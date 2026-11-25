"""GNN explainability (Section 20): node-feature importance + supporting neighbors.

Uses PyG's GNNExplainer when available, with a gradient (saliency) fallback that always works.
Produces, per high-ranking gene, a table with feature importances and top contributing network
neighbors. IMPORTANT: explanations are model-internal attributions, NOT causal evidence, and are
distinct from classical network centrality.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

logger = logging.getLogger(__name__)


class _EdgeWeightWrapper(nn.Module):
    """Adapt a (x, edge_index, edge_weight) model to a (x, edge_index) call for explainers."""

    def __init__(self, model: nn.Module, edge_weight: torch.Tensor | None):
        super().__init__()
        self.model = model
        self._ew = edge_weight

    def forward(self, x, edge_index, **kwargs):  # noqa: ANN001
        return self.model(x, edge_index, self._ew)


def saliency_feature_importance(
    model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor,
    edge_weight: torch.Tensor | None, node_idx: int, feature_names: list[str],
) -> dict[str, float]:
    """|d(logit)/d(x_node)| saliency per input feature for one node."""
    model.eval()
    xin = x.clone().detach().requires_grad_(True)
    logit = model(xin, edge_index, edge_weight).squeeze(-1)[node_idx]
    logit.backward()
    grad = xin.grad[node_idx].abs().detach().cpu().numpy()
    return {feature_names[i]: float(grad[i]) for i in range(len(feature_names))}


def top_contributing_neighbors(
    node_idx: int, edge_index: torch.Tensor, scores: np.ndarray, symbols: np.ndarray, k: int = 5
) -> list[str]:
    """Immediate graph neighbors of a node, ranked by their own model score."""
    ei = edge_index.cpu().numpy()
    nbrs = ei[1][ei[0] == node_idx]
    nbrs = np.unique(nbrs)
    if len(nbrs) == 0:
        return []
    order = nbrs[np.argsort(scores[nbrs])[::-1]][:k]
    return [str(symbols[i]) for i in order]


def try_gnn_explainer(model, x, edge_index, edge_weight, node_idx) -> dict[str, Any] | None:
    """Attempt PyG GNNExplainer; return per-feature importances or None if unavailable."""
    try:
        from torch_geometric.explain import Explainer, GNNExplainer

        wrapped = _EdgeWeightWrapper(model, edge_weight)
        explainer = Explainer(
            model=wrapped,
            algorithm=GNNExplainer(epochs=100),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(mode="binary_classification", task_level="node", return_type="raw"),
        )
        expl = explainer(x, edge_index, index=int(node_idx))
        node_mask = expl.get("node_mask")
        if node_mask is None:
            return None
        imp = node_mask[node_idx].detach().cpu().numpy() if node_mask.dim() == 2 else node_mask.detach().cpu().numpy()
        return {"feature_importance": imp.tolist()}
    except Exception as exc:  # noqa: BLE001
        logger.info("GNNExplainer unavailable/failed (%s); using saliency fallback.", exc)
        return None


def explain_top_genes(cfg, top_k: int = 20) -> pd.DataFrame:  # noqa: ANN001
    """Explain the top-ranked genes; write a per-gene explanation table."""
    from .config import get_device
    from .models import build_model

    device = get_device("cpu")  # explanations are cheap; CPU for determinism
    cohort = cfg.cohort
    graph_dir = cfg.path("paths.processed") / cohort / "graph"
    tables = cfg.path("paths.tables") / cohort

    data = torch.load(graph_dir / "pyg_data.pt", weights_only=False)
    node_table = pd.read_parquet(graph_dir / "node_table.parquet").sort_values("node_index")
    scores_npz = np.load(graph_dir / "node_scores.npz")
    ckpt = torch.load(cfg.path("paths.models") / cohort / "best_model.pt", weights_only=False)
    arch = ckpt["arch"]
    gnn_score = scores_npz[f"{arch}_mean"] if f"{arch}_mean" in scores_npz.files else \
        scores_npz[[k for k in scores_npz.files if k.endswith("_mean")][0]]

    # rebuild + load best model; scale features on the training fold (as in training)
    from .train import scale_features, _split_masks
    from sklearn.preprocessing import StandardScaler  # noqa: F401
    masks = np.load(graph_dir / "split_masks.npz")
    tr, va, te = _split_masks(masks, cfg.section("split")["strategies"][0])
    x_scaled = scale_features(data.x.numpy(), ~(va | te))
    x = torch.tensor(x_scaled, device=device)
    ei = data.edge_index.to(device)
    ew = data.edge_attr.to(device) if getattr(data, "edge_attr", None) is not None else None

    model = build_model(arch, x.shape[1], cfg.section("model"), out_channels=1).to(device)
    model.load_state_dict(ckpt["state_dict"])
    feature_names = list(data.feature_names)
    symbols = node_table["gene_name"].to_numpy()

    top_idx = np.argsort(gnn_score)[::-1][:top_k]
    rows = []
    for ni in top_idx:
        sal = saliency_feature_importance(model, x, ei, ew, int(ni), feature_names)
        nbrs = top_contributing_neighbors(int(ni), ei, gnn_score, symbols)
        nt = node_table.iloc[int(ni)]
        rows.append({
            "gene": symbols[ni], "gnn_score": float(gnn_score[ni]), "label": int(nt["label"]),
            "degree": int(nt["degree"]), "log2fc": float(nt["log2fc"]),
            "neglog10_fdr": float(nt["neglog10_fdr"]), "community": int(nt["community"]) if pd.notna(nt["community"]) else -1,
            "top_feature": max(sal, key=sal.get),
            "feature_importance": {k: round(v, 4) for k, v in sal.items()},
            "top_contributing_neighbors": ";".join(nbrs),
        })
    df = pd.DataFrame(rows)
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "gene_explanations.tsv", sep="\t", index=False)
    logger.info("Wrote gene explanations for %d genes -> %s", len(df), tables / "gene_explanations.tsv")
    return df
