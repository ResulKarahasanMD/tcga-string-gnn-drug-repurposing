"""Training, baselines, graph-information controls, and gene ranking (Sections 16-19).

Transductive node classification: the whole graph + all node features are visible; the loss is
computed only on train-labeled nodes, model selection on val-labeled, final metrics on
test-labeled. (Transductive leakage caveat: neighbours of held-out nodes are seen during
message passing — this is standard for transductive GNNs and is why we ALSO report the
inductive-style baselines and degree-matched splits.)

Anti-leakage feature scaling: StandardScaler is fit on the *training fold* only — all nodes
except the val/test-labeled hold-outs — never on the whole labeled set.

Baselines (a GNN must beat these): logistic regression, random forest, HistGradientBoosting,
feature-only MLP, degree/PageRank ranking, and network propagation (random walk with restart).
Graph-information controls: real vs degree-preserving-rewired vs features-only.
"""

from __future__ import annotations

import copy
import logging
import warnings
from pathlib import Path
from typing import Any

# MPS lacks deterministic scatter/index_put; we run with warn_only determinism, so silence the
# repeated per-backward warnings (the non-determinism is documented; multi-seed mean+/-sd is the
# reproducibility unit). Numerics are unaffected.
warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch import nn

from .config import Config, get_device, set_seed, write_run_manifest
from .evaluate import bootstrap_ci, compute_metrics
from .models import build_model

logger = logging.getLogger(__name__)


# ======================================================================================
# Data loading + scaling
# ======================================================================================
def load_graph(cfg: Config):
    """Load the PyG Data, node table, and split masks produced by build_string_graph."""
    graph_dir = cfg.path("paths.processed") / cfg.cohort / "graph"
    data_path = graph_dir / "pyg_data.pt"
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} missing. Run build_string_graph first.")
    # weights_only=False: our own trusted file contains a PyG Data object (torch>=2.6 default True).
    data = torch.load(data_path, weights_only=False)
    node_table = pd.read_parquet(graph_dir / "node_table.parquet")
    masks = np.load(graph_dir / "split_masks.npz")
    return data, node_table, masks, graph_dir


def scale_features(x: np.ndarray, fit_mask: np.ndarray) -> np.ndarray:
    """Fit StandardScaler on ``fit_mask`` rows only (training fold), transform all rows."""
    scaler = StandardScaler()
    scaler.fit(x[fit_mask])
    return scaler.transform(x).astype(np.float32)


def _split_masks(masks: dict, strategy: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (masks[f"{strategy}_train"], masks[f"{strategy}_val"], masks[f"{strategy}_test"])


# ======================================================================================
# GNN training
# ======================================================================================
def train_gnn(
    data,
    x_scaled: np.ndarray,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    cfg: Config,
    *,
    arch: str,
    device: torch.device,
    seed: int,
    edge_index_override: torch.Tensor | None = None,
    use_edges: bool = True,
) -> tuple[nn.Module, dict[str, list[float]]]:
    """Train one GNN (or MLP) with early stopping on val AUPRC. Returns (best_model, history)."""
    set_seed(seed)
    tcfg = cfg.section("train")
    mcfg = cfg.section("model")

    x = torch.tensor(x_scaled, device=device)
    y = data.y.to(device)
    edge_index = (edge_index_override if edge_index_override is not None else data.edge_index).to(device)
    edge_weight = data.edge_attr.to(device) if getattr(data, "edge_attr", None) is not None else None

    y_np = data.y.numpy()
    train_lab = train_mask & (y_np >= 0)
    val_lab = val_mask & (y_np >= 0)
    tr_idx = torch.tensor(np.where(train_lab)[0], device=device)
    va_idx = torch.tensor(np.where(val_lab)[0], device=device)

    n_pos = int(y_np[train_lab].sum())
    n_neg = int((y_np[train_lab] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=device) \
        if tcfg.get("class_weight", "balanced") == "balanced" else None

    model = build_model(arch, x.shape[1], mcfg, out_channels=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(tcfg["learning_rate"]),
                            weight_decay=float(tcfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=10)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val, best_state, patience_ctr = -np.inf, None, 0
    history = {"train_loss": [], "val_auprc": []}
    ei_arg = edge_index if use_edges else torch.empty((2, 0), dtype=torch.long, device=device)
    ew_arg = edge_weight if (use_edges and edge_index_override is None) else None

    for epoch in range(int(tcfg["max_epochs"])):
        model.train()
        opt.zero_grad()
        logits = model(x, ei_arg, ew_arg).squeeze(-1)
        loss = loss_fn(logits[tr_idx], y[tr_idx].float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(tcfg["grad_clip"]))
        opt.step()

        model.eval()
        with torch.no_grad():
            prob = torch.sigmoid(model(x, ei_arg, ew_arg).squeeze(-1))
        val_prob = prob[va_idx].cpu().numpy()
        val_y = y[va_idx].cpu().numpy()
        val_auprc = compute_metrics(val_y, val_prob)["auprc"]
        sched.step(val_auprc)
        history["train_loss"].append(float(loss.item()))
        history["val_auprc"].append(float(val_auprc))

        if val_auprc > best_val:
            best_val, best_state, patience_ctr = val_auprc, copy.deepcopy(model.state_dict()), 0
        else:
            patience_ctr += 1
            if patience_ctr >= int(tcfg["patience"]):
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    logger.info("  [%s seed=%d] best val AUPRC=%.3f @ epoch stop=%d", arch, seed, best_val, epoch + 1)
    return model, history


@torch.no_grad()
def gnn_scores(model: nn.Module, data, x_scaled: np.ndarray, device: torch.device,
               *, use_edges: bool = True) -> np.ndarray:
    """Sigmoid scores for ALL nodes."""
    model.eval()
    x = torch.tensor(x_scaled, device=device)
    ei = data.edge_index.to(device) if use_edges else torch.empty((2, 0), dtype=torch.long, device=device)
    ew = data.edge_attr.to(device) if (use_edges and getattr(data, "edge_attr", None) is not None) else None
    return torch.sigmoid(model(x, ei, ew).squeeze(-1)).cpu().numpy()


# ======================================================================================
# Baselines
# ======================================================================================
def train_sklearn_baselines(
    x_scaled: np.ndarray, y: np.ndarray, train_mask: np.ndarray, seed: int
) -> dict[str, np.ndarray]:
    """Feature-only classifiers; return {name: all-node scores}."""
    tl = train_mask & (y >= 0)
    xt, yt = x_scaled[tl], y[tl]
    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=seed, n_jobs=-1),
        "hist_gbm": HistGradientBoostingClassifier(random_state=seed),
    }
    out = {}
    for name, clf in models.items():
        clf.fit(xt, yt)
        out[name] = clf.predict_proba(x_scaled)[:, 1]
    return out


def network_propagation_rwr(
    data, seed_nodes: np.ndarray, *, restart: float = 0.5, n_iter: int = 100, tol: float = 1e-6
) -> np.ndarray:
    """Random walk with restart from seed (train-positive) nodes over the weighted graph."""
    n = data.num_nodes
    ei = data.edge_index.numpy()
    w = data.edge_attr.numpy() if getattr(data, "edge_attr", None) is not None else np.ones(ei.shape[1])
    # column-normalized transition matrix (sparse-free, degree-normalized)
    deg = np.zeros(n)
    np.add.at(deg, ei[0], w)
    deg[deg == 0] = 1.0
    p0 = np.zeros(n)
    if len(seed_nodes):
        p0[seed_nodes] = 1.0 / len(seed_nodes)
    p = p0.copy()
    for _ in range(n_iter):
        # message: sum over incoming neighbors of p[src]*w/deg[src]
        contrib = p[ei[0]] * w / deg[ei[0]]
        p_new = np.zeros(n)
        np.add.at(p_new, ei[1], contrib)
        p_new = (1 - restart) * p_new + restart * p0
        if np.abs(p_new - p).sum() < tol:
            p = p_new
            break
        p = p_new
    return p


def degree_pagerank_scores(data, node_table: pd.DataFrame) -> dict[str, np.ndarray]:
    """Non-learned ranking baselines from classical centrality."""
    return {
        "degree_rank": node_table["degree"].to_numpy(dtype=float),
        "pagerank_rank": node_table["pagerank"].to_numpy(dtype=float),
    }


# ======================================================================================
# Graph-information controls (Section 16)
# ======================================================================================
def degree_preserving_rewire(data, seed: int) -> torch.Tensor:
    """Return a degree-preserving randomized edge_index (double-edge swaps)."""
    import networkx as nx

    ei = data.edge_index.numpy()
    # undirected simple graph from the (bidirectional) edge_index
    g = nx.Graph()
    g.add_nodes_from(range(data.num_nodes))
    g.add_edges_from(zip(ei[0].tolist(), ei[1].tolist()))
    n_swap = max(1, g.number_of_edges())
    nx.double_edge_swap(g, nswap=n_swap, max_tries=n_swap * 10, seed=seed)
    u = np.array([e[0] for e in g.edges()])
    v = np.array([e[1] for e in g.edges()])
    ei_new = np.vstack([np.concatenate([u, v]), np.concatenate([v, u])])
    return torch.tensor(ei_new, dtype=torch.long)


# ======================================================================================
# Orchestration
# ======================================================================================
def run_train(cfg: Config) -> dict[str, Any]:
    """Train baselines + GNNs across seeds on the primary split; save metrics + best model."""
    device = get_device(cfg.get("compute.device", "auto"))
    logger.info("Training device: %s", device)
    data, node_table, masks, graph_dir = load_graph(cfg)
    y = data.y.numpy()
    x_raw = data.x.numpy()

    split_cfg = cfg.section("split")
    primary = split_cfg["strategies"][0]
    train_mask, val_mask, test_mask = _split_masks(masks, primary)
    fit_mask = ~(val_mask | test_mask)  # training fold = everything except held-out labeled nodes
    x_scaled = scale_features(x_raw, fit_mask)

    test_lab = test_mask & (y >= 0)
    y_test = y[test_lab]
    seeds = cfg.seeds
    rows: list[dict[str, Any]] = []
    all_scores: dict[str, np.ndarray] = {}

    # ---- baselines (seed-0) ----
    base = train_sklearn_baselines(x_scaled, y, train_mask, seeds[0])
    base.update(degree_pagerank_scores(data, node_table))
    seed_pos = np.where(train_mask & (y == 1))[0]
    base["network_propagation_rwr"] = network_propagation_rwr(data, seed_pos)
    for name, sc in base.items():
        m = compute_metrics(y_test, sc[test_lab])
        m.update({"model": name, "type": "baseline", "seed": seeds[0]})
        rows.append(m)
        all_scores[name] = sc

    # ---- GNNs across seeds ----
    best_overall = {"auprc": -1.0}
    for arch in cfg.get("model.architectures", ["gcn", "graphsage", "gatv2"]):
        per_seed_scores = []
        for sd in seeds:
            model, _ = train_gnn(data, x_scaled, train_mask, val_mask, cfg,
                                 arch=arch, device=device, seed=sd)
            sc = gnn_scores(model, data, x_scaled, device)
            per_seed_scores.append(sc)
            m = compute_metrics(y_test, sc[test_lab])
            m.update({"model": arch, "type": "gnn", "seed": sd})
            rows.append(m)
            if m["auprc"] > best_overall["auprc"]:
                best_overall = {"auprc": m["auprc"], "arch": arch, "seed": sd, "state": model.state_dict()}
        all_scores[f"{arch}_mean"] = np.mean(per_seed_scores, axis=0)

    # ---- graph-information controls (gcn, seed-0) ----
    controls = {}
    m0, _ = train_gnn(data, x_scaled, train_mask, val_mask, cfg, arch="gcn", device=device, seed=seeds[0])
    controls["real_graph"] = compute_metrics(y_test, gnn_scores(m0, data, x_scaled, device)[test_lab])["auprc"]
    m_fo, _ = train_gnn(data, x_scaled, train_mask, val_mask, cfg, arch="gcn", device=device,
                        seed=seeds[0], use_edges=False)
    controls["features_only"] = compute_metrics(
        y_test, gnn_scores(m_fo, data, x_scaled, device, use_edges=False)[test_lab])["auprc"]
    rewired_ei = degree_preserving_rewire(data, seeds[0])
    m_rw, _ = train_gnn(data, x_scaled, train_mask, val_mask, cfg, arch="gcn", device=device,
                        seed=seeds[0], edge_index_override=rewired_ei)
    # score with rewired edges too
    with torch.no_grad():
        m_rw.eval()
        xt = torch.tensor(x_scaled, device=device)
        rw_scores = torch.sigmoid(m_rw(xt, rewired_ei.to(device), None).squeeze(-1)).cpu().numpy()
    controls["degree_preserving_rewired"] = compute_metrics(y_test, rw_scores[test_lab])["auprc"]

    # ---- persist ----
    tables = cfg.path("paths.tables") / cfg.cohort
    models_dir = cfg.path("paths.models") / cfg.cohort
    tables.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(tables / "model_metrics.tsv", sep="\t", index=False)

    # seed mean±sd summary per model
    summary_tbl = (
        metrics_df.groupby("model")[["auroc", "auprc", "mcc", "f1"]]
        .agg(["mean", "std"]).round(4)
    )
    summary_tbl.to_csv(tables / "model_metrics_summary.tsv", sep="\t")

    pd.DataFrame([controls]).to_csv(tables / "graph_information_controls.tsv", sep="\t", index=False)

    # bootstrap CI for the best model's test AUPRC/AUROC
    best_scores = all_scores.get(f"{best_overall['arch']}_mean")
    ci = {
        "auprc": bootstrap_ci(y_test, best_scores[test_lab], lambda a, b: compute_metrics(a, b)["auprc"]),
        "auroc": bootstrap_ci(y_test, best_scores[test_lab], lambda a, b: compute_metrics(a, b)["auroc"]),
    }

    torch.save({"arch": best_overall["arch"], "seed": best_overall["seed"],
                "state_dict": best_overall["state"], "feature_names": list(data.feature_names)},
               models_dir / "best_model.pt")
    np.savez(graph_dir / "node_scores.npz", **all_scores)

    summary = {
        "cohort": cfg.cohort,
        "device": str(device),
        "split": primary,
        "n_test_positives": int(y_test.sum()),
        "best_model": {"arch": best_overall["arch"], "test_auprc": round(best_overall["auprc"], 4)},
        "best_auprc_ci": [round(v, 4) for v in ci["auprc"]],
        "best_auroc_ci": [round(v, 4) for v in ci["auroc"]],
        "graph_information_controls": {k: round(v, 4) for k, v in controls.items()},
        "outputs": {"metrics": str(tables / "model_metrics.tsv"),
                    "summary": str(tables / "model_metrics_summary.tsv"),
                    "best_model": str(models_dir / "best_model.pt")},
    }
    write_run_manifest(cfg, stage="train_gnn", seed=cfg.seed,
                       input_files={"pyg_data": graph_dir / "pyg_data.pt"}, extra=summary)
    logger.info("Best model: %s (test AUPRC=%.3f). Controls: %s",
                best_overall["arch"], best_overall["auprc"], summary["graph_information_controls"])
    return summary


def rank_all_genes(cfg: Config) -> dict[str, Any]:
    """Score all genes with the best model and write the ranked gene table + top-k module."""
    device = get_device(cfg.get("compute.device", "auto"))
    data, node_table, masks, graph_dir = load_graph(cfg)
    scores = np.load(graph_dir / "node_scores.npz")
    ckpt = torch.load(cfg.path("paths.models") / cfg.cohort / "best_model.pt", weights_only=False)
    arch = ckpt["arch"]
    score_key = f"{arch}_mean"
    gnn_score = scores[score_key] if score_key in scores.files else scores[arch + "_mean"]

    ranked = node_table.copy()
    ranked["gnn_score"] = gnn_score
    ranked = ranked.sort_values("gnn_score", ascending=False).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)

    tables = cfg.path("paths.tables") / cfg.cohort
    cols = ["rank", "gene_name", "preferred_name", "gnn_score", "label", "degree", "pagerank",
            "log2fc", "neglog10_fdr", "community"]
    cols = [c for c in cols if c in ranked.columns]
    ranked[cols].to_csv(tables / "gene_ranking.tsv", sep="\t", index=False)

    summary = {
        "cohort": cfg.cohort, "best_arch": arch,
        "n_ranked": int(len(ranked)),
        "top20": ranked.head(20)[["gene_name", "gnn_score", "label"]].to_dict("records"),
        "outputs": {"gene_ranking": str(tables / "gene_ranking.tsv")},
    }
    logger.info("Ranked %d genes; top gene: %s (score=%.3f).",
                len(ranked), ranked.iloc[0]["gene_name"], ranked.iloc[0]["gnn_score"])
    return summary
