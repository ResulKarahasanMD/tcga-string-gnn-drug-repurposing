"""Figures (Section 31). DE figures live in differential_expression.py; this module adds the
network / model / drug-repurposing figures. Saved as high-res PNG (+ SVG where possible)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .config import Config  # noqa: E402

logger = logging.getLogger(__name__)


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    try:
        fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    except Exception:  # noqa: BLE001
        pass
    plt.close(fig)


def plot_degree_distribution(node_table: pd.DataFrame, path: Path) -> None:
    deg = node_table["degree"].to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.log10(deg + 1), bins=50, color="steelblue", alpha=0.8)
    ax.set_xlabel("log10(degree + 1)")
    ax.set_ylabel("number of genes")
    ax.set_title(f"STRING network degree distribution (n={len(deg)} nodes)")
    _save(fig, path)


def plot_roc_pr(y_true: np.ndarray, scores: dict[str, np.ndarray], path: Path) -> None:
    from sklearn.metrics import precision_recall_curve, roc_curve

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax1.plot(fpr, tpr, label=name, lw=1.5)
        prec, rec, _ = precision_recall_curve(y_true, s)
        ax2.plot(rec, prec, label=name, lw=1.5)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.6)
    ax1.set(xlabel="FPR", ylabel="TPR", title="ROC (labeled nodes)")
    ax2.axhline(y_true.mean(), ls="--", c="k", lw=0.6)
    ax2.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    ax1.legend(fontsize=7)
    _save(fig, path)


def plot_model_comparison(metrics_df: pd.DataFrame, path: Path) -> None:
    agg = metrics_df.groupby("model")["auprc"].agg(["mean", "std"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(agg.index, agg["mean"], xerr=agg["std"].fillna(0), color="mediumseagreen", alpha=0.85)
    ax.set_xlabel("test AUPRC (mean ± sd over seeds)")
    ax.set_title("Model comparison: baselines vs GNNs")
    _save(fig, path)


def plot_drug_null_distribution(d_random: np.ndarray, d_obs: float, drug_name: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(d_random, bins=40, color="lightgray", alpha=0.9, label="degree-matched null")
    ax.axvline(d_obs, color="crimson", lw=2, label=f"observed d={d_obs:.3f}")
    ax.set_xlabel("mean closest distance to disease module")
    ax.set_ylabel("random target sets")
    ax.set_title(f"Network-proximity null: {drug_name}")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_top_drugs(ranking: pd.DataFrame, path: Path, n: int = 20) -> None:
    top = ranking.nsmallest(n, "proximity_zscore").iloc[::-1]
    labels = top["drug_name"].fillna(top["drug_id"]).astype(str)
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["crimson" if f < 0.05 else "steelblue" for f in top["fdr"]]
    ax.barh(range(len(top)), -top["proximity_zscore"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("-proximity z-score (higher = closer to module)")
    ax.set_title(f"Top {n} drugs by network proximity\n(red = FDR<0.05)")
    _save(fig, path)


def plot_three_layer_network(cfg: Config, ranking: pd.DataFrame, path: Path, n_drugs: int = 8) -> None:
    """Drug -> target -> disease-module three-layer schematic for the top drugs."""
    import networkx as nx

    graph_dir = cfg.path("paths.processed") / cfg.cohort / "graph"
    interim = cfg.path("paths.interim") / cfg.cohort
    node_table = pd.read_parquet(graph_dir / "node_table.parquet").sort_values("node_index")
    symbols = node_table["gene_name"].to_numpy()
    with open(interim / "drug_target_indices.json") as fh:
        drug_targets = {k: list(map(int, v)) for k, v in json.load(fh).items()}

    top = ranking.nsmallest(n_drugs, "proximity_zscore")
    g = nx.Graph()
    layers = {"drug": [], "target": [], "module": []}
    module_syms = set(node_table.loc[node_table["label"] == 1, "gene_name"])
    for r in top.itertuples(index=False):
        dname = r.drug_name or r.drug_id
        g.add_node(dname, layer="drug")
        layers["drug"].append(dname)
        for ti in drug_targets.get(r.drug_id, [])[:6]:
            tsym = symbols[ti]
            g.add_node(tsym, layer="target")
            g.add_edge(dname, tsym)
            if tsym not in layers["target"]:
                layers["target"].append(tsym)
    pos = {}
    for i, n_ in enumerate(layers["drug"]):
        pos[n_] = (0, i)
    for i, n_ in enumerate(layers["target"]):
        pos[n_] = (1, i * len(layers["drug"]) / max(len(layers["target"]), 1))
    fig, ax = plt.subplots(figsize=(9, 8))
    node_colors = ["#e4572e" if g.nodes[n_]["layer"] == "drug"
                   else ("#2e86ab" if n_ in module_syms else "#a5c9d6") for n_ in g.nodes]
    nx.draw(g, pos, ax=ax, node_color=node_colors, with_labels=True, font_size=6,
            node_size=350, edge_color="lightgray")
    ax.set_title("Drug (red) -> target (dark blue = in disease module) network")
    _save(fig, path)


def run_visualizations(cfg: Config) -> dict[str, Any]:
    """Generate the network/model/drug figures from saved pipeline outputs."""
    cohort = cfg.cohort
    figures = cfg.path("paths.figures") / cohort
    tables = cfg.path("paths.tables") / cohort
    graph_dir = cfg.path("paths.processed") / cohort / "graph"
    figures.mkdir(parents=True, exist_ok=True)
    made = []

    node_table = pd.read_parquet(graph_dir / "node_table.parquet")
    plot_degree_distribution(node_table, figures / "degree_distribution.png"); made.append("degree_distribution")

    if (tables / "model_metrics.tsv").exists():
        plot_model_comparison(pd.read_csv(tables / "model_metrics.tsv", sep="\t"),
                              figures / "model_comparison.png"); made.append("model_comparison")

    # ROC/PR on labeled nodes for a few methods
    if (graph_dir / "node_scores.npz").exists():
        scores = np.load(graph_dir / "node_scores.npz")
        y = node_table["label"].to_numpy()
        labeled = y >= 0
        want = [k for k in ["gcn_mean", "gatv2_mean", "network_propagation_rwr", "logreg"] if k in scores.files]
        sc = {k.replace("_mean", ""): scores[k][labeled] for k in want}
        if sc:
            plot_roc_pr(y[labeled], sc, figures / "roc_pr_curves.png"); made.append("roc_pr_curves")

    if (tables / "drug_ranking.tsv").exists():
        ranking = pd.read_csv(tables / "drug_ranking.tsv", sep="\t")
        plot_top_drugs(ranking, figures / "top20_drugs.png"); made.append("top20_drugs")
        try:
            plot_three_layer_network(cfg, ranking, figures / "three_layer_network.png"); made.append("three_layer_network")
        except Exception as exc:  # noqa: BLE001
            logger.warning("three-layer network skipped: %s", exc)

    logger.info("Generated figures: %s", made)
    return {"cohort": cohort, "figures": made, "figures_dir": str(figures)}
