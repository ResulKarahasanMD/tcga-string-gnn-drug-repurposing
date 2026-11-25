"""Disease-module construction and network-proximity drug ranking (Sections 21, 25-26).

Proximity uses the Guney et al. (2016) *closest* measure:

    d(S, T) = (1/|T|) * sum_{t in T} min_{s in S} dist(t, s)

with STRING confidence turned into edge distance (``-log(conf)`` or ``1/conf``). Significance
comes from a **degree-matched** null: for each real target we draw random proteins of similar
degree, preserve target-set size, build >=1000 random sets, and compute

    z = (d_obs - mean(d_random)) / std(d_random)     (more negative = closer than expected)

plus an empirical p-value and Benjamini-Hochberg FDR. Proximity is hypothesis-generating; it
does NOT prove binding, exposure, or efficacy.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components, dijkstra

from .config import Config, write_run_manifest

logger = logging.getLogger(__name__)


# ======================================================================================
# Graph -> weighted distance matrix
# ======================================================================================
def build_sparse_graph(
    edge_index: np.ndarray, edge_weight: np.ndarray, n_nodes: int, *, transform: str = "neglog"
) -> sp.csr_matrix:
    """Build a sparse distance graph from edges. ``transform`` maps confidence->distance."""
    w = np.clip(edge_weight.astype(float), 1e-6, 1.0)
    if transform == "neglog":
        dist = -np.log(w)          # conf=1 -> 0, conf=0.7 -> 0.357
    elif transform == "inverse":
        dist = 1.0 / w             # conf=1 -> 1, conf=0.7 -> 1.43
    else:
        raise ValueError(f"Unknown distance transform {transform!r}.")
    dist = np.maximum(dist, 1e-9)  # avoid zero-distance edges (breaks shortest-path semantics)
    g = sp.csr_matrix((dist, (edge_index[0], edge_index[1])), shape=(n_nodes, n_nodes))
    return g


def distance_to_module(graph: sp.csr_matrix, module_nodes: list[int]) -> np.ndarray:
    """Multi-source Dijkstra: min weighted distance from every node to the nearest module node."""
    if not module_nodes:
        raise ValueError("Empty disease module.")
    dmat = dijkstra(graph, directed=False, indices=module_nodes)
    return np.min(dmat, axis=0)  # nearest module node per target


# ======================================================================================
# Disease module (Section 21)
# ======================================================================================
def random_walk_restart(graph_adj: sp.csr_matrix, seeds: list[int], *, restart: float = 0.5,
                        n_iter: int = 100, tol: float = 1e-8) -> np.ndarray:
    """RWR propagation from seed nodes over a (confidence-weighted) adjacency; returns scores."""
    n = graph_adj.shape[0]
    deg = np.asarray(graph_adj.sum(axis=1)).ravel()
    deg[deg == 0] = 1.0
    W = graph_adj.multiply(1.0 / deg[:, None]).tocsr()  # row-normalized transition
    p0 = np.zeros(n)
    p0[seeds] = 1.0 / len(seeds)
    p = p0.copy()
    for _ in range(n_iter):
        p_new = (1 - restart) * (W.T @ p) + restart * p0
        if np.abs(p_new - p).sum() < tol:
            p = p_new
            break
        p = p_new
    return p


def build_disease_module(
    adjacency: sp.csr_matrix, seeds: list[int], size: int, *, restart: float = 0.5
) -> list[int]:
    """Module = seeds + top RWR-propagated genes, up to ``size`` (seeds always included)."""
    seeds = list(dict.fromkeys(seeds))
    if size <= len(seeds):
        return seeds
    scores = random_walk_restart(adjacency, seeds, restart=restart)
    order = np.argsort(scores)[::-1]
    module = list(seeds)
    seen = set(seeds)
    for node in order:
        if len(module) >= size:
            break
        if node not in seen:
            module.append(int(node))
            seen.add(int(node))
    return module


def module_connectivity(adjacency: sp.csr_matrix, module_nodes: list[int]) -> dict[str, float]:
    """Internal connectivity of the module subgraph (edges, giant-component fraction)."""
    idx = np.array(module_nodes)
    sub = adjacency[idx][:, idx]
    n_comp, labels = connected_components(sub, directed=False)
    sizes = np.bincount(labels)
    return {
        "module_size": int(len(module_nodes)),
        "internal_edges": int(sub.nnz // 2),
        "n_components": int(n_comp),
        "giant_fraction": float(sizes.max() / len(module_nodes)) if len(module_nodes) else 0.0,
    }


# ======================================================================================
# Degree-matched null (Section 25)
# ======================================================================================
def build_degree_bins(degrees: np.ndarray, giant_mask: np.ndarray, n_bins: int = 50) -> dict[int, np.ndarray]:
    """Map each degree bin -> array of candidate node indices (within the giant component)."""
    valid = np.where(giant_mask)[0]
    deg_valid = degrees[valid]
    edges = np.unique(np.quantile(deg_valid, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.digitize(deg_valid, edges)
    bins: dict[int, np.ndarray] = {}
    for b in np.unique(bin_ids):
        bins[int(b)] = valid[bin_ids == b]
    node_bin = np.full(len(degrees), -1)
    node_bin[valid] = bin_ids
    bins["_node_bin"] = node_bin  # type: ignore[assignment]
    bins["_edges"] = edges         # type: ignore[assignment]
    return bins


def degree_matched_sample(
    targets: list[int], bins: dict[int, np.ndarray], n_random: int, rng: np.random.Generator
) -> np.ndarray:
    """Return an [n_random, len(targets)] array of degree-matched random node indices."""
    node_bin = bins["_node_bin"]
    out = np.empty((n_random, len(targets)), dtype=int)
    for j, t in enumerate(targets):
        b = int(node_bin[t])
        pool = bins.get(b)
        if pool is None or len(pool) == 0:
            pool = np.concatenate([v for k, v in bins.items() if isinstance(k, int)])
        out[:, j] = rng.choice(pool, size=n_random, replace=True)
    return out


def proximity_zscore(
    dist_to_module: np.ndarray,
    targets: list[int],
    bins: dict[int, np.ndarray],
    *,
    n_random: int = 1000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Observed closest-distance + degree-matched null z-score, empirical p (lower=closer)."""
    rng = rng or np.random.default_rng(11)
    reachable = [t for t in targets if np.isfinite(dist_to_module[t])]
    if not reachable:
        return {"d_obs": float("nan"), "zscore": float("nan"), "empirical_p": float("nan"),
                "n_targets": len(targets), "n_reachable": 0}
    d_obs = float(np.mean(dist_to_module[reachable]))
    rand_sets = degree_matched_sample(reachable, bins, n_random, rng)
    rand_d = dist_to_module[rand_sets]
    rand_means = np.where(np.isfinite(rand_d), rand_d, np.nan)
    d_random = np.nanmean(rand_means, axis=1)
    d_random = d_random[np.isfinite(d_random)]
    mu, sd = float(np.mean(d_random)), float(np.std(d_random))
    z = (d_obs - mu) / sd if sd > 0 else 0.0
    emp_p = float((np.sum(d_random <= d_obs) + 1) / (len(d_random) + 1))  # closer-than-random
    return {"d_obs": d_obs, "zscore": float(z), "empirical_p": emp_p,
            "null_mean": mu, "null_std": sd, "n_targets": len(targets), "n_reachable": len(reachable)}


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """BH FDR-adjusted p-values (NaNs preserved)."""
    p = np.asarray(pvals, dtype=float)
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan)
    idx = np.where(mask)[0]
    ps = p[idx]
    order = np.argsort(ps)
    n = len(ps)
    ranked = ps[order]
    adj = ranked * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    q_sorted = np.clip(adj, 0, 1)
    out = np.empty(n)
    out[order] = q_sorted
    q[idx] = out
    return q


# ======================================================================================
# Orchestration: run_rank_drugs
# ======================================================================================
def _enrich_with_mechanism(ranking, cfg, node_table, drug_targets, interim):
    """Add direction-consistency, target expression, mechanism score, and summaries."""
    from .mechanism import (
        DEFAULT_WEIGHTS,
        compute_mechanism_scores,
        direction_consistency,
        mechanistic_summary,
    )

    nt = node_table.sort_values("node_index").reset_index(drop=True)
    symbols = nt["gene_name"].to_numpy()
    mean_expr = nt["mean_expression"].to_numpy()
    de_by_symbol = nt.set_index("gene_name")[["log2fc"]]

    inter_path = interim / "drug_interactions.parquet"
    actions_by_drug: dict[str, dict[str, str]] = {}
    if inter_path.exists():
        inter = pd.read_parquet(inter_path)
        for drug_id, grp in inter.groupby("drug_id"):
            actions_by_drug[drug_id] = dict(zip(grp["gene_name"], grp["action"]))

    dir_frac, mean_texpr, tgt_symbols_map = [], [], {}
    for row in ranking.itertuples(index=False):
        idxs = drug_targets.get(row.drug_id, [])
        tsyms = [symbols[i] for i in idxs]
        tgt_symbols_map[row.drug_id] = tsyms
        dc = direction_consistency(tsyms, actions_by_drug.get(row.drug_id, {}), de_by_symbol,
                                   lfc_threshold=float(cfg.get("differential_expression.abs_log2fc_threshold", 1.0)))
        dir_frac.append(dc["direction_consistency"])
        mean_texpr.append(float(np.mean([mean_expr[i] for i in idxs])) if idxs else np.nan)

    ranking = ranking.copy()
    ranking["direction_consistency"] = dir_frac
    ranking["mean_target_expression"] = mean_texpr

    weights = {**DEFAULT_WEIGHTS, **cfg.get("mechanism.weights", {})}
    ranking["mechanism_score"] = compute_mechanism_scores(ranking, weights).to_numpy()

    # mechanistic summaries for the 20 closest (by proximity z-score)
    top = ranking.nsmallest(20, "proximity_zscore")
    summaries = {}
    for r in top.itertuples(index=False):
        d = r._asdict()
        summaries[d["drug_id"]] = mechanistic_summary(d, tgt_symbols_map.get(d["drug_id"], []))
    ranking["mechanistic_summary"] = ranking["drug_id"].map(summaries).fillna("")
    return ranking


def run_rank_drugs(cfg: Config) -> dict[str, Any]:
    """Rank drugs by degree-matched network proximity to the disease module."""
    import torch

    cohort = cfg.cohort
    graph_dir = cfg.path("paths.processed") / cohort / "graph"
    interim = cfg.path("paths.interim") / cohort
    tables = cfg.path("paths.tables") / cohort
    figures = cfg.path("paths.figures") / cohort
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    data = torch.load(graph_dir / "pyg_data.pt", weights_only=False)
    node_table = pd.read_parquet(graph_dir / "node_table.parquet")
    with open(interim / "drug_target_indices.json") as fh:
        drug_targets = {k: list(map(int, v)) for k, v in json.load(fh).items()}
    drug_meta = pd.read_parquet(interim / "drug_meta.parquet").set_index("drug_id")

    n_nodes = data.num_nodes
    edge_index = data.edge_index.numpy()
    edge_weight = data.edge_attr.numpy()
    degrees = node_table["degree"].to_numpy()

    pcfg = cfg.section("proximity")
    graph = build_sparse_graph(edge_index, edge_weight, n_nodes, transform=pcfg["distance_transform"])
    adjacency = sp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(n_nodes, n_nodes))

    # giant component (for reachability + degree-matched pool)
    n_comp, comp_labels = connected_components(adjacency, directed=False)
    giant_label = np.bincount(comp_labels).argmax()
    giant_mask = comp_labels == giant_label

    # disease module: seeds = Open Targets positives, expanded by RWR to primary size
    seeds = np.where(node_table["label"].to_numpy() == 1)[0].tolist()
    seeds = [s for s in seeds if giant_mask[s]]
    module_sizes = pcfg.get("module_sizes", [50, 100, 250, 500])
    primary_size = module_sizes[0] if module_sizes else len(seeds)
    module = build_disease_module(adjacency, seeds, primary_size)
    conn = module_connectivity(adjacency, module)
    logger.info("Disease module: %d genes (%d seeds), internal edges=%d, giant_frac=%.2f",
                len(module), len(seeds), conn["internal_edges"], conn["giant_fraction"])

    dist_to_module = distance_to_module(graph, module)
    bins = build_degree_bins(degrees, giant_mask, n_bins=int(pcfg.get("degree_bins", 50)))

    # per-drug proximity
    n_random = int(pcfg.get("n_random", 1000))
    rng = np.random.default_rng(cfg.seed)
    module_set = set(module)
    node_scores = np.load(graph_dir / "node_scores.npz")
    gnn_key = next((k for k in node_scores.files if k.endswith("_mean")), None)
    gnn_score = node_scores[gnn_key] if gnn_key else np.zeros(n_nodes)

    rows = []
    for drug_id, targets in drug_targets.items():
        res = proximity_zscore(dist_to_module, targets, bins, n_random=n_random, rng=rng)
        meta = drug_meta.loc[drug_id] if drug_id in drug_meta.index else {}
        rows.append({
            "drug_id": drug_id,
            "drug_name": meta.get("drug_name") if isinstance(meta, pd.Series) else None,
            "approved": meta.get("approved") if isinstance(meta, pd.Series) else None,
            "n_targets": res["n_targets"],
            "n_reachable": res["n_reachable"],
            "targets_in_module": len([t for t in targets if t in module_set]),
            "network_distance": res["d_obs"],
            "proximity_zscore": res["zscore"],
            "empirical_pvalue": res["empirical_p"],
            "mean_target_gnn_score": float(np.mean([gnn_score[t] for t in targets])) if targets else np.nan,
        })
    ranking = pd.DataFrame(rows)
    ranking = ranking[ranking["n_reachable"] > 0].copy()
    ranking["fdr"] = benjamini_hochberg(ranking["empirical_pvalue"].to_numpy())

    # --- mechanism / directionality enrichment (Section 26/28) ---
    ranking = _enrich_with_mechanism(ranking, cfg, node_table, drug_targets, interim)

    ranking = ranking.sort_values("proximity_zscore").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(tables / "drug_ranking.tsv", sep="\t", index=False)

    # positive controls: where do known cohort drugs rank?
    controls = cfg.get("controls.positive_drugs_examples", [])
    ctrl_hits = ranking[ranking["drug_name"].str.lower().isin([c.lower() for c in controls])] \
        if len(ranking) and controls else pd.DataFrame()

    summary = {
        "cohort": cohort,
        "disease_module": conn,
        "n_seeds": len(seeds),
        "n_drugs_ranked": int(len(ranking)),
        "n_significant_fdr05": int((ranking["fdr"] < float(pcfg.get("fdr_threshold", 0.05))).sum()),
        "distance_transform": pcfg["distance_transform"],
        "n_random": n_random,
        "top10": ranking.head(10)[["rank", "drug_name", "proximity_zscore", "empirical_pvalue", "fdr"]].to_dict("records"),
        "positive_control_ranks": ctrl_hits[["rank", "drug_name", "proximity_zscore", "fdr"]].to_dict("records") if len(ctrl_hits) else [],
        "outputs": {"drug_ranking": str(tables / "drug_ranking.tsv")},
    }
    write_run_manifest(cfg, stage="rank_drugs", seed=cfg.seed, extra=summary)
    logger.info("Ranked %d drugs; %d significant at FDR<0.05. Top: %s",
                len(ranking), summary["n_significant_fdr05"],
                ranking.head(3)["drug_name"].tolist())
    return summary
