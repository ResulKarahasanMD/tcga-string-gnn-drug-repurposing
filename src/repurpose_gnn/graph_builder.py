"""Assemble the PyTorch Geometric ``Data`` object (Sections 11, 14, 15).

Ties together DE features, the STRING network, and Open Targets labels into a single graph:

    nodes   = protein-coding genes that are expressed (in the DE table) AND map to a STRING
              protein  (NOT "significant-DE-only" — avoids selection bias)
    x       = per-gene features (DE + expression stats); RAW here, standardized per-fold in
              train.py so scaling never leaks across the train/test boundary
    edge_*  = STRING edges >= threshold, undirected -> bidirectional edge_index + edge_attr
    y       = PU labels (1 positive / 0 sampled-negative / -1 unlabeled)
    masks   = train/val/test over LABELED nodes (>=2 split strategies)

Also persisted: index<->string_id<->symbol<->ensembl maps, a node table, and split masks.
Validation asserts shapes/dtypes, edge-index bounds, and absence of NaNs (Section 14).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .config import Config, write_run_manifest
from .differential_expression import log_cpm
from .identifiers import map_symbols_to_string
from .labels import load_positive_labels, make_pu_label_vector, sample_negatives
from .string_network import (
    build_networkx_graph,
    detect_communities_leiden,
    download_string_files,
    filter_links_to_nodes,
    load_string_aliases,
    load_string_info,
    load_string_links,
    per_node_centralities,
    recompute_combined_score,
    save_network,
    DETAILED_CHANNELS,
)

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["log2fc", "neglog10_fdr", "mean_expression", "expression_variance", "effect_size"]


# ======================================================================================
# Node features
# ======================================================================================
def collapse_de_to_symbol(de: pd.DataFrame, expr_stats: pd.DataFrame) -> pd.DataFrame:
    """Collapse Ensembl-level DE + expression stats to one representative row per symbol.

    Representative = the Ensembl id with the largest ``baseMean`` (most-expressed isoform),
    which resolves the multiple-Ensembl-per-symbol case deterministically.
    """
    df = de.join(expr_stats, how="left")
    df = df[df["gene_name"].notna() & (df["gene_name"] != "")]
    df = df.sort_values("baseMean", ascending=False)
    rep = df.drop_duplicates(subset="gene_name", keep="first").copy()
    logger.info("Collapsed DE %d Ensembl rows -> %d unique symbols.", len(de), len(rep))
    return rep.set_index("gene_name")


def expression_stats(counts: pd.DataFrame) -> pd.DataFrame:
    """Per-gene mean and variance of log-CPM across samples (index = gene_id)."""
    lcpm = log_cpm(counts)
    return pd.DataFrame(
        {"mean_expression": lcpm.mean(axis=1).to_numpy(),
         "expression_variance": lcpm.var(axis=1).to_numpy()},
        index=counts.index,
    )


def build_node_feature_frame(de_by_symbol: pd.DataFrame) -> pd.DataFrame:
    """Assemble the raw feature frame (index = symbol) with the FEATURE_COLUMNS."""
    feats = pd.DataFrame(index=de_by_symbol.index)
    feats["log2fc"] = de_by_symbol["log2fc"].to_numpy()
    feats["neglog10_fdr"] = de_by_symbol["neglog10_fdr"].to_numpy()
    feats["mean_expression"] = de_by_symbol["mean_expression"].to_numpy()
    feats["expression_variance"] = de_by_symbol["expression_variance"].to_numpy()
    # effect size = Wald statistic (log2fc / lfcSE), already in DE 'stat' column
    stat = de_by_symbol["stat"] if "stat" in de_by_symbol.columns else de_by_symbol["log2fc"]
    feats["effect_size"] = np.asarray(stat, dtype=float)
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return feats


# ======================================================================================
# Edge construction
# ======================================================================================
def build_bidirectional_edge_index(
    edges: pd.DataFrame, node_index: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Undirected STRING edges -> bidirectional PyG edge_index [2, 2E] + edge_attr [2E]."""
    src = edges["protein1"].map(node_index).to_numpy()
    dst = edges["protein2"].map(node_index).to_numpy()
    w = edges["edge_weight"].to_numpy(dtype=np.float32)
    if np.isnan(src).any() or np.isnan(dst).any():
        raise ValueError("Edge endpoints contain nodes not present in node_index.")
    src = src.astype(np.int64)
    dst = dst.astype(np.int64)
    ei = np.vstack([np.concatenate([src, dst]), np.concatenate([dst, src])])
    ea = np.concatenate([w, w])
    return torch.from_numpy(ei), torch.from_numpy(ea)


# ======================================================================================
# Splits (Section 15)
# ======================================================================================
def make_splits(
    y: np.ndarray,
    degrees: np.ndarray,
    *,
    strategy: str,
    val_frac: float,
    test_frac: float,
    seed: int,
    n_degree_bins: int = 5,
) -> dict[str, np.ndarray]:
    """Train/val/test boolean masks over labeled nodes (y in {0,1}).

    - ``stratified_random``: stratify by class only.
    - ``degree_aware``: stratify by (class, degree-quantile-bin) so the degree distribution is
      matched across splits (reduces the hub confound / optimistic bias from random splits).
    """
    rng = np.random.default_rng(seed)
    labeled = np.where(y >= 0)[0]
    n = len(labeled)
    train = np.zeros(len(y), dtype=bool)
    val = np.zeros(len(y), dtype=bool)
    test = np.zeros(len(y), dtype=bool)

    if strategy == "stratified_random":
        strata = [labeled[y[labeled] == c] for c in (0, 1)]
    elif strategy == "degree_aware":
        deg = degrees[labeled]
        edges = np.unique(np.quantile(deg, np.linspace(0, 1, n_degree_bins + 1)))
        dbin = np.digitize(deg, edges)
        strata = []
        for c in (0, 1):
            for b in np.unique(dbin):
                strata.append(labeled[(y[labeled] == c) & (dbin == b)])
    else:
        raise ValueError(f"Unknown split strategy {strategy!r}.")

    for group in strata:
        group = rng.permutation(group)
        n_g = len(group)
        n_test = int(round(test_frac * n_g))
        n_val = int(round(val_frac * n_g))
        test[group[:n_test]] = True
        val[group[n_test:n_test + n_val]] = True
        train[group[n_test + n_val:]] = True

    logger.info("Split %s: train=%d val=%d test=%d (labeled=%d).",
                strategy, train.sum(), val.sum(), test.sum(), n)
    return {"train": train, "val": val, "test": test}


# ======================================================================================
# Orchestration: run_build_graph (CLI stage)
# ======================================================================================
def run_build_graph(cfg: Config) -> dict[str, Any]:
    """Build + persist the STRING graph and the PyG Data object for the cohort."""
    cohort = cfg.cohort
    processed = cfg.path("paths.processed") / cohort
    interim = cfg.path("paths.interim") / cohort
    graph_dir = processed / "graph"
    graph_dir.mkdir(parents=True, exist_ok=True)

    # --- inputs from Phase 2 ---
    de_path = processed / "differential_expression_full.tsv"
    counts_path = processed / "filtered_counts.parquet"
    for p in (de_path, counts_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing. Run prepare_expression first.")
    de = pd.read_csv(de_path, sep="\t", index_col=0)
    counts = pd.read_parquet(counts_path)

    # --- node features (symbol level) ---
    expr = expression_stats(counts)
    de_sym = collapse_de_to_symbol(de, expr)
    feats = build_node_feature_frame(de_sym)

    # --- STRING mapping ---
    s = cfg.section("string")
    files = download_string_files(cfg)
    aliases = load_string_aliases(files.aliases)
    info = load_string_info(files.info)
    mapping = map_symbols_to_string(list(feats.index), aliases)
    mapping.unmatched.to_csv(graph_dir / "unmapped_genes_qc.tsv", sep="\t", index=False)
    mapped = mapping.mapped.drop_duplicates("string_protein_id").drop_duplicates("gene_name")
    logger.info("Mapped %d/%d symbols to STRING (%.1f%%).",
                len(mapped), len(feats), 100 * len(mapped) / len(feats))

    # node table: one row per (symbol, string_id) with features
    node_df = mapped.merge(feats, left_on="gene_name", right_index=True, how="inner")
    node_df = node_df.reset_index(drop=True)
    node_df["node_index"] = np.arange(len(node_df))
    node_index = dict(zip(node_df["string_protein_id"], node_df["node_index"]))
    node_string_ids = set(node_df["string_protein_id"])

    # --- STRING edges among nodes ---
    network_type = s["network_type"]
    detailed = network_type == "functional"
    links = load_string_links(
        files.links_detailed if detailed else files.physical_links, detailed=detailed
    )
    if detailed and bool(s.get("drop_textmining_channel", False)):
        channels = [c for c in DETAILED_CHANNELS if c != "textmining"]
        links = links.copy()
        links["combined_score"] = recompute_combined_score(links, channels)
        logger.info("Recomputed combined score WITHOUT text-mining channel (sensitivity).")
    edges = filter_links_to_nodes(links, node_string_ids, score_threshold=int(s["score_threshold"]))
    logger.info("Edges among %d nodes: %d (score>=%d, %s).",
                len(node_df), len(edges), s["score_threshold"], network_type)

    # --- networkx + centralities + communities ---
    g = build_networkx_graph(edges, list(node_df["string_protein_id"]))
    cent = per_node_centralities(g, seed=cfg.seed)
    communities = detect_communities_leiden(g, seed=cfg.seed)
    node_df = node_df.merge(cent, on="string_protein_id", how="left")
    node_df["community"] = node_df["string_protein_id"].map(communities).astype("Int64")
    node_df["preferred_name"] = node_df["string_protein_id"].map(
        info.set_index("string_protein_id")["preferred_name"]
    )

    # --- labels (PU) ---
    labels = load_positive_labels(cfg)
    pos_ids = node_df.index[node_df["gene_name"].isin(labels.positives)].to_numpy()
    degrees = node_df["degree"].to_numpy()
    lab_cfg = cfg.section("labels")
    unlabeled = [i for i in range(len(node_df)) if i not in set(pos_ids)]
    neg_ids, neg_qc = sample_negatives(
        list(pos_ids), unlabeled, degrees,
        strategy=lab_cfg.get("negative_sampling", "degree_matched"), seed=cfg.seed,
    )
    y = make_pu_label_vector(len(node_df), list(pos_ids), neg_ids)
    node_df["label"] = y
    logger.info("Labels: %d positives (in-graph), %d negatives, %d unlabeled.",
                int((y == 1).sum()), int((y == 0).sum()), int((y == -1).sum()))

    # --- edge_index / tensors ---
    edge_index, edge_attr = build_bidirectional_edge_index(edges, node_index)
    x = torch.tensor(node_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32))
    y_t = torch.tensor(y, dtype=torch.long)

    # --- splits (>=2 strategies) ---
    split_cfg = cfg.section("split")
    all_masks: dict[str, dict[str, np.ndarray]] = {}
    for strat in split_cfg["strategies"]:
        all_masks[strat] = make_splits(
            y, degrees, strategy=strat,
            val_frac=float(split_cfg["val_frac"]), test_frac=float(split_cfg["test_frac"]),
            seed=cfg.seed,
        )
    primary = split_cfg["strategies"][0]
    data = Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_t,
        train_mask=torch.tensor(all_masks[primary]["train"]),
        val_mask=torch.tensor(all_masks[primary]["val"]),
        test_mask=torch.tensor(all_masks[primary]["test"]),
    )
    data.feature_names = FEATURE_COLUMNS
    data.num_nodes = len(node_df)

    _validate_data(data, len(node_df))

    # --- persist ---
    torch.save(data, graph_dir / "pyg_data.pt")
    node_df.to_parquet(graph_dir / "node_table.parquet")
    save_network(edges, node_df[["node_index", "string_protein_id", "gene_name",
                                 "preferred_name", "degree", "community", "label"]],
                 g, graph_dir)
    np.savez(graph_dir / "split_masks.npz",
             **{f"{strat}_{k}": v for strat, m in all_masks.items() for k, v in m.items()})
    mappings = {
        "index_to_string": dict(zip(node_df["node_index"].astype(str), node_df["string_protein_id"])),
        "symbol_to_index": dict(zip(node_df["gene_name"], node_df["node_index"].astype(int))),
        "symbol_to_string": dict(zip(node_df["gene_name"], node_df["string_protein_id"])),
    }
    with open(graph_dir / "mappings.json", "w") as fh:
        json.dump(mappings, fh)

    summary = {
        "cohort": cohort,
        "n_nodes": int(len(node_df)),
        "n_edges_undirected": int(len(edges)),
        "n_edges_bidirectional": int(edge_index.shape[1]),
        "n_features": len(FEATURE_COLUMNS),
        "n_positives": int((y == 1).sum()),
        "n_negatives": int((y == 0).sum()),
        "network_type": network_type,
        "score_threshold": int(s["score_threshold"]),
        "mapping_rate": round(len(mapped) / len(feats), 4),
        "negative_sampling_qc": neg_qc,
        "outputs": {"pyg_data": str(graph_dir / "pyg_data.pt"),
                    "node_table": str(graph_dir / "node_table.parquet"),
                    "graph_dir": str(graph_dir)},
    }
    write_run_manifest(cfg, stage="build_string_graph", seed=cfg.seed,
                       input_files={"de_full": de_path, "filtered_counts": counts_path},
                       extra=summary)
    logger.info("Graph built: %d nodes, %d undirected edges, %d positives.",
                summary["n_nodes"], summary["n_edges_undirected"], summary["n_positives"])
    return summary


def _validate_data(data: Data, n_nodes: int) -> None:
    """Assert PyG Data integrity (Section 14)."""
    assert data.x.shape[0] == n_nodes, "x rows != n_nodes"
    assert data.edge_index.dtype == torch.long, "edge_index must be long"
    if data.edge_index.numel():
        assert int(data.edge_index.max()) < n_nodes, "edge_index out of bounds"
        assert int(data.edge_index.min()) >= 0, "edge_index has negative index"
    assert not torch.isnan(data.x).any(), "NaN in node features"
    assert not torch.isinf(data.x).any(), "Inf in node features"
    assert data.y.shape[0] == n_nodes, "y length != n_nodes"
    logger.info("Data validation passed: x=%s, edge_index=%s.",
                tuple(data.x.shape), tuple(data.edge_index.shape))
