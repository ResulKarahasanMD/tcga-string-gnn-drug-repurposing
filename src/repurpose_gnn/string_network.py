"""STRING protein-protein interaction network (Sections 12-13).

STRING explicitly asks large-scale users to download the bulk flat files rather than hammer
the API (1 req/sec fair-use). We therefore use the species-filtered v12.0 bulk files:

    protein.info.v12.0        preferred names / annotation
    protein.aliases.v12.0     symbol/Ensembl/UniProt -> STRING id mapping
    protein.links.detailed    functional network with per-channel scores
    protein.physical.links    physical-interaction subnetwork

Host verified 2026-07: ``https://stringdb-downloads.org/download/<type>.v12.0/9606.<type>.v12.0.txt.gz``

Edge weight is ``combined_score / 1000`` (Section 12). We can recompute the combined score
with the text-mining channel removed (the STRING probabilistic-OR formula with prior
correction) for the required text-mining sensitivity analysis.
"""

from __future__ import annotations

import gzip
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import requests

from .config import Config

logger = logging.getLogger(__name__)

STRING_DOWNLOAD_HOST = "https://stringdb-downloads.org/download"
STRING_PRIOR = 0.041  # STRING combined-score prior probability (Szklarczyk et al.)

# per-channel score columns in the *.links.detailed file (space-separated)
DETAILED_CHANNELS = [
    "neighborhood", "fusion", "cooccurence", "coexpression",
    "experimental", "database", "textmining",
]


@dataclass(frozen=True)
class StringFileSet:
    info: Path
    aliases: Path
    links_detailed: Path
    physical_links: Path


# ======================================================================================
# Download (idempotent; URLs verified before use)
# ======================================================================================
def string_file_urls(version: str, species: int) -> dict[str, str]:
    """Return {local_name: url} for the species-filtered STRING v12 bulk files."""
    v = f"v{version}"
    types = {
        "info": f"protein.info.{v}/{species}.protein.info.{v}.txt.gz",
        "aliases": f"protein.aliases.{v}/{species}.protein.aliases.{v}.txt.gz",
        "links_detailed": f"protein.links.detailed.{v}/{species}.protein.links.detailed.{v}.txt.gz",
        "physical_links": f"protein.physical.links.{v}/{species}.protein.physical.links.{v}.txt.gz",
    }
    return {name: f"{STRING_DOWNLOAD_HOST}/{path}" for name, path in types.items()}


def download_string_files(cfg: Config, *, timeout: int = 600) -> StringFileSet:
    """Download the four bulk files into ``paths.raw_string`` if absent. Returns their paths."""
    s = cfg.section("string")
    out_dir = cfg.path("paths.raw_string")
    out_dir.mkdir(parents=True, exist_ok=True)
    urls = string_file_urls(str(s["version"]), int(s["species"]))
    paths: dict[str, Path] = {}
    with requests.Session() as session:
        session.headers.update({"User-Agent": "repurpose-gnn/0.1 (academic research)"})
        for name, url in urls.items():
            dest = out_dir / url.rsplit("/", 1)[-1]
            paths[name] = dest
            if dest.exists() and dest.stat().st_size > 0:
                logger.info("STRING %s cached: %s", name, dest.name)
                continue
            logger.info("Downloading STRING %s -> %s", name, dest.name)
            with session.get(url, stream=True, timeout=timeout) as resp:
                if resp.status_code != 200:
                    raise RuntimeError(f"STRING download failed [{resp.status_code}] for {url}")
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(resp.raw, fh)
                tmp.replace(dest)
    return StringFileSet(paths["info"], paths["aliases"],
                         paths["links_detailed"], paths["physical_links"])


# ======================================================================================
# Loaders
# ======================================================================================
def load_string_info(path: Path) -> pd.DataFrame:
    """Load protein.info: columns [string_protein_id, preferred_name, protein_size, annotation]."""
    df = pd.read_csv(path, sep="\t", compression="gzip")
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    df = df.rename(columns={"#string_protein_id": "string_protein_id",
                            "string_protein_id": "string_protein_id"})
    return df


def load_string_aliases(path: Path) -> pd.DataFrame:
    """Load protein.aliases into columns [string_protein_id, alias, source] (identifiers-ready)."""
    df = pd.read_csv(path, sep="\t", compression="gzip")
    df.columns = [c.lstrip("#").strip() for c in df.columns]
    ren = {}
    for c in df.columns:
        cl = c.lower()
        if "protein" in cl:
            ren[c] = "string_protein_id"
        elif cl == "alias":
            ren[c] = "alias"
        elif "source" in cl:
            ren[c] = "source"
    df = df.rename(columns=ren)
    return df[["string_protein_id", "alias", "source"]]


def load_string_links(path: Path, *, detailed: bool) -> pd.DataFrame:
    """Load a STRING links file (space-separated). detailed=True expects per-channel columns."""
    df = pd.read_csv(path, sep=" ", compression="gzip")
    df.columns = [c.strip() for c in df.columns]
    needed = {"protein1", "protein2", "combined_score"}
    if not needed.issubset(df.columns):
        raise ValueError(f"{path.name}: expected columns {needed}, got {list(df.columns)}.")
    return df


# ======================================================================================
# Combined-score recomputation (text-mining sensitivity)
# ======================================================================================
def _remove_prior(score01: np.ndarray, prior: float = STRING_PRIOR) -> np.ndarray:
    """STRING prior correction: s' = (s - p)/(1 - p), clipped to [0, 1]."""
    out = (score01 - prior) / (1.0 - prior)
    return np.clip(out, 0.0, 1.0)


def recompute_combined_score(
    links: pd.DataFrame,
    channels: list[str],
    *,
    prior: float = STRING_PRIOR,
) -> np.ndarray:
    """Recompute the STRING combined score from a subset of channels (values 0..1000).

    Uses the documented probabilistic-OR with prior correction:
        s'_i = (s_i/1000 - p)/(1 - p);  combined' = 1 - prod(1 - s'_i);
        combined = combined' * (1 - p) + p   (then * 1000).
    Dropping ``textmining`` from ``channels`` yields the text-mining-free network.
    """
    missing = [c for c in channels if c not in links.columns]
    if missing:
        raise ValueError(f"Cannot recompute combined score; missing channels {missing}.")
    one_minus = np.ones(len(links), dtype=float)
    for ch in channels:
        s_prime = _remove_prior(links[ch].to_numpy(dtype=float) / 1000.0, prior)
        one_minus *= (1.0 - s_prime)
    combined_prime = 1.0 - one_minus
    combined = combined_prime * (1.0 - prior) + prior
    return np.round(combined * 1000.0).astype(int)


# ======================================================================================
# Network assembly
# ======================================================================================
def filter_links_to_nodes(
    links: pd.DataFrame,
    node_string_ids: set[str],
    *,
    score_threshold: int,
    score_column: str = "combined_score",
) -> pd.DataFrame:
    """Keep edges whose BOTH endpoints are in ``node_string_ids`` and score >= threshold."""
    mask = (
        links["protein1"].isin(node_string_ids)
        & links["protein2"].isin(node_string_ids)
        & (links[score_column] >= score_threshold)
    )
    edges = links.loc[mask, ["protein1", "protein2", score_column]].copy()
    edges = edges.rename(columns={score_column: "combined_score"})
    # STRING lists each undirected pair twice (A-B and B-A); keep one canonical orientation.
    a = np.minimum(edges["protein1"].to_numpy(), edges["protein2"].to_numpy())
    b = np.maximum(edges["protein1"].to_numpy(), edges["protein2"].to_numpy())
    edges["protein1"], edges["protein2"] = a, b
    edges = edges.drop_duplicates(subset=["protein1", "protein2"]).reset_index(drop=True)
    edges["edge_weight"] = edges["combined_score"] / 1000.0
    return edges


def build_networkx_graph(edges: pd.DataFrame, node_ids: list[str]) -> nx.Graph:
    """Build an undirected weighted graph including isolated nodes (degree 0)."""
    g = nx.Graph()
    g.add_nodes_from(node_ids)
    for row in edges.itertuples(index=False):
        g.add_edge(row.protein1, row.protein2,
                   weight=float(row.edge_weight), combined_score=int(row.combined_score))
    return g


# ======================================================================================
# Classical network analysis (Section 13)
# ======================================================================================
def classical_network_stats(g: nx.Graph, *, betweenness_k: int | None = 500,
                            seed: int = 11) -> dict[str, Any]:
    """Summary statistics + per-node centralities. Betweenness is approximate (k-sampled)."""
    n, m = g.number_of_nodes(), g.number_of_edges()
    components = list(nx.connected_components(g))
    giant = max(components, key=len) if components else set()
    degrees = dict(g.degree())
    isolated = [x for x, d in degrees.items() if d == 0]
    stats = {
        "n_nodes": n,
        "n_edges": m,
        "n_components": len(components),
        "giant_component_fraction": len(giant) / n if n else 0.0,
        "n_isolated": len(isolated),
        "mean_degree": float(np.mean(list(degrees.values()))) if n else 0.0,
        "median_degree": float(np.median(list(degrees.values()))) if n else 0.0,
        "max_degree": int(max(degrees.values())) if n else 0,
    }
    logger.info("Network: %d nodes, %d edges, %d components (giant=%.1f%%), %d isolated.",
                n, m, stats["n_components"], 100 * stats["giant_component_fraction"], len(isolated))
    return stats


def per_node_centralities(g: nx.Graph, *, betweenness_k: int | None = 500,
                          seed: int = 11) -> pd.DataFrame:
    """Per-node degree, weighted degree, PageRank, clustering, (approx) betweenness."""
    degree = dict(g.degree())
    wdegree = dict(g.degree(weight="weight"))
    pagerank = nx.pagerank(g, weight="weight") if g.number_of_edges() else {x: 0.0 for x in g}
    clustering = nx.clustering(g, weight="weight")
    if betweenness_k and g.number_of_nodes() > betweenness_k and g.number_of_edges():
        betw = nx.betweenness_centrality(g, k=betweenness_k, weight=None, seed=seed)
    elif g.number_of_edges():
        betw = nx.betweenness_centrality(g, weight=None, seed=seed)
    else:
        betw = {x: 0.0 for x in g}
    return pd.DataFrame(
        {
            "string_protein_id": list(g.nodes()),
            "degree": [degree[x] for x in g.nodes()],
            "weighted_degree": [wdegree[x] for x in g.nodes()],
            "pagerank": [pagerank[x] for x in g.nodes()],
            "clustering": [clustering[x] for x in g.nodes()],
            "betweenness_approx": [betw.get(x, 0.0) for x in g.nodes()],
        }
    )


def detect_communities_leiden(g: nx.Graph, *, seed: int = 11) -> dict[str, int]:
    """Leiden community detection (via igraph). Returns {string_protein_id: community_id}."""
    if g.number_of_edges() == 0:
        return {x: i for i, x in enumerate(g.nodes())}
    import igraph as ig
    import leidenalg as la

    nodes = list(g.nodes())
    idx = {x: i for i, x in enumerate(nodes)}
    edges = [(idx[u], idx[v]) for u, v in g.edges()]
    weights = [g[u][v]["weight"] for u, v in g.edges()]
    ig_graph = ig.Graph(n=len(nodes), edges=edges)
    ig_graph.es["weight"] = weights
    part = la.find_partition(
        ig_graph, la.RBConfigurationVertexPartition, weights="weight", seed=seed
    )
    membership = part.membership
    return {nodes[i]: int(membership[i]) for i in range(len(nodes))}


# ======================================================================================
# Persistence
# ======================================================================================
def save_network(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    graph: nx.Graph,
    out_dir: Path,
) -> dict[str, str]:
    """Write string_edges.tsv, string_nodes.tsv, graph.graphml."""
    out_dir.mkdir(parents=True, exist_ok=True)
    edges_path = out_dir / "string_edges.tsv"
    nodes_path = out_dir / "string_nodes.tsv"
    graphml_path = out_dir / "graph.graphml"
    edges.to_csv(edges_path, sep="\t", index=False)
    nodes.to_csv(nodes_path, sep="\t", index=False)
    # graphml can't hold arbitrary dtypes; write with clean attrs
    nx.write_graphml(graph, graphml_path)
    logger.info("Saved network: %s, %s, %s", edges_path.name, nodes_path.name, graphml_path.name)
    return {"edges": str(edges_path), "nodes": str(nodes_path), "graphml": str(graphml_path)}


def _gzip_head(path: Path, n: int = 3) -> list[str]:
    """Peek at the first n lines of a gzipped file (debug/validation helper)."""
    with gzip.open(path, "rt") as fh:
        return [next(fh).rstrip("\n") for _ in range(n)]
