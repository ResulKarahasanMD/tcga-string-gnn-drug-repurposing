"""Disease-gene labels for supervised prioritization (Sections 2, 15).

DE results are NEVER used as labels (that would be circular). Positives come from external
curated evidence. Default: **Open Targets** (open, no login), with loaders for COSMIC Cancer
Gene Census, OncoKB, and DisGeNET (the latter three need registration/files/tokens and error
clearly when unavailable).

Positive-unlabeled (PU) framing (Section 15): genes NOT in the positive set are *unlabeled*,
not guaranteed negatives. We therefore sample a defensible negative set (optionally
degree-matched) and leave the rest unlabeled (excluded from the loss). The unsupervised GAE
path (Approach B, ``models.py``) needs no labels at all.

Open Targets GraphQL verified live 2026-07:
    endpoint https://api.platform.opentargets.org/api/v4/graphql
    disease(efoId){ associatedTargets(page){ rows{ target{approvedSymbol} score } } }
    (disease ids are MONDO/EFO, e.g. lung adenocarcinoma = MONDO_0005061)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from .config import Config

logger = logging.getLogger(__name__)

OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"

_OT_ASSOC_QUERY = """
query($id: String!, $index: Int!, $size: Int!) {
  disease(efoId: $id) {
    id
    name
    associatedTargets(page: {index: $index, size: $size}) {
      count
      rows { target { id approvedSymbol } score }
    }
  }
}
"""


@dataclass
class LabelSet:
    """Positive gene symbols + provenance for the supervised task."""

    positives: set[str]
    source: str
    detail: pd.DataFrame  # gene_name, score/tier, ...
    meta: dict[str, Any]


# ======================================================================================
# Open Targets (default)
# ======================================================================================
def fetch_opentargets_targets(
    disease_id: str,
    *,
    min_score: float = 0.5,
    page_size: int = 200,
    session: requests.Session | None = None,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch disease-associated targets sorted by descending overall association score.

    Rows are returned score-descending, so we stop paging once a page falls entirely below
    ``min_score``. Returns columns ``[gene_name, target_id, score]`` (only score >= min_score).
    """
    own = session is None
    session = session or requests.Session()
    rows: list[dict[str, Any]] = []
    try:
        index = 0
        disease_name = None
        while True:
            resp = session.post(
                OPENTARGETS_GQL,
                json={"query": _OT_ASSOC_QUERY,
                      "variables": {"id": disease_id, "index": index, "size": page_size}},
                timeout=timeout,
            )
            if resp.status_code != 200:
                raise RuntimeError(f"Open Targets HTTP {resp.status_code}: {resp.text[:300]}")
            payload = resp.json()
            if payload.get("errors"):
                raise RuntimeError(f"Open Targets GraphQL errors: {payload['errors']}")
            disease = payload["data"]["disease"]
            if disease is None:
                raise ValueError(
                    f"Open Targets returned no disease for id={disease_id!r}. "
                    "Check the id (use the OT search endpoint; ids are MONDO/EFO)."
                )
            disease_name = disease["name"]
            page = disease["associatedTargets"]
            total = page["count"]
            batch = page["rows"]
            if not batch:
                break
            for r in batch:
                rows.append({"gene_name": r["target"]["approvedSymbol"],
                             "target_id": r["target"]["id"], "score": float(r["score"])})
            if batch[-1]["score"] < min_score:  # sorted desc -> everything below is < min_score
                break
            index += 1
            if index * page_size >= total:
                break
    finally:
        if own:
            session.close()

    df = pd.DataFrame(rows, columns=["gene_name", "target_id", "score"])
    df = df[df["score"] >= min_score].drop_duplicates("gene_name").reset_index(drop=True)
    logger.info("Open Targets: %d targets with score>=%.2f for %s (%s).",
                len(df), min_score, disease_id, disease_name)
    return df


# ======================================================================================
# Alternative curated sources
# ======================================================================================
def load_cosmic_cgc(path: Path, tiers: list[int]) -> pd.DataFrame:
    """Load COSMIC Cancer Gene Census CSV (user-supplied; registration required)."""
    if not path.exists():
        raise FileNotFoundError(
            f"COSMIC CGC file not found: {path}. Register at "
            "https://cancer.sanger.ac.uk/cosmic and place cancer_gene_census.csv there, "
            "or set labels.source=opentargets."
        )
    df = pd.read_csv(path)
    sym_col = next((c for c in df.columns if c.lower() in {"gene symbol", "gene_symbol", "symbol"}), None)
    tier_col = next((c for c in df.columns if c.lower() == "tier"), None)
    if sym_col is None:
        raise ValueError(f"Could not find a gene-symbol column in {path}. Columns: {list(df.columns)}")
    out = df[[sym_col] + ([tier_col] if tier_col else [])].rename(
        columns={sym_col: "gene_name", **({tier_col: "tier"} if tier_col else {})}
    )
    if tier_col:
        out = out[out["tier"].isin(tiers)]
    return out.drop_duplicates("gene_name").reset_index(drop=True)


def load_oncokb_genes(session: requests.Session | None = None, timeout: int = 60) -> pd.DataFrame:
    """Load OncoKB curated cancer genes (requires ONCOKB_TOKEN env var / registration)."""
    token = os.environ.get("ONCOKB_TOKEN")
    if not token:
        raise RuntimeError(
            "OncoKB requires an API token. Register at https://www.oncokb.org/, set "
            "ONCOKB_TOKEN in your environment, or use labels.source=opentargets."
        )
    own = session is None
    session = session or requests.Session()
    try:
        resp = session.get(
            "https://www.oncokb.org/api/v1/utils/allCuratedGenes",
            headers={"Authorization": f"Bearer {token}"}, timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"OncoKB HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
    finally:
        if own:
            session.close()
    df = pd.DataFrame(data)
    return pd.DataFrame({"gene_name": df["hugoSymbol"].drop_duplicates().reset_index(drop=True)})


def load_disgenet_genes(disease_umls: str, timeout: int = 60) -> pd.DataFrame:
    """Load DisGeNET gene-disease associations (requires DISGENET_API_KEY; account since 2023)."""
    api_key = os.environ.get("DISGENET_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DisGeNET requires an API key. Register at https://www.disgenet.com/, set "
            "DISGENET_API_KEY, or use labels.source=opentargets."
        )
    headers = {"Authorization": api_key, "accept": "application/json"}
    url = f"https://api.disgenet.com/api/v1/gda/summary?disease={disease_umls}"
    resp = requests.get(url, headers=headers, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"DisGeNET HTTP {resp.status_code}: {resp.text[:200]}")
    payload = resp.json().get("payload", [])
    syms = sorted({row["gene_symbol"] for row in payload if row.get("gene_symbol")})
    return pd.DataFrame({"gene_name": syms})


# ======================================================================================
# Dispatch
# ======================================================================================
def load_positive_labels(cfg: Config, session: requests.Session | None = None) -> LabelSet:
    """Load positive disease genes according to ``labels.source``."""
    lab = cfg.section("labels")
    source = lab["source"]
    if source == "opentargets":
        ot = lab["opentargets"]
        df = fetch_opentargets_targets(
            ot["disease_id"], min_score=float(ot["min_association_score"]), session=session
        )
        return LabelSet(set(df["gene_name"]), "opentargets", df,
                        {"disease_id": ot["disease_id"], "min_score": ot["min_association_score"]})
    if source == "cosmic_cgc":
        cc = lab["cosmic_cgc"]
        df = load_cosmic_cgc(Path(cc["file"]) if Path(cc["file"]).is_absolute()
                             else cfg.resolve_path(cc["file"]), cc["tier"])
        return LabelSet(set(df["gene_name"]), "cosmic_cgc", df, {"tiers": cc["tier"]})
    if source == "oncokb":
        df = load_oncokb_genes(session=session)
        return LabelSet(set(df["gene_name"]), "oncokb", df, {})
    if source == "disgenet":
        df = load_disgenet_genes(lab.get("disgenet", {}).get("disease_umls", ""))
        return LabelSet(set(df["gene_name"]), "disgenet", df, {})
    raise ValueError(f"Unknown labels.source={source!r}.")


# ======================================================================================
# PU labels + negative sampling (Section 15)
# ======================================================================================
def sample_negatives(
    positive_ids: list[int],
    candidate_ids: list[int],
    degrees: np.ndarray,
    *,
    strategy: str = "degree_matched",
    ratio: float = 1.0,
    n_bins: int = 50,
    seed: int = 11,
) -> tuple[list[int], dict[str, Any]]:
    """Sample negatives from unlabeled candidates (random or degree-matched).

    ``degrees`` is indexed by node id (position). ``candidate_ids`` are unlabeled node ids.
    Degree-matched sampling draws negatives with a node-degree distribution similar to the
    positives (mitigating the hub-bias confound). Returns ``(negative_ids, qc)``.
    """
    rng = np.random.default_rng(seed)
    n_neg = int(round(ratio * len(positive_ids)))
    candidate_ids = [c for c in candidate_ids if c not in set(positive_ids)]
    n_neg = min(n_neg, len(candidate_ids))

    if strategy == "random":
        negatives = list(rng.choice(candidate_ids, size=n_neg, replace=False))
    elif strategy == "degree_matched":
        pos_deg = degrees[positive_ids]
        cand_arr = np.array(candidate_ids)
        cand_deg = degrees[cand_arr]
        # bin by degree quantiles of the whole graph; match counts per bin to positives
        edges = np.unique(np.quantile(degrees, np.linspace(0, 1, n_bins + 1)))
        pos_bins = np.digitize(pos_deg, edges)
        cand_bins = np.digitize(cand_deg, edges)
        negatives: list[int] = []
        for b in np.unique(pos_bins):
            want = int(np.sum(pos_bins == b))
            pool = cand_arr[cand_bins == b]
            if len(pool) == 0:
                continue
            take = min(want, len(pool))
            negatives.extend(rng.choice(pool, size=take, replace=False).tolist())
        # top up if under-filled due to sparse bins
        if len(negatives) < n_neg:
            remaining = list(set(candidate_ids) - set(negatives))
            extra = rng.choice(remaining, size=min(n_neg - len(negatives), len(remaining)),
                               replace=False)
            negatives.extend(int(x) for x in extra)
    else:
        raise ValueError(f"Unknown negative-sampling strategy {strategy!r}.")

    negatives = [int(x) for x in negatives]
    qc = {
        "strategy": strategy,
        "n_positive": len(positive_ids),
        "n_negative": len(negatives),
        "pos_degree_mean": float(np.mean(degrees[positive_ids])) if positive_ids else 0.0,
        "neg_degree_mean": float(np.mean(degrees[negatives])) if negatives else 0.0,
    }
    logger.info("Negatives (%s): %d sampled | pos deg mean=%.1f vs neg deg mean=%.1f",
                strategy, len(negatives), qc["pos_degree_mean"], qc["neg_degree_mean"])
    return negatives, qc


def make_pu_label_vector(
    n_nodes: int,
    positive_ids: list[int],
    negative_ids: list[int],
) -> np.ndarray:
    """Return a label vector: 1=positive, 0=negative, -1=unlabeled (excluded from loss)."""
    y = np.full(n_nodes, -1, dtype=np.int64)
    y[np.asarray(positive_ids, dtype=int)] = 1
    if negative_ids:
        y[np.asarray(negative_ids, dtype=int)] = 0
    return y
