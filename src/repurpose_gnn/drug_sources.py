"""Open drug-target sources (Section 4, 23-24). Default: DGIdb (open, no login).

Field mapping across sources (all normalized to the columns below):
    gene_name | drug_name | drug_id | approved | action | source | interaction_score
- **DGIdb**  drug-gene interactions with interaction types (inhibitor/agonist/...) -> `action`.
- **Open Targets** `knownDrugs` with `mechanismOfAction`/`actionType` -> `action`.
- **ChEMBL**  mechanisms endpoint (`mechanism_of_action`, `action_type`).
- **DrugBank** (license-restricted) parsed in ``drugbank.py``; same normalized columns.

DGIdb v5 GraphQL verified 2026-07: endpoint https://dgidb.org/api/graphql,
``genes(names:[...]){nodes{interactions{drug{name conceptId approved} interactionScore
interactionTypes{type directionality}}}}``.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import Config

logger = logging.getLogger(__name__)

DGIDB_GQL = "https://dgidb.org/api/graphql"

_DGIDB_QUERY = """
query($names: [String!]!) {
  genes(names: $names) {
    nodes {
      name
      conceptId
      interactions {
        drug { name conceptId approved }
        interactionScore
        interactionTypes { type directionality }
      }
    }
  }
}
"""


# ======================================================================================
# DGIdb (default)
# ======================================================================================
def fetch_dgidb_interactions(
    gene_symbols: list[str],
    *,
    chunk_size: int = 120,
    session: requests.Session | None = None,
    timeout: int = 90,
    pause: float = 0.2,
) -> pd.DataFrame:
    """Fetch drug-gene interactions for a list of gene symbols (batched GraphQL)."""
    own = session is None
    session = session or requests.Session()
    rows: list[dict[str, Any]] = []
    try:
        n = len(gene_symbols)
        for i in range(0, n, chunk_size):
            names = gene_symbols[i:i + chunk_size]
            resp = session.post(DGIDB_GQL, json={"query": _DGIDB_QUERY, "variables": {"names": names}},
                                timeout=timeout)
            if resp.status_code != 200:
                raise RuntimeError(f"DGIdb HTTP {resp.status_code}: {resp.text[:200]}")
            payload = resp.json()
            if payload.get("errors"):
                raise RuntimeError(f"DGIdb GraphQL errors: {payload['errors']}")
            for node in payload["data"]["genes"]["nodes"]:
                gene = node["name"]
                for inter in node["interactions"]:
                    actions = sorted({t["type"] for t in inter["interactionTypes"] if t.get("type")})
                    rows.append({
                        "gene_name": gene,
                        "drug_name": inter["drug"]["name"],
                        "drug_id": inter["drug"]["conceptId"],
                        "approved": bool(inter["drug"]["approved"]),
                        "action": ";".join(actions) if actions else "",
                        "interaction_score": inter.get("interactionScore"),
                        "source": "dgidb",
                    })
            if (i // chunk_size) % 10 == 0:
                logger.info("DGIdb: queried %d/%d genes, %d interactions so far.",
                            min(i + chunk_size, n), n, len(rows))
            time.sleep(pause)
    finally:
        if own:
            session.close()
    df = pd.DataFrame(rows).drop_duplicates(subset=["drug_id", "gene_name", "action"])
    logger.info("DGIdb: %d interactions across %d drugs, %d genes.",
                len(df), df["drug_id"].nunique() if len(df) else 0, df["gene_name"].nunique() if len(df) else 0)
    return df


# ======================================================================================
# Open Targets drug data (alternative)
# ======================================================================================
_OT_KNOWN_DRUGS = """
query($ensemblId: String!, $size: Int!) {
  target(ensemblId: $ensemblId) {
    knownDrugs(size: $size) {
      rows { drugId prefName mechanismOfAction drugType }
    }
  }
}
"""


def fetch_opentargets_drugs(ensembl_ids: dict[str, str], *, session=None, timeout=60) -> pd.DataFrame:
    """Fetch known drugs per target from Open Targets (``ensembl_ids`` maps symbol->ENSG)."""
    own = session is None
    session = session or requests.Session()
    endpoint = "https://api.platform.opentargets.org/api/v4/graphql"
    rows: list[dict[str, Any]] = []
    try:
        for sym, ensg in ensembl_ids.items():
            resp = session.post(endpoint, json={"query": _OT_KNOWN_DRUGS,
                                "variables": {"ensemblId": ensg, "size": 200}}, timeout=timeout)
            if resp.status_code != 200:
                continue
            tgt = resp.json().get("data", {}).get("target")
            if not tgt or not tgt.get("knownDrugs"):
                continue
            for r in tgt["knownDrugs"]["rows"]:
                rows.append({"gene_name": sym, "drug_name": r.get("prefName"), "drug_id": r.get("drugId"),
                             "approved": None, "action": (r.get("mechanismOfAction") or ""),
                             "drug_type": r.get("drugType"), "source": "opentargets"})
            time.sleep(0.05)
    finally:
        if own:
            session.close()
    return pd.DataFrame(rows).drop_duplicates(subset=["drug_id", "gene_name"])


# ======================================================================================
# Normalization -> drug -> target sets
# ======================================================================================
def interactions_to_drug_targets(
    interactions: pd.DataFrame,
    *,
    approved_only: bool = True,
    known_action_only: bool = True,
) -> tuple[dict[str, set[str]], pd.DataFrame]:
    """Collapse interactions into {drug_id: set(gene targets)} + a drug metadata table."""
    df = interactions.copy()
    if approved_only and "approved" in df.columns and df["approved"].notna().any():
        df = df[df["approved"] == True]  # noqa: E712
    if known_action_only and "action" in df.columns:
        df = df[df["action"].astype(str).str.len() > 0]
    if df.empty:
        logger.warning("No interactions left after approved/known-action filters.")
    drug_targets = df.groupby("drug_id")["gene_name"].apply(lambda s: set(s)).to_dict()
    drug_meta = (
        df.groupby("drug_id")
        .agg(drug_name=("drug_name", "first"),
             approved=("approved", "first"),
             n_targets=("gene_name", "nunique"),
             actions=("action", lambda s: ";".join(sorted({a for x in s for a in str(x).split(";") if a}))),
             source=("source", "first"))
        .reset_index()
    )
    return drug_targets, drug_meta


def load_drug_targets(cfg: Config, gene_symbols: list[str], session=None):
    """Dispatch by ``drugs.source``; returns (interactions_df, drug_targets, drug_meta)."""
    dcfg = cfg.section("drugs")
    source = dcfg["source"]
    if source == "drugbank":
        from .drugbank import load_drugbank_targets
        return load_drugbank_targets(cfg, gene_symbols)
    if source == "dgidb":
        inter = fetch_dgidb_interactions(gene_symbols, session=session)
    elif source == "opentargets":
        raise NotImplementedError("opentargets drug source needs symbol->ENSG map; use dgidb default.")
    else:
        raise ValueError(f"Unknown drugs.source={source!r}.")
    drug_targets, drug_meta = interactions_to_drug_targets(
        inter, approved_only=bool(dcfg.get("approved_only", True)),
        known_action_only=bool(dcfg.get("known_action_only", True)),
    )
    return inter, drug_targets, drug_meta


# ======================================================================================
# CLI stage: map drug targets into the STRING network (Section 24)
# ======================================================================================
def run_map_drug_targets(cfg: Config) -> dict[str, Any]:
    """Fetch drug targets, map them to STRING node indices, report mapping stats."""
    import json

    from .config import write_run_manifest

    graph_dir = cfg.path("paths.processed") / cfg.cohort / "graph"
    node_table = pd.read_parquet(graph_dir / "node_table.parquet")
    with open(graph_dir / "mappings.json") as fh:
        mappings = json.load(fh)
    symbol_to_index = mappings["symbol_to_index"]
    network_symbols = list(symbol_to_index.keys())

    # DrugBank if a real XML is present, else the open default (config drives this).
    drugbank_xml = cfg.get("drugs.drugbank_xml", None)
    import os
    if os.environ.get("DRUGBANK_XML"):
        drugbank_xml = os.environ["DRUGBANK_XML"]
    if cfg.get("drugs.source") == "dgidb" and drugbank_xml and Path(drugbank_xml).exists():
        logger.info("Local DrugBank XML found -> using DrugBank instead of DGIdb.")
        cfg.raw["drugs"]["source"] = "drugbank"

    inter, drug_targets, drug_meta = load_drug_targets(cfg, network_symbols)

    interim = cfg.path("paths.interim") / cfg.cohort
    interim.mkdir(parents=True, exist_ok=True)
    inter.to_parquet(interim / "drug_interactions.parquet")

    # map each drug's targets to node indices; report drugs with no mapped targets separately
    mapped_targets: dict[str, list[int]] = {}
    no_target_drugs = []
    for drug_id, targets in drug_targets.items():
        idxs = [symbol_to_index[g] for g in targets if g in symbol_to_index]
        if idxs:
            mapped_targets[drug_id] = idxs
        else:
            no_target_drugs.append(drug_id)
    drug_meta["n_targets_in_network"] = drug_meta["drug_id"].map(
        lambda d: len(mapped_targets.get(d, []))
    )
    drug_meta.to_parquet(interim / "drug_meta.parquet")
    with open(interim / "drug_target_indices.json", "w") as fh:
        json.dump({k: v for k, v in mapped_targets.items()}, fh)

    summary = {
        "cohort": cfg.cohort,
        "source": cfg.get("drugs.source"),
        "n_drugs_total": int(drug_meta.shape[0]),
        "n_drugs_with_network_targets": int(len(mapped_targets)),
        "n_drugs_no_mapped_targets": int(len(no_target_drugs)),
        "n_interactions": int(len(inter)),
        "outputs": {"drug_meta": str(interim / "drug_meta.parquet"),
                    "drug_target_indices": str(interim / "drug_target_indices.json")},
    }
    write_run_manifest(cfg, stage="map_drug_targets", seed=cfg.seed, extra=summary)
    logger.info("Mapped %d drugs with >=1 network target (%d had none).",
                len(mapped_targets), len(no_target_drugs))
    return summary
