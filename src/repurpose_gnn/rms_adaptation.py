"""Rhabdomyosarcoma (RMS) + bosutinib adaptation (Section 36).

READ THIS FIRST — data-availability truth:
- **TCGA does NOT contain an independent, sufficiently large pediatric RMS cohort.**
- **TCGA-SARC is NOT rhabdomyosarcoma** (it is predominantly adult soft-tissue sarcomas:
  leiomyosarcoma, dedifferentiated liposarcoma, UPS, MPNST, etc.). Do not substitute it.
- RMS adaptation requires an RMS transcriptomic cohort from another source (see below).

This module therefore does NOT fabricate any RMS or bosutinib-in-RMS result. It provides:
1. A documented list of legitimate RMS data sources.
2. The biological stratification RMS requires (eRMS vs aRMS; PAX3/7::FOXO1 fusion status).
3. A live, evidence-graded verification of bosutinib's targets (real data).
4. A DepMap/CRISPR + wet-lab validation plan.
5. Instructions to plug an RMS expression matrix into the SAME pipeline.

Rule: produce NO RMS/bosutinib efficacy claim without real RMS data. If a computed proximity
is negative, that is a scientific result, not a pipeline failure.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# 1. Legitimate RMS data sources (none is TCGA)
# --------------------------------------------------------------------------------------
RMS_DATA_SOURCES = {
    "TARGET (NCI)": {
        "what": "Therapeutically Applicable Research To Generate Effective Treatments — includes "
                "pediatric RMS (TARGET-RT) RNA-seq + clinical + fusion status.",
        "access": "NCI GDC (open tier) + dbGaP for controlled data; https://ocg.cancer.gov/programs/target",
        "note": "Closest analogue to a 'TCGA for pediatric RMS'; use the GDC client with project TARGET-RT.",
    },
    "St. Jude Cloud": {
        "what": "Pediatric cancer genomics incl. RMS (PCGP / Genomes for Kids) RNA-seq + fusions.",
        "access": "https://www.stjude.cloud/ (registration + data-access agreement).",
    },
    "GEO / ArrayExpress": {
        "what": "Published RMS transcriptomic series (e.g., fusion-stratified cohorts).",
        "access": "https://www.ncbi.nlm.nih.gov/geo/ — search 'rhabdomyosarcoma RNA-seq'.",
    },
    "SRA": {
        "what": "Raw RMS sequencing runs for custom re-quantification (STAR/Salmon).",
        "access": "https://www.ncbi.nlm.nih.gov/sra",
    },
    "DepMap": {
        "what": "RMS cell-line expression + CRISPR/RNAi dependencies + drug sensitivity — for "
                "orthogonal target/drug validation (NOT a patient cohort).",
        "access": "https://depmap.org/portal/ (open downloads).",
    },
}

# --------------------------------------------------------------------------------------
# 2. Required biological stratification
# --------------------------------------------------------------------------------------
RMS_STRATIFICATION = """
RMS is not one disease. Analyze subtypes SEPARATELY:

- Embryonal RMS (eRMS): usually fusion-NEGATIVE; frequent RAS-pathway alterations; better prognosis.
- Alveolar RMS (aRMS): frequently fusion-POSITIVE, driven by PAX3::FOXO1 (t2;13) or
  PAX7::FOXO1 (t1;13). PAX3::FOXO1 and PAX7::FOXO1 differ clinically/biologically and should
  themselves be considered separately where sample size allows.

Consequences for this pipeline:
- Run differential expression and disease-module construction within a subtype/fusion stratum,
  not pooled — pooling mixes distinct driver programs.
- 'Normal comparator' is a hard problem: the cell-of-origin approximates myogenic progenitors /
  regenerating muscle; adult skeletal muscle is an IMPERFECT normal. Consider fetal/myoblast
  references and interpret tumor-vs-normal DE with that caveat. GTEx skeletal muscle is adult.
- Fusion status must come from the data source's annotation or be called from RNA-seq
  (fusion callers) — do not assume it.
"""

# --------------------------------------------------------------------------------------
# 3. Bosutinib target verification (real data)
# --------------------------------------------------------------------------------------
_DGIDB_DRUG_QUERY = """
query($names: [String!]!) {
  drugs(names: $names) {
    nodes {
      name conceptId approved
      interactions {
        gene { name }
        interactionScore
        interactionTypes { type }
      }
    }
  }
}
"""


def verify_bosutinib_targets(session: requests.Session | None = None, timeout: int = 60) -> pd.DataFrame:
    """Fetch bosutinib's drug-target interactions from DGIdb (live, real evidence).

    Returns a frame [record, drug_approved, gene, interaction_score, actions]. The caller
    should treat ABL1, ABL2, and SRC-family members as DISTINCT targets with distinct
    evidence — do not collapse them. Cross-reference ChEMBL/DrugBank for a fuller picture
    (DGIdb aggregates a subset and may split records by source/identifier).
    """
    own = session is None
    session = session or requests.Session()
    try:
        resp = session.post("https://dgidb.org/api/graphql",
                            json={"query": _DGIDB_DRUG_QUERY, "variables": {"names": ["BOSUTINIB"]}},
                            timeout=timeout)
        resp.raise_for_status()
        nodes = resp.json()["data"]["drugs"]["nodes"]
    finally:
        if own:
            session.close()
    rows = []
    for n in nodes:
        for it in n["interactions"]:
            rows.append({
                "record": n["name"], "drug_approved": n["approved"],
                "gene": it["gene"]["name"],
                "interaction_score": it["interactionScore"],
                "actions": ";".join(sorted({t["type"] for t in it["interactionTypes"] if t.get("type")})),
                "source": "dgidb",
            })
    df = pd.DataFrame(rows).sort_values("interaction_score", ascending=False, na_position="last")
    logger.info("Bosutinib: %d DGIdb interactions across %d record(s).",
                len(df), df["record"].nunique() if len(df) else 0)
    return df


BOSUTINIB_TARGET_EVIDENCE_NOTES = """
Interpreting bosutinib targets (evidence-graded):
- Canonical high-affinity targets: BCR-ABL1 (ABL1), SRC-family kinases. Bosutinib is a dual
  SRC/ABL inhibitor approved for CML.
- ABL2 (ARG): inhibited per ChEMBL/biochemical assays, but may be ABSENT from a given DGIdb
  record — evidence is source-dependent. DO NOT assume ABL1 ~ ABL2 ~ SRC equivalence.
- Extensive off-targets (STE20-family STK24/STK10/STK4, SIK1/2/3, CAMK, TXK, HCK, TNK2, ...):
  reported in kinome profiling; relevant for polypharmacology and toxicity but each needs its
  own evidence level (biochemical IC50 vs cellular vs inferred).
- Always grade per target: (a) DrugBank curated target, (b) ChEMBL pChEMBL/assay, (c) DGIdb
  aggregated, (d) inferred. Report the level in the final table; never flatten to 'kinase'.
"""

# --------------------------------------------------------------------------------------
# 4. Validation plan
# --------------------------------------------------------------------------------------
VALIDATION_PLAN = """
If the network analysis nominates bosutinib (or any kinase inhibitor) for an RMS stratum:

In-silico / orthogonal (compute):
- DepMap: is the nominated target a CRISPR/RNAi DEPENDENCY in RMS cell lines? Is bosutinib (or
  a congener) active in DepMap/PRISM drug screens for RMS lines?
- LINCS L1000: does the drug signature REVERSE the RMS disease signature? (optional module).
- Compare bosutinib's disease-module proximity to OTHER kinase inhibitors (dasatinib, ponatinib,
  imatinib, nilotinib) with degree-matched nulls — do not evaluate bosutinib in isolation.

Wet-lab (proposed; requires collaborators):
- Dose-response in fusion+ (aRMS) and fusion- (eRMS) lines -> IC50, Emax.
- Apoptosis assays (caspase-3/7, Annexin V).
- Target engagement (CETSA / phospho-target immunoblot) to confirm on-target inhibition.
- Phosphoproteomics to map pathway modulation vs predicted mechanism.
- Genetic rescue / target knockdown to test on-target dependence.
- Combination testing with standard-of-care.
- Organoid / PDX or xenograft validation for in-vivo activity.

Negative results are reported AS results. Proximity/dependency signals are hypotheses for these
experiments, not evidence of clinical benefit.
"""


def adaptation_summary() -> dict[str, Any]:
    """Return a machine-readable summary of the RMS/bosutinib adaptation guidance."""
    return {
        "tcga_has_rms": False,
        "tcga_sarc_is_rms": False,
        "recommended_sources": list(RMS_DATA_SOURCES.keys()),
        "subtypes": ["embryonal_RMS_fusion_negative", "alveolar_RMS_fusion_positive"],
        "fusions": ["PAX3::FOXO1", "PAX7::FOXO1"],
        "how_to_run": (
            "Provide an RMS raw-count matrix (genes x samples) + sample metadata with a "
            "'condition' column, write it to data/processed/<COHORT>/filtered_counts.parquet and "
            "differential_expression_full.tsv (or run prepare_expression on a GDC/GEO source), set "
            "labels.opentargets.disease_id to an RMS disease id (e.g. search Open Targets for "
            "'rhabdomyosarcoma'), then run build_string_graph -> train_gnn -> map_drug_targets -> "
            "rank_drugs exactly as for LUAD. Stratify by subtype/fusion (separate COHORT per stratum)."
        ),
    }
