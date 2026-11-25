"""Directionality & mechanism-consistency scoring (Section 26) + candidate table (Section 28).

Proximity to the disease module does NOT tell us whether a drug would *reverse* the disease.
This module adds mechanism reasoning:

- **Direction consistency**: does the drug's action oppose the target's dysregulation?
  An *inhibitor* of a gene UP in tumor, or an *activator* of a gene DOWN in tumor, is
  "disease-reversing"-consistent. Agonist of an up-gene is anti-consistent.
- **Mechanism-consistency score**: a transparent weighted sum of proximity, direction,
  GNN gene score, and target expression, MINUS toxicity / essential-normal-tissue penalties.
  Penalty terms need external data (tox/DepMap); when unavailable they are 0 and flagged,
  never fabricated.

Default weights are labelled **hypothesis-generation weights** and are swept in the
sensitivity analysis — they are not claimed to be optimal.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Drug-action vocabulary (DGIdb / DrugBank action strings), lowercased.
_INHIBITORY = {"inhibitor", "antagonist", "blocker", "suppressor", "negative modulator",
               "inverse agonist", "antisense", "inhibitory allosteric modulator", "degrader"}
_ACTIVATING = {"agonist", "activator", "inducer", "positive modulator", "stimulator",
               "partial agonist", "positive allosteric modulator"}

DEFAULT_WEIGHTS = {
    "proximity": 0.35,          # -z-score (closer than random)
    "direction": 0.25,          # fraction of direction-consistent targets
    "gnn": 0.20,                # mean GNN disease-gene score of targets
    "expression": 0.20,         # targets expressed in tumor
    "toxicity_penalty": 0.0,    # requires external tox data (not modeled here)
    "essential_penalty": 0.0,   # requires DepMap essentiality (not modeled here)
}


def classify_action(action_str: str) -> str:
    """Map an action string to 'inhibitory' | 'activating' | 'unknown'."""
    if not action_str:
        return "unknown"
    parts = {a.strip().lower() for a in str(action_str).split(";") if a.strip()}
    if parts & _INHIBITORY:
        return "inhibitory"
    if parts & _ACTIVATING:
        return "activating"
    return "unknown"


def direction_consistency(
    targets: list[str],
    actions: dict[str, str],
    de_by_symbol: pd.DataFrame,
    *,
    lfc_threshold: float = 1.0,
) -> dict[str, Any]:
    """Fraction of a drug's targets whose action opposes their tumor dysregulation.

    ``de_by_symbol`` indexed by gene symbol with a ``log2fc`` column. Returns the consistency
    fraction plus counts (consistent / anti-consistent / neutral / unknown).
    """
    consistent = anti = neutral = unknown = 0
    for g in targets:
        act = classify_action(actions.get(g, ""))
        if g not in de_by_symbol.index:
            unknown += 1
            continue
        lfc = float(de_by_symbol.loc[g, "log2fc"])
        if abs(lfc) < lfc_threshold or act == "unknown":
            neutral += 1
            continue
        up = lfc > 0
        # reversing: inhibit an up gene, or activate a down gene
        if (up and act == "inhibitory") or (not up and act == "activating"):
            consistent += 1
        else:
            anti += 1
    scored = consistent + anti
    frac = consistent / scored if scored else np.nan
    return {"direction_consistency": frac, "n_consistent": consistent, "n_anti": anti,
            "n_neutral": neutral, "n_unknown_dir": unknown}


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = np.isfinite(x)
    if finite.sum() == 0:
        return np.zeros_like(x)
    lo, hi = np.nanmin(x[finite]), np.nanmax(x[finite])
    if hi - lo < 1e-12:
        return np.where(finite, 0.5, 0.0)
    out = (x - lo) / (hi - lo)
    return np.where(finite, out, 0.0)


def compute_mechanism_scores(ranking: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Transparent weighted mechanism score from normalized components (higher = better)."""
    proximity = _minmax(-ranking["proximity_zscore"].to_numpy())       # closer -> higher
    direction = ranking["direction_consistency"].fillna(0.0).to_numpy()  # already 0..1
    gnn = _minmax(ranking["mean_target_gnn_score"].to_numpy())
    expr = _minmax(ranking.get("mean_target_expression", pd.Series(np.zeros(len(ranking)))).to_numpy())
    score = (
        weights["proximity"] * proximity
        + weights["direction"] * direction
        + weights["gnn"] * gnn
        + weights["expression"] * expr
        - weights["toxicity_penalty"] * 0.0
        - weights["essential_penalty"] * 0.0
    )
    return pd.Series(score, index=ranking.index)


def mechanistic_summary(row: pd.Series, target_symbols: list[str]) -> str:
    """Auto-generate the hypothesis-generating, non-clinical summary (Section 28)."""
    name = row.get("drug_name") or row.get("drug_id")
    tset = ", ".join(target_symbols[:3]) + ("…" if len(target_symbols) > 3 else "")
    z = row.get("proximity_zscore", float("nan"))
    dc = row.get("direction_consistency", float("nan"))
    dirtxt = (f"{dc:.0%} of scored targets have a disease-reversing action direction"
              if np.isfinite(dc) else "target action directions are largely unannotated")
    return (
        f"{name} sits close to the LUAD disease module (proximity z={z:.2f}) through targets "
        f"{tset}; {dirtxt}. This is hypothesis-generating and does NOT demonstrate binding, "
        f"adequate exposure, cellular activity, or clinical efficacy."
    )
