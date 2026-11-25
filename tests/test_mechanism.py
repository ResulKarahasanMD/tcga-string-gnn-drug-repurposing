"""Tests for directionality / mechanism-consistency scoring (Section 26)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from repurpose_gnn.mechanism import (
    DEFAULT_WEIGHTS,
    classify_action,
    compute_mechanism_scores,
    direction_consistency,
)


def test_classify_action():
    assert classify_action("inhibitor") == "inhibitory"
    assert classify_action("antagonist;blocker") == "inhibitory"
    assert classify_action("agonist") == "activating"
    assert classify_action("") == "unknown"
    assert classify_action("binder") == "unknown"


def test_direction_consistency_reversing_and_anti():
    de = pd.DataFrame({"log2fc": [3.0, -3.0, 0.2]}, index=["UP", "DOWN", "FLAT"])
    # inhibiting an UP gene = consistent; activating a DOWN gene = consistent
    good = direction_consistency(["UP", "DOWN"], {"UP": "inhibitor", "DOWN": "agonist"}, de)
    assert good["n_consistent"] == 2 and good["direction_consistency"] == 1.0
    # activating an UP gene = anti-consistent
    bad = direction_consistency(["UP"], {"UP": "agonist"}, de)
    assert bad["n_anti"] == 1 and bad["direction_consistency"] == 0.0
    # |lfc| below threshold -> neutral, not scored
    neu = direction_consistency(["FLAT"], {"FLAT": "inhibitor"}, de)
    assert neu["n_neutral"] == 1 and np.isnan(neu["direction_consistency"])


def test_direction_consistency_unknown_gene():
    de = pd.DataFrame({"log2fc": [3.0]}, index=["UP"])
    res = direction_consistency(["NOTINDE"], {"NOTINDE": "inhibitor"}, de)
    assert res["n_unknown_dir"] == 1


def test_compute_mechanism_scores_monotone_in_proximity():
    df = pd.DataFrame({
        "proximity_zscore": [-3.0, 0.0, 2.0],
        "direction_consistency": [1.0, 0.5, 0.0],
        "mean_target_gnn_score": [0.9, 0.5, 0.1],
        "mean_target_expression": [5.0, 3.0, 1.0],
    })
    s = compute_mechanism_scores(df, DEFAULT_WEIGHTS)
    # the most proximal + most direction-consistent drug scores highest
    assert s.iloc[0] > s.iloc[1] > s.iloc[2]
