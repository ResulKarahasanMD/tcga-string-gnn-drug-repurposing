"""Offline tests for DE preprocessing/annotation logic (not the DESeq2 fit itself)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from repurpose_gnn.config import Config
from repurpose_gnn.differential_expression import (
    annotate_and_flag,
    compute_sample_qc,
    filter_low_expression,
    log_cpm,
)


def test_filter_low_expression():
    counts = pd.DataFrame(
        {
            "s1": [100, 0, 5],
            "s2": [100, 0, 5],
            "s3": [100, 0, 5],
            "s4": [100, 1, 0],
            "s5": [100, 1, 0],
        },
        index=["HIGH", "LOW", "MED"],
    )
    filt, qc = filter_low_expression(counts, min_count=10, min_samples_frac=0.5)
    assert "HIGH" in filt.index
    assert "LOW" not in filt.index  # never reaches 10
    assert "MED" not in filt.index  # only 3/5 >=... actually 0 reach >=10
    assert qc.loc[qc.gene_id == "HIGH", "kept"].iloc[0]


def test_log_cpm_shape():
    counts = pd.DataFrame({"s1": [10, 20], "s2": [30, 40]}, index=["A", "B"])
    lc = log_cpm(counts)
    assert lc.shape == counts.shape
    assert np.isfinite(lc.to_numpy()).all()


def test_compute_sample_qc_outlier_flag():
    # s3 has a tiny library -> robust outlier
    counts = pd.DataFrame(
        {"s1": [1000, 1000], "s2": [1100, 900], "s3": [1, 1]}, index=["A", "B"]
    )
    sample_df = pd.DataFrame(
        {"sample_barcode": ["s1", "s2", "s3"], "condition": ["tumor", "tumor", "normal"]}
    )
    qc = compute_sample_qc(counts, sample_df)
    assert set(["library_size", "n_detected_genes", "outlier_flag"]).issubset(qc.columns)
    assert bool(qc.loc[qc.sample_barcode == "s3", "outlier_flag"].iloc[0])


def _cfg() -> Config:
    return Config(
        raw={
            "project": {"cohort": "TEST"},
            "differential_expression": {"fdr_threshold": 0.05, "abs_log2fc_threshold": 1.0,
                                        "lfc_shrink": False, "reference_level": "normal"},
            "features": {"clip_neglog10_fdr": 50.0},
        },
        config_path=Path("x"),
    )


def test_annotate_and_flag_handles_fdr_zero_and_flags():
    res = pd.DataFrame(
        {
            "baseMean": [100.0, 50.0, 10.0],
            "log2fc": [3.0, -0.2, 2.0],
            "fdr": [0.0, 0.9, 0.01],  # fdr==0 must NOT produce inf
            "pval": [1e-300, 0.8, 1e-4],
        },
        index=["ENSG1.1", "ENSG2.2", "ENSG3.3"],
    )
    ann = pd.DataFrame(
        {"gene_id": ["ENSG1.1", "ENSG2.2", "ENSG3.3"], "gene_name": ["AAA", "BBB", "CCC"],
         "gene_type": ["protein_coding"] * 3}
    )
    out = annotate_and_flag(res, ann, _cfg())
    assert np.isfinite(out["neglog10_fdr"].to_numpy()).all()
    assert out.loc["ENSG1.1", "neglog10_fdr"] == 50.0  # clipped, not inf
    # ENSG1: fdr<0.05 and |lfc|>=1 -> significant, up
    assert bool(out.loc["ENSG1.1", "significant"]) and out.loc["ENSG1.1", "direction"] == "up"
    # ENSG2: not significant
    assert not bool(out.loc["ENSG2.2", "significant"])
    assert out.loc["ENSG1.1", "ensembl"] == "ENSG1"
    assert out.loc["ENSG1.1", "gene_name"] == "AAA"
