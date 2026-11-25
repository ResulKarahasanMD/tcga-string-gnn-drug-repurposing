"""Tests for TCGA barcode parsing, GDC filter building, STAR reader, and dedup logic.

All offline — no network required.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from repurpose_gnn.config import Config
from repurpose_gnn.tcga import (
    build_files_filter,
    build_sample_metadata,
    parse_tcga_barcode,
    read_star_counts,
    report_counts,
)


# ---------------- barcode parsing ----------------
def test_parse_sample_barcode_tumor():
    bc = parse_tcga_barcode("TCGA-05-4384-01A")
    assert bc.project == "TCGA"
    assert bc.tss == "05"
    assert bc.participant_id == "4384"
    assert bc.sample_type_code == "01"
    assert bc.vial == "A"
    assert bc.participant_barcode == "TCGA-05-4384"
    assert bc.is_tumor and not bc.is_normal
    assert bc.condition == "tumor"


def test_parse_aliquot_barcode_normal():
    bc = parse_tcga_barcode("TCGA-50-5931-11A-01D-1752-01")
    assert bc.sample_type_code == "11"
    assert bc.vial == "A"
    assert bc.portion == "01"
    assert bc.analyte == "D"
    assert bc.plate == "1752"
    assert bc.center == "01"
    assert bc.is_normal and not bc.is_tumor
    assert bc.condition == "normal"


def test_parse_barcode_rejects_non_tcga():
    with pytest.raises(ValueError):
        parse_tcga_barcode("FOO-05-4384-01A")


def test_parse_barcode_rejects_too_few_fields():
    with pytest.raises(ValueError):
        parse_tcga_barcode("TCGA-05-4384")


# ---------------- GDC filter ----------------
def _mini_cfg(**gdc_over) -> Config:
    raw = {
        "project": {"cohort": "TCGA-LUAD"},
        "reproducibility": {"seed": 11},
        "gdc": {
            "api_base": "https://api.gdc.cancer.gov",
            "data_category": "Transcriptome Profiling",
            "data_type": "Gene Expression Quantification",
            "workflow_type": "STAR - Counts",
            "experimental_strategy": "RNA-Seq",
            "sample_types": {"tumor": ["Primary Tumor"], "normal": ["Solid Tissue Normal"]},
            "dedup_multiple_per_participant": True,
            "max_samples_per_group": None,
            **gdc_over,
        },
    }
    return Config(raw=raw, config_path=Path("x"))


def test_build_files_filter_shape():
    filt = build_files_filter(_mini_cfg())
    assert filt["op"] == "and"
    fields = {c["content"]["field"] for c in filt["content"]}
    assert "cases.project.project_id" in fields
    assert "analysis.workflow_type" in fields
    assert "cases.samples.sample_type" in fields


# ---------------- STAR reader ----------------
def test_read_star_counts_skips_summary_rows(tmp_path):
    content = (
        "# gene-model: GENCODE v36\n"  # real GDC files start with this comment line
        "gene_id\tgene_name\tgene_type\tunstranded\tstranded_first\tstranded_second\ttpm_unstranded\tfpkm_unstranded\tfpkm_uq_unstranded\n"
        "N_unmapped\t\t\t1000\t500\t500\t0\t0\t0\n"
        "N_multimapping\t\t\t200\t100\t100\t0\t0\t0\n"
        "N_noFeature\t\t\t50\t25\t25\t0\t0\t0\n"
        "N_ambiguous\t\t\t10\t5\t5\t0\t0\t0\n"
        "ENSG00000146648.18\tEGFR\tprotein_coding\t42\t20\t22\t3.1\t2.0\t2.1\n"
        "ENSG00000141510.16\tTP53\tprotein_coding\t7\t3\t4\t0.5\t0.3\t0.3\n"
    )
    f = tmp_path / "sample.star.tsv"
    f.write_text(content)
    df = read_star_counts(f, count_column="unstranded")
    assert list(df.index) == ["ENSG00000146648.18", "ENSG00000141510.16"]
    assert df.loc["ENSG00000146648.18", "gene_name"] == "EGFR"
    assert df.loc["ENSG00000146648.18", "unstranded"] == 42
    assert "N_unmapped" not in df.index


def test_read_star_counts_bad_column(tmp_path):
    f = tmp_path / "s.tsv"
    f.write_text("gene_id\tgene_name\tgene_type\tunstranded\nENSG1\tA\tprotein_coding\t1\n")
    with pytest.raises(ValueError):
        read_star_counts(f, count_column="nonexistent")


# ---------------- dedup + counts ----------------
def _files_df() -> pd.DataFrame:
    # two tumor samples from same participant (should dedup to 1), plus one normal.
    return pd.DataFrame(
        [
            {"file_id": "f1", "file_name": "a", "md5sum": "m1", "file_size": 1,
             "case_barcode": "TCGA-05-4384", "sample_barcode": "TCGA-05-4384-01A",
             "gdc_sample_type": "Primary Tumor", "condition": "tumor"},
            {"file_id": "f2", "file_name": "b", "md5sum": "m2", "file_size": 1,
             "case_barcode": "TCGA-05-4384", "sample_barcode": "TCGA-05-4384-01B",
             "gdc_sample_type": "Primary Tumor", "condition": "tumor"},
            {"file_id": "f3", "file_name": "c", "md5sum": "m3", "file_size": 1,
             "case_barcode": "TCGA-50-5931", "sample_barcode": "TCGA-50-5931-11A",
             "gdc_sample_type": "Solid Tissue Normal", "condition": "normal"},
        ]
    )


def test_dedup_one_per_participant_condition():
    sample_df = build_sample_metadata(_files_df(), _mini_cfg())
    # participant TCGA-05-4384 tumor collapses 2->1; normal kept
    assert len(sample_df) == 2
    counts = report_counts(sample_df)
    assert counts["n_tumor"] == 1
    assert counts["n_normal"] == 1
    assert counts["n_participants"] == 2


def test_max_samples_per_group_cap():
    df = _files_df()
    # add a second normal participant
    df = pd.concat([df, pd.DataFrame([{
        "file_id": "f4", "file_name": "d", "md5sum": "m4", "file_size": 1,
        "case_barcode": "TCGA-44-2665", "sample_barcode": "TCGA-44-2665-11A",
        "gdc_sample_type": "Solid Tissue Normal", "condition": "normal"}])], ignore_index=True)
    sample_df = build_sample_metadata(df, _mini_cfg(max_samples_per_group=1))
    counts = report_counts(sample_df)
    assert counts["n_tumor"] == 1
    assert counts["n_normal"] == 1
