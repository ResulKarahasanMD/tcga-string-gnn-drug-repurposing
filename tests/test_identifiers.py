"""Tests for identifier mapping (Section 8, 32): version stripping, symbol collapse,
STRING mapping, species validation, one-to-many resolution. All offline."""

from __future__ import annotations

import pandas as pd
import pytest

from repurpose_gnn.identifiers import (
    build_gene_annotation,
    collapse_duplicate_symbols,
    filter_protein_coding,
    is_par_y,
    map_symbols_to_string,
    strip_ensembl_version,
    validate_species,
)


# ---------------- version stripping ----------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ENSG00000146648.18", "ENSG00000146648"),
        ("ENSG00000146648", "ENSG00000146648"),
        ("ENSG00000182378.14_PAR_Y", "ENSG00000182378"),
        ("SOMEGENE.3", "SOMEGENE"),
        ("PLAINSYMBOL", "PLAINSYMBOL"),
    ],
)
def test_strip_ensembl_version(raw, expected):
    assert strip_ensembl_version(raw) == expected


def test_is_par_y():
    assert is_par_y("ENSG00000182378.14_PAR_Y")
    assert not is_par_y("ENSG00000146648.18")


def test_strip_type_error():
    with pytest.raises(TypeError):
        strip_ensembl_version(12345)


# ---------------- annotation ----------------
def test_build_gene_annotation_and_filter():
    star = pd.DataFrame(
        {"gene_name": ["EGFR", "MALAT1"], "gene_type": ["protein_coding", "lncRNA"],
         "unstranded": [10, 20]},
        index=pd.Index(["ENSG00000146648.18", "ENSG00000251562.8"], name="gene_id"),
    )
    ann = build_gene_annotation(star)
    assert set(ann.columns) == {"gene_id", "ensembl", "gene_name", "gene_type", "is_par_y"}
    assert ann.loc[ann.gene_name == "EGFR", "ensembl"].iloc[0] == "ENSG00000146648"
    pc = filter_protein_coding(ann)
    assert list(pc["gene_name"]) == ["EGFR"]


# ---------------- duplicate-symbol collapse ----------------
def test_collapse_duplicate_symbols_sum():
    counts = pd.DataFrame(
        {"s1": [5, 7, 3], "s2": [1, 1, 1]},
        index=["ENSG_A1", "ENSG_A2", "ENSG_B"],
    )
    ann = pd.DataFrame(
        {"gene_id": ["ENSG_A1", "ENSG_A2", "ENSG_B"], "gene_name": ["A", "A", "B"]}
    )
    collapsed, qc = collapse_duplicate_symbols(counts, ann, strategy="sum")
    assert collapsed.loc["A", "s1"] == 12  # 5+7
    assert collapsed.loc["B", "s1"] == 3
    assert qc.loc[qc.gene_name == "A", "n_ensembl_ids"].iloc[0] == 2


def test_collapse_duplicate_symbols_max_total():
    counts = pd.DataFrame({"s1": [5, 100], "s2": [5, 100]}, index=["ENSG_A1", "ENSG_A2"])
    ann = pd.DataFrame({"gene_id": ["ENSG_A1", "ENSG_A2"], "gene_name": ["A", "A"]})
    collapsed, _ = collapse_duplicate_symbols(counts, ann, strategy="max_total")
    assert collapsed.loc["A", "s1"] == 100  # keeps the higher-total Ensembl id


# ---------------- STRING mapping ----------------
def _aliases() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "string_protein_id": [
                "9606.ENSP00000275493", "9606.ENSP00000275493",
                "9606.ENSP00000269305", "9606.ENSP00000000001",
            ],
            "alias": ["EGFR", "EGFR", "TP53", "egfr"],
            "source": ["Ensembl_HGNC", "BioMart_HUGO", "Ensembl_HGNC", "OtherSource"],
        }
    )


def test_map_symbols_to_string_basic():
    res = map_symbols_to_string(["EGFR", "TP53"], _aliases())
    assert res.n_mapped == 2
    egfr = res.mapped[res.mapped.gene_name == "EGFR"].iloc[0]
    assert egfr["string_protein_id"] == "9606.ENSP00000275493"


def test_map_symbols_to_string_unmatched_recorded():
    res = map_symbols_to_string(["EGFR", "NOTAGENE"], _aliases())
    assert res.n_mapped == 1
    assert "NOTAGENE" in set(res.unmatched["gene_name"])
    assert res.mapping_rate() == 0.5


def test_map_symbols_case_insensitive_fallback():
    # 'Egfr' has no exact alias but matches 'egfr'/'EGFR' via lowercase fallback
    res = map_symbols_to_string(["Egfr"], _aliases())
    assert res.n_mapped == 1


def test_validate_species_rejects_wrong_taxon():
    bad = pd.Series(["10090.ENSMUSP00000000001"])
    with pytest.raises(ValueError):
        validate_species(bad, taxon=9606)
