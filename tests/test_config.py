"""Tests for the config system: extends/merge, dotted access, overrides, path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from repurpose_gnn.config import (
    Config,
    _apply_overrides,
    _deep_merge,
    load_config,
    package_versions,
    set_seed,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[1]


def test_deep_merge_nested():
    base = {"a": 1, "b": {"x": 1, "y": 2}}
    override = {"b": {"y": 20, "z": 30}, "c": 3}
    merged = _deep_merge(base, override)
    assert merged == {"a": 1, "b": {"x": 1, "y": 20, "z": 30}, "c": 3}
    # original untouched
    assert base["b"] == {"x": 1, "y": 2}


def test_dotted_get_and_default():
    cfg = Config(raw={"string": {"score_threshold": 700}}, config_path=Path("x"))
    assert cfg.get("string.score_threshold") == 700
    assert cfg.get("string.missing", 123) == 123
    with pytest.raises(KeyError):
        cfg.get("string.missing")


def test_apply_overrides_parses_yaml_scalars():
    raw = {"gdc": {"retries": 5}}
    out = _apply_overrides(raw, ["gdc.retries=8", "gdc.max_samples_per_group=4", "model.residual=true"])
    assert out["gdc"]["retries"] == 8
    assert out["gdc"]["max_samples_per_group"] == 4
    assert out["model"]["residual"] is True


def test_load_luad_config_extends_default():
    cfg = load_config(REPO / "configs" / "luad.yaml")
    assert cfg.cohort == "TCGA-LUAD"
    # inherited from default.yaml
    assert cfg.get("string.score_threshold") == 700
    assert cfg.get("gdc.workflow_type") == "STAR - Counts"
    # overridden in luad.yaml
    assert cfg.get("labels.opentargets.disease_id") == "MONDO_0005061"
    assert cfg.seeds == [11, 22, 33, 44, 55]


def test_resolve_path_relative_to_root():
    cfg = load_config(REPO / "configs" / "luad.yaml")
    p = cfg.path("paths.raw_tcga")
    assert p.is_absolute()
    assert p.name == "tcga"


def test_set_seed_is_reproducible():
    import random

    set_seed(123)
    a = [random.random() for _ in range(5)]
    set_seed(123)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_sha256_file(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes(b"hello")
    # sha256("hello")
    assert sha256_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"


def test_package_versions_includes_torch():
    versions = package_versions()
    assert "torch" in versions and versions["torch"] != "not installed"
