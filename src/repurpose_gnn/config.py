"""Configuration, reproducibility, logging, and run-manifest utilities.

A single YAML file (optionally ``extends``-ing another) drives every stage. The :class:`Config`
object gives dotted-key access (``cfg.get("string.score_threshold")``) plus helpers to resolve
paths relative to the project root.

Reproducibility harness (used by every CLI entrypoint):
- :func:`set_seed`            seed python / numpy / torch / cuda and request determinism
- :func:`get_device`          cuda -> mps -> cpu selection with a safe MPS->CPU fallback
- :func:`sha256_file`         hash inputs for provenance
- :func:`write_run_manifest`  record config, package versions, seed, input hashes, timestamp

Determinism note: exact bitwise reproducibility is NOT guaranteed across devices/BLAS builds.
On MPS some ops are non-deterministic or unimplemented; we set ``PYTORCH_ENABLE_MPS_FALLBACK=1``
so unsupported ops silently fall back to CPU. See docstrings for details.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import random
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Project root = repository root = two levels up from this file (src/repurpose_gnn/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIGS_DIR = PROJECT_ROOT / "configs"

# Sentinel so that get(key) can distinguish "missing" from a legitimate None/default value.
_MISSING = object()


# --------------------------------------------------------------------------------------
# Config object
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    """Thin wrapper over a nested config dict with dotted-key access + path resolution."""

    raw: dict[str, Any]
    config_path: Path
    project_root: Path = field(default=PROJECT_ROOT)

    # -- access -------------------------------------------------------------------------
    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        """Return a nested value by dotted key, e.g. ``"string.score_threshold"``.

        Raises :class:`KeyError` if the key is absent and no ``default`` is supplied.
        """
        node: Any = self.raw
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                if default is _MISSING:
                    raise KeyError(f"Missing required config key: {dotted_key!r}")
                return default
        return node

    def section(self, name: str) -> dict[str, Any]:
        """Return a top-level section dict (empty dict if absent)."""
        val = self.raw.get(name, {})
        if not isinstance(val, dict):
            raise TypeError(f"Config section {name!r} is not a mapping (got {type(val).__name__}).")
        return val

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    # -- paths --------------------------------------------------------------------------
    def resolve_path(self, path_like: str | os.PathLike, *, create_parent: bool = False) -> Path:
        """Resolve a path relative to the project root (absolute paths pass through)."""
        p = Path(path_like)
        if not p.is_absolute():
            p = self.project_root / p
        p = p.resolve()
        if create_parent:
            p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def path(self, dotted_key: str, *, create_parent: bool = False) -> Path:
        """Resolve a path stored under ``dotted_key`` in the config."""
        return self.resolve_path(self.get(dotted_key), create_parent=create_parent)

    @property
    def cohort(self) -> str:
        return str(self.get("project.cohort"))

    @property
    def seed(self) -> int:
        return int(self.get("reproducibility.seed", 11))

    @property
    def seeds(self) -> list[int]:
        return [int(s) for s in self.get("reproducibility.seeds", [11, 22, 33, 44, 55])]


# --------------------------------------------------------------------------------------
# Loading / merging
# --------------------------------------------------------------------------------------
def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (override wins). Returns a new dict."""
    out = dict(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"Top level of {path} must be a mapping, got {type(data).__name__}.")
    return data


def _apply_overrides(raw: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply ``key.subkey=value`` CLI overrides (values parsed as YAML scalars)."""
    if not overrides:
        return raw
    out = raw
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got: {item!r}")
        dotted, _, val_str = item.partition("=")
        value = yaml.safe_load(val_str)  # parses ints/floats/bools/lists/None sensibly
        node = out
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"Cannot set {dotted}: {part} is not a mapping.")
        node[parts[-1]] = value
    return out


def load_config(
    config_path: str | os.PathLike,
    *,
    overrides: list[str] | None = None,
) -> Config:
    """Load a YAML config, resolving an optional ``extends: <file>`` base, then CLI overrides.

    ``extends`` is resolved relative to the config file's directory. Cohort configs
    (``configs/luad.yaml``) typically ``extends: default.yaml``.
    """
    config_path = Path(config_path).resolve()
    raw = _read_yaml(config_path)

    parent = raw.pop("extends", None)
    if parent is not None:
        base_path = (config_path.parent / parent).resolve()
        base = load_config(base_path).raw
        raw = _deep_merge(base, raw)

    raw = _apply_overrides(raw, overrides)
    cfg = Config(raw=raw, config_path=config_path)
    return cfg


# --------------------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------------------
def set_seed(seed: int, *, deterministic: bool = True) -> None:
    """Seed python / numpy / torch / cuda RNGs.

    ``deterministic=True`` also requests deterministic cuDNN/torch algorithms where
    available. This does NOT guarantee identical results across devices, PyTorch
    versions, or BLAS builds; on MPS several ops are non-deterministic. Treat multi-seed
    mean +/- sd (see ``reproducibility.seeds``) as the reproducibility unit, not a single run.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # numpy always present in practice; keep import-light for tests
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except AttributeError:
                pass
            # Best-effort; some ops raise if no deterministic impl exists — do not hard-fail.
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except (AttributeError, RuntimeError):
                pass
    except ImportError:
        logger.debug("torch not importable during set_seed; skipping torch seeding.")


def get_device(prefer: str | None = None) -> "object":
    """Select a compute device: cuda -> mps -> cpu (unless ``prefer`` forces one).

    Also sets ``PYTORCH_ENABLE_MPS_FALLBACK=1`` so ops unimplemented on MPS fall back to
    CPU instead of raising. Returns a ``torch.device``.
    """
    import torch

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

    if prefer and prefer.lower() != "auto":
        prefer = prefer.lower()
        if prefer == "cpu":
            return torch.device("cpu")
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        logger.warning("Requested device %r unavailable; falling back to auto-selection.", prefer)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------------------
# Provenance: hashing, package versions, run manifest
# --------------------------------------------------------------------------------------
def sha256_file(path: str | os.PathLike, *, chunk_size: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (memory-safe for large downloads)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


_TRACKED_PACKAGES = (
    "numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "pydeseq2", "anndata",
    "networkx", "python-igraph", "leidenalg", "torch", "torch-geometric", "gseapy",
    "matplotlib", "plotly", "requests", "lxml", "pyarrow", "pyyaml",
)


def package_versions() -> dict[str, str]:
    """Best-effort {package: version} for tracked dependencies (missing -> 'not installed')."""
    out: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "not installed"
    return out


def write_run_manifest(
    cfg: Config,
    *,
    stage: str,
    seed: int,
    input_files: dict[str, str | os.PathLike] | None = None,
    extra: dict[str, Any] | None = None,
    out_dir: str | os.PathLike | None = None,
) -> Path:
    """Write a JSON manifest recording config, versions, seed, input hashes, and timestamp.

    ``input_files`` maps a logical name -> path; each existing file is hashed. Returns the
    manifest path (``results/logs/manifest_<stage>_<UTC-timestamp>.json`` by default).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(out_dir) if out_dir else (cfg.project_root / "results" / "logs")
    out_dir.mkdir(parents=True, exist_ok=True)

    hashes: dict[str, Any] = {}
    for name, p in (input_files or {}).items():
        p = Path(p)
        if p.exists() and p.is_file():
            hashes[name] = {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        else:
            hashes[name] = {"path": str(p), "sha256": None, "bytes": None, "note": "missing"}

    manifest = {
        "stage": stage,
        "cohort": cfg.get("project.cohort", None),
        "timestamp_utc": ts,
        "seed": seed,
        "config_path": str(cfg.config_path),
        "config": cfg.raw,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions(),
        "input_files": hashes,
        "extra": extra or {},
    }
    out_path = out_dir / f"manifest_{stage}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)
    logger.info("Wrote run manifest: %s", out_path)
    return out_path


# --------------------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------------------
def setup_logging(
    *,
    level: str = "INFO",
    log_file: str | os.PathLike | None = None,
) -> None:
    """Configure root logging to stderr (and optionally a file). Idempotent-ish."""
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )
