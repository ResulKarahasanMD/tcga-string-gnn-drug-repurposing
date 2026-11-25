"""Shared CLI plumbing for every pipeline stage.

Each stage exposes a ``*_main()`` used both by ``scripts/<stage>.py`` (``python scripts/...``)
and by the console-script entrypoints declared in ``pyproject.toml``. All share
:func:`run_stage_cli`, which handles argument parsing, logging setup, config loading,
device override, error handling, and exit codes.

Exit codes: 0 success · 1 stage error (handled) · 2 bad usage/config · 130 interrupted.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import Config, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)

# Stages implemented so far. Later phases flip these on as their modules land.
_NOT_YET = "This stage is implemented in a later build phase."


def _build_parser(prog: str, description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=description)
    p.add_argument("--config", required=True, help="Path to a YAML config (e.g. configs/luad.yaml).")
    p.add_argument(
        "--set", dest="overrides", action="append", default=[], metavar="KEY.SUB=VALUE",
        help="Override a config value (repeatable), e.g. --set gdc.max_samples_per_group=8",
    )
    p.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"],
                   help="Force compute device (default: auto cuda->mps->cpu).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def run_stage_cli(
    stage_name: str,
    stage_fn: Callable[[Config], dict[str, Any]],
    description: str,
    argv: list[str] | None = None,
) -> int:
    """Parse args, load config, run ``stage_fn(cfg)``, return an exit code."""
    parser = _build_parser(f"repurpose-{stage_name}", description)
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, overrides=args.overrides)
    except (FileNotFoundError, ValueError, TypeError, KeyError) as exc:
        print(f"[config error] {exc}", file=sys.stderr)
        return 2

    if args.device is not None:
        cfg.raw.setdefault("compute", {})["device"] = args.device

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_dir = cfg.path("paths.logs")
    log_file = log_dir / f"{stage_name}_{cfg.cohort}_{ts}.log"
    setup_logging(level=args.log_level, log_file=log_file)

    set_seed(cfg.seed, deterministic=bool(cfg.get("reproducibility.deterministic", True)))
    logger.info("=== stage=%s cohort=%s config=%s ===", stage_name, cfg.cohort, args.config)
    logger.info("Logging to %s", log_file)

    try:
        summary = stage_fn(cfg)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        return 130
    except NotImplementedError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001 - stage boundary: log, don't crash the shell ugly
        logger.exception("Stage %s failed: %s", stage_name, exc)
        return 1

    logger.info("Stage %s complete.", stage_name)
    if isinstance(summary, dict):
        print(json.dumps(summary, indent=2, default=str))
    return 0


# --------------------------------------------------------------------------------------
# Per-stage entrypoints
# --------------------------------------------------------------------------------------
def download_tcga_main(argv: list[str] | None = None) -> int:
    from .tcga import run_download

    return run_stage_cli(
        "download-tcga", run_download,
        "Download TCGA STAR-Counts RNA-seq + build sample metadata from the GDC.", argv,
    )


def _pending(name: str) -> Callable[[Config], dict[str, Any]]:
    def _fn(cfg: Config) -> dict[str, Any]:
        raise NotImplementedError(f"{name}: {_NOT_YET}")

    return _fn


def prepare_expression_main(argv: list[str] | None = None) -> int:
    try:
        from .differential_expression import run_prepare_and_de as fn  # type: ignore
    except ImportError:
        fn = _pending("prepare-expression")
    return run_stage_cli("prepare-expression", fn,
                         "Build count matrix, QC, and run differential expression.", argv)


def build_string_graph_main(argv: list[str] | None = None) -> int:
    try:
        from .graph_builder import run_build_graph as fn  # type: ignore
    except ImportError:
        fn = _pending("build-string-graph")
    return run_stage_cli("build-string-graph", fn,
                         "Build the STRING PPI network and PyTorch Geometric Data object.", argv)


def train_gnn_main(argv: list[str] | None = None) -> int:
    try:
        from .train import run_train as fn  # type: ignore
    except ImportError:
        fn = _pending("train-gnn")
    return run_stage_cli("train-gnn", fn, "Train baselines + GNN models and evaluate.", argv)


def rank_genes_main(argv: list[str] | None = None) -> int:
    try:
        from .evaluate import run_rank_genes as fn  # type: ignore
    except ImportError:
        fn = _pending("rank-genes")
    return run_stage_cli("rank-genes", fn, "Rank genes and build the disease module.", argv)


def map_drug_targets_main(argv: list[str] | None = None) -> int:
    try:
        from .drug_sources import run_map_drug_targets as fn  # type: ignore
    except ImportError:
        fn = _pending("map-drug-targets")
    return run_stage_cli("map-drug-targets", fn, "Load drug targets and map to STRING nodes.", argv)


def rank_drugs_main(argv: list[str] | None = None) -> int:
    try:
        from .network_proximity import run_rank_drugs as fn  # type: ignore
    except ImportError:
        fn = _pending("rank-drugs")
    return run_stage_cli("rank-drugs", fn, "Network-proximity drug ranking with null model.", argv)
