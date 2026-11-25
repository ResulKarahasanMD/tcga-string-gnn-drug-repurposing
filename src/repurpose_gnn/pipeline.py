"""End-to-end pipeline orchestrator.

    python -m repurpose_gnn.pipeline --config configs/luad.yaml
    python scripts/run_pipeline.py --config configs/luad.yaml   (thin shim)

Runs stages in order, stopping cleanly at the first stage not yet implemented (later build
phases extend the chain). Each stage is idempotent/resumable where possible.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .config import Config, load_config, set_seed, setup_logging

logger = logging.getLogger(__name__)


def _stage_chain() -> list[tuple[str, Callable[[Config], dict[str, Any]] | None]]:
    """Return (name, fn) pairs; fn is None if that stage's module isn't available yet."""
    chain: list[tuple[str, Callable[[Config], dict[str, Any]] | None]] = []

    from .tcga import run_download
    chain.append(("download_tcga", run_download))

    def _try(modpath: str, attr: str) -> Callable[[Config], dict[str, Any]] | None:
        try:
            mod = __import__(f"repurpose_gnn.{modpath}", fromlist=[attr])
            return getattr(mod, attr)
        except (ImportError, AttributeError):
            return None

    chain.append(("prepare_expression", _try("differential_expression", "run_prepare_and_de")))
    chain.append(("build_string_graph", _try("graph_builder", "run_build_graph")))
    chain.append(("train_gnn", _try("train", "run_train")))
    chain.append(("rank_genes", _try("evaluate", "run_rank_genes")))
    chain.append(("map_drug_targets", _try("drug_sources", "run_map_drug_targets")))
    chain.append(("rank_drugs", _try("network_proximity", "run_rank_drugs")))
    return chain


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="repurpose-pipeline", description="Run the full TCGA->STRING->GNN->drug pipeline."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--device", default=None, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--stop-after", default=None, help="Stop after this stage name.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config, overrides=args.overrides)
    except Exception as exc:  # noqa: BLE001
        print(f"[config error] {exc}", file=sys.stderr)
        return 2
    if args.device is not None:
        cfg.raw.setdefault("compute", {})["device"] = args.device

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_file = cfg.path("paths.logs") / f"pipeline_{cfg.cohort}_{ts}.log"
    setup_logging(level=args.log_level, log_file=log_file)
    set_seed(cfg.seed, deterministic=bool(cfg.get("reproducibility.deterministic", True)))

    logger.info("=== PIPELINE cohort=%s config=%s ===", cfg.cohort, args.config)
    results: dict[str, Any] = {}
    for name, fn in _stage_chain():
        if fn is None:
            logger.warning("Stage %r is not implemented yet — stopping pipeline here.", name)
            break
        logger.info("--- stage: %s ---", name)
        try:
            results[name] = fn(cfg)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline failed at stage %s: %s", name, exc)
            return 1
        if args.stop_after and name == args.stop_after:
            logger.info("Stopping after %s as requested.", name)
            break

    logger.info("Pipeline finished. Completed stages: %s", list(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
