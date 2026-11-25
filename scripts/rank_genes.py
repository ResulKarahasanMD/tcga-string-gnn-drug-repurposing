#!/usr/bin/env python
"""Thin CLI shim: Rank genes and build the disease module.
Real logic lives in src/repurpose_gnn/. See: repurpose_gnn.cli.rank_genes_main
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a plain checkout (no install) by making src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repurpose_gnn.cli import rank_genes_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(rank_genes_main())
