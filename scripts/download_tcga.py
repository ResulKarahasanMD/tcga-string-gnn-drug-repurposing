#!/usr/bin/env python
"""Thin CLI shim: Download TCGA STAR-Counts RNA-seq + sample metadata from the GDC.
Real logic lives in src/repurpose_gnn/. See: repurpose_gnn.cli.download_tcga_main
"""
from __future__ import annotations

import sys
from pathlib import Path

# Allow running from a plain checkout (no install) by making src/ importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repurpose_gnn.cli import download_tcga_main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(download_tcga_main())
