#!/usr/bin/env python
"""Thin shim: run the full pipeline. See repurpose_gnn.pipeline.main"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from repurpose_gnn.pipeline import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
