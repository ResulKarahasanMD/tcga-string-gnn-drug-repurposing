"""repurpose_gnn — TCGA -> STRING -> GNN -> network-proximity drug repurposing.

Modular research pipeline. Core analysis logic lives in this package; notebooks under
``notebooks/`` and thin scripts under ``scripts/`` are orchestration/teaching layers only.

Design principles enforced across modules:
- No fabricated results, endpoints, or credentials.
- Differential-expression results are node *features*, never disease-gene *labels*.
- No train/test leakage; feature scaling is fit on the training fold only.
- Network proximity is hypothesis-generating, not evidence of clinical efficacy.
- When required data are missing, stop the stage and report — never invent.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
