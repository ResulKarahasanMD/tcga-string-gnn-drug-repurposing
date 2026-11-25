"""Evaluation metrics, bootstrap CIs, model comparison, and gene ranking (Sections 19-21).

Classes are highly imbalanced (few known disease genes), so accuracy is NOT primary. We report
AUROC, AUPRC (primary), precision/recall/F1, MCC, balanced accuracy, Brier, and ranking metrics
(precision@k, recall@k, enrichment@k) with bootstrap confidence intervals.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def _safe_auroc(y: np.ndarray, p: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    k = min(k, len(scores))
    if k == 0:
        return float("nan")
    top = np.argsort(scores)[::-1][:k]
    return float(y[top].sum() / k)


def recall_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    top = np.argsort(scores)[::-1][: min(k, len(scores))]
    return float(y[top].sum() / n_pos)


def enrichment_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    """Fold-enrichment of positives in top-k vs the base rate."""
    base = y.mean()
    if base == 0:
        return float("nan")
    return precision_at_k(y, scores, k) / base


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    threshold: float = 0.5,
    ks: tuple[int, ...] = (10, 20, 50, 100),
) -> dict[str, float]:
    """Full imbalance-aware metric bundle at a given probability threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n_pos = int(y_true.sum())

    # AUROC/AUPRC/@k are rank-based -> valid for ANY monotone score (probabilities or raw
    # centrality). Brier/calibration require true probabilities; threshold classification
    # needs a cut-point. For non-probability scores (e.g. degree/PageRank baselines) we use a
    # rank cut (top-n_pos) for the classification metrics and leave Brier undefined.
    is_prob = bool(np.all(np.isfinite(y_prob)) and y_prob.min() >= 0.0 and y_prob.max() <= 1.0)
    if is_prob:
        y_pred = (y_prob >= threshold).astype(int)
    else:
        y_pred = np.zeros_like(y_true)
        if n_pos > 0:
            y_pred[np.argsort(y_prob)[::-1][:n_pos]] = 1

    out: dict[str, float] = {
        "auroc": _safe_auroc(y_true, y_prob),
        "auprc": float(average_precision_score(y_true, y_prob)) if n_pos else float("nan"),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "brier": float(brier_score_loss(y_true, y_prob)) if (is_prob and n_pos) else float("nan"),
        "is_probability": is_prob,
        "n_pos": n_pos,
        "n": int(len(y_true)),
    }
    for k in ks:
        out[f"precision@{k}"] = precision_at_k(y_true, y_prob, k)
        out[f"recall@{k}"] = recall_at_k(y_true, y_prob, k)
        out[f"enrichment@{k}"] = enrichment_at_k(y_true, y_prob, k)
    return out


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    seed: int = 11,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a metric. Returns (point, lo, hi)."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    point = metric_fn(y_true, y_prob)
    n = len(y_true)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        stats.append(metric_fn(y_true[idx], y_prob[idx]))
    if not stats:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def delong_auroc_compare(y_true: np.ndarray, prob_a: np.ndarray, prob_b: np.ndarray) -> dict[str, float]:
    """Bootstrap comparison of two models' AUROC (paired). Returns delta + empirical p.

    (A robust, assumption-light alternative to the analytic DeLong test; paired over samples.)
    """
    rng = np.random.default_rng(11)
    y_true = np.asarray(y_true)
    a, b = np.asarray(prob_a), np.asarray(prob_b)
    obs = _safe_auroc(y_true, a) - _safe_auroc(y_true, b)
    n = len(y_true)
    deltas = []
    for _ in range(2000):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        deltas.append(_safe_auroc(y_true[idx], a[idx]) - _safe_auroc(y_true[idx], b[idx]))
    deltas = np.array(deltas)
    # two-sided empirical p that delta crosses 0
    p = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean()) if len(deltas) else float("nan")
    return {"auroc_delta": float(obs), "p_value": float(p), "n_boot": int(len(deltas))}


def run_rank_genes(cfg):  # noqa: ANN001 - CLI stage entrypoint
    """CLI stage: score all nodes with the trained model and build the ranked gene table.

    Implemented in the training/ranking phase; requires a trained model checkpoint.
    """
    from .train import rank_all_genes

    return rank_all_genes(cfg)
