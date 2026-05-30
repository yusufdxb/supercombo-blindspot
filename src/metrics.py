"""Threshold-free OOD detection metrics and bootstrap confidence intervals.

Conventions:
- Higher score = more OOD.
- Label 1 = OOD, label 0 = ID.

This module is a thin numpy + sklearn wrapper around the standard suite
(AUROC, AUPR, FPR@95TPR) plus a stratified bootstrap CI helper. It is
deliberately decoupled from the E6 / baseline scoring code so the same
metrics can be applied to any per-frame OOD score arrays (e.g. the cached
arrays in report/baselines_collected.npz, or fresh scores from
src/pca_mahalanobis.py).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


def _clean(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    mask = np.isfinite(s)
    return s[mask], y[mask]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve. Higher score = more OOD; label 1 = OOD."""
    s, y = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def aupr(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the precision-recall curve (average precision)."""
    s, y = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def fpr_at_tpr(scores: np.ndarray, labels: np.ndarray,
               tpr_target: float = 0.95) -> float:
    """FPR at the smallest threshold whose TPR is >= tpr_target.

    Returns NaN if no threshold reaches the target TPR (which should not
    happen for non-degenerate inputs since TPR=1 at the lowest threshold).
    """
    s, y = _clean(scores, labels)
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, s)
    idx = np.searchsorted(tpr, tpr_target, side="left")
    if idx >= len(tpr):
        return float("nan")
    return float(fpr[idx])


def roc_curve_points(scores: np.ndarray, labels: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(fpr, tpr, thresholds) for plotting / further analysis."""
    s, y = _clean(scores, labels)
    fpr, tpr, thr = roc_curve(y, s)
    return fpr, tpr, thr


def pr_curve_points(scores: np.ndarray, labels: np.ndarray
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(precision, recall, thresholds) for plotting / further analysis."""
    s, y = _clean(scores, labels)
    p, r, thr = precision_recall_curve(y, s)
    return p, r, thr


def bootstrap_ci(metric_fn: Callable[[np.ndarray, np.ndarray], float],
                 scores: np.ndarray, labels: np.ndarray,
                 n_bootstrap: int = 1000, ci: float = 0.95,
                 seed: int = 42) -> tuple[float, float, float]:
    """Stratified bootstrap CI for any (scores, labels) -> float metric.

    Resamples ID and OOD frames independently with replacement so the
    per-class count is preserved across resamples (the alternative is
    catastrophic for AUROC when one class is small). Returns (mean, lo, hi)
    where lo/hi are the central-ci quantiles across bootstrap replicates.

    NaN replicates (degenerate label distribution after resample, which is
    impossible for stratified resampling unless one class is empty
    upstream) are dropped before quantile.
    """
    s, y = _clean(scores, labels)
    rng = np.random.RandomState(int(seed))
    idx_pos = np.where(y == 1)[0]
    idx_neg = np.where(y == 0)[0]
    if len(idx_pos) == 0 or len(idx_neg) == 0:
        return float("nan"), float("nan"), float("nan")
    vals = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        rp = rng.choice(idx_pos, size=len(idx_pos), replace=True)
        rn = rng.choice(idx_neg, size=len(idx_neg), replace=True)
        sel = np.concatenate([rp, rn])
        vals[b] = metric_fn(s[sel], y[sel])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan"), float("nan")
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(vals, alpha))
    hi = float(np.quantile(vals, 1.0 - alpha))
    return float(vals.mean()), lo, hi
