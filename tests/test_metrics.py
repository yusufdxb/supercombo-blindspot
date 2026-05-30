"""Unit tests for src/metrics.py."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.metrics import (
    aupr,
    auroc,
    bootstrap_ci,
    fpr_at_tpr,
    pr_curve_points,
    roc_curve_points,
)


def _two_class(n_id: int = 200, n_ood: int = 200, sep: float = 2.0,
               seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    s_id = rng.normal(0.0, 1.0, size=n_id)
    s_ood = rng.normal(sep, 1.0, size=n_ood)
    scores = np.concatenate([s_id, s_ood])
    labels = np.concatenate([np.zeros(n_id, dtype=int),
                              np.ones(n_ood, dtype=int)])
    return scores, labels


def test_auroc_perfect_separation():
    # Perfect separation: all OOD > all ID. AUROC = 1.0.
    scores = np.concatenate([np.zeros(50), np.ones(50)])
    labels = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    assert auroc(scores, labels) == pytest.approx(1.0)


def test_auroc_random_is_half():
    # Identical distributions: AUROC ~ 0.5.
    rng = np.random.RandomState(7)
    scores = rng.normal(size=1000)
    labels = rng.randint(0, 2, size=1000)
    v = auroc(scores, labels)
    assert 0.45 < v < 0.55


def test_auroc_known_value():
    # Known AUROC against sklearn ground truth on a fixed synthetic set.
    from sklearn.metrics import roc_auc_score
    scores, labels = _two_class(sep=1.5, seed=42)
    assert auroc(scores, labels) == pytest.approx(
        float(roc_auc_score(labels, scores)))


def test_aupr_perfect_separation():
    scores = np.concatenate([np.zeros(50), np.ones(50)])
    labels = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    assert aupr(scores, labels) == pytest.approx(1.0)


def test_fpr_at_tpr_perfect():
    # Perfect separation -> FPR@95TPR = 0.
    scores = np.concatenate([np.zeros(50), np.ones(50)])
    labels = np.concatenate([np.zeros(50, dtype=int), np.ones(50, dtype=int)])
    assert fpr_at_tpr(scores, labels, 0.95) == pytest.approx(0.0)


def test_fpr_at_tpr_monotone_in_target():
    # Easier target (lower TPR) cannot need a higher FPR.
    scores, labels = _two_class(sep=1.0, seed=3)
    f95 = fpr_at_tpr(scores, labels, 0.95)
    f50 = fpr_at_tpr(scores, labels, 0.50)
    assert f50 <= f95 + 1e-9


def test_roc_curve_points_shapes():
    scores, labels = _two_class(seed=1)
    fpr, tpr, thr = roc_curve_points(scores, labels)
    assert fpr.shape == tpr.shape
    assert fpr[0] == 0.0 and tpr[0] == 0.0
    assert fpr[-1] == pytest.approx(1.0) and tpr[-1] == pytest.approx(1.0)


def test_pr_curve_points_shapes():
    scores, labels = _two_class(seed=1)
    p, r, thr = pr_curve_points(scores, labels)
    assert p.shape == r.shape
    # sklearn convention: recall ends at 0 (last point) and starts at 1.
    assert r[0] == pytest.approx(1.0)


def test_bootstrap_reproducibility():
    scores, labels = _two_class(seed=5)
    a = bootstrap_ci(auroc, scores, labels, n_bootstrap=200, seed=42)
    b = bootstrap_ci(auroc, scores, labels, n_bootstrap=200, seed=42)
    assert a == b


def test_bootstrap_ci_tightens_with_more_samples():
    # The CI WIDTH around the bootstrap mean should not grow as we add more
    # replicates; on a fixed dataset it should be approximately stable, and
    # in practice slightly narrower because the quantile estimate is less
    # noisy. We assert non-increase up to a tolerance.
    scores, labels = _two_class(n_id=300, n_ood=300, sep=1.0, seed=11)
    _, lo_a, hi_a = bootstrap_ci(auroc, scores, labels, n_bootstrap=100,
                                 seed=42)
    _, lo_b, hi_b = bootstrap_ci(auroc, scores, labels, n_bootstrap=1000,
                                 seed=42)
    width_a = hi_a - lo_a
    width_b = hi_b - lo_b
    # Allow a small slack: the 1000-replicate CI may be marginally wider
    # than the 100-replicate one if the 100-replicate one happened to miss
    # the tails, but should never be wider by more than 20%.
    assert width_b <= width_a * 1.20 + 1e-6


def test_bootstrap_mean_near_point_estimate():
    scores, labels = _two_class(n_id=400, n_ood=400, sep=2.0, seed=9)
    point = auroc(scores, labels)
    mean, lo, hi = bootstrap_ci(auroc, scores, labels, n_bootstrap=500,
                                seed=42)
    # The bootstrap mean should be within ~2% of the point estimate.
    assert abs(mean - point) < 0.02
    assert lo <= point <= hi or (lo - 1e-3) <= point <= (hi + 1e-3)


def test_bootstrap_stratified_preserves_class_counts():
    # Heavily imbalanced data: with non-stratified resampling, replicates
    # would frequently land with zero OOD examples and AUROC would be NaN.
    # Our stratified bootstrap should never produce NaN here.
    rng = np.random.RandomState(0)
    scores = np.concatenate([rng.normal(0, 1, 990), rng.normal(3, 1, 10)])
    labels = np.concatenate([np.zeros(990, dtype=int),
                              np.ones(10, dtype=int)])
    mean, lo, hi = bootstrap_ci(auroc, scores, labels, n_bootstrap=200,
                                seed=42)
    assert np.isfinite(mean) and np.isfinite(lo) and np.isfinite(hi)


def test_metrics_handle_nan_scores():
    scores = np.array([0.1, np.nan, 0.5, 0.9, 0.2])
    labels = np.array([0, 1, 1, 1, 0])
    # auroc should drop the NaN and still compute on the remaining 4.
    v = auroc(scores, labels)
    assert np.isfinite(v)
