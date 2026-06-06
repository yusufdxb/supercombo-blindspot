"""Tests for the LOCO FPR@95%TPR fair operating-point metric."""

from __future__ import annotations

import numpy as np
import pytest

from src.loco_threshold_free import (
    _threshold_at_tpr,
    loco_fpr_at_tpr,
)


def test_threshold_at_tpr_recovers_quantile():
    # higher = more OOD; threshold at 95% TPR is the 5th percentile of collapse.
    collapse = np.linspace(0.0, 1.0, 1001)
    thr = _threshold_at_tpr(collapse, tpr=0.95)
    assert thr == pytest.approx(0.05, abs=1e-2)
    assert float(np.mean(collapse > thr)) == pytest.approx(0.95, abs=1e-2)


def test_e6_zero_fpr_when_collapse_is_frozen():
    # Clean corpora: high-variance real hidden state. Collapse: a frozen vector
    # (near-zero rolling spread). E6 must flag the collapse at 95% TPR with 0%
    # false positives on every held-out real corpus.
    rng = np.random.RandomState(0)
    clean = {f"c{i}": rng.normal(0, 1.0, size=(120, 32)).astype(np.float32)
             for i in range(3)}
    collapse = np.ones((120, 32), dtype=np.float32) + rng.normal(0, 1e-4, (120, 32))
    res = loco_fpr_at_tpr("e6", clean, collapse, window=30, tpr=0.95)
    assert res["fpr_mean"] == pytest.approx(0.0, abs=1e-9)
    assert res["fpr_max"] == pytest.approx(0.0, abs=1e-9)
    # sensitivity met on the collapse set (within finite-sample / strict-> slack)
    for fold in res["folds"].values():
        assert fold["realised_tpr"] == pytest.approx(0.95, abs=0.02)


def test_e6_nonzero_fpr_when_real_overlaps_collapse():
    # If a "clean" corpus also has frozen (low-spread) stretches, the fixed
    # operating point must catch some of them: FPR > 0. This guards against the
    # metric trivially returning 0.
    rng = np.random.RandomState(1)
    clean = {
        "lively": rng.normal(0, 1.0, size=(120, 16)).astype(np.float32),
        "sometimes_frozen": np.vstack([
            rng.normal(0, 1.0, size=(60, 16)),
            np.ones((60, 16)) + rng.normal(0, 1e-4, (60, 16)),
        ]).astype(np.float32),
    }
    collapse = np.ones((120, 16), dtype=np.float32) + rng.normal(0, 1e-4, (120, 16))
    res = loco_fpr_at_tpr("e6", clean, collapse, window=30, tpr=0.95)
    assert res["fpr_max"] > 0.0
