"""Tests for E6: hidden_state-spread detector."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.e6_detector import rolling_spread


def test_rolling_spread_constant_is_zero():
    h = np.ones((50, 8), dtype=np.float32)
    s = rolling_spread(h, window=10)
    assert s[-1] == pytest.approx(0.0, abs=1e-7)


def test_rolling_spread_is_window_var_trace():
    rng = np.random.RandomState(0)
    h = rng.randn(100, 8).astype(np.float32)
    s = rolling_spread(h, window=20)
    direct = float(np.var(h[-20:], axis=0).sum())
    assert s[-1] == pytest.approx(direct, rel=1e-5)


from src.e6_detector import calibrate_threshold


def test_calibrate_threshold_is_low_percentile():
    rng = np.random.RandomState(0)
    spreads = rng.uniform(1.0, 2.0, size=500)
    thr = calibrate_threshold(spreads, percentile=1.0)
    assert 1.0 <= thr <= 1.05


def test_e6_runs_on_caches_and_fires_below_alpha_1():
    teardown = Path("report/teardown_collected.npz")
    e4 = Path("report/e4_collected.npz")
    if not teardown.exists() or not e4.exists():
        pytest.skip("caches missing")
    from src.e6_detector import (calibrate_threshold, evaluate_on_e4,
                                  rolling_spread, _real_calibration_hidden)
    real_h = _real_calibration_hidden()
    thr = calibrate_threshold(rolling_spread(real_h, 30), 1.0)
    res = evaluate_on_e4(e4, thr, 30)
    assert res["fires_at"] < 1.0


def test_loco_fpr_two_corpora():
    rng = np.random.RandomState(0)
    sub = rng.normal(1.0, 0.1, size=(200, 16)).astype(np.float32)
    ram = rng.normal(1.1, 0.1, size=(200, 16)).astype(np.float32)
    from src.e6_detector import loco_fpr
    res = loco_fpr({"subaru": sub, "ram": ram}, window=20, percentile=1.0)
    assert set(res["folds"].keys()) == {"subaru", "ram"}
    assert 0.0 <= res["fpr_mean"] <= 1.0
    assert 0.0 <= res["fpr_max"] <= 1.0
    assert res["fpr_max"] >= res["fpr_mean"]
