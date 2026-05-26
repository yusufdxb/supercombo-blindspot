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
