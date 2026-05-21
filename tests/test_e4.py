"""Tests for E4: real-to-sim interpolation sweep."""

from __future__ import annotations

import numpy as np
import pytest

from src.e4_interp import blend


def test_blend_endpoints_and_midpoint():
    r = np.full((6, 4, 4), 10.0, dtype=np.float32)
    c = np.full((6, 4, 4), 50.0, dtype=np.float32)
    assert np.allclose(blend(r, c, 0.0), 10.0)
    assert np.allclose(blend(r, c, 1.0), 50.0)
    assert np.allclose(blend(r, c, 0.5), 30.0)


def test_blend_is_float32_and_in_range():
    r = np.zeros((6, 2, 2), dtype=np.uint8)
    c = np.full((6, 2, 2), 255, dtype=np.uint8)
    out = blend(r, c, 0.25)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 255.0


from src.e4_interp import transition_width


def test_transition_width_sharp_cliff():
    # activity holds near 1.0 then drops inside one step: a cliff
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    norm = {a: (1.0 if a <= 0.5 else 0.02) for a in alphas}
    a90, a10 = transition_width(alphas, norm)
    assert a10 - a90 < 0.2, (a90, a10)


def test_transition_width_smooth_gradient():
    # activity falls linearly across the whole sweep: a gradient
    alphas = [round(0.1 * i, 4) for i in range(11)]
    norm = {a: 1.0 - a for a in alphas}
    a90, a10 = transition_width(alphas, norm)
    assert a10 - a90 > 0.6, (a90, a10)


from src.e4_interp import load_cache, save_cache


def test_cache_round_trip(tmp_path):
    collected = {
        0.0: {"accel_t0": np.arange(5.0), "hidden_state": np.ones((5, 512))},
        0.5: {"accel_t0": np.arange(5.0) * 2, "hidden_state": np.zeros((5, 512))},
        1.0: {"accel_t0": np.arange(5.0) * 3, "hidden_state": np.full((5, 512), 9.0)},
    }
    path = tmp_path / "e4.npz"
    save_cache(path, collected)
    back = load_cache(path)
    assert sorted(back) == [0.0, 0.5, 1.0]
    for a in collected:
        for k in collected[a]:
            assert np.array_equal(back[a][k], collected[a][k])
