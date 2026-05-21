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


from src.e4_interp import (activity_per_head, feature_centroid,
                           feature_projection, feature_spread,
                           mean_uncertainty, normalized_activity)


def _synthetic_segment(scale: float, n: int = 40):
    """A collected-style dict whose per-frame variation scales with `scale`."""
    rng = np.random.default_rng(0)
    d = {}
    for name in ["accel_t0", "desired_curv", "lead_prob"]:
        d[name] = rng.normal(0, scale, n)
    for name in ["plan", "lane_lines", "road_edges", "lead", "pose",
                 "desire_state", "meta"]:
        d[name] = rng.normal(0, scale, (n, 8))
    d["hidden_state"] = rng.normal(0, scale, (n, 512)) + scale * 100.0
    d["plan_std"] = np.full((n, 8), scale)
    return d


def test_activity_per_head_tracks_scale():
    lo = activity_per_head(_synthetic_segment(0.1))
    hi = activity_per_head(_synthetic_segment(1.0))
    assert all(hi[h] > lo[h] for h in lo)


def test_normalized_activity_is_one_at_alpha_zero():
    per_head = {0.0: activity_per_head(_synthetic_segment(1.0)),
                1.0: activity_per_head(_synthetic_segment(0.01))}
    norm = normalized_activity(per_head)
    assert abs(norm[0.0] - 1.0) < 1e-9
    assert norm[1.0] < 0.5


def test_feature_projection_endpoints():
    cents = {0.0: np.zeros(512), 0.5: np.full(512, 0.5), 1.0: np.ones(512)}
    proj = feature_projection(cents)
    assert abs(proj[0.0] - 0.0) < 1e-9
    assert abs(proj[1.0] - 1.0) < 1e-9
    assert abs(proj[0.5] - 0.5) < 1e-9


def test_feature_spread_and_uncertainty():
    seg = _synthetic_segment(1.0)
    assert feature_spread(seg) > 0.0
    assert feature_centroid(seg).shape == (512,)
    assert abs(mean_uncertainty(seg) - 1.0) < 1e-9


from src.e4_interp import CACHE, CLIFF_WIDTH
from src.teardown import WARMUP as _WARMUP
from src.e4_interp import _post as _e4_post  # re-exported from teardown


@pytest.mark.skipif(not CACHE.exists(),
                    reason=f"E4 cache {CACHE.name} not present")
class TestE4Regression:
    """Pins the E4 result from the committed cache. numpy only."""

    @pytest.fixture(scope="class")
    def swept(self):
        cached = load_cache(CACHE)
        post = {a: _e4_post(cached[a], _WARMUP) for a in cached}
        alphas = sorted(post)
        per_head = {a: activity_per_head(post[a]) for a in alphas}
        norm = normalized_activity(per_head)
        cents = {a: feature_centroid(post[a]) for a in alphas}
        return alphas, norm, feature_projection(cents)

    def test_alpha_zero_is_real_alive(self, swept):
        alphas, norm, fproj = swept
        assert alphas[0] == 0.0 and alphas[-1] == 1.0
        assert abs(norm[0.0] - 1.0) < 1e-6
        assert abs(fproj[0.0]) < 1e-6

    def test_alpha_one_is_carla_collapsed(self, swept):
        _, norm, fproj = swept
        # alpha=1 must reproduce E1's CARLA collapse
        assert norm[1.0] < 0.2, norm[1.0]
        assert abs(fproj[1.0] - 1.0) < 1e-6

    def test_transition_is_a_cliff(self, swept):
        alphas, norm, _ = swept
        a90, a10 = transition_width(alphas, norm)
        assert np.isfinite(a90) and np.isfinite(a10)
        width = a10 - a90
        assert 0.0 <= width <= 1.0
        # the measured E4 verdict: the collapse is a cliff, not a gradient
        assert width < CLIFF_WIDTH, width
