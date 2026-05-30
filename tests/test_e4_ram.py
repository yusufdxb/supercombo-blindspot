"""Tests for E4-RAM: real-to-sim interpolation sweep with RAM source."""

from __future__ import annotations

import numpy as np
import pytest

from src.e4_ram import CACHE, load_cache, save_cache


def test_cache_round_trip(tmp_path):
    """save_cache -> load_cache preserves all alpha keys and arrays."""
    collected = {
        0.0: {"accel_t0": np.arange(5.0), "hidden_state": np.ones((5, 512))},
        0.5: {"accel_t0": np.arange(5.0) * 2, "hidden_state": np.zeros((5, 512))},
        1.0: {"accel_t0": np.arange(5.0) * 3, "hidden_state": np.full((5, 512), 9.0)},
    }
    path = tmp_path / "e4_ram.npz"
    save_cache(path, collected)
    back = load_cache(path)
    assert sorted(back) == [0.0, 0.5, 1.0]
    for a in collected:
        for k in collected[a]:
            assert np.array_equal(back[a][k], collected[a][k])


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


from src.e4_ram import (
    _load_subaru_cliff,
    activity_per_head,
    feature_centroid,
    feature_projection,
    normalized_activity,
    transition_width,
)
from src.teardown import WARMUP as _WARMUP


def test_analysis_from_synthetic():
    """Verify analysis pipeline runs on synthetic data without error."""
    seg_real = _synthetic_segment(1.0, 40)
    seg_carla = _synthetic_segment(0.01, 40)
    per_head = {
        0.0: activity_per_head(seg_real),
        1.0: activity_per_head(seg_carla),
    }
    norm = normalized_activity(per_head)
    assert abs(norm[0.0] - 1.0) < 1e-9
    assert norm[1.0] < 0.5


@pytest.mark.skipif(not CACHE.exists(),
                    reason=f"E4-RAM cache {CACHE.name} not present")
class TestE4RamRegression:
    """Pins the E4-RAM result from the committed cache."""

    @pytest.fixture(scope="class")
    def swept(self):
        from src.e4_ram import _post
        cached = load_cache(CACHE)
        post = {a: _post(cached[a], _WARMUP) for a in cached}
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
        assert norm[1.0] < 0.2, norm[1.0]
        assert abs(fproj[1.0] - 1.0) < 1e-6

    def test_transition_is_a_gradient(self, swept):
        alphas, norm, _ = swept
        a90, a10 = transition_width(alphas, norm)
        assert np.isfinite(a90) and np.isfinite(a10)
        width = a10 - a90
        assert 0.0 <= width <= 1.0
        # RAM shows a gradient (width ~0.27), unlike Subaru's cliff (0.015)
        assert width > 0.15, width
