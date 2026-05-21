"""Regression test for the distribution-shift teardown headline result.

Runs the E1/E2/E3 analysis on the committed collected-output cache
(report/teardown_collected.npz) and asserts the published verdict. This pins
the finding: if the analysis logic or the cache changes, CI catches it. Needs
only numpy -- no model, no CARLA, no raw frames.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.teardown import (
    CACHE,
    COLLAPSE,
    WARMUP,
    _load_cache,
    _post,
    e1_collapse_map,
    e2_feature_ood,
    e3_confidence,
)

pytestmark = pytest.mark.skipif(
    not CACHE.exists(),
    reason=f"collected-output cache {CACHE.name} not present",
)


@pytest.fixture(scope="module")
def analysis():
    """Mirror src.teardown.main's analysis prep: load the cache, drop the
    per-segment warmup transient, concatenate the two real segments."""
    segments = _load_cache(CACHE)
    subaru = _post(segments["subaru"], WARMUP)
    ram = _post(segments["ram"], WARMUP)
    carla = _post(segments["carla"], WARMUP)
    real = {k: np.concatenate([subaru[k], ram[k]]) for k in subaru}
    return real, carla


def test_e1_eight_of_ten_heads_collapse(analysis):
    real, carla = analysis
    rows = e1_collapse_map(real, carla)
    assert len(rows) == 10
    survived = {r["head"] for r in rows if r["ratio"] >= COLLAPSE}
    collapsed = {r["head"] for r in rows if r["ratio"] < COLLAPSE}
    # 8 of 10 heads collapse on CARLA; pose and meta are the two that survive.
    assert len(collapsed) == 8, f"expected 8 collapsed heads, got {sorted(collapsed)}"
    assert survived == {"pose", "meta"}


def test_e2_feature_space_collapses(analysis):
    real, carla = analysis
    e2 = e2_feature_ood(real, carla)
    # CARLA's recurrent feature spread collapses to a tiny fraction of real.
    assert e2["spread_ratio"] < 1e-3, e2["spread_ratio"]
    # and the two distributions are cleanly separable (out of distribution).
    assert e2["dprime"] > 2.0, e2["dprime"]


def test_e3_failure_is_silent(analysis):
    real, carla = analysis
    rows = e3_confidence(real, carla)
    assert len(rows) == 3
    # outputs lose ~99% of their activity ...
    assert all(r["out_ratio"] < 0.05 for r in rows), [r["out_ratio"] for r in rows]
    # ... yet not one CARLA frame's predicted uncertainty leaves the model's
    # normal real-driving range. The failure is undetectable from the outputs.
    assert all(r["carla_above_real_p95"] == 0.0 for r in rows)
