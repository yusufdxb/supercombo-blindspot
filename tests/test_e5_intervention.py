"""Tests for src.e5_intervention (summarizer-bottleneck causal intervention)."""
from __future__ import annotations

import numpy as np
import pytest

from src import e5_intervention as ev


def test_make_conditions_keys_and_shapes():
    real = np.random.randn(20, 512).astype(np.float32)
    carla = np.random.randn(20, 512).astype(np.float32) + 3.0
    vr = np.random.randn(20, 2048).astype(np.float32)
    vc = np.random.randn(20, 2048).astype(np.float32)
    conds = ev.make_conditions(real, carla, vr, vc)
    assert set(conds) == {"real_baseline", "carla_baseline", "real_div_only",
                          "mu_swap", "scale_swap", "real_history"}
    for cur, buf, vis in conds.values():
        assert cur.shape == (20, 512) and buf.shape == (20, 512)
        assert vis.shape == (20, 2048)


def test_mu_swap_matches_real_mean():
    real = np.random.randn(50, 512).astype(np.float32)
    carla = np.random.randn(50, 512).astype(np.float32) + 5.0
    vc = np.zeros((50, 2048), np.float32)
    cur, _, _ = ev.make_conditions(real, carla, vc[:, :512], vc)["mu_swap"]
    # mean replaced by real mean, fluctuations preserved
    np.testing.assert_allclose(cur.mean(0), real.mean(0), atol=1e-4)
    np.testing.assert_allclose(cur - cur.mean(0), carla - carla.mean(0), atol=1e-4)


def test_scale_swap_matches_real_std():
    real = np.random.randn(50, 512).astype(np.float32) * 2.0
    carla = np.random.randn(50, 512).astype(np.float32) * 0.5 + 1.0
    vc = np.zeros((50, 2048), np.float32)
    cur, _, _ = ev.make_conditions(real, carla, vc[:, :512], vc)["scale_swap"]
    np.testing.assert_allclose(cur.std(0), real.std(0), rtol=0.05, atol=1e-3)


def test_real_history_decouples_current_and_buffer():
    real = np.random.randn(10, 512).astype(np.float32)
    carla = np.random.randn(10, 512).astype(np.float32) + 9.0
    vc = np.zeros((10, 2048), np.float32)
    cur, buf, _ = ev.make_conditions(real, carla, vc[:, :512], vc)["real_history"]
    np.testing.assert_array_equal(cur, carla)   # current token is CARLA
    np.testing.assert_array_equal(buf, real)    # history buffer is real


def test_activity_table_real_baseline_excluded_and_ratio_one_for_identical():
    # two identical recs -> ratio 1.0 for every head
    n = 30
    rec = {"plan": np.random.randn(n, 100).astype(np.float32),
           "accel_t0": np.random.randn(n).astype(np.float32),
           "desired_curv": np.random.randn(n).astype(np.float32),
           "lead_prob": np.random.randn(n).astype(np.float32),
           "lane_lines": np.random.randn(n, 50).astype(np.float32),
           "road_edges": np.random.randn(n, 50).astype(np.float32),
           "lead": np.random.randn(n, 30).astype(np.float32),
           "pose": np.random.randn(n, 12).astype(np.float32),
           "desire_state": np.random.randn(n, 8).astype(np.float32),
           "meta": np.random.randn(n, 48).astype(np.float32)}
    results = {"real_baseline": rec, "carla_baseline": {k: v.copy() for k, v in rec.items()}}
    rows = ev.activity_table(results)
    assert all("real_baseline" not in r for r in rows)
    for r in rows:
        assert r["carla_baseline"] == pytest.approx(1.0, abs=1e-6)


def test_summarise_falsifies_dc_offset_when_mu_swap_collapsed():
    # real_div_only ~1 (by construction), mu_swap collapsed -> verdict must say
    # the DC-offset hypothesis is falsified and not claim a simple mean fix.
    rows = []
    for h in ["plan", "lane_lines", "road_edges", "lead", "desire_state"]:
        rows.append({"head": h, "carla_baseline": 0.03, "real_div_only": 1.0,
                     "mu_swap": 0.01, "scale_swap": 0.18, "real_history": 1.3})
    s = ev.summarise(rows)
    assert s["carla_collapsed"] == 5
    assert "falsified" in s["verdict"].lower()
    assert "construction" in s["verdict"].lower()
