"""Tests for scripts/attractor_diagnostic.py (H1+H2 offline diagnostic)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers from the script
# ---------------------------------------------------------------------------

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location(
    "attractor_diagnostic",
    Path(__file__).parent.parent / "scripts" / "attractor_diagnostic.py",
)
_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_mod)  # type: ignore[union-attr]

NORM_THRESHOLD = _mod.NORM_THRESHOLD
_regime_labels = _mod._regime_labels
h1_curvature_lag = _mod.h1_curvature_lag
h2_basin_similarity = _mod.h2_basin_similarity


def test_regime_labels_bimodal():
    norms = np.array([0.004, 0.005, 0.8, 1.0, 0.003])
    labels = _regime_labels(norms)
    assert labels[0] == 0  # low
    assert labels[1] == 0  # low
    assert labels[2] == 1  # high
    assert labels[3] == 1  # high
    assert labels[4] == 0  # low


def test_regime_labels_gap_is_minus_one():
    norms = np.array([0.3])  # in gap between threshold and 0.5
    labels = _regime_labels(norms)
    assert labels[0] == -1


def test_regime_labels_threshold():
    just_below = np.array([NORM_THRESHOLD - 1e-6])
    just_above = np.array([NORM_THRESHOLD + 1e-6])
    assert _regime_labels(just_below)[0] == 0
    assert _regime_labels(just_above)[0] == -1  # in gap, not yet high


def test_h1_low_curvature_detects_h1():
    """Synthetic H1 scenario: low-norm frames have near-zero curvature."""
    rng = np.random.RandomState(0)
    n = 100
    labels = np.zeros(n, dtype=np.int8)
    labels[50:] = 1  # second half is high-norm
    desired_curv = np.zeros(n)
    desired_curv[50:] = rng.uniform(0.1, 0.2, 50)  # high-norm has nonzero curv
    result = h1_curvature_lag(desired_curv, labels, max_lag=3)
    # At lag=0, low should have lower |curv| than high: ratio < 1
    assert result[0]["ratio"] < 1.0


def test_h1_returns_all_lags():
    n = 50
    labels = np.array([0] * 25 + [1] * 25, dtype=np.int8)
    curv = np.random.rand(n)
    result = h1_curvature_lag(curv, labels, max_lag=4)
    assert set(result.keys()) == {0, 1, 2, 3, 4}
    for v in result.values():
        assert "low_mean_abs_curv" in v
        assert "high_mean_abs_curv" in v
        assert "ratio" in v


def test_h2_shared_basin_detected():
    """Synthetic H2 scenario: low-norm states are cosine-close to CARLA."""
    pytest.importorskip("sklearn")
    rng = np.random.RandomState(42)
    d = 8  # low-d so cosine is stable despite small cluster radii
    carla_dir = np.ones(d, dtype=np.float32)
    carla_dir /= np.linalg.norm(carla_dir)
    high_dir = -carla_dir  # opposite direction; unambiguous separation

    # Very tight clusters so k-means reliably finds the right centroids
    carla_states = (carla_dir * 0.02 + rng.randn(30, d).astype(np.float32) * 2e-4)
    dc_low = (carla_dir * 0.015 + rng.randn(40, d).astype(np.float32) * 2e-4)
    dc_high = (high_dir * 0.8 + rng.randn(20, d).astype(np.float32) * 2e-4)
    dc_states = np.vstack([dc_low, dc_high]).astype(np.float32)
    labels = np.array([0] * 40 + [1] * 20, dtype=np.int8)
    result = h2_basin_similarity(dc_states, labels, carla_states)
    assert result["cosine_low_vs_carla"] > 0.9, (
        f"Expected cosine > 0.9 for shared-basin synthetic, got {result['cosine_low_vs_carla']:.3f}")


def test_h2_different_basin_detected():
    """H2 different-basin: low-norm states point in different direction from CARLA."""
    pytest.importorskip("sklearn")
    rng = np.random.RandomState(7)
    d = 32
    carla_dir = rng.randn(d); carla_dir /= np.linalg.norm(carla_dir)
    orthogonal_dir = rng.randn(d)
    orthogonal_dir -= np.dot(orthogonal_dir, carla_dir) * carla_dir
    orthogonal_dir /= np.linalg.norm(orthogonal_dir)
    carla_states = (carla_dir * 0.02 + rng.randn(30, d) * 0.001).astype(np.float32)
    dc_low = (orthogonal_dir * 0.015 + rng.randn(40, d) * 0.001).astype(np.float32)
    dc_high = (rng.randn(20, d)).astype(np.float32)
    dc_states = np.vstack([dc_low, dc_high]).astype(np.float32)
    labels = np.array([0] * 40 + [1] * 20, dtype=np.int8)
    result = h2_basin_similarity(dc_states, labels, carla_states)
    assert result["cosine_low_vs_carla"] < 0.4, (
        f"Expected cosine < 0.4 for orthogonal-basin synthetic, got {result['cosine_low_vs_carla']:.3f}")


def test_h2_purity_near_one_for_clean_bimodal():
    """k=2 should perfectly recover the regime labels for clean synthetic data."""
    pytest.importorskip("sklearn")
    rng = np.random.RandomState(0)
    d = 16
    low = rng.randn(50, d).astype(np.float32) * 0.005
    high = (rng.randn(50, d).astype(np.float32) + 5)
    dc_states = np.vstack([low, high])
    carla = rng.randn(10, d).astype(np.float32) * 0.004
    labels = np.array([0] * 50 + [1] * 50, dtype=np.int8)
    result = h2_basin_similarity(dc_states, labels, carla)
    assert result["label_purity"] > 0.95


def test_diagnostic_on_real_caches():
    """Full end-to-end: run on the real cached data and check result file written."""
    rw = Path("report/real_weather_collected.npz")
    td = Path("report/teardown_collected.npz")
    if not rw.exists() or not td.exists():
        pytest.skip("real weather or teardown caches not present")
    pytest.importorskip("sklearn")
    results = Path("report/attractor_diagnostic_results.md")
    main = _mod.main
    rc = main()
    assert rc == 0
    assert results.exists()
    text = results.read_text()
    assert "H1 verdict" in text
    assert "H2 verdict" in text
    # Verify the hard-bimodal gap is still zero
    assert "Gap (0.1-0.5, should be 0): 0" in text
