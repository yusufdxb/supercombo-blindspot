"""Unit tests for src/ablations.py.

Tests sweep logic, output format, and edge cases using synthetic data.
No model, GPU, or real data caches required.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.ablations import (
    fmt_ci,
    sweep_knn_k,
    sweep_e6_window,
    write_cache,
    write_report,
)


# --- synthetic data helpers ------------------------------------------------

def _make_id_ood(
    n_id: int = 200,
    n_ood: int = 100,
    dim: int = 32,
    sep: float = 3.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Separated ID and OOD feature arrays for unit tests."""
    rng = np.random.RandomState(seed)
    id_h = rng.normal(0.0, 1.0, size=(n_id, dim)).astype(np.float32)
    ood_h = rng.normal(sep, 0.5, size=(n_ood, dim)).astype(np.float32)
    return id_h, ood_h


def _stub_fires_at_alpha(value: float = 0.55):
    """Return a callable that ignores all args and returns `value`."""
    def _stub(*_args, **_kwargs):
        return value
    return _stub


# --- KNN k sweep tests ----------------------------------------------------

class TestSweepKnnK:
    """Tests for sweep_knn_k()."""

    def test_returns_one_dict_per_k(self):
        id_h, ood_h = _make_id_ood()
        ks = [5, 10, 20]
        results = sweep_knn_k(id_h, ood_h, k_values=ks, n_bootstrap=10,
                              seed=0)
        assert len(results) == len(ks)
        for r, k in zip(results, ks):
            assert r["k"] == k

    def test_result_keys(self):
        id_h, ood_h = _make_id_ood()
        results = sweep_knn_k(id_h, ood_h, k_values=[5], n_bootstrap=10,
                              seed=0)
        r = results[0]
        assert "k" in r
        assert "auroc_point" in r
        assert "auroc_ci" in r
        assert len(r["auroc_ci"]) == 3  # (mean, lo, hi)

    def test_auroc_is_valid(self):
        id_h, ood_h = _make_id_ood(sep=5.0)
        results = sweep_knn_k(id_h, ood_h, k_values=[5, 20],
                              n_bootstrap=10, seed=0)
        for r in results:
            assert 0.0 <= r["auroc_point"] <= 1.0
            m, lo, hi = r["auroc_ci"]
            assert lo <= m <= hi

    def test_ci_contains_point_estimate(self):
        id_h, ood_h = _make_id_ood(sep=5.0)
        results = sweep_knn_k(id_h, ood_h, k_values=[10],
                              n_bootstrap=50, seed=0)
        r = results[0]
        _, lo, hi = r["auroc_ci"]
        # Point estimate should be close to CI bounds (within 0.1 tolerance
        # for small bootstrap samples).
        assert r["auroc_point"] > lo - 0.1
        assert r["auroc_point"] < hi + 0.1

    def test_well_separated_data_high_auroc(self):
        # High-dimensional tight ID cluster with OOD far away.
        # After L2-normalisation the ID cluster maps to a small cap
        # on the unit sphere (mean=[1,0,...] + small perturbations)
        # while OOD maps to the opposite side (mean=[-1,0,...]).
        # k-th NN distance for OOD-to-ID is then genuinely large.
        rng = np.random.RandomState(7)
        dim = 64
        id_h = rng.normal(0.0, 0.1, size=(200, dim)).astype(np.float32)
        id_h[:, 0] += 5.0  # cluster around [5, 0, ..., 0]
        ood_h = rng.normal(0.0, 0.1, size=(100, dim)).astype(np.float32)
        ood_h[:, 0] -= 5.0  # cluster around [-5, 0, ..., 0]
        results = sweep_knn_k(id_h, ood_h, k_values=[5], n_bootstrap=10,
                              seed=0)
        assert results[0]["auroc_point"] > 0.9

    def test_k_larger_than_id_set(self):
        """k > n_id should not crash; knn_distance clamps k."""
        id_h, ood_h = _make_id_ood(n_id=10)
        results = sweep_knn_k(id_h, ood_h, k_values=[50], n_bootstrap=10,
                              seed=0)
        assert len(results) == 1
        assert np.isfinite(results[0]["auroc_point"])


# --- E6 window sweep tests ------------------------------------------------

class TestSweepE6Window:
    """Tests for sweep_e6_window() with injected fires_at_alpha stub."""

    def test_returns_one_dict_per_window(self):
        id_h, ood_h = _make_id_ood(n_id=100, n_ood=80)
        ws = [10, 20]
        results = sweep_e6_window(
            id_h, ood_h, window_values=ws, n_bootstrap=10, seed=0,
            _fires_at_alpha_fn=_stub_fires_at_alpha(0.55))
        assert len(results) == len(ws)
        for r, w in zip(results, ws):
            assert r["window"] == w

    def test_result_keys(self):
        id_h, ood_h = _make_id_ood(n_id=100, n_ood=80)
        results = sweep_e6_window(
            id_h, ood_h, window_values=[10], n_bootstrap=10, seed=0,
            _fires_at_alpha_fn=_stub_fires_at_alpha(0.55))
        r = results[0]
        assert "window" in r
        assert "auroc_point" in r
        assert "auroc_ci" in r
        assert "fires_at_alpha" in r
        assert len(r["auroc_ci"]) == 3

    def test_auroc_is_valid(self):
        id_h, ood_h = _make_id_ood(n_id=100, n_ood=80, sep=5.0)
        results = sweep_e6_window(
            id_h, ood_h, window_values=[10, 20], n_bootstrap=10, seed=0,
            _fires_at_alpha_fn=_stub_fires_at_alpha(0.55))
        for r in results:
            assert 0.0 <= r["auroc_point"] <= 1.0
            m, lo, hi = r["auroc_ci"]
            assert lo <= m <= hi

    def test_fires_at_alpha_propagated(self):
        id_h, ood_h = _make_id_ood(n_id=100, n_ood=80)
        sentinel = 0.42
        results = sweep_e6_window(
            id_h, ood_h, window_values=[10], n_bootstrap=10, seed=0,
            _fires_at_alpha_fn=_stub_fires_at_alpha(sentinel))
        assert results[0]["fires_at_alpha"] == sentinel

    def test_window_larger_than_sequence(self):
        """Window > T should produce all-NaN spread, yielding NaN AUROC."""
        id_h, ood_h = _make_id_ood(n_id=5, n_ood=5)
        # window=100 > n_id=5: all frames will be NaN after rolling_spread.
        results = sweep_e6_window(
            id_h, ood_h, window_values=[100], n_bootstrap=10, seed=0,
            _fires_at_alpha_fn=_stub_fires_at_alpha(float("nan")))
        # With 0 valid frames, auroc returns NaN.
        assert np.isnan(results[0]["auroc_point"])


# --- report writer tests --------------------------------------------------

class TestWriteReport:
    """Tests for write_report() output format."""

    def test_writes_markdown_file(self, tmp_path):
        knn = [
            {"k": 5, "auroc_point": 0.99, "auroc_ci": (0.99, 0.98, 1.0)},
            {"k": 50, "auroc_point": 1.0, "auroc_ci": (1.0, 1.0, 1.0)},
        ]
        e6 = [
            {"window": 10, "auroc_point": 0.95, "auroc_ci": (0.95, 0.93, 0.97),
             "fires_at_alpha": 0.6},
            {"window": 30, "auroc_point": 0.99, "auroc_ci": (0.99, 0.98, 1.0),
             "fires_at_alpha": 0.55},
        ]
        out = tmp_path / "ablations_results.md"
        write_report(knn, e6, out)
        assert out.exists()
        text = out.read_text()
        assert "# Hyperparameter Sensitivity Ablations" in text

    def test_contains_knn_table(self, tmp_path):
        knn = [{"k": 5, "auroc_point": 0.99, "auroc_ci": (0.99, 0.98, 1.0)}]
        e6 = [{"window": 10, "auroc_point": 0.95,
               "auroc_ci": (0.95, 0.93, 0.97), "fires_at_alpha": 0.6}]
        out = tmp_path / "ablations_results.md"
        write_report(knn, e6, out)
        text = out.read_text()
        assert "KNN k-sensitivity" in text
        assert "| k | AUROC" in text
        assert "| 5 |" in text

    def test_contains_e6_table(self, tmp_path):
        knn = [{"k": 50, "auroc_point": 1.0, "auroc_ci": (1.0, 1.0, 1.0)}]
        e6 = [
            {"window": 10, "auroc_point": 0.95,
             "auroc_ci": (0.95, 0.93, 0.97), "fires_at_alpha": 0.6},
            {"window": 30, "auroc_point": 0.99,
             "auroc_ci": (0.99, 0.98, 1.0), "fires_at_alpha": 0.55},
        ]
        out = tmp_path / "ablations_results.md"
        write_report(knn, e6, out)
        text = out.read_text()
        assert "E6 window-size sensitivity" in text
        assert "| window | AUROC" in text
        assert "| 10 |" in text
        assert "| 30 |" in text
        assert "0.550" in text  # fires_at_alpha formatted

    def test_contains_summary_section(self, tmp_path):
        knn = [
            {"k": 5, "auroc_point": 0.99, "auroc_ci": (0.99, 0.98, 1.0)},
            {"k": 50, "auroc_point": 1.0, "auroc_ci": (1.0, 1.0, 1.0)},
        ]
        e6 = [
            {"window": 10, "auroc_point": 0.95,
             "auroc_ci": (0.95, 0.93, 0.97), "fires_at_alpha": 0.6},
        ]
        out = tmp_path / "ablations_results.md"
        write_report(knn, e6, out)
        text = out.read_text()
        assert "## Summary" in text

    def test_nan_fires_at_alpha_renders_as_na(self, tmp_path):
        knn = [{"k": 50, "auroc_point": 1.0, "auroc_ci": (1.0, 1.0, 1.0)}]
        e6 = [{"window": 10, "auroc_point": 0.95,
               "auroc_ci": (0.95, 0.93, 0.97),
               "fires_at_alpha": float("nan")}]
        out = tmp_path / "ablations_results.md"
        write_report(knn, e6, out)
        text = out.read_text()
        assert "n/a" in text


# --- cache writer tests ---------------------------------------------------

class TestWriteCache:
    """Tests for write_cache() .npz output."""

    def test_writes_npz(self, tmp_path):
        knn = [
            {"k": 5, "auroc_point": 0.99, "auroc_ci": (0.99, 0.98, 1.0)},
            {"k": 50, "auroc_point": 1.0, "auroc_ci": (1.0, 1.0, 1.0)},
        ]
        e6 = [
            {"window": 10, "auroc_point": 0.95,
             "auroc_ci": (0.95, 0.93, 0.97), "fires_at_alpha": 0.6},
        ]
        out = tmp_path / "ablations_collected.npz"
        write_cache(knn, e6, out)
        assert out.exists()
        d = np.load(out)
        assert "knn_k_values" in d.files
        assert "knn_auroc_points" in d.files
        assert "knn_auroc_cis" in d.files
        assert "e6_window_values" in d.files
        assert "e6_auroc_points" in d.files
        assert "e6_auroc_cis" in d.files
        assert "e6_fires_at_alpha" in d.files
        np.testing.assert_array_equal(d["knn_k_values"], [5, 50])
        np.testing.assert_array_equal(d["e6_window_values"], [10])

    def test_cache_shapes(self, tmp_path):
        knn = [
            {"k": 5, "auroc_point": 0.99, "auroc_ci": (0.99, 0.98, 1.0)},
            {"k": 10, "auroc_point": 0.995, "auroc_ci": (0.995, 0.99, 1.0)},
            {"k": 20, "auroc_point": 1.0, "auroc_ci": (1.0, 1.0, 1.0)},
        ]
        e6 = [
            {"window": 10, "auroc_point": 0.95,
             "auroc_ci": (0.95, 0.93, 0.97), "fires_at_alpha": 0.6},
            {"window": 30, "auroc_point": 0.99,
             "auroc_ci": (0.99, 0.98, 1.0), "fires_at_alpha": 0.55},
        ]
        out = tmp_path / "ablations_collected.npz"
        write_cache(knn, e6, out)
        d = np.load(out)
        assert d["knn_k_values"].shape == (3,)
        assert d["knn_auroc_points"].shape == (3,)
        assert d["knn_auroc_cis"].shape == (3, 3)
        assert d["e6_window_values"].shape == (2,)
        assert d["e6_auroc_points"].shape == (2,)
        assert d["e6_auroc_cis"].shape == (2, 3)
        assert d["e6_fires_at_alpha"].shape == (2,)


# --- fmt_ci test -----------------------------------------------------------

def test_fmt_ci():
    s = fmt_ci((0.996, 0.992, 1.000))
    assert s == "0.996 [0.992, 1.000]"
