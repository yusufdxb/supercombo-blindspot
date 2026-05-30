"""Tests for the baseline OOD detectors in src/baselines.py.

Covers: shape, monotonicity sanity (ID-like inputs score lower than
OOD-like), cache reproducibility, not-applicable baselines raise, LOCO
protocol mirrors e6_detector behaviour. Same calibration corpus
(teardown_collected.npz) and the same E4 cache as E6.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import baselines as B


# ---------------------- shape tests ------------------------------------

def test_mahalanobis_shape_and_finite():
    rng = np.random.RandomState(0)
    feats_id = rng.randn(200, 32).astype(np.float32)
    feats_test = rng.randn(50, 32).astype(np.float32)
    s = B.mahalanobis(feats_id, feats_test)
    assert s.shape == (50,)
    assert np.isfinite(s).all()


def test_relative_mahalanobis_shape_and_finite():
    rng = np.random.RandomState(0)
    feats_id = rng.randn(200, 16).astype(np.float32)
    feats_test = rng.randn(50, 16).astype(np.float32)
    s = B.relative_mahalanobis(feats_id, feats_test)
    assert s.shape == (50,)
    assert np.isfinite(s).all()


def test_knn_shape_and_finite():
    rng = np.random.RandomState(0)
    feats_id = rng.randn(200, 16).astype(np.float32)
    feats_test = rng.randn(50, 16).astype(np.float32)
    s = B.knn_distance(feats_id, feats_test, k=10)
    assert s.shape == (50,)
    assert np.isfinite(s).all()
    assert (s >= 0).all()


# ---------------------- monotonicity sanity -----------------------------

def test_mahalanobis_ood_scores_higher():
    rng = np.random.RandomState(1)
    feats_id = rng.normal(0.0, 1.0, size=(500, 16)).astype(np.float32)
    near = rng.normal(0.0, 1.0, size=(200, 16)).astype(np.float32)
    far = rng.normal(5.0, 1.0, size=(200, 16)).astype(np.float32)
    s_near = B.mahalanobis(feats_id, near)
    s_far = B.mahalanobis(feats_id, far)
    assert s_far.mean() > s_near.mean() * 2


def test_relative_mahalanobis_ood_scores_higher():
    rng = np.random.RandomState(2)
    feats_id = rng.normal(0.0, 1.0, size=(500, 16)).astype(np.float32)
    near = rng.normal(0.0, 1.0, size=(200, 16)).astype(np.float32)
    far = rng.normal(8.0, 1.0, size=(200, 16)).astype(np.float32)
    s_near = B.relative_mahalanobis(feats_id, near)
    s_far = B.relative_mahalanobis(feats_id, far)
    assert s_far.mean() > s_near.mean()


def test_knn_ood_scores_higher():
    rng = np.random.RandomState(3)
    feats_id = rng.normal(0.0, 1.0, size=(500, 16)).astype(np.float32)
    near = rng.normal(0.0, 1.0, size=(200, 16)).astype(np.float32)
    far = rng.normal(0.0, 1.0, size=(200, 16)).astype(np.float32) + 10.0
    s_near = B.knn_distance(feats_id, near, k=10)
    s_far = B.knn_distance(feats_id, far, k=10)
    assert s_far.mean() > s_near.mean()


# ---------------------- not-applicable raises --------------------------

@pytest.mark.parametrize("fn", [B.msp, B.energy, B.vim])
def test_not_applicable_baselines_raise(fn):
    with pytest.raises(NotImplementedError):
        fn()


# ---------------------- LOCO mirrors E6 protocol -----------------------

def test_loco_two_corpora_returns_per_fold_stats():
    rng = np.random.RandomState(0)
    sub = rng.normal(1.0, 0.1, size=(200, 16)).astype(np.float32)
    ram = rng.normal(1.1, 0.1, size=(200, 16)).astype(np.float32)
    res = B.loco_fpr("mahalanobis", {"subaru": sub, "ram": ram}, percentile=99.0)
    assert set(res["folds"].keys()) == {"subaru", "ram"}
    assert 0.0 <= res["fpr_mean"] <= 1.0
    assert 0.0 <= res["fpr_max"] <= 1.0
    assert res["fpr_max"] >= res["fpr_mean"]


# ---------------------- cache reproducibility --------------------------

def test_mahalanobis_is_deterministic():
    rng = np.random.RandomState(0)
    feats_id = rng.randn(200, 32).astype(np.float32)
    feats_test = rng.randn(50, 32).astype(np.float32)
    s1 = B.mahalanobis(feats_id, feats_test)
    s2 = B.mahalanobis(feats_id, feats_test)
    np.testing.assert_allclose(s1, s2)


def test_knn_is_deterministic():
    rng = np.random.RandomState(0)
    feats_id = rng.randn(200, 16).astype(np.float32)
    feats_test = rng.randn(50, 16).astype(np.float32)
    s1 = B.knn_distance(feats_id, feats_test, k=10)
    s2 = B.knn_distance(feats_id, feats_test, k=10)
    np.testing.assert_allclose(s1, s2)


def test_relative_mahalanobis_deterministic_at_fixed_seed():
    # The background GMM uses a fixed internal seed=0, so RMD should be
    # bit-identical across runs given the same ID set.
    rng = np.random.RandomState(0)
    feats_id = rng.randn(200, 16).astype(np.float32)
    feats_test = rng.randn(50, 16).astype(np.float32)
    s1 = B.relative_mahalanobis(feats_id, feats_test)
    s2 = B.relative_mahalanobis(feats_id, feats_test)
    np.testing.assert_allclose(s1, s2)


# ---------------------- end-to-end cache build -------------------------

def test_build_outputs_on_real_caches(tmp_path):
    teardown = Path("report/teardown_collected.npz")
    e4 = Path("report/e4_collected.npz")
    if not teardown.exists() or not e4.exists():
        pytest.skip("caches missing")
    out_npz = tmp_path / "baselines_collected.npz"
    out_md = tmp_path / "baselines_results.md"
    res = B._build_outputs(out_npz, out_md, percentile=99.0)
    assert set(res.keys()) == set(B.APPLICABLE_BASELINES)
    for name, r in res.items():
        assert 0.0 <= r["loco"]["fpr_mean"] <= 1.0
        assert len(r["alphas"]) == len(r["fired_fraction"])
    assert out_npz.exists()
    assert out_md.exists()
    # Reload and confirm keys for every baseline x alpha exist.
    d = np.load(out_npz)
    for name in B.APPLICABLE_BASELINES:
        assert f"{name}__id_scores" in d.files
