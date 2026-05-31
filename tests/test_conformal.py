"""Tests for the split-conformal OOD detector in src/baselines.py.

Covers: shape, p-value super-uniformity for ID, high OOD scores for
out-of-distribution inputs, LOCO protocol, and end-to-end report generation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src import baselines as B


# ---- shape and range -------------------------------------------------------

def test_conformal_pvalues_shape():
    rng = np.random.RandomState(0)
    cal = rng.uniform(0, 1, size=200).astype(np.float64)
    test = rng.uniform(0, 1, size=50).astype(np.float64)
    pv = B.conformal_pvalues(cal, test)
    assert pv.shape == (50,)


def test_conformal_pvalues_in_range():
    rng = np.random.RandomState(1)
    cal = rng.uniform(0, 1, size=300).astype(np.float64)
    test = rng.uniform(0, 1, size=100).astype(np.float64)
    pv = B.conformal_pvalues(cal, test)
    assert np.all(np.isfinite(pv))
    # 1 - pvalue is in [0, 1)
    assert np.all(pv >= 0.0)
    assert np.all(pv < 1.0)


def test_conformal_ood_shape():
    rng = np.random.RandomState(0)
    feats_id = rng.randn(200, 16).astype(np.float32)
    feats_test = rng.randn(50, 16).astype(np.float32)
    s = B.conformal_ood(feats_id, feats_test, k=10)
    assert s.shape == (50,)
    assert np.isfinite(s).all()


# ---- p-value super-uniformity (ID samples) ----------------------------------

def test_conformal_id_scores_near_uniform():
    """Under exchangeability, p-values on ID-like test points should be
    approximately uniform in [0,1], so 1-pvalue should have mean ~0.5."""
    rng = np.random.RandomState(42)
    feats_id = rng.normal(0.0, 1.0, size=(500, 8)).astype(np.float32)
    feats_test = rng.normal(0.0, 1.0, size=(200, 8)).astype(np.float32)
    s = B.conformal_ood(feats_id, feats_test, k=20)
    # Mean should be near 0.5 (uniform distribution over [0, 1))
    assert 0.3 < s.mean() < 0.7, f"mean={s.mean():.3f}, expected near 0.5"


# ---- OOD inputs score high --------------------------------------------------

def test_conformal_ood_scores_high_for_far_points():
    """OOD points from a far distribution should score near 1.0."""
    rng = np.random.RandomState(10)
    feats_id = rng.normal(0.0, 1.0, size=(400, 16)).astype(np.float32)
    feats_ood = rng.normal(10.0, 1.0, size=(100, 16)).astype(np.float32)
    s_ood = B.conformal_ood(feats_id, feats_ood, k=20)
    # All OOD points should score > 0.9
    assert np.all(s_ood > 0.9), f"min OOD score={s_ood.min():.3f}"


# ---- LOCO protocol ----------------------------------------------------------

def test_conformal_loco_two_corpora():
    """LOCO for conformal should return per-fold stats via the generic
    loco_fpr dispatcher."""
    rng = np.random.RandomState(0)
    sub = rng.normal(0.0, 0.1, size=(200, 16)).astype(np.float32)
    ram = rng.normal(0.0, 0.1, size=(200, 16)).astype(np.float32)
    res = B.loco_fpr("conformal", {"subaru": sub, "ram": ram}, percentile=99.0)
    assert set(res["folds"].keys()) == {"subaru", "ram"}
    assert 0.0 <= res["fpr_mean"] <= 1.0
    assert res["fpr_max"] >= res["fpr_mean"]


def test_conformal_in_applicable_baselines():
    assert "conformal" in B.APPLICABLE_BASELINES


def test_conformal_score_dispatcher():
    """_score('conformal', ...) should return the same as conformal_ood."""
    rng = np.random.RandomState(5)
    feats_id = rng.randn(100, 8).astype(np.float32)
    feats_test = rng.randn(30, 8).astype(np.float32)
    s1 = B._score("conformal", feats_id, feats_test)
    s2 = B.conformal_ood(feats_id, feats_test, k=50)
    np.testing.assert_allclose(s1, s2)


# ---- end-to-end on real caches --------------------------------------------

def test_conformal_on_real_caches(tmp_path):
    teardown = Path("report/teardown_collected.npz")
    e4 = Path("report/e4_collected.npz")
    if not teardown.exists() or not e4.exists():
        pytest.skip("caches missing")
    d = np.load(teardown)
    all_real = np.concatenate([d["subaru__hidden_state"], d["ram__hidden_state"]])
    d4 = np.load(e4)
    ood_h = d4["1.0000__hidden_state"]
    s_id = B._score("conformal", all_real, all_real)
    s_ood = B._score("conformal", all_real, ood_h)
    assert np.isfinite(s_id).all()
    assert np.isfinite(s_ood).all()
    # OOD should score substantially higher than ID
    assert s_ood.mean() > s_id.mean()


def test_conformal_results_module_on_real_caches(tmp_path):
    teardown = Path("report/teardown_collected.npz")
    e4 = Path("report/e4_collected.npz")
    if not teardown.exists() or not e4.exists():
        pytest.skip("caches missing")
    from src.conformal_results import run
    out = tmp_path / "conformal_results.md"
    results = run(out)
    auroc_m = results["auroc"][0]
    assert 0.5 < auroc_m <= 1.0
    assert out.exists()


# ---- build_outputs includes conformal -------------------------------------

def test_build_outputs_includes_conformal(tmp_path):
    teardown = Path("report/teardown_collected.npz")
    e4 = Path("report/e4_collected.npz")
    if not teardown.exists() or not e4.exists():
        pytest.skip("caches missing")
    out_npz = tmp_path / "baselines_collected.npz"
    out_md = tmp_path / "baselines_results.md"
    res = B._build_outputs(out_npz, out_md, percentile=99.0)
    assert "conformal" in res
    d = np.load(out_npz)
    assert "conformal__id_scores" in d.files
