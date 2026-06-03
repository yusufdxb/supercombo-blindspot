"""Tests for E8b PCA-reduced per-vehicle Mahalanobis (src/e8_pca_hybrid.py).

Fast tests use synthetic features and small seed sets. Cache-backed tests skip
if the report/*.npz caches are absent. The figure/headline smoke is marked slow
(it touches the raw 512-D inverse). Run slow with: pytest -m slow.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

from src.e8_pca_hybrid import (
    K_STAR,
    k_sweep,
    maha_at_k,
    maha_auroc,
    within_vehicle_fpr,
)

_CACHES = all(
    Path(p).exists()
    for p in [
        "report/teardown_collected.npz",
        "report/e4_collected.npz",
        "report/e7_collected.npz",
    ]
)
_need_caches = pytest.mark.skipif(not _CACHES, reason="report caches not present")


@pytest.fixture
def rng():
    return np.random.RandomState(0)


class TestMahaAtK:
    def test_shape_and_finite(self, rng):
        fit = rng.randn(150, 64).astype(np.float32)
        sc = rng.randn(40, 64).astype(np.float32)
        out = maha_at_k(fit, sc, k=16)
        assert out.shape == (40,)
        assert np.all(np.isfinite(out))

    def test_reduced_k_is_lower_dim_but_valid(self, rng):
        fit = rng.randn(120, 64).astype(np.float32)
        sc = rng.randn(30, 64).astype(np.float32)
        # k larger than n_features falls back to raw; still finite
        assert np.all(np.isfinite(maha_at_k(fit, sc, k=64)))
        assert np.all(np.isfinite(maha_at_k(fit, sc, k=8)))

    def test_shifted_features_score_higher(self, rng):
        fit = rng.randn(200, 48).astype(np.float32)
        near = rng.randn(50, 48).astype(np.float32)
        far = (rng.randn(50, 48) + 8.0).astype(np.float32)
        assert maha_at_k(fit, far, k=16).mean() > maha_at_k(fit, near, k=16).mean()


class TestConstants:
    def test_k_star_is_32(self):
        assert K_STAR == 32


@_need_caches
class TestWithinVehicleFpr:
    def test_pca_k32_controls_fpr(self):
        """The headline fix: per-vehicle PCA k=32 gives a controlled FPR
        (well under the raw-512D 30-45% failure), near the 1% calibration target."""
        td = np.load("report/teardown_collected.npz")
        for veh in ["subaru", "ram"]:
            mean, _ = within_vehicle_fpr(td[f"{veh}__hidden_state"], k=32, seeds=range(5))
            assert mean < 0.10, f"{veh} PCA k=32 FPR should be controlled, got {mean:.3f}"

    def test_raw512_fpr_is_uncontrolled(self):
        """Sanity that the problem is real: raw 512-D FPR is much higher than PCA."""
        td = np.load("report/teardown_collected.npz")
        raw, _ = within_vehicle_fpr(td["subaru__hidden_state"], k=512, seeds=range(3))
        pca, _ = within_vehicle_fpr(td["subaru__hidden_state"], k=32, seeds=range(3))
        assert raw > pca + 0.10, f"raw {raw:.3f} should be much worse than pca {pca:.3f}"


@_need_caches
class TestCoverage:
    def test_corruption_auroc_high_at_k32(self):
        td = np.load("report/teardown_collected.npz")
        e7 = np.load("report/e7_collected.npz")
        auc = maha_auroc(td["subaru__hidden_state"], e7["fog__5__hidden_state"], k=32)
        assert auc > 0.95, f"PCA-Maha k=32 corruption AUROC should be high, got {auc:.3f}"

    def test_corruption_auroc_recovers_with_k(self):
        """The OOD signal needs ~32 components: AUROC at k=32 > at k=8."""
        td = np.load("report/teardown_collected.npz")
        e7 = np.load("report/e7_collected.npz")
        lo = maha_auroc(td["subaru__hidden_state"], e7["fog__5__hidden_state"], k=8)
        hi = maha_auroc(td["subaru__hidden_state"], e7["fog__5__hidden_state"], k=32)
        assert hi > lo, f"corruption AUROC should recover with k: k8={lo:.3f} k32={hi:.3f}"


@_need_caches
class TestKSweep:
    def test_fpr_rises_with_k(self):
        """FPR climbs as k grows past the well-conditioned regime."""
        rows = k_sweep(ks=(16, 128), seeds=range(3))
        by_k = {r["k"]: r["fpr_mean"] for r in rows}
        assert by_k[128] > by_k[16], f"FPR should rise with k: {by_k}"

    def test_rows_have_expected_keys(self):
        rows = k_sweep(ks=(32,), seeds=range(2))
        assert rows[0]["k"] == 32
        for key in ["fpr_mean", "corruption_auroc", "collapse_auroc"]:
            assert key in rows[0]


@_need_caches
@pytest.mark.slow
class TestFigureSmoke:
    def test_make_figure_writes(self, tmp_path):
        from src.e8_pca_hybrid import make_figure
        out = make_figure(out=tmp_path / "e8_demo.png")
        assert out.exists() and out.stat().st_size > 0
