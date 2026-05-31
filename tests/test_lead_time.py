"""Tests for src/lead_time.py.

Covers: fires_at helper, sign convention (positive lead = before cliff),
cliff constant, and end-to-end table on real caches.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.lead_time import (
    CLIFF_ALPHA,
    FIRE_THRESHOLD,
    _fires_at,
    compute_all_lead_times,
    write_report,
)


# ---- constants -------------------------------------------------------------

def test_cliff_alpha_value():
    """Cliff onset must be the published value from report/e4_results.md."""
    assert CLIFF_ALPHA == pytest.approx(0.784)


def test_fire_threshold_value():
    assert FIRE_THRESHOLD == pytest.approx(0.50)


# ---- _fires_at helper ------------------------------------------------------

def test_fires_at_returns_first_crossing():
    alphas = np.array([0.0, 0.1, 0.3, 0.5, 0.7])
    fracs = np.array([0.0, 0.2, 0.6, 0.8, 0.9])
    assert _fires_at(alphas, fracs) == pytest.approx(0.3)


def test_fires_at_returns_nan_when_never_fires():
    alphas = np.array([0.0, 0.5, 1.0])
    fracs = np.array([0.1, 0.2, 0.4])
    result = _fires_at(alphas, fracs)
    assert np.isnan(result)


def test_fires_at_threshold_boundary():
    """Exactly at FIRE_THRESHOLD (0.50) should NOT fire; just above should."""
    alphas = np.array([0.2, 0.4, 0.6])
    fracs_at = np.array([0.50, 0.50, 0.50])  # exactly at threshold, not above
    assert np.isnan(_fires_at(alphas, fracs_at))

    fracs_above = np.array([0.49, 0.51, 0.60])
    assert _fires_at(alphas, fracs_above) == pytest.approx(0.4)


# ---- sign convention -------------------------------------------------------

def test_e6_lead_is_positive():
    """E6 fires at 0.550, cliff at 0.784 -> lead = +0.234 > 0."""
    rows = compute_all_lead_times() if _caches_available() else None
    if rows is None:
        pytest.skip("caches missing")
    e6_row = next(r for r in rows if "E6" in r["detector"])
    assert e6_row["lead_blend"] > 0.0, f"E6 lead should be positive, got {e6_row['lead_blend']}"
    assert e6_row["lead_blend"] == pytest.approx(0.234, abs=0.01)


def test_e6_fires_before_cliff():
    if not _caches_available():
        pytest.skip("caches missing")
    rows = compute_all_lead_times()
    e6_row = next(r for r in rows if "E6" in r["detector"])
    assert e6_row["fires_at"] < CLIFF_ALPHA


# ---- full table on real caches ---------------------------------------------

def test_compute_all_lead_times_shape():
    if not _caches_available():
        pytest.skip("caches missing")
    rows = compute_all_lead_times()
    # E6 + 3 baselines + conformal + pca_maha = 6 rows
    assert len(rows) == 6
    for r in rows:
        assert "detector" in r
        assert "single_auroc" in r
        assert "loco_mean_fpr" in r
        assert "fires_at" in r
        assert "lead_blend" in r


def test_e6_auroc_near_one():
    if not _caches_available():
        pytest.skip("caches missing")
    rows = compute_all_lead_times()
    e6_row = next(r for r in rows if "E6" in r["detector"])
    assert e6_row["single_auroc"] > 0.99


def test_knn50_and_conformal_same_fires_at():
    """KNN-50 and conformal both use KNN score; fires_at must be identical."""
    if not _caches_available():
        pytest.skip("caches missing")
    rows = compute_all_lead_times()
    knn_row = next(r for r in rows if r["detector"] == "knn50")
    conf_row = next(r for r in rows if r["detector"] == "conformal")
    assert knn_row["fires_at"] == pytest.approx(conf_row["fires_at"])


def test_write_report_creates_file(tmp_path):
    if not _caches_available():
        pytest.skip("caches missing")
    rows = compute_all_lead_times()
    out = tmp_path / "lead_time_results.md"
    write_report(rows, out)
    assert out.exists()
    content = out.read_text()
    assert "E6" in content
    assert "cliff" in content.lower()
    assert "AUROC" in content


# ---- helpers ---------------------------------------------------------------

def _caches_available() -> bool:
    return (
        Path("report/teardown_collected.npz").exists()
        and Path("report/e4_collected.npz").exists()
        and Path("report/metrics_collected.npz").exists()
    )
