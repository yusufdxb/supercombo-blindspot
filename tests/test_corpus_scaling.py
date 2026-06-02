"""Tests for src.corpus_scaling (E6 LOCO FPR + segment-level bootstrap)."""
from __future__ import annotations

import numpy as np
import pytest

from src import corpus_scaling as cs


def test_bootstrap_mean_ci_brackets_mean_and_is_deterministic():
    vals = [0.01, 0.02, 0.03, 0.07]
    lo, hi = cs.bootstrap_mean_ci(vals, b=5000, seed=1)
    assert lo <= np.mean(vals) <= hi
    lo2, hi2 = cs.bootstrap_mean_ci(vals, b=5000, seed=1)
    assert (lo, hi) == (lo2, hi2)  # seeded -> reproducible


def test_bootstrap_ci_collapses_for_identical_values():
    lo, hi = cs.bootstrap_mean_ci([0.05, 0.05, 0.05], b=1000, seed=0)
    assert lo == pytest.approx(0.05) and hi == pytest.approx(0.05)


def test_corpus_fpr_all_below_threshold_is_one():
    # constant hidden -> zero rolling spread -> below any positive threshold
    hidden = np.ones((60, 8), dtype=np.float32)
    assert cs.corpus_fpr(hidden, threshold=1.0, window=30) == pytest.approx(1.0)


def test_corpus_fpr_high_variance_not_flagged():
    rng = np.random.default_rng(0)
    hidden = rng.standard_normal((100, 8)).astype(np.float32) * 10.0
    # spread is large; a tiny threshold flags nothing
    assert cs.corpus_fpr(hidden, threshold=1e-6, window=30) == pytest.approx(0.0)


def test_load_real_corpora_splits_daytime_as_near_collapse():
    clean, near = cs.load_real_corpora()
    # daytime_control must never be in the clean calibration set
    assert "daytime_control" not in clean
    if "daytime_control" in near or near:
        assert "daytime_control" in near
    # the original N=2 corpora are clean-eligible when present
    for n in ("subaru", "ram"):
        assert n in clean


def test_run_uses_at_least_four_clean_corpora_and_reports_ci():
    res = cs.run()
    assert len(res["clean"]) >= 4
    lo, hi = res["ci"]
    assert lo <= res["loco"]["fpr_mean"] <= hi
    # before/after delta is present (original N=2)
    assert res["orig"] is not None
