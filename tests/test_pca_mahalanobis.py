"""Unit tests for src/pca_mahalanobis.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.pca_mahalanobis import fit_pca, loco_fpr, pca_mahalanobis


def test_pca_components_capped_by_max():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(200, 512))
    pca = fit_pca(X, var_target=0.999, max_components=50)
    assert pca.n_components_ <= 50
    assert pca.n_components_ >= 1


def test_pca_components_capped_by_samples():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 512))
    pca = fit_pca(X, var_target=0.999, max_components=200)
    # cap = min(200, 20-1, 512) = 19
    assert pca.n_components_ <= 19


def test_pca_mahalanobis_shape_and_id_low():
    rng = np.random.RandomState(1)
    X_id = rng.normal(size=(300, 512))
    X_ood = rng.normal(loc=5.0, size=(100, 512))
    s_id = pca_mahalanobis(X_id, X_id)
    s_ood = pca_mahalanobis(X_id, X_ood)
    assert s_id.shape == (300,)
    assert s_ood.shape == (100,)
    # OOD distribution should sit higher than ID.
    assert float(np.median(s_ood)) > float(np.median(s_id))


def test_pca_mahalanobis_separates_clear_shift():
    rng = np.random.RandomState(2)
    X_id = rng.normal(size=(500, 64))
    X_ood = rng.normal(loc=3.0, size=(200, 64))
    from src.metrics import auroc
    scores = np.concatenate([pca_mahalanobis(X_id, X_id),
                             pca_mahalanobis(X_id, X_ood)])
    labels = np.concatenate([np.zeros(500, dtype=int),
                             np.ones(200, dtype=int)])
    assert auroc(scores, labels) > 0.9


def test_loco_fpr_runs_on_real_corpora():
    # Smoke test: run LOCO on the actual subaru/ram cache. This test
    # documents the empirical result; we assert the result is a number,
    # not a specific value.
    d = np.load("report/teardown_collected.npz")
    by_corpus = {"subaru": d["subaru__hidden_state"],
                 "ram": d["ram__hidden_state"]}
    res = loco_fpr(by_corpus, percentile=99.0)
    assert set(res["folds"].keys()) == {"subaru", "ram"}
    assert 0.0 <= res["fpr_mean"] <= 1.0
    assert 0.0 <= res["fpr_max"] <= 1.0
