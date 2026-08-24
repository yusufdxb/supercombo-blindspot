"""PCA-Mahalanobis ablation baseline.

The vanilla Mahalanobis baseline in src/baselines.py LOCO-fails at 100%
held-out FPR. The most predictable reviewer objection is: "you fit a full
512-D Gaussian on ~300 samples per corpus; of course it failed. Did you
try PCA first?" This module gives Mahalanobis its fairest shot: project
to the top-K principal components of the ID feature set (K chosen as the
smaller of "components needed for 90% variance" and 50), then fit a
Gaussian and compute Mahalanobis in the reduced space.

Higher score = more OOD, label 1 = OOD (same convention as the rest of
the OOD pipeline).

Reference: Mahalanobis++ (arXiv:2505.18032, May 2025) makes the same
shrinkage + normalisation argument; PCA pre-projection is the older and
cruder version of the same idea.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA


def fit_pca(features_id: np.ndarray, var_target: float = 0.90,
            max_components: int = 50) -> PCA:
    """Fit PCA on ID features. Component count = min(comps for `var_target`
    cumulative variance, `max_components`, n_samples - 1, n_features)."""
    n_samples, n_features = features_id.shape
    cap = min(max_components, n_samples - 1, n_features)
    cap = max(cap, 1)
    pca_full = PCA(n_components=cap, random_state=0).fit(features_id)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    k_var = int(np.searchsorted(cumvar, var_target) + 1)
    k = min(k_var, cap)
    k = max(k, 1)
    return PCA(n_components=k, random_state=0).fit(features_id)


def _fit_gaussian(features_id: np.ndarray, reg: float = 0.1
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Trace-shrinkage Gaussian fit, same convention as src/baselines.py."""
    mu = features_id.mean(axis=0)
    X = features_id - mu
    cov = (X.T @ X) / max(len(X) - 1, 1)
    trace_mean = float(np.trace(cov) / cov.shape[0]) if cov.shape[0] else 1.0
    cov = cov + (reg * trace_mean + 1e-6) * np.eye(cov.shape[0], dtype=cov.dtype)
    prec = np.linalg.inv(cov)
    return mu, prec


def pca_mahalanobis(features_id: np.ndarray, features_test: np.ndarray,
                    var_target: float = 0.90, max_components: int = 50,
                    reg: float = 0.1) -> np.ndarray:
    """Per-frame squared Mahalanobis distance in a PCA-reduced ID space.

    Fits the PCA basis and the Gaussian moments on `features_id`, projects
    both ID and test through the same basis, and returns squared
    Mahalanobis. Returns shape (len(features_test),). Higher = more OOD.
    """
    pca = fit_pca(features_id, var_target=var_target,
                  max_components=max_components)
    Z_id = pca.transform(features_id)
    Z_te = pca.transform(features_test)
    mu, prec = _fit_gaussian(Z_id, reg=reg)
    X = Z_te - mu
    return np.einsum("ij,jk,ik->i", X, prec, X).astype(np.float64)


# --- LOCO calibration mirroring src/baselines.py protocol ---------------

def _calibrate_threshold_high(scores_id: np.ndarray, percentile: float = 99.0
                              ) -> float:
    s = scores_id[~np.isnan(scores_id)]
    return float(np.percentile(s, percentile))


def loco_fpr(real_hidden_by_corpus: dict[str, np.ndarray],
             percentile: float = 99.0,
             var_target: float = 0.90,
             max_components: int = 50) -> dict:
    """LOCO held-out FPR for PCA-Mahalanobis. Same protocol as
    src/baselines.loco_fpr: fit PCA + Gaussian on N-1 corpora, threshold at
    the `percentile`-th of the calibration's self-scores, score the
    held-out corpus, count fraction above threshold."""
    folds = {}
    for held_out in real_hidden_by_corpus:
        calib_keys = [k for k in real_hidden_by_corpus if k != held_out]
        calib = np.concatenate([real_hidden_by_corpus[k] for k in calib_keys],
                                axis=0)
        s_calib = pca_mahalanobis(calib, calib, var_target=var_target,
                                  max_components=max_components)
        thr = _calibrate_threshold_high(s_calib, percentile)
        s_held = pca_mahalanobis(calib, real_hidden_by_corpus[held_out],
                                 var_target=var_target,
                                 max_components=max_components)
        valid = s_held[~np.isnan(s_held)]
        fpr = float(np.mean(valid > thr)) if len(valid) else float("nan")
        folds[held_out] = {"threshold": thr, "fpr": fpr,
                            "calibrated_on": calib_keys}
    fprs = np.array([f["fpr"] for f in folds.values()])
    return {"folds": folds, "fpr_mean": float(fprs.mean()),
            "fpr_max": float(fprs.max())}


def _real_calibration_by_corpus() -> dict[str, np.ndarray]:
    d = np.load("report/teardown_collected.npz")
    return {"subaru": d["subaru__hidden_state"],
            "ram": d["ram__hidden_state"]}


def _e4_alphas(cache: Path) -> np.ndarray:
    d = np.load(cache)
    return np.array(sorted({float(k.split("__")[0])
                            for k in d.files if "__hidden_state" in k}))


def _e4_hidden_at(cache: Path, alpha: float) -> np.ndarray:
    d = np.load(cache)
    return d[f"{alpha:.4f}__hidden_state"]
