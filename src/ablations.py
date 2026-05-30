"""Hyperparameter-sensitivity ablation sweep logic for KNN k and E6 window.

Core functions for computing sensitivity sweeps. The CLI entry point is
scripts/ablations.py; the functions here are importable for tests.

Sweeps:
  - KNN k in {5, 10, 20, 50, 100}: AUROC (ID vs alpha=1.0 OOD) + bootstrap CI
  - E6 window in {10, 20, 30, 50}: AUROC + fires-at-alpha + bootstrap CI

Uses the same data caches as build_metrics.py (report/teardown_collected.npz,
report/e4_collected.npz).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines import knn_distance, _real_calibration_by_corpus
from src.e6_detector import calibrate_threshold, rolling_spread
from src.metrics import auroc, bootstrap_ci

REPORT = Path("report")
N_BOOTSTRAP = 1000
SEED = 42

KNN_K_VALUES = [5, 10, 20, 50, 100]
E6_WINDOW_VALUES = [10, 20, 30, 50]


# --- data loading (mirrors build_metrics.py) --------------------------------

def load_id_hidden() -> np.ndarray:
    """Concatenate subaru + ram hidden_state from teardown_collected.npz."""
    by_corpus = _real_calibration_by_corpus()
    return np.concatenate(list(by_corpus.values()), axis=0)


def load_ood_hidden(alpha: float = 1.0) -> np.ndarray:
    """Load hidden_state for a specific alpha from the E4 cache."""
    d = np.load(REPORT / "e4_collected.npz")
    return d[f"{alpha:.4f}__hidden_state"]


def e4_alphas() -> np.ndarray:
    d = np.load(REPORT / "e4_collected.npz")
    return np.array(sorted({float(k.split("__")[0])
                            for k in d.files if "__hidden_state" in k}))


# --- KNN k ablation ---------------------------------------------------------

def sweep_knn_k(
    id_hidden: np.ndarray,
    ood_hidden: np.ndarray,
    k_values: list[int] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> list[dict]:
    """For each k, compute KNN AUROC (ID vs OOD) with bootstrap CI.

    Returns a list of dicts, one per k, each with keys:
      k, auroc_point, auroc_ci (mean, lo, hi)
    """
    if k_values is None:
        k_values = KNN_K_VALUES
    results = []
    for k in k_values:
        id_scores = knn_distance(id_hidden, id_hidden, k=k)
        ood_scores = knn_distance(id_hidden, ood_hidden, k=k)
        scores = np.concatenate([id_scores, ood_scores])
        labels = np.concatenate([
            np.zeros(len(id_scores), dtype=int),
            np.ones(len(ood_scores), dtype=int),
        ])
        auc_point = auroc(scores, labels)
        auc_ci = bootstrap_ci(auroc, scores, labels,
                              n_bootstrap=n_bootstrap, seed=seed)
        results.append({
            "k": k,
            "auroc_point": auc_point,
            "auroc_ci": auc_ci,
        })
    return results


# --- E6 window ablation -----------------------------------------------------

def fires_at_alpha(
    id_hidden: np.ndarray,
    e4_cache: Path,
    window: int,
    percentile: float = 1.0,
) -> float:
    """Smallest alpha where >50% of E4 frames are flagged OOD by E6.

    E6 convention: spread below threshold = OOD. Calibration threshold
    is the `percentile`-th percentile of the real-driving spread distribution.
    """
    real_spreads = rolling_spread(id_hidden, window=window)
    thr = calibrate_threshold(real_spreads, percentile)
    alphas = e4_alphas()
    d = np.load(e4_cache)
    for a in alphas:
        H = d[f"{float(a):.4f}__hidden_state"]
        s = rolling_spread(H, window=window)
        valid = s[~np.isnan(s)]
        frac = float(np.mean(valid < thr)) if len(valid) else 0.0
        if frac > 0.5:
            return float(a)
    return float("nan")


def sweep_e6_window(
    id_hidden: np.ndarray,
    ood_hidden: np.ndarray,
    window_values: list[int] | None = None,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
    _fires_at_alpha_fn=None,
) -> list[dict]:
    """For each window, compute E6 AUROC (ID vs OOD at alpha=1.0) with
    bootstrap CI, plus fires-at-alpha along the E4 sweep.

    _fires_at_alpha_fn: injectable for testing (avoids needing E4 caches).
    Defaults to the real fires_at_alpha function.

    Returns a list of dicts, one per window, each with keys:
      window, auroc_point, auroc_ci (mean, lo, hi), fires_at_alpha
    """
    if window_values is None:
        window_values = E6_WINDOW_VALUES
    if _fires_at_alpha_fn is None:
        _fires_at_alpha_fn = fires_at_alpha
    e4_cache = REPORT / "e4_collected.npz"
    results = []
    for w in window_values:
        # E6 native: lower spread = more OOD. Negate so higher = more OOD
        # (same convention as build_metrics.py _e6_scores_higher_more_ood).
        id_scores = -rolling_spread(id_hidden, window=w)
        ood_scores = -rolling_spread(ood_hidden, window=w)
        # Drop NaN frames (before the window fills).
        id_valid_mask = np.isfinite(id_scores)
        ood_valid_mask = np.isfinite(ood_scores)
        id_s = id_scores[id_valid_mask]
        ood_s = ood_scores[ood_valid_mask]
        scores = np.concatenate([id_s, ood_s])
        labels = np.concatenate([
            np.zeros(len(id_s), dtype=int),
            np.ones(len(ood_s), dtype=int),
        ])
        auc_point = auroc(scores, labels)
        auc_ci = bootstrap_ci(auroc, scores, labels,
                              n_bootstrap=n_bootstrap, seed=seed)
        fa = _fires_at_alpha_fn(id_hidden, e4_cache, window=w)
        results.append({
            "window": w,
            "auroc_point": auc_point,
            "auroc_ci": auc_ci,
            "fires_at_alpha": fa,
        })
    return results


# --- output -----------------------------------------------------------------

def fmt_ci(triple: tuple[float, float, float]) -> str:
    m, lo, hi = triple
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def write_report(
    knn_results: list[dict],
    e6_results: list[dict],
    out: Path,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> None:
    """Write report/ablations_results.md with the two sweep tables."""
    lines: list[str] = []
    lines.append("# Hyperparameter Sensitivity Ablations")
    lines.append("")
    lines.append(
        "Ablation sweeps for the two key hyperparameters flagged in the "
        "publication-readiness audit: KNN k (neighbour count) and E6 window "
        "(rolling-spread window size). Eval split: ID = subaru + ram real "
        "driving (concatenated), OOD = E4 alpha=1.0 CARLA frames. Bootstrap: "
        f"stratified by label, n={n_bootstrap}, seed={seed}."
    )
    lines.append("")

    # KNN k sweep
    lines.append("## KNN k-sensitivity")
    lines.append("")
    lines.append(
        "KNN OOD score = L2-normalised distance to the k-th nearest "
        "neighbour in the ID feature set (Sun et al., ICML 2022). Higher = "
        "more OOD. Default k=50 in build_metrics.py."
    )
    lines.append("")
    lines.append("| k | AUROC (mean [95% CI]) |")
    lines.append("|---|---|")
    for r in knn_results:
        lines.append(f"| {r['k']} | {fmt_ci(r['auroc_ci'])} |")
    lines.append("")

    # E6 window sweep
    lines.append("## E6 window-size sensitivity")
    lines.append("")
    lines.append(
        "E6 monitor = rolling trace(Var) of the 512-D recurrent feature over "
        "a sliding window. Lower spread = more OOD (negated for AUROC "
        "convention). fires-at-alpha = smallest alpha where >50% of frames "
        "flagged. Default window=30 in build_metrics.py."
    )
    lines.append("")
    lines.append("| window | AUROC (mean [95% CI]) | fires-at-alpha |")
    lines.append("|---|---|---|")
    for r in e6_results:
        fa = f"{r['fires_at_alpha']:.3f}" if np.isfinite(r['fires_at_alpha']) else "n/a"
        lines.append(
            f"| {r['window']} | {fmt_ci(r['auroc_ci'])} | {fa} |"
        )
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    knn_aurocs = [r["auroc_point"] for r in knn_results]
    knn_range = max(knn_aurocs) - min(knn_aurocs)
    e6_aurocs = [r["auroc_point"] for r in e6_results]
    e6_range = max(e6_aurocs) - min(e6_aurocs)
    lines.append(
        f"KNN AUROC range across k in {{{', '.join(str(r['k']) for r in knn_results)}}}: "
        f"{knn_range:.4f} (min {min(knn_aurocs):.3f}, max {max(knn_aurocs):.3f}). "
        f"E6 AUROC range across window in "
        f"{{{', '.join(str(r['window']) for r in e6_results)}}}: "
        f"{e6_range:.4f} (min {min(e6_aurocs):.3f}, max {max(e6_aurocs):.3f}). "
        f"Both detectors show {'low' if knn_range < 0.05 and e6_range < 0.05 else 'moderate'} "
        f"sensitivity to their primary hyperparameter at alpha=1.0."
    )
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def write_cache(
    knn_results: list[dict],
    e6_results: list[dict],
    out: Path,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = SEED,
) -> None:
    """Save results to an .npz for reproducibility / downstream use."""
    cache: dict[str, np.ndarray] = {}
    cache["knn_k_values"] = np.array([r["k"] for r in knn_results], dtype=np.int32)
    cache["knn_auroc_points"] = np.array(
        [r["auroc_point"] for r in knn_results], dtype=np.float32)
    cache["knn_auroc_cis"] = np.array(
        [list(r["auroc_ci"]) for r in knn_results], dtype=np.float32)

    cache["e6_window_values"] = np.array(
        [r["window"] for r in e6_results], dtype=np.int32)
    cache["e6_auroc_points"] = np.array(
        [r["auroc_point"] for r in e6_results], dtype=np.float32)
    cache["e6_auroc_cis"] = np.array(
        [list(r["auroc_ci"]) for r in e6_results], dtype=np.float32)
    cache["e6_fires_at_alpha"] = np.array(
        [r["fires_at_alpha"] for r in e6_results], dtype=np.float32)

    cache["meta__n_bootstrap"] = np.int32(n_bootstrap)
    cache["meta__seed"] = np.int32(seed)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cache)
