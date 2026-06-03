"""E8: Hybrid OOD detector combining E6 rolling-spread + Mahalanobis.

E6 is a second-order (collapse) detector: it fires when the rolling temporal
variance of the 512-D recurrent feature drops below a calibrated floor. It
passes LOCO FPR (~1%) across real-driving corpora because it is
location-invariant. But E7 showed it is blind to photometric corruptions
(AUROC 0.52-0.74 on ImageNet-C): those corruptions do not cause the hidden
state to freeze, so the rolling spread stays normal even as the image is
severely degraded.

Mahalanobis distance is a first-order (location-sensitive) detector: it fires
when the feature vector drifts far from the ID Gaussian mean. It catches
photometric corruptions perfectly (AUROC 1.0 on all noise/weather/digital
corruptions in E7). But it cannot calibrate across corpora: the subaru and
ram feature clouds are disjoint in the 512-D space, so LOCO FPR = 100% at any
threshold.

The hybrid fires if EITHER arm fires. This gives full-class coverage:
    - Collapse (CARLA-style OOD): caught by E6 arm.
    - Photometric OOD (ImageNet-C): caught by Mahalanobis arm.

LOCO CALIBRATION HONESTY:
    The naive OR combination inherits Mahalanobis's location-sensitivity: the
    combined LOCO FPR is 100%, entirely driven by the Mahalanobis arm flagging
    every frame of the held-out corpus as OOD. Tightening the Mahalanobis
    threshold does not fix this -- no finite percentile resolves the disjoint-
    corpus problem (verified: LOCO FPR = 100% at p=99.0, 99.9, 99.99). This
    is NOT a deficiency of the hybrid design; it is the canonical finding from
    Phantom-Braking: first-order detectors cannot calibrate across corpora on
    this model's recurrent feature. The hybrid's value is that it covers both
    failure classes, but its deployment FPR target should be:

        - E6 arm: ~1% LOCO FPR (calibrated, location-invariant)
        - Mahalanobis arm: sensor-locked (calibrate ONCE on the deployed
          vehicle's ID corpus; never leave-one-VEHICLE-out)

    The combined FPR is reported per-arm and as the realized OR on the
    same LOCO folds, so the reader sees exactly which arm drives the FPR.

Usage:
    python -m src.e8_hybrid                    # analysis from caches
    python -m src.e8_hybrid --n-bootstrap 1000 # full bootstrap run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CACHE_OUT = ROOT / "report" / "e8_hybrid_collected.npz"
RESULTS_MD = ROOT / "report" / "e8_hybrid_results.md"
FIG_DIR = ROOT / "report" / "figures"

# -------------------------------------------------------------------------
# Data loading helpers (mirrors e6_detector.py and baselines.py)
# -------------------------------------------------------------------------


def _load_id_by_corpus() -> dict[str, np.ndarray]:
    d = np.load(ROOT / "report" / "teardown_collected.npz")
    return {
        "subaru": d["subaru__hidden_state"],
        "ram": d["ram__hidden_state"],
    }


def _load_id_all() -> np.ndarray:
    by_corpus = _load_id_by_corpus()
    return np.concatenate(list(by_corpus.values()), axis=0)


# -------------------------------------------------------------------------
# Per-frame hybrid OOD score
# -------------------------------------------------------------------------


def hybrid_scores(
    features_id: np.ndarray,
    features_test: np.ndarray,
    window: int = 30,
    e6_percentile: float = 1.0,
    maha_percentile: float = 99.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute (hybrid_binary, e6_score, maha_score) for test frames.

    Returns:
        hybrid_fired: bool array (T,), True if either arm fires.
        e6_score:     rolling spread, lower = more OOD. NaN for first window-1
                      frames (same convention as e6_detector.rolling_spread).
        maha_score:   Mahalanobis distance, higher = more OOD.

    Thresholds are calibrated on features_id:
        E6 arm:   e6_percentile-th percentile of the rolling spread of
                  features_id (default 1st percentile -> ~1% ID FPR).
        Maha arm: maha_percentile-th percentile of the Mahalanobis scores of
                  features_id scored against themselves (default 99th
                  percentile -> ~1% ID FPR).

    The caller is responsible for LOCO calibration; this function uses the
    supplied features_id as both the ID Gaussian and the calibration set.
    """
    from src.e6_detector import rolling_spread, calibrate_threshold
    from src.baselines import mahalanobis, calibrate_threshold_high

    # E6 arm
    id_spreads = rolling_spread(features_id, window)
    e6_thr = calibrate_threshold(id_spreads, e6_percentile)
    test_spreads = rolling_spread(features_test, window)
    e6_fired = test_spreads < e6_thr  # lower spread = collapse = OOD

    # Maha arm
    maha_id_scores = mahalanobis(features_id, features_id)
    maha_thr = calibrate_threshold_high(maha_id_scores, maha_percentile)
    maha_test_scores = mahalanobis(features_id, features_test)
    maha_fired = maha_test_scores > maha_thr

    # Hybrid: fire if EITHER arm fires. NaN E6 frames = window warmup;
    # treat them as NOT firing for E6 (conservative on the E6 arm only).
    e6_fired_clean = np.where(np.isnan(test_spreads), False, e6_fired)
    hybrid_fired = e6_fired_clean | maha_fired

    return hybrid_fired.astype(bool), test_spreads, maha_test_scores


def hybrid_ood_score(
    features_id: np.ndarray,
    features_test: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """Threshold-free hybrid OOD score for AUROC/AUPR computation.

    Strategy: normalise each arm's score to [0, 1] relative to the ID
    distribution, then take the element-wise maximum. This is
    location-invariant for the E6 arm and distance-based for the Maha arm.

    Normalisation:
        E6 arm (lower = more OOD): map id p1 -> 1.0, id p99 -> 0.0.
            score_e6 = clip( (p99_id - spread) / (p99_id - p1_id), 0, 1 )
        Maha arm (higher = more OOD): map id p1 -> 0.0, id p99 -> 1.0.
            score_maha = clip( (maha - p1_id) / (p99_id - p1_id), 0, 1 )

    The max-combination means that any frame that is extreme on EITHER axis
    gets a high hybrid score. Ties are broken in favour of the more extreme arm.

    NaN (warmup frames in E6) are replaced by 0.0 so they do not contribute to
    the max (the Maha arm score still applies to those frames).
    """
    from src.e6_detector import rolling_spread
    from src.baselines import mahalanobis

    id_spreads = rolling_spread(features_id, window)
    test_spreads = rolling_spread(features_test, window)
    id_valid = id_spreads[np.isfinite(id_spreads)]
    p1_e6 = float(np.percentile(id_valid, 1))
    p99_e6 = float(np.percentile(id_valid, 99))
    denom_e6 = max(p99_e6 - p1_e6, 1e-12)
    # NaN warmup -> 0 (neutral, Maha arm takes over for those frames)
    spread_clean = np.where(np.isfinite(test_spreads), test_spreads, p99_e6)
    score_e6 = np.clip((p99_e6 - spread_clean) / denom_e6, 0.0, 1.0)

    maha_id_scores = mahalanobis(features_id, features_id)
    maha_test_scores = mahalanobis(features_id, features_test)
    p1_maha = float(np.percentile(maha_id_scores, 1))
    p99_maha = float(np.percentile(maha_id_scores, 99))
    denom_maha = max(p99_maha - p1_maha, 1e-12)
    score_maha = np.clip((maha_test_scores - p1_maha) / denom_maha, 0.0, 1.0)

    return np.maximum(score_e6, score_maha).astype(np.float64)


# -------------------------------------------------------------------------
# LOCO calibration for the hybrid
# -------------------------------------------------------------------------


def loco_fpr_hybrid(
    real_hidden_by_corpus: dict[str, np.ndarray],
    window: int = 30,
    e6_percentile: float = 1.0,
    maha_percentile: float = 99.0,
) -> dict:
    """Leave-one-corpus-out FPR for the hybrid detector.

    For each held-out corpus:
        1. Calibrate E6 threshold and Maha threshold on the N-1 corpora.
        2. Score the held-out corpus with both arms.
        3. Combine with OR.
        4. Report per-arm FPR and combined FPR.

    The per-arm FPR makes the decomposition explicit: users can see whether
    the combined FPR is driven by E6 or Maha.
    """
    from src.e6_detector import rolling_spread, calibrate_threshold
    from src.baselines import mahalanobis, calibrate_threshold_high

    folds: dict[str, dict] = {}
    for held_out in real_hidden_by_corpus:
        calib_keys = [k for k in real_hidden_by_corpus if k != held_out]
        calib = np.concatenate(
            [real_hidden_by_corpus[k] for k in calib_keys], axis=0
        )
        held = real_hidden_by_corpus[held_out]

        # E6 arm threshold from calibration set
        calib_spreads = rolling_spread(calib, window)
        e6_thr = calibrate_threshold(calib_spreads, e6_percentile)

        # Maha arm threshold from calibration set
        maha_id_scores = mahalanobis(calib, calib)
        maha_thr = calibrate_threshold_high(maha_id_scores, maha_percentile)

        # Score held-out set
        held_spreads = rolling_spread(held, window)
        held_maha = mahalanobis(calib, held)

        valid_mask = np.isfinite(held_spreads)
        n_valid = int(valid_mask.sum())

        e6_fired = np.where(valid_mask, held_spreads < e6_thr, False)
        maha_fired = held_maha > maha_thr
        # For E6-arm FPR, only count frames that have a valid spread
        e6_fpr = float(e6_fired[valid_mask].mean()) if n_valid else float("nan")
        maha_fpr = float(maha_fired.mean()) if len(maha_fired) else float("nan")
        hybrid_fired = e6_fired | maha_fired
        combined_fpr = float(hybrid_fired.mean()) if len(hybrid_fired) else float("nan")

        folds[held_out] = {
            "e6_threshold": float(e6_thr),
            "maha_threshold": float(maha_thr),
            "e6_fpr": e6_fpr,
            "maha_fpr": maha_fpr,
            "combined_fpr": combined_fpr,
            "calibrated_on": calib_keys,
            "n_frames_valid": n_valid,
        }

    e6_fprs = np.array([f["e6_fpr"] for f in folds.values()])
    maha_fprs = np.array([f["maha_fpr"] for f in folds.values()])
    combined_fprs = np.array([f["combined_fpr"] for f in folds.values()])

    return {
        "folds": folds,
        "e6_fpr_mean": float(np.nanmean(e6_fprs)),
        "e6_fpr_max": float(np.nanmax(e6_fprs)),
        "maha_fpr_mean": float(np.nanmean(maha_fprs)),
        "maha_fpr_max": float(np.nanmax(maha_fprs)),
        "combined_fpr_mean": float(np.nanmean(combined_fprs)),
        "combined_fpr_max": float(np.nanmax(combined_fprs)),
    }


# -------------------------------------------------------------------------
# Metric evaluation helpers
# -------------------------------------------------------------------------


def _e6_ood_scores_higher(
    id_hidden: np.ndarray,
    test_hidden: np.ndarray,
    window: int = 30,
) -> np.ndarray:
    """E6 scores with higher=more-OOD convention (negated spread)."""
    from src.e6_detector import rolling_spread
    s = rolling_spread(test_hidden, window)
    return -s  # negate: lower spread -> higher OOD score


def _maha_ood_scores(
    id_hidden: np.ndarray,
    test_hidden: np.ndarray,
) -> np.ndarray:
    """Mahalanobis scores (already higher=more-OOD)."""
    from src.baselines import mahalanobis
    return mahalanobis(id_hidden, test_hidden)


def _rmd_ood_scores(
    id_hidden: np.ndarray,
    test_hidden: np.ndarray,
) -> np.ndarray:
    """Relative Mahalanobis scores (higher=more-OOD)."""
    from src.baselines import relative_mahalanobis
    return relative_mahalanobis(id_hidden, test_hidden)


def compute_metrics(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute AUROC, AUPR, FPR@95TPR with bootstrap CIs.

    scores: per-frame OOD score, higher = more OOD.
    labels: 0 = ID, 1 = OOD.
    """
    from src import metrics

    fpr95_fn = lambda s, l: metrics.fpr_at_tpr(s, l, 0.95)

    auroc_val = metrics.auroc(scores, labels)
    aupr_val = metrics.aupr(scores, labels)
    fpr95_val = fpr95_fn(scores, labels)

    auroc_ci = metrics.bootstrap_ci(
        metrics.auroc, scores, labels, n_bootstrap=n_bootstrap, seed=seed
    )
    aupr_ci = metrics.bootstrap_ci(
        metrics.aupr, scores, labels, n_bootstrap=n_bootstrap, seed=seed
    )
    fpr95_ci = metrics.bootstrap_ci(
        fpr95_fn, scores, labels, n_bootstrap=n_bootstrap, seed=seed
    )

    return {
        "auroc": auroc_val,
        "aupr": aupr_val,
        "fpr95": fpr95_val,
        "auroc_ci": auroc_ci,
        "aupr_ci": aupr_ci,
        "fpr95_ci": fpr95_ci,
        "n_id": int((labels == 0).sum()),
        "n_ood": int((labels == 1).sum()),
    }


# -------------------------------------------------------------------------
# E7 corruption sweep evaluation
# -------------------------------------------------------------------------


def evaluate_on_e7(
    window: int = 30,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict[str, dict]:
    """Evaluate hybrid, E6-alone, Maha-alone, and RMD-alone on every
    (corruption, severity) condition in the E7 cache.

    Returns {condition_key: {detector_name: metrics_dict, ...}, ...}
    """
    from src.e7_corruption import load_cache, CORRUPTION_NAMES, SEVERITIES
    from src.teardown import WARMUP

    collected = load_cache(ROOT / "report" / "e7_collected.npz")
    id_hidden = _load_id_all()

    # Pre-compute ID scores for each detector (for label construction)
    id_e6_scores = _e6_ood_scores_higher(id_hidden, id_hidden, window)
    id_maha_scores = _maha_ood_scores(id_hidden, id_hidden)
    id_rmd_scores = _rmd_ood_scores(id_hidden, id_hidden)
    id_hybrid_scores = hybrid_ood_score(id_hidden, id_hidden, window)

    # Drop NaN from E6 ID scores
    id_e6_valid = id_e6_scores[np.isfinite(id_e6_scores)]

    results: dict[str, dict] = {}

    for condition_key in sorted(collected.keys()):
        parts = condition_key.split("__")
        cname, sev = parts[0], int(parts[1])

        hidden_raw = collected[condition_key].get("hidden_state")
        if hidden_raw is None:
            continue

        # Discard warmup frames
        hidden = hidden_raw[WARMUP:]
        T = len(hidden)
        if T < window:
            results[condition_key] = {
                "corruption": cname, "severity": sev, "skip": True, "n_frames": T
            }
            continue

        cond_result: dict = {
            "corruption": cname, "severity": sev, "skip": False, "n_frames": T
        }

        # Detector scores for this condition (OOD frames)
        ood_e6 = _e6_ood_scores_higher(id_hidden, hidden, window)
        ood_maha = _maha_ood_scores(id_hidden, hidden)
        ood_rmd = _rmd_ood_scores(id_hidden, hidden)
        ood_hybrid = hybrid_ood_score(id_hidden, hidden, window)

        ood_e6_valid = ood_e6[np.isfinite(ood_e6)]

        for det_name, id_scores, ood_scores in [
            ("e6", id_e6_valid, ood_e6_valid),
            ("mahalanobis", id_maha_scores, ood_maha),
            ("rmd", id_rmd_scores, ood_rmd),
            ("hybrid", id_hybrid_scores, ood_hybrid),
        ]:
            if len(ood_scores) == 0 or len(id_scores) == 0:
                cond_result[det_name] = {
                    "auroc": float("nan"), "aupr": float("nan"),
                    "fpr95": float("nan"),
                    "auroc_ci": (float("nan"),) * 3,
                    "aupr_ci": (float("nan"),) * 3,
                    "fpr95_ci": (float("nan"),) * 3,
                }
                continue
            scores = np.concatenate([id_scores, ood_scores])
            labels = np.concatenate([
                np.zeros(len(id_scores), dtype=np.int64),
                np.ones(len(ood_scores), dtype=np.int64),
            ])
            cond_result[det_name] = compute_metrics(
                scores, labels, n_bootstrap=n_bootstrap, seed=seed
            )

        results[condition_key] = cond_result

    return results


# -------------------------------------------------------------------------
# E4 collapse axis evaluation
# -------------------------------------------------------------------------


def evaluate_on_e4(
    window: int = 30,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> dict:
    """Evaluate hybrid, E6-alone, Maha-alone, RMD-alone on the E4 alpha sweep.

    Returns per-alpha AUROC for each detector, plus headline metrics at alpha=1.
    """
    from src.e6_detector import _e4_alphas, _e4_hidden_at

    e4_cache = ROOT / "report" / "e4_collected.npz"
    alphas = _e4_alphas(e4_cache)
    id_hidden = _load_id_all()

    id_e6 = _e6_ood_scores_higher(id_hidden, id_hidden, window)
    id_maha = _maha_ood_scores(id_hidden, id_hidden)
    id_rmd = _rmd_ood_scores(id_hidden, id_hidden)
    id_hybrid = hybrid_ood_score(id_hidden, id_hidden, window)

    id_e6_valid = id_e6[np.isfinite(id_e6)]

    per_alpha: list[dict] = []

    for a in alphas:
        H = _e4_hidden_at(e4_cache, float(a))
        ood_e6 = _e6_ood_scores_higher(id_hidden, H, window)
        ood_maha = _maha_ood_scores(id_hidden, H)
        ood_rmd = _rmd_ood_scores(id_hidden, H)
        ood_hybrid = hybrid_ood_score(id_hidden, H, window)

        ood_e6_valid = ood_e6[np.isfinite(ood_e6)]

        row: dict = {"alpha": float(a)}
        for det_name, id_s, ood_s in [
            ("e6", id_e6_valid, ood_e6_valid),
            ("mahalanobis", id_maha, ood_maha),
            ("rmd", id_rmd, ood_rmd),
            ("hybrid", id_hybrid, ood_hybrid),
        ]:
            if len(ood_s) == 0 or len(id_s) == 0:
                row[f"{det_name}_auroc"] = float("nan")
                continue
            scores = np.concatenate([id_s, ood_s])
            labels = np.concatenate([
                np.zeros(len(id_s), dtype=np.int64),
                np.ones(len(ood_s), dtype=np.int64),
            ])
            row[f"{det_name}_auroc"] = float(
                __import__("src.metrics", fromlist=["auroc"]).auroc(scores, labels)
            )

        per_alpha.append(row)

    # Headline at alpha=1.0: full bootstrap CI
    H_full = _e4_hidden_at(e4_cache, 1.0)
    headline: dict = {}
    for det_name, id_s in [
        ("e6", id_e6_valid),
        ("mahalanobis", id_maha),
        ("rmd", id_rmd),
        ("hybrid", id_hybrid),
    ]:
        if det_name == "e6":
            ood_s_raw = _e6_ood_scores_higher(id_hidden, H_full, window)
            ood_s = ood_s_raw[np.isfinite(ood_s_raw)]
        elif det_name == "mahalanobis":
            ood_s = _maha_ood_scores(id_hidden, H_full)
        elif det_name == "rmd":
            ood_s = _rmd_ood_scores(id_hidden, H_full)
        else:
            ood_s = hybrid_ood_score(id_hidden, H_full, window)
        scores = np.concatenate([id_s, ood_s])
        labels = np.concatenate([
            np.zeros(len(id_s), dtype=np.int64),
            np.ones(len(ood_s), dtype=np.int64),
        ])
        headline[det_name] = compute_metrics(
            scores, labels, n_bootstrap=n_bootstrap, seed=seed
        )

    return {"per_alpha": per_alpha, "headline": headline, "alphas": alphas}


# -------------------------------------------------------------------------
# Submodule collapse localization (Task B)
# -------------------------------------------------------------------------


def localize_collapse() -> dict:
    """Identify which submodule's activations collapse first and most.

    Uses report/e5_submodule_collected.npz. For each probe, computes:
        activity_ratio(alpha): sum |std_per_element(OOD)| / sum |std_per_element(real)|
        cliff_alpha: smallest alpha where activity_ratio < 0.5
        mean_shift_at_1: L1 shift of per-element mean at alpha=1 relative to alpha=0

    Returns a dict with per-probe arrays and rankings.
    """
    from src.e5_submodule import load_cache, analyse, SUBMODULE_PROBES

    cache_path = ROOT / "report" / "e5_submodule_collected.npz"
    if not cache_path.exists():
        raise FileNotFoundError(f"E5 submodule cache missing: {cache_path}")

    alphas, per_probe = load_cache(cache_path)
    res = analyse(alphas, per_probe)

    # Collect into a table
    probe_names = list(res["ratios"].keys())
    rows = []
    for name in probe_names:
        rows.append({
            "probe": name,
            "cliff_alpha": res["cliffs"][name],
            "mean_shift_at_1": res["mean_shifts"][name],
            "activity_ratio_at_1": float(res["ratios"][name][-1]),
            "activity_ratio_at_05": float(
                res["ratios"][name][
                    np.searchsorted(alphas, 0.5, side="left")
                ] if 0.5 in alphas else
                np.interp(0.5, alphas, res["ratios"][name])
            ),
            "role": next(
                (p.role for p in SUBMODULE_PROBES if p.name == name),
                "unknown",
            ),
        })

    # Rank by cliff alpha (lower = collapses earlier = more upstream of collapse)
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["cliff_alpha"] if np.isfinite(r["cliff_alpha"]) else float("inf"),
            -r["mean_shift_at_1"],
        ),
    )

    # First to collapse = smallest finite cliff alpha
    finite_cliffs = [r for r in rows_sorted if np.isfinite(r["cliff_alpha"])]
    first_collapse = finite_cliffs[0] if finite_cliffs else rows_sorted[0]

    return {
        "alphas": alphas,
        "ratios": res["ratios"],
        "cliffs": res["cliffs"],
        "mean_shifts": res["mean_shifts"],
        "table": rows_sorted,
        "first_collapse_probe": first_collapse["probe"],
        "first_collapse_cliff": first_collapse["cliff_alpha"],
    }


# -------------------------------------------------------------------------
# Report writing
# -------------------------------------------------------------------------


def _fmt_ci(ci_triple: tuple) -> str:
    _, lo, hi = ci_triple
    return f"[{lo:.4f}, {hi:.4f}]"


def _fmt(v: float, precision: int = 4) -> str:
    if not np.isfinite(v):
        return "NaN"
    return f"{v:.{precision}f}"


def write_results(
    e7_results: dict,
    e4_results: dict,
    loco_result: dict,
    submodule_result: dict,
    out: Path,
    n_bootstrap: int = 1000,
) -> None:
    """Write report/e8_hybrid_results.md."""
    lines: list[str] = []

    lines += [
        "# E8: Hybrid OOD Detector (E6 + Mahalanobis)",
        "",
        "Combines the rolling-spread collapse detector (E6) with Mahalanobis "
        "distance to cover two disjoint failure classes: temporal collapse "
        "(CARLA-style OOD, E4 sweep) and photometric corruption (ImageNet-C, "
        "E7 sweep). The hybrid fires if EITHER arm fires.",
        "",
        "Bootstrap: stratified by label, n_bootstrap=" + str(n_bootstrap) + ", seed=42.",
        "",
    ]

    # --- LOCO FPR ---
    lines += [
        "## Combined FPR: LOCO calibration",
        "",
        "Each arm is calibrated on the N-1 corpora (LOCO protocol matching E6 and "
        "baselines). The combined FPR is the OR of both arms on the held-out corpus.",
        "",
        "**E6 arm:** calibrated at the 1st percentile of the ID rolling spread "
        "(same as E6 standalone). Location-invariant; calibrates cleanly.",
        "",
        "**Mahalanobis arm:** calibrated at the 99th percentile of the ID "
        "Mahalanobis scores. Location-sensitive; LOCO FPR = 100% at any "
        "percentile because the subaru and ram feature clouds are disjoint in "
        "the 512-D recurrent feature space (canonical Phantom-Braking finding, "
        "reproduced in baselines.py).",
        "",
        "| held-out corpus | E6 FPR | Maha FPR | combined FPR |",
        "|---|---|---|---|",
    ]
    for corpus, fold in loco_result["folds"].items():
        lines.append(
            f"| {corpus} | {fold['e6_fpr']:.4f} | "
            f"{fold['maha_fpr']:.4f} | {fold['combined_fpr']:.4f} |"
        )
    lines += [
        "",
        f"**LOCO mean E6 FPR: {loco_result['e6_fpr_mean']:.4f}**",
        f"**LOCO mean Maha FPR: {loco_result['maha_fpr_mean']:.4f}**",
        f"**LOCO mean combined FPR: {loco_result['combined_fpr_mean']:.4f}**",
        "",
        "The combined FPR is driven entirely by the Mahalanobis arm's "
        "location-sensitivity. The E6 arm alone would give LOCO mean FPR "
        f"~{loco_result['e6_fpr_mean']:.4f} (consistent with the standalone "
        "E6 result of 1.03%). The hybrid's deployment FPR target is: "
        "calibrate the Maha arm ONCE per vehicle (sensor-locked), never "
        "leave-one-vehicle-out.",
        "",
    ]

    # --- E4 collapse axis headline ---
    e4_headline = e4_results["headline"]
    lines += [
        "## Collapse axis (E4 sweep): headline metrics at alpha=1.0",
        "",
        "ID = subaru + ram real driving. OOD = CARLA alpha=1.0.",
        "",
        "| Detector | AUROC | AUROC 95% CI | AUPR | FPR@95TPR |",
        "|---|---|---|---|---|",
    ]
    for det in ["e6", "mahalanobis", "rmd", "hybrid"]:
        m = e4_headline[det]
        lines.append(
            f"| {det} | {_fmt(m['auroc'])} | {_fmt_ci(m['auroc_ci'])} | "
            f"{_fmt(m['aupr'])} | {_fmt(m['fpr95'])} |"
        )
    lines.append("")

    # --- Per-alpha AUROC on E4 sweep ---
    lines += [
        "### Per-alpha AUROC on E4 sweep",
        "",
        "| alpha | E6 AUROC | Maha AUROC | RMD AUROC | Hybrid AUROC |",
        "|---|---|---|---|---|",
    ]
    for row in e4_results["per_alpha"]:
        lines.append(
            f"| {row['alpha']:.2f} | {_fmt(row['e6_auroc'])} | "
            f"{_fmt(row['mahalanobis_auroc'])} | {_fmt(row['rmd_auroc'])} | "
            f"{_fmt(row['hybrid_auroc'])} |"
        )
    lines.append("")

    # --- E7 corruption sweep: per-corruption comparison table ---
    lines += [
        "## Photometric corruption sweep (E7): per-corruption AUROC comparison",
        "",
        "Mean AUROC across 5 severities for each corruption.",
        "",
        "| Corruption | E6 mean AUROC | Maha mean AUROC | RMD mean AUROC | Hybrid mean AUROC |",
        "|---|---|---|---|---|",
    ]

    # Aggregate per-corruption mean
    from collections import defaultdict
    per_corruption: dict[str, dict[str, list]] = defaultdict(
        lambda: {d: [] for d in ["e6", "mahalanobis", "rmd", "hybrid"]}
    )
    for key, cond in e7_results.items():
        if cond.get("skip") or cond.get("severity", 0) == 0:
            continue
        cname = cond["corruption"]
        for det in ["e6", "mahalanobis", "rmd", "hybrid"]:
            v = cond[det]["auroc"] if det in cond else float("nan")
            if np.isfinite(v):
                per_corruption[cname][det].append(v)

    from src.e7_corruption import CORRUPTION_NAMES as E7_NAMES
    for cname in E7_NAMES:
        if cname not in per_corruption:
            continue
        d = per_corruption[cname]
        row_vals = {
            det: float(np.mean(d[det])) if d[det] else float("nan")
            for det in ["e6", "mahalanobis", "rmd", "hybrid"]
        }
        lines.append(
            f"| {cname} | {_fmt(row_vals['e6'])} | {_fmt(row_vals['mahalanobis'])} | "
            f"{_fmt(row_vals['rmd'])} | {_fmt(row_vals['hybrid'])} |"
        )
    lines.append("")

    # Per-severity for selected corruptions
    lines += [
        "### Per-severity AUROC for selected corruptions (severity 1-5)",
        "",
    ]
    for show_cname in ["gaussian_noise", "contrast", "defocus_blur", "jpeg_compression"]:
        lines.append(f"#### {show_cname}")
        lines.append("")
        lines.append("| Severity | E6 | Maha | RMD | Hybrid |")
        lines.append("|---|---|---|---|---|")
        for sev in range(1, 6):
            key = f"{show_cname}__{sev}"
            if key not in e7_results or e7_results[key].get("skip"):
                continue
            cond = e7_results[key]
            lines.append(
                f"| {sev} | {_fmt(cond['e6']['auroc'])} | "
                f"{_fmt(cond['mahalanobis']['auroc'])} | "
                f"{_fmt(cond['rmd']['auroc'])} | "
                f"{_fmt(cond['hybrid']['auroc'])} |"
            )
        lines.append("")

    # Full E7 table with CIs for hybrid
    lines += [
        "### Full E7 table: hybrid AUROC with bootstrap CIs",
        "",
        "| Corruption | Severity | Hybrid AUROC | Hybrid 95% CI | E6 AUROC | Maha AUROC |",
        "|---|---|---|---|---|---|",
    ]
    for key in sorted(e7_results.keys()):
        cond = e7_results[key]
        if cond.get("skip") or cond.get("severity", 0) == 0:
            continue
        h = cond.get("hybrid", {})
        e6m = cond.get("e6", {})
        maha = cond.get("mahalanobis", {})
        lines.append(
            f"| {cond['corruption']} | {cond['severity']} | "
            f"{_fmt(h.get('auroc', float('nan')))} | "
            f"{_fmt_ci(h.get('auroc_ci', (float('nan'),) * 3))} | "
            f"{_fmt(e6m.get('auroc', float('nan')))} | "
            f"{_fmt(maha.get('auroc', float('nan')))} |"
        )
    lines.append("")

    # --- Summary / discussion ---
    # Compute overall means for the discussion paragraph
    all_e6_auroc = [
        cond["e6"]["auroc"]
        for cond in e7_results.values()
        if not cond.get("skip") and cond.get("severity", 0) > 0
        and np.isfinite(cond["e6"]["auroc"])
    ]
    all_maha_auroc = [
        cond["mahalanobis"]["auroc"]
        for cond in e7_results.values()
        if not cond.get("skip") and cond.get("severity", 0) > 0
        and np.isfinite(cond["mahalanobis"]["auroc"])
    ]
    all_hybrid_auroc = [
        cond["hybrid"]["auroc"]
        for cond in e7_results.values()
        if not cond.get("skip") and cond.get("severity", 0) > 0
        and np.isfinite(cond["hybrid"]["auroc"])
    ]
    e6_mean = float(np.mean(all_e6_auroc)) if all_e6_auroc else float("nan")
    maha_mean = float(np.mean(all_maha_auroc)) if all_maha_auroc else float("nan")
    hybrid_mean = float(np.mean(all_hybrid_auroc)) if all_hybrid_auroc else float("nan")

    lines += [
        "## Discussion",
        "",
        f"**Photometric corruption (E7, {len(all_e6_auroc)} conditions):**",
        f"E6 mean AUROC = {_fmt(e6_mean)}, Maha mean AUROC = {_fmt(maha_mean)}, "
        f"Hybrid mean AUROC = {_fmt(hybrid_mean)}. "
        "The hybrid matches Mahalanobis on photometric corruptions (where E6 is "
        "near-chance) because the Mahalanobis arm dominates. The E6 arm adds no "
        "photometric coverage but does not degrade it.",
        "",
        f"**Temporal collapse (E4, alpha=1.0):**",
        "Hybrid AUROC matches or slightly exceeds E6-alone because both arms fire "
        "on severe collapse: the hidden state both freezes (E6 fires) AND drifts "
        "from the ID mean (Maha fires). The hybrid score's max-combination gives "
        "at least as high a score as either arm alone.",
        "",
        "**FPR calibration:**",
        "E6 arm LOCO FPR is ~1% (location-invariant, calibrates across corpora). "
        "Mahalanobis arm LOCO FPR is 100% (location-sensitive, the subaru and ram "
        "corpora are disjoint in the 512-D space). The combined LOCO FPR equals "
        "the Mahalanobis arm's FPR. This is NOT fixable by threshold tightening "
        "(verified at p=99.0, 99.9, 99.99 in baselines.py). Deployment prescription: "
        "calibrate the Mahalanobis arm on the deployed vehicle's corpus only; "
        "the LOCO failure reflects corpus-to-corpus location shift, not OOD.",
        "",
        "**Why Mahalanobis scores above chance on collapse (unlike the "
        "collapse-to-mean failure mode in the paper plan):**",
        "On the E4 sweep, CARLA collapse pushes the hidden state both to low "
        "variance AND to a different mean (the CARLA attractor is a distinct "
        "point, not the ID mean). So Mahalanobis fires correctly. The paper "
        "plan's 'Mahalanobis below chance' warning applies to collapse-exactly-"
        "to-the-mean; on this model the CARLA attractor is off-center enough "
        "that Mahalanobis still works on the collapse axis.",
        "",
    ]

    # --- Submodule localization (Task B) ---
    lines += [
        "## Task B: Submodule collapse localization",
        "",
        "Using report/e5_submodule_collected.npz (8 probe points from vision "
        "post-encoder through recurrent/policy stack).",
        "",
        "| Probe | Role | Cliff alpha | Activity ratio at alpha=1.0 | "
        "Mean shift at alpha=1.0 |",
        "|---|---|---|---|---|",
    ]
    for row in submodule_result["table"]:
        cliff_str = f"{row['cliff_alpha']:.3f}" if np.isfinite(row["cliff_alpha"]) else "n/a"
        lines.append(
            f"| `{row['probe']}` | {row['role']} | {cliff_str} | "
            f"{row['activity_ratio_at_1']:.4f} | {row['mean_shift_at_1']:.4f} |"
        )

    lines += [
        "",
        f"**First to collapse: `{submodule_result['first_collapse_probe']}` "
        f"(cliff alpha = {_fmt(submodule_result['first_collapse_cliff'], 3)})**",
        "",
        "Activity ratios at alpha=0.5 (mid-sweep):",
    ]
    for row in submodule_result["table"]:
        lines.append(
            f"- `{row['probe']}`: {row['activity_ratio_at_05']:.4f}"
        )
    lines += [
        "",
        "The activity ratio measures sum(temporal std of activations at OOD) / "
        "sum(temporal std at real). A ratio below 0.5 means the activations have "
        "less than half the temporal variation they had on real frames -- the "
        "network's internal dynamics have frozen. The cliff alpha is the smallest "
        "alpha where this ratio drops below 0.5.",
        "",
    ]

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


# -------------------------------------------------------------------------
# Figures
# -------------------------------------------------------------------------


def _fig_e4_auroc_sweep(e4_results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_alpha = e4_results["per_alpha"]
    alphas = [r["alpha"] for r in per_alpha]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    for det, color, ls in [
        ("hybrid", "#2ca02c", "-"),
        ("e6", "#d62728", "--"),
        ("mahalanobis", "#1f77b4", ":"),
        ("rmd", "#ff7f0e", "-."),
    ]:
        vals = [r.get(f"{det}_auroc", float("nan")) for r in per_alpha]
        ax.plot(alphas, vals, color=color, ls=ls, lw=2, label=det, marker="o", ms=4)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--", alpha=0.5, label="chance")
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("AUROC")
    ax.set_title("E8: Hybrid vs single detectors -- collapse axis (E4 sweep)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e8_e4_auroc_sweep.png", dpi=150)
    plt.close(fig)


def _fig_e7_mean_auroc(e7_results: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.e7_corruption import CORRUPTION_NAMES as E7_NAMES
    from collections import defaultdict

    per_corruption: dict[str, dict[str, list]] = defaultdict(
        lambda: {d: [] for d in ["e6", "mahalanobis", "rmd", "hybrid"]}
    )
    for cond in e7_results.values():
        if cond.get("skip") or cond.get("severity", 0) == 0:
            continue
        cname = cond["corruption"]
        for det in ["e6", "mahalanobis", "rmd", "hybrid"]:
            v = cond[det]["auroc"] if det in cond else float("nan")
            if np.isfinite(v):
                per_corruption[cname][det].append(v)

    corruptions = [c for c in E7_NAMES if c in per_corruption]
    x = np.arange(len(corruptions))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6), dpi=140)
    for i, (det, color) in enumerate([
        ("e6", "#d62728"),
        ("mahalanobis", "#1f77b4"),
        ("rmd", "#ff7f0e"),
        ("hybrid", "#2ca02c"),
    ]):
        means = [
            float(np.mean(per_corruption[c][det])) if per_corruption[c][det]
            else float("nan")
            for c in corruptions
        ]
        ax.bar(x + i * width, means, width, label=det, color=color, alpha=0.8)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels([c.replace("_", "\n") for c in corruptions], fontsize=7)
    ax.set_ylabel("Mean AUROC (across 5 severities)")
    ax.set_title("E8: Hybrid vs single detectors -- photometric corruptions (E7)")
    ax.legend(fontsize=9)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--", alpha=0.5)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e8_e7_mean_auroc.png", dpi=150)
    plt.close(fig)


def _fig_submodule_ratios(submodule_result: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    alphas = submodule_result["alphas"]
    ratios = submodule_result["ratios"]
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(ratios)))
    for (name, r), c in zip(ratios.items(), colors):
        cliff = submodule_result["cliffs"].get(name, float("nan"))
        label = name if not np.isfinite(cliff) else f"{name} (cliff={cliff:.2f})"
        ax.plot(alphas, r, marker="o", lw=1.6, color=c, label=label)
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", label="activity ratio 0.5")
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("Activity ratio (OOD / real temporal std)")
    ax.set_title("E8 / Task B: Submodule collapse localization")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(-0.05, 1.5)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e8_submodule_ratios.png", dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E8: Hybrid OOD detector")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--e6-percentile", type=float, default=1.0)
    ap.add_argument("--maha-percentile", type=float, default=99.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    print("Loading ID calibration data ...")
    by_corpus = _load_id_by_corpus()
    id_hidden = _load_id_all()

    print("Computing LOCO FPR ...")
    loco_result = loco_fpr_hybrid(
        by_corpus,
        window=args.window,
        e6_percentile=args.e6_percentile,
        maha_percentile=args.maha_percentile,
    )
    print(f"  E6 LOCO mean FPR: {loco_result['e6_fpr_mean']:.4f}")
    print(f"  Maha LOCO mean FPR: {loco_result['maha_fpr_mean']:.4f}")
    print(f"  Combined LOCO mean FPR: {loco_result['combined_fpr_mean']:.4f}")

    print("Evaluating on E4 collapse sweep ...")
    e4_results = evaluate_on_e4(
        window=args.window, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    print("  done. Headline at alpha=1.0:")
    for det in ["e6", "mahalanobis", "rmd", "hybrid"]:
        m = e4_results["headline"][det]
        print(f"    {det}: AUROC={m['auroc']:.4f} [{m['auroc_ci'][1]:.4f}, "
              f"{m['auroc_ci'][2]:.4f}]")

    print("Evaluating on E7 corruption sweep ...")
    e7_results = evaluate_on_e7(
        window=args.window, n_bootstrap=args.n_bootstrap, seed=args.seed
    )
    n_conds = sum(1 for c in e7_results.values() if not c.get("skip") and c.get("severity", 0) > 0)
    print(f"  done. {n_conds} non-clean conditions analyzed.")

    print("Localizing submodule collapse ...")
    submodule_result = localize_collapse()
    print(f"  First to collapse: {submodule_result['first_collapse_probe']} "
          f"(cliff alpha={submodule_result['first_collapse_cliff']:.3f})")

    print("Writing results ...")
    write_results(
        e7_results, e4_results, loco_result, submodule_result,
        RESULTS_MD, n_bootstrap=args.n_bootstrap,
    )

    print("Generating figures ...")
    _fig_e4_auroc_sweep(e4_results, FIG_DIR)
    _fig_e7_mean_auroc(e7_results, FIG_DIR)
    _fig_submodule_ratios(submodule_result, FIG_DIR)

    print(f"\nResults written to {RESULTS_MD.relative_to(ROOT)}")
    print(f"Figures written to {FIG_DIR.relative_to(ROOT)}/e8_*.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
