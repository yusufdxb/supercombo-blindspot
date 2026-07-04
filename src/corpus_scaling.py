"""Corpus scaling for the E6 monitor's false-positive rate.

The audit's #1 blocker: the E6 cross-corpus FPR rested on N=2 real corpora
(subaru + ram), which is effectively a two-fold estimate. This module lifts the
clean-real calibration set to every real corpus whose hidden_state is cached, and
reports the leave-one-corpus-out (LOCO) FPR with a SEGMENT-LEVEL bootstrap CI
(resampling corpora, not frames -- frame-level bootstrap is invalid here because
recurrent features are strongly autocorrelated, as codex flagged).

Clean-real corpora (calibration-eligible): subaru, ram (teardown cache) +
ev6_night, bronco_night (real_weather cache). The daytime_control segment is a
real recording on which E6 FIRES (a near-zero recurrent attractor); it is NOT
clean calibration data, so it is held out and reported separately as the real
near-collapse case.

This runs from cached hidden_states only (no GPU, no raw frames). Scaling beyond
these corpora requires fetching + running additional public comma segments
(scripts/fetch_upgrade_data.py pattern + a GPU --collect); that path is documented
in docs/REPRODUCIBILITY.md and left as the next increment.

Run:
    python -m src.corpus_scaling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.e6_detector import calibrate_threshold, loco_fpr, rolling_spread

ROOT = Path(__file__).resolve().parent.parent
TEARDOWN = ROOT / "report" / "teardown_collected.npz"
REAL_WEATHER = ROOT / "report" / "real_weather_collected.npz"
OUT_MD = ROOT / "report" / "corpus_scaling_results.md"
FIG = ROOT / "report" / "figures" / "corpus_scaling.png"

WINDOW = 30
PERCENTILE = 1.0
# subaru + ram were the original N=2 calibration set.
ORIGINAL = ["subaru", "ram"]


def load_real_corpora() -> tuple[dict, dict]:
    """Return (clean_real, near_collapse) dicts of name -> hidden_state (T, 512).

    clean_real are calibration-eligible real drives; near_collapse holds real
    recordings on which the monitor fires (daytime_control)."""
    clean: dict[str, np.ndarray] = {}
    if TEARDOWN.exists():
        d = np.load(TEARDOWN)
        for name in ("subaru", "ram"):
            key = f"{name}__hidden_state"
            if key in d.files:
                clean[name] = d[key]
    near: dict[str, np.ndarray] = {}
    if REAL_WEATHER.exists():
        d = np.load(REAL_WEATHER)
        for name in ("ev6_night", "bronco_night"):
            key = f"{name}__hidden_state"
            if key in d.files:
                clean[name] = d[key]
        if "daytime_control__hidden_state" in d.files:
            near["daytime_control"] = d["daytime_control__hidden_state"]
    return clean, near


def bootstrap_mean_ci(values: list[float], b: int = 10000, ci: float = 95.0,
                      seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean of per-corpus FPRs (segment-level:
    each bootstrap draw resamples whole corpora with replacement)."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return float("nan"), float("nan")
    means = arr[rng.integers(0, len(arr), size=(b, len(arr)))].mean(axis=1)
    lo = float(np.percentile(means, (100 - ci) / 2))
    hi = float(np.percentile(means, 100 - (100 - ci) / 2))
    return lo, hi


def corpus_fpr(hidden: np.ndarray, threshold: float, window: int = WINDOW) -> float:
    s = rolling_spread(hidden, window)
    v = s[~np.isnan(s)]
    return float(np.mean(v < threshold)) if len(v) else float("nan")


def run(window: int = WINDOW, percentile: float = PERCENTILE) -> dict:
    clean, near = load_real_corpora()
    if len(clean) < 2:
        raise RuntimeError(f"need >=2 clean real corpora, found {list(clean)}")

    # scaled LOCO over all clean corpora
    loco = loco_fpr(clean, window, percentile)
    per_fold = {k: v["fpr"] for k, v in loco["folds"].items()}
    lo, hi = bootstrap_mean_ci(list(per_fold.values()))

    # original N=2 LOCO (subaru + ram only) for the before/after delta
    orig_set = {k: clean[k] for k in ORIGINAL if k in clean}
    orig = loco_fpr(orig_set, window, percentile) if len(orig_set) == 2 else None

    # threshold calibrated on ALL clean corpora; score the near-collapse segment
    all_clean = np.concatenate(list(clean.values()), axis=0)
    thr_all = calibrate_threshold(rolling_spread(all_clean, window), percentile)
    near_fpr = {k: corpus_fpr(v, thr_all, window) for k, v in near.items()}

    return {"clean": list(clean), "per_fold": per_fold, "loco": loco,
            "ci": (lo, hi), "orig": orig, "thr_all": thr_all,
            "near_fpr": near_fpr, "window": window, "percentile": percentile}


def write_results(res: dict) -> None:
    L = ["# E6 corpus scaling: LOCO FPR with segment-level bootstrap", ""]
    n = len(res["clean"])
    L += [
        f"Clean-real calibration corpora (N={n}): {', '.join(res['clean'])}.",
        f"window={res['window']}, percentile={res['percentile']}. Rolling-spread "
        "monitor; FPR = fraction of a held-out corpus's frames flagged OOD by a "
        "threshold calibrated on the other corpora (leave-one-corpus-out).",
        "",
        "## Per-held-out-corpus FPR",
        "",
        "| held-out corpus | held-out FPR |",
        "|---|---|",
    ]
    for k, v in res["per_fold"].items():
        L.append(f"| {k} | {v:.4f} |")
    lo, hi = res["ci"]
    L += [
        "",
        f"**LOCO mean FPR: {res['loco']['fpr_mean'] * 100:.2f}%** "
        f"(segment-level bootstrap 95% CI [{lo * 100:.2f}%, {hi * 100:.2f}%]); "
        f"LOCO max {res['loco']['fpr_max'] * 100:.2f}%.",
        "",
    ]
    if res["orig"] is not None:
        L += [
            f"Before/after: the original N=2 estimate (subaru+ram only) gave LOCO "
            f"mean {res['orig']['fpr_mean'] * 100:.2f}% / max "
            f"{res['orig']['fpr_max'] * 100:.2f}%. Scaling to N={n} clean corpora "
            f"{'tightens' if res['loco']['fpr_max'] <= res['orig']['fpr_max'] else 'widens'} "
            "the held-out spread and gives the first cross-corpus CI.",
            "",
        ]
    if res["near_fpr"]:
        L += [
            "## Real near-collapse (held out of calibration)",
            "",
            "These are real recordings on which the monitor fires; they are NOT "
            "clean calibration data and are excluded from the LOCO set above. "
            f"Scored at the all-clean threshold {res['thr_all']:.6f}:",
            "",
            "| segment | fraction flagged |",
            "|---|---|",
        ]
        for k, v in res["near_fpr"].items():
            L.append(f"| {k} | {v:.4f} |")
        L.append("")
    L += [
        "## Scope",
        "",
        f"N={n} clean real corpora is an honest lift from the prior N=2 but is not "
        "yet fleet-scale. The pipeline consumes any cached real hidden_state, so "
        "reaching N=30-50 is a matter of fetching + running additional public comma "
        "segments (scripts/fetch_upgrade_data.py pattern + GPU --collect); the "
        "segment-level bootstrap and LOCO machinery here are unchanged by adding "
        "corpora.",
    ]
    OUT_MD.write_text("\n".join(L) + "\n")


def make_figure(res: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import physx_style as _physx_style  # editorial-print theme
        _physx_style.apply()
    except Exception:
        return
    names = list(res["per_fold"])
    vals = [res["per_fold"][k] * 100 for k in names]
    lo, hi = res["ci"]
    mean = res["loco"]["fpr_mean"] * 100
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.bar(names, vals, color="#2a78d6", label="held-out FPR")
    ax.axhline(mean, color="black", lw=1.2, label=f"LOCO mean {mean:.2f}%")
    ax.axhspan(lo * 100, hi * 100, color="grey", alpha=0.25,
               label=f"95% CI [{lo*100:.2f}, {hi*100:.2f}]")
    ax.set_ylabel("held-out FPR (%)")
    ax.set_title(f"E6 LOCO FPR across N={len(names)} clean real corpora")
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def main() -> int:
    res = run()
    write_results(res)
    make_figure(res)
    print(f"clean corpora (N={len(res['clean'])}): {', '.join(res['clean'])}")
    for k, v in res["per_fold"].items():
        print(f"  held-out {k:14s} FPR {v*100:6.2f}%")
    lo, hi = res["ci"]
    print(f"  LOCO mean {res['loco']['fpr_mean']*100:.2f}%  "
          f"95% CI [{lo*100:.2f}%, {hi*100:.2f}%]  max {res['loco']['fpr_max']*100:.2f}%")
    if res["orig"]:
        print(f"  (was N=2: mean {res['orig']['fpr_mean']*100:.2f}% "
              f"max {res['orig']['fpr_max']*100:.2f}%)")
    for k, v in res["near_fpr"].items():
        print(f"  near-collapse {k}: {v*100:.1f}% flagged")
    print(f"  wrote {OUT_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
