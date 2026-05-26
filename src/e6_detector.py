"""E6: a self-aware OOD detector from supercombo's own internal state.

E2 showed the model's 512-D recurrent feature vector freezes to a single
point on CARLA. E6 asks whether a downstream monitor watching the rolling
spread of that vector, calibrated on real driving, would catch the
collapse, and at what alpha along the E4 sweep it fires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def rolling_spread(hidden: np.ndarray, window: int) -> np.ndarray:
    """Per-frame trace of the rolling covariance of the hidden state.
    `hidden` is shape (T, D). Returns shape (T,), NaN before the window fills."""
    T, D = hidden.shape
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T + 1):
        out[t - 1] = float(np.var(hidden[t - window:t], axis=0).sum())
    return out


def calibrate_threshold(real_spreads: np.ndarray, percentile: float = 1.0) -> float:
    """Below this spread = OOD. Pick the `percentile`-th percentile of the
    real-driving spread distribution so real drives stay above it ~99% of
    the time."""
    s = real_spreads[~np.isnan(real_spreads)]
    return float(np.percentile(s, percentile))


def _e4_alphas(cache: Path) -> np.ndarray:
    """All unique alpha values present in the E4 cache, sorted ascending."""
    d = np.load(cache)
    alphas = sorted({float(k.split("__")[0]) for k in d.files if "__hidden_state" in k})
    return np.array(alphas)


def _e4_hidden_at(cache: Path, alpha: float) -> np.ndarray:
    """Pull the hidden_state array for a specific alpha out of the E4 cache."""
    d = np.load(cache)
    return d[f"{alpha:.4f}__hidden_state"]


def evaluate_on_e4(e4_cache: Path, thr: float, window: int = 30) -> dict:
    """For each alpha in the E4 cache, compute the fraction of frames whose
    rolling-spread sits below the calibrated threshold. Return alphas,
    fired_fraction, and the smallest alpha where >50% of frames fired."""
    alphas = _e4_alphas(e4_cache)
    fired = []
    for a in alphas:
        H = _e4_hidden_at(e4_cache, float(a))
        s = rolling_spread(H, window)
        s = s[~np.isnan(s)]
        fired.append(float(np.mean(s < thr)) if len(s) else float("nan"))
    fired = np.array(fired)
    above = fired > 0.5
    fires_at = float(alphas[np.argmax(above)]) if above.any() else float("nan")
    return {"alphas": alphas, "fired_fraction": fired, "fires_at": fires_at,
            "threshold": thr}


def _real_calibration_hidden() -> np.ndarray:
    """Concatenate subaru + ram hidden_state from teardown_collected.npz.
    This is the real-driving calibration corpus for the threshold."""
    d = np.load("report/teardown_collected.npz")
    return np.concatenate([d["subaru__hidden_state"], d["ram__hidden_state"]], axis=0)


def _figure(res: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.plot(res["alphas"], res["fired_fraction"], "o-", color="#d63a3a", lw=1.8)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--",
               label="fire threshold (50% of frames)")
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("fraction of frames flagged OOD")
    ax.set_title("E6: hidden_state-spread detector along the E4 sweep")
    if not np.isnan(res["fires_at"]):
        ax.axvline(res["fires_at"], color="black", lw=0.8,
                   label=f"detector fires @ alpha={res['fires_at']:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def _write_results(res: dict, real_spreads: np.ndarray, out: Path) -> None:
    valid = real_spreads[~np.isnan(real_spreads)]
    fpr = float(np.mean(valid < res["threshold"])) if len(valid) else float("nan")
    lines = ["# E6 Results: Self-Aware OOD Detector", ""]
    lines.append(f"- threshold: {res['threshold']:.6f} "
                 "(1st percentile of real-driving rolling spread)")
    lines.append(f"- real-driving false-positive rate at this threshold: {fpr:.4f}")
    lines.append(f"- detector fires (>50% of frames) at alpha = {res['fires_at']:.3f}")
    lines.append("")
    lines.append("| alpha | fired fraction |")
    lines.append("|---|---|")
    for a, f in zip(res["alphas"], res["fired_fraction"]):
        lines.append(f"| {a:.4f} | {f:.3f} |")
    out.write_text("\n".join(lines) + "\n")


import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--percentile", type=float, default=1.0)
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args(argv)

    real_hidden = _real_calibration_hidden()
    real_spreads = rolling_spread(real_hidden, args.window)
    thr = calibrate_threshold(real_spreads, args.percentile)

    res = evaluate_on_e4(Path("report/e4_collected.npz"), thr, args.window)

    _figure(res, Path("report/figures/e6_detector.png"))
    _write_results(res, real_spreads, Path("report/e6_results.md"))
    print("E6 done. threshold=", thr, "fires_at_alpha=", res["fires_at"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
