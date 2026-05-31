"""E6 v0.9.6: OOD detector on the v0.9.6 supercombo hidden state.

Calibrates the rolling-spread threshold on Subaru+RAM hidden states from
teardown_v096_collected.npz, then evaluates it on the E4 v0.9.6 sweep
(e4_v096_collected.npz). Writes report/e6_v096_results.md and the figure.
Does NOT touch any v0.9.7 cached file.

    env -u PYTHONPATH .venv/bin/python -m src.e6_detector_v096
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.e6_detector import (
    calibrate_threshold, evaluate_on_e4, loco_fpr, rolling_spread,
)

ROOT = Path(__file__).resolve().parents[1]
TEARDOWN_CACHE = ROOT / "report" / "teardown_v096_collected.npz"
E4_CACHE = ROOT / "report" / "e4_v096_collected.npz"
FIG_OUT = ROOT / "report" / "figures" / "e6_detector_v096.png"
RESULTS_OUT = ROOT / "report" / "e6_v096_results.md"


def _real_by_corpus_v096() -> dict[str, np.ndarray]:
    d = np.load(TEARDOWN_CACHE)
    return {"subaru": d["subaru__hidden_state"],
            "ram": d["ram__hidden_state"]}


def _figure_v096(res: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4), dpi=140)
    ax.plot(res["alphas"], res["fired_fraction"], "o-", color="#d63a3a", lw=1.8)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--",
               label="fire threshold (50% of frames)")
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("fraction of frames flagged OOD")
    ax.set_title("E6 v0.9.6: hidden_state-spread detector along the E4 sweep")
    if not np.isnan(res["fires_at"]):
        ax.axvline(res["fires_at"], color="black", lw=0.8,
                   label=f"detector fires @ alpha={res['fires_at']:.2f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def _write_results_v096(res: dict, all_real_hidden: np.ndarray, out: Path) -> None:
    real_spreads = rolling_spread(all_real_hidden, 30)
    valid = real_spreads[~np.isnan(real_spreads)]
    in_sample_fpr = float(np.mean(valid < res["threshold"])) if len(valid) else float("nan")

    loco = res["loco"]
    lines = ["# E6 v0.9.6 Results: Self-Aware OOD Detector", ""]
    lines.append("## Threshold and false-positive rate")
    lines.append("")
    lines.append(f"- threshold (calibrated on all real corpora, p=1.0): "
                 f"{res['threshold']:.6f}")
    lines.append(f"- in-sample FPR at this threshold (definitional): "
                 f"{in_sample_fpr:.4f}")
    lines.append("")
    lines.append("## Held-out FPR (leave-one-corpus-out across {subaru, ram})")
    lines.append("")
    lines.append("| held-out corpus | calibrated on | threshold | held-out FPR |")
    lines.append("|---|---|---|---|")
    for name, fold in loco["folds"].items():
        calib = ", ".join(fold["calibrated_on"])
        lines.append(f"| {name} | {calib} | {fold['threshold']:.6f} | "
                     f"{fold['fpr']:.4f} |")
    lines.append("")
    lines.append(f"**LOCO mean FPR: {loco['fpr_mean']:.4f} "
                 f"({loco['fpr_mean'] * 100:.2f}%)**")
    lines.append(f"**LOCO max FPR: {loco['fpr_max']:.4f} "
                 f"({loco['fpr_max'] * 100:.2f}%)**")
    lines.append("")
    lines.append("## Detector response on the E4 v0.9.6 sweep")
    lines.append("")
    fires_str = f"{res['fires_at']:.3f}" if not np.isnan(res["fires_at"]) else "never"
    lines.append(f"- detector fires (>50% of frames flagged) at alpha = {fires_str}")
    lines.append("")
    lines.append("| alpha | fired fraction |")
    lines.append("|---|---|")
    for a, f in zip(res["alphas"], res["fired_fraction"]):
        lines.append(f"| {a:.4f} | {f:.3f} |")
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="E6 v0.9.6 OOD detector")
    p.add_argument("--percentile", type=float, default=1.0)
    p.add_argument("--window", type=int, default=30)
    args = p.parse_args(argv)

    for cache in (TEARDOWN_CACHE, E4_CACHE):
        if not cache.exists():
            print(f"Missing cache: {cache}")
            print("Run teardown_v096 --collect and e4_interp_v096 --collect first.")
            return 1

    by_corpus = _real_by_corpus_v096()
    loco = loco_fpr(by_corpus, args.window, args.percentile)

    all_real = np.concatenate(list(by_corpus.values()), axis=0)
    full_thr = calibrate_threshold(rolling_spread(all_real, args.window),
                                   args.percentile)

    res = evaluate_on_e4(E4_CACHE, full_thr, args.window)
    res["loco"] = loco

    _figure_v096(res, FIG_OUT)
    _write_results_v096(res, all_real, RESULTS_OUT)

    fires_str = f"{res['fires_at']:.3f}" if not np.isnan(res["fires_at"]) else "never"
    print(f"E6 v0.9.6 done.")
    print(f"  full-corpus threshold = {full_thr:.6f}")
    print(f"  LOCO fpr mean = {loco['fpr_mean']:.4f} ({loco['fpr_mean']*100:.2f}%)")
    print(f"  LOCO fpr max  = {loco['fpr_max']:.4f} ({loco['fpr_max']*100:.2f}%)")
    print(f"  fires_at_alpha = {fires_str}")
    print(f"\nE4 v0.9.6 sweep -- alpha / fired_fraction:")
    for a, f in zip(res["alphas"], res["fired_fraction"]):
        print(f"  alpha={a:.4f}  fired={f:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
