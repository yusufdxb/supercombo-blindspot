"""E4-RAM: real-to-sim interpolation sweep using the RAM segment.

Re-runs the E4 alpha-blend sweep with the RAM real-driving source instead
of Subaru, to test whether the cliff location and transition width are
vehicle-invariant.

    python -m src.e4_ram              # analysis + figure, from cache
    env -u PYTHONPATH .venv/bin/python -m src.e4_ram --collect  # re-run model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.e4_interp import (
    BASE_ALPHAS,
    CLIFF_WIDTH,
    REFINE_GAP,
    REFINE_ROUNDS,
    activity_per_head,
    blend,
    feature_centroid,
    feature_projection,
    feature_spread,
    load_cache,
    mean_uncertainty,
    normalized_activity,
    save_cache,
    transition_width,
)
from src.teardown import (
    CARLA_C, HEAD_NAMES, N, REAL_C, SCALARS, WARMUP,
    WARN_C, _flat, _plt, _post,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "report" / "e4_ram_collected.npz"
FIG = ROOT / "report" / "figures" / "e4_ram_interpolation.png"
RESULTS = ROOT / "report" / "e4_ram_results.md"
RAM_HEVC = DATA / "ram_source" / "fcamera.hevc"
RAM_RLOG = DATA / "ram_source" / "rlog.bz2"
CARLA_NPY = DATA / "domain_gap" / "carla_rgb.npy"
SUBARU_CACHE = ROOT / "report" / "e4_collected.npz"


def _collect_live() -> dict[float, dict]:
    """Run supercombo over blended RAM + CARLA sequences."""
    from src.probe_model import collect, load_carla_six, load_real_six
    from src.state import build_session, load_output_slices

    sess, slices = build_session(), load_output_slices()
    print(f"Loading RAM + CARLA frame sequences (N={N}) ...")
    real_six = load_real_six(RAM_HEVC, RAM_RLOG, N)
    carla_six = load_carla_six(CARLA_NPY, N)

    collected: dict[float, dict] = {}

    def run(alpha: float) -> None:
        blended = [blend(r, c, alpha) for r, c in zip(real_six, carla_six)]
        collected[alpha] = collect(blended, sess, slices)
        print(f"  alpha={alpha:.4f} collected")

    print(f"First pass: {len(BASE_ALPHAS)} alphas "
          f"(first warm triggers ~28 s PTX JIT) ...")
    for a in BASE_ALPHAS:
        run(a)

    for rnd in range(REFINE_ROUNDS):
        post = {a: _post(collected[a], WARMUP) for a in collected}
        norm = normalized_activity(
            {a: activity_per_head(post[a]) for a in post})
        xs = sorted(collected)
        added = []
        for a1, a2 in zip(xs, xs[1:]):
            if abs(norm[a1] - norm[a2]) > REFINE_GAP:
                mid = round((a1 + a2) / 2, 4)
                if mid not in collected:
                    added.append(mid)
        if not added:
            break
        print(f"Refinement round {rnd + 1}: inserting {added}")
        for a in added:
            run(a)

    return collected


def _e6_fires_at_alpha(collected: dict[float, dict]) -> float:
    """Smallest alpha where E6 rolling-spread fires on >50% of frames."""
    from src.e6_detector import rolling_spread, calibrate_threshold
    from src.baselines import _real_calibration_by_corpus

    by_corpus = _real_calibration_by_corpus()
    id_hidden = np.concatenate(list(by_corpus.values()), axis=0)
    real_spreads = rolling_spread(id_hidden, window=30)
    thr = calibrate_threshold(real_spreads, 1.0)

    for a in sorted(collected):
        H = collected[a]["hidden_state"][WARMUP:]
        s = rolling_spread(H, window=30)
        valid = s[~np.isnan(s)]
        frac = float(np.mean(valid < thr)) if len(valid) else 0.0
        if frac > 0.5:
            return float(a)
    return float("nan")


def _load_subaru_cliff() -> tuple[float, float, float]:
    """Load the Subaru E4 cliff parameters from its cache for comparison."""
    if not SUBARU_CACHE.exists():
        return float("nan"), float("nan"), float("nan")
    sub = load_cache(SUBARU_CACHE)
    post = {a: _post(sub[a], WARMUP) for a in sub}
    alphas = sorted(post)
    per_head = {a: activity_per_head(post[a]) for a in alphas}
    norm = normalized_activity(per_head)
    a90, a10 = transition_width(alphas, norm)
    return a90, a10, a10 - a90


def fig_interp(alphas, norm, fproj, unc, a90, a10,
               sub_a90: float = float("nan"),
               sub_a10: float = float("nan")) -> None:
    """Two-panel figure: (top) output activity + feature collapse vs alpha,
    (bottom) predicted uncertainty vs alpha. Overlays Subaru cliff region."""
    plt = _plt()
    xs = sorted(alphas)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(xs, [norm[a] for a in xs], "-o", color=REAL_C, lw=2,
             label="output activity (1.0 = real)")
    ax1.plot(xs, [fproj[a] for a in xs], "-s", color=CARLA_C, lw=2,
             label="feature collapse (1.0 = CARLA centroid)")
    if np.isfinite(a90) and np.isfinite(a10):
        ax1.axvspan(a90, a10, color=WARN_C, alpha=0.15,
                    label=f"RAM transition ({a10 - a90:.2f})")
    if np.isfinite(sub_a90) and np.isfinite(sub_a10):
        ax1.axvspan(sub_a90, sub_a10, color="#b39ddb", alpha=0.12,
                    label=f"Subaru transition ({sub_a10 - sub_a90:.2f})")
    ax1.set_ylabel("normalized")
    ax1.set_title("E4-RAM  supercombo across the real-to-sim interpolation (RAM source)")
    ax1.legend(facecolor="#161a22", edgecolor="#3a3f4b", fontsize=8)

    ax2.plot(xs, [unc[a] for a in xs], "-o", color="#9aa0aa", lw=2,
             label="predicted plan uncertainty")
    ax2.set_xlabel("alpha   (0 = real RAM frame,  1 = CARLA frame)")
    ax2.set_ylabel("mean plan_std")
    ax2.legend(facecolor="#161a22", edgecolor="#3a3f4b")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def write_results(alphas, norm, fproj, spread, unc,
                  a90, a10, width, verdict,
                  sub_a90, sub_a10, sub_width,
                  e6_fires: float = float("nan")) -> None:
    """Write report/e4_ram_results.md with the per-alpha table and comparison."""
    xs = sorted(alphas)
    L = ["# E4-RAM results: real-to-sim interpolation (RAM source)", "",
         f"Pixel alpha-blend of the RAM real sequence and the CARLA "
         f"sequence (N={N} frames, {WARMUP} warmup discarded). alpha=0 is the "
         f"real frame, alpha=1 is the CARLA frame.", "",
         f"**Verdict: {verdict}.** Output activity falls from 0.9 to 0.1 of "
         f"the real baseline over alpha {a90:.3f} to {a10:.3f} "
         f"(transition width {width:.3f}; < {CLIFF_WIDTH} reads as a cliff).",
         ""]

    # Comparison with Subaru
    L.append("## Comparison with Subaru E4")
    L.append("")
    L.append("| Source | a90 | a10 | Transition width | E6 fires-at-alpha | E6 headroom | Verdict |")
    L.append("|---|---|---|---|---|---|---|")
    sub_verdict = "cliff" if sub_width < CLIFF_WIDTH else "gradient"
    L.append(f"| Subaru | {sub_a90:.3f} | {sub_a10:.3f} | "
             f"{sub_width:.3f} | 0.550 | {sub_a90 - 0.550:.3f} | {sub_verdict} |")
    L.append(f"| RAM | {a90:.3f} | {a10:.3f} | {width:.3f} | "
             f"{e6_fires:.3f} | {a90 - e6_fires:.3f} | {verdict} |")
    L.append("")

    L.append("## Per-alpha table")
    L.append("")
    L.append("| alpha | output activity | feature collapse | feature spread | plan uncertainty |")
    L.append("|---|---|---|---|---|")
    for a in xs:
        L.append(f"| {a:.4f} | {norm[a]:.4f} | {fproj[a]:.4f} "
                 f"| {spread[a]:.2f} | {unc[a]:.4f} |")
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E4-RAM real-to-sim interpolation sweep (RAM source)")
    ap.add_argument("--collect", action="store_true",
                    help=f"re-run the model instead of using {CACHE.name}")
    args = ap.parse_args(argv)

    if args.collect or not CACHE.exists():
        collected = _collect_live()
        save_cache(CACHE, collected)
        print(f"  cached collected outputs -> {CACHE.relative_to(ROOT)}")
    else:
        print(f"Loading {CACHE.relative_to(ROOT)} "
              f"(no model / CARLA needed; pass --collect to re-run) ...")
        collected = load_cache(CACHE)

    post = {a: _post(collected[a], WARMUP) for a in collected}
    alphas = sorted(post)
    per_head = {a: activity_per_head(post[a]) for a in alphas}
    norm = normalized_activity(per_head)
    centroids = {a: feature_centroid(post[a]) for a in alphas}
    fproj = feature_projection(centroids)
    spread = {a: feature_spread(post[a]) for a in alphas}
    unc = {a: mean_uncertainty(post[a]) for a in alphas}
    a90, a10 = transition_width(alphas, norm)
    width = a10 - a90
    verdict = "cliff" if width < CLIFF_WIDTH else "gradient"

    # Load Subaru cliff for comparison
    sub_a90, sub_a10, sub_width = _load_subaru_cliff()

    # E6 fires-at-alpha on RAM sweep
    e6_fires = _e6_fires_at_alpha(collected)

    print(f"\n=== E4-RAM  REAL-TO-SIM INTERPOLATION ({len(alphas)} alphas) ===")
    print(f"  {'alpha':>7} {'activity':>10} {'feat collapse':>14} "
          f"{'uncertainty':>12}")
    for a in alphas:
        print(f"  {a:>7.4f} {norm[a]:>10.4f} {fproj[a]:>14.4f} "
              f"{unc[a]:>12.4f}")
    print(f"\n  RAM transition width: alpha {a90:.3f} -> {a10:.3f} "
          f"= {width:.3f}  ->  {verdict.upper()}")
    print(f"  E6 fires at alpha {e6_fires:.3f} "
          f"(Subaru was 0.550, headroom = {a90 - e6_fires:.3f})")
    if np.isfinite(sub_width):
        print(f"  Subaru transition width: alpha {sub_a90:.3f} -> {sub_a10:.3f} "
              f"= {sub_width:.3f}")
    else:
        print("  Subaru cache not found; no comparison available.")

    fig_interp(alphas, norm, fproj, unc, a90, a10, sub_a90, sub_a10)
    write_results(alphas, norm, fproj, spread, unc, a90, a10, width, verdict,
                  sub_a90, sub_a10, sub_width, e6_fires)
    print(f"  figure -> {FIG.relative_to(ROOT)}   "
          f"results -> {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
