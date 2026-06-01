"""E4 v0.9.6: real-to-sim interpolation sweep on supercombo v0.9.6.

Identical methodology to src/e4_interp.py. Blends Subaru real inputs
toward CARLA inputs; writes _v096-suffixed outputs only.

    python -m src.e4_interp_v096              # analysis from cache
    env -u PYTHONPATH .venv/bin/python -m src.e4_interp_v096 --collect
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.teardown import (CARLA_C, HEAD_NAMES, N, REAL_C, SCALARS, WARMUP,
                          WARN_C, _flat, _plt, _post)
from src.e4_interp import (
    BASE_ALPHAS, CLIFF_WIDTH, REFINE_GAP, REFINE_ROUNDS,
    activity_per_head, blend, feature_centroid, feature_projection,
    feature_spread, mean_uncertainty, normalized_activity, transition_width,
    load_cache, save_cache, _fc_transition_width, _is_finite,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "report" / "e4_v096_collected.npz"
FIG = ROOT / "report" / "figures" / "e4_interpolation_v096.png"
RESULTS = ROOT / "report" / "e4_v096_results.md"
SUBARU_HEVC = DATA / "subaru_source" / "fcamera.hevc"
SUBARU_RLOG = DATA / "subaru_source" / "rlog.bz2"
CARLA_NPY = DATA / "domain_gap" / "carla_rgb.npy"
MODEL_PATH = ROOT / "models" / "supercombo_v096.onnx"


def _collect_live() -> dict[float, dict]:
    from src.probe_model import collect, load_carla_six, load_real_six
    from src.state import build_session, load_output_slices

    sess = build_session(MODEL_PATH)
    slices = load_output_slices(MODEL_PATH)
    providers = sess.get_providers()
    print(f"v0.9.6 session providers: {providers}")
    print(f"Loading Subaru + CARLA frame sequences (N={N}) ...")
    real_six = load_real_six(SUBARU_HEVC, SUBARU_RLOG, N)
    carla_six = load_carla_six(CARLA_NPY, N)

    collected: dict[float, dict] = {}

    def run(alpha: float) -> None:
        blended = [blend(r, c, alpha) for r, c in zip(real_six, carla_six)]
        collected[alpha] = collect(blended, sess, slices)
        print(f"  alpha={alpha:.4f} collected")

    print(f"v0.9.6 first pass: {len(BASE_ALPHAS)} alphas "
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


def fig_interp_v096(alphas, norm, fproj, unc, a90, a10) -> None:
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
                    label=f"transition (width {a10 - a90:.3f})")
    ax1.set_ylabel("normalized")
    ax1.set_title("E4 v0.9.6  supercombo across the real-to-sim interpolation")
    ax1.legend(facecolor="#161a22", edgecolor="#3a3f4b")

    ax2.plot(xs, [unc[a] for a in xs], "-o", color="#9aa0aa", lw=2,
             label="predicted plan uncertainty")
    ax2.set_xlabel("alpha   (0 = real Subaru frame,  1 = CARLA frame)")
    ax2.set_ylabel("mean plan_std")
    ax2.legend(facecolor="#161a22", edgecolor="#3a3f4b")

    fig.tight_layout()
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=150)
    plt.close(fig)


def write_results_v096(alphas, norm, fproj, spread, unc,
                       a90, a10, fc_10, fc_90, fc_width, verdict) -> None:
    xs = sorted(alphas)
    norm_min = min(norm[a] for a in xs)
    L = ["# E4 v0.9.6 results: real-to-sim interpolation", "",
         f"Pixel alpha-blend of the Subaru real sequence and the CARLA "
         f"sequence (N={N} frames, {WARMUP} warmup discarded). alpha=0 is the "
         f"real frame, alpha=1 is the CARLA frame. Model: supercombo_v096.onnx.", ""]
    if a90 is not None and a10 is not None:
        width = a10 - a90
        L.append(
            f"**Verdict: {verdict}.** Output activity falls from 0.9 to 0.1 of "
            f"the real baseline over alpha {a90:.3f} to {a10:.3f} "
            f"(transition width {width:.3f}; < {CLIFF_WIDTH} reads as a cliff)."
        )
    else:
        L.append(
            f"**Verdict: {verdict} (feature-collapse signal).** "
            f"Output-activity floor never reaches 0.1x (min {norm_min:.2f}x); "
            f"transition computed on feature-collapse signal instead: "
            f"alpha {fc_10:.3f} to {fc_90:.3f} "
            f"(transition width {fc_width:.3f}; < {CLIFF_WIDTH} reads as a cliff)."
        )
    L += ["",
          "| alpha | output activity | feature collapse | feature spread | plan uncertainty |",
          "|---|---|---|---|---|"]
    for a in xs:
        L.append(f"| {a:.4f} | {norm[a]:.4f} | {fproj[a]:.4f} "
                 f"| {spread[a]:.2f} | {unc[a]:.4f} |")
    RESULTS.write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E4 v0.9.6 real-to-sim interpolation sweep")
    ap.add_argument("--collect", action="store_true",
                    help=f"re-run the v0.9.6 model instead of using {CACHE.name}")
    args = ap.parse_args(argv)

    if args.collect or not CACHE.exists():
        collected = _collect_live()
        save_cache(CACHE, collected)
        print(f"  cached v0.9.6 E4 outputs -> {CACHE.relative_to(ROOT)}")
    else:
        print(f"Loading {CACHE.relative_to(ROOT)} (pass --collect to re-run) ...")
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

    # Compute feature-collapse transition width (monotone 0->1, always valid)
    fc_10, fc_90, fc_width = _fc_transition_width(alphas, fproj)

    # Base verdict on the signal that crosses cleanly.
    if a90 is not None and a10 is not None:
        width = a10 - a90
        verdict = "cliff" if width < CLIFF_WIDTH else "gradient"
    else:
        verdict = "cliff" if fc_width < CLIFF_WIDTH else "gradient"

    print(f"\n=== E4 v0.9.6  REAL-TO-SIM INTERPOLATION ({len(alphas)} alphas) ===")
    print(f"  {'alpha':>7} {'activity':>10} {'feat collapse':>14} "
          f"{'uncertainty':>12}")
    for a in alphas:
        print(f"  {a:>7.4f} {norm[a]:>10.4f} {fproj[a]:>14.4f} "
              f"{unc[a]:>12.4f}")
    if a90 is not None and a10 is not None:
        print(f"\n  transition width (output-activity): alpha {a90:.3f} -> {a10:.3f} "
              f"= {a10 - a90:.3f}  ->  {verdict.upper()}")
    else:
        norm_min = min(norm[a] for a in alphas)
        print(f"\n  output-activity floor never reaches 0.1x (min {norm_min:.2f}x); "
              f"transition on feature-collapse: alpha {fc_10:.3f} -> {fc_90:.3f} "
              f"= {fc_width:.3f}  ->  {verdict.upper()}")

    # For figure: use fc crossings since output-activity doesn't cross
    fig_a90 = a90 if a90 is not None else fc_10
    fig_a10 = a10 if a10 is not None else fc_90
    fig_interp_v096(alphas, norm, fproj, unc, fig_a90, fig_a10)
    write_results_v096(alphas, norm, fproj, spread, unc,
                       a90, a10, fc_10, fc_90, fc_width, verdict)
    print(f"  figure -> {FIG.relative_to(ROOT)}   "
          f"results -> {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
