"""E4: real-to-sim interpolation sweep.

Blends real Subaru model-frame inputs toward CARLA inputs and measures
supercombo along the path, to test whether the output collapse from the
E1/E2/E3 teardown is a sharp cliff or a smooth gradient.

    python -m src.e4_interp              # analysis + figure, from the cache
    env -u PYTHONPATH .venv/bin/python -m src.e4_interp --collect  # re-run model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.teardown import (CARLA_C, HEAD_NAMES, N, REAL_C, SCALARS, WARMUP,
                          WARN_C, _flat, _plt, _post)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "report" / "e4_collected.npz"
FIG = ROOT / "report" / "figures" / "e4_interpolation.png"
RESULTS = ROOT / "report" / "e4_results.md"
SUBARU_HEVC = DATA / "subaru_source" / "fcamera.hevc"
SUBARU_RLOG = DATA / "subaru_source" / "rlog.bz2"
CARLA_NPY = DATA / "domain_gap" / "carla_rgb.npy"

BASE_ALPHAS = [round(0.1 * i, 4) for i in range(11)]  # 0.0, 0.1, ... 1.0
REFINE_GAP = 0.2      # insert a midpoint when adjacent activity differs by > this
REFINE_ROUNDS = 2     # refinement passes after the first sweep
CLIFF_WIDTH = 0.2     # transition width below this reads as a cliff


def blend(real_six: np.ndarray, carla_six: np.ndarray, alpha: float) -> np.ndarray:
    """Convex blend (1-alpha)*real + alpha*carla in float32."""
    r = np.asarray(real_six, dtype=np.float32)
    c = np.asarray(carla_six, dtype=np.float32)
    return (1.0 - alpha) * r + alpha * c


def _crossing(xs: list[float], ys: list[float], level: float) -> float:
    """First x where the piecewise-linear curve (xs, ys) crosses `level`."""
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            x0, x1 = xs[i - 1], xs[i]
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")


def transition_width(alphas, norm_activity: dict) -> tuple[float, float]:
    """Return (alpha@0.9, alpha@0.1): the alphas where normalized activity
    crosses 0.9 and 0.1. The transition width is the difference."""
    xs = sorted(alphas)
    ys = [norm_activity[a] for a in xs]
    return _crossing(xs, ys, 0.9), _crossing(xs, ys, 0.1)


def save_cache(path: Path, collected: dict[float, dict]) -> None:
    """Persist {alpha: {head: array}} as one compressed .npz."""
    flat = {f"{a:.4f}__{k}": v
            for a, d in collected.items() for k, v in d.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_cache(path: Path) -> dict[float, dict]:
    """Inverse of `save_cache`."""
    z = np.load(path)
    out: dict[float, dict] = {}
    for key in z.files:
        a_str, _, name = key.partition("__")
        out.setdefault(float(a_str), {})[name] = z[key]
    return out


def activity_per_head(d: dict) -> dict[str, float]:
    """Per-head temporal activity: sum of per-element std over the frames."""
    return {name: float(_flat(d[name]).std(axis=0).sum())
            for name in SCALARS + HEAD_NAMES}


def normalized_activity(per_head_by_alpha: dict[float, dict]) -> dict[float, float]:
    """Median over heads of (head activity / head activity at alpha 0).
    Equals 1.0 at alpha 0 by construction.  Median is used instead of mean
    to be robust to the 2 heads (pose, meta) that E1 found do NOT collapse on
    CARLA frames; their ratios >> 1 would inflate the arithmetic mean at
    alpha -> 1 and mask the true collapse of the other 8 heads."""
    base = per_head_by_alpha[0.0]
    out: dict[float, float] = {}
    for a, ph in per_head_by_alpha.items():
        ratios = [ph[h] / base[h] for h in base if base[h] > 1e-12]
        out[a] = float(np.median(ratios))
    return out


def feature_centroid(d: dict) -> np.ndarray:
    """Mean hidden_state vector (512,) over the frames."""
    return d["hidden_state"].astype(np.float64).mean(axis=0)


def feature_projection(centroid_by_alpha: dict[float, np.ndarray]) -> dict[float, float]:
    """Project each centroid onto the real->CARLA axis w = mu_c - mu_r:
    f(alpha) = ((mu_a - mu_r) . w) / (w . w). f(0)=0, f(1)=1."""
    mu_r = centroid_by_alpha[0.0]
    mu_c = centroid_by_alpha[1.0]
    w = mu_c - mu_r
    denom = float(w @ w) + 1e-12
    return {a: float((mu - mu_r) @ w / denom)
            for a, mu in centroid_by_alpha.items()}


def feature_spread(d: dict) -> float:
    """Trace of the hidden_state covariance (total variance)."""
    return float(np.var(d["hidden_state"].astype(np.float64), axis=0).sum())


def mean_uncertainty(d: dict) -> float:
    """Mean predicted plan uncertainty over the frames."""
    return float(_flat(d["plan_std"]).mean())


def _collect_live() -> dict[float, dict]:
    """Run supercombo over the blended sequences. Needs supercombo.onnx, the
    Subaru HEVC/rlog, and data/domain_gap/carla_rgb.npy. Imports are local so
    the cached path stays free of the onnxruntime / OpenCV dependency."""
    from src.probe_model import collect, load_carla_six, load_real_six
    from src.state import build_session, load_output_slices

    sess, slices = build_session(), load_output_slices()
    print(f"Loading Subaru + CARLA frame sequences (N={N}) ...")
    real_six = load_real_six(SUBARU_HEVC, SUBARU_RLOG, N)
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


def fig_interp(alphas, norm, fproj, unc, a90, a10) -> None:
    """Two-panel figure: (top) output activity + feature collapse vs alpha,
    (bottom) predicted uncertainty vs alpha."""
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
                    label=f"transition (width {a10 - a90:.2f})")
    ax1.set_ylabel("normalized")
    ax1.set_title("E4  supercombo across the real-to-sim interpolation")
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


def write_results(alphas, norm, fproj, spread, unc,
                  a90, a10, width, verdict) -> None:
    """Write report/e4_results.md: the per-alpha table and the verdict."""
    xs = sorted(alphas)
    L = ["# E4 results: real-to-sim interpolation", "",
         f"Pixel alpha-blend of the Subaru real sequence and the CARLA "
         f"sequence (N={N} frames, {WARMUP} warmup discarded). alpha=0 is the "
         f"real frame, alpha=1 is the CARLA frame.", "",
         f"**Verdict: {verdict}.** Output activity falls from 0.9 to 0.1 of "
         f"the real baseline over alpha {a90:.3f} to {a10:.3f} "
         f"(transition width {width:.3f}; < {CLIFF_WIDTH} reads as a cliff).",
         "",
         "| alpha | output activity | feature collapse | feature spread | plan uncertainty |",
         "|---|---|---|---|---|"]
    for a in xs:
        L.append(f"| {a:.4f} | {norm[a]:.4f} | {fproj[a]:.4f} "
                 f"| {spread[a]:.2f} | {unc[a]:.4f} |")
    RESULTS.write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E4 real-to-sim interpolation sweep")
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

    print(f"\n=== E4  REAL-TO-SIM INTERPOLATION ({len(alphas)} alphas) ===")
    print(f"  {'alpha':>7} {'activity':>10} {'feat collapse':>14} "
          f"{'uncertainty':>12}")
    for a in alphas:
        print(f"  {a:>7.4f} {norm[a]:>10.4f} {fproj[a]:>14.4f} "
              f"{unc[a]:>12.4f}")
    print(f"\n  transition width: alpha {a90:.3f} -> {a10:.3f} "
          f"= {width:.3f}  ->  {verdict.upper()}")

    fig_interp(alphas, norm, fproj, unc, a90, a10)
    write_results(alphas, norm, fproj, spread, unc, a90, a10, width, verdict)
    print(f"  figure -> {FIG.relative_to(ROOT)}   "
          f"results -> {RESULTS.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
