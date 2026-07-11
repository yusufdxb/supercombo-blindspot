"""Distribution-shift teardown of openpilot v0.9.6 supercombo.

Identical methodology to src/teardown.py (v0.9.7) applied to the v0.9.6
model. All outputs are written to _v096-suffixed files; no v0.9.7 file
is touched.

  env -u PYTHONPATH .venv/bin/python -m src.teardown_v096 --collect
  env -u PYTHONPATH .venv/bin/python -m src.teardown_v096          # from cache
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.teardown import (
    COLLAPSE, HEAD_NAMES, N, SCALARS, WARMUP, WARN_C, CARLA_C, REAL_C,
    _flat, _plt, _post,
    e1_collapse_map, e2_feature_ood, e3_confidence,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG_DIR = ROOT / "report" / "figures"
RESULTS_MD = ROOT / "report" / "teardown_v096_results.md"
CACHE = ROOT / "report" / "teardown_v096_collected.npz"
MODEL_PATH = ROOT / "models" / "supercombo_v096.onnx"


def _save_cache(path: Path, segments: dict[str, dict]) -> None:
    flat = {f"{seg}__{k}": v for seg, d in segments.items() for k, v in d.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def _load_cache(path: Path) -> dict[str, dict]:
    z = np.load(path)
    segs: dict[str, dict] = {}
    for key in z.files:
        seg, _, name = key.partition("__")
        segs.setdefault(seg, {})[name] = z[key]
    return segs


def _collect_live() -> dict[str, dict]:
    from src.probe_model import collect, load_carla_six, load_real_six
    from src.state import build_session, load_output_slices

    sess = build_session(MODEL_PATH)
    slices = load_output_slices(MODEL_PATH)
    providers = sess.get_providers()
    print(f"v0.9.6 session providers: {providers}")
    print(f"Collecting v0.9.6 (N={N}/segment) -- first warm triggers ~28 s PTX JIT ...")
    return {
        "subaru": collect(load_real_six(DATA / "subaru_source" / "fcamera.hevc",
                                        DATA / "subaru_source" / "rlog.bz2", N),
                          sess, slices),
        "ram": collect(load_real_six(DATA / "ram_source" / "fcamera.hevc",
                                     DATA / "ram_source" / "rlog.bz2", N),
                       sess, slices),
        "carla": collect(load_carla_six(DATA / "domain_gap" / "carla_rgb.npy", N),
                         sess, slices),
    }


def fig_collapse_v096(rows: list[dict]) -> None:
    plt = _plt()
    rows = sorted([r for r in rows if np.isfinite(r["ratio"])], key=lambda r: r["ratio"])
    names = [r["head"] for r in rows]
    ratios = [max(r["ratio"], 1e-4) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(names, ratios, color=[CARLA_C if r < COLLAPSE else "#898781" for r in ratios])
    ax.set_xscale("log")
    ax.axvline(COLLAPSE, color=WARN_C, ls="--", lw=1.2,
               label=f"collapse threshold ({COLLAPSE})")
    ax.axvline(1.0, color="#898781", ls=":", lw=1.0, label="parity with real")
    ax.set_xlabel("CARLA output activity / real output activity  (log scale)")
    ax.set_title("E1 v0.9.6  supercombo output activity on CARLA vs real footage")
    ax.legend(loc="lower right", facecolor="#ffffff", edgecolor="#c3c2b7")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_head_collapse_v096.png", dpi=150)
    plt.close(fig)


def fig_features_v096(e2: dict) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.scatter(*e2["pca_real"].T, s=20, c=REAL_C, alpha=0.75,
               label=f"real comma footage (n={len(e2['pca_real'])})", edgecolors="none")
    ax.scatter(*e2["pca_carla"].T, s=40, c=CARLA_C, alpha=0.9,
               label=f"CARLA renders (n={len(e2['pca_carla'])})", edgecolors="none")
    cx, cy = e2["pca_carla"].mean(0)
    ax.annotate(f"{len(e2['pca_carla'])} CARLA frames\ncollapse to one point",
                xy=(cx, cy), xytext=(cx + 0.15, cy + 0.18), color=CARLA_C, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=CARLA_C))
    ax.set_xlabel("PC1 of supercombo hidden_state (v0.9.6)")
    ax.set_ylabel("PC2 of supercombo hidden_state (v0.9.6)")
    ax.set_title("E2 v0.9.6  CARLA in the model's own 512-D feature space\n"
                 f"feature spread {e2['spread_ratio']:.4f}x of real   "
                 f"separability {100*e2['separability']:.0f}%   d'={e2['dprime']:.1f}")
    ax.legend(facecolor="#ffffff", edgecolor="#c3c2b7")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e2_feature_ood_v096.png", dpi=150)
    plt.close(fig)


def fig_confidence_v096(e3: list[dict]) -> None:
    plt = _plt()
    heads = [r["head"] for r in e3]
    lost = [100 * (1 - r["out_ratio"]) for r in e3]
    flagged = [100 * r["carla_above_real_p95"] for r in e3]
    x = np.arange(len(heads))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.8))
    b1 = ax.bar(x - w / 2, lost, w, color=CARLA_C,
                label="output activity lost on CARLA")
    b2 = ax.bar(x + w / 2, flagged, w, color=REAL_C,
                label="CARLA frames the model flags abnormal (uncertainty > real p95)")
    ax.bar_label(b1, fmt="%.1f%%", padding=3, color="#0b0b0b", fontsize=10)
    ax.bar_label(b2, fmt="%.0f%%", padding=3, color="#0b0b0b", fontsize=10)
    ax.set_ylim(0, 112)
    ax.set_xticks(x, heads)
    ax.set_ylabel("percent")
    ax.set_title("E3 v0.9.6  outputs vs uncertainty on CARLA")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1,
              facecolor="#ffffff", edgecolor="#c3c2b7")
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(FIG_DIR / "e3_confidence_v096.png", dpi=150)
    plt.close(fig)


def _write_results_md(e1, e2, e3) -> None:
    L = ["# Teardown results (v0.9.6)", "",
         f"Real: Subaru + RAM segments (320 frames captured, {WARMUP} warmup discarded; "
         f"319 stored after pair-processing, 219 analysis frames). "
         f"CARLA: 319 stored clean-road frames. supercombo openpilot v0.9.6.", "",
         "## E1  output collapse map", "",
         "| head | real activity | CARLA activity | CARLA/real | collapsed elems | state |",
         "|---|---|---|---|---|---|"]
    for r in e1:
        st = "**COLLAPSED**" if r["ratio"] < COLLAPSE else "alive"
        L.append(f"| {r['head']} | {r['real']:.4f} | {r['carla']:.4f} | {r['ratio']:.4f} "
                 f"| {100*r['collapsed_frac']:.0f}% | {st} |")
    L += ["", "## E2  internal feature-space OOD", "",
          f"- CARLA feature spread is **{e2['spread_ratio']:.5f}x** the real spread "
          f"(trace of `hidden_state` covariance).",
          f"- real vs CARLA separability **{100*e2['separability']:.1f}%**, "
          f"d' = **{e2['dprime']:.2f}** along the centroid-difference direction.", "",
          "## E3  confidence response", "",
          "| head | output retained | pred. unc. real | pred. unc. CARLA | "
          "unc. ratio | CARLA above real p95 |", "|---|---|---|---|---|---|"]
    for r in e3:
        L.append(f"| {r['head']} | {100*r['out_ratio']:.1f}% | {r['unc_real']:.4f} | "
                 f"{r['unc_carla']:.4f} | {r['unc_ratio']:.2f}x | "
                 f"{100*r['carla_above_real_p95']:.0f}% |")
    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(L) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="v0.9.6 supercombo distribution-shift teardown")
    ap.add_argument("--collect", action="store_true",
                    help="re-run the v0.9.6 model over the raw segments instead of "
                         f"using the cached collected outputs ({CACHE.name})")
    args = ap.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.collect or not CACHE.exists():
        segments = _collect_live()
        _save_cache(CACHE, segments)
        print(f"  cached v0.9.6 collected outputs -> {CACHE.relative_to(ROOT)}")
    else:
        print(f"Loading v0.9.6 collected outputs from {CACHE.relative_to(ROOT)} -- "
              f"pass --collect to re-run.")
        segments = _load_cache(CACHE)

    subaru = _post(segments["subaru"], WARMUP)
    ram = _post(segments["ram"], WARMUP)
    carla = _post(segments["carla"], WARMUP)
    real = {k: np.concatenate([subaru[k], ram[k]]) for k in subaru}
    print(f"  post-warmup: real {len(real['accel_t0'])} (Subaru+RAM), "
          f"CARLA {len(carla['accel_t0'])}")

    e1 = e1_collapse_map(real, carla)
    e2 = e2_feature_ood(real, carla)
    e3 = e3_confidence(real, carla)

    print("\n=== E1  OUTPUT COLLAPSE MAP v0.9.6 (post-warmup temporal activity) ===")
    print(f"  {'head':<16} {'real':>11} {'CARLA':>11} {'CARLA/real':>11} "
          f"{'collapsed':>10}  state")
    for r in e1:
        st = "COLLAPSED" if r["ratio"] < COLLAPSE else "alive"
        print(f"  {r['head']:<16} {r['real']:>11.4f} {r['carla']:>11.4f} "
              f"{r['ratio']:>11.4f} {100*r['collapsed_frac']:>9.0f}%  {st}")

    print("\n=== E2  INTERNAL FEATURE-SPACE OOD v0.9.6 (supercombo hidden_state) ===")
    print(f"  CARLA feature spread : {e2['spread_ratio']:.5f}x of real "
          f"({'collapsed cluster' if e2['spread_ratio'] < 0.5 else 'comparable spread'})")
    print(f"  real vs CARLA        : separability {100*e2['separability']:.1f}%  "
          f"d'={e2['dprime']:.2f}  ({'cleanly separated -> OOD' if e2['dprime'] > 2 else 'overlapping'})")

    print("\n=== E3  CONFIDENCE RESPONSE v0.9.6 (output collapse vs predicted uncertainty) ===")
    print(f"  {'head':<14} {'out retained':>13} {'unc real':>10} {'unc CARLA':>11} "
          f"{'unc ratio':>10} {'CARLA>real p95':>15}")
    for r in e3:
        print(f"  {r['head']:<14} {100*r['out_ratio']:>12.1f}% {r['unc_real']:>10.4f} "
              f"{r['unc_carla']:>11.4f} {r['unc_ratio']:>10.2f}x "
              f"{100*r['carla_above_real_p95']:>14.0f}%")

    fig_collapse_v096(e1)
    fig_features_v096(e2)
    fig_confidence_v096(e3)
    _write_results_md(e1, e2, e3)
    print(f"\nfigures -> {FIG_DIR}/   results -> {RESULTS_MD}")

    collapsed = [r["head"] for r in e1 if r["ratio"] < COLLAPSE]
    print(f"\n=== VERDICT v0.9.6 ===")
    print(f"  {len(collapsed)}/{len(e1)} output heads collapse on CARLA: {collapsed}")
    print(f"  feature space: CARLA spread {e2['spread_ratio']:.2f}x real, "
          f"{100*e2['separability']:.0f}% separable")
    return 0


if __name__ == "__main__":
    sys.exit(main())
