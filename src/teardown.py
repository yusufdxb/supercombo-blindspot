"""Distribution-shift teardown of openpilot v0.9.7 supercombo.

Compares the model on real comma footage vs CARLA renders across three
experiments and writes a results table + figures.

  E1  Output collapse map  — per-head temporal activity, real vs CARLA. A head
      whose frame-to-frame variation collapses is one the model has stopped
      driving from.
  E2  Internal OOD         — the model's own 512-D recurrent feature vector:
      does the CARLA cluster collapse (low spread) and separate from the real
      feature distribution? OOD measured inside the model, not in pixels.
  E3  Confidence response  — the model's predicted uncertainties (plan / lead /
      curvature std). When outputs collapse, does predicted uncertainty rise to
      flag it, and does it rise *enough* to leave the model's normal range?

Methodology notes (these matter):
  * Each segment is warmed from a fresh recurrent state and the first WARMUP
    frames are discarded — the warmup transient is not scene response.
  * Real = Subaru + RAM, collected separately (no domain seam in one warm run),
    post-warmup portions concatenated. Two cars, two drives.
  * E1 aggregates per-element std as sum(CARLA)/sum(real): the ratio of real
    per-element variation that CARLA reproduces, weighted by real activity.

    env -u PYTHONPATH .venv/bin/python -m src.teardown
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG_DIR = ROOT / "report" / "figures"
RESULTS_MD = ROOT / "report" / "teardown_results.md"
# Cached collected model outputs — lets the teardown run from a fresh clone
# with no model, no CARLA, and no multi-GB raw frames. See README "Reproduce".
CACHE = ROOT / "report" / "teardown_collected.npz"

N = 320          # frames collected per segment
WARMUP = 100     # discarded per segment (recurrent state fill, not scene response)
SCALARS = ["accel_t0", "desired_curv", "lead_prob"]
# Heads mapped by E1 — mirrors src.probe_model._HEADS (both are stable
# supercombo output groups; kept in sync by hand).
HEAD_NAMES = ["plan", "lane_lines", "road_edges", "lead", "pose",
              "desire_state", "meta"]
COLLAPSE = 0.10  # head activity ratio below this = collapsed


def _post(d: dict, warmup: int) -> dict:
    return {k: v[warmup:] for k, v in d.items()}


def _flat(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(len(arr), -1)


# --------------------------------------------------------------------------
# E1 — output collapse map
# --------------------------------------------------------------------------

def e1_collapse_map(real: dict, carla: dict) -> list[dict]:
    rows = []
    for name in SCALARS + HEAD_NAMES:
        r = _flat(real[name])
        c = _flat(carla[name])
        rstd = r.std(axis=0)
        cstd = c.std(axis=0)
        ratio = float(cstd.sum() / rstd.sum()) if rstd.sum() > 1e-12 else float("nan")
        active = rstd > (0.05 * rstd.max() + 1e-9)  # elements that move on real data
        per_elem = cstd[active] / rstd[active]
        collapsed_frac = float(np.mean(per_elem < COLLAPSE)) if active.any() else float("nan")
        rows.append({"head": name, "real": float(rstd.sum()), "carla": float(cstd.sum()),
                     "ratio": ratio, "collapsed_frac": collapsed_frac})
    return rows


# --------------------------------------------------------------------------
# E2 — internal feature-space OOD
# --------------------------------------------------------------------------

def e2_feature_ood(real: dict, carla: dict) -> dict:
    R = real["hidden_state"].astype(np.float64)
    C = carla["hidden_state"].astype(np.float64)
    mu_r, mu_c = R.mean(0), C.mean(0)

    # spread = total variance (trace of covariance)
    spread_r = float(np.var(R, axis=0).sum())
    spread_c = float(np.var(C, axis=0).sum())

    # separability along the centroid-difference direction
    w = mu_c - mu_r
    w /= (np.linalg.norm(w) + 1e-12)
    pr, pc = R @ w, C @ w
    dprime = float(abs(pc.mean() - pr.mean())
                   / np.sqrt(0.5 * (pr.var() + pc.var()) + 1e-12))
    thr = 0.5 * (pr.mean() + pc.mean())
    flip = 1 if pc.mean() > pr.mean() else -1
    acc = float((np.mean((pr * flip) < (thr * flip))
                 + np.mean((pc * flip) >= (thr * flip))) / 2)

    # PCA (fit on real) for the scatter figure
    Rc = R - mu_r
    _, _, Vt = np.linalg.svd(Rc, full_matrices=False)
    basis = Vt[:2]
    return {
        "spread_real": spread_r, "spread_carla": spread_c,
        "spread_ratio": spread_c / spread_r if spread_r > 1e-12 else float("nan"),
        "dprime": dprime, "separability": acc,
        "pca_real": Rc @ basis.T, "pca_carla": (C - mu_r) @ basis.T,
    }


# --------------------------------------------------------------------------
# E3 — confidence response
# --------------------------------------------------------------------------

def e3_confidence(real: dict, carla: dict) -> list[dict]:
    rows = []
    for label, out_key, std_key in [("plan", "plan", "plan_std"),
                                    ("lead", "lead", "lead_std"),
                                    ("desired_curv", "desired_curv", "desired_curv_std")]:
        ro, co = _flat(real[out_key]), _flat(carla[out_key])
        out_ratio = float(co.std(0).sum() / ro.std(0).sum())
        # predicted uncertainty: per-frame mean std
        ru = _flat(real[std_key]).mean(1)
        cu = _flat(carla[std_key]).mean(1)
        # does CARLA's predicted uncertainty leave the model's normal real range?
        real_p95 = float(np.percentile(ru, 95))
        frac_above = float(np.mean(cu > real_p95))
        rows.append({
            "head": label, "out_ratio": out_ratio,
            "unc_real": float(ru.mean()), "unc_carla": float(cu.mean()),
            "unc_ratio": float(cu.mean() / ru.mean()) if ru.mean() > 1e-12 else float("nan"),
            "carla_above_real_p95": frac_above,
        })
    return rows


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    return plt


# physx-newton-bench editorial palette: blue primary, rust + amber secondaries
REAL_C, CARLA_C, WARN_C = "#2a78d6", "#1baf7a", "#52514e"
# light chrome for legends/boxes on the cream canvas
LEGEND_FC, LEGEND_EC, MUTED_C, INK_C = "#ffffff", "#c3c2b7", "#898781", "#0b0b0b"


def fig_collapse(rows: list[dict]) -> None:
    plt = _plt()
    rows = sorted([r for r in rows if np.isfinite(r["ratio"])], key=lambda r: r["ratio"])
    names = [r["head"] for r in rows]
    ratios = [max(r["ratio"], 1e-4) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.barh(names, ratios, color=[CARLA_C if r < COLLAPSE else "#898781" for r in ratios])
    ax.set_xscale("log")
    ax.axvline(COLLAPSE, color=WARN_C, ls="--", lw=1.2, label=f"collapse threshold ({COLLAPSE})")
    ax.axvline(1.0, color="#898781", ls=":", lw=1.0, label="parity with real")
    ax.set_xlabel("CARLA output activity / real output activity  (log scale)")
    ax.set_title("E1  supercombo output activity on CARLA vs real footage")
    ax.legend(loc="lower right", facecolor="#ffffff", edgecolor="#c3c2b7")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e1_head_collapse.png", dpi=150)
    plt.close(fig)


def fig_features(e2: dict) -> None:
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
    ax.set_xlabel("PC1 of supercombo hidden_state")
    ax.set_ylabel("PC2 of supercombo hidden_state")
    ax.set_title("E2  CARLA in the model's own 512-D feature space\n"
                 f"feature spread {e2['spread_ratio']:.4f}x of real   "
                 f"separability {100*e2['separability']:.0f}%   d'={e2['dprime']:.1f}")
    ax.legend(facecolor="#ffffff", edgecolor="#c3c2b7")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "e2_feature_ood.png", dpi=150)
    plt.close(fig)


def fig_confidence(e3: list[dict]) -> None:
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
    ax.set_title("E3  outputs collapse ~99%, the model's uncertainty flags none of it\n"
                 "silent failure: confidently blind")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=1,
              facecolor="#ffffff", edgecolor="#c3c2b7")
    fig.subplots_adjust(bottom=0.24)
    fig.savefig(FIG_DIR / "e3_confidence.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# collected-output cache
# --------------------------------------------------------------------------

def _save_cache(path: Path, segments: dict[str, dict]) -> None:
    """Persist the per-segment collected outputs as one compressed .npz."""
    flat = {f"{seg}__{k}": v for seg, d in segments.items() for k, v in d.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def _load_cache(path: Path) -> dict[str, dict]:
    """Inverse of `_save_cache` — return {segment: {head: array}}."""
    z = np.load(path)
    segs: dict[str, dict] = {}
    for key in z.files:
        seg, _, name = key.partition("__")
        segs.setdefault(seg, {})[name] = z[key]
    return segs


def _collect_live() -> dict[str, dict]:
    """Run supercombo over the real + CARLA segments. Needs the model, the real
    HEVC/rlog segments, and data/domain_gap/carla_rgb.npy. Imports are local so
    the cached path stays free of the onnxruntime / OpenCV dependency."""
    from src.probe_model import collect, load_carla_six, load_real_six
    from src.state import build_session, load_output_slices

    sess, slices = build_session(), load_output_slices()
    print(f"Collecting (N={N}/segment) — first warm triggers ~28 s PTX JIT ...")
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


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="supercombo distribution-shift teardown")
    ap.add_argument("--collect", action="store_true",
                    help="re-run the model over the raw segments instead of "
                         f"using the cached collected outputs ({CACHE.name})")
    args = ap.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    if args.collect or not CACHE.exists():
        segments = _collect_live()
        _save_cache(CACHE, segments)
        print(f"  cached collected outputs -> {CACHE.relative_to(ROOT)}")
    else:
        print(f"Loading collected outputs from {CACHE.relative_to(ROOT)} — no "
              f"model / CARLA / raw frames needed (pass --collect to re-run).")
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

    print("\n=== E1  OUTPUT COLLAPSE MAP (post-warmup temporal activity) ===")
    print(f"  {'head':<16} {'real':>11} {'CARLA':>11} {'CARLA/real':>11} "
          f"{'collapsed':>10}  state")
    for r in e1:
        st = "COLLAPSED" if r["ratio"] < COLLAPSE else "alive"
        print(f"  {r['head']:<16} {r['real']:>11.4f} {r['carla']:>11.4f} "
              f"{r['ratio']:>11.4f} {100*r['collapsed_frac']:>9.0f}%  {st}")

    print("\n=== E2  INTERNAL FEATURE-SPACE OOD (supercombo hidden_state) ===")
    print(f"  CARLA feature spread : {e2['spread_ratio']:.5f}x of real "
          f"({'collapsed cluster' if e2['spread_ratio'] < 0.5 else 'comparable spread'})")
    print(f"  real vs CARLA        : separability {100*e2['separability']:.1f}%  "
          f"d'={e2['dprime']:.2f}  ({'cleanly separated -> OOD' if e2['dprime'] > 2 else 'overlapping'})")

    print("\n=== E3  CONFIDENCE RESPONSE (output collapse vs predicted uncertainty) ===")
    print(f"  {'head':<14} {'out retained':>13} {'unc real':>10} {'unc CARLA':>11} "
          f"{'unc ratio':>10} {'CARLA>real p95':>15}")
    for r in e3:
        print(f"  {r['head']:<14} {100*r['out_ratio']:>12.1f}% {r['unc_real']:>10.4f} "
              f"{r['unc_carla']:>11.4f} {r['unc_ratio']:>10.2f}x "
              f"{100*r['carla_above_real_p95']:>14.0f}%")

    fig_collapse(e1)
    fig_features(e2)
    fig_confidence(e3)
    _write_results_md(e1, e2, e3)
    print(f"\nfigures -> {FIG_DIR}/   results -> {RESULTS_MD}")

    collapsed = [r["head"] for r in e1 if r["ratio"] < COLLAPSE]
    print(f"\n=== VERDICT ===")
    print(f"  {len(collapsed)}/{len(e1)} output heads collapse on CARLA: {collapsed}")
    print(f"  feature space: CARLA spread {e2['spread_ratio']:.2f}x real, "
          f"{100*e2['separability']:.0f}% separable")
    return 0


def _write_results_md(e1, e2, e3) -> None:
    L = ["# Teardown results", "",
         f"Real: Subaru + RAM segments (320 frames captured, {WARMUP} warmup discarded; "
         f"319 stored after pair-processing, 219 analysis frames). "
         f"CARLA: 319 stored clean-road frames. supercombo openpilot v0.9.7.", "",
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


if __name__ == "__main__":
    sys.exit(main())
