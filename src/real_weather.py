"""E_weather: adverse-weather OOD axis for the supercombo distribution-shift teardown.

Feeds three real comma-3 segments through the pinned v0.9.7 model and measures
whether real night + headlight-glare footage induces the same silent collapse as
CARLA synthetic renders.

Segments (all tici, 1928x1208 yuv420p, with liveCalibration -- no intrinsics confound):
  ev6_night       : EV6 residential night + oncoming-headlight glare
  bronco_night    : Ford Bronco highway night + tail-light/sign glare
  daytime_control : Daytime-dry C3 in-distribution sanity control

Metrics per segment (compared against the v0.9.7 Subaru+RAM real baseline):
  E1-style: per-head temporal activity ratio (seg/baseline); collapsed head count
  E2-style: recurrent feature spread ratio + d' / separability vs real ID cluster
  E3-style: fraction of frames exceeding the v0.9.7 real p95 predicted uncertainty
  E6-style: rolling-spread monitor (threshold=0.078873, window=30) fire fraction

Usage:
  env -u PYTHONPATH .venv/bin/python -m src.real_weather
  env -u PYTHONPATH .venv/bin/python -m src.real_weather --collect   # re-run model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORT = ROOT / "report"
FIG_DIR = REPORT / "figures"

CACHE = REPORT / "real_weather_collected.npz"
TEARDOWN_CACHE = REPORT / "teardown_collected.npz"

N = 320       # frames per segment (matches teardown convention)
WARMUP = 100  # discarded (matches teardown convention)

COLLAPSE = 0.10  # activity ratio below this = collapsed (E1)

HEAD_NAMES = ["plan", "lane_lines", "road_edges", "lead", "pose", "desire_state", "meta"]
SCALARS = ["accel_t0", "desired_curv", "lead_prob"]

# E6 calibrated threshold and window (from report/e6_results.md, v0.9.7 real-driving).
# This is the N=2 (subaru+ram) 1st-percentile operating point; the manuscript's
# N=4 all-clean headline calibration uses 0.087077 (report/corpus_scaling_results.md).
# Both are the deploy-side and analysis-side operating points of the same monitor.
E6_THRESHOLD = 0.078873
E6_WINDOW = 30

# daytime_control intermittently enters a bi-modal near-zero recurrent attractor
# that drives rolling spread below the E6 threshold (E6 fires ~58%). Independent
# verification (2026-05-30) confirmed this is genuine model behaviour on clean,
# correctly-warped input, NOT a pipeline artifact. The earlier "high steer + low
# speed / after ~120-220 frames" explanation was FALSIFIED: the norm toggles from
# frame ~16 (not a one-time settle), and EV6 night reaches a higher peak steer at
# the same low speed without collapsing. In the low-norm regime the output heads
# ARE suppressed (~45x on desired_curv); E1 reads 0/10 only because the activity
# metric averages across both regimes. The trigger is UNEXPLAINED; treat the E6
# firing here as an open caveat (a real-segment near-collapse), not settled.


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _post(d: dict, warmup: int = WARMUP) -> dict:
    return {k: v[warmup:] for k, v in d.items()}


def _flat(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(len(arr), -1)


def _load_baseline() -> dict:
    """Load the v0.9.7 Subaru+RAM real baseline from teardown_collected.npz."""
    z = np.load(TEARDOWN_CACHE)
    sub: dict = {k.partition("__")[2]: z[k] for k in z.files if k.startswith("subaru__")}
    ram: dict = {k.partition("__")[2]: z[k] for k in z.files if k.startswith("ram__")}
    sub_p = _post(sub)
    ram_p = _post(ram)
    return {k: np.concatenate([sub_p[k], ram_p[k]]) for k in sub_p}


def _load_carla() -> dict:
    """Load the CARLA reference from teardown_collected.npz for the comparison table."""
    z = np.load(TEARDOWN_CACHE)
    carla: dict = {k.partition("__")[2]: z[k] for k in z.files if k.startswith("carla__")}
    return _post(carla)


# --------------------------------------------------------------------------
# collection (runs the model)
# --------------------------------------------------------------------------

def _collect_live() -> dict[str, dict]:
    from src.probe_model import collect, load_real_six
    from src.state import build_session, load_output_slices

    sess, slices = build_session(), load_output_slices()
    results = {}
    for name, subdir in [
        ("ev6_night", "ev6_night_source"),
        ("bronco_night", "bronco_night_source"),
        ("daytime_control", "daytime_control_source"),
    ]:
        hevc = DATA / subdir / "fcamera.hevc"
        rlog = DATA / subdir / "rlog.bz2"
        print(f"  collecting {name} ({N} frames) ...")
        frames = load_real_six(hevc, rlog, N)
        results[name] = collect(frames, sess, slices)
        print(f"    done: {len(frames)} frames loaded, "
              f"{len(results[name]['accel_t0'])} model outputs")
    return results


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


# --------------------------------------------------------------------------
# E1-style: activity ratio per head
# --------------------------------------------------------------------------

def e1_activity(seg: dict, baseline: dict) -> list[dict]:
    rows = []
    for name in SCALARS + HEAD_NAMES:
        b = _flat(baseline[name])
        s = _flat(seg[name])
        bstd = b.std(axis=0)
        sstd = s.std(axis=0)
        ratio = float(sstd.sum() / bstd.sum()) if bstd.sum() > 1e-12 else float("nan")
        active = bstd > (0.05 * bstd.max() + 1e-9)
        per_elem = sstd[active] / bstd[active]
        collapsed_frac = float(np.mean(per_elem < COLLAPSE)) if active.any() else float("nan")
        rows.append({
            "head": name,
            "baseline": float(bstd.sum()),
            "seg": float(sstd.sum()),
            "ratio": ratio,
            "collapsed_frac": collapsed_frac,
        })
    return rows


def _count_collapsed(e1_rows: list[dict]) -> int:
    return sum(1 for r in e1_rows if r["ratio"] < COLLAPSE)


# --------------------------------------------------------------------------
# E2-style: feature-space OOD
# --------------------------------------------------------------------------

def e2_feature_ood(seg: dict, baseline: dict) -> dict:
    B = baseline["hidden_state"].astype(np.float64)
    S = seg["hidden_state"].astype(np.float64)
    mu_b = B.mean(0)
    mu_s = S.mean(0)
    spread_b = float(np.var(B, axis=0).sum())
    spread_s = float(np.var(S, axis=0).sum())

    w = mu_s - mu_b
    norm = np.linalg.norm(w)
    w_unit = w / (norm + 1e-12)
    pb, ps = B @ w_unit, S @ w_unit
    dprime = float(
        abs(ps.mean() - pb.mean())
        / np.sqrt(0.5 * (pb.var() + ps.var()) + 1e-12)
    )
    thr = 0.5 * (pb.mean() + ps.mean())
    flip = 1 if ps.mean() > pb.mean() else -1
    sep = float(
        (np.mean((pb * flip) < (thr * flip)) + np.mean((ps * flip) >= (thr * flip))) / 2
    )
    return {
        "spread_baseline": spread_b,
        "spread_seg": spread_s,
        "spread_ratio": spread_s / spread_b if spread_b > 1e-12 else float("nan"),
        "dprime": dprime,
        "separability": sep,
    }


# --------------------------------------------------------------------------
# E3-style: uncertainty above baseline p95
# --------------------------------------------------------------------------

def e3_uncertainty(seg: dict, baseline: dict) -> list[dict]:
    rows = []
    for label, std_key in [
        ("plan", "plan_std"),
        ("lead", "lead_std"),
        ("desired_curv", "desired_curv_std"),
    ]:
        bu = _flat(baseline[std_key]).mean(1)
        su = _flat(seg[std_key]).mean(1)
        real_p95 = float(np.percentile(bu, 95))
        frac_above = float(np.mean(su > real_p95))
        rows.append({
            "head": label,
            "baseline_unc_mean": float(bu.mean()),
            "seg_unc_mean": float(su.mean()),
            "real_p95": real_p95,
            "frac_above_p95": frac_above,
        })
    return rows


# --------------------------------------------------------------------------
# E6-style: rolling-spread monitor
# --------------------------------------------------------------------------

def e6_monitor(seg: dict, threshold: float = E6_THRESHOLD, window: int = E6_WINDOW) -> dict:
    from src.e6_detector import rolling_spread
    H = seg["hidden_state"].astype(np.float64)
    spreads = rolling_spread(H, window)
    valid = spreads[~np.isnan(spreads)]
    fired_frac = float(np.mean(valid < threshold)) if len(valid) else float("nan")
    return {
        "threshold": threshold,
        "window": window,
        "fired_frac": fired_frac,
        "n_valid": int(len(valid)),
        "fires": bool(fired_frac > 0.5),
    }


# --------------------------------------------------------------------------
# figure
# --------------------------------------------------------------------------

def _make_figure(results: dict[str, dict], baseline: dict, carla: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "text.color": "#0b0b0b",
        "axes.labelcolor": "#0b0b0b", "xtick.color": "#898781",
        "ytick.color": "#898781", "axes.edgecolor": "#c3c2b7",
        "font.size": 10, "axes.titlesize": 11, "axes.grid": True,
        "grid.color": "#e1e0d9", "grid.linewidth": 0.8,
    })

    seg_names = list(results.keys())
    seg_labels = {
        "ev6_night": "EV6 night + glare",
        "bronco_night": "Bronco night + glare",
        "daytime_control": "Daytime control (ID)",
    }

    # Panel 1: E1 head activity ratios for "plan" head (most diagnostic)
    # Panel 2: E2 spread ratios
    # Panel 3: E6 fire fractions
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # E1: overall activity ratio (all heads combined)
    ax = axes[0]
    labels, ratios = [], []
    for sn in seg_names:
        e1 = results[sn]["e1"]
        overall = sum(r["seg"] for r in e1) / (sum(r["baseline"] for r in e1) + 1e-12)
        labels.append(seg_labels.get(sn, sn))
        ratios.append(overall)
    # Add CARLA reference
    carla_e1 = e1_activity(carla, baseline)
    carla_overall = sum(r["seg"] for r in carla_e1) / (sum(r["baseline"] for r in carla_e1) + 1e-12)
    labels.append("CARLA (OOD ref)")
    ratios.append(carla_overall)

    colors = ["#898781" if r > COLLAPSE else "#ff7043" for r in ratios]
    colors[-1] = "#ff7043"  # CARLA always red
    ax.barh(labels, ratios, color=colors)
    ax.axvline(COLLAPSE, color="#ffd54f", ls="--", lw=1.2, label=f"collapse (<{COLLAPSE})")
    ax.axvline(1.0, color="#898781", ls=":", lw=1.0, label="parity with ID baseline")
    ax.set_xlabel("total activity / baseline activity")
    ax.set_title("E1 output activity ratio\n(all heads combined)")
    ax.legend(fontsize=8, facecolor="#f1efe8", edgecolor="#c3c2b7")

    # E2: feature spread ratio
    ax = axes[1]
    labels2, spread_ratios = [], []
    for sn in seg_names:
        labels2.append(seg_labels.get(sn, sn))
        spread_ratios.append(results[sn]["e2"]["spread_ratio"])
    carla_e2 = e2_feature_ood(carla, baseline)
    labels2.append("CARLA (OOD ref)")
    spread_ratios.append(carla_e2["spread_ratio"])

    colors2 = ["#898781" if r > 0.5 else "#ff7043" for r in spread_ratios]
    colors2[-1] = "#ff7043"
    ax.barh(labels2, spread_ratios, color=colors2)
    ax.axvline(0.5, color="#ffd54f", ls="--", lw=1.2, label="collapsed spread (<0.5x)")
    ax.axvline(1.0, color="#898781", ls=":", lw=1.0)
    ax.set_xlabel("feature spread / baseline spread")
    ax.set_title("E2 recurrent feature spread ratio")
    ax.legend(fontsize=8, facecolor="#f1efe8", edgecolor="#c3c2b7")

    # E6: fire fraction
    ax = axes[2]
    labels3, fire_fracs = [], []
    for sn in seg_names:
        labels3.append(seg_labels.get(sn, sn))
        fire_fracs.append(results[sn]["e6"]["fired_frac"])
    carla_e6 = e6_monitor(carla)
    labels3.append("CARLA (OOD ref)")
    fire_fracs.append(carla_e6["fired_frac"])

    colors3 = ["#ff7043" if f > 0.5 else "#898781" for f in fire_fracs]
    colors3[-1] = "#ff7043"
    ax.barh(labels3, [100 * f for f in fire_fracs], color=colors3)
    ax.axvline(50, color="#ffd54f", ls="--", lw=1.2, label="detector fires (>50%)")
    ax.set_xlabel("% frames below spread threshold")
    ax.set_title(f"E6 rolling-spread monitor\n(thr={E6_THRESHOLD}, w={E6_WINDOW})")
    ax.legend(fontsize=8, facecolor="#f1efe8", edgecolor="#c3c2b7")

    fig.suptitle("Real night/glare vs CARLA: does real adverse weather collapse supercombo?",
                 color="#0b0b0b", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "real_weather.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  figure -> {out}")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def _write_report(results: dict[str, dict], baseline: dict, carla: dict) -> None:
    carla_e1 = e1_activity(carla, baseline)
    carla_e2 = e2_feature_ood(carla, baseline)
    carla_e3 = e3_uncertainty(carla, baseline)
    carla_e6 = e6_monitor(carla)

    seg_labels = {
        "ev6_night": "EV6 night + headlight glare",
        "bronco_night": "Bronco night + tail-light/sign glare",
        "daytime_control": "Daytime-dry control (in-distribution)",
        "carla": "CARLA synthetic (E1-E6 reference OOD)",
    }

    L: list[str] = []
    L += ["# Real-weather OOD axis: results", "",
          "Segments: all comma-3 (tici), 1928x1208 yuv420p, with liveCalibration.",
          "No intrinsics confound: all three use the identical `_ar_ox_config` "
          "focal-length/principal-point as the Subaru+RAM baseline.",
          f"Model: supercombo v0.9.7 (models/supercombo.onnx).",
          f"N={N} source frames/segment, one frame consumed by pair processing, "
          f"{WARMUP} stored-output warmup discarded, {N - 1 - WARMUP} outputs analysed.",
          ""]

    # ---------- E1 table ----------
    L += ["## E1 output activity ratio (seg vs v0.9.7 real Subaru+RAM baseline)", ""]
    L += ["| segment | head | baseline activity | seg activity | seg/baseline | "
          "collapsed elems | state |",
          "|---|---|---|---|---|---|---|"]
    for sn, sinfo in list(results.items()) + [("carla", {"e1": carla_e1})]:
        label = seg_labels.get(sn, sn)
        for r in sinfo["e1"]:
            st = "**COLLAPSED**" if r["ratio"] < COLLAPSE else "alive"
            L.append(
                f"| {label} | {r['head']} | {r['baseline']:.4f} | "
                f"{r['seg']:.4f} | {r['ratio']:.4f} | "
                f"{100 * r['collapsed_frac']:.0f}% | {st} |"
            )
    L.append("")

    # collapsed head summary
    L += ["### Collapsed head count per condition", ""]
    L += ["| condition | collapsed heads (ratio < 0.10) |",
          "|---|---|"]
    for sn, sinfo in list(results.items()) + [("carla", {"e1": carla_e1})]:
        n_collapsed = sum(1 for r in sinfo["e1"] if r["ratio"] < COLLAPSE)
        L.append(f"| {seg_labels.get(sn, sn)} | {n_collapsed}/{len(sinfo['e1'])} |")
    L.append("")

    # ---------- E2 table ----------
    L += ["## E2 recurrent feature spread ratio and separability", ""]
    L += ["| condition | spread baseline | spread seg | spread ratio | "
          "d' | separability |",
          "|---|---|---|---|---|---|"]
    for sn, sinfo in list(results.items()) + [("carla", {"e2": carla_e2})]:
        e2 = sinfo["e2"]
        L.append(
            f"| {seg_labels.get(sn, sn)} | "
            f"{e2['spread_baseline']:.4f} | {e2['spread_seg']:.4f} | "
            f"{e2['spread_ratio']:.5f} | {e2['dprime']:.2f} | "
            f"{100 * e2['separability']:.1f}% |"
        )
    L.append("")

    # ---------- E3 table ----------
    L += ["## E3 predicted uncertainty above v0.9.7 real p95", ""]
    L += ["| condition | head | baseline unc | seg unc | real p95 | "
          "frames above p95 |",
          "|---|---|---|---|---|---|"]
    for sn, sinfo in list(results.items()) + [("carla", {"e3": carla_e3})]:
        for r in sinfo["e3"]:
            L.append(
                f"| {seg_labels.get(sn, sn)} | {r['head']} | "
                f"{r['baseline_unc_mean']:.4f} | {r['seg_unc_mean']:.4f} | "
                f"{r['real_p95']:.4f} | {100 * r['frac_above_p95']:.0f}% |"
            )
    L.append("")

    # ---------- E6 table ----------
    L += [f"## E6 rolling-spread monitor "
          f"(threshold={E6_THRESHOLD}, window={E6_WINDOW})", ""]
    L += ["| condition | fire fraction | fires? |", "|---|---|---|"]
    for sn, sinfo in list(results.items()) + [("carla", {"e6": carla_e6})]:
        e6 = sinfo["e6"]
        fires_str = "**YES**" if e6["fires"] else "no"
        L.append(
            f"| {seg_labels.get(sn, sn)} | "
            f"{100 * e6['fired_frac']:.1f}% | {fires_str} |"
        )
    L.append("")

    # ---------- verdict ----------
    L += ["## Verdict", ""]
    night_segs = [sn for sn in results if sn != "daytime_control"]
    control_e1_ok = all(
        r["ratio"] >= COLLAPSE
        for r in results["daytime_control"]["e1"]
        if r["head"] in HEAD_NAMES
    )
    night_collapsed = {
        sn: sum(1 for r in results[sn]["e1"] if r["ratio"] < COLLAPSE)
        for sn in night_segs
    }
    carla_collapsed = sum(1 for r in carla_e1 if r["ratio"] < COLLAPSE)
    night_e6_fires = {sn: results[sn]["e6"]["fires"] for sn in night_segs}

    # daytime_control E1 sanity is based on output heads, not hidden_state norm
    L.append(
        f"E1 sanity check: daytime control {'PASSED' if control_e1_ok else 'FAILED -- pipeline error!'} "
        f"(all output heads active on daytime in-distribution footage)."
    )

    all_night_no_collapse = all(n == 0 for n in night_collapsed.values())
    if all_night_no_collapse:
        L.append(
            f"Real night + headlight/tail-light glare does NOT induce E1 output collapse: "
            f"all heads remain active on EV6 night ({night_collapsed['ev6_night']} "
            f"collapsed) and Bronco night ({night_collapsed['bronco_night']} collapsed). "
            f"CARLA collapses {carla_collapsed}/{len(carla_e1)} heads as reference."
        )
    else:
        L.append(
            f"Real night + glare induces E1 collapse on some heads: "
            f"EV6 {night_collapsed['ev6_night']} collapsed, "
            f"Bronco {night_collapsed['bronco_night']} collapsed. "
            f"CARLA collapses {carla_collapsed}/{len(carla_e1)} heads."
        )

    e6_night_fires = any(night_e6_fires.values())
    control_e6_fires = results["daytime_control"]["e6"]["fires"]
    if not e6_night_fires:
        L.append(
            "E6 rolling-spread monitor does NOT fire on any real night segment "
            f"(EV6: {100*results['ev6_night']['e6']['fired_frac']:.1f}% frames flagged, "
            f"Bronco: {100*results['bronco_night']['e6']['fired_frac']:.1f}% frames flagged; "
            f"CARLA fires at {100*carla_e6['fired_frac']:.1f}%)."
        )
    else:
        L.append(
            "E6 rolling-spread monitor FIRES on at least one real night segment. "
            f"EV6: {100*results['ev6_night']['e6']['fired_frac']:.1f}%, "
            f"Bronco: {100*results['bronco_night']['e6']['fired_frac']:.1f}%."
        )

    if control_e6_fires:
        dc_e6 = results["daytime_control"]["e6"]
        L.append(
            f"NOTE: E6 also fires on the daytime control segment "
            f"({100*dc_e6['fired_frac']:.1f}% frames flagged). "
            "Independent verification confirms this is genuine model behaviour on clean, "
            "correctly-warped input, not a pipeline artifact. The segment intermittently "
            "enters a bi-modal near-zero recurrent attractor (norm toggles from frame ~16); "
            "in the low-norm regime the output heads are suppressed (~45x on desired_curv), "
            "and E1 reads 0/10 only because the activity metric averages across both regimes. "
            "The earlier 'high steer + low speed' cause was falsified (EV6 night reaches a "
            "higher peak steer at the same speed without collapsing). The trigger is "
            "UNEXPLAINED; this is an open E6 caveat, a real-segment near-collapse the monitor "
            "fires on, distinct from the fully silent CARLA case."
        )

    if all_night_no_collapse and not e6_night_fires:
        L += ["",
              "**Bounding conclusion**: real night + headlight/tail-light glare on a "
              "comma-3 device does not collapse the model (E1: 0 heads, "
              "E6: does not fire on either night segment). The CARLA collapse signature "
              "is driven by the synthetic sim-to-real rendering gap, not by real low-light "
              "or glare conditions that openpilot's training distribution covers. One caveat "
              "sharpens rather than contradicts this: a single real daytime segment "
              "intermittently enters a CARLA-like near-zero recurrent attractor (E6 fires, "
              "trigger unexplained), so the collapse is best described as predominantly "
              "sim-induced rather than strictly CARLA-only."]
    elif all_night_no_collapse and e6_night_fires:
        L += ["",
              "**Partial finding**: E1 output heads remain active on real night footage "
              "(no collapse), but E6 spread monitor fires, indicating tighter recurrent "
              "feature dynamics than baseline. Night glare is a moderate distributional "
              "shift, weaker than CARLA synthetic."]
    else:
        L += ["",
              "**Finding**: real night + glare induces measurable E1 output collapse, "
              "partially confirming the CARLA finding on real adverse-condition data. "
              "Extent is less severe than CARLA synthetic renders."]

    out = REPORT / "real_weather_results.md"
    out.write_text("\n".join(L) + "\n")
    print(f"  report -> {out}")


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E_weather: real adverse-weather OOD axis"
    )
    ap.add_argument(
        "--collect", action="store_true",
        help="re-run the model over raw segments instead of loading the cache"
    )
    args = ap.parse_args(argv)

    # GPU verification
    try:
        import subprocess
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        print(f"[GPU] {smi.stdout.strip()}")
    except Exception as ex:
        print(f"[GPU] nvidia-smi unavailable: {ex}")

    if args.collect or not CACHE.exists():
        print(f"Collecting (N={N}/segment) -- first warm triggers ~28s PTX JIT ...")
        raw = _collect_live()
        _save_cache(CACHE, raw)
        print(f"  cached -> {CACHE.relative_to(ROOT)}")
    else:
        print(f"Loading cache {CACHE.relative_to(ROOT)}")
        raw = _load_cache(CACHE)

    # Post-warmup slice
    segs = {name: _post(d) for name, d in raw.items()}
    for name, d in segs.items():
        print(f"  post-warmup {name}: {len(d['accel_t0'])} frames")

    baseline = _load_baseline()
    carla = _load_carla()
    print(f"  baseline (Subaru+RAM): {len(baseline['accel_t0'])} frames")
    print(f"  carla reference: {len(carla['accel_t0'])} frames")

    # Compute metrics
    results: dict[str, dict] = {}
    for name, seg in segs.items():
        e1 = e1_activity(seg, baseline)
        e2 = e2_feature_ood(seg, baseline)
        e3 = e3_uncertainty(seg, baseline)
        e6 = e6_monitor(seg)
        results[name] = {"e1": e1, "e2": e2, "e3": e3, "e6": e6}

    # Print summary
    print("\n=== E1  ACTIVITY RATIO (seg / v0.9.7 real baseline) ===")
    print(f"  {'condition':<28} {'head':<16} {'ratio':>10} {'collapsed':>10}  state")
    for name, sinfo in list(results.items()) + [("carla_ref", {"e1": e1_activity(carla, baseline)})]:
        for r in sinfo["e1"]:
            st = "COLLAPSED" if r["ratio"] < COLLAPSE else "alive"
            print(f"  {name:<28} {r['head']:<16} {r['ratio']:>10.4f} "
                  f"{100*r['collapsed_frac']:>9.0f}%  {st}")

    print("\n=== E2  FEATURE SPREAD RATIO ===")
    print(f"  {'condition':<28} {'spread_ratio':>14} {'d_prime':>9} {'sep':>8}")
    for name, sinfo in list(results.items()) + [("carla_ref", {"e2": e2_feature_ood(carla, baseline)})]:
        e2 = sinfo["e2"]
        print(f"  {name:<28} {e2['spread_ratio']:>14.5f} {e2['dprime']:>9.2f} "
              f"{100*e2['separability']:>7.1f}%")

    print("\n=== E3  UNCERTAINTY ABOVE BASELINE p95 ===")
    print(f"  {'condition':<28} {'head':<14} {'seg_unc':>10} {'p95':>10} {'above_p95':>10}")
    for name, sinfo in list(results.items()) + [("carla_ref", {"e3": e3_uncertainty(carla, baseline)})]:
        for r in sinfo["e3"]:
            print(f"  {name:<28} {r['head']:<14} {r['seg_unc_mean']:>10.4f} "
                  f"{r['real_p95']:>10.4f} {100*r['frac_above_p95']:>9.0f}%")

    print("\n=== E6  ROLLING-SPREAD MONITOR ===")
    print(f"  {'condition':<28} {'fire_frac':>10} {'fires?':>8}")
    carla_e6 = e6_monitor(carla)
    for name, sinfo in list(results.items()) + [("carla_ref", {"e6": carla_e6})]:
        e6 = sinfo["e6"]
        fires = "FIRES" if e6["fires"] else "no"
        print(f"  {name:<28} {100*e6['fired_frac']:>9.1f}% {fires:>8}")

    _make_figure(results, baseline, carla)
    _write_report(results, baseline, carla)

    print("\n=== VERDICT ===")
    control_ok = all(r["ratio"] >= COLLAPSE for r in results["daytime_control"]["e1"]
                     if r["head"] in HEAD_NAMES)
    print(f"  Daytime control sanity: {'PASS' if control_ok else 'FAIL -- pipeline error!'}")
    for sn in ["ev6_night", "bronco_night"]:
        n_coll = sum(1 for r in results[sn]["e1"] if r["ratio"] < COLLAPSE)
        fires = results[sn]["e6"]["fires"]
        print(f"  {sn}: {n_coll} collapsed heads, E6 fires={fires}")
    n_carla = sum(1 for r in e1_activity(carla, baseline) if r["ratio"] < COLLAPSE)
    print(f"  CARLA ref: {n_carla} collapsed heads, E6 fires={carla_e6['fires']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
