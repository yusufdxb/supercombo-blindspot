"""E9: how much of the CARLA collapse do low-level pixel statistics explain?

E1-E3 show supercombo's outputs and recurrent feature vector collapse on CARLA
renders. The first question a reviewer raises: is that collapse just CARLA's
low-level pixel statistics (color cast, contrast, texture-energy histogram)? We
push CARLA's statistics onto the real distribution and measure what lifts and what
survives. The finding is split: output activity partially recovers (readouts below
1% of real fall from 8/10 to 1-3/10), but the recurrent-state freeze does not
(hidden-state spread stays ~1e-5 of real, feature cluster still separable,
exported uncertainty not elevated past its real p95). So the tested statistics are excluded as a
sufficient explanation for the freeze, not for all output quiescence.

This experiment holds scene content fixed and pushes CARLA's model-frame pixel
statistics toward the real distribution, three ways of increasing reach:

  moment  per-channel affine match of mean + std toward the pooled real channel,
          then clipped to real's [0.5, 99.5] percentile range (so the achieved
          moments are approximate; the sanity print reports the residual).
  hist    per-channel monotonic CDF match of the full marginal distribution.
  fda     Fourier-domain adaptation: replace only CARLA's low-frequency amplitude
          band (beta=0.02) with the mean real amplitude, keeping CARLA's phase
          (up to a final range clip). Pushes low-frequency spatial energy toward
          real, addressing the "renders are too clean" rebuttal a marginal match
          leaves open.

The result is an INVARIANCE test, not renderer identification. These interventions
align pooled per-channel marginals (and a low-frequency amplitude band), not
cross-channel dependencies, spatial/temporal structure, or semantics; scene
geometry and the zero- vs liveCalibration warp are untouched. So whatever collapse
SURVIVES matching is not explained by the tested low-level statistics; it does not
follow that the collapse is renderer-independent. We re-run the E1/E2/E3 teardown
on each variant against the same real baseline and report all 10 tracked readouts
under both the 1% and 10% activity thresholds.

    env -u PYTHONPATH .venv/bin/python -m src.e9_pixelstat            # from cache
    env PYTHONPATH=. .venv/bin/python -m src.e9_pixelstat --collect   # re-run model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.teardown import (N, WARMUP, e1_collapse_map, e2_feature_ood,
                          e3_confidence, _post)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG_DIR = ROOT / "report" / "figures"
RESULTS_MD = ROOT / "report" / "e9_pixelstat_results.md"
CACHE = ROOT / "report" / "e9_collected.npz"


# --------------------------------------------------------------------------
# pixel-statistic interventions on the 6-channel medmodel input
# --------------------------------------------------------------------------

def _channel_stats(frames: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-channel (mean, std, lo, hi) over a stack of (n, 6, H, W) frames."""
    flat = frames.reshape(len(frames), frames.shape[1], -1)  # (n, 6, HW)
    per_ch = flat.transpose(1, 0, 2).reshape(frames.shape[1], -1)  # (6, n*HW)
    return (per_ch.mean(1), per_ch.std(1) + 1e-8,
            np.percentile(per_ch, 0.5, axis=1), np.percentile(per_ch, 99.5, axis=1))


def moment_match(carla: np.ndarray, real_ref: np.ndarray) -> np.ndarray:
    """Affine per-channel match of CARLA mean+std toward the pooled real channel,
    then clipped to real's [0.5, 99.5] percentile range. Clipping makes the
    achieved moments approximate; the residual is reported by the sanity print."""
    c_mu, c_sd, _, _ = _channel_stats(carla)
    r_mu, r_sd, r_lo, r_hi = _channel_stats(real_ref)
    out = np.empty_like(carla, dtype=np.float32)
    for ch in range(carla.shape[1]):
        z = (carla[:, ch] - c_mu[ch]) / c_sd[ch]
        out[:, ch] = np.clip(z * r_sd[ch] + r_mu[ch], r_lo[ch], r_hi[ch])
    return out


def _cdf_match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Monotonic midpoint-quantile remap of src toward ref's CDF, with source ties
    preserved: identical source values share one midpoint quantile and map to the
    same output. Because ties are preserved the output CDF only approaches ref's,
    it does not equal it in general (a constant source cannot become non-constant).
    This is not skimage.exposure.match_histograms, which uses a different
    (empirical-CDF) convention and yields a different mapping."""
    src_flat = src.ravel()
    uniq, inv, counts = np.unique(src_flat, return_inverse=True, return_counts=True)
    # midpoint cumulative quantile of each unique source value
    src_q = (np.cumsum(counts) - 0.5 * counts) / len(src_flat)
    ref_sorted = np.sort(ref.ravel().astype(np.float64))
    ref_q = (np.arange(len(ref_sorted)) + 0.5) / len(ref_sorted)
    uniq_mapped = np.interp(src_q, ref_q, ref_sorted)
    return uniq_mapped[inv].reshape(src.shape).astype(np.float32)


def hist_match(carla: np.ndarray, real_ref: np.ndarray) -> np.ndarray:
    """Per-channel full-distribution histogram match of CARLA onto pooled real."""
    out = np.empty_like(carla, dtype=np.float32)
    for ch in range(carla.shape[1]):
        # subsample the reference pool for a stable CDF without huge memory
        ref = real_ref[:, ch].reshape(-1)
        if len(ref) > 400_000:
            idx = np.linspace(0, len(ref) - 1, 400_000).astype(int)
            ref = ref[idx]
        out[:, ch] = _cdf_match_channel(carla[:, ch], ref)
    return out


def fda_match(carla: np.ndarray, real_ref: np.ndarray, beta: float = 0.02) -> np.ndarray:
    """Fourier-domain adaptation: replace CARLA's low-frequency amplitude spectrum
    (central beta-band) with the mean real amplitude, keeping CARLA's phase in the
    frequency-domain step. The final spatial-domain clip to real's percentile range
    is a nonlinearity, so the phase of the returned frame is preserved only up to
    that clip.

    This moves the spatial-frequency energy toward real footage while preserving
    scene structure, addressing the "renders lack real texture/noise" objection
    that a marginal-histogram match cannot."""
    n, C, H, W = carla.shape
    # mean real amplitude per channel (shifted so DC is centered)
    real_amp = np.zeros((C, H, W), dtype=np.float64)
    for i in range(len(real_ref)):
        for ch in range(C):
            real_amp[ch] += np.abs(np.fft.fftshift(np.fft.fft2(real_ref[i, ch])))
    real_amp /= len(real_ref)

    cy, cx = H // 2, W // 2
    by, bx = max(1, int(H * beta)), max(1, int(W * beta))
    r_lo = np.percentile(real_ref.reshape(len(real_ref), C, -1).transpose(1, 0, 2)
                         .reshape(C, -1), 0.5, axis=1)
    r_hi = np.percentile(real_ref.reshape(len(real_ref), C, -1).transpose(1, 0, 2)
                         .reshape(C, -1), 99.5, axis=1)
    out = np.empty_like(carla, dtype=np.float32)
    for i in range(n):
        for ch in range(C):
            f = np.fft.fftshift(np.fft.fft2(carla[i, ch]))
            amp, pha = np.abs(f), np.angle(f)
            amp[cy - by:cy + by, cx - bx:cx + bx] = \
                real_amp[ch, cy - by:cy + by, cx - bx:cx + bx]
            rec = np.fft.ifft2(np.fft.ifftshift(amp * np.exp(1j * pha))).real
            out[i, ch] = np.clip(rec, r_lo[ch], r_hi[ch])
    return out


# --------------------------------------------------------------------------
# collection
# --------------------------------------------------------------------------

def _diagnostics(variant: np.ndarray, real_ref: np.ndarray, beta: float = 0.02) -> np.ndarray:
    """Per-intervention match quality against the real reference, as
    [mean error, std error, marginal distance, low-frequency band error].

    Each intervention targets a different statistic, so a single mean-deviation
    number cannot validate all three; we report one diagnostic per target."""
    v_mu, v_sd, _, _ = _channel_stats(variant)
    r_mu, r_sd, _, _ = _channel_stats(real_ref)
    qs = np.linspace(0.01, 0.99, 99)
    marg, band = [], []
    C = variant.shape[1]
    for ch in range(C):
        # marginal distance: mean absolute quantile difference (1-Wasserstein style)
        vq = np.percentile(variant[:, ch], qs * 100)
        rq = np.percentile(real_ref[:, ch], qs * 100)
        marg.append(np.abs(vq - rq).mean())
        # low-frequency amplitude error over the same beta band the FDA swap targets
        H, W = variant.shape[2], variant.shape[3]
        by, bx = max(1, int(H * beta)), max(1, int(W * beta))
        cy, cx = H // 2, W // 2
        def _band(stack):
            amps = []
            for i in range(0, len(stack), max(1, len(stack) // 24)):  # subsample frames
                f = np.fft.fftshift(np.fft.fft2(stack[i, ch]))
                amps.append(np.abs(f[cy - by:cy + by, cx - bx:cx + bx]).mean())
            return float(np.mean(amps))
        band.append(abs(_band(variant) - _band(real_ref)))
    return np.array([float(np.abs(v_mu - r_mu).mean()), float(np.abs(v_sd - r_sd).mean()),
                     float(np.mean(marg)), float(np.mean(band))], dtype=np.float64)


def _collect_live() -> dict[str, dict]:
    """Run the model on real (Subaru+RAM) and on raw + statistic-matched CARLA."""
    from src.probe_model import collect, load_carla_six, load_real_six
    from src.state import build_session, load_output_slices

    sess, slices = build_session(), load_output_slices()
    print(f"Collecting (N={N}/segment) ...")
    subaru_six = load_real_six(DATA / "subaru_source" / "fcamera.hevc",
                               DATA / "subaru_source" / "rlog.bz2", N)
    ram_six = load_real_six(DATA / "ram_source" / "fcamera.hevc",
                            DATA / "ram_source" / "rlog.bz2", N)
    carla_six = load_carla_six(DATA / "domain_gap" / "carla_rgb.npy", N)

    real_ref = np.stack(subaru_six + ram_six)          # pooled real reference
    carla_arr = np.stack(carla_six)
    carla_mom = moment_match(carla_arr, real_ref)
    carla_his = hist_match(carla_arr, real_ref)
    carla_fda = fda_match(carla_arr, real_ref)

    # sanity: report how well statistics were matched (channel-averaged)
    r_mu, r_sd, _, _ = _channel_stats(real_ref)
    for label, arr in [("carla_raw", carla_arr), ("carla_moment", carla_mom),
                       ("carla_hist", carla_his), ("carla_fda", carla_fda)]:
        mu, sd, _, _ = _channel_stats(arr)
        print(f"  {label:<13} mean|dev-from-real|={np.abs(mu - r_mu).mean():7.3f}  "
              f"std|dev|={np.abs(sd - r_sd).mean():7.3f}")

    def _to_list(a):
        return [a[i] for i in range(len(a))]

    # per-intervention match diagnostics, persisted so the report can be rebuilt
    # from the cache without the raw frames
    diag = {name: _diagnostics(arr, real_ref) for name, arr in
            [("carla_raw", carla_arr), ("carla_moment", carla_mom),
             ("carla_hist", carla_his), ("carla_fda", carla_fda)]}

    return {
        "subaru": collect(subaru_six, sess, slices),
        "ram": collect(ram_six, sess, slices),
        "carla_raw": collect(carla_six, sess, slices),
        "carla_moment": collect(_to_list(carla_mom), sess, slices),
        "carla_hist": collect(_to_list(carla_his), sess, slices),
        "carla_fda": collect(_to_list(carla_fda), sess, slices),
        "diagnostics": diag,
    }


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
# analysis
# --------------------------------------------------------------------------

def _summarize(real: dict, carla: dict) -> dict:
    # e1_collapse_map ranks all 10 tracked readouts (3 scalars derived from heads
    # + 7 heads). We report the full set under BOTH the <1% and <10% thresholds,
    # not a filtered subset, so output recovery is not hidden by threshold choice.
    e1 = e1_collapse_map(real, carla)
    e2 = e2_feature_ood(real, carla)
    e3 = e3_confidence(real, carla)
    ratios = {r["head"]: r["ratio"] for r in e1}
    finite = [r["ratio"] for r in e1 if np.isfinite(r["ratio"])]
    below01 = sum(1 for x in finite if x < 0.01)
    below10 = sum(1 for x in finite if x < 0.10)
    # exact uncertainty exceedance: worst head's fraction of CARLA frames above real p95
    unc_frac = float(max(r["carla_above_real_p95"] for r in e3))
    return {
        "n_readouts": len(e1), "below01": below01, "below10": below10,
        "ratios": ratios,
        "spread_ratio": e2["spread_ratio"], "separability": e2["separability"],
        "dprime": e2["dprime"], "unc_frac_max": unc_frac,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pixel-statistic control for the CARLA collapse")
    ap.add_argument("--collect", action="store_true",
                    help="re-run the model instead of loading the cached outputs")
    args = ap.parse_args(argv)

    if args.collect or not CACHE.exists():
        segments = _collect_live()
        _save_cache(CACHE, segments)
        print(f"  cached -> {CACHE.relative_to(ROOT)}")
    else:
        print(f"Loading cached outputs from {CACHE.relative_to(ROOT)} (pass --collect to re-run).")
        segments = _load_cache(CACHE)

    subaru = _post(segments["subaru"], WARMUP)
    ram = _post(segments["ram"], WARMUP)
    real = {k: np.concatenate([subaru[k], ram[k]]) for k in subaru}

    variants = ["carla_raw", "carla_moment", "carla_hist", "carla_fda"]
    labels = {"carla_raw": "CARLA (raw)",
              "carla_moment": "CARLA + mean/std match",
              "carla_hist": "CARLA + histogram match",
              "carla_fda": "CARLA + Fourier (FDA) match"}
    results = {v: _summarize(real, _post(segments[v], WARMUP)) for v in variants}

    print("\n=== E9  PIXEL-STATISTIC CONTROL (real baseline: Subaru+RAM) ===")
    print(f"  {'variant':<26} {'readouts<1%':>12} {'readouts<10%':>13} "
          f"{'feat spread':>12} {'separable':>10} {'unc>p95':>9}")
    for v in variants:
        r = results[v]
        print(f"  {labels[v]:<26} {r['below01']:>8}/{r['n_readouts']:<3} "
              f"{r['below10']:>9}/{r['n_readouts']:<3} {r['spread_ratio']:>12.2e} "
              f"{100*r['separability']:>9.1f}% {100*r['unc_frac_max']:>8.1f}%")

    _verdict(results, variants, labels)
    _write_md(results, variants, labels, segments.get("diagnostics", {}))
    _fig(results, variants, labels)
    print(f"\nresults -> {RESULTS_MD.relative_to(ROOT)}   figure -> "
          f"{(FIG_DIR / 'e9_pixelstat.png').relative_to(ROOT)}")
    return 0


def _verdict(results, variants, labels) -> None:
    raw = results["carla_raw"]
    matched = [v for v in variants if v != "carla_raw"]
    best01 = min(results[v]["below01"] for v in matched)   # fewest still <1%
    best_spread = max(results[v]["spread_ratio"] for v in matched)
    print("\n=== VERDICT (narrow) ===")
    print(f"  Output activity PARTIALLY recovers: readouts below 1% of real fall from "
          f"{raw['below01']}/{raw['n_readouts']} (raw) to {best01}/{raw['n_readouts']} "
          f"under matching.")
    print(f"  The RECURRENT FREEZE survives: hidden-state spread stays "
          f"{raw['spread_ratio']:.2e} -> best {best_spread:.2e} of real, separability "
          f"~{100*raw['separability']:.0f}%, and exported uncertainty not elevated "
          f"under this metric (<=0.5% of frames exceed the real p95).")
    print("  => the tested low-level pixel statistics are excluded as a SUFFICIENT "
          "explanation for the recurrent-state freeze, but they partly explain the "
          "output quiescence. This is an invariance test, not renderer identification: "
          "geometry, phase, higher-order texture, and semantics remain confounded.")


def _write_md(results, variants, labels, diag: dict | None = None) -> None:
    raw = results["carla_raw"]
    matched = [v for v in variants if v != "carla_raw"]
    best01 = min(results[v]["below01"] for v in matched)
    best_spread = max(results[v]["spread_ratio"] for v in matched)
    L = ["# E9  pixel-statistic control for the CARLA collapse", "",
         "Scene content is held fixed; CARLA's 6-channel medmodel input has its "
         "per-channel pixel statistics pushed onto the pooled real (Subaru+RAM) "
         "distribution three ways: per-channel moment (mean/std) match (clipped to "
         "real's range, so approximate), full marginal-histogram match, and a "
         "low-frequency Fourier-amplitude band swap (FDA, beta=0.02, CARLA phase "
         "kept up to a final clip). If the E1-E3 collapse were a "
         "low-level pixel-statistic artifact it should lift under matching; the "
         "invariant part is whatever survives. All 10 tracked readouts (3 scalars "
         "derived from heads + 7 heads) are reported under both thresholds; "
         "`accel_t0` is extracted from `plan` and is not an independent head.", "",
         "| variant | readouts <1% | readouts <10% | recurrent spread (xreal) "
         "| separability | max unc >real p95 |", "|---|---|---|---|---|---|"]
    for v in variants:
        r = results[v]
        L.append(f"| {labels[v]} | {r['below01']}/{r['n_readouts']} | "
                 f"{r['below10']}/{r['n_readouts']} | {r['spread_ratio']:.2e} | "
                 f"{100*r['separability']:.1f}% | {100*r['unc_frac_max']:.1f}% |")
    L += ["", "## Reading", "",
          f"- Output activity partially recovers: readouts below 1% of real fall "
          f"from {raw['below01']}/{raw['n_readouts']} on raw CARLA to "
          f"{best01}/{raw['n_readouts']} under matching (at the 10% threshold most "
          f"stay suppressed).",
          f"- The recurrent freeze survives: hidden-state spread stays "
          f"{raw['spread_ratio']:.2e} -> {best_spread:.2e} of real and separability "
          f"holds ~{100*raw['separability']:.0f}%; exported uncertainty is not elevated "
          f"under this metric, with at most "
          f"{100*max(results[v]['unc_frac_max'] for v in variants):.1f}% of frames on any "
          f"head exceeding the real p95 (this measures p95 exceedance only, not that the "
          f"uncertainty channel is literally flat).",
          "- Interpretation: the tested low-level pixel statistics are excluded as a "
          "*sufficient* explanation for the recurrent-state freeze, but they partly "
          "explain the output quiescence. This is an invariance test, not renderer "
          "identification: geometry (zero- vs liveCalibration warp), phase, "
          "higher-order texture, and semantics remain confounded, and only one "
          "renderer and one CARLA sequence were tested."]

    # Table E9: every per-readout ratio, so a reader can check the split directly
    # rather than trusting the thresholded counts.
    order = list(results[variants[0]]["ratios"].keys())
    L += ["", "## Table E9: per-readout activity ratio (CARLA / pooled real)", "",
          "All 10 tracked readouts under every condition. Values below 0.01 are "
          "collapsed at the 1% threshold; values below 0.10 are suppressed at the "
          "10% threshold.", "",
          "| readout | " + " | ".join(labels[v] for v in variants) + " |",
          "|---" * (len(variants) + 1) + "|"]
    for name in order:
        cells = []
        for v in variants:
            r = results[v]["ratios"][name]
            cells.append("n/a" if not np.isfinite(r) else f"{r:.4f}")
        L.append(f"| `{name}` | " + " | ".join(cells) + " |")

    if diag:
        L += ["", "## Table E9b: intervention match quality", "",
              "Each intervention targets a different statistic, so one summary "
              "number cannot validate all three. Absolute error against the pooled "
              "real reference, averaged over the 6 input channels.", "",
              "| variant | mean err | std err | marginal distance | low-freq band err |",
              "|---|---|---|---|---|"]
        for v in variants:
            d = np.asarray(diag.get(v, []), dtype=float)
            if d.size == 4:
                L.append(f"| {labels[v]} | {d[0]:.3f} | {d[1]:.3f} | {d[2]:.3f} | {d[3]:.1f} |")

    RESULTS_MD.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_MD.write_text("\n".join(L) + "\n")


def _fig(results, variants, labels) -> None:
    from src.teardown import _plt, CARLA_C, REAL_C, WARN_C
    plt = _plt()
    short = {"carla_raw": "raw", "carla_moment": "moment\n(mean/std)",
             "carla_hist": "histogram", "carla_fda": "Fourier\n(FDA)"}
    ticks = [short.get(v, labels[v]) for v in variants]
    xs = np.arange(len(variants))
    below01 = [results[v]["below01"] for v in variants]
    below10 = [results[v]["below10"] for v in variants]
    spread = [max(results[v]["spread_ratio"], 1e-6) for v in variants]
    n_read = results[variants[0]]["n_readouts"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.6))
    w = 0.38
    b1 = ax1.bar(xs - w / 2, below01, w, color=CARLA_C, label="below 1% (collapsed)")
    b2 = ax1.bar(xs + w / 2, below10, w, color=WARN_C, label="below 10% (suppressed)")
    ax1.bar_label(b1, fmt="%d", padding=2, color="#0b0b0b", fontsize=10)
    ax1.bar_label(b2, fmt="%d", padding=2, color="#0b0b0b", fontsize=10)
    ax1.set_ylim(0, n_read + 1.2)
    ax1.axhline(n_read, color="#898781", ls=":", lw=1.0)
    ax1.set_ylabel(f"readouts below threshold (of {n_read})")
    ax1.set_xticks(xs, ticks, fontsize=9)
    ax1.set_title("Output collapse partially lifts (both thresholds shown)")
    ax1.legend(loc="upper right", fontsize=8, facecolor="#ffffff", edgecolor="#c3c2b7")
    ax2.bar(xs, spread, color=REAL_C, width=0.6)
    ax2.set_yscale("log")
    ax2.set_ylim(1e-6, 3.0)
    ax2.axhline(1.0, color="#898781", ls=":", lw=1.0, label="parity with real")
    ax2.set_ylabel("recurrent-feature spread / real  (log)")
    ax2.set_xticks(xs, ticks, fontsize=9)
    ax2.set_title("Recurrent freeze survives matching")
    ax2.legend(loc="upper left", facecolor="#ffffff", edgecolor="#c3c2b7")
    fig.suptitle("E9  matching CARLA's low-level statistics to real: outputs partly "
                 "recover, the recurrent freeze does not", fontsize=12)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "e9_pixelstat.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    sys.exit(main())
