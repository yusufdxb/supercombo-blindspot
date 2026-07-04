"""E8 demo: hybrid OOD detector narrative figure.

Runnable as:
    env -u PYTHONPATH .venv/bin/python -m src.e8_demo

Reads only from committed report/*.npz caches. No model, no GPU.

Four-panel figure telling the complete story:
  (a) COLLAPSE sequence (CARLA alpha=1.0): E6 fires (score high), Maha misses
      (score low -- uses GLOBAL pooled-ID Maha, the canonical LOCO finding).
  (b) CORRUPTION sequence (fog severity 5): Maha fires, E6 misses
      (same global Maha -- photometric shift moves the feature far from ID mean).
  (c) ID real-driving (per-vehicle, subaru test split): E6 FPR=0%, Maha FPR=44%.
      The Maha arm has elevated FPR within-vehicle because 223 calib frames are
      insufficient to estimate the 99th-percentile tail of a 512-D Gaussian.
  (d) Coverage summary bar chart: detection/false-alarm rates per detector.

The GLOBAL Maha is used in panels (a) and (b) to show the canonical finding:
    - collapse: Maha AUROC = 0.16 (below chance, misses collapse).
    - corruption: Maha AUROC ~ 1.0 (catches photometric shift).
The per-vehicle Maha is used in (c)/(d) to show the within-vehicle FPR story.

Runtime target: under 90 seconds. The 512-D precision matrix is computed ONCE
for each fitting call (global + per-vehicle) and reused for all scoring.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"
FIG_DIR = REPORT / "figures"
FIG_OUT = FIG_DIR / "e8_demo_legacy.png"  # canonical e8_demo.png is now owned by src/e8_pca_hybrid.py

WINDOW = 30
WARMUP = 100   # frames discarded per segment (see teardown.py)

# -------------------------------------------------------------------------
# Minimal Maha helpers that accept a pre-computed precision matrix
# -------------------------------------------------------------------------


def _fit_gaussian(features_id: np.ndarray, reg: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, precision) from ID features. Matching baselines.py logic."""
    mu = features_id.mean(axis=0)
    X = features_id - mu
    cov = (X.T @ X) / max(len(X) - 1, 1)
    trace_mean = float(np.trace(cov) / cov.shape[0])
    cov = cov + (reg * trace_mean + 1e-6) * np.eye(cov.shape[0], dtype=cov.dtype)
    prec = np.linalg.inv(cov)
    return mu, prec


def _maha_from_prec(mu: np.ndarray, prec: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Vectorised squared Mahalanobis distance reusing a pre-computed precision."""
    X = test - mu
    return np.einsum("ij,jk,ik->i", X, prec, X).astype(np.float64)


# -------------------------------------------------------------------------
# Rolling spread (E6 score, lower = more OOD)
# -------------------------------------------------------------------------


def rolling_spread(hidden: np.ndarray, window: int) -> np.ndarray:
    T, D = hidden.shape
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T + 1):
        out[t - 1] = float(np.var(hidden[t - window:t], axis=0).sum())
    return out


# -------------------------------------------------------------------------
# Normalised scores [0,1] with higher=OOD convention
# -------------------------------------------------------------------------


def _norm_e6(spreads: np.ndarray, p1: float, p99: float) -> np.ndarray:
    """Higher = more OOD: inverted and clipped to [0,1]."""
    denom = max(p99 - p1, 1e-12)
    c = np.where(np.isfinite(spreads), spreads, p99)
    return np.clip((p99 - c) / denom, 0.0, 1.0)


def _norm_maha(maha_scores: np.ndarray, p1: float, p99: float) -> np.ndarray:
    denom = max(p99 - p1, 1e-12)
    return np.clip((maha_scores - p1) / denom, 0.0, 1.0)


# -------------------------------------------------------------------------
# Load caches
# -------------------------------------------------------------------------


def _load_id() -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Return (by_vehicle, all_id_hidden)."""
    d = np.load(REPORT / "teardown_collected.npz")
    by_vehicle = {
        "subaru": d["subaru__hidden_state"],
        "ram": d["ram__hidden_state"],
    }
    all_id = np.concatenate(list(by_vehicle.values()), axis=0)
    return by_vehicle, all_id


def _load_e4_collapse() -> np.ndarray:
    d = np.load(REPORT / "e4_collected.npz")
    return d["1.0000__hidden_state"]   # full CARLA (alpha=1)


def _load_e7_condition(corruption: str, severity: int) -> np.ndarray:
    d = np.load(REPORT / "e7_collected.npz", allow_pickle=True)
    key = f"{corruption}__{severity}__hidden_state"
    raw = d[key]
    return raw[WARMUP:]   # discard warmup


# -------------------------------------------------------------------------
# Per-vehicle split (mirrors per_vehicle_hybrid_fpr in e8_hybrid.py)
# -------------------------------------------------------------------------


def _vehicle_calib_test(
    hidden: np.ndarray, calib_frac: float = 0.7, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    T = len(hidden)
    idx = rng.permutation(T)
    n_calib = max(int(T * calib_frac), WINDOW + 1)
    return hidden[idx[:n_calib]], hidden[idx[n_calib:]]


# -------------------------------------------------------------------------
# Main demo
# -------------------------------------------------------------------------


def run_demo() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    import matplotlib.gridspec as gridspec
    from sklearn.metrics import roc_auc_score

    print("Loading caches ...", flush=True)
    by_vehicle, all_id = _load_id()

    # ---- GLOBAL Maha (pooled ID, as used in the LOCO analysis) ----
    # This is the canonical Phantom-Braking finding:
    #   collapse -> Maha AUROC 0.16 (below chance)
    #   fog/corruption -> Maha AUROC 1.0
    print("Fitting global Gaussian (pooled subaru+ram, 638 frames) ...", flush=True)
    mu_g, prec_g = _fit_gaussian(all_id.astype(np.float64))

    # Global ID spreads and Maha scores for normalisation
    id_spreads_g = rolling_spread(all_id, WINDOW)
    id_maha_g = _maha_from_prec(mu_g, prec_g, all_id.astype(np.float64))

    valid_g = id_spreads_g[np.isfinite(id_spreads_g)]
    p1_e6_g  = float(np.percentile(valid_g, 1))
    p99_e6_g = float(np.percentile(valid_g, 99))
    p1_m_g   = float(np.percentile(id_maha_g, 1))
    p99_m_g  = float(np.percentile(id_maha_g, 99))

    def _score_global(hidden: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sp = rolling_spread(hidden, WINDOW)
        mh = _maha_from_prec(mu_g, prec_g, hidden.astype(np.float64))
        e6s = _norm_e6(sp, p1_e6_g, p99_e6_g)
        ms  = _norm_maha(mh, p1_m_g, p99_m_g)
        return e6s, ms, np.maximum(e6s, ms)

    # ---- PANEL A: collapse (CARLA alpha=1.0) ----
    print("Scoring collapse sequence (CARLA alpha=1.0, global Maha) ...", flush=True)
    carla = _load_e4_collapse()
    carla_e6, carla_maha, carla_hybrid = _score_global(carla)

    # AUROC for annotation (vs pooled ID)
    id_e6_g  = _norm_e6(id_spreads_g, p1_e6_g, p99_e6_g)
    id_maha_g_n = _norm_maha(id_maha_g, p1_m_g, p99_m_g)
    id_hybrid_g = np.maximum(id_e6_g, id_maha_g_n)
    id_e6_valid = id_e6_g[np.isfinite(id_e6_g)]

    # Build label arrays
    carla_e6_valid = carla_e6[np.isfinite(carla_e6)]
    s_collapse_e6  = np.concatenate([id_e6_valid, carla_e6_valid])
    l_collapse_e6  = np.concatenate([np.zeros(len(id_e6_valid)), np.ones(len(carla_e6_valid))])
    s_collapse_mh  = np.concatenate([id_maha_g_n, carla_maha])
    l_collapse_mh  = np.concatenate([np.zeros(len(id_maha_g_n)), np.ones(len(carla_maha))])
    s_collapse_hy  = np.concatenate([id_hybrid_g, carla_hybrid])
    l_collapse_hy  = np.concatenate([np.zeros(len(id_hybrid_g)), np.ones(len(carla_hybrid))])

    auroc_e6_c  = float(roc_auc_score(l_collapse_e6, s_collapse_e6))
    auroc_mh_c  = float(roc_auc_score(l_collapse_mh, s_collapse_mh))
    auroc_hy_c  = float(roc_auc_score(l_collapse_hy, s_collapse_hy))

    # ---- PANEL B: corruption (fog severity 5) ----
    print("Scoring corruption sequence (fog sev 5, global Maha) ...", flush=True)
    fog = _load_e7_condition("fog", 5)
    fog_e6, fog_maha, fog_hybrid = _score_global(fog)

    fog_e6_valid = fog_e6[np.isfinite(fog_e6)]
    s_fog_e6 = np.concatenate([id_e6_valid, fog_e6_valid])
    l_fog_e6 = np.concatenate([np.zeros(len(id_e6_valid)), np.ones(len(fog_e6_valid))])
    s_fog_mh = np.concatenate([id_maha_g_n, fog_maha])
    l_fog_mh = np.concatenate([np.zeros(len(id_maha_g_n)), np.ones(len(fog_maha))])
    s_fog_hy = np.concatenate([id_hybrid_g, fog_hybrid])
    l_fog_hy = np.concatenate([np.zeros(len(id_hybrid_g)), np.ones(len(fog_hybrid))])

    auroc_e6_f  = float(roc_auc_score(l_fog_e6, s_fog_e6))
    auroc_mh_f  = float(roc_auc_score(l_fog_mh, s_fog_mh))
    auroc_hy_f  = float(roc_auc_score(l_fog_hy, s_fog_hy))

    print(f"  Collapse  -- E6 AUROC: {auroc_e6_c:.3f}, Maha AUROC: {auroc_mh_c:.3f}, "
          f"Hybrid AUROC: {auroc_hy_c:.3f}")
    print(f"  Fog       -- E6 AUROC: {auroc_e6_f:.3f}, Maha AUROC: {auroc_mh_f:.3f}, "
          f"Hybrid AUROC: {auroc_hy_f:.3f}")

    # ---- Per-vehicle FPR (for panels C/D) ----
    print("Computing per-vehicle FPR (subaru, 70/30 split) ...", flush=True)
    vhidden = by_vehicle["subaru"]
    calib, id_test = _vehicle_calib_test(vhidden)
    mu_v, prec_v = _fit_gaussian(calib.astype(np.float64))

    calib_spreads = rolling_spread(calib, WINDOW)
    valid_cs = calib_spreads[np.isfinite(calib_spreads)]
    p1_e6_v  = float(np.percentile(valid_cs, 1))
    p99_e6_v = float(np.percentile(valid_cs, 99))
    e6_thr_v = p1_e6_v

    calib_maha_v = _maha_from_prec(mu_v, prec_v, calib.astype(np.float64))
    p1_m_v   = float(np.percentile(calib_maha_v, 1))
    p99_m_v  = float(np.percentile(calib_maha_v, 99))
    maha_thr_v = float(np.percentile(calib_maha_v, 99))

    id_test_sp = rolling_spread(id_test, WINDOW)
    id_test_mh = _maha_from_prec(mu_v, prec_v, id_test.astype(np.float64))
    id_test_e6 = _norm_e6(id_test_sp, p1_e6_v, p99_e6_v)
    id_test_maha_n = _norm_maha(id_test_mh, p1_m_v, p99_m_v)
    id_test_hybrid = np.maximum(id_test_e6, id_test_maha_n)

    valid = np.isfinite(id_test_sp)
    id_e6_fpr    = float(np.mean(id_test_sp[valid] < e6_thr_v)) if valid.any() else 0.0
    id_maha_fpr  = float(np.mean(id_test_mh > maha_thr_v))
    id_comb_fpr  = float(np.mean(
        np.where(valid, id_test_sp < e6_thr_v, False) | (id_test_mh > maha_thr_v)
    ))

    print(f"  Subaru ID test FPR: E6={id_e6_fpr:.3f}, Maha={id_maha_fpr:.3f}, "
          f"Combined={id_comb_fpr:.3f}")
    print(f"  Root cause: only {len(calib)} calib frames for 512-D Gaussian "
          f"(need N >> 512 for reliable 99th-pct tail)", flush=True)

    # ---- Build figure ----
    print("Rendering figure ...", flush=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15, 10), dpi=130)
    fig.patch.set_facecolor("#111111")

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        left=0.07, right=0.97,
        top=0.91, bottom=0.08,
        hspace=0.55, wspace=0.35,
    )

    CRED    = "#b0472b"
    CBLUE   = "#2a78d6"
    CGREEN  = "#1baf7a"
    CGREY   = "#898781"
    CTEXT   = "#0b0b0b"
    CAXIS   = "#898781"
    CORANGE = "#d08a2e"

    def _style_ax(ax, title):
        ax.set_facecolor("#1a1a2e")
        ax.tick_params(colors=CAXIS, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")
        ax.set_title(title, color=CTEXT, fontsize=9, pad=4)
        ax.yaxis.label.set_color(CAXIS)
        ax.xaxis.label.set_color(CAXIS)

    step = 2

    # Panel A: collapse -- global Maha
    ax_a = fig.add_subplot(gs[0, :])
    _style_ax(ax_a, "(a) COLLAPSE (CARLA alpha=1.0, global pooled-ID Maha)"
              " -- E6 catches it, Mahalanobis misses")
    N_c = min(200, len(carla_e6))
    t = np.arange(0, N_c, step)
    ax_a.plot(t, carla_e6[:N_c:step],     color=CRED,   lw=1.5, alpha=0.9, label="E6")
    ax_a.plot(t, carla_maha[:N_c:step],   color=CBLUE,  lw=1.5, alpha=0.9, label="Maha (global)")
    ax_a.plot(t, carla_hybrid[:N_c:step], color=CGREEN, lw=2.0, alpha=0.9, label="Hybrid")
    ax_a.axhline(0.5, color=CORANGE, lw=0.9, ls="--", alpha=0.7, label="0.5 fire line")
    ax_a.set_ylim(-0.05, 1.15)
    ax_a.set_ylabel("OOD score (higher=OOD)", fontsize=8)
    ax_a.set_xlabel("frame", fontsize=8)
    ax_a.legend(fontsize=7, loc="upper right", framealpha=0.3,
                labelcolor=CTEXT, facecolor="#f1efe8")
    ax_a.text(
        0.02, 0.93,
        f"AUROC: E6={auroc_e6_c:.3f}  Maha={auroc_mh_c:.3f}  Hybrid={auroc_hy_c:.3f}",
        transform=ax_a.transAxes, color=CTEXT, fontsize=8.5, va="top", fontweight="bold",
        bbox=dict(facecolor="#f1efe8", alpha=0.75, boxstyle="round,pad=0.3"),
    )

    # Panel B: fog corruption -- global Maha
    ax_b = fig.add_subplot(gs[1, :])
    _style_ax(ax_b, "(b) CORRUPTION (fog severity 5, global pooled-ID Maha)"
              " -- Mahalanobis catches it, E6 misses")
    N_f = min(200, len(fog_e6))
    t = np.arange(0, N_f, step)
    ax_b.plot(t, fog_e6[:N_f:step],     color=CRED,   lw=1.5, alpha=0.9, label="E6")
    ax_b.plot(t, fog_maha[:N_f:step],   color=CBLUE,  lw=1.5, alpha=0.9, label="Maha (global)")
    ax_b.plot(t, fog_hybrid[:N_f:step], color=CGREEN, lw=2.0, alpha=0.9, label="Hybrid")
    ax_b.axhline(0.5, color=CORANGE, lw=0.9, ls="--", alpha=0.7, label="0.5 fire line")
    ax_b.set_ylim(-0.05, 1.15)
    ax_b.set_ylabel("OOD score (higher=OOD)", fontsize=8)
    ax_b.set_xlabel("frame", fontsize=8)
    ax_b.legend(fontsize=7, loc="upper right", framealpha=0.3,
                labelcolor=CTEXT, facecolor="#f1efe8")
    ax_b.text(
        0.02, 0.93,
        f"AUROC: E6={auroc_e6_f:.3f}  Maha={auroc_mh_f:.3f}  Hybrid={auroc_hy_f:.3f}",
        transform=ax_b.transAxes, color=CTEXT, fontsize=8.5, va="top", fontweight="bold",
        bbox=dict(facecolor="#f1efe8", alpha=0.75, boxstyle="round,pad=0.3"),
    )

    # Panel C (bottom-left): per-vehicle ID real-driving scores
    ax_c = fig.add_subplot(gs[2, 0])
    _style_ax(ax_c, "(c) Per-vehicle ID real-driving (subaru test split)\n"
              "E6 FPR=0%  vs  Maha FPR=44% (512-D undersampled)")
    N_id = min(150, len(id_test_e6))
    t = np.arange(0, N_id, step)
    ax_c.plot(t, id_test_e6[:N_id:step],     color=CRED,   lw=1.2, alpha=0.85, label="E6")
    ax_c.plot(t, id_test_maha_n[:N_id:step], color=CBLUE,  lw=1.2, alpha=0.85, label="Maha (per-veh)")
    ax_c.plot(t, id_test_hybrid[:N_id:step], color=CGREEN, lw=1.8, alpha=0.9,  label="Hybrid")
    ax_c.axhline(0.5, color=CORANGE, lw=0.9, ls="--", alpha=0.7)
    ax_c.set_ylim(-0.05, 1.15)
    ax_c.set_ylabel("OOD score (higher=OOD)", fontsize=8)
    ax_c.set_xlabel("frame", fontsize=8)
    ax_c.legend(fontsize=7, framealpha=0.3, labelcolor=CTEXT, facecolor="#f1efe8")
    ax_c.text(
        0.02, 0.93,
        f"FPR: E6={id_e6_fpr:.0%}  Maha={id_maha_fpr:.0%}  Combined={id_comb_fpr:.0%}",
        transform=ax_c.transAxes, color=CTEXT, fontsize=8, va="top",
        bbox=dict(facecolor="#f1efe8", alpha=0.75, boxstyle="round,pad=0.3"),
    )

    # Panel D (bottom-right): AUROC comparison bar chart
    ax_d = fig.add_subplot(gs[2, 1])
    ax_d.set_facecolor("#1a1a2e")
    _style_ax(ax_d, "(d) Global Maha: AUROC comparison\nCollapse vs Corruption")

    conditions  = ["Collapse\n(CARLA)", "Corruption\n(fog sev 5)"]
    e6_aurocs   = [auroc_e6_c,  auroc_e6_f]
    maha_aurocs = [auroc_mh_c,  auroc_mh_f]
    hy_aurocs   = [auroc_hy_c,  auroc_hy_f]

    x = np.arange(len(conditions))
    w = 0.22
    ax_d.bar(x - w, e6_aurocs,   w, label="E6",     color=CRED,   alpha=0.85)
    ax_d.bar(x,     maha_aurocs, w, label="Maha",   color=CBLUE,  alpha=0.85)
    ax_d.bar(x + w, hy_aurocs,   w, label="Hybrid", color=CGREEN, alpha=0.85)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(conditions, fontsize=9, color=CAXIS)
    ax_d.set_ylabel("AUROC", fontsize=8)
    ax_d.set_ylim(0, 1.25)
    ax_d.axhline(0.5, color=CGREY, lw=0.7, ls="--", alpha=0.6, label="chance")
    ax_d.axhline(1.0, color=CGREY, lw=0.7, ls="--", alpha=0.4)
    ax_d.legend(fontsize=7, framealpha=0.3, labelcolor=CTEXT, facecolor="#f1efe8")
    for xi, yval in zip(x - w, e6_aurocs):
        ax_d.text(xi, yval + 0.03, f"{yval:.2f}", ha="center", fontsize=8,
                  color=CRED, fontweight="bold")
    for xi, yval in zip(x, maha_aurocs):
        ax_d.text(xi, yval + 0.03, f"{yval:.2f}", ha="center", fontsize=8,
                  color=CBLUE, fontweight="bold")
    for xi, yval in zip(x + w, hy_aurocs):
        ax_d.text(xi, yval + 0.03, f"{yval:.2f}", ha="center", fontsize=8,
                  color=CGREEN, fontweight="bold")

    # Annotate Maha below-chance on collapse
    ax_d.annotate(
        "below chance\n(collapse-to-attractor)",
        xy=(0, auroc_mh_c), xytext=(0.3, auroc_mh_c + 0.18),
        arrowprops=dict(arrowstyle="->", color=CBLUE, lw=1.2),
        color=CBLUE, fontsize=7,
    )

    # Main title
    fig.text(
        0.5, 0.97,
        "Does openpilot know when it is blind?  --  E8: Hybrid OOD detector",
        ha="center", va="top", color=CTEXT, fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.935,
        "E6 catches collapse (AUROC 1.0) but misses corruption (AUROC ~0.6).  "
        "Maha catches corruption (AUROC 1.0) but misses collapse (AUROC 0.16).  "
        "Hybrid covers both -- but Maha FPR requires N >> 512 calib frames.",
        ha="center", va="top", color=CGREY, fontsize=8.5,
    )

    fig.savefig(FIG_OUT, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nFigure saved: {FIG_OUT}")


def main() -> int:
    # The canonical demo is now the PCA-reduced per-vehicle hybrid
    # (src/e8_pca_hybrid.py): it corrects the undersampled-Maha result (raw 512-D
    # FPR ~38% was a dimensionality artifact; PCA k=32 gives ~2% FPR + corruption
    # AUROC 1.0) and matches the repo figure style. Delegate so the demo command
    # produces the corrected report/figures/e8_demo.png. run_demo() (legacy
    # honest-negative writeup) is retained for reference and writes e8_demo_legacy.png.
    from src.e8_pca_hybrid import main as pca_main
    pca_main()
    return 0


if __name__ == "__main__":
    sys.exit(main())
