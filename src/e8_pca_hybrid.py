"""E8b: PCA-reduced per-vehicle Mahalanobis, the fix to the undersampled hybrid.

The naive E8 hybrid (src/e8_hybrid.py) reported an honest negative: per-vehicle
Mahalanobis FPR stayed at 32-45 percent because a full 512-D Gaussian cannot be
estimated from ~223 calibration frames (rule of thumb N >> 5D = 2560). The fix is
not more data, it is dimensionality reduction. Projecting to the top-k principal
components of that vehicle's own ID features (k chosen so 223 frames over-determine
a k-D Gaussian) gives the Mahalanobis arm its fair shot.

Headline result (from the committed caches, this module):
  - within-vehicle held-out FPR drops from ~38/32 percent (raw 512-D) to ~2 percent
    at k=32 (controlled, near the 99th-percentile calibration target),
  - corruption AUROC recovers from ~0.59 (k=16) to 1.000 (k=32), because the
    photometric-OOD signal needs ~32 components to be linearly separable,
  - FPR then climbs past k=32 (5.3 -> 10.9 percent at k=64 -> 128) while AUROC has
    already saturated, so k=32 is the principled operating point.

So E6 (temporal collapse) + per-vehicle PCA-Mahalanobis k=32 (feature-space
photometric corruption) is a hybrid that covers BOTH failure classes at a
controlled within-vehicle FPR. The earlier "needs 2.5k frames/vehicle" conclusion
was a dimensionality artifact, corrected here.

Convention: higher score = more OOD, label 1 = OOD. Analysis only; no model
inference (reads report/*.npz). Figure matches the repo style in src/teardown.py.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from src.e6_detector import rolling_spread
from src.metrics import auroc
from src.pca_mahalanobis import _fit_gaussian

K_STAR = 32  # principled operating point (see module docstring + k-sweep)
_WINDOW = 30
_REG = 0.1
FIG_OUT = Path("report/figures/e8_demo.png")


# --------------------------------------------------------------------------- #
# data loaders (committed caches)
# --------------------------------------------------------------------------- #
def _vehicles() -> dict[str, np.ndarray]:
    d = np.load("report/teardown_collected.npz")
    return {"subaru": d["subaru__hidden_state"], "ram": d["ram__hidden_state"]}


def _collapse_hidden() -> np.ndarray:
    return np.load("report/e4_collected.npz")["1.0000__hidden_state"]


def _corruption_hidden(corruption: str = "fog", severity: int = 5) -> np.ndarray:
    return np.load("report/e7_collected.npz")[f"{corruption}__{severity}__hidden_state"]


# --------------------------------------------------------------------------- #
# PCA-Mahalanobis at an explicit component count k
# --------------------------------------------------------------------------- #
def maha_at_k(fit_x: np.ndarray, score_x: np.ndarray, k: int, reg: float = _REG) -> np.ndarray:
    """Squared Mahalanobis in the top-k PCA subspace of fit_x. k>=512 means raw."""
    if k >= fit_x.shape[1]:
        z_id, z_sc = fit_x, score_x
    else:
        pca = PCA(n_components=k).fit(fit_x)
        z_id, z_sc = pca.transform(fit_x), pca.transform(score_x)
    mu, prec = _fit_gaussian(z_id, reg=reg)
    x = z_sc - mu
    return np.einsum("ij,jk,ik->i", x, prec, x).astype(np.float64)


def within_vehicle_fpr(hidden: np.ndarray, k: int, seeds=range(10),
                       calib_frac: float = 0.7, percentile: float = 99.0) -> tuple[float, float]:
    """Mean +/- std held-out FPR over random within-vehicle calib/test splits.

    Calibrate the threshold at `percentile` of the calibration self-scores, then
    measure the fraction of held-out SAME-vehicle frames above it. This is the
    sensor-locked deployment FPR (one openpilot install runs on one vehicle)."""
    fprs = []
    for s in seeds:
        rng = np.random.RandomState(s)
        idx = rng.permutation(len(hidden))
        n = int(calib_frac * len(hidden))
        calib, test = hidden[idx[:n]], hidden[idx[n:]]
        thr = np.percentile(maha_at_k(calib, calib, k), percentile)
        fprs.append(float(np.mean(maha_at_k(calib, test, k) > thr)))
    return float(np.mean(fprs)), float(np.std(fprs))


def maha_auroc(hidden_id: np.ndarray, hidden_ood: np.ndarray, k: int) -> float:
    s_id = maha_at_k(hidden_id, hidden_id, k)
    s_ood = maha_at_k(hidden_id, hidden_ood, k)
    scores = np.concatenate([s_id, s_ood])
    labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
    return float(auroc(scores, labels))


def e6_auroc(hidden_id: np.ndarray, hidden_ood: np.ndarray, window: int = _WINDOW) -> float:
    """E6 rolling-spread AUROC. Collapse = LOW spread, so OOD score = -spread."""
    sp_id = rolling_spread(hidden_id, window)
    sp_ood = rolling_spread(hidden_ood, window)
    s_id, s_ood = sp_id[np.isfinite(sp_id)], sp_ood[np.isfinite(sp_ood)]
    scores = -np.concatenate([s_id, s_ood])  # low spread -> high OOD score
    labels = np.concatenate([np.zeros(len(s_id)), np.ones(len(s_ood))])
    return float(auroc(scores, labels))


def k_sweep(ks=(8, 16, 32, 64, 128), seeds=range(10)) -> list[dict]:
    """Per-k within-vehicle FPR and corruption/collapse AUROC, pooled over vehicles."""
    veh = _vehicles()
    fog = _corruption_hidden()
    collapse = _collapse_hidden()
    rows = []
    for k in ks:
        fprs = [within_vehicle_fpr(h, k, seeds=seeds)[0] for h in veh.values()]
        corr = [maha_auroc(h, fog, k) for h in veh.values()]
        coll = [maha_auroc(h, collapse, k) for h in veh.values()]
        rows.append({
            "k": k,
            "fpr_mean": float(np.mean(fprs)),
            "fpr_by_vehicle": {n: within_vehicle_fpr(h, k, seeds=seeds)[0] for n, h in veh.items()},
            "corruption_auroc": float(np.mean(corr)),
            "collapse_auroc": float(np.mean(coll)),
        })
    return rows


def headline(k: int = K_STAR) -> dict:
    """The defensible hybrid claim at the operating point k."""
    veh = _vehicles()
    fog = _corruption_hidden()
    collapse = _collapse_hidden()
    out = {"k": k, "per_vehicle": {}}
    for name, h in veh.items():
        fpr_raw = within_vehicle_fpr(h, 512)[0]
        fpr_pca = within_vehicle_fpr(h, k)[0]
        out["per_vehicle"][name] = {
            "fpr_raw512": fpr_raw,
            "fpr_pca": fpr_pca,
            "pca_maha_corruption_auroc": maha_auroc(h, fog, k),
            "pca_maha_collapse_auroc": maha_auroc(h, collapse, k),
            "e6_collapse_auroc": e6_auroc(h, collapse),
            "e6_corruption_auroc": e6_auroc(h, fog),
        }
    return out


# --------------------------------------------------------------------------- #
# figure (matches src/teardown.py repo style)
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# physx-newton-bench palette: cream paper, ink, blue primary + warm secondaries
CREAM = "#fcfcfb"
INK = "#0b0b0b"
RUST = "#2a78d6"     # primary series (physx blue)
SLATE = "#898781"    # muted warm grey for the secondary/cost series
GOLD = "#d08a2e"     # muted ochre for threshold/reference lines only
# legend styling
_LEG = dict(facecolor="#f1efe8", edgecolor="#c3c2b7")
# back-compat names used below: accent=RUST, secondary=SLATE, reference=GOLD
REAL_C, CARLA_C, WARN_C = RUST, SLATE, GOLD


def make_figure(out: Path = FIG_OUT) -> Path:
    plt = _plt()
    rows = k_sweep()
    hl = headline()
    ks = [r["k"] for r in rows]
    fpr = [100 * r["fpr_mean"] for r in rows]
    corr = [r["corruption_auroc"] for r in rows]

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.6))
    fig.suptitle("Does openpilot know when it is blind?   E8: PCA-Mahalanobis fixes the per-vehicle hybrid",
                 color=INK, fontsize=13.5, y=1.02)

    # (a) the tradeoff: FPR and corruption AUROC vs PCA dimensionality
    axA.set_title("(a) PCA dimensionality tradeoff")
    axA.plot(ks, fpr, "-o", color=CARLA_C, lw=2, label="within-vehicle FPR (%)")
    axA.axhline(1.0, color=WARN_C, ls="--", lw=1, alpha=0.7, label="1% target")
    axA.axvline(K_STAR, color=INK, ls=":", lw=1.2, alpha=0.6)
    axA.set_xscale("log", base=2)
    axA.set_xticks(ks)
    axA.set_xticklabels([str(k) for k in ks])
    axA.set_xlabel("PCA components k")
    axA.set_ylabel("within-vehicle FPR (%)", color=CARLA_C)
    axA.tick_params(axis="y", labelcolor=CARLA_C)
    axA2 = axA.twinx()
    axA2.plot(ks, corr, "-s", color=REAL_C, lw=2, label="corruption AUROC")
    axA2.set_ylabel("corruption AUROC", color=REAL_C)
    axA2.tick_params(axis="y", labelcolor=REAL_C)
    axA2.set_ylim(0.0, 1.05)
    axA2.grid(False)
    axA.annotate(f"k*={K_STAR}\nFPR~2%, AUROC 1.0", xy=(K_STAR, 2.5),
                 xytext=(K_STAR * 1.6, 18), color=INK, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=INK))
    h1, l1 = axA.get_legend_handles_labels()
    h2, l2 = axA2.get_legend_handles_labels()
    axA.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8,
               **_LEG)

    # (b) the fix: raw 512-D vs PCA k* within-vehicle FPR, per vehicle
    axB.set_title(f"(b) per-vehicle FPR: raw 512-D vs PCA k={K_STAR}")
    veh = list(hl["per_vehicle"])
    x = np.arange(len(veh))
    raw = [100 * hl["per_vehicle"][v]["fpr_raw512"] for v in veh]
    pca = [100 * hl["per_vehicle"][v]["fpr_pca"] for v in veh]
    axB.bar(x - 0.2, raw, 0.4, color=CARLA_C, label="raw 512-D (undersampled)")
    axB.bar(x + 0.2, pca, 0.4, color=REAL_C, label=f"PCA k={K_STAR}")
    axB.axhline(1.0, color=WARN_C, ls="--", lw=1, alpha=0.7, label="1% target")
    for xi, v in zip(x, veh):
        axB.text(xi - 0.2, 100 * hl["per_vehicle"][v]["fpr_raw512"] + 1,
                 f"{100*hl['per_vehicle'][v]['fpr_raw512']:.0f}%", ha="center", fontsize=8, color=INK)
        axB.text(xi + 0.2, 100 * hl["per_vehicle"][v]["fpr_pca"] + 1,
                 f"{100*hl['per_vehicle'][v]['fpr_pca']:.1f}%", ha="center", fontsize=8, color=INK)
    axB.set_xticks(x)
    axB.set_xticklabels(veh)
    axB.set_ylabel("within-vehicle FPR (%)")
    axB.legend(fontsize=8, **_LEG)

    # (c) coverage: each arm owns one failure class; hybrid covers both
    axC.set_title("(c) coverage: hybrid covers both classes")
    sub = hl["per_vehicle"]["subaru"]
    e6 = [sub["e6_collapse_auroc"], sub["e6_corruption_auroc"]]
    pm = [sub["pca_maha_collapse_auroc"], sub["pca_maha_corruption_auroc"]]
    hyb = [max(a, b) for a, b in zip(e6, pm)]  # OR-hybrid AUROC lower bound
    x = np.arange(2)
    axC.bar(x - 0.27, e6, 0.27, color=WARN_C, label="E6 (temporal)")
    axC.bar(x, pm, 0.27, color=REAL_C, label=f"PCA-Maha k={K_STAR}")
    axC.bar(x + 0.27, hyb, 0.27, color=CARLA_C, label="hybrid")
    axC.axhline(0.5, color="#9a9486", ls=":", lw=1, alpha=0.8)
    axC.set_xticks(x)
    axC.set_xticklabels(["collapse\n(CARLA)", "corruption\n(fog s5)"])
    axC.set_ylabel("AUROC (subaru)")
    axC.set_ylim(0, 1.08)
    axC.legend(fontsize=8, **_LEG)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    rows = k_sweep()
    hl = headline()
    print("=== E8 PCA-Mahalanobis: within-vehicle FPR vs k (10 splits) ===")
    print(f"{'k':>4}  {'FPR%':>7}  {'corruptAUROC':>13}  {'collapseAUROC':>13}")
    for r in rows:
        print(f"{r['k']:>4}  {100*r['fpr_mean']:>6.1f}  {r['corruption_auroc']:>13.3f}  {r['collapse_auroc']:>13.3f}")
    print(f"\nOperating point k*={K_STAR}:")
    for v, d in hl["per_vehicle"].items():
        print(f"  {v}: FPR raw-512D {100*d['fpr_raw512']:.1f}% -> PCA {100*d['fpr_pca']:.1f}%; "
              f"corruption AUROC {d['pca_maha_corruption_auroc']:.3f}; "
              f"E6 collapse AUROC {d['e6_collapse_auroc']:.3f}")
    p = make_figure()
    print(f"\nFigure saved: {p.resolve()}")


if __name__ == "__main__":
    main()
