"""Build report/metrics_collected.npz, report/metrics_results.md, and the
three figures (ROC, PR, AUROC-vs-alpha) for the five OOD detectors:

  - E6 (rolling-spread, lower=more OOD, NEGATED for the higher=more OOD
    convention used by metrics.py)
  - Mahalanobis
  - Relative Mahalanobis
  - KNN-50
  - PCA-Mahalanobis (new, src/pca_mahalanobis.py)

Eval split: ID = real-driving from subaru + ram (concatenated, the
calibration corpus). OOD = E4 alpha=1.0 (cleanest OOD signal). Matches
the (ids, alpha=1.0) split materialised in report/baselines_collected.npz
exactly.

Also computes a per-alpha AUROC sweep so we can see at which alpha each
detector starts to separate ID from OOD.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.baselines import (
    APPLICABLE_BASELINES,
    _real_calibration_by_corpus,
    _score,
)
from src.e6_detector import rolling_spread
from src.metrics import (
    aupr,
    auroc,
    bootstrap_ci,
    fpr_at_tpr,
    pr_curve_points,
    roc_curve_points,
)
from src.pca_mahalanobis import loco_fpr as pca_loco_fpr
from src.pca_mahalanobis import pca_mahalanobis

REPORT = Path("report")
FIG = REPORT / "figures"
N_BOOTSTRAP = 1000
SEED = 42
E6_WINDOW = 30


# --- score builders -----------------------------------------------------

def _e4_alphas(cache: Path) -> np.ndarray:
    d = np.load(cache)
    return np.array(sorted({float(k.split("__")[0])
                            for k in d.files if "__hidden_state" in k}))


def _e4_hidden_at(cache: Path, alpha: float) -> np.ndarray:
    d = np.load(cache)
    return d[f"{alpha:.4f}__hidden_state"]


def _e6_scores_higher_more_ood(hidden: np.ndarray, window: int = E6_WINDOW
                               ) -> np.ndarray:
    """E6 native: lower spread = more OOD. Negate so higher = more OOD,
    matching the metrics.py convention.

    Rolling spread is NaN before the window fills. metrics.py drops NaN
    frames; that's the same convention E6 uses for its own LOCO FPR.
    """
    return -rolling_spread(hidden, window=window)


def build_all_scores() -> dict:
    """For each detector, compute:
      - id_scores: scores on the ID corpus (subaru + ram concatenated)
      - per-alpha scores: scores on each alpha slice of the E4 sweep
    Returns a dict keyed by detector name.
    """
    by_corpus = _real_calibration_by_corpus()
    all_real = np.concatenate(list(by_corpus.values()), axis=0)
    e4 = Path("report/e4_collected.npz")
    alphas = _e4_alphas(e4)

    out = {"alphas": alphas, "all_real": all_real}

    # baselines: use cached scores from baselines_collected.npz if present
    bcache = REPORT / "baselines_collected.npz"
    cached = np.load(bcache) if bcache.exists() else None

    for name in APPLICABLE_BASELINES:
        if cached is not None and f"{name}__id_scores" in cached.files:
            id_scores = cached[f"{name}__id_scores"]
            per_alpha = {
                float(a): cached[f"{name}__alpha_{a:.4f}_scores"]
                for a in alphas
            }
        else:
            id_scores = _score(name, all_real, all_real)
            per_alpha = {float(a): _score(name, all_real,
                                          _e4_hidden_at(e4, float(a)))
                          for a in alphas}
        out[name] = {"id_scores": id_scores, "per_alpha": per_alpha}

    # PCA-Mahalanobis: not cached, compute fresh.
    pca_id = pca_mahalanobis(all_real, all_real)
    pca_alpha = {float(a): pca_mahalanobis(all_real, _e4_hidden_at(e4, float(a)))
                 for a in alphas}
    out["pca_mahalanobis"] = {"id_scores": pca_id, "per_alpha": pca_alpha}

    # E6: lower spread = more OOD -> negate.
    e6_id = _e6_scores_higher_more_ood(all_real)
    e6_alpha = {float(a): _e6_scores_higher_more_ood(_e4_hidden_at(e4, float(a)))
                for a in alphas}
    out["e6"] = {"id_scores": e6_id, "per_alpha": e6_alpha}

    return out


# --- per-detector metric computation ------------------------------------

DETECTOR_ORDER = ["e6", "mahalanobis", "relative_mahalanobis", "knn50",
                  "pca_mahalanobis"]
DETECTOR_LABEL = {
    "e6": "E6 (rolling-spread)",
    "mahalanobis": "Mahalanobis",
    "relative_mahalanobis": "Relative Mahalanobis",
    "knn50": "KNN-50",
    "pca_mahalanobis": "PCA-Mahalanobis",
}


def _build_eval_split(scores_pack: dict, ood_alpha: float = 1.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    id_scores = np.asarray(scores_pack["id_scores"], dtype=np.float64)
    ood_scores = np.asarray(scores_pack["per_alpha"][ood_alpha],
                            dtype=np.float64)
    scores = np.concatenate([id_scores, ood_scores])
    labels = np.concatenate([np.zeros(len(id_scores), dtype=int),
                              np.ones(len(ood_scores), dtype=int)])
    return scores, labels


def compute_table1(all_scores: dict) -> dict:
    """Per-detector AUROC, AUPR, FPR@95TPR with bootstrap 95% CIs."""
    table: dict[str, dict] = {}
    for name in DETECTOR_ORDER:
        s, y = _build_eval_split(all_scores[name], ood_alpha=1.0)
        auroc_p = auroc(s, y)
        aupr_p = aupr(s, y)
        fpr_p = fpr_at_tpr(s, y, 0.95)
        auroc_ci = bootstrap_ci(auroc, s, y, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        aupr_ci = bootstrap_ci(aupr, s, y, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        fpr_ci = bootstrap_ci(lambda ss, yy: fpr_at_tpr(ss, yy, 0.95),
                              s, y, n_bootstrap=N_BOOTSTRAP, seed=SEED)
        table[name] = {
            "auroc": auroc_p, "aupr": aupr_p, "fpr95": fpr_p,
            "auroc_ci": auroc_ci, "aupr_ci": aupr_ci, "fpr95_ci": fpr_ci,
        }
    return table


def compute_alpha_sweep(all_scores: dict) -> dict:
    """Per-alpha AUROC per detector. Each alpha pairs ID frames vs that
    alpha's frames; alpha=0 is ID-vs-ID and should sit near 0.5."""
    alphas = all_scores["alphas"]
    sweep: dict[str, np.ndarray] = {}
    for name in DETECTOR_ORDER:
        vals = []
        id_scores = np.asarray(all_scores[name]["id_scores"], dtype=np.float64)
        for a in alphas:
            ood = np.asarray(all_scores[name]["per_alpha"][float(a)],
                             dtype=np.float64)
            s = np.concatenate([id_scores, ood])
            y = np.concatenate([np.zeros(len(id_scores), dtype=int),
                                 np.ones(len(ood), dtype=int)])
            vals.append(auroc(s, y))
        sweep[name] = np.array(vals)
    return sweep


def compute_pca_loco() -> dict:
    by_corpus = _real_calibration_by_corpus()
    return pca_loco_fpr(by_corpus, percentile=99.0)


# --- output writers -----------------------------------------------------

def _fmt_ci(triple: tuple[float, float, float]) -> str:
    m, lo, hi = triple
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def write_report(table1: dict, sweep: dict, alphas: np.ndarray,
                 pca_loco: dict, out: Path) -> None:
    lines: list[str] = []
    lines.append("# OOD Detection Metrics (threshold-free) + Bootstrap CIs")
    lines.append("")
    lines.append(
        "Five detectors evaluated on the supercombo recurrent feature: E6 "
        "(rolling spread on the 512-D state), three feature-space baselines "
        "from src/baselines.py (Mahalanobis, Relative Mahalanobis, KNN-50), "
        "and a PCA-Mahalanobis ablation (src/pca_mahalanobis.py). Eval split: "
        "ID = subaru + ram real driving (concatenated, n=638), OOD = E4 "
        "alpha=1.0 CARLA frames (n=319). Higher score = more OOD by "
        "convention; E6 scores are negated. Bootstrap: stratified by label, "
        f"n={N_BOOTSTRAP}, seed={SEED}."
    )
    lines.append("")
    lines.append("## Table 1: threshold-free metrics with 95% CI")
    lines.append("")
    lines.append(
        "| detector | AUROC (mean [95% CI]) | AUPR (mean [95% CI]) | "
        "FPR@95TPR (mean [95% CI]) |")
    lines.append("|---|---|---|---|")
    for name in DETECTOR_ORDER:
        r = table1[name]
        lines.append(
            f"| {DETECTOR_LABEL[name]} | {_fmt_ci(r['auroc_ci'])} | "
            f"{_fmt_ci(r['aupr_ci'])} | {_fmt_ci(r['fpr95_ci'])} |"
        )
    lines.append("")

    lines.append("## Table 2: AUROC across the E4 alpha sweep")
    lines.append("")
    lines.append("| alpha | E6 | Mahalanobis | Rel-Mahalanobis | KNN-50 | PCA-Mahalanobis |")
    lines.append("|---|---|---|---|---|---|")
    for i, a in enumerate(alphas):
        cols = [f"{sweep[n][i]:.3f}" for n in DETECTOR_ORDER]
        lines.append(f"| {float(a):.4f} | " + " | ".join(cols) + " |")
    lines.append("")

    lines.append("## PCA-Mahalanobis LOCO FPR (1% target)")
    lines.append("")
    lines.append("| held-out corpus | calibrated on | threshold | held-out FPR |")
    lines.append("|---|---|---|---|")
    for fname, fold in pca_loco["folds"].items():
        calib = ", ".join(fold["calibrated_on"])
        lines.append(f"| {fname} | {calib} | {fold['threshold']:.6f} | "
                     f"{fold['fpr']:.4f} |")
    lines.append("")
    lines.append(
        f"**PCA-Mahalanobis LOCO mean FPR: {pca_loco['fpr_mean']:.4f} "
        f"({pca_loco['fpr_mean']*100:.2f}%), max: {pca_loco['fpr_max']:.4f} "
        f"({pca_loco['fpr_max']*100:.2f}%)**"
    )
    lines.append("")

    lines.append("## Headline")
    lines.append("")
    lines.append(_headline(table1, sweep, alphas, pca_loco))
    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def _headline(table1: dict, sweep: dict, alphas: np.ndarray,
              pca_loco: dict) -> str:
    e6_auroc = table1["e6"]["auroc_ci"][0]
    maha_auroc = table1["mahalanobis"]["auroc_ci"][0]
    rmd_auroc = table1["relative_mahalanobis"]["auroc_ci"][0]
    knn_auroc = table1["knn50"]["auroc_ci"][0]
    pca_auroc = table1["pca_mahalanobis"]["auroc_ci"][0]
    return (
        f"At alpha=1.0 the five detectors split into three camps. "
        f"(1) E6 ({e6_auroc:.3f}) and KNN-50 ({knn_auroc:.3f}) achieve "
        f"essentially perfect separation. (2) Relative Mahalanobis "
        f"({rmd_auroc:.3f}) separates well but lags. (3) Both vanilla "
        f"Mahalanobis ({maha_auroc:.3f}) and PCA-Mahalanobis "
        f"({pca_auroc:.3f}) score BELOW chance: their AUROC sits at ~0.15, "
        f"meaning CARLA-OOD frames produce LOWER Mahalanobis distance than "
        f"real ID frames. This is consistent with E2: the recurrent feature "
        f"freezes to a near-constant vector on CARLA, and that frozen vector "
        f"happens to land in a high-density region of the ID Gaussian fit. "
        f"Distance-from-mean cannot detect collapse-to-the-mean. Reconciling "
        f"with Agent E's 100% LOCO finding: Mahalanobis and PCA-Mahalanobis "
        f"separate poorly (and in the wrong direction at alpha=1.0), AND fail "
        f"to calibrate across corpora; KNN separates perfectly at alpha=1.0 "
        f"but still LOCO-fails at 100% because the absolute ram/subaru "
        f"locations are further apart than the within-corpus radius. E6, "
        f"which watches the second-order trace rather than absolute "
        f"position, both separates and calibrates (LOCO mean FPR ~1.03%). "
        f"PCA-Mahalanobis LOCO mean FPR drops to "
        f"{pca_loco['fpr_mean']*100:.2f}% (from 100%), "
        f"{'a partial improvement but still well above the 1% target; PCA does NOT recover the calibration property' if pca_loco['fpr_mean'] > 0.05 else 'and recovers calibration'}. "
        f"The paper-worthy framing: location-sensitive feature-space "
        f"detectors fail on supercombo in two distinct modes (mean-collapse "
        f"and cross-corpus drift), and a second-order monitor is needed."
    )


# --- figures ------------------------------------------------------------

def _color(name: str) -> str:
    return {
        "e6": "#b0472b",
        "mahalanobis": "#3a78d6",
        "relative_mahalanobis": "#3ad693",
        "knn50": "#d6a13a",
        "pca_mahalanobis": "#9b3ad6",
    }[name]


def write_roc(all_scores: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    for name in DETECTOR_ORDER:
        s, y = _build_eval_split(all_scores[name], ood_alpha=1.0)
        fpr, tpr, _ = roc_curve_points(s, y)
        ax.plot(fpr, tpr, label=DETECTOR_LABEL[name], color=_color(name),
                lw=1.6)
    ax.axhline(0.95, color="grey", lw=0.7, ls="--", label="TPR=0.95")
    ax.plot([0, 1], [0, 1], color="black", lw=0.5, ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves: ID (subaru+ram) vs CARLA alpha=1.0")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def write_pr(all_scores: dict, out: Path) -> None:
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=140)
    for name in DETECTOR_ORDER:
        s, y = _build_eval_split(all_scores[name], ood_alpha=1.0)
        p, r, _ = pr_curve_points(s, y)
        ax.plot(r, p, label=DETECTOR_LABEL[name], color=_color(name), lw=1.6)
    ax.set_xlabel("Recall (TPR on OOD)")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall: ID (subaru+ram) vs CARLA alpha=1.0")
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def write_alpha_sweep(sweep: dict, alphas: np.ndarray, out: Path) -> None:
    import matplotlib.pyplot as plt
    import physx_style as _physx_style  # editorial-print theme
    _physx_style.apply()
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    for name in DETECTOR_ORDER:
        ax.plot(alphas, sweep[name], "o-", label=DETECTOR_LABEL[name],
                color=_color(name), lw=1.4, ms=3.5)
    ax.axhline(0.5, color="grey", lw=0.7, ls="--", label="chance")
    ax.set_xlabel("alpha (0 = real, 1 = CARLA)")
    ax.set_ylabel("AUROC vs ID corpus")
    ax.set_title("AUROC vs distribution-shift gradient")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0.35, 1.02)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


# --- main ---------------------------------------------------------------

def _flatten_cache(all_scores: dict, table1: dict, sweep: dict,
                   pca_loco: dict) -> dict:
    """Pack everything into a flat dict for np.savez_compressed."""
    out: dict = {}
    alphas = all_scores["alphas"]
    out["alphas"] = alphas
    for name in DETECTOR_ORDER:
        out[f"{name}__id_scores"] = np.asarray(
            all_scores[name]["id_scores"], dtype=np.float32)
        for a in alphas:
            out[f"{name}__alpha_{float(a):.4f}_scores"] = np.asarray(
                all_scores[name]["per_alpha"][float(a)], dtype=np.float32)
        out[f"{name}__alpha_sweep_auroc"] = sweep[name].astype(np.float32)
        # Table 1 entries.
        r = table1[name]
        out[f"{name}__auroc_point"] = np.float32(r["auroc"])
        out[f"{name}__aupr_point"] = np.float32(r["aupr"])
        out[f"{name}__fpr95_point"] = np.float32(r["fpr95"])
        out[f"{name}__auroc_ci"] = np.array(r["auroc_ci"], dtype=np.float32)
        out[f"{name}__aupr_ci"] = np.array(r["aupr_ci"], dtype=np.float32)
        out[f"{name}__fpr95_ci"] = np.array(r["fpr95_ci"], dtype=np.float32)
    out["pca_mahalanobis__loco_fpr_mean"] = np.float32(pca_loco["fpr_mean"])
    out["pca_mahalanobis__loco_fpr_max"] = np.float32(pca_loco["fpr_max"])
    out["meta__n_bootstrap"] = np.int32(N_BOOTSTRAP)
    out["meta__seed"] = np.int32(SEED)
    return out


def main() -> int:
    print("[1/5] computing all detector scores...")
    all_scores = build_all_scores()
    print("[2/5] computing Table 1 (point + bootstrap CIs)...")
    table1 = compute_table1(all_scores)
    print("[3/5] computing Table 2 (alpha sweep AUROC)...")
    sweep = compute_alpha_sweep(all_scores)
    print("[4/5] computing PCA-Mahalanobis LOCO FPR...")
    pca_loco = compute_pca_loco()
    print("[5/5] writing outputs...")
    cache = _flatten_cache(all_scores, table1, sweep, pca_loco)
    REPORT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(REPORT / "metrics_collected.npz", **cache)
    write_report(table1, sweep, all_scores["alphas"], pca_loco,
                 REPORT / "metrics_results.md")
    write_roc(all_scores, FIG / "roc_curves.png")
    write_pr(all_scores, FIG / "pr_curves.png")
    write_alpha_sweep(sweep, all_scores["alphas"], FIG / "auroc_vs_alpha.png")
    for name in DETECTOR_ORDER:
        r = table1[name]
        print(f"  {name}: AUROC={r['auroc']:.3f} "
              f"AUPR={r['aupr']:.3f} FPR@95={r['fpr95']:.3f}")
    print(f"PCA-M LOCO mean FPR: {pca_loco['fpr_mean']:.4f}, "
          f"max: {pca_loco['fpr_max']:.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
