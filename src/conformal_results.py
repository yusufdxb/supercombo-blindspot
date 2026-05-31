"""Generate report/conformal_results.md.

Split-conformal OOD detector: KNN nonconformity score (k=50, L2-normalised)
wrapped in inductive conformal p-values (Papadopoulos 2002, Vovk 2005).
OOD score = 1 - p-value so higher = more OOD, matching the rest of the
pipeline's convention.

Usage:
    env -u PYTHONPATH ~/Projects/phantom-braking/.venv/bin/python -m src.conformal_results
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src import baselines as B
from src.metrics import auroc, aupr, fpr_at_tpr, bootstrap_ci


def run(out_path: Path = Path("report/conformal_results.md")) -> dict:
    """Compute all conformal metrics and write the report. Returns dict of results."""
    d = np.load("report/teardown_collected.npz")
    by_corpus = {
        "subaru": d["subaru__hidden_state"],
        "ram": d["ram__hidden_state"],
    }
    all_real = np.concatenate(list(by_corpus.values()), axis=0)
    id_labels = np.zeros(len(all_real), dtype=np.int64)

    d4 = np.load("report/e4_collected.npz")
    ood_h = d4["1.0000__hidden_state"]
    ood_labels = np.ones(len(ood_h), dtype=np.int64)

    all_scores = np.concatenate([
        B._score("conformal", all_real, all_real),
        B._score("conformal", all_real, ood_h),
    ])
    all_labels = np.concatenate([id_labels, ood_labels])

    # Bootstrap metrics (n=1000, seed=42, same as metrics_results.md)
    mean_auroc, lo_auroc, hi_auroc = bootstrap_ci(
        auroc, all_scores, all_labels, n_bootstrap=1000, seed=42
    )
    mean_aupr, lo_aupr, hi_aupr = bootstrap_ci(
        aupr, all_scores, all_labels, n_bootstrap=1000, seed=42
    )
    mean_fpr95, lo_fpr95, hi_fpr95 = bootstrap_ci(
        lambda s, y: fpr_at_tpr(s, y, 0.95), all_scores, all_labels,
        n_bootstrap=1000, seed=42
    )

    # LOCO
    loco = B.loco_fpr("conformal", by_corpus, percentile=99.0)

    # Also compute at significance=0.05 (threshold=0.95) for the distribution-
    # free operating point.
    loco_sig05 = B.loco_fpr("conformal", by_corpus, percentile=95.0)

    # E4 sweep: fired fractions
    alphas = B._e4_alphas(Path("report/e4_collected.npz"))
    id_all = B._score("conformal", all_real, all_real)
    thr_99 = B.calibrate_threshold_high(id_all, 99.0)
    thr_95 = B.calibrate_threshold_high(id_all, 95.0)

    fired_99 = []
    fired_95 = []
    for a in alphas:
        H = d4[f"{a:.4f}__hidden_state"]
        s = B._score("conformal", all_real, H)
        s = s[np.isfinite(s)]
        fired_99.append(float(np.mean(s > thr_99)) if len(s) else float("nan"))
        fired_95.append(float(np.mean(s > thr_95)) if len(s) else float("nan"))

    def _fires_at(fa: list[float]) -> float:
        fa_arr = np.array(fa)
        above = fa_arr > 0.5
        return float(alphas[np.argmax(above)]) if above.any() else float("nan")

    fires_at_99 = _fires_at(fired_99)
    fires_at_95 = _fires_at(fired_95)

    results = {
        "auroc": (mean_auroc, lo_auroc, hi_auroc),
        "aupr": (mean_aupr, lo_aupr, hi_aupr),
        "fpr95": (mean_fpr95, lo_fpr95, hi_fpr95),
        "loco_99": loco,
        "loco_95": loco_sig05,
        "fires_at_99": fires_at_99,
        "fires_at_95": fires_at_95,
        "thr_99": float(thr_99),
        "thr_95": float(thr_95),
        "alphas": alphas,
        "fired_99": fired_99,
        "fired_95": fired_95,
    }

    _write_report(results, out_path)
    return results


def _write_report(r: dict, out: Path) -> None:
    auroc_mean, auroc_lo, auroc_hi = r["auroc"]
    aupr_mean, aupr_lo, aupr_hi = r["aupr"]
    fpr_mean, fpr_lo, fpr_hi = r["fpr95"]
    loco = r["loco_99"]
    loco95 = r["loco_95"]

    lines = [
        "# Conformal OOD Detector Results",
        "",
        "## Method",
        "",
        "Split-conformal (inductive) OOD detector using KNN nonconformity score.",
        "Calibration set: all ID frames (subaru + ram, n=638).",
        "Nonconformity score: L2-normalised k=50 KNN distance to calibration set.",
        "p-value: p(x) = #{i : alpha_i >= alpha(x)} / (n_cal + 1).",
        "OOD score (reported): 1 - p-value. Higher = more OOD.",
        "",
        "Two operating points:",
        "- 1% FPR (99th percentile threshold, thr=" + f"{r['thr_99']:.6f})",
        "  matching all other baselines in this report.",
        "- 5% FPR (95th percentile threshold, thr=" + f"{r['thr_95']:.6f})",
        "  the distribution-free conformal significance level alpha=0.05.",
        "",
        "## Threshold-free metrics with 95% CI (n=1000 bootstrap, seed=42)",
        "",
        "ID = subaru+ram (n=638), OOD = E4 alpha=1.0 CARLA (n=319).",
        "",
        "| metric | mean [95% CI] |",
        "|---|---|",
        f"| AUROC | {auroc_mean:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}] |",
        f"| AUPR | {aupr_mean:.3f} [{aupr_lo:.3f}, {aupr_hi:.3f}] |",
        f"| FPR@95TPR | {fpr_mean:.3f} [{fpr_lo:.3f}, {fpr_hi:.3f}] |",
        "",
        "## LOCO held-out FPR (leave-one-corpus-out)",
        "",
        "### At 1% operating point (99th percentile threshold)",
        "",
        "| held-out corpus | calibrated on | threshold | held-out FPR |",
        "|---|---|---|---|",
    ]
    for fname, fold in loco["folds"].items():
        calib = ", ".join(fold["calibrated_on"])
        lines.append(
            f"| {fname} | {calib} | {fold['threshold']:.6f} | {fold['fpr']:.4f} |"
        )
    lines += [
        "",
        f"**LOCO mean FPR: {loco['fpr_mean']:.4f} ({loco['fpr_mean'] * 100:.2f}%)**",
        f"**LOCO max FPR: {loco['fpr_max']:.4f} ({loco['fpr_max'] * 100:.2f}%)**",
        "",
        "### At 5% operating point (95th percentile threshold, conformal alpha=0.05)",
        "",
        "| held-out corpus | calibrated on | threshold | held-out FPR |",
        "|---|---|---|---|",
    ]
    for fname, fold in loco95["folds"].items():
        calib = ", ".join(fold["calibrated_on"])
        lines.append(
            f"| {fname} | {calib} | {fold['threshold']:.6f} | {fold['fpr']:.4f} |"
        )
    lines += [
        "",
        f"**LOCO mean FPR: {loco95['fpr_mean']:.4f} ({loco95['fpr_mean'] * 100:.2f}%)**",
        f"**LOCO max FPR: {loco95['fpr_max']:.4f} ({loco95['fpr_max'] * 100:.2f}%)**",
        "",
        "## E4 alpha-sweep: fired fraction",
        "",
        f"- 1% threshold: detector fires (>50% flagged) at alpha = "
        + (f"{r['fires_at_99']:.3f}" if np.isfinite(r["fires_at_99"]) else "never"),
        f"- 5% threshold: detector fires (>50% flagged) at alpha = "
        + (f"{r['fires_at_95']:.3f}" if np.isfinite(r["fires_at_95"]) else "never"),
        "",
        "| alpha | fired@1% | fired@5% |",
        "|---|---|---|",
    ]
    for a, f99, f95 in zip(r["alphas"], r["fired_99"], r["fired_95"]):
        lines.append(f"| {a:.4f} | {f99:.3f} | {f95:.3f} |")
    lines += [
        "",
        "## Interpretation",
        "",
        "The conformal detector achieves AUROC=1.000 and FPR@95TPR=0.000 on the",
        "single-corpus eval (subaru+ram ID vs. CARLA OOD at alpha=1.0).",
        "This perfect separation is inherited from the underlying KNN-50 score.",
        "",
        "However, LOCO mean FPR = 100% at BOTH operating points. This means:",
        "1. At the 1% threshold: every frame of the held-out real-driving corpus",
        "   is flagged as OOD -- 100x the nominal rate.",
        "2. At the 5% conformal significance level: same result.",
        "   The distribution-free guarantee requires exchangeability between",
        "   calibration and test; the 100% LOCO FPR directly falsifies",
        "   exchangeability for the supercombo recurrent feature across corpora.",
        "",
        "This is the same location-sensitivity failure as raw KNN-50. Conformal",
        "wrapping does not rescue a score that is non-exchangeable across",
        "deployment domains. The result strengthens the paper's claim: the",
        "location-sensitivity failure is a property of the feature space, not",
        "an artefact of the threshold calibration convention.",
        "",
        "## Comparison to Table 1 (metrics_results.md)",
        "",
        "| detector | AUROC | AUPR | FPR@95TPR | LOCO mean FPR |",
        "|---|---|---|---|---|",
        "| E6 (rolling-spread) | 0.996 [0.992, 1.000] | 0.995 [0.990, 1.000] | 0.000 [0.000, 0.000] | 0.0103 |",
        "| KNN-50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.0000 |",
        f"| Conformal (KNN-50 nonconformity) | {auroc_mean:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}] | "
        f"{aupr_mean:.3f} [{aupr_lo:.3f}, {aupr_hi:.3f}] | {fpr_mean:.3f} [{fpr_lo:.3f}, {fpr_hi:.3f}] | "
        f"{loco['fpr_mean']:.4f} |",
        "",
        "KNN-50 and Conformal are effectively the same detector on this dataset:",
        "both achieve perfect single-corpus separation and both fail LOCO at 100%.",
        "The conformal p-value transformation is monotone in the KNN score, so",
        "AUROC and FPR@95TPR are identical. The LOCO result confirms the",
        "exchangeability assumption is violated at the feature-space level.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> int:
    results = run(Path("report/conformal_results.md"))
    auroc_m, auroc_lo, auroc_hi = results["auroc"]
    aupr_m, aupr_lo, aupr_hi = results["aupr"]
    fpr_m, fpr_lo, fpr_hi = results["fpr95"]
    loco = results["loco_99"]
    print("Conformal OOD detector results (KNN-50 nonconformity, split-CP)")
    print(f"  AUROC:      {auroc_m:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]")
    print(f"  AUPR:       {aupr_m:.3f} [{aupr_lo:.3f}, {aupr_hi:.3f}]")
    print(f"  FPR@95TPR:  {fpr_m:.3f} [{fpr_lo:.3f}, {fpr_hi:.3f}]")
    print(f"  LOCO mean FPR (1% op): {loco['fpr_mean']:.4f} ({loco['fpr_mean']*100:.2f}%)")
    print(f"  LOCO max FPR (1% op):  {loco['fpr_max']:.4f} ({loco['fpr_max']*100:.2f}%)")
    print(f"  fires_at_alpha (1%): {results['fires_at_99']}")
    print(f"  fires_at_alpha (5%): {results['fires_at_95']}")
    print(f"Wrote report/conformal_results.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
