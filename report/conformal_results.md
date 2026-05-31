# Conformal OOD Detector Results

## Method

Split-conformal (inductive) OOD detector using KNN nonconformity score.
Calibration set: all ID frames (subaru + ram, n=638).
Nonconformity score: L2-normalised k=50 KNN distance to calibration set.
p-value: p(x) = #{i : alpha_i >= alpha(x)} / (n_cal + 1).
OOD score (reported): 1 - p-value. Higher = more OOD.

Two operating points:
- 1% FPR (99th percentile threshold, thr=0.986901)
  matching all other baselines in this report.
- 5% FPR (95th percentile threshold, thr=0.947027)
  the distribution-free conformal significance level alpha=0.05.

## Threshold-free metrics with 95% CI (n=1000 bootstrap, seed=42)

ID = subaru+ram (n=638), OOD = E4 alpha=1.0 CARLA (n=319).

| metric | mean [95% CI] |
|---|---|
| AUROC | 1.000 [1.000, 1.000] |
| AUPR | 1.000 [1.000, 1.000] |
| FPR@95TPR | 0.000 [0.000, 0.000] |

## LOCO held-out FPR (leave-one-corpus-out)

### At 1% operating point (99th percentile threshold)

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 0.983812 | 1.0000 |
| ram | subaru | 0.983812 | 1.0000 |

**LOCO mean FPR: 1.0000 (100.00%)**
**LOCO max FPR: 1.0000 (100.00%)**

### At 5% operating point (95th percentile threshold, conformal alpha=0.05)

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 0.944062 | 1.0000 |
| ram | subaru | 0.944062 | 1.0000 |

**LOCO mean FPR: 1.0000 (100.00%)**
**LOCO max FPR: 1.0000 (100.00%)**

## E4 alpha-sweep: fired fraction

- 1% threshold: detector fires (>50% flagged) at alpha = 0.325
- 5% threshold: detector fires (>50% flagged) at alpha = 0.300

| alpha | fired@1% | fired@5% |
|---|---|---|
| 0.0000 | 0.000 | 0.000 |
| 0.1000 | 0.000 | 0.000 |
| 0.1500 | 0.000 | 0.000 |
| 0.2000 | 0.031 | 0.100 |
| 0.2250 | 0.147 | 0.235 |
| 0.2500 | 0.282 | 0.401 |
| 0.2750 | 0.414 | 0.473 |
| 0.3000 | 0.470 | 0.530 |
| 0.3250 | 0.514 | 0.658 |
| 0.3500 | 0.589 | 0.815 |
| 0.3750 | 0.799 | 0.956 |
| 0.4000 | 0.950 | 0.997 |
| 0.4250 | 0.994 | 1.000 |
| 0.4500 | 1.000 | 1.000 |
| 0.4750 | 1.000 | 1.000 |
| 0.5000 | 1.000 | 1.000 |
| 0.5250 | 1.000 | 1.000 |
| 0.5500 | 1.000 | 1.000 |
| 0.5750 | 1.000 | 1.000 |
| 0.6000 | 1.000 | 1.000 |
| 0.6500 | 1.000 | 1.000 |
| 0.6750 | 1.000 | 1.000 |
| 0.7000 | 1.000 | 1.000 |
| 0.7250 | 1.000 | 1.000 |
| 0.7500 | 1.000 | 1.000 |
| 0.7750 | 1.000 | 1.000 |
| 0.8000 | 1.000 | 1.000 |
| 0.9000 | 1.000 | 1.000 |
| 1.0000 | 1.000 | 1.000 |

## Interpretation

The conformal detector achieves AUROC=1.000 and FPR@95TPR=0.000 on the
single-corpus eval (subaru+ram ID vs. CARLA OOD at alpha=1.0).
This perfect separation is inherited from the underlying KNN-50 score.

However, LOCO mean FPR = 100% at BOTH operating points. This means:
1. At the 1% threshold: every frame of the held-out real-driving corpus
   is flagged as OOD -- 100x the nominal rate.
2. At the 5% conformal significance level: same result.
   The distribution-free guarantee requires exchangeability between
   calibration and test; the 100% LOCO FPR directly falsifies
   exchangeability for the supercombo recurrent feature across corpora.

This is the same location-sensitivity failure as raw KNN-50. Conformal
wrapping does not rescue a score that is non-exchangeable across
deployment domains. The result strengthens the paper's claim: the
location-sensitivity failure is a property of the feature space, not
an artefact of the threshold calibration convention.

## Comparison to Table 1 (metrics_results.md)

| detector | AUROC | AUPR | FPR@95TPR | LOCO mean FPR |
|---|---|---|---|---|
| E6 (rolling-spread) | 0.996 [0.992, 1.000] | 0.995 [0.990, 1.000] | 0.000 [0.000, 0.000] | 0.0103 |
| KNN-50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.0000 |
| Conformal (KNN-50 nonconformity) | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] | 1.0000 |

KNN-50 and Conformal are effectively the same detector on this dataset:
both achieve perfect single-corpus separation and both fail LOCO at 100%.
The conformal p-value transformation is monotone in the KNN score, so
AUROC and FPR@95TPR are identical. The LOCO result confirms the
exchangeability assumption is violated at the feature-space level.

