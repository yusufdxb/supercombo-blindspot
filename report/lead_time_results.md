# Detection Lead-Time vs AUROC

For each detector calibrated at its 1% FPR operating point (99th-percentile
of ID scores), we report:

- **Single-corpus AUROC**: AUC on the standard eval split
  (ID = subaru+ram n=638, OOD = E4 alpha=1.0 CARLA n=319).
- **LOCO mean FPR**: mean false-positive rate across leave-one-corpus-out folds
  {subaru, ram}. Measures cross-corpus calibration stability.
- **fires_at_alpha**: smallest alpha in the E4 blend sweep where
  >50% of frames are flagged at the 1% threshold.
- **lead (blend-units)**: cliff_alpha - fires_at_alpha.
  Cliff onset = 0.784 (output-activity drop from >0.9 to <0.1
  of real baseline, established in report/e4_results.md).
  Positive lead = detector fires BEFORE the cliff (useful early warning).
  Negative lead = detector fires AFTER the cliff or not at all.
  Baseline detectors that calibrate to 100% LOCO FPR are NOT
  at their advertised 1% operating point cross-corpus; their fires-at
  values and apparent leads are invalid under deployment conditions.

## Summary table

| detector | single-corpus AUROC | LOCO mean FPR | fires_at_alpha | lead (blend-units) |
|---|---|---|---|---|
| E6 (rolling-spread) | 0.996 | 0.0103 | 0.550 | +0.234 |
| mahalanobis | 0.159 | 1.0000 | 0.150 | +0.634 |
| relative_mahalanobis | 0.934 | 1.0000 | 0.100 | +0.684 |
| knn50 | 1.000 | 1.0000 | 0.325 | +0.459 |
| conformal | 1.000 | 1.0000 | 0.325 | +0.459 |
| pca_mahalanobis | 0.152 | 0.1191 | never | n/a |

## Interpretation

A high single-corpus AUROC does NOT imply useful early warning, and a
positive apparent lead-time does NOT imply cross-corpus reliability.

KNN-50 and Conformal both achieve AUROC=1.000 and appear to fire at
alpha=0.325 (lead +0.459 blend-units). However both detectors have
LOCO mean FPR of 100%. This means the 1% threshold calibrated on one
corpus flags every frame of the held-out corpus as OOD. The apparent
lead is an artefact of threshold collapse: the detector is not at its
advertised 1% operating point cross-corpus, so the fires-at-alpha
value is meaningless under deployment conditions.

Relative Mahalanobis shows even earlier apparent firing (alpha=0.100,
lead +0.684) but with the same 100% LOCO FPR invalidation.

E6 fires at alpha=0.550 (lead +0.234 blend-units) and maintains LOCO
mean FPR of 1.03%, confirming that BOTH the lead-time AND the
calibration hold across corpora. This is the paper headline: E6 is
the only detector with a VERIFIED lead-time that survives cross-corpus
evaluation.

Conformal wraps the same KNN nonconformity score in a distribution-free
p-value framework. Under exchangeability the FPR guarantee is exact at
the chosen significance level (0.05 -> 5%). However, conformal LOCO
FPR = 100% confirms the exchangeability assumption fails: the
supercombo recurrent feature is non-exchangeable across corpora.
The conformal guarantee does not transfer. This is the same
location-sensitivity failure as raw KNN, now confirmed via the
distribution-free framework.

## Cliff reference

Cliff onset (output-activity cliff): alpha = 0.784
(from report/e4_results.md: activity crosses 0.9*baseline at
alpha=0.784 and 0.1*baseline at alpha=0.799; transition width 0.015).

