# Hyperparameter Sensitivity Ablations

Ablation sweeps for the two key hyperparameters flagged in the publication-readiness audit: KNN k (neighbour count) and E6 window (rolling-spread window size). Eval split: ID = subaru + ram real driving (concatenated), OOD = E4 alpha=1.0 CARLA frames. Bootstrap: stratified by label, n=1000, seed=42.

## KNN k-sensitivity

KNN OOD score = L2-normalised distance to the k-th nearest neighbour in the ID feature set (Sun et al., ICML 2022). Higher = more OOD. Default k=50 in build_metrics.py.

| k | AUROC (mean [95% CI]) |
|---|---|
| 5 | 1.000 [1.000, 1.000] |
| 10 | 1.000 [1.000, 1.000] |
| 20 | 1.000 [1.000, 1.000] |
| 50 | 1.000 [1.000, 1.000] |
| 100 | 1.000 [1.000, 1.000] |

## E6 window-size sensitivity

E6 monitor = rolling trace(Var) of the 512-D recurrent feature over a sliding window. Lower spread = more OOD (negated for AUROC convention). fires-at-alpha = smallest alpha where >50% of frames flagged. Default window=30 in build_metrics.py.

| window | AUROC (mean [95% CI]) | fires-at-alpha |
|---|---|---|
| 10 | 0.957 [0.940, 0.972] | 1.000 |
| 20 | 0.983 [0.972, 0.992] | 1.000 |
| 30 | 0.996 [0.992, 1.000] | 0.550 |
| 50 | 1.000 [1.000, 1.000] | 0.525 |

## Summary

KNN AUROC range across k in {5, 10, 20, 50, 100}: 0.0000 (min 1.000, max 1.000). E6 AUROC range across window in {10, 20, 30, 50}: 0.0431 (min 0.957, max 1.000). Both detectors show low sensitivity to their primary hyperparameter at alpha=1.0.

