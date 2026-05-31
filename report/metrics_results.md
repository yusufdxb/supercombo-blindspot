# OOD Detection Metrics (threshold-free) + Bootstrap CIs

Five detectors evaluated on the supercombo recurrent feature: E6 (rolling spread on the 512-D state), three feature-space baselines from src/baselines.py (Mahalanobis, Relative Mahalanobis, KNN-50), and a PCA-Mahalanobis ablation (src/pca_mahalanobis.py). Eval split: ID = subaru + ram real driving (concatenated, n=638), OOD = E4 alpha=1.0 CARLA frames (n=319). Higher score = more OOD by convention; E6 scores are negated. Bootstrap: stratified by label, n=1000, seed=42.

## Table 1: threshold-free metrics with 95% CI

| detector | AUROC (mean [95% CI]) | AUPR (mean [95% CI]) | FPR@95TPR (mean [95% CI]) |
|---|---|---|---|
| E6 (rolling-spread) | 0.996 [0.992, 1.000] | 0.995 [0.990, 1.000] | 0.000 [0.000, 0.000] |
| Mahalanobis | 0.159 [0.130, 0.190] | 0.230 [0.217, 0.245] | 0.854 [0.828, 0.881] |
| Relative Mahalanobis | 0.934 [0.914, 0.952] | 0.732 [0.684, 0.784] | 0.067 [0.049, 0.088] |
| KNN-50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 [0.000, 0.000] |
| PCA-Mahalanobis | 0.152 [0.124, 0.179] | 0.214 [0.209, 0.219] | 0.854 [0.828, 0.881] |

## Table 2: AUROC across the E4 alpha sweep

| alpha | E6 | Mahalanobis | Rel-Mahalanobis | KNN-50 | PCA-Mahalanobis |
|---|---|---|---|---|---|
| 0.0000 | 0.686 | 0.465 | 0.525 | 0.279 | 0.458 |
| 0.1000 | 0.609 | 0.956 | 1.000 | 0.355 | 0.427 |
| 0.1500 | 0.540 | 0.994 | 1.000 | 0.491 | 0.389 |
| 0.2000 | 0.433 | 1.000 | 1.000 | 0.708 | 0.341 |
| 0.2250 | 0.399 | 1.000 | 1.000 | 0.793 | 0.308 |
| 0.2500 | 0.366 | 0.992 | 0.999 | 0.855 | 0.263 |
| 0.2750 | 0.331 | 0.976 | 0.998 | 0.896 | 0.235 |
| 0.3000 | 0.311 | 0.949 | 0.995 | 0.926 | 0.219 |
| 0.3250 | 0.294 | 0.917 | 0.992 | 0.952 | 0.214 |
| 0.3500 | 0.283 | 0.839 | 0.984 | 0.975 | 0.211 |
| 0.3750 | 0.303 | 0.778 | 0.979 | 0.990 | 0.208 |
| 0.4000 | 0.323 | 0.722 | 0.974 | 0.997 | 0.198 |
| 0.4250 | 0.400 | 0.582 | 0.962 | 1.000 | 0.176 |
| 0.4500 | 0.540 | 0.507 | 0.956 | 1.000 | 0.166 |
| 0.4750 | 0.605 | 0.462 | 0.952 | 1.000 | 0.178 |
| 0.5000 | 0.627 | 0.464 | 0.952 | 1.000 | 0.205 |
| 0.5250 | 0.693 | 0.507 | 0.956 | 1.000 | 0.233 |
| 0.5500 | 0.727 | 0.529 | 0.958 | 1.000 | 0.252 |
| 0.5750 | 0.784 | 0.537 | 0.959 | 1.000 | 0.265 |
| 0.6000 | 0.836 | 0.513 | 0.957 | 1.000 | 0.265 |
| 0.6500 | 0.891 | 0.470 | 0.954 | 1.000 | 0.251 |
| 0.6750 | 0.904 | 0.459 | 0.954 | 1.000 | 0.241 |
| 0.7000 | 0.911 | 0.435 | 0.953 | 1.000 | 0.230 |
| 0.7250 | 0.832 | 0.387 | 0.950 | 1.000 | 0.215 |
| 0.7500 | 0.824 | 0.357 | 0.949 | 1.000 | 0.209 |
| 0.7750 | 0.851 | 0.309 | 0.945 | 1.000 | 0.198 |
| 0.8000 | 0.912 | 0.264 | 0.941 | 1.000 | 0.185 |
| 0.9000 | 0.965 | 0.181 | 0.935 | 1.000 | 0.158 |
| 1.0000 | 0.996 | 0.159 | 0.934 | 1.000 | 0.152 |

## PCA-Mahalanobis LOCO FPR (1% target)

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 25.978046 | 0.0000 |
| ram | subaru | 24.011308 | 0.2382 |

**PCA-Mahalanobis LOCO mean FPR: 0.1191 (11.91%), max: 0.2382 (23.82%)**

## Headline

At alpha=1.0 the five detectors split into three camps. (1) E6 (0.996) and KNN-50 (1.000) achieve essentially perfect separation. (2) Relative Mahalanobis (0.934) separates well but lags. (3) Both vanilla Mahalanobis (0.159) and PCA-Mahalanobis (0.152) score BELOW chance: their AUROC sits at ~0.15, meaning CARLA-OOD frames produce LOWER Mahalanobis distance than real ID frames. This is consistent with E2: the recurrent feature freezes to a near-constant vector on CARLA, and that frozen vector happens to land in a high-density region of the ID Gaussian fit. Distance-from-mean cannot detect collapse-to-the-mean. Reconciling with Agent E's 100% LOCO finding: Mahalanobis and PCA-Mahalanobis separate poorly (and in the wrong direction at alpha=1.0), AND fail to calibrate across corpora; KNN separates perfectly at alpha=1.0 but still LOCO-fails at 100% because the absolute ram/subaru locations are further apart than the within-corpus radius. E6, which watches the second-order trace rather than absolute position, both separates and calibrates (LOCO mean FPR ~1.03%). PCA-Mahalanobis LOCO mean FPR drops to 11.91% (from 100%), a partial improvement but still well above the 1% target; PCA does NOT recover the calibration property. The paper-worthy framing: location-sensitive feature-space detectors fail on supercombo in two distinct modes (mean-collapse and cross-corpus drift), and a second-order monitor is needed.

