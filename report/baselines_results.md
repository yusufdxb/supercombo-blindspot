# Baseline OOD Detectors: head-to-head with E6

Implements the post-hoc OOD baselines listed in docs/paper_plan.md section 1 on the same 512-D recurrent feature E6 watches. Same real-driving calibration corpus (subaru + ram), same LOCO protocol as report/e6_results.md. Higher score = more OOD; threshold is the 99th percentile of the ID score distribution (target FPR 1% by construction).

## Applicability of the six paper-plan baselines

| Baseline | Applies to supercombo? | Reason |
|---|---|---|
| MSP | No | No softmax classification head; lane_lines_prob is independent sigmoids, not a softmax. |
| Energy | No | No logits; output heads are Gaussian-mixture regressions and existence probabilities. |
| Mahalanobis | Yes | Single-Gaussian fit on the 512-D recurrent feature. |
| Relative Mahalanobis | Yes | Background = coarse 2-component GMM on the same ID features (documented in src/baselines.py). |
| KNN | Yes | k=50 L2-normalised distance to ID features. |
| ViM | No | Requires a classifier weight matrix + logits; supercombo has neither on the recurrent feature. |

## mahalanobis

- threshold (calibrated on all real corpora, p=99.0): 191.746897

### Held-out FPR (leave-one-corpus-out across {subaru, ram})

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 165.372067 | 1.0000 |
| ram | subaru | 124.577565 | 1.0000 |

**LOCO mean FPR: 1.0000 (100.00%)**
**LOCO max FPR: 1.0000 (100.00%)**

### Detector response on the E4 sweep

- detector fires (>50% of frames flagged) at alpha = 0.150

| alpha | fired fraction |
|---|---|
| 0.0000 | 0.000 |
| 0.1000 | 0.411 |
| 0.1500 | 0.793 |
| 0.2000 | 0.994 |
| 0.2250 | 1.000 |
| 0.2500 | 0.991 |
| 0.2750 | 0.972 |
| 0.3000 | 0.940 |
| 0.3250 | 0.903 |
| 0.3500 | 0.812 |
| 0.3750 | 0.740 |
| 0.4000 | 0.674 |
| 0.4250 | 0.511 |
| 0.4500 | 0.423 |
| 0.4750 | 0.370 |
| 0.5000 | 0.373 |
| 0.5250 | 0.423 |
| 0.5500 | 0.448 |
| 0.5750 | 0.458 |
| 0.6000 | 0.429 |
| 0.6500 | 0.379 |
| 0.6750 | 0.367 |
| 0.7000 | 0.339 |
| 0.7250 | 0.282 |
| 0.7500 | 0.248 |
| 0.7750 | 0.191 |
| 0.8000 | 0.138 |
| 0.9000 | 0.041 |
| 1.0000 | 0.016 |

## relative_mahalanobis

- threshold (calibrated on all real corpora, p=99.0): 1.563328

### Held-out FPR (leave-one-corpus-out across {subaru, ram})

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 5.365824 | 1.0000 |
| ram | subaru | 18.058466 | 1.0000 |

**LOCO mean FPR: 1.0000 (100.00%)**
**LOCO max FPR: 1.0000 (100.00%)**

### Detector response on the E4 sweep

- detector fires (>50% of frames flagged) at alpha = 0.100

| alpha | fired fraction |
|---|---|
| 0.0000 | 0.013 |
| 0.1000 | 0.987 |
| 0.1500 | 1.000 |
| 0.2000 | 1.000 |
| 0.2250 | 1.000 |
| 0.2500 | 0.991 |
| 0.2750 | 0.972 |
| 0.3000 | 0.940 |
| 0.3250 | 0.903 |
| 0.3500 | 0.812 |
| 0.3750 | 0.740 |
| 0.4000 | 0.674 |
| 0.4250 | 0.511 |
| 0.4500 | 0.423 |
| 0.4750 | 0.370 |
| 0.5000 | 0.373 |
| 0.5250 | 0.423 |
| 0.5500 | 0.448 |
| 0.5750 | 0.458 |
| 0.6000 | 0.429 |
| 0.6500 | 0.379 |
| 0.6750 | 0.367 |
| 0.7000 | 0.339 |
| 0.7250 | 0.282 |
| 0.7500 | 0.248 |
| 0.7750 | 0.191 |
| 0.8000 | 0.138 |
| 0.9000 | 0.041 |
| 1.0000 | 0.016 |

## knn50

- threshold (calibrated on all real corpora, p=99.0): 1.068717

### Held-out FPR (leave-one-corpus-out across {subaru, ram})

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 1.122270 | 1.0000 |
| ram | subaru | 0.750344 | 1.0000 |

**LOCO mean FPR: 1.0000 (100.00%)**
**LOCO max FPR: 1.0000 (100.00%)**

### Detector response on the E4 sweep

- detector fires (>50% of frames flagged) at alpha = 0.325

| alpha | fired fraction |
|---|---|
| 0.0000 | 0.000 |
| 0.1000 | 0.000 |
| 0.1500 | 0.000 |
| 0.2000 | 0.028 |
| 0.2250 | 0.147 |
| 0.2500 | 0.276 |
| 0.2750 | 0.414 |
| 0.3000 | 0.470 |
| 0.3250 | 0.514 |
| 0.3500 | 0.589 |
| 0.3750 | 0.799 |
| 0.4000 | 0.944 |
| 0.4250 | 0.994 |
| 0.4500 | 1.000 |
| 0.4750 | 1.000 |
| 0.5000 | 1.000 |
| 0.5250 | 1.000 |
| 0.5500 | 1.000 |
| 0.5750 | 1.000 |
| 0.6000 | 1.000 |
| 0.6500 | 1.000 |
| 0.6750 | 1.000 |
| 0.7000 | 1.000 |
| 0.7250 | 1.000 |
| 0.7500 | 1.000 |
| 0.7750 | 1.000 |
| 0.8000 | 1.000 |
| 0.9000 | 1.000 |
| 1.0000 | 1.000 |

## Comparison to E6

E6 fires at alpha = 0.550 with LOCO mean FPR 1.03% / max 2.07% (report/e6_results.md). The fires-at-alpha column below is the smallest alpha where >50% of frames are flagged.

| detector | LOCO mean FPR | LOCO max FPR | fires-at alpha |
|---|---|---|---|
| E6 (rolling-spread, ref) | 0.0103 | 0.0207 | 0.550 |
| mahalanobis | 1.0000 | 1.0000 | 0.150 |
| relative_mahalanobis | 1.0000 | 1.0000 | 0.100 |
| knn50 | 1.0000 | 1.0000 | 0.325 |

## Discussion

All three applicable feature-space baselines hit 100% LOCO held-out FPR, regardless of ridge regularisation or k. The subaru and ram calibration corpora occupy disjoint regions of the 512-D recurrent feature space: held out either way, every sample of the unseen corpus sits in the far tail of the other's ID distribution. KNN (L2-normalised) median distance ram-on-subaru is 1.34 vs subaru-self 99th percentile 0.75; Mahalanobis with reg=0.1 gives ram-on-subaru median 7670 vs subaru-self 99th percentile 125. This is not a regularisation bug; it is a property of supercombo's recurrent feature: it encodes per-platform state (Subaru vs Ram cabin and camera mount differ) at magnitudes that dwarf the within-platform variance.

The fires-at-alpha column for the baselines is therefore not directly comparable to E6's 0.550. Mahalanobis 'fires' at alpha=0.15 and KNN at alpha=0.325, but at thresholds that already flag 100% of the held-out real-driving corpus as OOD: the baselines are not calibrated to the same 1% FPR operating point as E6. Reading them as 'earlier detection' would be wrong.

E6's rolling-spread monitor is location-invariant (it watches the second-order trace of the feature, not its absolute position), and is the reason it passes LOCO at ~1% FPR where the location-sensitive baselines cannot. This is the headline result: on the supercombo recurrent feature, vanilla post-hoc feature-space OOD scores from the OpenOOD baseline set do not transfer; a second-order monitor does.

Caveats: (a) only two real-driving corpora are available for LOCO; with more corpora the per-platform shift might be averaged out. (b) Mahalanobis and RMD use a 10% trace shrinkage; results are qualitatively unchanged across reg in [1e-3, 10]. (c) The not-applicable verdict for MSP, Energy, and ViM is structural to supercombo's regression-head design, not a deficiency of those methods on classification models.

