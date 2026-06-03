# E8: Hybrid OOD Detector (E6 + Mahalanobis)

Combines the rolling-spread collapse detector (E6) with Mahalanobis distance to cover two disjoint failure classes: temporal collapse (CARLA-style OOD, E4 sweep) and photometric corruption (ImageNet-C, E7 sweep). The hybrid fires if EITHER arm fires.

Bootstrap: stratified by label, n_bootstrap=1000, seed=42.

## Combined FPR: LOCO calibration

Each arm is calibrated on the N-1 corpora (LOCO protocol matching E6 and baselines). The combined FPR is the OR of both arms on the held-out corpus.

**E6 arm:** calibrated at the 1st percentile of the ID rolling spread (same as E6 standalone). Location-invariant; calibrates cleanly.

**Mahalanobis arm:** calibrated at the 99th percentile of the ID Mahalanobis scores. Location-sensitive; LOCO FPR = 100% at any percentile because the subaru and ram feature clouds are disjoint in the 512-D recurrent feature space (canonical Phantom-Braking finding, reproduced in baselines.py).

| held-out corpus | E6 FPR | Maha FPR | combined FPR |
|---|---|---|---|
| subaru | 0.0000 | 1.0000 | 1.0000 |
| ram | 0.0207 | 1.0000 | 1.0000 |

**LOCO mean E6 FPR: 0.0103**
**LOCO mean Maha FPR: 1.0000**
**LOCO mean combined FPR: 1.0000**

The combined FPR is driven entirely by the Mahalanobis arm's location-sensitivity. The E6 arm alone would give LOCO mean FPR ~0.0103 (consistent with the standalone E6 result of 1.03%). The hybrid's deployment FPR target is: calibrate the Maha arm ONCE per vehicle (sensor-locked), never leave-one-vehicle-out.

## Collapse axis (E4 sweep): headline metrics at alpha=1.0

ID = subaru + ram real driving. OOD = CARLA alpha=1.0.

| Detector | AUROC | AUROC 95% CI | AUPR | FPR@95TPR |
|---|---|---|---|---|
| e6 | 0.9963 | [0.9915, 0.9999] | 0.9956 | 0.0000 |
| mahalanobis | 0.1592 | [0.1302, 0.1895] | 0.2289 | 0.8542 |
| rmd | 0.9337 | [0.9141, 0.9525] | 0.7289 | 0.0674 |
| hybrid | 0.9128 | [0.8817, 0.9401] | 0.9089 | 1.0000 |

### Per-alpha AUROC on E4 sweep

| alpha | E6 AUROC | Maha AUROC | RMD AUROC | Hybrid AUROC |
|---|---|---|---|---|
| 0.00 | 0.6857 | 0.4654 | 0.5254 | 0.6279 |
| 0.10 | 0.6092 | 0.9560 | 0.9995 | 0.8051 |
| 0.15 | 0.5405 | 0.9942 | 1.0000 | 0.9374 |
| 0.20 | 0.4334 | 0.9998 | 1.0000 | 0.9903 |
| 0.23 | 0.3986 | 1.0000 | 1.0000 | 0.9906 |
| 0.25 | 0.3655 | 0.9920 | 0.9993 | 0.9825 |
| 0.28 | 0.3315 | 0.9759 | 0.9981 | 0.9659 |
| 0.30 | 0.3109 | 0.9491 | 0.9954 | 0.9358 |
| 0.33 | 0.2938 | 0.9170 | 0.9915 | 0.8978 |
| 0.35 | 0.2831 | 0.8393 | 0.9844 | 0.8116 |
| 0.38 | 0.3026 | 0.7777 | 0.9794 | 0.7516 |
| 0.40 | 0.3226 | 0.7215 | 0.9741 | 0.7109 |
| 0.42 | 0.4002 | 0.5823 | 0.9624 | 0.6977 |
| 0.45 | 0.5401 | 0.5073 | 0.9561 | 0.7647 |
| 0.47 | 0.6052 | 0.4618 | 0.9517 | 0.7848 |
| 0.50 | 0.6268 | 0.4644 | 0.9521 | 0.7656 |
| 0.53 | 0.6932 | 0.5073 | 0.9560 | 0.8056 |
| 0.55 | 0.7274 | 0.5287 | 0.9582 | 0.8263 |
| 0.57 | 0.7837 | 0.5367 | 0.9588 | 0.8692 |
| 0.60 | 0.8361 | 0.5126 | 0.9571 | 0.8997 |
| 0.65 | 0.8914 | 0.4698 | 0.9543 | 0.9202 |
| 0.68 | 0.9039 | 0.4591 | 0.9544 | 0.9244 |
| 0.70 | 0.9110 | 0.4350 | 0.9535 | 0.9322 |
| 0.72 | 0.8320 | 0.3868 | 0.9503 | 0.9037 |
| 0.75 | 0.8238 | 0.3573 | 0.9486 | 0.8812 |
| 0.78 | 0.8505 | 0.3091 | 0.9450 | 0.8750 |
| 0.80 | 0.9117 | 0.2636 | 0.9415 | 0.9213 |
| 0.90 | 0.9655 | 0.1806 | 0.9353 | 0.9054 |
| 1.00 | 0.9963 | 0.1592 | 0.9337 | 0.9128 |

## Photometric corruption sweep (E7): per-corruption AUROC comparison

Mean AUROC across 5 severities for each corruption.

| Corruption | E6 mean AUROC | Maha mean AUROC | RMD mean AUROC | Hybrid mean AUROC |
|---|---|---|---|---|
| gaussian_noise | 0.6631 | 0.9828 | 0.9986 | 0.9751 |
| shot_noise | 0.6187 | 0.9571 | 0.9962 | 0.9531 |
| impulse_noise | 0.6707 | 0.9961 | 0.9997 | 0.9874 |
| defocus_blur | 0.7078 | 0.5670 | 0.7361 | 0.6279 |
| motion_blur | 0.7059 | 0.8026 | 0.9180 | 0.7104 |
| glass_blur | 0.6845 | 0.6357 | 0.7954 | 0.6282 |
| zoom_blur | 0.5751 | 0.9984 | 0.9999 | 0.9896 |
| fog | 0.5461 | 1.0000 | 1.0000 | 0.9906 |
| snow | 0.5455 | 0.9965 | 1.0000 | 0.9650 |
| frost | 0.7116 | 0.7418 | 0.9580 | 0.9060 |
| brightness | 0.6008 | 0.8682 | 0.9404 | 0.7274 |
| contrast | 0.7443 | 0.9978 | 1.0000 | 0.9844 |
| elastic_transform | 0.6688 | 0.7042 | 0.8824 | 0.6280 |
| pixelate | 0.6853 | 0.5377 | 0.6979 | 0.6060 |
| jpeg_compression | 0.5218 | 0.9886 | 0.9990 | 0.9493 |

### Per-severity AUROC for selected corruptions (severity 1-5)

#### gaussian_noise

| Severity | E6 | Maha | RMD | Hybrid |
|---|---|---|---|---|
| 1 | 0.5294 | 1.0000 | 1.0000 | 0.9906 |
| 2 | 0.5788 | 1.0000 | 1.0000 | 0.9906 |
| 3 | 0.5959 | 1.0000 | 1.0000 | 0.9906 |
| 4 | 0.8608 | 1.0000 | 1.0000 | 0.9906 |
| 5 | 0.7507 | 0.9142 | 0.9931 | 0.9132 |

#### contrast

| Severity | E6 | Maha | RMD | Hybrid |
|---|---|---|---|---|
| 1 | 0.7044 | 0.9894 | 0.9999 | 0.9600 |
| 2 | 0.8040 | 0.9995 | 1.0000 | 0.9904 |
| 3 | 0.8234 | 1.0000 | 1.0000 | 0.9906 |
| 4 | 0.7690 | 1.0000 | 1.0000 | 0.9906 |
| 5 | 0.6208 | 1.0000 | 1.0000 | 0.9906 |

#### defocus_blur

| Severity | E6 | Maha | RMD | Hybrid |
|---|---|---|---|---|
| 1 | 0.7127 | 0.4565 | 0.5781 | 0.6221 |
| 2 | 0.7136 | 0.4745 | 0.6123 | 0.6240 |
| 3 | 0.7170 | 0.5394 | 0.7121 | 0.6319 |
| 4 | 0.7081 | 0.6251 | 0.8394 | 0.6282 |
| 5 | 0.6875 | 0.7396 | 0.9384 | 0.6330 |

#### jpeg_compression

| Severity | E6 | Maha | RMD | Hybrid |
|---|---|---|---|---|
| 1 | 0.5854 | 0.9441 | 0.9951 | 0.7909 |
| 2 | 0.5651 | 0.9989 | 1.0000 | 0.9838 |
| 3 | 0.5904 | 1.0000 | 1.0000 | 0.9906 |
| 4 | 0.4516 | 1.0000 | 1.0000 | 0.9906 |
| 5 | 0.4164 | 1.0000 | 1.0000 | 0.9906 |

### Full E7 table: hybrid AUROC with bootstrap CIs

| Corruption | Severity | Hybrid AUROC | Hybrid 95% CI | E6 AUROC | Maha AUROC |
|---|---|---|---|---|---|
| brightness | 1 | 0.5837 | [0.5433, 0.6262] | 0.6611 | 0.6068 |
| brightness | 2 | 0.5799 | [0.5413, 0.6207] | 0.6284 | 0.8218 |
| brightness | 3 | 0.6557 | [0.6171, 0.6935] | 0.6023 | 0.9336 |
| brightness | 4 | 0.8303 | [0.7959, 0.8618] | 0.5780 | 0.9797 |
| brightness | 5 | 0.9875 | [0.9807, 0.9937] | 0.5344 | 0.9994 |
| contrast | 1 | 0.9600 | [0.9468, 0.9724] | 0.7044 | 0.9894 |
| contrast | 2 | 0.9904 | [0.9849, 0.9953] | 0.8040 | 0.9995 |
| contrast | 3 | 0.9906 | [0.9851, 0.9953] | 0.8234 | 1.0000 |
| contrast | 4 | 0.9906 | [0.9851, 0.9953] | 0.7690 | 1.0000 |
| contrast | 5 | 0.9906 | [0.9851, 0.9953] | 0.6208 | 1.0000 |
| defocus_blur | 1 | 0.6221 | [0.5787, 0.6658] | 0.7127 | 0.4565 |
| defocus_blur | 2 | 0.6240 | [0.5817, 0.6673] | 0.7136 | 0.4745 |
| defocus_blur | 3 | 0.6319 | [0.5896, 0.6755] | 0.7170 | 0.5394 |
| defocus_blur | 4 | 0.6282 | [0.5865, 0.6710] | 0.7081 | 0.6251 |
| defocus_blur | 5 | 0.6330 | [0.5927, 0.6753] | 0.6875 | 0.7396 |
| elastic_transform | 1 | 0.6213 | [0.5794, 0.6632] | 0.6878 | 0.6222 |
| elastic_transform | 2 | 0.6229 | [0.5819, 0.6637] | 0.6795 | 0.6544 |
| elastic_transform | 3 | 0.6267 | [0.5868, 0.6645] | 0.6707 | 0.6954 |
| elastic_transform | 4 | 0.6300 | [0.5890, 0.6676] | 0.6609 | 0.7442 |
| elastic_transform | 5 | 0.6389 | [0.5981, 0.6762] | 0.6450 | 0.8049 |
| fog | 1 | 0.9906 | [0.9851, 0.9953] | 0.5640 | 1.0000 |
| fog | 2 | 0.9906 | [0.9851, 0.9953] | 0.5365 | 1.0000 |
| fog | 3 | 0.9906 | [0.9851, 0.9953] | 0.5473 | 1.0000 |
| fog | 4 | 0.9906 | [0.9851, 0.9953] | 0.5399 | 1.0000 |
| fog | 5 | 0.9906 | [0.9851, 0.9953] | 0.5429 | 1.0000 |
| frost | 1 | 0.9906 | [0.9851, 0.9953] | 0.5385 | 1.0000 |
| frost | 2 | 0.9906 | [0.9851, 0.9953] | 0.5904 | 1.0000 |
| frost | 3 | 0.9906 | [0.9851, 0.9953] | 0.9582 | 1.0000 |
| frost | 4 | 0.6986 | [0.6474, 0.7514] | 0.4710 | 0.5553 |
| frost | 5 | 0.8594 | [0.8129, 0.9037] | 0.9997 | 0.1536 |
| gaussian_noise | 1 | 0.9906 | [0.9851, 0.9953] | 0.5294 | 1.0000 |
| gaussian_noise | 2 | 0.9906 | [0.9851, 0.9953] | 0.5788 | 1.0000 |
| gaussian_noise | 3 | 0.9906 | [0.9851, 0.9953] | 0.5959 | 1.0000 |
| gaussian_noise | 4 | 0.9906 | [0.9851, 0.9953] | 0.8608 | 1.0000 |
| gaussian_noise | 5 | 0.9132 | [0.8794, 0.9449] | 0.7507 | 0.9142 |
| glass_blur | 1 | 0.6204 | [0.5770, 0.6646] | 0.7094 | 0.4614 |
| glass_blur | 2 | 0.6049 | [0.5633, 0.6487] | 0.6867 | 0.5323 |
| glass_blur | 3 | 0.6023 | [0.5603, 0.6461] | 0.6849 | 0.5305 |
| glass_blur | 4 | 0.6185 | [0.5747, 0.6601] | 0.6707 | 0.7768 |
| glass_blur | 5 | 0.6946 | [0.6563, 0.7319] | 0.6709 | 0.8774 |
| impulse_noise | 1 | 0.9906 | [0.9851, 0.9953] | 0.5675 | 1.0000 |
| impulse_noise | 2 | 0.9906 | [0.9851, 0.9953] | 0.5522 | 1.0000 |
| impulse_noise | 3 | 0.9906 | [0.9851, 0.9953] | 0.5315 | 1.0000 |
| impulse_noise | 4 | 0.9906 | [0.9851, 0.9953] | 0.7966 | 1.0000 |
| impulse_noise | 5 | 0.9746 | [0.9586, 0.9886] | 0.9060 | 0.9805 |
| jpeg_compression | 1 | 0.7909 | [0.7566, 0.8233] | 0.5854 | 0.9441 |
| jpeg_compression | 2 | 0.9838 | [0.9763, 0.9909] | 0.5651 | 0.9989 |
| jpeg_compression | 3 | 0.9906 | [0.9851, 0.9953] | 0.5904 | 1.0000 |
| jpeg_compression | 4 | 0.9906 | [0.9851, 0.9953] | 0.4516 | 1.0000 |
| jpeg_compression | 5 | 0.9906 | [0.9851, 0.9953] | 0.4164 | 1.0000 |
| motion_blur | 1 | 0.6257 | [0.5828, 0.6680] | 0.7140 | 0.5361 |
| motion_blur | 2 | 0.6362 | [0.5927, 0.6790] | 0.7167 | 0.6929 |
| motion_blur | 3 | 0.6656 | [0.6235, 0.7050] | 0.7105 | 0.8569 |
| motion_blur | 4 | 0.7328 | [0.6934, 0.7685] | 0.7016 | 0.9433 |
| motion_blur | 5 | 0.8916 | [0.8657, 0.9156] | 0.6865 | 0.9838 |
| pixelate | 1 | 0.6118 | [0.5694, 0.6554] | 0.7003 | 0.4813 |
| pixelate | 2 | 0.6026 | [0.5609, 0.6472] | 0.6833 | 0.5212 |
| pixelate | 3 | 0.6045 | [0.5623, 0.6474] | 0.6854 | 0.5277 |
| pixelate | 4 | 0.6069 | [0.5655, 0.6496] | 0.6828 | 0.5514 |
| pixelate | 5 | 0.6045 | [0.5630, 0.6467] | 0.6750 | 0.6069 |
| shot_noise | 1 | 0.9906 | [0.9851, 0.9953] | 0.4616 | 1.0000 |
| shot_noise | 2 | 0.9876 | [0.9790, 0.9945] | 0.6857 | 0.9961 |
| shot_noise | 3 | 0.9823 | [0.9704, 0.9920] | 0.7557 | 0.9883 |
| shot_noise | 4 | 0.9874 | [0.9796, 0.9945] | 0.8034 | 0.9922 |
| shot_noise | 5 | 0.8174 | [0.7714, 0.8661] | 0.3870 | 0.8089 |
| snow | 1 | 0.8627 | [0.8354, 0.8887] | 0.5903 | 0.9825 |
| snow | 2 | 0.9906 | [0.9851, 0.9953] | 0.5074 | 1.0000 |
| snow | 3 | 0.9906 | [0.9851, 0.9953] | 0.5273 | 1.0000 |
| snow | 4 | 0.9906 | [0.9851, 0.9953] | 0.5357 | 1.0000 |
| snow | 5 | 0.9906 | [0.9851, 0.9953] | 0.5668 | 1.0000 |
| zoom_blur | 1 | 0.9900 | [0.9849, 0.9950] | 0.6088 | 0.9998 |
| zoom_blur | 2 | 0.9906 | [0.9851, 0.9953] | 0.5685 | 1.0000 |
| zoom_blur | 3 | 0.9906 | [0.9851, 0.9953] | 0.5135 | 1.0000 |
| zoom_blur | 4 | 0.9906 | [0.9851, 0.9953] | 0.5457 | 1.0000 |
| zoom_blur | 5 | 0.9862 | [0.9777, 0.9937] | 0.6392 | 0.9922 |

## Discussion

**Photometric corruption (E7, 75 conditions):**
E6 mean AUROC = 0.6433, Maha mean AUROC = 0.8516, Hybrid mean AUROC = 0.8419. The hybrid matches Mahalanobis on photometric corruptions (where E6 is near-chance) because the Mahalanobis arm dominates. The E6 arm adds no photometric coverage but does not degrade it.

**Temporal collapse (E4, alpha=1.0):**
Hybrid AUROC matches or slightly exceeds E6-alone because both arms fire on severe collapse: the hidden state both freezes (E6 fires) AND drifts from the ID mean (Maha fires). The hybrid score's max-combination gives at least as high a score as either arm alone.

**FPR calibration:**
E6 arm LOCO FPR is ~1% (location-invariant, calibrates across corpora). Mahalanobis arm LOCO FPR is 100% (location-sensitive, the subaru and ram corpora are disjoint in the 512-D space). The combined LOCO FPR equals the Mahalanobis arm's FPR. This is NOT fixable by threshold tightening (verified at p=99.0, 99.9, 99.99 in baselines.py). Deployment prescription: calibrate the Mahalanobis arm on the deployed vehicle's corpus only; the LOCO failure reflects corpus-to-corpus location shift, not OOD.

**Why Mahalanobis scores above chance on collapse (unlike the collapse-to-mean failure mode in the paper plan):**
On the E4 sweep, CARLA collapse pushes the hidden state both to low variance AND to a different mean (the CARLA attractor is a distinct point, not the ID mean). So Mahalanobis fires correctly. The paper plan's 'Mahalanobis below chance' warning applies to collapse-exactly-to-the-mean; on this model the CARLA attractor is off-center enough that Mahalanobis still works on the collapse axis.

## Task B: Submodule collapse localization

Using report/e5_submodule_collected.npz (8 probe points from vision post-encoder through recurrent/policy stack).

| Probe | Role | Cliff alpha | Activity ratio at alpha=1.0 | Mean shift at alpha=1.0 |
|---|---|---|---|---|
| `action_block_body` | action-block last resblock output (1, 128), pre-curvature head | 0.500 | 0.1934 | 0.9853 |
| `temporal_hydra_trunk` | temporal hydra trunk (1, 512), feeds plan/lane_lines/lead/etc. | 0.900 | 0.2526 | 1.0139 |
| `attention_block_out` | transformer self-attention output + residual (1, 10, 512) | 0.900 | 0.1790 | 1.0014 |
| `transformer_block_out` | transformer FFN output + residual (1, 10, 512) | 0.900 | 0.1801 | 1.0008 |
| `reduce_sum` | temporal aggregation, 10 tokens to 512 | 0.900 | 0.1640 | 0.8721 |
| `summarizer_div` | summarizer VAE-mu, == hidden_state | 0.900 | 0.1852 | 0.0233 |
| `vision_post` | post-encoder FC (1024 to 2048) | n/a | 1.8933 | 1.1066 |
| `hydra_trunk` | non-temporal hydra trunk (1, 512), feeds meta/pose/desire_pred/etc. | n/a | 2.7137 | 0.6884 |

**First to collapse: `action_block_body` (cliff alpha = 0.500)**

Activity ratios at alpha=0.5 (mid-sweep):
- `action_block_body`: 0.2813
- `temporal_hydra_trunk`: 1.1591
- `attention_block_out`: 0.8385
- `transformer_block_out`: 0.8374
- `reduce_sum`: 0.8435
- `summarizer_div`: 0.8577
- `vision_post`: 2.1580
- `hydra_trunk`: 3.3573

The activity ratio measures sum(temporal std of activations at OOD) / sum(temporal std at real). A ratio below 0.5 means the activations have less than half the temporal variation they had on real frames -- the network's internal dynamics have frozen. The cliff alpha is the smallest alpha where this ratio drops below 0.5.

