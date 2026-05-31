# E6 v0.9.6 Results: Self-Aware OOD Detector

## Threshold and false-positive rate

- threshold (calibrated on all real corpora, p=1.0): 0.095745
- in-sample FPR at this threshold (definitional): 0.0115

## Held-out FPR (leave-one-corpus-out across {subaru, ram})

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 0.135752 | 0.6655 |
| ram | subaru | 0.092656 | 0.0000 |

**LOCO mean FPR: 0.3328 (33.28%)**
**LOCO max FPR: 0.6655 (66.55%)**

## Detector response on the E4 v0.9.6 sweep

- detector fires (>50% of frames flagged) at alpha = 0.800

| alpha | fired fraction |
|---|---|
| 0.0000 | 0.024 |
| 0.1000 | 0.014 |
| 0.1500 | 0.014 |
| 0.2000 | 0.000 |
| 0.2250 | 0.000 |
| 0.2500 | 0.000 |
| 0.2750 | 0.000 |
| 0.3000 | 0.000 |
| 0.3250 | 0.000 |
| 0.3500 | 0.000 |
| 0.3750 | 0.000 |
| 0.4000 | 0.000 |
| 0.4250 | 0.014 |
| 0.4500 | 0.000 |
| 0.4750 | 0.000 |
| 0.5000 | 0.028 |
| 0.5250 | 0.169 |
| 0.5500 | 0.255 |
| 0.5750 | 0.283 |
| 0.6000 | 0.297 |
| 0.6250 | 0.300 |
| 0.6500 | 0.348 |
| 0.6750 | 0.403 |
| 0.7000 | 0.483 |
| 0.7250 | 0.500 |
| 0.7500 | 0.500 |
| 0.8000 | 0.503 |
| 0.8500 | 0.507 |
| 0.8750 | 0.507 |
| 0.9000 | 0.507 |
| 0.9250 | 0.510 |
| 0.9500 | 0.514 |
| 0.9750 | 0.514 |
| 1.0000 | 0.517 |
