# E6 Results: Self-Aware OOD Detector

## Threshold and false-positive rate

- threshold (calibrated on all real corpora, p=1.0): 0.078873
- in-sample FPR at this threshold (definitional, not a generalisation claim): 0.0115

## Held-out FPR (leave-one-corpus-out across {subaru, ram})

| held-out corpus | calibrated on | threshold | held-out FPR |
|---|---|---|---|
| subaru | ram | 0.061982 | 0.0000 |
| ram | subaru | 0.082197 | 0.0207 |

**LOCO mean FPR: 0.0103 (1.03%)**
**LOCO max FPR: 0.0207 (2.07%)**

This is the honest generalisation estimate: on a held-out corpus the detector did not see during threshold calibration.

## Detector response on the E4 sweep

- detector fires (>50% of frames flagged) at alpha = 0.550

| alpha | fired fraction |
|---|---|
| 0.0000 | 0.003 |
| 0.1000 | 0.000 |
| 0.1500 | 0.000 |
| 0.2000 | 0.000 |
| 0.2250 | 0.000 |
| 0.2500 | 0.000 |
| 0.2750 | 0.000 |
| 0.3000 | 0.000 |
| 0.3250 | 0.000 |
| 0.3500 | 0.000 |
| 0.3750 | 0.000 |
| 0.4000 | 0.000 |
| 0.4250 | 0.083 |
| 0.4500 | 0.190 |
| 0.4750 | 0.393 |
| 0.5000 | 0.338 |
| 0.5250 | 0.438 |
| 0.5500 | 0.517 |
| 0.5750 | 0.593 |
| 0.6000 | 0.724 |
| 0.6500 | 0.693 |
| 0.6750 | 0.734 |
| 0.7000 | 0.762 |
| 0.7250 | 0.759 |
| 0.7500 | 0.745 |
| 0.7750 | 0.745 |
| 0.8000 | 0.834 |
| 0.9000 | 0.883 |
| 1.0000 | 0.986 |
