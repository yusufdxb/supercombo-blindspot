# LOCO FPR@95%TPR: fair cross-corpus operating point

Clean-real corpora (N=4): subaru, ram, ev6_night, bronco_night. Collapse set: E4 alpha=1.0 CARLA frames. Window=30. Operating point: threshold fixed at 95% TPR on the collapse set (per fold, under the calibration-corpus ID model), FPR measured on the held-out real corpus. Identical protocol for every detector.

## LOCO FPR@95%TPR per detector

| detector | LOCO mean FPR | 95% CI | LOCO max FPR | per-fold FPR |
|---|---|---|---|---|
| e6 | 0.00% | [0.00%, 0.00%] | 0.00% | subaru=0.00%, ram=0.00%, ev6_night=0.00%, bronco_night=0.00% |
| mahalanobis | 95.14% | [91.77%, 98.51%] | 100.00% | subaru=100.00%, ram=90.28%, ev6_night=96.24%, bronco_night=94.04% |
| relative_mahalanobis | 99.69% | [99.06%, 100.00%] | 100.00% | subaru=100.00%, ram=98.75%, ev6_night=100.00%, bronco_night=100.00% |
| knn50 | 60.82% | [35.89%, 85.74%] | 100.00% | subaru=100.00%, ram=71.47%, ev6_night=36.05%, bronco_night=35.74% |

## Reading

At a sensitivity-matched operating point (95% collapse detection), E6's second-order spread monitor false-positives on 0 of the held-out real corpora, because the collapse spread sits orders of magnitude below the real steady-driving floor. The location-based baselines separate collapse within a corpus but their absolute scores do not transfer across real corpora, so the same operating point misfires on held-out real driving. This is the percentile-free counterpart to the LOCO percentile FPR and isolates the cross-corpus calibration property.
