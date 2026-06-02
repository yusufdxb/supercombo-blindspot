# E6 corpus scaling: LOCO FPR with segment-level bootstrap

Clean-real calibration corpora (N=4): subaru, ram, ev6_night, bronco_night.
window=30, percentile=1.0. Rolling-spread monitor; FPR = fraction of a held-out corpus's frames flagged OOD by a threshold calibrated on the other corpora (leave-one-corpus-out).

## Per-held-out-corpus FPR

| held-out corpus | held-out FPR |
|---|---|
| subaru | 0.0276 |
| ram | 0.0690 |
| ev6_night | 0.0000 |
| bronco_night | 0.0000 |

**LOCO mean FPR: 2.41%** (segment-level bootstrap 95% CI [0.00%, 5.17%]); LOCO max 6.90%.

Before/after: the original N=2 estimate (subaru+ram only) gave LOCO mean 1.03% / max 2.07%. Scaling to N=4 clean corpora widens the held-out spread and gives the first cross-corpus CI.

## Real near-collapse (held out of calibration)

These are real recordings on which the monitor fires; they are NOT clean calibration data and are excluded from the LOCO set above. Scored at the all-clean threshold 0.087077:

| segment | fraction flagged |
|---|---|
| daytime_control | 0.6034 |

## Scope

N=4 clean real corpora is an honest lift from the prior N=2 but is not yet fleet-scale. The pipeline consumes any cached real hidden_state, so reaching N=30-50 is a matter of fetching + running additional public comma segments (scripts/fetch_upgrade_data.py pattern + GPU --collect); the segment-level bootstrap and LOCO machinery here are unchanged by adding corpora.
