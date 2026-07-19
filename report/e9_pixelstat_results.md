# E9  pixel-statistic control for the CARLA collapse

Scene content is held fixed; CARLA's 6-channel medmodel input has its per-channel pixel statistics pushed onto the pooled real (Subaru+RAM) distribution three ways: per-channel moment (mean/std) match (clipped to real's range, so approximate), full marginal-histogram match, and a low-frequency Fourier-amplitude band swap (FDA, beta=0.02, CARLA phase kept up to a final clip). If the E1-E3 collapse were a low-level pixel-statistic artifact it should lift under matching; the invariant part is whatever survives. All 10 tracked readouts (3 scalars derived from heads + 7 heads) are reported under both thresholds; `accel_t0` is extracted from `plan` and is not an independent head.

| variant | readouts <1% | readouts <10% | recurrent spread (xreal) | separability | max unc >real p95 |
|---|---|---|---|---|---|
| CARLA (raw) | 8/10 | 8/10 | 1.29e-05 | 87.9% | 0.0% |
| CARLA + mean/std match | 1/10 | 7/10 | 1.26e-05 | 87.9% | 0.5% |
| CARLA + histogram match | 2/10 | 8/10 | 1.29e-05 | 87.9% | 0.0% |
| CARLA + Fourier (FDA) match | 3/10 | 8/10 | 1.35e-05 | 87.9% | 0.0% |

## Reading

- Output activity partially recovers: readouts below 1% of real fall from 8/10 on raw CARLA to 1/10 under matching (at the 10% threshold most stay suppressed).
- The recurrent freeze survives: hidden-state spread stays 1.29e-05 -> 1.35e-05 of real and separability holds ~88%; exported uncertainty is not elevated under this metric, with at most 0.5% of frames on any head exceeding the real p95 (this measures p95 exceedance only, not that the uncertainty channel is literally flat).
- Interpretation: the tested low-level pixel statistics are excluded as a *sufficient* explanation for the recurrent-state freeze, but they partly explain the output quiescence. This is an invariance test, not renderer identification: geometry (zero- vs liveCalibration warp), phase, higher-order texture, and semantics remain confounded, and only one renderer and one CARLA sequence were tested.
