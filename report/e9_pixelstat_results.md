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

## Table E9: per-readout activity ratio (CARLA / pooled real)

All 10 tracked readouts under every condition. Values below 0.01 are collapsed at the 1% threshold; values below 0.10 are suppressed at the 10% threshold.

| readout | CARLA (raw) | CARLA + mean/std match | CARLA + histogram match | CARLA + Fourier (FDA) match |
|---|---|---|---|---|
| `accel_t0` | 0.0040 | 0.1105 | 0.0478 | 0.0235 |
| `desired_curv` | 0.0018 | 0.0030 | 0.0020 | 0.0020 |
| `lead_prob` | 0.0058 | 0.0403 | 0.0380 | 0.0111 |
| `plan` | 0.0057 | 0.0435 | 0.0267 | 0.0132 |
| `lane_lines` | 0.0054 | 0.0383 | 0.0177 | 0.0079 |
| `road_edges` | 0.0076 | 0.0258 | 0.0143 | 0.0115 |
| `lead` | 0.0042 | 0.0453 | 0.0302 | 0.0181 |
| `pose` | 0.1788 | 0.2090 | 0.2015 | 0.1704 |
| `desire_state` | 0.0049 | 0.0140 | 0.0060 | 0.0052 |
| `meta` | 0.7181 | 0.6296 | 0.6324 | 0.6532 |

## Table E9b: intervention match quality

Each intervention targets a different statistic, so one summary number cannot validate all three. Absolute error against the pooled real reference, averaged over the 6 input channels.

| variant | mean err | std err | marginal distance | low-freq band err |
|---|---|---|---|---|
| CARLA (raw) | 46.305 | 15.615 | 46.643 | 55431.2 |
| CARLA + mean/std match | 0.078 | 0.192 | 6.157 | 10778.1 |
| CARLA + histogram match | 0.043 | 0.091 | 0.311 | 7639.7 |
| CARLA + Fourier (FDA) match | 0.613 | 0.625 | 3.521 | 1856.6 |
