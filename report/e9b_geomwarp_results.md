# E9b  geometry (zero-warp) control for the CARLA collapse

The CARLA path warps to the medmodel frame with a zero-calibration euler; real segments use their liveCalibration euler (same intrinsics K, same get_warp_matrix construction). E9b holds the pixels real and swaps only that calibration euler, to isolate the calibration-warp confound the E9 pixel-statistic control leaves untouched. This isolates the preprocessing warp only; it does not equate the two cameras or their scene content. All 10 tracked readouts are reported under both thresholds.

| comparison | readouts <1% | readouts <10% | feature spread (xbaseline) | separability |
|---|---|---|---|---|
| A: real zero-warp vs real calibrated | 0/10 | 0/10 | 5.44e-01 | 89.4% |
| B: CARLA vs real zero-warp (identical warp) | 5/10 | 8/10 | 2.37e-05 | 75.0% |
| (ref) CARLA vs real calibrated | 8/10 | 8/10 | 1.29e-05 | 87.9% |

## Reading

- Real footage under the zero-calibration warp is not collapsed but representation-shifted: 0/10 readouts below 10% of the calibrated baseline and feature spread 0.54x of calibrated real (far above CARLA's freeze), yet 89.4% separable from the calibrated representation. The warp shifts the features without freezing them.
- CARLA still freezes against the identical-warp real baseline: 5/10 readouts below 1% and spread 2.37e-05 (the freeze also present in the reference against calibrated real: 8/10, 1.29e-05). The below-1% counts differ (5 vs 8) because the zero-warped real baseline re-normalises per-readout activity; the recurrent freeze is the invariant.
- Interpretation: on these sequences the zero- vs liveCalibration warp is not sufficient by itself to explain the freeze. It is not equated with camera geometry or content; renderer, content, and semantic differences remain confounded.
