# Teardown results

Real: Subaru + RAM segments (320 frames each, 100 warmup discarded). CARLA: 320 captured clean-road frames. supercombo openpilot v0.9.7.

## E1  output collapse map

| head | real activity | CARLA activity | CARLA/real | collapsed elems | state |
|---|---|---|---|---|---|
| accel_t0 | 0.3277 | 0.0013 | 0.0040 | 100% | **COLLAPSED** |
| desired_curv | 0.1318 | 0.0002 | 0.0018 | 100% | **COLLAPSED** |
| lead_prob | 0.1974 | 0.0011 | 0.0058 | 100% | **COLLAPSED** |
| plan | 1193.8486 | 6.8342 | 0.0057 | 100% | **COLLAPSED** |
| lane_lines | 267.9777 | 1.4401 | 0.0054 | 100% | **COLLAPSED** |
| road_edges | 280.3987 | 2.1272 | 0.0076 | 100% | **COLLAPSED** |
| lead | 232.3695 | 0.9693 | 0.0042 | 100% | **COLLAPSED** |
| pose | 5.3518 | 0.9572 | 0.1788 | 100% | alive |
| desire_state | 0.3312 | 0.0016 | 0.0049 | 100% | **COLLAPSED** |
| meta | 4.0278 | 2.8924 | 0.7181 | 24% | alive |

## E2  internal feature-space OOD

- CARLA feature spread is **0.00001x** the real spread (trace of `hidden_state` covariance).
- real vs CARLA separability **87.9%**, d' = **2.19** along the centroid-difference direction.

## E3  confidence response

| head | output retained | pred. unc. real | pred. unc. CARLA | unc. ratio | CARLA above real p95 |
|---|---|---|---|---|---|
| plan | 0.6% | 0.4103 | 0.5522 | 1.35x | 0% |
| lead | 0.4% | 1.2832 | 1.5412 | 1.20x | 0% |
| desired_curv | 0.2% | 0.0833 | 0.1534 | 1.84x | 0% |
