# Teardown results (v0.9.6)

Real: Subaru + RAM segments (320 frames each, 100 warmup discarded). CARLA: 320 captured clean-road frames. supercombo openpilot v0.9.6.

## E1  output collapse map

| head | real activity | CARLA activity | CARLA/real | collapsed elems | state |
|---|---|---|---|---|---|
| accel_t0 | 0.5412 | 0.2335 | 0.4315 | 0% | alive |
| desired_curv | 0.0811 | 0.0767 | 0.9461 | 0% | alive |
| lead_prob | 0.4460 | 0.0162 | 0.0363 | 100% | **COLLAPSED** |
| plan | 1343.4014 | 315.5723 | 0.2349 | 0% | alive |
| lane_lines | 517.7581 | 721.5779 | 1.3937 | 0% | alive |
| road_edges | 459.6908 | 335.2443 | 0.7293 | 0% | alive |
| lead | 297.6508 | 49.3049 | 0.1656 | 73% | alive |
| pose | 6.3677 | 0.7918 | 0.1243 | 100% | alive |
| desire_state | 0.0334 | 0.2278 | 6.8283 | 50% | alive |
| meta | 4.4454 | 2.1171 | 0.4763 | 25% | alive |

## E2  internal feature-space OOD

- CARLA feature spread is **0.44269x** the real spread (trace of `hidden_state` covariance).
- real vs CARLA separability **100.0%**, d' = **6.77** along the centroid-difference direction.

## E3  confidence response

| head | output retained | pred. unc. real | pred. unc. CARLA | unc. ratio | CARLA above real p95 |
|---|---|---|---|---|---|
| plan | 23.5% | 0.4322 | 0.4855 | 1.12x | 0% |
| lead | 16.6% | 1.7033 | 1.5751 | 0.92x | 0% |
| desired_curv | 94.6% | 0.0975 | 0.0939 | 0.96x | 14% |
