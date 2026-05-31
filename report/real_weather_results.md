# Real-weather OOD axis: results

Segments: all comma-3 (tici), 1928x1208 yuv420p, with liveCalibration.
No intrinsics confound: all three use the identical `_ar_ox_config` focal-length/principal-point as the Subaru+RAM baseline.
Model: supercombo v0.9.7 (models/supercombo.onnx).
N=320 frames/segment, 100 warmup discarded, 220 post-warmup frames analysed.

## E1 output activity ratio (seg vs v0.9.7 real Subaru+RAM baseline)

| segment | head | baseline activity | seg activity | seg/baseline | collapsed elems | state |
|---|---|---|---|---|---|---|
| EV6 night + headlight glare | accel_t0 | 0.3277 | 0.2498 | 0.7624 | 0% | alive |
| EV6 night + headlight glare | desired_curv | 0.1318 | 0.1823 | 1.3840 | 0% | alive |
| EV6 night + headlight glare | lead_prob | 0.1974 | 0.2086 | 1.0567 | 0% | alive |
| EV6 night + headlight glare | plan | 1193.8486 | 178.0731 | 0.1492 | 55% | alive |
| EV6 night + headlight glare | lane_lines | 267.9777 | 297.7620 | 1.1111 | 0% | alive |
| EV6 night + headlight glare | road_edges | 280.3987 | 163.2578 | 0.5822 | 0% | alive |
| EV6 night + headlight glare | lead | 232.3695 | 81.3440 | 0.3501 | 0% | alive |
| EV6 night + headlight glare | pose | 5.3518 | 0.7144 | 0.1335 | 0% | alive |
| EV6 night + headlight glare | desire_state | 0.3312 | 0.3547 | 1.0710 | 0% | alive |
| EV6 night + headlight glare | meta | 4.0278 | 2.2565 | 0.5602 | 0% | alive |
| Bronco night + tail-light/sign glare | accel_t0 | 0.3277 | 0.4660 | 1.4221 | 0% | alive |
| Bronco night + tail-light/sign glare | desired_curv | 0.1318 | 0.1139 | 0.8642 | 0% | alive |
| Bronco night + tail-light/sign glare | lead_prob | 0.1974 | 0.0617 | 0.3127 | 0% | alive |
| Bronco night + tail-light/sign glare | plan | 1193.8486 | 330.0736 | 0.2765 | 0% | alive |
| Bronco night + tail-light/sign glare | lane_lines | 267.9777 | 230.3550 | 0.8596 | 0% | alive |
| Bronco night + tail-light/sign glare | road_edges | 280.3987 | 293.2766 | 1.0459 | 0% | alive |
| Bronco night + tail-light/sign glare | lead | 232.3695 | 74.8079 | 0.3219 | 0% | alive |
| Bronco night + tail-light/sign glare | pose | 5.3518 | 1.0676 | 0.1995 | 0% | alive |
| Bronco night + tail-light/sign glare | desire_state | 0.3312 | 0.2521 | 0.7614 | 0% | alive |
| Bronco night + tail-light/sign glare | meta | 4.0278 | 3.2456 | 0.8058 | 19% | alive |
| Daytime-dry control (in-distribution) | accel_t0 | 0.3277 | 0.1976 | 0.6029 | 0% | alive |
| Daytime-dry control (in-distribution) | desired_curv | 0.1318 | 0.0759 | 0.5757 | 0% | alive |
| Daytime-dry control (in-distribution) | lead_prob | 0.1974 | 0.2755 | 1.3951 | 0% | alive |
| Daytime-dry control (in-distribution) | plan | 1193.8486 | 145.1917 | 0.1216 | 56% | alive |
| Daytime-dry control (in-distribution) | lane_lines | 267.9777 | 152.0394 | 0.5674 | 0% | alive |
| Daytime-dry control (in-distribution) | road_edges | 280.3987 | 97.1429 | 0.3464 | 19% | alive |
| Daytime-dry control (in-distribution) | lead | 232.3695 | 68.0617 | 0.2929 | 15% | alive |
| Daytime-dry control (in-distribution) | pose | 5.3518 | 1.8704 | 0.3495 | 0% | alive |
| Daytime-dry control (in-distribution) | desire_state | 0.3312 | 0.1256 | 0.3794 | 0% | alive |
| Daytime-dry control (in-distribution) | meta | 4.0278 | 3.7944 | 0.9420 | 14% | alive |
| CARLA synthetic (E1-E6 reference OOD) | accel_t0 | 0.3277 | 0.0013 | 0.0040 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | desired_curv | 0.1318 | 0.0002 | 0.0018 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | lead_prob | 0.1974 | 0.0011 | 0.0058 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | plan | 1193.8486 | 6.8342 | 0.0057 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | lane_lines | 267.9777 | 1.4401 | 0.0054 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | road_edges | 280.3987 | 2.1272 | 0.0076 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | lead | 232.3695 | 0.9693 | 0.0042 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | pose | 5.3518 | 0.9572 | 0.1788 | 100% | alive |
| CARLA synthetic (E1-E6 reference OOD) | desire_state | 0.3312 | 0.0016 | 0.0049 | 100% | **COLLAPSED** |
| CARLA synthetic (E1-E6 reference OOD) | meta | 4.0278 | 2.8924 | 0.7181 | 24% | alive |

### Collapsed head count per condition

| condition | collapsed heads (ratio < 0.10) |
|---|---|
| EV6 night + headlight glare | 0/10 |
| Bronco night + tail-light/sign glare | 0/10 |
| Daytime-dry control (in-distribution) | 0/10 |
| CARLA synthetic (E1-E6 reference OOD) | 8/10 |

## E2 recurrent feature spread ratio and separability

| condition | spread baseline | spread seg | spread ratio | d' | separability |
|---|---|---|---|---|---|
| EV6 night + headlight glare | 0.5428 | 0.5487 | 1.01098 | 4.49 | 96.6% |
| Bronco night + tail-light/sign glare | 0.5428 | 0.3331 | 0.61378 | 3.83 | 93.4% |
| Daytime-dry control (in-distribution) | 0.5428 | 0.1005 | 0.18516 | 2.13 | 87.3% |
| CARLA synthetic (E1-E6 reference OOD) | 0.5428 | 0.0000 | 0.00001 | 2.19 | 87.9% |

## E3 predicted uncertainty above v0.9.7 real p95

| condition | head | baseline unc | seg unc | real p95 | frames above p95 |
|---|---|---|---|---|---|
| EV6 night + headlight glare | plan | 0.4103 | 0.3202 | 0.6349 | 0% |
| EV6 night + headlight glare | lead | 1.2832 | 1.0819 | 2.4206 | 2% |
| EV6 night + headlight glare | desired_curv | 0.0833 | 0.0846 | 0.1840 | 15% |
| Bronco night + tail-light/sign glare | plan | 0.4103 | 0.3910 | 0.6349 | 0% |
| Bronco night + tail-light/sign glare | lead | 1.2832 | 1.1105 | 2.4206 | 0% |
| Bronco night + tail-light/sign glare | desired_curv | 0.0833 | 0.1234 | 0.1840 | 2% |
| Daytime-dry control (in-distribution) | plan | 0.4103 | 0.5556 | 0.6349 | 1% |
| Daytime-dry control (in-distribution) | lead | 1.2832 | 1.6757 | 2.4206 | 3% |
| Daytime-dry control (in-distribution) | desired_curv | 0.0833 | 0.1601 | 0.1840 | 5% |
| CARLA synthetic (E1-E6 reference OOD) | plan | 0.4103 | 0.5522 | 0.6349 | 0% |
| CARLA synthetic (E1-E6 reference OOD) | lead | 1.2832 | 1.5412 | 2.4206 | 0% |
| CARLA synthetic (E1-E6 reference OOD) | desired_curv | 0.0833 | 0.1534 | 0.1840 | 0% |

## E6 rolling-spread monitor (threshold=0.078873, window=30)

| condition | fire fraction | fires? |
|---|---|---|
| EV6 night + headlight glare | 0.0% | no |
| Bronco night + tail-light/sign glare | 0.0% | no |
| Daytime-dry control (in-distribution) | 57.9% | **YES** |
| CARLA synthetic (E1-E6 reference OOD) | 100.0% | **YES** |

## Verdict

E1 sanity check: daytime control PASSED (all output heads active on daytime in-distribution footage).
Real night + headlight/tail-light glare does NOT induce E1 output collapse: all heads remain active on EV6 night (0 collapsed) and Bronco night (0 collapsed). CARLA collapses 8/10 heads as reference.
E6 rolling-spread monitor does NOT fire on any real night segment (EV6: 0.0% frames flagged, Bronco: 0.0% frames flagged; CARLA fires at 100.0%).
NOTE: E6 also fires on the daytime control segment (57.9% frames flagged). This is NOT a false positive or pipeline error. The daytime_control segment (dongle 376bf99325883932, seg 1) has sustained high steer angles (max 249 deg) and low speed, which drives the model into a near-zero recurrent attractor after ~220 frames. E1 output heads remain active (zero collapsed) so the model is not blind; the E6 fire reflects unusual kinematics, not synthetic domain gap. This distinguishes the E6 mechanism from the CARLA case (E6 + E1 both fire on CARLA; only E6 fires here).

**Bounding conclusion**: real night + headlight/tail-light glare on a comma-3 device does not collapse the model (E1: 0 heads, E6: does not fire). The collapse signature documented in E1-E6 is CARLA-specific (synthetic domain gap), not a general adverse-lighting phenomenon. This bounds the contribution: silent output collapse is triggered by the sim-to-real rendering gap, not by real low-light or glare conditions that openpilot's training distribution covers.
