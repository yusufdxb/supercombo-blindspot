# Evidence Register

Last audited: 2026-07-17

This ledger defines the strongest public wording supported by committed artifacts. `Verified` means
the result is reproduced in a committed result file or cache. It does not mean independently
replicated, peer reviewed, deployed on road, or accepted for publication.

| ID | Status | Scoped claim | Primary evidence | Required caveat |
|---|---|---|---|---|
| C1 | Verified | The v0.9.7 reimplementation matched comma's reference acceleration output on 100% of 1,159 post-warmup frames within +/-0.5 m/s2; median absolute delta was 0.04 m/s2. | `report/parity_results.md`, `src/run_parity.py` | This tolerance is task-specific parity, not bitwise equality. |
| C2 | Verified | On the CARLA corpus, 8 of 10 v0.9.7 output heads retained less than 1% of real-footage temporal activity. | `report/teardown_results.md`, `report/teardown_collected.npz` | One model version and one simulator corpus; pose and meta did not meet the collapse threshold. |
| C3 | Verified | The v0.9.7 recurrent feature spread on CARLA was approximately 0.00001 times the real spread. | `report/teardown_results.md` | This describes the recorded corpus and the defined covariance-trace metric. |
| C4 | Verified | Zero of 219 CARLA analysis frames exceeded the real p95 for the monitored plan, lead, or desired-curvature uncertainty channels. | `report/teardown_results.md` | This does not prove that every model output or downstream safety signal is silent. |
| C5 | Verified | On the Subaru real-to-CARLA overlay sweep, output activity crossed from 0.9 to 0.1 of real within an alpha width of 0.015. | `report/e4_results.md`, `report/e4_collected.npz` | The sweep is a double-exposure overlay probe, not a content-preserving sim-to-real morph. |
| C6 | Verified | The RAM sweep reached the same collapsed endpoint but followed a gradient with width 0.274. | `report/e4_ram_results.md` | Transition shape and early-warning headroom are source-dependent. The large cache is not shipped. |
| C7 | Verified | Encoder-stage temporal activity did not collapse; submodule probes place the cliff downstream of the encoder. | `report/e5_results.md`, `report/e5_submodule_results.md` | Localization is based on selected probes and does not establish a complete causal mechanism. The large cache is not shipped. |
| C8 | Verified | Across four v0.9.7 real corpora, the rolling recurrent-spread detector had 2.41% mean LOCO FPR, segment-bootstrap 95% CI [0, 5.17], and 6.90% worst-fold FPR. | `report/corpus_scaling_results.md` | This is not near-zero, fleet-scale, or deployment-grade evidence. |
| C9 | Verified | On the Subaru overlay sweep, the detector crossed 50% fired frames at alpha 0.550, before the output cliff near alpha 0.784. | `report/e6_results.md`, `report/e4_results.md` | The same early-warning claim failed on RAM; no universal lead-time claim is supported. |
| C10 | Verified negative | The detector did not transfer to v0.9.6; its reported LOCO FPR was 33%. | `report/e6_v096_results.md` | Adjacent model versions require separate calibration or a different monitor. |
| C11 | Verified negative | v0.9.6 remained separable in recurrent feature space but did not reproduce the v0.9.7 silent-freeze phenotype; 1 of 10 heads collapsed and other outputs amplified. | `report/teardown_v096_results.md`, `report/e4_v096_results.md` | Do not describe silent output collapse as version-invariant. |
| C12 | Verified bounded result | Across 15 ImageNet-C corruptions at five severities, the E6 detector was near chance on many corruptions and no corruption reproduced the CARLA output-collapse pattern. | `report/e7_results.md`, `report/e7_overlay_results.md` | The E7 cache is not shipped; E6 is a collapse detector, not a general OOD detector. |
| C13 | Verified bounded result | Two real night/glare corpora did not reproduce the CARLA output collapse. | `report/real_weather_results.md`, `report/real_weather_collected.npz` | Rain, fog, geography, sensor degradation, and broader real OOD remain untested. |
| C14 | Open | A clean daytime segment intermittently entered a near-zero recurrent attractor and fired E6 on 60.34% of analyzed frames; the trigger remains unexplained. | `report/corpus_scaling_results.md`, `report/daytime_attractor_analysis.md`, `report/attractor_diagnostic_results.md` | This is an unresolved observation, not evidence of a real driving failure or validated alert. |
| C15 | Verified implementation | The C++ monitor numerically agrees with the Python path on the documented fixture and has a measured x86 desktop latency. | `report/deployment_results.md`, `deploy/cpp/` | No on-road or live openpilot deployment was performed. |
| C16 | Verified diagnostic | With thresholds chosen using labeled collapse data to target 95% TPR, E6 produced 0% held-out real FPR at 94.8% realized collapse detection; KNN-50, Mahalanobis, and RMD produced 60.82%, 95.14%, and 99.69% FPR. | `report/loco_threshold_free_results.md` | This collapse-aware operating point is a diagnostic comparison, not a deployable threshold learned from clean calibration data. |

## Unsupported wording

These phrases overstate what the evidence in the table above supports. Do not use them
in the manuscript, the README, figure captions, or any public description of this work:

- `near-zero false alarm`
- `production-ready monitor`
- `universal OOD detector`
- `version-invariant collapse`
- `on-road deployment`
- `fleet-scale validation`
- `the model exposes no signal whatsoever`
- `0.23 alpha units of early warning` without naming the Subaru overlay protocol

## Reproducibility boundary

The fresh-clone path re-derives E1-E4 from committed caches. E5 submodule and E7 large caches are
not tracked and require regeneration. Model files and source video segments are fetched or supplied
separately. `report/MANIFEST.json` records the audited environment and hashes, but its recorded Git
state was dirty and must be regenerated for a release tag.

## Publication status

Manuscript draft only. No submission receipt, preprint identifier, acceptance, or independent
replication was present at the 2026-07-17 audit.
