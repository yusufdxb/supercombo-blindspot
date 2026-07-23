# Claim-to-Evidence Matrix

This matrix defines the review boundary for the manuscript. `Recomputed` means the public cache path
regenerates the result from tracked arrays. `Inspect only` means the committed table and figure can be
checked, but the underlying large activation cache is not in the repository. None of these claims has
been independently replicated or peer reviewed.

| ID | Manuscript claim | Evidence | Reproduction level | Required boundary |
|---|---|---|---|---|
| C1 | v0.9.7 acceleration parity: 1,159/1,159 frames within +/-0.5 m/s2, median absolute delta 0.0409 m/s2 | `report/parity_results.md`, `src/run_parity.py` | External inputs required | Task-specific tolerance parity, not bitwise equality |
| C2 | Eight of ten v0.9.7 heads retain less than 1% of real temporal activity on CARLA | `report/teardown_results.md`, `report/teardown_collected.npz` | Recomputed | One model version and CARLA corpus |
| C3 | Recurrent spread contracts to approximately 0.00001 times the real spread | `report/teardown_results.md` | Recomputed | Metric-specific observation, not a causal mechanism |
| C4 | Zero of 219 CARLA frames exceed real p95 on the three monitored uncertainty channels | `report/teardown_results.md` | Recomputed | Does not cover every model or system output |
| C5 | Subaru overlay transition width is 0.015 | `report/e4_results.md`, `report/e4_collected.npz` | Recomputed | Double-exposure overlay, not a content-preserving morph |
| C6 | RAM overlay transition width is 0.274 and warning headroom is negative | `report/e4_ram_results.md` | Inspect only | Large activation cache is not distributed |
| C7 | Selected probes place contraction downstream of the encoder | `report/e5_results.md`, `report/e5_submodule_results.md` | Inspect only | Partial localization; mu-versus-sigma ambiguity remains |
| C8 | Collapse-unaware E6 calibration has 2.41% mean LOCO FPR, 95% CI [0%, 5.17%], worst fold 6.90% across four real corpora | `report/corpus_scaling_results.md` | Recomputed | Not fleet-scale or deployment-grade |
| C9 | On the Subaru overlay, E6 fires at alpha 0.550 before the output cliff near 0.784 | `report/e6_results.md`, `report/e4_results.md` | Recomputed | No warning lead on RAM |
| C10 | At full CARLA shift E6 has AUROC 0.996 [0.992, 1.000] | `report/metrics_results.md`, `report/metrics_collected.npz` | Recomputed | Threshold-free separation is not operational calibration |
| C11 | Collapse-aware 95% TPR diagnostic: E6 0% held-out FPR, baselines 60.82% to 99.69% | `report/loco_threshold_free_results.md` | Recomputed | Uses labeled collapse data; not a clean-only threshold |
| C12 | ImageNet-C reproduces zero output-collapsed cells out of 75 | `report/e7_overlay_results.md` | Inspect only | Corruption cache is not distributed |
| C13 | v0.9.6 does not reproduce the freeze and E6 has 33.28% LOCO FPR | `report/teardown_v096_results.md`, `report/e6_v096_results.md` | Recomputed | Adjacent versions fail differently |
| C14 | Real night/glare segments reproduce zero collapsed heads and zero E6 fire rate | `report/real_weather_results.md`, `report/real_weather_collected.npz` | Recomputed | Other real shifts remain untested |
| C15 | A daytime segment is flagged on 60.34% of analyzed frames | `report/corpus_scaling_results.md` | Recomputed | Trigger is unexplained; not a validated safety alert |
| C16 | C++ monitor agrees with the Python fixture and is measured at about 0.4 microseconds on x86 | `report/deployment_results.md`, `deploy/cpp/` | Test and benchmark | No target-platform or vehicle timing |

The machine-readable checks are implemented in `scripts/verify_paper.py`; the artifact hashes are in
`paper/artifact_manifest.json`.
