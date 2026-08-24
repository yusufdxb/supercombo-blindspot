# Silent Failure Under Distribution Shift: A Teardown of a Production Driving Model and a Recurrent-Feature Monitor

> **Superseded outline.** The current manuscript source is `paper/manuscript.md`. This file is retained
> only for provenance and may contain stale draft wording. Do not cite it as the paper or build from it.

> Working title. Alternatives: "Does a Production L2 Driving Model Know When It Is Blind?" / "Collapse Without Warning: Internal-Feature Monitoring for Shipped Driving Models"

**Draft skeleton, v0.** Numbers below are pulled from the committed `report/*.md` result files. Sections marked `[AUTHOR TODO]` remain open. Claims and their release boundaries are registered in `docs/evidence_register.md` and were last audited on 2026-07-17.

Target venue is undecided. Reverify the official call, scope, deadline, and archival status before selecting a venue.

---

## Abstract

`[DRAFT]` Production L2 driver-assistance systems are validated, in part, in simulation. We ask whether a shipped end-to-end driving model can signal when rendered input drives it into a silent failure mode. We instrument openpilot v0.9.7's `supercombo` network and build a parity-controlled reimplementation of its inference path, matching comma's reference acceleration output within +/-0.5 m/s2 on 100% of 1,159 real-footage frames (median absolute delta 0.04 m/s2). On a CARLA-rendered corpus, 8 of 10 output heads fall below 1% of their real-driving temporal activity and the 512-D recurrent feature spread contracts to 1e-5 of its real value, while monitored predictive-uncertainty channels remain below their real-driving 95th percentile on all 219 analysis frames. A real-to-CARLA overlay sweep exhibits a 0.015-wide output cliff on the Subaru source, and selected probes place the collapse downstream of the vision encoder. A zero-retraining rolling-spread monitor on the recurrent state exposes the Subaru collapse before the output cliff, but the broader evidence sharply limits that result: across four v0.9.7 real corpora its mean leave-one-corpus-out false-positive rate is 2.41% (95% CI [0, 5.17%], worst fold 6.90%); it provides no early-warning headroom on the RAM overlay; and it fails to transfer to v0.9.6 (33% LOCO FPR). The adjacent v0.9.6 model also reacts abnormally to CARLA but does not reproduce v0.9.7's silent-freeze phenotype. These results establish a version- and source-specific silent-collapse failure, not a deployment-ready or universal OOD detector.

A corruption sweep (15 ImageNet-C corruptions x 5 severities on real frames) bounds the claim two ways. The silent collapse is **sim-specific**: no corruption reproduces it (at most 1 of 10 output heads collapses on any corruption-severity cell, versus 7 of 10 under CARLA). And E6 is **collapse-specific**: it stays near chance on the photometric shifts the model tolerates and rises only on a few severe corruptions (frost AUROC 1.00, impulse-noise 0.91, gaussian-noise 0.86), but a cell-for-cell E1-vs-E6 overlay shows those firings track a recurrent-feature-spread shift, not output collapse, which never occurs here. The contribution is a targeted monitor for the sim-induced silent-collapse mode, not a universal OOD detector.

---

## 1. Introduction

`[DRAFT outline]`

- **Hook.** Every L2/autonomous program validates in simulation. The validity of that practice rests on an unstated assumption: that the model under test behaves the same way on simulated input as on real input, or at least fails loudly when it does not.
- **The question.** When a shipped driving model is shown input it was never trained on, does it fail loudly or silently? "Silently" is the dangerous answer, because the downstream stack (planner, AEB, safety monitors) trusts the model's outputs and its uncertainty signal.
- **What we do.** We take a single, real, shipped model (openpilot v0.9.7 `supercombo`) and run a controlled distribution-shift teardown: parity-verify the inference, instrument every output head and the recurrent state, characterize the failure as input drifts from real toward simulated, localize it inside the network, and test whether an internal-feature monitor recovers the signal the outputs hide.
- **Why it is credible.** The result is a negative finding about a production model, so the harness must be trustworthy. We establish parity first (Section 4.1) so the collapse is the model, not our reimplementation.
- **Contributions (4):**
  1. A **parity-controlled** reimplementation of openpilot v0.9.7 `supercombo` inference, verified to 100% of 1159 frames within +/-0.5 m/s^2 of comma's reference output (median abs delta 0.04 m/s^2), including correct recurrent state threading and unnormalized YUV input handling. Released.
  2. An empirical demonstration of **silent failure** under visual distribution shift: output collapse (E1), feature-space freeze (E2), and a non-responsive uncertainty channel (E3) occurring simultaneously, with 0 of 219 CARLA OOD frames exceeding the model's real-driving uncertainty p95 (0% across plan, lead, and desired_curv).
  3. A characterization of the collapse as a **hard cliff** (E4, transition width 0.015) **localized downstream of the vision encoder** (E5: encoder stages stay at or above real activity; the cliff enters at the recurrent summarizer VAE-mu bottleneck and the action-block feedback path).
  4. A **0-retraining recurrent-feature monitor** (E6) whose benefit and failure boundary are both measured: it fires before the Subaru overlay cliff, but reaches 2.41% mean LOCO FPR across four v0.9.7 real corpora, offers no early-warning headroom on the RAM overlay, and does not transfer to v0.9.6 (33% LOCO FPR). ImageNet-C and real night/glare controls do not reproduce the CARLA collapse. E6 is a scoped collapse probe, not a universal OOD detector.

---

## 2. Related Work

`[DRAFT, source: docs/related_work.md and docs/paper_plan.md §2. Pull the table prose into paragraphs.]`

Organize into five short paragraphs:

1. **OOD detection in driving / AV perception.** Keser et al. 2025 (arXiv:2501.08083, closest neighbor: feature-space density monitoring of a frozen perception model as a safety monitor); Henriksson et al. (SEAA 2019, doi:10.1109/SEAA.2019.00026, citation key `henriksson2019performance`); Hodge, Paterson, and Habli's OOD-for-safety-assurance review (arXiv:2510.21254); Guo and Su's trajectory-prediction OOD work (arXiv:2509.13577); Yuhas and Easwaran's OOD-for-AEB co-design (arXiv:2307.13419); and Saemann and Gross's online domain-exit monitor (arXiv:2310.14675).
2. **Feature-space and internal monitors.** Cheng et al. runtime neuron-activation monitoring (arXiv:1809.06573, intellectual ancestor of E6); Stocco et al. SelfOracle (ICSE 2020) and the uncertainty-quantification follow-up (arXiv:2404.18573); Parallel Activations Drift Detector (arXiv:2404.07776); Topological Uncertainty (arXiv:2105.04404). Position E6: same lineage, but watches the *second-order spread* of recurrent features rather than reconstruction error or absolute position, which is why it survives leave-one-corpus-out where location-based scores do not.
3. **openpilot / supercombo prior work.** Chen et al. Openpilot-Deepdive (arXiv:2206.08176, anchor citation for the model description); Geretti et al. falsification (GPCE/SPLASH 2022); adversarial study (arXiv:2505.11532); commaai issue #20704 / discussion #22212 as primary evidence that phantom braking under distribution shift is a *known, user-reported* failure of the shipped model.
4. **Simulation testing of driving DNNs (DeepRoad line).** DeepXplore (SOSP 2017), DeepTest (ICSE 2018), DeepRoad (ASE 2018, arXiv:1802.02295), MarMot (arXiv:2310.07414). Our angle: these *generate* tests assuming the sim is in-distribution; we show the sim itself can be OOD to the model, which undercuts coverage claims from sim-based testing.
5. **OOD benchmarks and corruption robustness.** OpenOOD (NeurIPS 2022, arXiv:2210.07242, taxonomy/baselines); Hendrycks & Dietterich ImageNet-C (ICLR 2019) and Michaelis et al. Cityscapes-C (arXiv:1907.07484), basis for our E7 corruption axis; Mahalanobis++ (arXiv:2505.18032) and the geometry view (arXiv:2510.15202) to establish Mahalanobis is a respected, current baseline, not a strawman.

The seven citations flagged by the publication-readiness audit were reverified against primary records on 2026-07-17 and corrected in `docs/related_work.md`. The remaining bibliography and all comparative positioning still require a source-level sweep when this outline is converted into the venue template.

---

## 3. Threat Model

`[DRAFT, source: docs/threat_model.md, near-complete.]`

- **The threat.** A shipped driving model deployed in a visually shifted context (rendered sim, weather, glare, novel geography, sensor degradation). Under sufficient shift, three things happen at once and silently: output collapse to a plausible constant, no rise in the uncertainty channel, and a frozen recurrent state.
- **Existing defenses and why each misses this mode** (one paragraph each, all evidence-backed):
  - Predictive-uncertainty heads: E3, ratios 1.20x to 1.84x, 0 of 219 CARLA frames above real p95 for any monitored head (0%), a real-calibrated threshold never fires.
  - Plan-feasibility / accel / lateral-accel limits: collapsed outputs (plan 0.0057x, accel_t0 0.0040x) look like a benign stationary scene, so plausibility passes.
  - Output-disagreement / temporal-jitter monitors: frozen outputs have *lower* variance, so a jitter monitor reads the freeze as increased stability.
  - Same-architecture ensembles: E5 localizes the collapse downstream of the encoder, so an ensemble of the same architecture shares the collapse path.
  - Input image-quality checks: CARLA-clean is sharper and less noisy than real footage, so quality monitors rate it as good.
- **Our claim.** The model's own internal features carry an OOD signal the output heads do not surface. E6 is a complementary layer, not a replacement: cheap (one O(d) statistic on a forward pass already happening), shipped-model compatible (no retraining, no architecture change), and calibrated against real-driving FPR rather than against simulated negatives.

---

## 4. Method

### 4.1 Parity-controlled reimplementation `[verified]`

- Reconstruct openpilot v0.9.7 `supercombo` inference from the released ONNX and the reference files in `references/openpilot-v0.9.7/` (modeld, parse_model_outputs, loadyuv.cl, constants).
- **Two non-obvious correctness points** (worth a paragraph each, these are the reviewer-convincing details):
  - **Recurrent state threading.** `features_buffer` and `prev_desired_curv` must shift-and-append after each inference; zero-init only on frame 1. Per-frame zero reset produces a multi-second init transient that masquerades as a phantom brake. (See `src/state.py`.)
  - **Unnormalized YUV input.** `loadyuv.cl` does `convert_float8()` with no scaling; the model consumes uint8 Y/U/V in 0..255, not divided by 255. (See `src/preprocessor.py`.)
- **Parity result:** 100% of 1159 frames within +/-0.5 m/s^2 of comma's reference output, median absolute delta 0.04 m/s^2. This is the load-bearing claim for the negative result; report the parity figure prominently.

### 4.2 Data `[verified]`

- **Real (ID):** two comma corpora, `subaru` and `ram`, 320 raw frames collected per segment (`src/teardown.py` constant `N=320`). The inference loop processes consecutive frame pairs, so the first frame is consumed as the initial recurrent-state seed and is never emitted as output; this leaves 319 stored frames per corpus in `report/teardown_collected.npz`. For the E1/E2/E3 collapse and confidence analyses (`src/teardown.py` main), the first 100 stored frames are then discarded as recurrent-state warmup (`WARMUP=100`, function `_post`), leaving **219 analysis frames per corpus** (438 real total). For the E6 monitor and the threshold-free metrics (`src/e6_detector.py`, `src/baselines.py`), the full 319 stored frames per corpus are used directly (the rolling-spread window handles the initial NaN frames internally), giving **ID n=638** (319 subaru + 319 ram).
- **OOD:** 320 raw CARLA-rendered clean-road frames collected (`N=320`), pair-processed to 319 stored frames. For E1/E2/E3: 219 CARLA analysis frames (100 warmup discarded). For E6/metrics: **319 CARLA OOD frames** (`n=319`).
- **Interpolation (E4):** pixel alpha-blend of the Subaru real sequence and the CARLA sequence, alpha=0 real, alpha=1 sim, 29 alpha points.

### 4.3 Metrics `[verified]`

- **Activity:** sum of per-element temporal std over a window; "collapse" = CARLA/real activity ratio.
- **Feature spread:** trace of the recurrent-state covariance over a rolling window.
- **Threshold-free OOD metrics:** AUROC, AUPR, FPR@95TPR with stratified bootstrap CIs (n=1000, seed=42), ID = subaru+ram (n=638), OOD = E4 alpha=1.0 CARLA (n=319).
- **Calibration protocol:** the original detector/baseline comparison uses LOCO across {subaru, ram}; the current E6 generalization estimate uses four v0.9.7 real corpora {subaru, ram, ev6_night, bronco_night} and segment-level bootstrap intervals.

### 4.4 The E6 monitor `[verified]`

- Monitored quantity: rolling temporal spread of the 512-D recurrent feature emitted by `supercombo` (`src/e6_detector.py`).
- Threshold: 1st-percentile of the training real-driving rolling-spread distribution, calibrated LOCO. The nominal 1% training target does not imply 1% held-out FPR.
- One O(d) statistic per inference; no retraining, no extra heads.

### 4.5 Baselines `[verified]`

- Three applicable post-hoc feature-space scores on the *same* 512-D feature: Mahalanobis, Relative Mahalanobis (RMD), KNN-50; plus a PCA-Mahalanobis ablation.
- **Not applicable, with structural reasons** (state these so reviewers do not read them as omissions): MSP (no softmax head), Energy (no logits), ViM (no classifier weight matrix on the recurrent feature). supercombo's heads are Gaussian-mixture regressions and existence probabilities.
- RMD's "background" model is a 2-component diagonal GMM rather than the single pooled Gaussian of Ren et al. (arXiv:2106.09022): supercombo's recurrent feature carries no class labels, so a single pooled background Gaussian would be identical to the (single-class) foreground Gaussian and force RMD identically to zero. The relative score is therefore the class-Mahalanobis distance minus the best-component background Mahalanobis. Source: `src/baselines.py` lines 148-217.

---

## 5. Experiments and Results

### E1: Output collapse map `[verified: report/teardown_results.md]`

8 of 10 heads collapse to under 1% of real-driving temporal activity on CARLA-clean input. Two heads survive (pose 0.18x, meta 0.72x). Figure: `report/figures/e1_head_collapse.png`.

| head | CARLA/real activity | state |
|---|---|---|
| desired_curv | 0.0018 | COLLAPSED |
| accel_t0 | 0.0040 | COLLAPSED |
| lead | 0.0042 | COLLAPSED |
| desire_state | 0.0049 | COLLAPSED |
| lane_lines | 0.0054 | COLLAPSED |
| plan | 0.0057 | COLLAPSED |
| lead_prob | 0.0058 | COLLAPSED |
| road_edges | 0.0076 | COLLAPSED |
| pose | 0.1788 | alive |
| meta | 0.7181 | alive |

### E2: Internal feature-space OOD `[verified]`

CARLA recurrent-feature spread is 1e-5 of the real spread (trace of `hidden_state` covariance). Real-vs-CARLA separability 87.9%, d' = 2.19 along the centroid-difference direction. The recurrent state freezes to a near-constant point. Figure: `report/figures/e2_feature_ood.png`.

### E3: Silent failure (the centerpiece) `[verified]`

Outputs lose ~99.5% of their activity, but predictive-uncertainty heads rise only 1.20x to 1.84x, and 0 of 219 CARLA frames (0%) exceed the real-driving p95 of any monitored head. Figure: `report/figures/e3_confidence.png`.

| head | output retained | unc. ratio | OOD frames above real p95 |
|---|---|---|---|
| plan | 0.6% | 1.35x | 0% (0 / 219) |
| lead | 0.4% | 1.20x | 0% (0 / 219) |
| desired_curv | 0.2% | 1.84x | 0% (0 / 219) |

### E4: Cliff, not gradient `[verified]`

Alpha-blending real toward CARLA: output activity first balloons to 6.32x the real baseline (ghosted-input thrash, peak at alpha=0.425), then collapses in a hard cliff, falling from 0.9x to 0.1x of real over alpha 0.784 to 0.799 (transition width 0.015). Feature spread crashes from 0.25 to 0.00 by alpha=0.78. Predictive uncertainty never spikes through the transition. Figure: `report/figures/e4_interpolation.png`.

### E5: Localization, downstream of the encoder `[verified]`

Every vision-encoder stage stays at or above the real-driving activity baseline across the full alpha sweep (stem 1.43x, stage3 2.06x, head 2.14x at alpha=1; minimum 0.96). The collapse is therefore not the encoder failing. Submodule probing pins the cliff entry to the recurrent summarizer VAE-mu bottleneck (`summarizer_div`, cliff alpha 0.900) with earlier amplification in the action-block feedback path (`action_block_body`, cliff alpha 0.500, driven by the `prev_desired_curv` loop); the transformer + reduce-sum stage is a passive relay (`vision_post` and `hydra_trunk` show no cliff). Figures: `report/figures/e5_layer_localization.png`, `report/figures/e5_submodule_localization.png`.

`[AUTHOR TODO]` Add a bridge sentence: the per-stage `nan` cliff-alpha values in report/e5_results.md mean "no cliff in the encoder," reconciled with the submodule table.

### E6: A monitor could have caught it `[verified]`

The rolling-spread monitor on the 512-D recurrent feature fires on more than 50% of frames at alpha=0.550 on the Subaru overlay, before the E4 output-collapse cliff near alpha=0.784. This source-specific gap is about 0.23 alpha units. It does not generalize to the RAM overlay, where the detector fires at alpha=0.850 after the gradient has begun. Expanding the held-out real-driving evaluation to four v0.9.7 corpora produces 2.41% mean LOCO FPR (segment-bootstrap 95% CI [0, 5.17%]) and 6.90% worst-fold FPR. Figure: `report/figures/e6_detector.png`, `report/figures/auroc_vs_alpha.png`; generalization: `report/corpus_scaling_results.md`.

**Threshold-free comparison at alpha=1.0** (AUROC mean [95% CI], `report/metrics_results.md`):

| detector | AUROC | AUPR | FPR@95TPR | LOCO mean FPR |
|---|---|---|---|---|
| E6 (rolling-spread) | 0.996 [0.992, 1.000] | 0.995 [0.990, 1.000] | 0.000 | **2.41% across four corpora** |
| KNN-50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 | 100% |
| Relative Mahalanobis | 0.934 [0.914, 0.952] | 0.732 [0.684, 0.784] | 0.067 | 100% |
| Mahalanobis | 0.159 [0.130, 0.190] | 0.230 [0.217, 0.245] | 0.854 | 100% |
| PCA-Mahalanobis | 0.152 [0.124, 0.179] | 0.214 [0.209, 0.219] | 0.854 | 11.91% |

**The headline finding.** Location-sensitive feature-space scores fail on the original two-corpus `supercombo` comparison in two distinct ways. Vanilla and PCA-Mahalanobis score below chance at alpha=1.0 because the recurrent state contracts toward the ID mean, while all three applicable absolute-position baselines hit 100% two-corpus LOCO FPR because the Subaru and RAM feature distributions are strongly separated. E6's second-order trace separates CARLA from the original real set, but its four-corpus 2.41% mean FPR, 6.90% worst fold, source-dependent lead time, and v0.9.6 transfer failure prevent a general transfer claim. The defensible claim is narrower: second-order recurrent spread exposes this v0.9.7 collapse more robustly than the tested location scores, while still requiring source/version validation.

### E7: Corruption sweep, E6 is collapse-specific (a bounded result) `[verified: report/e7_results.md, 2026-05-29]`

We applied the 15 ImageNet-C corruptions (Hendrycks & Dietterich, ICLR 2019) at 5 severities to the real Subaru frames (raw RGB, pre-YUV), plus a clean baseline (76 conditions total: 15 corruptions x 5 severities + 1 clean), re-ran `supercombo` with correct recurrent state handling on each corrupted sequence (320 raw frames collected per condition, 319 stored after pair-processing, 219 post-warmup analysis frames per condition after discarding the first 100 as recurrent-state warmup, `src/e7_corruption.py` `WARMUP=100`), and evaluated E6 and the baselines against clean real driving. Figures: `report/figures/e7_auroc_heatmap.png`, `report/figures/e7_severity_sweep.png`.

**The honest finding: E6 does not generalize as a generic corruption detector; it is specific to the recurrent-state collapse.** On most corruptions E6's AUROC sits near chance (mean per-corruption AUROC: jpeg 0.52, fog 0.55, snow 0.55, brightness 0.60, zoom_blur 0.58, defocus/motion/glass blur 0.68-0.71, pixelate 0.69, elastic 0.67), and at the CARLA-calibrated 1% threshold E6's fire rate is ~0.000 across nearly all corruption/severity cells. E6 detects strongly only on a few severe corruption-and-severity combinations (heavy frost, extreme noise) that pull the recurrent-feature spread far enough to separate from real driving (these do **not** reproduce the output collapse; see the overlay below):

| condition | E6 AUROC [95% CI] | E6 fire rate @1% thr |
|---|---|---|
| frost, sev 5 | 1.000 [0.999, 1.000] | 1.000 |
| frost, sev 3 | 0.958 [0.944, 0.971] | 0.063 |
| impulse_noise, sev 5 | 0.906 [0.877, 0.932] | 0.547 |
| gaussian_noise, sev 4 | 0.861 [0.833, 0.887] | 0.200 |
| contrast, sev 3 | 0.823 [0.787, 0.855] | 0.089 |
| shot_noise, sev 4 | 0.803 [0.770, 0.835] | 0.111 |
| (typical photometric/blur, all sev) | 0.50 - 0.72 | 0.000 |

**Interpretation: the collapse is sim-specific, and E6's corruption firings are decoupled from it.** To resolve whether E6's quiet response on most corruptions is correct (no collapse) or a miss (undetected collapse), we ran the canonical E1 output-collapse metric per corruption-severity cell (`src/e7_overlay.py`, results in `report/e7_overlay_results.md`), behind a validation gate that first reproduces the published CARLA collapse (7 of 10 output heads). The result is unambiguous: **no ImageNet-C corruption reproduces the output collapse.** At most 1 of 10 output heads collapses on any of the 75 cells, versus 7 of 10 under CARLA. Two consequences follow. First, there are **zero false negatives**: there is no output collapse for E6 to miss, so its quiet response on fog, brightness, blur, and the rest is correct, not a silent failure. Second, E6's four high-AUROC cells (frost at severities 3 and 5, gaussian-noise severity 4, impulse-noise severity 5) fire **without** any output collapse, so on this corpus E6 is responding to a recurrent-feature-spread shift, not to the silent-collapse mode. The honest reading is therefore narrower than "E6 generalizes beyond CARLA": the silent collapse is a property of full-sim rendering, and E6 is a collapse-specific monitor that is correctly quiet on real-frame corruptions and fires on only a few severe ones through a feature shift that is not itself an output failure.

**On the baselines.** Mahalanobis and Relative Mahalanobis "fire" on a large fraction of corruptions (fog, frost, gaussian/impulse/shot noise, snow, zoom, contrast all at ~1.0 fire rate), but recall from E6/baselines that both carry 100% LOCO held-out FPR: they flag held-out real driving too, so their high corruption fire rates are not calibrated detection. KNN-50 fires only on the heaviest noise/frost. None of the baselines is calibrated to the 1% operating point E6 holds.

**What E7 changes about the paper's claim.** The earlier framing ("an internal monitor catches the OOD condition") must be narrowed: an internal monitor catches the *silent-collapse* condition, which the E1 overlay shows is **sim-specific** (no ImageNet-C corruption reproduces it). E6 is a targeted probe for that failure mode with 2.41% mean four-corpus LOCO FPR and 6.90% worst-fold FPR; on real-frame corruptions it is mostly quiet, and its few firings track a feature-spread shift rather than an output collapse. This is a narrower and more defensible contribution than "generalizes beyond CARLA."

---

## 6. Limitations `[DRAFT: source: docs/threat_model.md §4, strong]`

- **Two adjacent model versions, one silent-freeze phenotype:** v0.9.6 is also abnormal on CARLA, but only 1/10 heads collapse and other outputs amplify. E6 reaches 33% LOCO FPR on v0.9.6, so neither phenotype nor monitor transfer is version-invariant.
- **Four real corpora remain small:** the current v0.9.7 LOCO estimate is 2.41% mean FPR with CI [0, 5.17] and 6.90% worst fold. It is not a fleet-scale or deployment operating point.
- **OOD axis breadth and E6 selectivity:** CARLA-clean is an extreme shift. ImageNet-C and two real night/glare corpora do not reproduce the collapse. Rain, fog, novel geography, sensor degradation, and another simulator remain untested. A clean daytime segment exhibits an unexplained near-zero recurrent attractor, so real false-alert behavior is not resolved.
- **E5 localization is partial:** collapse pinned to summarizer VAE-mu + action-block feedback by ruling out the encoder and probing 8 submodules; a VAE-mu/sigma ambiguity remains (note it, source: report/e5_submodule_results.md).
- **No real-robot / on-road deployment of E6:** the monitor is demonstrated offline on logged + rendered + corrupted frames, not in the running stack.

---

## 7. Conclusion

`[DRAFT]` In the tested v0.9.7 model and CARLA corpus, a shipped L2 driving model collapses to low-activity outputs without raising the monitored uncertainty channels. Recurrent spread exposes this failure on one overlay source, but its false-positive rate, source-dependent lead time, cross-version failure, and unexplained real attractor make it an offline diagnostic candidate rather than a deployable monitor. The result motivates internal-state validation alongside output checks while showing that each monitor must be validated across versions, sources, and realistic shifts.

---

## 8. Reproducibility `[verified: README documents a GPU-free, CARLA-free fresh-clone reproduce path from committed caches]`

- E1-E4 result caches are committed and rerun from cache. Large E5 submodule and E7 caches are regenerated rather than shipped.
- Bootstrap params: n=1000, seed=42. Requirements pinned (Python 3.10, onnxruntime-gpu 1.23.2, numpy 2.2.6, carla 0.9.15).
- `[AUTHOR TODO, audit P1]` Package smaller sufficient E5/E7 summaries or a verified regeneration path; the large caches are not tracked. Regenerate `report/MANIFEST.json` from a clean release state.

---

## Figures (all exist in report/figures/ unless marked)

- `hero.png` (four findings at a glance), `e1_head_collapse.png`, `e2_feature_ood.png`, `e3_confidence.png`, `e4_interpolation.png`, `e5_layer_localization.png`, `e5_submodule_localization.png`, `e6_detector.png`, `roc_curves.png`, `pr_curves.png`, `auroc_vs_alpha.png`.
- `e7_auroc_heatmap.png`, `e7_severity_sweep.png` `[verified, generated 2026-05-29]`.

## Pre-submission checklist (from the 2026-05-29 publication-readiness audit; updated post-E7)

- [x] **P0:** E7 corruption sweep run (re-collected + analyzed 2026-05-29), subsection + 2 figures written. Result is a bounded negative (E6 collapse-specific), reflected in abstract/contrib-4/limitations.
- [x] **P0 (from E7):** E1 output-collapse-per-corruption overlay run (`src/e7_overlay.py`, `report/e7_overlay_results.md`, figure `e7_overlay.png`, 2026-05-30), with a validation gate that reproduces the CARLA collapse (7/10 heads). Resolved: collapse is sim-specific (<=1/10 heads on every corruption vs 7/10 CARLA), 0 false negatives, E6's four corruption firings decoupled from output collapse. Headline claim no longer gated; abstract/contrib-4/E7/limitations updated.
- [ ] **P1:** the E5 (3.9 GB) and E7 (110 MB) caches are `.gitignore`d, so they are NOT in the public repo at all (not merely un-LFS'd). The reproduce-from-cache path is broken for E5/E7. Decide: Git LFS the E7 cache (E5 at 3.9 GB exceeds even LFS norms, consider a regeneration script + smaller committed summary instead).
- [x] **P1:** reverify and correct the six flagged arXiv entries, add the near-neighbor online-domain monitor, and pin Henriksson to SEAA 2019 with DOI and citation key (2026-07-17).
- [x] **P1:** consolidate current claims, cross-version failures, and public wording in `docs/evidence_register.md` (2026-07-17).
- [ ] **P2:** RMD-background one-sentence justification in §4.5.
- [ ] **P2:** state N for every percentage; pin matplotlib.

## Open authorship / venue decisions `[AUTHOR]`

- Author list / affiliation line (Wayne State; advisor co-author?).
- Final venue and format after rechecking official 2026/2027 calls.
- Whether to include the completed real night/glare negative result and unresolved daytime attractor in the main paper or supplement.
