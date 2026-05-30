# Silent Failure Under Distribution Shift: A Teardown of a Production Driving Model and a Recurrent-Feature Monitor

> Working title. Alternatives: "Does a Production L2 Driving Model Know When It Is Blind?" / "Collapse Without Warning: Internal-Feature Monitoring for Shipped Driving Models"

**Draft skeleton, v0.** Numbers below are pulled verbatim from the committed `report/*.md` result files (verified). Sections marked `[AUTHOR TODO]` are gaps from the publication-readiness audit (2026-05-29). Sections marked `[PENDING E7]` are blocked on running `python -m src.e7_corruption` against the already-collected `report/e7_collected.npz` cache (no GPU, no CARLA; the single P0 blocker to a submittable draft).

Target venue: SafeAI @ UAI 2026 (confirmed open) or a NeurIPS 2026 workshop track. Format-agnostic for now; port to the venue template once chosen.

---

## Abstract

`[DRAFT]` Production L2 driver-assistance systems are validated, in large part, in simulation. We ask whether a shipped end-to-end driving model can tell when its input has left the distribution it was trained on, or whether it fails silently. We instrument openpilot v0.9.7's `supercombo` network, the model that drives comma hardware on public roads, and build a parity-exact reimplementation of its inference path, verified to within +/-0.5 m/s^2 of comma's own reference output on 100% of 1159 real-footage frames (median absolute delta 0.04 m/s^2). Running the verified model on CARLA-rendered driving scenes, we find that 8 of 10 output heads collapse to under 1% of their real-driving temporal activity and the 512-D recurrent feature vector freezes to 1e-5 of its real spread, while the model's own predictive-uncertainty heads rise only 1.2x to 1.8x and never exceed their real-driving 95th percentile on a single out-of-distribution (OOD) frame. The failure is therefore silent: nothing the model emits flags the collapse. An alpha-blend sweep shows the collapse is a hard cliff (transition width 0.015 in blend fraction), and a layer-by-layer probe localizes it downstream of the vision encoder, in the recurrent summarizer and action block, not in perception. Finally, we show a 0-retraining monitor on the model's own 512-D recurrent feature, the rolling temporal spread of the state, calibrated leave-one-corpus-out to a ~1% real-driving false-positive rate, detects the OOD condition (AUROC 0.996) roughly 0.23 blend-units before outputs cliff, where standard location-sensitive feature-space OOD scores (Mahalanobis, Relative Mahalanobis, KNN) fail to transfer across real corpora. The safety implication: a sim "pass" can be the model collapsed to a safe-looking default, not the model perceiving, and output-side monitors alone cannot catch it.

A corruption sweep (15 ImageNet-C corruptions x 5 severities on real frames) bounds the claim two ways. The silent collapse is **sim-specific**: no corruption reproduces it (at most 1 of 10 output heads collapses on any corruption-severity cell, versus 7 of 10 under CARLA). And E6 is **collapse-specific**: it stays near chance on the photometric shifts the model tolerates and rises only on a few severe corruptions (frost AUROC 1.00, impulse-noise 0.91, gaussian-noise 0.86), but a cell-for-cell E1-vs-E6 overlay shows those firings track a recurrent-feature-spread shift, not output collapse, which never occurs here. The contribution is a targeted monitor for the sim-induced silent-collapse mode, not a universal OOD detector.

---

## 1. Introduction

`[DRAFT outline]`

- **Hook.** Every L2/autonomous program validates in simulation. The validity of that practice rests on an unstated assumption: that the model under test behaves the same way on simulated input as on real input, or at least fails loudly when it does not.
- **The question.** When a shipped driving model is shown input it was never trained on, does it fail loudly or silently? "Silently" is the dangerous answer, because the downstream stack (planner, AEB, safety monitors) trusts the model's outputs and its uncertainty signal.
- **What we do.** We take a single, real, shipped model (openpilot v0.9.7 `supercombo`) and run a controlled distribution-shift teardown: parity-verify the inference, instrument every output head and the recurrent state, characterize the failure as input drifts from real toward simulated, localize it inside the network, and test whether an internal-feature monitor recovers the signal the outputs hide.
- **Why it is credible.** The result is a negative finding about a production model, so the harness must be trustworthy. We establish parity first (Section 4.1) so the collapse is the model, not our reimplementation.
- **Contributions (4):**
  1. A **parity-exact** reimplementation of openpilot v0.9.7 `supercombo` inference, verified to 100% of 1159 frames within +/-0.5 m/s^2 of comma's reference output (median abs delta 0.04 m/s^2), including correct recurrent state threading and unnormalized YUV input handling. Released.
  2. An empirical demonstration of **silent failure** under visual distribution shift: output collapse (E1), feature-space freeze (E2), and a non-responsive uncertainty channel (E3) occurring simultaneously, with 0% of OOD frames exceeding the model's real-driving uncertainty p95.
  3. A characterization of the collapse as a **hard cliff** (E4, transition width 0.015) **localized downstream of the vision encoder** (E5: encoder stages stay at or above real activity; the cliff enters at the recurrent summarizer VAE-mu bottleneck and the action-block feedback path).
  4. A **0-retraining recurrent-feature monitor** (E6): rolling temporal spread of the 512-D state, LOCO-calibrated to ~1% real-driving FPR, AUROC 0.996 on CARLA, firing ~0.23 blend-units before the output cliff, where standard location-sensitive OOD scores fail to transfer across corpora (100% LOCO FPR). An ImageNet-C corruption sweep bounds the claim two ways: the silent collapse is **sim-specific** (no corruption reproduces it, at most 1/10 output heads collapse vs 7/10 under CARLA), and a cell-for-cell E1-vs-E6 overlay shows E6's few firings (frost AUROC 1.00, impulse 0.91, gaussian 0.86) track a recurrent-feature shift rather than output collapse. E6 is collapse-specific, not a universal corruption detector.

---

## 2. Related Work

`[DRAFT — source: docs/related_work.md and docs/paper_plan.md §2. Pull the table prose into paragraphs.]`

Organize into five short paragraphs:

1. **OOD detection in driving / AV perception.** Keser et al. 2025 (arXiv:2501.08083, closest neighbor: feature-space density monitoring of a frozen perception model as a safety monitor); Henriksson et al. (RefSQ 2023) `[AUTHOR TODO: pin bibkey/DOI]`; OOD-as-assurance-evidence position paper (arXiv:2510.21254); trajectory-prediction OOD (arXiv:2509.13577); OOD-for-AEB co-design (arXiv:2307.13419).
2. **Feature-space and internal monitors.** Cheng et al. runtime neuron-activation monitoring (arXiv:1809.06573, intellectual ancestor of E6); Stocco et al. SelfOracle (ICSE 2020) and the uncertainty-quantification follow-up (arXiv:2404.18573); Parallel Activations Drift Detector (arXiv:2404.07776); Topological Uncertainty (arXiv:2105.04404). Position E6: same lineage, but watches the *second-order spread* of recurrent features rather than reconstruction error or absolute position, which is why it survives leave-one-corpus-out where location-based scores do not.
3. **openpilot / supercombo prior work.** Chen et al. Openpilot-Deepdive (arXiv:2206.08176, anchor citation for the model description); Geretti et al. falsification (GPCE/SPLASH 2022); adversarial study (arXiv:2505.11532); commaai issue #20704 / discussion #22212 as primary evidence that phantom braking under distribution shift is a *known, user-reported* failure of the shipped model.
4. **Simulation testing of driving DNNs (DeepRoad line).** DeepXplore (SOSP 2017), DeepTest (ICSE 2018), DeepRoad (ASE 2018, arXiv:1802.02295), MarMot (arXiv:2310.07414). Our angle: these *generate* tests assuming the sim is in-distribution; we show the sim itself can be OOD to the model, which undercuts coverage claims from sim-based testing.
5. **OOD benchmarks and corruption robustness.** OpenOOD (NeurIPS 2022, arXiv:2210.07242, taxonomy/baselines); Hendrycks & Dietterich ImageNet-C (ICLR 2019) and Michaelis et al. Cityscapes-C (arXiv:1907.07484), basis for our E7 corruption axis; Mahalanobis++ (arXiv:2505.18032) and the geometry view (arXiv:2510.15202) to establish Mahalanobis is a respected, current baseline, not a strawman.

`[AUTHOR TODO]` Resolve the six [UNVERIFIED] citations in docs/related_work.md and the Henriksson bibkey before submission (audit P1). Consider adding arXiv:2310.14675 (Online Out-of-Domain Detection for Automated Driving) as a near-neighbor.

---

## 3. Threat Model

`[DRAFT — source: docs/threat_model.md, near-complete. Clean the "Agent A" artifact on line 43 of that file.]`

- **The threat.** A shipped driving model deployed in a visually shifted context (rendered sim, weather, glare, novel geography, sensor degradation). Under sufficient shift, three things happen at once and silently: output collapse to a plausible constant, no rise in the uncertainty channel, and a frozen recurrent state.
- **Existing defenses and why each misses this mode** (one paragraph each, all evidence-backed):
  - Predictive-uncertainty heads: E3, ratios 1.20x to 1.84x, 0% of OOD frames above real p95, a real-calibrated threshold never fires.
  - Plan-feasibility / accel / lateral-accel limits: collapsed outputs (plan 0.0057x, accel_t0 0.0040x) look like a benign stationary scene, so plausibility passes.
  - Output-disagreement / temporal-jitter monitors: frozen outputs have *lower* variance, so a jitter monitor reads the freeze as increased stability.
  - Same-architecture ensembles: E5 localizes the collapse downstream of the encoder, so an ensemble of the same architecture shares the collapse path.
  - Input image-quality checks: CARLA-clean is sharper and less noisy than real footage, so quality monitors rate it as good.
- **Our claim.** The model's own internal features carry an OOD signal the output heads do not surface. E6 is a complementary layer, not a replacement: cheap (one O(d) statistic on a forward pass already happening), shipped-model compatible (no retraining, no architecture change), and calibrated against real-driving FPR rather than against simulated negatives.

---

## 4. Method

### 4.1 Parity-exact reimplementation `[verified]`

- Reconstruct openpilot v0.9.7 `supercombo` inference from the released ONNX and the reference files in `references/openpilot-v0.9.7/` (modeld, parse_model_outputs, loadyuv.cl, constants).
- **Two non-obvious correctness points** (worth a paragraph each, these are the reviewer-convincing details):
  - **Recurrent state threading.** `features_buffer` and `prev_desired_curv` must shift-and-append after each inference; zero-init only on frame 1. Per-frame zero reset produces a multi-second init transient that masquerades as a phantom brake. (See `src/state.py`.)
  - **Unnormalized YUV input.** `loadyuv.cl` does `convert_float8()` with no scaling; the model consumes uint8 Y/U/V in 0..255, not divided by 255. (See `src/preprocessor.py`.)
- **Parity result:** 100% of 1159 frames within +/-0.5 m/s^2 of comma's reference output, median absolute delta 0.04 m/s^2. This is the load-bearing claim for the negative result; report the parity figure prominently.

### 4.2 Data `[verified]`

- **Real (ID):** two comma corpora, `subaru` and `ram`, 320 frames each, 100 warmup frames discarded (220 analysis frames each).
- **OOD:** 320 CARLA-rendered clean-road frames (openpilot v0.9.7 sim pipeline).
- **Interpolation (E4):** pixel alpha-blend of the Subaru real sequence and the CARLA sequence, alpha=0 real, alpha=1 sim, 29 alpha points.
- `[AUTHOR TODO]` State N explicitly wherever a percentage appears (e.g., E3 "0%" = 0 of 220 CARLA frames).

### 4.3 Metrics `[verified]`

- **Activity:** sum of per-element temporal std over a window; "collapse" = CARLA/real activity ratio.
- **Feature spread:** trace of the recurrent-state covariance over a rolling window.
- **Threshold-free OOD metrics:** AUROC, AUPR, FPR@95TPR with stratified bootstrap CIs (n=1000, seed=42), ID = subaru+ram (n=638), OOD = E4 alpha=1.0 CARLA (n=319).
- **Calibration protocol:** leave-one-corpus-out (LOCO) across {subaru, ram}.

### 4.4 The E6 monitor `[verified]`

- Monitored quantity: rolling temporal spread of the 512-D recurrent feature emitted by `supercombo` (`src/e6_detector.py`).
- Threshold: 1st-percentile of the real-driving rolling-spread distribution (target ~1% FPR by construction), calibrated LOCO.
- One O(d) statistic per inference; no retraining, no extra heads.

### 4.5 Baselines `[verified]`

- Three applicable post-hoc feature-space scores on the *same* 512-D feature: Mahalanobis, Relative Mahalanobis (RMD), KNN-50; plus a PCA-Mahalanobis ablation.
- **Not applicable, with structural reasons** (state these so reviewers do not read them as omissions): MSP (no softmax head), Energy (no logits), ViM (no classifier weight matrix on the recurrent feature). supercombo's heads are Gaussian-mixture regressions and existence probabilities.
- `[AUTHOR TODO]` One-sentence note that RMD's "background" uses a 2-component GMM because with a single ID class the Ren et al. marginal Gaussian collapses to the class Gaussian (making RMD identically zero). Source: `src/baselines.py` lines 148-191.

---

## 5. Experiments and Results

### E1: Output collapse map `[verified — report/teardown_results.md]`

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

Outputs lose ~99.5% of their activity, but predictive-uncertainty heads rise only 1.20x to 1.84x, and 0% of CARLA frames exceed the real-driving p95 of any monitored head. Figure: `report/figures/e3_confidence.png`.

| head | output retained | unc. ratio | OOD frames above real p95 |
|---|---|---|---|
| plan | 0.6% | 1.35x | 0% (0 / 220) |
| lead | 0.4% | 1.20x | 0% (0 / 220) |
| desired_curv | 0.2% | 1.84x | 0% (0 / 220) |

### E4: Cliff, not gradient `[verified]`

Alpha-blending real toward CARLA: output activity first balloons to 6.32x the real baseline (ghosted-input thrash, peak at alpha=0.425), then collapses in a hard cliff, falling from 0.9x to 0.1x of real over alpha 0.784 to 0.799 (transition width 0.015). Feature spread crashes from 0.25 to 0.00 by alpha=0.78. Predictive uncertainty never spikes through the transition. Figure: `report/figures/e4_interpolation.png`.

### E5: Localization, downstream of the encoder `[verified]`

Every vision-encoder stage stays at or above the real-driving activity baseline across the full alpha sweep (stem 1.43x, stage3 2.06x, head 2.14x at alpha=1; minimum 0.96). The collapse is therefore not the encoder failing. Submodule probing pins the cliff entry to the recurrent summarizer VAE-mu bottleneck (`summarizer_div`, cliff alpha 0.900) with earlier amplification in the action-block feedback path (`action_block_body`, cliff alpha 0.500, driven by the `prev_desired_curv` loop); the transformer + reduce-sum stage is a passive relay (`vision_post` and `hydra_trunk` show no cliff). Figures: `report/figures/e5_layer_localization.png`, `report/figures/e5_submodule_localization.png`.

`[AUTHOR TODO]` Add a bridge sentence: the per-stage `nan` cliff-alpha values in report/e5_results.md mean "no cliff in the encoder," reconciled with the submodule table.

### E6: A monitor could have caught it `[verified]`

The rolling-spread monitor on the 512-D recurrent feature, calibrated LOCO to mean FPR 1.03% / max 2.07% (in-sample 1.15%), fires (>50% of frames flagged) at alpha=0.550, well before the E4 output-collapse cliff at alpha~0.784. The gap is ~0.23 alpha-units of early warning. Figure: `report/figures/e6_detector.png`, `report/figures/auroc_vs_alpha.png`.

**Threshold-free comparison at alpha=1.0** (AUROC mean [95% CI], `report/metrics_results.md`):

| detector | AUROC | AUPR | FPR@95TPR | LOCO mean FPR |
|---|---|---|---|---|
| E6 (rolling-spread) | 0.996 [0.992, 1.000] | 0.995 [0.990, 1.000] | 0.000 | **1.03%** |
| KNN-50 | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 0.000 | 100% |
| Relative Mahalanobis | 0.934 [0.914, 0.952] | 0.732 [0.684, 0.784] | 0.067 | 100% |
| Mahalanobis | 0.159 [0.130, 0.190] | 0.230 [0.217, 0.245] | 0.854 | 100% |
| PCA-Mahalanobis | 0.152 [0.124, 0.179] | 0.214 [0.209, 0.219] | 0.854 | 11.91% |

**The headline finding.** Location-sensitive feature-space scores fail on `supercombo` in two distinct ways. (1) Vanilla and PCA-Mahalanobis score *below chance* at alpha=1.0 (AUROC ~0.15): the recurrent state collapses *to the mean* of the ID Gaussian, and distance-from-mean cannot detect collapse-to-the-mean. (2) All three applicable baselines hit 100% LOCO FPR: the subaru and ram corpora occupy disjoint regions of the 512-D feature space (the feature encodes per-platform state at magnitudes that dwarf within-platform variance), so any absolute-position score that calibrates on one corpus flags the entire other corpus. E6 watches the *second-order trace* (location-invariant), so it both separates (AUROC 0.996) and calibrates across corpora (~1% FPR). The paper-worthy claim: on this production recurrent feature, standard OpenOOD post-hoc scores do not transfer; a second-order monitor does.

### E7: Corruption sweep, E6 is collapse-specific (a bounded result) `[verified — report/e7_results.md, 2026-05-29]`

We applied the 15 ImageNet-C corruptions (Hendrycks & Dietterich, ICLR 2019) at 5 severities to the real Subaru frames (raw RGB, pre-YUV), re-ran `supercombo` with correct recurrent state handling on each corrupted sequence (76 conditions, 319 frames each), and evaluated E6 and the baselines against clean real driving. Figures: `report/figures/e7_auroc_heatmap.png`, `report/figures/e7_severity_sweep.png`.

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

**What E7 changes about the paper's claim.** The earlier framing ("an internal monitor catches the OOD condition") must be narrowed: an internal monitor catches the *silent-collapse* condition, which the E1 overlay shows is **sim-specific** (no ImageNet-C corruption reproduces it). E6 is a targeted monitor for that specific, dangerous failure mode at ~1% real-driving FPR, where output-side and location-based feature detectors do not; on real-frame corruptions it is correctly quiet, and its few firings track a feature-spread shift rather than an output collapse. This is a narrower and more defensible contribution than "generalizes beyond CARLA," and it should be written as such.

---

## 6. Limitations `[DRAFT — source: docs/threat_model.md §4, strong]`

- **N=1 model:** supercombo v0.9.7 only; no other openpilot version, Tesla, Mobileye, Waymo, or research IL stack.
- **N=2 real corpora:** LOCO is a two-fold estimate; variance is not meaningfully reportable at N=2. A third corpus is needed before quoting a single production FPR.
- **OOD axis breadth and E6 selectivity:** CARLA-clean is an extreme, easy shift. The E7 ImageNet-C corruption axis (Subaru-only; RAM not in the sweep, `src/e7_corruption.py` lines 32-33) shows E6 is collapse-specific. The per-corruption E1 overlay (`report/e7_overlay_results.md`) resolves the earlier ambiguity: no corruption reproduces the output collapse (at most 1/10 heads vs 7/10 under CARLA), so E6's quiet response on the low-AUROC corruptions is correct (no collapse to miss), and its few firings track a feature-spread shift rather than collapse. The honest residual limitation is that this corruption axis therefore never confronts E6 with a *non-CARLA* output collapse; real adverse-weather footage (rain/night/glare) that actually induces collapse remains pending.
- **E5 localization is partial:** collapse pinned to summarizer VAE-mu + action-block feedback by ruling out the encoder and probing 8 submodules; a VAE-mu/sigma ambiguity remains (note it, source: report/e5_submodule_results.md).
- **No real-robot / on-road deployment of E6:** the monitor is demonstrated offline on logged + rendered + corrupted frames, not in the running stack.

---

## 7. Conclusion

`[DRAFT]` A shipped L2 driving model, shown out-of-distribution input, collapses to a plausible constant and does not raise its own uncertainty, so simulation "passes" can be false confidence. The signal the outputs hide is recoverable from the model's own recurrent features with a single O(d) statistic, no retraining, and a real-driving-calibrated false-positive rate. Output-side monitoring alone is insufficient for the safety case of a shipped driving model; an internal-feature monitor is a cheap, deployable complement.

---

## 8. Reproducibility `[verified — README documents a GPU-free, CARLA-free fresh-clone reproduce path from committed caches]`

- All result caches committed (`report/*_collected.npz`); analysis reruns from cache.
- Bootstrap params: n=1000, seed=42. Requirements pinned (Python 3.10, onnxruntime-gpu 1.23.2, numpy 2.2.6, carla 0.9.15).
- `[AUTHOR TODO — audit P1]` Verify `report/e5_collected.npz` (3.9 GB) and `report/e7_collected.npz` (110 MB) are tracked by Git LFS and survive a fresh clone + `git lfs pull`; if not, the E5/E7 "reproduce from cache" claim is false on the public repo. Pin `matplotlib` in requirements.txt.

---

## Figures (all exist in report/figures/ unless marked)

- `hero.png` (four findings at a glance), `e1_head_collapse.png`, `e2_feature_ood.png`, `e3_confidence.png`, `e4_interpolation.png`, `e5_layer_localization.png`, `e5_submodule_localization.png`, `e6_detector.png`, `roc_curves.png`, `pr_curves.png`, `auroc_vs_alpha.png`.
- `e7_auroc_heatmap.png`, `e7_severity_sweep.png` `[verified — generated 2026-05-29]`.

## Pre-submission checklist (from the 2026-05-29 publication-readiness audit; updated post-E7)

- [x] **P0:** E7 corruption sweep run (re-collected + analyzed 2026-05-29), subsection + 2 figures written. Result is a bounded negative (E6 collapse-specific), reflected in abstract/contrib-4/limitations.
- [x] **P0 (from E7):** E1 output-collapse-per-corruption overlay run (`src/e7_overlay.py`, `report/e7_overlay_results.md`, figure `e7_overlay.png`, 2026-05-30), with a validation gate that reproduces the CARLA collapse (7/10 heads). Resolved: collapse is sim-specific (<=1/10 heads on every corruption vs 7/10 CARLA), 0 false negatives, E6's four corruption firings decoupled from output collapse. Headline claim no longer gated; abstract/contrib-4/E7/limitations updated.
- [ ] **P1:** the E5 (3.9 GB) and E7 (110 MB) caches are `.gitignore`d, so they are NOT in the public repo at all (not merely un-LFS'd). The reproduce-from-cache path is broken for E5/E7. Decide: Git LFS the E7 cache (E5 at 3.9 GB exceeds even LFS norms, consider a regeneration script + smaller committed summary instead).
- [ ] **P1:** resolve six [UNVERIFIED] citations + Henriksson bibkey.
- [ ] **P2:** RMD-background one-sentence justification in §4.5.
- [ ] **P2:** clean "Agent A" artifact in docs/threat_model.md line 43.
- [ ] **P2:** state N for every percentage; pin matplotlib.

## Open authorship / venue decisions `[AUTHOR]`

- Author list / affiliation line (Wayne State; advisor co-author?).
- Final venue: SafeAI @ UAI 2026 (confirmed open) vs a NeurIPS 2026 workshop.
- Whether to fold in a second OOD axis (real adverse-weather) before submission or defer to the extended version.
