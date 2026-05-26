# Phantom-Braking, paper plan

Scope: position E6 (rolling-spread monitor on the 512-D recurrent feature of openpilot v0.9.7 supercombo) in the OOD-for-AV literature, lock the baseline set, and pick a non-CARLA OOD axis. Dates verified via arXiv ID where shown. Claims sourced only from search; training-data-only claims are flagged [UNVERIFIED].

---

## 1. Confirmed baseline list for E6

E6 produces a scalar score per frame from the recurrent feature vector, then thresholds it at a real-driving-calibrated FPR. The natural baseline set is therefore scalar post-hoc OOD scores operating on either the same feature vector or on the model outputs.

| Baseline | One-line method | Why include | Reference |
|---|---|---|---|
| MSP (Maximum Softmax Probability) | Max post-softmax probability of class head; lower means more OOD | Mandatory floor baseline in every OOD paper since 2017; cheap; if E6 cannot beat MSP, story collapses | Hendrycks and Gimpel, ICLR 2017, arXiv:1610.02136 |
| Energy score | LogSumExp over logits as a free-energy score | Standard 2020-2026 baseline; still cited as the strong simple post-hoc method in 2024-2025 surveys and in arXiv:2501.08083 (Jan 2025, AV input monitoring) | Liu et al., NeurIPS 2020, arXiv:2010.03759 |
| Mahalanobis on the same 512-D feature | Class-conditional Gaussian fit on ID features, score = min Mahalanobis distance | Direct apples-to-apples comparison with E6 (same layer, different score); still a "respected baseline" in 2024-2025, though performance varies with backbone (per arXiv:2505.18032, Mahalanobis++, May 2025) | Lee et al., NeurIPS 2018, arXiv:1807.03888 |
| Relative Mahalanobis (RMD) | Class-Mahalanobis minus global-Gaussian Mahalanobis | The fix to vanilla Mahalanobis for near-OOD; widely cited 2023-2025 as the fair upgrade; cheap to add on top of the Mahalanobis fit | Ren et al., 2021, arXiv:2106.09022 |
| KNN distance in feature space | k-th nearest neighbor distance to ID training features | Strong non-parametric baseline; 2022 ICML; still the reference non-parametric method in OpenOOD and in 2024 surveys | Sun et al., ICML 2022, arXiv:2204.06507 |
| ViM (Virtual-logit Matching) | Combines residual in feature null-space with logits | Modern hybrid baseline; the head-to-head NECO/Mahalanobis++ papers benchmark against ViM in 2024-2025 | Wang et al., CVPR 2022, arXiv:2203.10807 |

Flags and exclusions:
- ODIN (temperature + input perturbation). Treated as dated in 2024-2026 OOD-for-AV work; superseded by energy and Mahalanobis variants. Cite once for lineage, skip as a baseline.
- Deep ensemble disagreement. Not applicable: supercombo is a single shipped ONNX, no ensemble exists, and retraining for an ensemble is out of scope. Mention in limitations.
- GradNorm, ASH. Strong on ImageNet-style image classifiers operating on logits; supercombo has multi-head regression and YUV uint8 input, so logit-gradient and activation-shaping semantics do not transfer cleanly. Cite, do not include.
- NECO (ICLR 2024, arXiv:2403.18051 [UNVERIFIED arXiv ID, paper confirmed at ICLR 2024). Built on Neural Collapse, which is a classification-head property. Skip; mention as recent SOTA on classification OOD.

Recommended minimum set if space is tight: MSP, Energy, Mahalanobis (same layer), KNN. Add ViM and Relative Mahalanobis if reviewer pushes on "modern" baselines.

---

## 2. Related work, by theme

### 2.1 OOD detection in driving / AV perception

| Ref | Venue, year, ID | One-line relevance |
|---|---|---|
| Henriksson et al., Performance Analysis of OOD Detection on Various Trained Neural Networks | RefSQ 2023 / IEEE; warg.org PDF | Empirical OOD-detector comparison on AV datasets; standard citation for "OOD as part of the AV safety lifecycle" |
| Keser et al., Benchmarking Vision Foundation Models for Input Monitoring in Autonomous Driving | arXiv:2501.08083, Jan 2025 | Closest neighbor to E6: density estimation in feature space of a frozen perception model, framed as a safety monitor. Must-cite |
| Out-of-Distribution Detection for Safety Assurance of AI and Autonomous Systems | arXiv:2510.21254, Oct 2025 | Recent position paper on OOD as assurance evidence for shipped AI |
| Dynamic-Aware Adaptive Multi-Mode OOD Detection for Trajectory Prediction | arXiv:2509.13577, Sep 2025 | OOD on the trajectory-prediction stage of AV stacks; same problem class one layer downstream of supercombo |
| Co-Design of OOD Detectors for Autonomous Emergency Braking | arXiv:2307.13419, Jul 2023 | Connects OOD detection to AEB decisions; relevant because phantom braking is the AEB failure mode |
| Detecting What Matters: OOD 3D Object Detection in AVs | arXiv:2506.23426, Jun 2025 | 2025 reference for sensor-level OOD in shipped-style AV perception |

### 2.2 Feature-space and internal monitors

| Ref | Venue, year, ID | Relevance |
|---|---|---|
| Cheng et al., Runtime Monitoring Neuron Activation Patterns | arXiv:1809.06573, 2018; extended in arXiv:2011.11959 (Provably-Robust Runtime Monitoring) | Foundational paper on monitoring neural-network internals at runtime; direct intellectual ancestor of E6 |
| Stocco et al., SelfOracle / Misbehaviour Prediction for Autonomous Driving Systems | ICSE 2020, dl.acm.org/doi/10.1145/3377811.3380353; Tonella co-author on follow-ups | The canonical "predict misbehaviour from internal model confidence" paper for end-to-end driving; E6 is in the same lineage but uses recurrent features, not autoencoder reconstruction |
| Stocco et al., Predicting Safety Misbehaviours in ADS using Uncertainty Quantification | arXiv:2404.18573, Apr 2024 | 2024 update of the SelfOracle line; uncertainty-quantification angle |
| Parallel Activations Drift Detector | arXiv:2404.07776, Apr 2024 | 2024 baseline for unsupervised drift detection from activations; same family as E6 |
| Topological Uncertainty: Monitoring NNs through persistence of activation graphs | arXiv:2105.04404, 2021 | Alternative internal-monitor formulation; cite for completeness |

### 2.3 openpilot / supercombo prior work

| Ref | Venue, year, ID | Relevance |
|---|---|---|
| Chen et al., Level 2 Autonomous Driving on a Single Device: Diving into the Devils of Openpilot | arXiv:2206.08176, Jun 2022 (OpenDriveLab Openpilot-Deepdive) | The reference teardown of supercombo: input format, output heads, training pipeline. Anchor citation for the model description |
| Geretti et al., Finding Property Violations through Network Falsification: Lessons from OpenPilot | GPCE/SPLASH 2022, dl.acm.org/doi/10.1145/3551349.3559500 | Formal falsification on supercombo; safety-property angle on the same network |
| Revisiting Adversarial Perception Attacks and Defense Methods on AV Systems | arXiv:2505.11532, May 2025 | 2025 adversarial-robustness study including supercombo; companion topic to OOD |
| openpilot-supercombo-model (MTammvee, GitHub) | github.com/MTammvee/openpilot-supercombo-model | Public docs on input/output tensor layout; useful methodological cite, not a paper |
| commaai blog, openpilot 0.9.0 / 0.9.3 / 0.9.8 release notes | blog.comma.ai/090release, blog.comma.ai/093release, blog.comma.ai/098release | Primary source for model-version provenance; cite for v0.9.7 ancestry |
| commaai/openpilot issue #20704 (large-shadow phantom braking), discussion #22212 (shadow phantom braking) | github.com/commaai/openpilot/issues/20704 | Primary evidence that phantom braking under distribution shift is a known, user-reported failure mode of the shipped model |

### 2.4 Neural-network simulation testing for AVs (DeepRoad-line)

| Ref | Venue, year, ID | Relevance |
|---|---|---|
| Pei et al., DeepXplore: Automated Whitebox Testing of DL Systems | SOSP 2017 | Differential testing of DNNs; lineage citation, not a baseline |
| Tian et al., DeepTest: Automated Testing of DNN-driven Autonomous Cars | ICSE 2018 | Affine + weather filter test generation; lineage citation |
| Zhang et al., DeepRoad: GAN-based Metamorphic Autonomous Driving System Testing | ASE 2018, arXiv:1802.02295 | GAN-based weather corruption tests on driving DNNs; lineage citation and a possible source of OOD test images |
| Stocco et al., MarMot: Metamorphic Runtime Monitoring of ADS | arXiv:2310.07414, Oct 2023 | Runtime metamorphic monitoring on ADS; bridges DeepRoad lineage and the runtime-monitor lineage |

### 2.5 Safety monitoring / assurance of shipped models, OOD benchmarks

| Ref | Venue, year, ID | Relevance |
|---|---|---|
| Yang et al., OpenOOD: Benchmarking Generalized OOD Detection | NeurIPS 2022, arXiv:2210.07242 | Standardized OOD benchmark and codebase; cite as the reference taxonomy for near-OOD vs far-OOD framing |
| OpenOOD-VLM | ECCV / NeurIPS 2024; github.com/YBZh/OpenOOD-VLM | 2024 extension of the benchmark to foundation-model features; supports the "feature-space monitoring" framing |
| Hendrycks and Dietterich, ImageNet-C / Benchmarking NN Robustness to Common Corruptions | ICLR 2019 | The reference corruption benchmark; basis for the synthetic-corruption OOD axis |
| Michaelis et al., Object Detection when Winter is Coming (Pascal-C/Coco-C/Cityscapes-C) | arXiv:1907.07484, 2019 | Direct AV extension of ImageNet-C; cite if we go the synthetic-corruption route |
| Mahalanobis++: Improving OOD Detection via Feature Normalization | arXiv:2505.18032, May 2025 | 2025 evidence that Mahalanobis is still an active, respected baseline (not a strawman), with feature-normalization fixing the variance-across-backbones problem |
| A Geometry-Based View of Mahalanobis OOD Detection | arXiv:2510.15202, Oct 2025 | 2025 theoretical re-grounding of Mahalanobis; reinforces "include Mahalanobis, do not skip it" |
| NECO: Neural Collapse-based OOD Detection | ICLR 2024 (proceedings.iclr.cc 2024 paper 04b84142b99dae8560b517401e6e5275) | Recent SOTA on classification OOD; cite, do not run (head incompatible with supercombo) |

---

## 3. Non-CARLA OOD axis, recommendation

E6 is currently calibrated and tested on (real subaru+ram driving) vs (CARLA). Reviewers will press on whether the monitor is fitted to CARLA. We need one more OOD axis where E6 still fires before output collapse.

### 3.1 Option A, MetaDrive bridge

State of the bridge, verified from commaai/openpilot:
- Issue #31711 (Strange behavior of OpenPilot experimental mode in the simulator) confirms openpilot drives erratically inside MetaDrive (random U-turns across solid yellow). Reporter inspected the bridge CAN with Cabana, data was clean; the suspected cause is missing model inputs (no mapsd, etc.), not bad sim pixels. Issue is OPEN as of the searches above.
- Related: issue #34044 flaky MetaDrive test; PR #34045 macOS run; issue #30693 CI integration; issue #30913 modeld not running under MetaDrive; issue #31797 offscreen-window failure.

Implication: MetaDrive is, at the bridge level, openpilot's own admission of an OOD-like failure on a non-CARLA sim. Engineering effort to run E6 on MetaDrive frames is moderate (the bridge already provides YUV frames at 256x512), but the confound is severe: erratic behavior in MetaDrive is at least partly explained by missing services, not by pixel-OOD. Hard to make a clean claim.

- Effort: 3-5 days (stand up bridge, hook E6 on frame tap, gather n>=1k frames per condition).
- Credibility: Medium-low. Reviewer can argue the bridge confound.
- Data availability: High, MetaDrive is permissively licensed and the bridge ships with openpilot.

### 3.2 Option B, real-comma rain / night / glare segments

Searches confirm comma2k19 (arXiv:1812.05752) is highway-California daytime, dry. It is NOT a clean source of rain/night/glare. Real adverse-weather comma footage would have to come from commadataci.blob.core.windows.net (the source used in our Step 3.5) by filtering for night and weather, which is a tagging effort, not an API.

- Effort: 5-8 days (curate, hand-tag, decode HEVC, re-run E6 calibration). Risk of small final n.
- Credibility: High. Real pixels, same sensor, same model deployment domain. Most reviewer-resistant.
- Data availability: Medium. Public blob exists, but rain/night curation is manual and capacity-limited.

### 3.3 Option C, ImageNet-C-style synthetic corruptions on real frames

Apply the 15 ImageNet-C corruptions (gaussian noise, defocus blur, motion blur, snow, fog, brightness, contrast, etc.), 5 severities, on the existing real-comma frames already in our pipeline. Re-run E6 and the baselines. Plot score vs severity.

- Effort: 1-2 days (corruption code is public; preprocessor already in src/preprocessor.py).
- Credibility: High for the ML4AD / SafeAI audience because Hendrycks ImageNet-C and Michaelis Cityscapes-C are the standard robustness yardsticks. Lower than real adverse weather but standard.
- Data availability: Trivial (corruption is synthetic on top of frames we already have).
- Risk: Per recent search results, many ImageNet-C corruptions now appear in web-scraped pretraining data, so some corruption types may be near-ID for modern backbones. Supercombo is small and AV-trained, so this risk is lower than for VFMs, but it should be acknowledged.

### 3.4 Recommendation

Pick Option C (ImageNet-C-style synthetic corruptions on real frames). Reason: it gives a per-severity dose-response curve for E6 vs all baselines at 1-2 days of effort, uses an established standard benchmark, and is decoupled from the CARLA pipeline so reviewers cannot claim E6 is sim-fitted. Option B is the stronger result but the curation risk is large for a paper deadline. Option A is the weakest because the MetaDrive bridge has a known confound (issue #31711) that contaminates the OOD signal.

If time allows after C, add B as a smaller adverse-weather case study. Skip A unless we have a separate reason (e.g., reviewer specifically asks for sim-to-sim).

---

## 4. Open uncertainties

- NECO exact arXiv ID not pinned by search (paper confirmed at ICLR 2024 via proceedings PDF link). Marked [UNVERIFIED] inline. Verify before bibliography freeze.
- commadataci.blob.core.windows.net rain/night availability not verified by search; only inferred from our prior Step 3.5 usage. Verify by sampling before committing to Option B.
- Henriksson et al. surfaces under multiple titles (RefSQ 2023; "Performance Analysis of OOD Detection on Various Trained Neural Networks"). Pin the specific paper and bibkey before submission.
