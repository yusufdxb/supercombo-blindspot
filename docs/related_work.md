# Related Work

Phantom-Braking sits at the intersection of out-of-distribution (OOD) detection, internal-feature monitoring, simulation-based testing of driving models, and reverse-engineering of shipped autonomy stacks. We organize prior work by theme and position each cluster relative to E1-E6.

Citations marked [UNVERIFIED] remain pending a venue/ID pin; the rest are reconciled against the authoritative literature search in `docs/paper_plan.md`.

## 1. OOD detection for neural networks (general)

The closest baselines for E6 are post-hoc OOD detectors that score a sample using a trained classifier's own outputs or activations. Phantom-Braking adapts this lineage to a temporal recurrent state on a shipped driving model.

- Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks," ICLR 2017, arXiv:1610.02136. Maximum softmax probability (MSP) as the canonical output-side OOD baseline; E3 shows this fails for supercombo because uncertainty heads barely move on CARLA (1.20-1.84x vs real, 0% of sim frames exceed real p95).
- Liang, Li, and Srikant, "Enhancing the Reliability of Out-of-Distribution Image Detection in Neural Networks (ODIN)," ICLR 2018, arXiv:1706.02690. [UNVERIFIED] Temperature scaling + input perturbation on softmax. Per paper_plan.md, ODIN is treated as dated in 2024-2026 OOD-for-AV work and superseded by energy and Mahalanobis variants; we cite it for lineage only and do not run it as a baseline.
- Lee, Lee, Lee, and Shin, "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks," NeurIPS 2018, arXiv:1807.03888. Class-conditional Mahalanobis distance at the penultimate layer; the closest methodological ancestor to E6, which monitors a recurrent feature vector (see Theme 3).
- Liu, Wang, Owens, and Li, "Energy-based Out-of-distribution Detection," NeurIPS 2020, arXiv:2010.03759. Free-energy (LogSumExp over logits) score; the standard strong post-hoc baseline through 2024-2025 surveys.
- Ren et al., "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection," 2021, arXiv:2106.09022. Relative Mahalanobis (RMD): class-Mahalanobis minus global-Gaussian Mahalanobis; the canonical near-OOD fix to the Lee 2018 fit and a planned baseline for E6.
- Sun, Ming, Zhu, and Li, "Out-of-Distribution Detection with Deep Nearest Neighbors," ICML 2022, arXiv:2204.06507. Non-parametric kNN distance in feature space; conceptually compatible with E6's spread monitor, and the reference non-parametric baseline in OpenOOD.
- Wang et al., "ViM: Out-Of-Distribution with Virtual-logit Matching," CVPR 2022, arXiv:2203.10807. Hybrid baseline combining feature null-space residual with logits; included as a "modern" baseline against E6.
- Lakshminarayanan, Pritzel, and Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," NeurIPS 2017, arXiv:1612.01474. [UNVERIFIED] Ensemble disagreement as an uncertainty signal. Per paper_plan.md, not applicable as a baseline: supercombo is a single shipped ONNX, no ensemble exists, and retraining for one is out of scope. Mentioned in limitations.

## 2. OOD detection in autonomous driving

This literature is smaller and tends to target research models with clean labels rather than shipped production stacks. Phantom-Braking's contribution here is to apply an internal-feature monitor to the exact ONNX bundle that runs on a deployed openpilot fleet.

- Keser et al., "Benchmarking Vision Foundation Models for Input Monitoring in Autonomous Driving," arXiv:2501.08083, Jan 2025. The closest neighbor to E6: density estimation in feature space of a frozen perception model, framed as a safety monitor. E6 differs by operating on a recurrent feature vector of a shipped end-to-end driving model rather than a foundation-model encoder.
- Henriksson et al., "Performance Analysis of OOD Detection on Various Trained Neural Networks," RefSQ 2023 / IEEE. Empirical OOD-detector comparison on AV datasets; standard citation for OOD as part of the AV safety lifecycle.
- "Out-of-Distribution Detection for Safety Assurance of AI and Autonomous Systems," arXiv:2510.21254, Oct 2025. Recent position paper on OOD as assurance evidence for shipped AI; frames the safety-case role that E6 is intended to fill.
- "Dynamic-Aware Adaptive Multi-Mode OOD Detection for Trajectory Prediction," arXiv:2509.13577, Sep 2025. OOD applied to the trajectory-prediction stage of AV stacks; same problem class one layer downstream of supercombo.
- "Co-Design of OOD Detectors for Autonomous Emergency Braking," arXiv:2307.13419, Jul 2023. Connects OOD detection to AEB decisions; directly relevant because phantom braking is the AEB failure mode E6 targets.
- "Detecting What Matters: OOD 3D Object Detection in AVs," arXiv:2506.23426, Jun 2025. 2025 reference for sensor-level OOD in shipped-style AV perception.
- Bogdoll, Nitsche, and Zoellner, "Anomaly Detection in Autonomous Driving: A Survey," CVPR Workshops 2022, arXiv:2204.07974. [UNVERIFIED] Survey across sensor-, object-, and scenario-level anomaly detection; positions where recurrent-state monitoring falls in the taxonomy.
- Filos, Tigkas, McAllister, Rhinehart, Levine, and Gal, "Can Autonomous Vehicles Identify, Recover From, and Adapt to Distribution Shifts?" ICML 2020, arXiv:2006.14911. [UNVERIFIED] Distribution shift on a research IL stack; complements our finding that on a shipped model the standard output-side uncertainty channel is silent (E3).
- McAllister et al., "Concrete Problems for Autonomous Vehicle Safety: Advantages of Bayesian Deep Learning," IJCAI 2017. [UNVERIFIED] Foundational argument for runtime uncertainty in driving; E3 is exactly the failure mode this paper warned about.

Position: E6 is one of the few OOD detectors evaluated on a shipped production driving model (supercombo v0.9.7) with parity-exact preprocessing. Most prior work uses research models trained from scratch by the authors.

## 3. Feature-space and internal-state monitors

E6 monitors the temporal spread of supercombo's 512-D recurrent feature vector. The lineage is feature-space OOD (Mahalanobis-style), but adapted to a temporal driving model where the relevant state is recurrent, not feed-forward.

- Lee et al. (2018), Mahalanobis distance OOD (see Theme 1). Canonical penultimate-layer feature monitor; assumes a single feed-forward feature vector per input. E6 differs because the monitored vector is a recurrent state that already integrates temporal history.
- Mueller et al., "Mahalanobis++: Improving OOD Detection via Feature Normalization," arXiv:2505.18032, May 2025. Evidence that Mahalanobis is still an active, respected baseline (not a strawman) once feature normalization fixes the variance-across-backbones problem; supports including the Lee 2018 fit head-to-head against E6.
- "A Geometry-Based View of Mahalanobis OOD Detection," arXiv:2510.15202, Oct 2025. 2025 theoretical re-grounding of Mahalanobis; reinforces inclusion rather than dismissal of the feature-Gaussian family.
- Keser et al., arXiv:2501.08083 (see Theme 2). The closest published lineage point for E6: density estimation in a frozen perception model's feature space, framed as an input monitor for AV. E6 inherits this framing and pushes it to a recurrent state on a shipped, end-to-end driving stack rather than a foundation-model encoder.
- Cheng et al., "Runtime Monitoring Neuron Activation Patterns," arXiv:1809.06573, 2018; extended in arXiv:2011.11959 (Provably-Robust Runtime Monitoring). Foundational work on monitoring neural-network internals at runtime; direct intellectual ancestor of E6.
- Stocco et al., "Misbehaviour Prediction for Autonomous Driving Systems (SelfOracle)," ICSE 2020. The canonical "predict misbehaviour from internal model confidence" paper for end-to-end driving; E6 is in the same lineage but uses recurrent features rather than autoencoder reconstruction.
- Stocco et al., "Predicting Safety Misbehaviours in ADS using Uncertainty Quantification," arXiv:2404.18573, Apr 2024. 2024 update of the SelfOracle line via uncertainty quantification.
- Parallel Activations Drift Detector, arXiv:2404.07776, Apr 2024. 2024 baseline for unsupervised drift detection from activations; same family as E6.
- Sastry and Oore, "Detecting Out-of-Distribution Examples with Gram Matrices," ICML 2020, arXiv:1912.12510. [UNVERIFIED] Higher-order feature statistics at intermediate layers; closer in spirit to a "spread" statistic than Mahalanobis.
- Sun and Li, "DICE: Leveraging Sparsification for Out-of-Distribution Detection," ECCV 2022, arXiv:2111.09805. [UNVERIFIED] Activation sparsification at the penultimate layer; relevant as a contrast: under CARLA, supercombo's recurrent state does not just sparsify, its temporal spread collapses (E4 feature-spread row drops to 0.00 by alpha=0.78).
- Guo and Su, "Latent Dynamics-Aware OOD Monitoring for Trajectory Prediction with Provable Guarantees," arXiv:2603.14603, Mar 2026. Direct neighbor of E6: monitors the latent/recurrent state of a trajectory prediction model for OOD using quickest changepoint detection with conformal coverage guarantees. Differentiation: E6 uses rolling spread on a shipped end-to-end driving stack with no provable guarantees; Guo & Su 2026 uses conformal/QCD on a standalone trajectory predictor with coverage guarantees but does not target a shipped production model.

Position: E6 inherits feature-space OOD's methodology and applies it to a temporal recurrent state on a shipped driving model. The closest published precedent is Keser et al. 2025 (input monitoring on a frozen perception model); the gap E6 closes is doing this on the recurrent state of an end-to-end shipped stack rather than on a foundation-model encoder. The monitored quantity is spread across a sliding window, which makes it sensitive to the "freeze" mode E4 identifies (feature spread crashes from 0.25 to 0.00 across the cliff).

## 4. Simulation-based testing of neural driving models

This thread assumes the simulator is in-distribution to the model under test. Phantom-Braking's E1-E4 invalidate that assumption for openpilot v0.9.7 on CARLA-clean: the model's outputs collapse on the cleanest possible sim input, before any adversarial perturbation is applied.

- Pei, Cao, Yang, and Jana, "DeepXplore: Automated Whitebox Testing of Deep Learning Systems," SOSP 2017. Neuron-coverage-guided differential testing; lineage citation, not a baseline.
- Tian, Pei, Jana, and Ray, "DeepTest: Automated Testing of Deep-Neural-Network-driven Autonomous Cars," ICSE 2018. Affine and weather metamorphic transformations on driving images; the transformations are assumed to preserve the driving label, but if the base image is already OOD the labels are not meaningful.
- Zhang et al., "DeepRoad: GAN-Based Metamorphic Testing and Input Validation Framework for Autonomous Driving Systems," ASE 2018, arXiv:1802.02295. Style-transfer-based testing; demonstrates the OOD problem we measure: a GAN-rendered sim frame can be far from the training distribution of the model under test.
- Stocco et al., "MarMot: Metamorphic Runtime Monitoring of ADS," arXiv:2310.07414, Oct 2023. Runtime metamorphic monitoring on ADS; bridges the DeepRoad lineage and the runtime-monitor lineage E6 sits in.
- Dosovitskiy, Ros, Codevilla, Lopez, and Koltun, "CARLA: An Open Urban Driving Simulator," CoRL 2017, arXiv:1711.03938. [UNVERIFIED] The simulator we use; we do not modify CARLA, we measure how supercombo responds to it.
- Norden, O'Kelly, and Sinha, "Efficient Black-box Assessment of Autonomous Vehicle Safety," arXiv:1912.03618, 2019. [UNVERIFIED] Adversarial scenario search; assumes the simulator is faithful enough that adversarial scenarios transfer to the real model. E1-E4 push back on this for visually-driven models.

Position: We do not propose a new testing method. We document that the prerequisite of these methods (sim is in-distribution to the model) fails empirically for a shipped driving model, and we provide a monitor (E6) that can detect when that prerequisite has failed at runtime.

## 5. Reverse engineering and teardowns of shipped driving models

Public teardowns of production driving models are rare. Most exist as community blog posts and reproductions rather than peer-reviewed papers.

- Chen et al., "Level 2 Autonomous Driving on a Single Device: Diving into the Devils of Openpilot," arXiv:2206.08176, Jun 2022 (OpenDriveLab Openpilot-Deepdive). The reference academic teardown of supercombo: input format, output heads, training pipeline. Anchor citation for our parity-exact reproduction; we extend it from a static input/output description to a runtime distribution-shift teardown.
- Geretti et al., "Finding Property Violations through Network Falsification: Lessons from OpenPilot," GPCE/SPLASH 2022. Formal falsification on supercombo; safety-property angle on the same network E6 monitors.
- "Revisiting Adversarial Perception Attacks and Defense Methods on AV Systems," arXiv:2505.11532, May 2025. 2025 adversarial-robustness study including supercombo; companion topic to OOD.
- comma.ai, "openpilot," github.com/commaai/openpilot. The model under test; the in-repo `selfdrive/modeld/` pipeline is the reference for our parity-exact preprocessing (YUV unnormalized, recurrent state rolled, two-channel Y plane ordering per the kernel).
- openpilot-supercombo-model, github.com/MTammvee/openpilot-supercombo-model. Public docs on input/output tensor layout; we cross-checked input layout against `loadyuv.cl` and `modeld` source rather than the README, which had Y channels 1 and 2 swapped.
- comma.ai blog, openpilot 0.9.0 / 0.9.3 / 0.9.8 release notes (blog.comma.ai/090release, /093release, /098release). Primary source for model-version provenance; cited for v0.9.7 ancestry.
- commaai/openpilot issue #20704 (large-shadow phantom braking) and discussion #22212. Primary user-reported evidence that phantom braking under distribution shift is a known failure mode of the shipped model.
- Codevilla, Santana, Lopez, and Gaidon, "Exploring the Limitations of Behavior Cloning for Autonomous Driving," ICCV 2019, arXiv:1904.08980. [UNVERIFIED] Limitations of IL-trained driving stacks; complements our finding that visual distribution shift collapses outputs even when the IL model is mature and shipped.

Position: Phantom-Braking contributes a parity-exact reproduction of supercombo v0.9.7 inference, plus a safety-relevant finding (E3 silent failure of the uncertainty heads) and a runtime monitor (E6) calibrated against the model's own real-driving behavior. The closest published precedent in the AV literature is Keser et al. 2025 (input monitoring on a frozen perception model, foundation-model features); E6 differs by operating on the recurrent state of a shipped end-to-end driving stack and pairing the monitor with a diagnostic teardown (E1-E5) under leave-one-corpus-out FPR.
