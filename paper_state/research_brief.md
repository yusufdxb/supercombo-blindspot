# Research Brief

Written by paper-deep-researcher on 2026-05-30. Bound by
`paper_state/contribution_contract.md` (locked 2026-05-30) and the framing in
`paper_state/framing_memo.md`. Scope: a bounded NEGATIVE result on runtime OOD /
silent-failure detection for ONE shipped end-to-end driving model (openpilot v0.9.7
supercombo), with a collapse-specific recurrent-feature monitor (E6). No claim in this
brief exceeds the contract boundary (N=1 model, collapse-specific monitor, no
generalization, N=2 LOCO is a two-fold estimate not a production FPR).

Verification rule applied: every source below carries a quote I FETCHED this run via
WebFetch or a direct-link WebSearch, with the URL/arXiv id. Sources I could not confirm,
or whose author/venue I found to be WRONG in the existing `docs/related_work.md` /
`docs/paper_plan.md`, are flagged explicitly in the "Citation corrections" section and
must be fixed before drafting. No citation here rests on model memory.

---

## Sub-questions (the decomposition)

1. What is the canonical OOD-detection lineage every reviewer expects to see cited for a
   feature-space / output-side monitor (MSP, Mahalanobis, RMD, KNN, Energy, ViM, neuron
   monitoring)? (Cluster A)
2. What are the 2024-2026 neighbors that define "recent" for OOD-on-a-recurrent-state /
   input-monitoring-for-AV, and exactly how close are the two closest (Keser 2025, Guo and
   Su 2026)? (Cluster B)
3. What benchmarks and datasets are table stakes for the corruption-bound and the OOD
   framing (OpenOOD, ImageNet-C, Cityscapes-C/Coco-C), and which is the right yardstick for
   the E7 bound? (Cluster C)
4. What is the competing-method set (the post-hoc baselines E6 is compared against), and
   what are their reported headline behaviors and known limits? (Cluster D)
5. What is the openpilot/supercombo prior-art anchor (Chen 2022) and the motivation
   evidence (the phantom-braking issue, falsification work), and is the existing citation
   metadata correct? (Cluster D + corrections)
6. What does the target venue (SafeAI @ UAI 2026 workshop) actually require: page limit,
   format, review model, archival status, topics, deadline? And what are arXiv-preprint
   norms for a safety/teardown negative result? (Cluster E)
7. Where is the field settled vs contested, and which open gaps can this paper exploit
   without exceeding N=1? (synthesis + open gaps)

---

## Cluster A: Canonical lineage

- **Hendrycks and Gimpel, "A Baseline for Detecting Misclassified and Out-of-Distribution
  Examples in Neural Networks," ICLR 2017.** FETCHED QUOTE: "Correctly classified examples
  tend to have greater maximum softmax probabilities than erroneously classified and
  out-of-distribution examples, allowing for their detection." Source: arXiv:1610.02136.
  Why it matters: the mandatory MSP floor baseline. E3 is precisely the failure of this
  output-side family on supercombo (uncertainty heads rise only 1.20x-1.84x; 0 of 220 OOD
  frames cross real p95), so MSP must be named and shown to fail.

- **Lee, Lee, Lee, Shin, "A Simple Unified Framework for Detecting Out-of-Distribution
  Samples and Adversarial Attacks," NeurIPS 2018.** FETCHED QUOTE: "We obtain the class
  conditional Gaussian distributions with respect to (low- and upper-level) features of the
  deep models under Gaussian discriminant analysis, which result in a confidence score
  based on the Mahalanobis distance." Source: arXiv:1807.03888. Why it matters: the
  Mahalanobis ancestor and the closest methodological lineage point for E6 (same feature
  vector, first-order distance-from-mean score). It is a baseline here and, per the contract,
  scores below chance and 100% LOCO FPR, which is the load-bearing "location-based scores do
  not transfer" comparison.

- **Ren, Fort, Liu, Roy, Padhy, Lakshminarayanan, "A Simple Fix to Mahalanobis Distance for
  Improving Near-OOD Detection," 2021.** FETCHED QUOTE: "We analyze its failure modes for
  near-OOD detection and propose a simple fix called relative Mahalanobis distance (RMD)
  which improves performance and is more robust to hyperparameter choice." Source:
  arXiv:2106.09022. Why it matters: RMD is the canonical near-OOD upgrade to Lee 2018; a
  reviewer demands it as the "fair upgrade" baseline. It is in the E6 baseline set and also
  fails to transfer (100% LOCO FPR per contract).

- **Sun, Ming, Zhu, Li, "Out-of-Distribution Detection with Deep Nearest Neighbors," ICML
  2022.** FETCHED QUOTE: "In this paper, we explore the efficacy of non-parametric
  nearest-neighbor distance for OOD detection, which has been largely overlooked in the
  literature." Source: arXiv:2204.06507. Why it matters: KNN is the strongest applicable
  baseline and the reference non-parametric method in OpenOOD. Per the contract, KNN-50
  TIES E6 at AUROC 1.000 at alpha=1.0, so the honest delta is transfer/calibration (KNN hits
  100% LOCO FPR), NOT raw separation. This row enforces the exclusion-list ban on "E6 beats
  baselines."

- **Liu, Wang, Owens, Li, "Energy-based Out-of-distribution Detection," NeurIPS 2020.**
  FETCHED QUOTE: "Unlike softmax confidence scores, energy scores are theoretically aligned
  with the probability density of the inputs and are less susceptible to the overconfidence
  issue." Source: arXiv:2010.03759. Why it matters: the standard strong simple post-hoc
  baseline through the 2024-2025 surveys; an output-side score that belongs in the baseline
  comparison alongside MSP.

- **Wang, Li, Feng, Zhang, "ViM: Out-Of-Distribution with Virtual-logit Matching," CVPR
  2022.** FETCHED QUOTE: "an additional logit representing the virtual OOD class is generated
  from the residual of the feature against the principal space." Source: arXiv:2203.10807.
  Why it matters: the modern hybrid (feature-residual + logit) baseline that 2024-2025
  head-to-head papers benchmark against; include it to pre-empt the "no modern baseline"
  reviewer.

- **Cheng, Nuhrenberg, Yasuoka, "Runtime Monitoring Neuron Activation Patterns," 2018.**
  FETCHED QUOTE: "In operation, a classification decision over an input is further
  supplemented by examining if a pattern similar (measured by Hamming distance) to the
  generated pattern is contained in the monitor." Source: arXiv:1809.06573. Why it matters:
  the intellectual ancestor of internal-state runtime monitoring. Delta is real: binarized
  per-frame activation patterns on a classifier vs a continuous second-order spread of a
  recurrent state on a temporal driver; the freeze mode (spread crashing across the cliff)
  is not what a Hamming-distance pattern check naturally captures.

- **Sastry and Oore, "Detecting Out-of-Distribution Examples with Gram Matrices," ICML
  2020.** FETCHED QUOTE (arXiv version): "We find that characterizing activity patterns by
  Gram matrices and identifying anomalies in gram matrix values can yield high OOD detection
  rates." Source: arXiv:1912.12510; proceedings: https://proceedings.mlr.press/v119/sastry20a.html
  (ICML 2020, pp. 8491-8501). Why it matters: the closest lineage point to a "higher-order
  feature statistic" framing (E6 is a second-order spread, not a first-order distance). The
  ICML 2020 venue in `related_work.md` is CORRECT; resolves its [UNVERIFIED] flag.

Note on lineage completeness: the brief is intentionally NOT padded with GradNorm/ASH/DICE
as baselines, matching the paper_plan exclusion logic (logit-gradient/activation-shaping
semantics do not transfer to a multi-head YUV-uint8 regression model). Cite-only, per plan.

---

## Cluster B: Recent neighbors (2024-2026)

- **Keser, Orhan, Amini-Naieni, Schwalbe, Knoll, Rottmann, "Benchmarking Vision Foundation
  Models for Input Monitoring in Autonomous Driving," 2025.** FETCHED QUOTE: "Find a full
  model of the training data's feature distribution, to then use its density at new points
  as in-distribution (ID) score." Source: arXiv:2501.08083. Why it matters: THE closest
  published neighbor. It monitors input/feature DENSITY (a location-based ID score) of a
  FROZEN vision-foundation-model encoder. E6's delta is substrate (recurrent state of a
  shipped end-to-end driver, one stage downstream of the encoder), statistic (second-order
  spread, location-invariant, vs density), and that the location-based class Keser
  instantiates is exactly the class that hits 100% LOCO FPR here.

- **Guo and Su, "Latent Dynamics-Aware OOD Monitoring for Trajectory Prediction with
  Provable Guarantees," 2026.** FETCHED QUOTES: "formulate OOD monitoring for trajectory
  prediction as a quickest changepoint detection (QCD) problem that offers a principled
  statistical framework with established theory"; "admitting provable guarantees on delay
  and false alarms"; "by leveraging this structure we extend the cumulative Maximum Mean
  Discrepancy approach to enable detection." Source: arXiv:2603.14603 (submitted 15 Mar
  2026; authors Tongfei Guo, Lili Su). Why it matters: the second-closest neighbor. It
  monitors latent dynamics of a STANDALONE trajectory predictor with PROVABLE QCD/MMD
  guarantees. E6's delta: a SHIPPED production end-to-end model (parity-verified harness),
  NO provable guarantee (empirical LOCO FPR instead), paired with a collapse teardown.
  CAVEAT: the arXiv id 2603.14603 is a forward-numbered 2026 preprint with no final venue
  yet; I confirmed title + authors + trajectory-prediction target in two fetches. Drafter
  must re-pin venue/DOI before camera-ready.

- **Guo and Su, "Dynamic Aware: Adaptive Multi-Mode Out-of-Distribution Detection for
  Trajectory Prediction in Autonomous Vehicles," 2025.** FETCHED QUOTE: "By explicitly
  modeling these error modes, our method achieves substantial improvements in both detection
  delay and false alarm rates." Source: arXiv:2509.13577 (same authors as the 2026 paper).
  Why it matters: the one-layer-downstream sibling (OOD on the trajectory-prediction stage).
  Reinforces that the active 2025-2026 frontier for AV OOD is at the predictor/latent-state
  level, NOT on the recurrent state of a shipped end-to-end stack, which is E6's open lane.

- **Grewal, Tonella, Stocco, "Predicting Safety Misbehaviours in Autonomous Driving Systems
  using Uncertainty Quantification," 2024.** FETCHED QUOTE: "This paper evaluates different
  Bayesian uncertainty quantification methods from the deep learning domain for the
  anticipatory testing of safety-critical misbehaviours during system-level simulation-based
  testing." Source: arXiv:2404.18573. Why it matters: the 2024 continuation of the SelfOracle
  line (predict misbehaviour from internal model confidence). E6 is in this lineage but uses
  a recurrent-feature second-order statistic, not Bayesian UQ or autoencoder reconstruction,
  and crucially shows the model's OWN uncertainty channel is silent (E3), which is the gap
  UQ-based monitors assume away.

- **Mueller and Hein, "Mahalanobis++: Improving OOD Detection via Feature Normalization,"
  2025.** FETCHED QUOTES: "We show that simple l2-normalization of the features mitigates
  this problem effectively"; Mahalanobis-on-pre-logit-features is "among the most effective
  for ImageNet-scale OOD detection." Source: arXiv:2505.18032. Why it matters: 2025 evidence
  that Mahalanobis is a LIVE, respected baseline (not a strawman), so running Lee 2018
  head-to-head and showing it fails to transfer is a fair, current comparison, not a beat-up
  of a dead method.

- **Hodge, Paterson, Habli, "Out-of-Distribution Detection for Safety Assurance of AI and
  Autonomous Systems," 2025.** FETCHED QUOTE: "Demonstrating the safety of autonomous systems
  rigorously is critical for their responsible adoption but it is challenging as it requires
  robust methodologies that can handle novel and uncertain situations throughout the system
  lifecycle, including detecting out-of-distribution (OoD) data." Source: arXiv:2510.21254.
  Why it matters: a 2025 position paper that frames the exact safety-case role E6 is meant to
  fill; good for the intro's stakes paragraph and for the SafeAI audience.

- **Ben Ammar, Belkhir, Popescu, Manzanera, Franchi, "NECO: NEural Collapse Based
  Out-of-distribution detection," ICLR 2024.** FETCHED QUOTE: "We introduce NECO, a novel
  post-hoc method for OOD detection, which leverages the geometric properties of 'neural
  collapse' and of principal component spaces to identify OOD data." Source: arXiv:2310.06823.
  Why it matters: recent (ICLR 2024) SOTA on classification OOD. Cite-only, do NOT run:
  neural collapse is a classification-head property and supercombo is multi-head regression.
  Naming it pre-empts the "why not the newest method" reviewer.

---

## Cluster C: Benchmarks and datasets

- **OpenOOD (Yang et al., "OpenOOD: Benchmarking Generalized Out-of-Distribution
  Detection," NeurIPS 2022 Datasets and Benchmarks).** FETCHED QUOTE: "we build a unified,
  well-structured codebase called OpenOOD, which implements over 30 methods developed in
  relevant fields and provides a comprehensive benchmark." Source: arXiv:2210.07242. What it
  measures: standardized near-OOD vs far-OOD detection across 30+ methods. Who uses it: it
  is THE reference taxonomy/codebase reviewers expect for the near-OOD/far-OOD framing and
  for the baseline definitions (MSP, Mahalanobis, KNN, Energy, ViM all live here). Standard
  metric: AUROC / FPR95. Venue note: cite as the taxonomy anchor; this paper is NOT a
  leaderboard submission (no SOTA claim, per contract), so OpenOOD grounds the baseline
  vocabulary, it is not a benchmark this paper "ranks on."

- **ImageNet-C (Hendrycks and Dietterich, "Benchmarking Neural Network Robustness to Common
  Corruptions and Perturbations," ICLR 2019).** FETCHED QUOTE: "Our first benchmark,
  ImageNet-C, standardizes and expands the corruption robustness topic, while showing which
  classifiers are preferable in safety-critical applications." Source: arXiv:1903.12261 (ICLR
  2019). What it measures: robustness to 15 algorithmically-generated corruption types x 5
  severities. Who uses it: the de-facto corruption-robustness yardstick for the ML4AD/SafeAI
  audience. Standard metric: mean corruption error (mCE); in this paper, applied as the OOD
  axis for the E7 bound (AUROC of E6 vs baselines per corruption-severity cell). Venue note:
  this IS the expected synthetic-corruption benchmark; the E7 result (no ImageNet-C
  corruption reproduces the collapse; E6 mean AUROC 0.52-0.74 on photometric families) is
  the bound that keeps the contribution inside "collapse-specific, not universal OOD."
  CORRECTION: `docs/paper_plan.md` line 79 lists ImageNet-C as "ICLR 2019" with no arXiv id;
  the arXiv id is 1903.12261. (The ID 1807.01697 is the older 2018 preprint title; the ICLR
  2019 version is 1903.12261. Pin 1903.12261.)

- **Cityscapes-C / Pascal-C / Coco-C (Michaelis et al., "Benchmarking Robustness in Object
  Detection: Autonomous Driving when Winter is Coming," 2019).** FETCHED QUOTE: "The three
  resulting benchmark datasets, termed Pascal-C, Coco-C and Cityscapes-C, contain a large
  variety of image corruptions." Source: arXiv:1907.07484. What it measures: the ImageNet-C
  corruption suite ported to AV object detection. Who uses it: the standard AV-specific
  corruption reference. Venue note: cite as the AV extension that justifies the corruption
  axis for a driving model; the existing `related_work.md`/`paper_plan.md` title is slightly
  off (they say "Object Detection when Winter is Coming"; the actual title leads with
  "Benchmarking Robustness in Object Detection: Autonomous Driving when Winter is Coming").
  Minor; pin the full title.

- **comma2k19 (Schafer, Santana, Haden, Biasini, "A Commute in Data: The comma2k19
  Dataset," 2018).** FETCHED QUOTE: "a dataset of over 33 hours of commute in California's
  280 highway ... 2019 segments, 1 minute long each, on a 20km section of highway driving
  between California's San Jose and San Francisco." Source: arXiv:1812.05752. What it
  measures: real comma sensor footage, highway/daytime/dry. Who uses it: comma's own public
  driving dataset. Venue note: confirms the paper_plan finding that comma2k19 is NOT a clean
  source of rain/night/glare (it is California highway daytime dry), which is why the real
  adverse-weather OOD axis (Option B) is curation-gated and correctly parked out of scope.

---

## Cluster D: Competing methods

The "competitors" for this paper are the post-hoc OOD scores E6 is benchmarked against
(all already in Cluster A with their canonical quotes). Their RELEVANT behavior here,
per the locked contract, is:

- **Mahalanobis (Lee 2018, arXiv:1807.03888):** distance-from-fitted-Gaussian-mean on the
  same 512-D recurrent feature. Reported behavior here: scores BELOW chance (AUROC 0.159 per
  framing memo) because the recurrent state collapses TO the ID mean, and 100% LOCO FPR.
  Known limitation (external, FETCHED): Mahalanobis++ (arXiv:2505.18032) shows vanilla
  Mahalanobis is variance-sensitive across backbones, fixed by feature normalization, so the
  failure here is a real design mismatch (first-order distance vs the collapse-to-mean +
  cross-corpus-drift modes), not a strawman.

- **Relative Mahalanobis / RMD (Ren 2021, arXiv:2106.09022):** the near-OOD fix. Reported
  behavior here: also 100% LOCO FPR (per contract). Known limitation: it is still a
  location-based score; the subaru and ram corpora occupy disjoint feature regions whose
  separation dwarfs within-corpus radius, so the cross-corpus calibration breaks.

- **KNN-50 (Sun 2022, arXiv:2204.06507):** absolute nearest-neighbor distance in feature
  space. Reported headline here: TIES E6 at AUROC 1.000 at alpha=1.0 (so NO "beats" claim),
  and per the ablation is insensitive to k (AUROC 1.000 for k in {5,10,20,50,100}). Known
  limitation: 100% LOCO FPR (absolute-position score does not transfer across corpora). This
  is the single most important competitor row: the contribution is transfer/calibration, not
  separation, and this row is the one a reviewer will probe.

- **MSP (Hendrycks 2017, arXiv:1610.02136) and Energy (Liu 2020, arXiv:2010.03759):** the
  output-side competitors. Reported behavior here: the output/uncertainty channel is the one
  E3 shows is silent (1.20x-1.84x, 0/220 over real p95), so these are shown insufficient by
  construction. Known limitation (general): both are output-confidence scores and cannot see
  a collapse that keeps outputs in-range while freezing the internal state.

- **PCA-Mahalanobis (variant run in this paper):** per contract, reaches 11.91% LOCO FPR,
  still far above 1%. Not an external paper, but the relevant competing-variant datapoint:
  even a dimensionality-reduced location score does not transfer to the 1% target.

External method-class comparators worth citing as competitors-by-lineage (already fetched):
- **Keser 2025 density score (arXiv:2501.08083):** the AV-native instantiation of the
  location-based class; its density-as-ID-score is the same family that fails LOCO here.
- **Guo and Su 2026 QCD/MMD monitor (arXiv:2603.14603):** the strongest recent
  latent-state OOD monitor, but on a standalone predictor with provable guarantees; the
  competing approach to "monitor a latent/recurrent state," differentiated by model class and
  guarantee type.

openpilot/supercombo prior art (anchor + motivation):

- **Chen et al., "Level 2 Autonomous Driving on a Single Device: Diving into the Devils of
  Openpilot," 2022.** FETCHED QUOTE: "With curiosity in mind, we deep-dive into Openpilot and
  conclude that its key to success is the end-to-end system design instead of a conventional
  modular framework." Source: arXiv:2206.08176 (authors: Li Chen, Tutian Tang, Zhitian Cai,
  Yang Li, Penghao Wu, Hongyang Li, Jianping Shi, Junchi Yan, Yu Qiao; OpenDriveLab
  Openpilot-Deepdive). Why it matters: the reference academic teardown of supercombo
  (input/output/architecture). This paper extends it from a static input/output description
  to a runtime distribution-shift teardown. THE anchor citation for the parity claim.

- **commaai/openpilot issue #20704, "Large Shadow phantom braking."** FETCHED QUOTE (title):
  "Large Shadow phantom braking"; body: "Tall vehicles when casting a shadow into the
  adjacent lane cause openpilot to mis-identify the shadows as vehicles, and abruptly brake
  even with no actual vehicle in front of you in your lane." Source:
  https://github.com/commaai/openpilot/issues/20704. Why it matters: primary user-reported
  evidence that phantom braking under distribution shift is a known failure mode of the
  shipped model. CITE ONLY AS MOTIVATION (contract lines 88-89, 118-119); no causal link to
  E6's collapse.

---

## Cluster E: Venue expectations

- **SafeAI @ UAI 2026 workshop (PRIMARY venue).** Source:
  https://safe-ai-workshop.github.io/uai-2026/ (all quotes FETCHED this run).
  - Page limit: FETCHED QUOTE: "Papers should be no longer than 4 pages, excluding
    references." Supplementary allowed in the same PDF with no page limit. Category-B
    (already-accepted/submitted-elsewhere) papers keep their original venue's format/limit.
  - Format: FETCHED QUOTE: "Original submissions must comply with the UAI style requirements
    and use the adjusted template SafeAI format."
  - Review model: FETCHED QUOTE: "Single-blind peer review by the program committee."
  - Archival status: FETCHED QUOTE: "There will be no proceedings, so authors are free to
    submit their work elsewhere." (Non-archival; arXiv preprint + later resubmission is
    fully compatible.)
  - Topics: FETCHED QUOTE list includes "Uncertainty, robustness, and control" with
    "Robustness to distribution shift," and "Interpretability, auditing, and evaluation"
    with "Methods to audit, measure, monitor, and evaluate agentic AI systems." A
    distribution-shift teardown + runtime monitor is squarely in scope.
  - Dates: FETCHED QUOTES: "Paper submission: May 28, 2026"; "Author notification: Jun 24,
    2026"; "Camera-ready deadline: Jul 9, 2026"; "Workshop date: Aug 21, 2026."
  - Negative/position/WIP results: NOT explicitly addressed on the page. The CFP solicits
    "original research, theoretical results, and applied work." ASSUMPTION: a bounded
    negative result with a working monitor is within "applied work," and non-archival 4-page
    workshops are the conventional home for exactly this kind of single-model negative
    finding; but the page does not state a negative-results track, so do not assume a
    negative-results-friendly review without checking the PC.
  - **DECISION-RELEVANT FLAG:** the submission deadline (May 28, 2026) is 2 days BEFORE
    today's date (2026-05-30), and the FETCHED page shows NO extension or second round. If
    this is accurate and final, the SafeAI@UAI 2026 primary-submission window has CLOSED.
    The orchestrator must verify the live deadline/extension status before treating SafeAI as
    the immediate target; the contract's "arXiv preprint (CoRL/RSS-style) as the immediate
    deliverable" path is unaffected and remains the safe primary deliverable. This is the
    single most consequential venue finding in this brief.

- **arXiv preprint norms for a safety/teardown negative result (the contract's immediate
  deliverable).** No single canonical "rulebook" URL to quote; this is documented practice,
  so marked ASSUMPTION where not directly sourced:
  - ASSUMPTION: the threshold-free + bootstrap-CI + leave-one-corpus-out reporting the
    project already adopted (per memory note "Distribution-shift teardown methodology") is
    the reviewer-resistant norm for a negative OOD result; AUROC with bootstrap CIs and LOCO
    FPR is what the OOD literature (OpenOOD-style AUROC/FPR95) expects, and the framing memo
    already reports AUROC 0.996 [0.992, 1.000], which matches.
  - ASSUMPTION: a teardown paper is expected to ship a parity/trust artifact before any
    negative claim; this paper has it (parity within +/-0.5 m/s^2 on 100% of 1159 frames),
    which is the correct order (trust the harness, then claim the collapse).
  - VERIFIED norm from the venue itself: non-archival workshop means an arXiv preprint now +
    SafeAI (or successor venue) later is allowed ("authors are free to submit their work
    elsewhere," fetched above).

---

## State of the field (synthesis)

The OOD-detection toolkit is SETTLED at the method level: MSP (2017), Mahalanobis (2018),
RMD (2021), Energy (2020), KNN (2022), and ViM (2022) are the fixed canonical baselines,
codified in OpenOOD (NeurIPS 2022), and 2025 work (Mahalanobis++, the geometry-of-Mahalanobis
line) confirms the feature-Gaussian family is still live rather than deprecated. What is
CONTESTED, and where the 2024-2026 frontier sits, is OOD/silent-failure detection for
DRIVING: the active recent work targets either a frozen perception/foundation-model encoder's
feature density (Keser 2025) or a standalone trajectory predictor's latent dynamics with
provable QCD/MMD guarantees (Guo and Su 2025/2026), plus the SelfOracle/UQ misbehaviour-
prediction line (Stocco/Grewal, through 2024). Nobody in the fetched neighbor set monitors
the RECURRENT STATE of a SHIPPED, parity-verified end-to-end production driver, and nobody
reports that the model's OWN uncertainty channel stays silent during a collapse. The
SafeAI/AV-safety-assurance community (Hodge et al. 2025) is actively framing OOD as
safety-case evidence, which is exactly the niche this paper speaks to. The contested, open
question this paper enters: when a shipped end-to-end model fails silently on shifted input,
do output-side and location-based feature scores suffice (the field's default assumption), or
is a location-invariant second-order recurrent-state statistic needed? The paper's answer is
bounded and negative-plus-constructive: for THIS one model, output-side and location-based
scores do not transfer, and a cheap spread monitor does, with no claim that this generalizes.

---

## Open gaps this paper could exploit (each anchored to a source that shows the gap is open)

- **Gap 1: existing AV input monitors watch the perception encoder, not the recurrent state
  of a shipped end-to-end driver.** Anchored to Keser et al. 2025 (arXiv:2501.08083), whose
  fetched method is density on a FROZEN perception/foundation-model encoder ("use its density
  at new points as in-distribution (ID) score"). The recurrent-state-of-a-shipped-E2E-driver
  substrate is demonstrably not what the closest AV-native monitor targets. E6 occupies it.
  Bounded: N=1 model, no generalization claim.

- **Gap 2: recent latent-state OOD monitors are built and validated on STANDALONE trajectory
  predictors with theoretical guarantees, not on a deployed production model with an
  empirically parity-verified harness.** Anchored to Guo and Su 2026 (arXiv:2603.14603),
  fetched as QCD on a trajectory predictor with "provable guarantees on delay and false
  alarms," and the 2025 sibling (arXiv:2509.13577) on the trajectory-prediction stage. The
  shipped-model-with-real-parity evaluation axis is open; this paper fills it empirically
  (and explicitly does NOT claim guarantees, staying inside the contract).

- **Gap 3: the assumption that a shipped model's own uncertainty/output channel signals
  distribution shift is widely relied on but, on this model, false.** Anchored to the UQ
  misbehaviour-prediction line (Grewal/Tonella/Stocco 2024, arXiv:2404.18573), which evaluates
  Bayesian UQ "for the anticipatory testing of safety-critical misbehaviours," i.e. assumes
  the confidence signal is informative. E3 (0/220 OOD frames over real p95) is direct evidence
  this assumption fails for supercombo v0.9.7. The gap is "does the output-side signal
  suffice for THIS shipped model," and the answer is no. Bounded to N=1.

- **Gap 4: corruption-robustness benchmarks (ImageNet-C / Cityscapes-C) bound what counts as
  the same OOD failure, and that bound is rarely drawn for sim-rendered input.** Anchored to
  Hendrycks and Dietterich 2019 (arXiv:1903.12261) and Michaelis et al. 2019 (arXiv:1907.07484)
  as the standard corruption yardsticks. The open move this paper makes is to USE that
  yardstick to BOUND its own claim (E7: no ImageNet-C corruption reproduces the sim collapse),
  turning the standard robustness benchmark into the device that keeps the contribution
  honest (collapse-specific, sim-specific), which the contract requires.

- **Gap 5 (parked, flagged as the most reviewer-resistant follow-up, NOT this paper):** a
  real adverse-weather OOD axis that induces a non-CARLA output collapse. Anchored to the
  comma2k19 dataset (arXiv:1812.05752), whose fetched description (California highway,
  daytime, dry) confirms there is no public clean rain/night/glare comma corpus, so the gap is
  genuinely open and curation-gated, not merely unattempted. Correctly out of scope per
  contract; recorded so the drafter frames it as known future work, not an oversight.

---

## Citation corrections (problems found in docs/related_work.md and docs/paper_plan.md)

These are errors in the EXISTING literature docs that this brief is correcting. The
drafter/litmapper must fix these; they are exactly the kind of thing a reviewer who knows
the field will catch.

1. **WRONG AUTHOR AND VENUE: the openpilot falsification paper.** `related_work.md` line 72
   and `paper_plan.md` line 58 cite "Geretti et al., Finding Property Violations through
   Network Falsification: Lessons from OpenPilot, GPCE/SPLASH 2022." The ACTUAL paper is
   **von Stein and Elbaum, "Finding Property Violations through Network Falsification:
   Challenges, Adaptations and Lessons Learned from OpenPilot," ASE 2022 (Industry
   Showcase)**, DOI 10.1145/3551349.3559500. Source (FETCHED via search, ACM + conf page):
   https://dl.acm.org/doi/10.1145/3551349.3559500 and
   https://conf.researchr.org/details/ase-2022/ase-2022-industry-showcase/7/ . "Geretti et
   al." and "GPCE/SPLASH" appear to be a fabricated author/venue; this MUST be replaced.

2. **UNDER-SPECIFIED, MULTIPLE VERSIONS: Henriksson OOD-on-AV.** `related_work.md` line 25 /
   `paper_plan.md` line 36 cite "Henriksson et al., Performance Analysis of OOD Detection on
   Various Trained Neural Networks, RefSQ 2023 / IEEE." There are at least three distinct
   real Henriksson papers (arXiv:2103.15580; arXiv:2204.12378; and a better-fit 2024 AV-
   datasets paper arXiv:2401.17013 "Evaluation of Out-of-Distribution Detection Performance on
   Autonomous Driving Datasets"). The "RefSQ 2023" attribution is not confirmed by my search.
   ACTION: pin ONE specific Henriksson paper to a specific id before drafting; arXiv:2401.17013
   is the most on-point for AV datasets. paper_plan.md line 132 already flags this; treat it
   as a hard blocker, not a soft note.

3. **WRONG TITLE (minor): Sastry and Oore.** `related_work.md` line 48 calls it "Detecting
   Out-of-Distribution Examples with Gram Matrices, ICML 2020, arXiv:1912.12510." The arXiv
   1912.12510 title is "Detecting Out-of-Distribution Examples with In-distribution Examples
   and Gram Matrices"; the ICML 2020 proceedings title (sastry20a) is "Detecting
   Out-of-Distribution Examples with Gram Matrices." The ICML 2020 venue IS correct (resolves
   the [UNVERIFIED] flag). Use the proceedings title + arXiv id together.

4. **MISSING arXiv id and possible wrong id: ImageNet-C.** `paper_plan.md` line 79 lists
   ImageNet-C as "ICLR 2019" with no id. The ICLR 2019 version is arXiv:1903.12261 (FETCHED,
   title "Benchmarking Neural Network Robustness to Common Corruptions and Perturbations,"
   Hendrycks and Dietterich). Pin 1903.12261, not the older 2018 preprint id.

5. **TITLE FORMATTING (minor): Michaelis Cityscapes-C.** Both docs say "Object Detection when
   Winter is Coming"; the full fetched title is "Benchmarking Robustness in Object Detection:
   Autonomous Driving when Winter is Coming" (arXiv:1907.07484). Use the full title.

6. **STILL [UNVERIFIED], not confirmed this run (treat as leads, not evidence):** the
   following `related_work.md` entries were NOT independently fetched this run and remain
   unverified: Liang/Li/Srikant ODIN (arXiv:1706.02690, cite-for-lineage only per plan);
   Lakshminarayanan deep ensembles (arXiv:1612.01474, limitations-only); McAllister "Concrete
   Problems for AV Safety" IJCAI 2017; Bogdoll "Anomaly Detection in Autonomous Driving: A
   Survey" (arXiv:2204.07974); Codevilla "Exploring the Limitations of Behavior Cloning"
   (arXiv:1904.08980); Dosovitskiy CARLA (arXiv:1711.03938); Norden "Efficient Black-box
   Assessment" (arXiv:1912.03618); Pei DeepXplore; Tian DeepTest; Zhang DeepRoad
   (arXiv:1802.02295); Stocco MarMot (arXiv:2310.07414); the "Parallel Activations Drift
   Detector" (arXiv:2404.07776); "Topological Uncertainty" (arXiv:2105.04404); the
   "Dynamic-Aware ... Multi-Mode OOD" (now CONFIRMED above as Guo and Su 2509.13577, so this
   one is upgraded from unverified to verified); "Detecting What Matters: OOD 3D Object
   Detection" (arXiv:2506.23426); "Revisiting Adversarial Perception Attacks ... on AV
   Systems" (arXiv:2505.11532); OpenOOD-VLM. These are plausible leads; the citation-verifier
   stage must fetch each before it enters the draft as fact. Filos et al. ICML 2020
   (arXiv:2006.14911), SelfOracle/Stocco ICSE 2020, and the Stocco/Grewal 2024 UQ paper ARE
   confirmed this run and can be moved off [UNVERIFIED].

VERIFIED-OFF-FLAG this run (were [UNVERIFIED] in related_work.md, now confirmed with fetched
quotes above): Filos et al. "Can Autonomous Vehicles Identify, Recover From, and Adapt to
Distribution Shifts?" ICML 2020 (arXiv:2006.14911); Stocco et al. SelfOracle ICSE 2020;
Sastry and Oore Gram matrices ICML 2020 (arXiv:1912.12510). And these never-flagged anchors
are independently re-confirmed: Chen 2022 (2206.08176), Keser 2025 (2501.08083), Guo and Su
2026 (2603.14603), all six canonical baselines, OpenOOD, ImageNet-C, Cityscapes-C, comma2k19,
Mahalanobis++ (2505.18032), Hodge 2025 (2510.21254), NECO (2310.06823), issue #20704.

---

## Honesty summary (end of turn)

What I VERIFIED this run, each with a fetched quote + URL/arXiv id in the body above:
- All 6 canonical baselines (MSP 1610.02136, Lee/Mahalanobis 1807.03888, RMD 2106.09022,
  KNN 2204.06507, Energy 2010.03759, ViM 2203.10807) plus Cheng neuron monitoring
  (1809.06573) and Sastry/Oore Gram matrices (1912.12510, ICML 2020).
- Both closest recent neighbors: Keser 2025 (2501.08083) and Guo and Su 2026 (2603.14603),
  plus Guo and Su 2025 (2509.13577), Grewal/Stocco UQ 2024 (2404.18573), Mahalanobis++
  (2505.18032), Hodge OOD-assurance 2025 (2510.21254), NECO ICLR 2024 (2310.06823).
- Benchmarks: OpenOOD (2210.07242), ImageNet-C (1903.12261), Cityscapes-C (1907.07484),
  comma2k19 (1812.05752).
- openpilot anchor Chen 2022 (2206.08176) and motivation issue #20704; corrected the
  openpilot falsification paper to von Stein and Elbaum ASE 2022.
- SafeAI@UAI 2026 venue norms (4-page limit, UAI/SafeAI template, single-blind, non-
  archival, distribution-shift in-scope) and ALL four dates, directly from the workshop site.

What I did NOT verify (left as [UNVERIFIED] leads, listed in correction item 6): roughly a
dozen secondary related-work entries (ODIN, deep ensembles, McAllister, Bogdoll survey,
Codevilla, CARLA, Norden, DeepXplore/DeepTest/DeepRoad, MarMot, Parallel-Activations-Drift,
Topological Uncertainty, the 3D-object-detection and adversarial-AV 2025 papers, OpenOOD-VLM).
These are plausible but must be fetched by the citation-verifier before entering the draft.

THE SINGLE RISKIEST UNVERIFIED ASSUMPTION: that SafeAI @ UAI 2026 is still a viable PRIMARY
submission target. The workshop page I fetched lists the paper-submission deadline as May 28,
2026, which is TWO DAYS before today (2026-05-30), with no extension or second round shown. If
that deadline is final, the primary venue window has already closed and the strategy must fall
back to the arXiv preprint (which the contract already names as the immediate deliverable) plus
a successor venue. The orchestrator should confirm the live deadline/extension status before
committing to SafeAI; everything downstream of the arXiv preprint is unaffected. A secondary,
lower-stakes risk: the Guo and Su 2026 id (2603.14603) is a forward-numbered preprint with no
final venue, so its venue/DOI must be re-pinned at camera-ready.
