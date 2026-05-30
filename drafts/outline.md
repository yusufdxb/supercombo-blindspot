# Paper Outline

## Target venue and section structure

- Venue: arXiv preprint, CoRL/RSS-style (two-column, 8 pages body + references); written
  to compress later to a 4-6 page SafeAI-style workshop submission without restructuring.
- Chosen structure: system-teardown empirical paper (Introduction, Related Work, Threat
  Model and Framing, Method, Experiments and Results [E1-E7], Limitations, Conclusion,
  Reproducibility note). This is the correct structure because the contribution is a
  controlled empirical teardown with a negative finding plus a monitor demonstration, not
  a theoretical proposal; the section order mirrors the experimental dependency chain
  (parity must precede collapse, collapse must precede localization, localization must
  precede monitor, monitor must precede bounds).
- Why this structure fits the venue and contribution: arXiv/CoRL empirical papers that
  center a negative finding need an explicit threat-model section to establish why the
  finding is dangerous, and they need the experiments to unfold in logical order so each
  result is unambiguous before the next one leans on it; a vanilla IMRaD structure would
  bury the threat-model argument inside the introduction and make the claim boundary
  harder to audit.

---

## Argument spine

Section 1 (Introduction) establishes the problem: sim-based validation of shipped
driving models rests on an assumption the model either behaves identically on sim input
or signals loudly when it does not. Section 2 (Related Work) names the gap: no prior
work reports this failure mode on this model, the closest monitor class (location-based
feature scores) has a known architecture mismatch to a recurrent state, and the closest
mechanism-twin (EigenTrack) lives on LLMs/VLMs. Section 3 (Threat Model) names the
danger precisely: the three defenses a safety case would rely on (uncertainty heads,
output plausibility, temporal jitter) are exactly the signals that stay silent. Section
4 (Method) delivers the contribution's infrastructure: parity-verified harness, data,
metrics, and the E6 monitor design. Sections 5.1-5.7 (Experiments E1-E7) deliver the
evidence in logical order: parity (trust), collapse (phenomenon), silence (the gap),
cliff-shape (characterization), localization (mechanism, partial), monitor + baselines
(solution), corruption bounds (scope). Section 6 (Limitations) enforces the N=1/N=2
boundary and the collapse-specific scope. Section 7 (Conclusion) names what the
evidence does and does not support. An inline Reproducibility note closes the evidence
chain.

---

## Sections

### 1. Introduction

- JOB (one sentence): Establish the problem (silent failure under sim-rendered input),
  state the gap (no prior monitor targets the recurrent state of a shipped end-to-end
  driver under cross-corpus LOCO), and announce the four contributions.
- Claims it carries:
  - That phantom braking under distribution shift is a known, user-reported failure mode
    of the shipped model (commaai issues #20704 / #22212) -- motivation only, no causal
    claim.
  - The four-contribution list (parity harness; silent-failure demonstration E1/E2/E3;
    cliff + localization E4/E5; monitor + bounds E6/E7), stated as a preview, not yet
    evidenced.
- Paragraph plan:
  - P1 (hook): Sim-based validation of L2/autonomous stacks is standard practice, and
    its validity rests on the assumption that the shipped model behaves the same on
    rendered input as on real input -- or at least fails loudly when it does not.
  - P2 (the dangerous answer): We test that assumption on one deployed model (openpilot
    v0.9.7 supercombo) and find the model fails silently -- 8 of 10 output heads collapse,
    the recurrent state freezes, and the model's own uncertainty channel stays quiet --
    with zero out-of-distribution frames exceeding the real-driving 95th-percentile
    uncertainty threshold.
  - P3 (safety relevance and phantom-braking motivation): Name the user-reported
    phantom-braking issue as evidence that this failure mode surfaces in the field
    (motivation only; no causal claim). Explain the safety-case implication: a sim "pass"
    can be the model collapsed to a plausible default, and the downstream signals a
    safety monitor would trust are precisely the signals that stay silent.
  - P4 (gap in prior monitoring work): The closest monitors either watch a frozen
    perception encoder's feature density (Keser 2025) or a standalone trajectory
    predictor's latent dynamics (Guo/Su 2026); neither targets the recurrent state of a
    shipped end-to-end driver. EigenTrack (arXiv:2509.15735) uses a second-order
    covariance-spectrum statistic on LLMs/VLMs, not on a driving model and not under
    cross-corpus LOCO transfer. Standard location-based scores (Mahalanobis, RMD, KNN)
    hit 100% leave-one-corpus-out FPR here.
  - P5 (this paper): Name the four contributions explicitly, with the N=1 / N=2 / LOCO
    two-fold qualifiers already attached. The bounded takeaway (collapse-specific monitor,
    not a universal OOD detector) must appear here so reviewers carry it into the paper.
- Figures/tables placed here:
  - Figure 1 (hero.png): Four-panel overview -- output collapse, feature freeze,
    uncertainty silence, and monitor detection -- provides the full argument at a glance
    and grounds the four-contribution list; supports the "silent failure" claim summary
    and the E6 early-warning claim.

---

### 2. Related Work

- JOB (one sentence): Place the contribution at the intersection of two ancestral lines
  (location-based feature OOD and internal-activation runtime monitoring), name every
  neighbor a reviewer will invoke, and state exactly what each neighbor does NOT have
  that this paper does.
- Claims it carries: none (this section disposes of prior work; it does not assert new
  evidence).
- Paragraph plan:
  - P1 (OOD detection for AV perception -- Line A ancestors): Cite MSP (Hendrycks/Gimpel
    2017), Energy (Liu 2020), Mahalanobis (Lee 2018, arXiv:1807.03888), RMD (Ren 2021),
    KNN (Sun 2022, arXiv:2204.06507), ViM (Wang 2022), Mahalanobis++ (Mueller/Hein 2025,
    arXiv:2505.18032). Establish these as the location-based family that E3/E6 runs as
    baselines. The honest framing: KNN ties E6 on single-corpus AUROC; the failure is
    cross-corpus LOCO (100% FPR), not single-corpus separation. State ViM and Energy as
    not applicable to supercombo's regression heads (structural exclusion), not just
    omitted.
  - P2 (AV-native OOD and uncertainty monitoring): Keser 2025 (arXiv:2501.08083) --
    density-based score on a frozen encoder, one stage upstream; Guo/Su 2026
    (arXiv:2603.14603) -- MMD/QCD latent-dynamics on a standalone predictor with
    provable guarantees; Filos 2020 (arXiv:2006.14911) -- canonical AV-distribution-shift
    framing; Grewal/Tonella/Stocco 2024 (arXiv:2404.18573) and SelfOracle 2020 -- assume
    confidence signal is informative; E3 is the contrary finding for this model.
    One-sentence statement of the OOD-for-AV-datasets line (Henriksson, to be pinned to
    a single verified arXiv id before drafting). State the delta: none of these targets
    the recurrent state of a shipped end-to-end driver under cross-corpus LOCO.
  - P3 (internal-activation and second-order monitors -- Line B): Cheng 2018
    (arXiv:1809.06573) -- binarized neuron activation patterns on classifiers (ancestor);
    Sastry/Oore 2020 (arXiv:1912.12510, ICML 2020) -- Gram-matrix (first move to
    higher-order feature statistic for OOD, direct ancestor of E6's second-order choice).
    EigenTrack 2025 (arXiv:2509.15735, HARD citation -- must cite and disposition):
    covariance-spectrum of hidden activations plus a trained recurrent classifier on
    LLMs/VLMs, second-order and early-warning, but wrong substrate (not a driving model),
    different statistic (full eigenspectrum/RMT vs single spread trace), and no cross-
    corpus LOCO evaluation. NECO (arXiv:2310.06823) -- classification-head property,
    inapplicable to multi-head regression, name and excuse. The drafter must NOT write
    "first to use a second-order hidden-activation statistic for OOD" -- EigenTrack
    pre-dates this work on that framing. The true, defensible claim: first to use a
    location-invariant second-order spread on the recurrent state of a SHIPPED end-to-end
    driving model, evaluated under cross-corpus LOCO transfer.
  - P4 (openpilot / supercombo prior work): Chen 2022 (arXiv:2206.08176, Openpilot-
    Deepdive) -- static architecture teardown and reimplementation, anchor this paper
    extends to a runtime distribution-shift teardown. Von Stein and Elbaum, ASE 2022
    (DOI 10.1145/3551349.3559500) -- falsification via adversarial input generation
    (WRONG prior citation in skeleton_source.md was "Geretti et al., GPCE/SPLASH 2022";
    the correct author/venue is confirmed in literature_map.md and must be used here).
    Adversarial supercombo study (arXiv:2505.11532) -- adversarial perturbations with
    input-level defenses, not a silent-collapse / recurrent-state-monitor study.
  - P5 (sim testing of driving DNNs and corruption robustness): DeepTest (ICSE 2018),
    DeepRoad (ASE 2018, arXiv:1802.02295), MarMot (arXiv:2310.07414) -- assume the sim
    is in-distribution; this paper shows the sim itself can be OOD to the model.
    ImageNet-C (Hendrycks/Dietterich, arXiv:1903.12261, ICLR 2019) -- the corruption
    axis used in E7; Cityscapes-C (Michaelis, arXiv:1907.07484). OpenOOD (arXiv:2210.07242,
    NeurIPS 2022) -- taxonomy vocabulary anchor. CARLA (Dosovitskiy 2017,
    arXiv:1711.03938) -- must cite as the OOD axis source (HARD citation gap per
    literature_map.md).
- Figures/tables placed here:
  - Table RW (competitor contrast table): a compact version of the literature_map.md
    contrast table showing the five axes (substrate, score type, calibration, target
    model, guarantee) for Keser 2025, Guo/Su 2026, Cheng 2018, Lee 2018, Sun 2022,
    EigenTrack 2025, and This Work; supports the bounded-novelty claim and lets reviewers
    audit the delta without re-reading five papers. Status: to-produce (compressing the
    existing literature_map.md table into a paper-ready format; all cell values are
    already verified).

---

### 3. Threat Model

- JOB (one sentence): Establish precisely why silent collapse is dangerous by showing
  that the three defenses a safety case would rely on are exactly the defenses that fail,
  and position E6 as a cheap complementary layer.
- Claims it carries:
  - That predictive-uncertainty heads are insufficient: uncertainty ratios 1.20x-1.84x,
    0/220 OOD frames above real p95 (E3, report/teardown_results.md). VERIFIED.
  - That output-plausibility and temporal-jitter checks are also insufficient
    (structural argument: collapsed outputs look benign; frozen state has lower variance).
    ASSUMPTION: this is a structural argument, not a new experiment; it is supportable
    from the E1 activity ratios (plan 0.0057x, accel_t0 0.0040x) already in the ledger.
  - That E6 is a complement (not replacement): one O(d) statistic per forward pass, no
    retraining, no architecture change, calibrated against real-driving FPR.
- Paragraph plan:
  - P1 (threat statement): Define the threat: a shipped driving model deployed in a
    visually shifted context (rendered sim, novel geography, sensor degradation) where
    three things happen simultaneously and silently -- output collapse, recurrent freeze,
    and uncertainty silence.
  - P2 (why uncertainty-head monitoring fails): E3 numbers (0/220 OOD frames above real
    p95; uncertainty ratios 1.20x-1.84x). The uncertainty head a safety monitor trusts
    is exactly the head that stays silent. (Contrasted against the Grewal/SelfOracle
    line that assumes confidence is informative.)
  - P3 (why output-plausibility and jitter monitors fail): Collapsed outputs look like a
    benign stationary scene (plan 0.6% retained, accel_t0 0.4% retained), so plausibility
    passes; frozen outputs have lower variance than active outputs, so a jitter monitor
    reads the freeze as increased stability.
  - P4 (why same-architecture ensembles and input-quality checks fail): E5 localizes the
    collapse downstream of the encoder, so an ensemble of the same architecture shares
    the collapse path; CARLA-clean renders are sharper than real footage, so image-quality
    checks rate sim as good. (Structural argument; not a new experiment.)
  - P5 (E6 as the complementary signal): The model's own recurrent features carry the OOD
    signal the output heads do not surface. E6 is a complement layer: cheap, compatible
    with the shipped model as-is, and calibrated against real-driving FPR. State the
    honesty bound here: it is collapse-specific, offline-only, and N=2-calibrated.
- Figures/tables placed here: none (argument is structural; figures arrive in the
  experiments section where the evidence is shown).

---

### 4. Method

- JOB (one sentence): Describe the parity-verified harness, the data, the metrics, the
  E6 monitor design, and the baseline set in sufficient detail for the teardown results
  to be trusted and reproduced.
- Claims it carries:
  - Parity-exact reimplementation: 100% of 1159 frames within +/-0.5 m/s^2, median abs
    delta 0.04 m/s^2 (report/parity_results.md). VERIFIED.
  - Two non-obvious correctness points: recurrent state must roll (shift-and-append, not
    zero-reset per frame); YUV input is unnormalized uint8 0..255 (contract, parity_results).
  - Data: N_ID = 638 frames (subaru 220 + ram 220 analysis frames each, 100 warmup
    discarded); N_OOD = 319 CARLA frames (analysis); 29-point alpha-blend sweep for E4.
  - Metrics: activity (temporal std sum), feature spread (trace of recurrent-state
    covariance), threshold-free OOD metrics (AUROC/AUPR/FPR@95TPR with stratified
    bootstrap, n=1000 seed=42), LOCO calibration protocol.
  - E6 design: rolling temporal spread of the 512-D recurrent state, threshold at 1st
    percentile of real-driving rolling-spread distribution, calibrated LOCO.
  - Baselines: Mahalanobis, RMD, KNN-50, PCA-Mahalanobis on the same 512-D feature.
    MSP/Energy/ViM structurally inapplicable (named and excused).
    RMD background GMM note (two-component because single ID class degenerates).
- Paragraph plan:
  - P1 (parity harness overview): Reconstruct openpilot v0.9.7 supercombo inference from
    the released ONNX and comma reference files; the parity test is the load-bearing
    trust claim.
  - P2 (recurrent state threading): Shift-and-append after each inference; zero-init
    only on frame 1; per-frame zero reset produces a multi-second transient. This detail
    is reviewer-convincing evidence that the reimplementation is correct.
  - P3 (YUV normalization): loadyuv.cl does convert_float8() with no scaling; the model
    consumes uint8 Y/U/V in 0..255. State the consequence of getting this wrong.
  - P4 (parity result): 100% of 1159 frames within +/-0.5 m/s^2 of comma's reference
    output (median abs delta 0.04 m/s^2). This is the load-bearing harness-trust claim
    for the negative result.
  - P5 (data): Real ID corpora (subaru, ram), CARLA OOD frames, alpha-blend interpolation
    for E4. State N explicitly everywhere a percentage will appear.
  - P6 (metrics and calibration protocol): Activity, feature spread, threshold-free OOD
    metrics with bootstrap CIs, and the LOCO calibration protocol. State the N=2 / two-
    fold qualifier for any FPR derived from LOCO.
  - P7 (E6 monitor): One O(d) statistic per forward pass; threshold calibration; the
    location-invariant property that distinguishes it from the baseline family.
  - P8 (baselines and structural exclusions): Mahalanobis / RMD / KNN-50 / PCA-
    Mahalanobis on the 512-D feature. MSP, Energy, ViM are structurally inapplicable
    (no softmax/logits/classifier weight matrix on supercombo's regression heads); state
    the exclusion explicitly so reviewers do not read it as an omission.
- Figures/tables placed here: none (all figures are in the experiments section; the
  method section's load-bearing content is the parity number and the design descriptions,
  which are stated in text).

---

### 5. Experiments and Results

#### 5.1 E1: Output collapse

- JOB (one sentence): Show that 8 of 10 output heads collapse to under 1% of real-driving
  temporal activity on CARLA-rendered clean input, with specific activity ratios.
- Claims it carries:
  - 8 of 10 heads collapse (CARLA/real activity ratio < 0.01): desired_curv 0.0018,
    accel_t0 0.0040, lead 0.0042, desire_state 0.0049, lane_lines 0.0054, plan 0.0057,
    lead_prob 0.0058, road_edges 0.0076. pose (0.1788) and meta (0.7181) survive.
    (report/teardown_results.md). VERIFIED.
- Paragraph plan:
  - P1: Run the parity-verified model on CARLA-rendered clean roads; present the
    activity-ratio table; state which heads collapse and which survive. 8 of 10 collapse
    to under 1% of real temporal activity.
- Figures/tables placed here:
  - Figure E1 (e1_head_collapse.png): Bar chart of CARLA/real activity ratio per output
    head, showing 8 collapsed and 2 surviving heads; supports the "8 of 10 heads collapse"
    claim. Status: exists.

---

#### 5.2 E2: Recurrent-feature freeze

- JOB (one sentence): Show that the 512-D recurrent state freezes to a near-constant
  vector under CARLA input, quantified by covariance trace ratio and separability.
- Claims it carries:
  - CARLA feature spread is 1e-5 of real spread (trace of hidden_state covariance).
  - Real-vs-CARLA separability 87.9%, d' = 2.19 along the centroid-difference direction.
    (report/teardown_results.md). VERIFIED.
- Paragraph plan:
  - P1: Measure the rolling covariance trace of the 512-D hidden_state on real vs CARLA
    frames; the trace collapses to 1e-5 of real, and the real/CARLA distributions separate
    at 87.9%. The recurrent state freezes to a near-constant point in feature space.
- Figures/tables placed here:
  - Figure E2 (e2_feature_ood.png): Scatter or distribution plot of the 512-D feature
    space (projected), showing real vs CARLA separation and the CARLA freeze point;
    supports the recurrent-feature-freeze claim. Status: exists.

---

#### 5.3 E3: Uncertainty silence (the centerpiece)

- JOB (one sentence): Show that the model's own predictive-uncertainty heads fail to
  register the collapse, with 0 of 220 OOD frames exceeding the real-driving p95, making
  the failure silent by the model's own signals.
- Claims it carries:
  - Uncertainty ratios 1.20x-1.84x (plan 1.35x, lead 1.20x, desired_curv 1.84x).
  - 0/220 CARLA frames exceed the real-driving p95 of any monitored head.
    (report/teardown_results.md, contract). VERIFIED.
  - Framing risk note from framing_memo.md applies: state E3 as an empirical finding on
    this model ("the uncertainty channel stays quiet for supercombo v0.9.7 under CARLA
    input"), never as a causal explanation of field phantom-braking incidents.
- Paragraph plan:
  - P1: Present the three-row E3 table (output retained, uncertainty ratio, OOD frames
    above real p95). 0 of 220 OOD frames cross the real p95 on plan, lead, or
    desired_curv; uncertainty rises only 1.20x-1.84x. Contrast with the SelfOracle/Grewal
    line that assumes confidence is informative.
  - P2: State the failure-mode implication: a real-calibrated uncertainty threshold never
    fires; nothing the model emits flags the collapse. This is the safety-relevant gap --
    the downstream safety case that relies on model uncertainty as a monitor cannot catch
    this mode.
- Figures/tables placed here:
  - Figure E3 (e3_confidence.png): Time-series or box plot showing uncertainty distributions
    for real vs CARLA frames alongside the real-driving p95 threshold, making the silence
    visually unambiguous; supports the "0/220 OOD frames above real p95" claim.
    Status: exists.

---

#### 5.4 E4: Cliff characterization and segment dependence

- JOB (one sentence): Characterize the collapse as a hard cliff on the real-to-sim blend
  axis for the Subaru source, and show the cliff is segment-dependent (gradient on the RAM
  source), so cliff headroom cannot be generalized.
- Claims it carries:
  - Cliff transition width 0.015 on Subaru: output activity falls from 0.9x to 0.1x of
    real over alpha 0.784 to 0.799; feature spread crashes from 0.25 to 0.00 by alpha 0.78.
  - Output activity first balloons to 6.32x at alpha=0.425 (ghosted-input thrash), then
    collapses.
  - Predictive uncertainty never spikes through the transition.
  - On RAM source: gradient with width 0.274 (E4-RAM), no early-warning headroom.
    (report/teardown_results.md, contract). VERIFIED.
  - Cliff headroom cannot be assumed to generalize across segments (contract claim boundary).
- Paragraph plan:
  - P1: The alpha-blend sweep from real (alpha=0) toward CARLA (alpha=1) shows a two-phase
    response: an initial thrash (activity peak 6.32x at alpha=0.425 from ghosted-input
    interference) followed by a hard cliff (transition width 0.015) at alpha 0.784-0.799.
    Feature spread crashes from 0.25 to 0.00 by alpha=0.78. Uncertainty never spikes.
  - P2: The cliff is segment-dependent. On the RAM source the same sweep produces a
    gradient (width 0.274) with no sharp cliff and no early-warning headroom. Cliff
    headroom cannot be assumed to generalize across segments; state as PARTIAL
    characterization.
- Figures/tables placed here:
  - Figure E4 (e4_interpolation.png + e4_ram_interpolation.png as two panels): Activity
    ratio and feature spread vs alpha for Subaru (showing the cliff at 0.784-0.799) and
    RAM (showing the gradient), side by side; supports both the "hard cliff" and the
    "segment-dependent" claims. Status: both figures exist.

---

#### 5.5 E5: Localization downstream of the vision encoder

- JOB (one sentence): Show that the collapse is not in the vision encoder (every stage
  stays at or above real activity) and enters at the recurrent summarizer and action-block
  feedback path, with the partial VAE-mu/sigma ambiguity stated.
- Claims it carries:
  - Every encoder stage stays at or above real activity (stem 1.43x, stage3 2.06x, head
    2.14x at alpha=1; minimum 0.96); the collapse is downstream of the encoder. VERIFIED.
  - Cliff entry pinned to summarizer_div (VAE-mu, cliff alpha 0.900) and action_block_body
    (cliff alpha 0.500, driven by prev_desired_curv feedback loop).
  - vision_post and hydra_trunk are passive relays (no cliff).
  - Localization is PARTIAL: VAE-mu/sigma ambiguity remains.
    (report/teardown_results.md, contract). VERIFIED.
- Paragraph plan:
  - P1: Present the layer-by-layer activity ratios at alpha=1; every encoder stage is at
    or above real (minimum 0.96). The collapse is therefore not the encoder failing. State
    the structural contrast explicitly: "the failure is in the summarizer and action block,
    not in perception."
  - P2: Submodule probing places the cliff entry at the recurrent summarizer VAE-mu
    bottleneck (cliff alpha 0.900) with earlier amplification in the action-block feedback
    path (cliff alpha 0.500, driven by the prev_desired_curv recurrent loop). The
    transformer and reduce-sum stage are passive relays. VAE-mu/sigma ambiguity remains
    -- state as partial localization.
- Figures/tables placed here:
  - Figure E5a (e5_layer_localization.png): Per-stage activity ratio vs alpha, showing all
    encoder stages staying at or above real while the model output collapses; supports the
    "downstream of the encoder" localization claim. Status: exists.
  - Figure E5b (e5_submodule_localization.png): Cliff-alpha per submodule, showing
    summarizer_div and action_block_body as cliff-entry points and vision_post/hydra_trunk
    as passive relays; supports the partial localization claim. Status: exists.

---

#### 5.6 E6: Monitor detection and baseline comparison

- JOB (one sentence): Show that the rolling-spread monitor detects the collapse with
  AUROC 0.996 and fires 0.23 blend-units before the output cliff, and that the location-
  based baselines fail to transfer across corpora (100% LOCO FPR each) -- where the
  distinction is cross-corpus calibration, not single-corpus AUROC (KNN ties E6 at
  AUROC 1.000).
- Claims it carries:
  - E6: AUROC 0.996 [0.992, 1.000], AUPR 0.995 [0.990, 1.000], FPR@95TPR 0.000,
    LOCO mean FPR 1.03% (max 2.07%), fires (>50% frames) at alpha=0.550 (0.23
    blend-units before cliff at 0.784). (report/e6_results.md, report/metrics_results.md).
    VERIFIED.
  - KNN-50: AUROC 1.000, 100% LOCO FPR. The contrast is TRANSFER/CALIBRATION, not raw
    separation (KNN ties E6; contract exclusion line 108-111 enforced here).
  - Mahalanobis: AUROC 0.159 (below chance); collapses to the ID mean, distance-from-mean
    cannot detect collapse-to-the-mean. VERIFIED.
  - RMD: AUROC 0.934, 100% LOCO FPR. VERIFIED.
  - PCA-Mahalanobis: AUROC 0.152, LOCO mean FPR 11.91%. VERIFIED.
  - All three applicable baselines: 100% LOCO FPR. (report/metrics_results.md). VERIFIED.
  - The mechanism behind the LOCO failure: subaru and ram corpora sit in disjoint 512-D
    feature regions at magnitudes that dwarf within-corpus radius; absolute-position scores
    calibrate on one corpus and flag the entire other corpus. (report/metrics_results.md).
    VERIFIED.
  - N=2 / LOCO two-fold qualifier MUST be attached to every FPR number in this section.
    A third corpus is needed before quoting a production FPR.
  - Novelty positioning: E6 is the first zero-retraining, location-invariant second-order
    (rolling-spread) monitor on the recurrent state of a parity-verified SHIPPED end-to-end
    driving model, evaluated under cross-corpus LOCO FPR where the named location-based
    scores fail to transfer. EigenTrack (arXiv:2509.15735) uses the same second-order
    family on LLMs/VLMs with a full eigenspectrum/RMT trained classifier -- wrong substrate,
    different statistic, different evaluation axis; not a pre-emption of this claim.
- Paragraph plan:
  - P1: Present the threshold-free comparison table (E6, KNN-50, RMD, Mahalanobis,
    PCA-Mahalanobis) at alpha=1.0 with 95% bootstrap CIs. Note that KNN-50 ties E6 at
    AUROC 1.000; the distinguishing axis is cross-corpus transfer, not single-corpus
    separation.
  - P2: Explain why Mahalanobis scores BELOW chance (AUROC 0.159): the recurrent state
    collapses to the mean of the ID Gaussian, and distance-from-mean cannot detect
    collapse-to-the-mean. This is the mechanism-level explanation, not just a number.
  - P3: Present the LOCO comparison: all three location-based baselines hit 100% LOCO FPR;
    E6 holds 1.03% (max 2.07%) (N=2, two-fold estimate, not a production FPR). Explain
    the geometric reason: the two real corpora sit in disjoint feature regions whose
    inter-corpus separation dwarfs the within-corpus radius, so any absolute-position score
    calibrated on one corpus flags the entire other. E6 uses the second-order trace
    (location-invariant), so it survives the corpus shift.
  - P4: Present the alpha-sweep AUROC plot; E6 crosses 0.5 AUROC at alpha=0.550 and fires
    (>50% of frames flagged) at alpha=0.550 -- 0.23 blend-units before the E4 cliff at
    0.784. Quantify the early-warning gap.
  - P5: Position against EigenTrack (arXiv:2509.15735) and Keser 2025 (arXiv:2501.08083)
    explicitly, using the contract's bounded-novelty framing: different substrate, different
    statistic, different evaluation axis. State the framing: this is not "outperforms
    baselines" -- it is "the location-based baselines fail to transfer and a second-order
    monitor calibrates where they cannot."
- Figures/tables placed here:
  - Table E6 (main detector comparison): AUROC/AUPR/FPR@95TPR/LOCO FPR for all five
    detectors at alpha=1.0 with 95% CIs; supports the transfer/calibration claim.
    Status: exists (report/metrics_results.md; to-produce as a formatted paper table).
  - Figure E6a (e6_detector.png): E6 fire rate vs alpha, showing the detector firing before
    the output cliff; supports the "fires 0.23 blend-units before cliff" claim.
    Status: exists.
  - Figure E6b (auroc_vs_alpha.png): AUROC vs alpha for all five detectors, showing E6
    rising earlier than the collapse onset; supports the early-warning and AUROC trajectory
    claims. Status: exists.
  - Figure E6c (roc_curves.png + pr_curves.png): ROC and PR curves at alpha=1.0 for all
    five detectors; supports the threshold-free comparison. Status: both exist. These two
    figures may be combined into one two-panel figure or placed in an appendix if space is
    tight.

---

#### 5.7 E7: Corruption sweep (two-way bound)

- JOB (one sentence): Show that the silent collapse is sim-specific (no ImageNet-C
  corruption reproduces it -- max 1 of 10 heads vs 7 of 10 under CARLA) and E6 is
  collapse-specific (correctly quiet on most corruptions; its 4 high-AUROC cells track a
  feature-spread shift, not an output collapse), bounding both the failure mode and the
  monitor's scope.
- Claims it carries:
  - No ImageNet-C corruption reproduces the output collapse: 0 of 75 cells collapse
    (>=5 of 10 heads); max 1 of 10 heads on any cell vs 7 of 10 under CARLA.
    (report/e7_overlay_results.md). VERIFIED (validation gate: reproduces 7/10 CARLA
    collapse).
  - Zero false negatives: no collapse exists to miss, so E6's quiet response on fog,
    brightness, blur is correct.
  - E6 fires on 4 cells with NO output collapse (frost sev3 AUROC 0.958, frost sev5
    AUROC 1.000, gaussian_noise sev4 AUROC 0.861, impulse_noise sev5 AUROC 0.906);
    these track a recurrent-feature-spread shift, not the collapse mode.
  - E6 is near chance on most photometric/blur corruptions (fog 0.54, snow 0.55,
    brightness 0.60, zoom_blur 0.58, mean AUROC 0.52-0.74 across the named families).
    (report/e7_results.md). VERIFIED.
  - The location-based baselines (Mahalanobis, RMD) fire on many corruptions at calibrated
    thresholds but carry 100% LOCO FPR from E6; their high fire rates are uncalibrated.
    (report/e7_results.md). VERIFIED.
  - The silent collapse is CARLA / full-sim specific; real-frame corruptions do NOT
    reproduce it. (report/e7_overlay_results.md). VERIFIED.
  - Boundary enforcement: DO NOT write "E6 generalizes beyond CARLA as a collapse
    detector" -- no non-CARLA output collapse was induced, so there is no non-CARLA
    collapse for E6 to have generalized to (contract exclusion lines 99-105).
- Paragraph plan:
  - P1: Describe the E7 setup: 15 ImageNet-C corruptions at 5 severities applied to real
    Subaru frames (RGB, pre-YUV), supercombo re-run with correct recurrent state handling
    on each corrupted sequence, E6 and baselines evaluated. Validation gate: confirm the
    CARLA collapse (7/10 heads) is reproduced before reading the corruption results.
  - P2: The silent collapse is sim-specific. Present the E1 overlay result: 0 of 75
    corruption-severity cells produce output collapse (>=5/10 heads), maximum 1 of 10
    heads on any cell vs 7 of 10 under CARLA. The collapse is a property of full-sim
    rendering, not photometric or blur corruptions of real frames.
  - P3: E6 is collapse-specific, not a universal corruption detector. On most corruptions
    E6 AUROC sits near chance (mean per-corruption AUROC: jpeg 0.52, fog 0.55, snow 0.55,
    brightness 0.60, zoom_blur 0.58, blur families 0.68-0.71). E6 fires on 4 cells
    (frost sev3/sev5, gaussian_noise sev4, impulse_noise sev5), but those cells have zero
    output collapse -- the firings track a recurrent-feature-spread shift rather than the
    collapse mode. There are zero false negatives because there is no collapse to miss.
  - P4: Baseline behavior under corruptions: Mahalanobis and RMD fire on many corruptions
    at high rates, but recall from E6 both carry 100% LOCO FPR -- their fire rates are
    not calibrated to the 1% operating point E6 holds. KNN fires only on the heaviest
    noise/frost. None is calibrated to the operating point.
  - P5: Synthesis: the E7 result narrows the contribution precisely. The silent collapse
    is the dangerous mode (sim-specific); the corruption sweep shows it is not a general
    model-robustness failure, and the E1 overlay shows E6 is not a general corruption
    detector. The contribution is a targeted monitor for the specific, dangerous,
    sim-induced silent-collapse mode.
- Figures/tables placed here:
  - Figure E7a (e7_auroc_heatmap.png): AUROC heatmap (15 corruptions x 5 severities) for
    E6, visually showing near-chance AUROC on most cells and the 4 high-AUROC outliers;
    supports the "near chance on most" claim. Status: exists.
  - Figure E7b (e7_severity_sweep.png): E6 AUROC vs severity for representative
    corruptions, showing the monotonic trend (or lack thereof) per corruption family;
    supports the "bounded to a few severe cases" claim. Status: exists.
  - Figure E7c (e7_overlay.png): Cell-for-cell E1 collapse count vs E6 AUROC, making the
    decoupling of E6 firings from output collapse visually explicit; this is the key
    figure for bounding the "collapse-specific" claim. Status: exists.

---

### 6. Limitations

- JOB (one sentence): State the four scope boundaries that define where the evidence does
  and does not extend, so no reader can mistake the bounded finding for a general result.
- Claims it carries: none (this section prevents overclaiming; it does not assert new
  evidence).
- Paragraph plan:
  - P1 (N=1 model): supercombo v0.9.7 only; no other openpilot version, no other vendor,
    no research IL stack. The silent-collapse phenomenon and the monitor are demonstrated
    on this one model; no generalization is claimed. (Contract exclusion lines 77-78.)
  - P2 (N=2 corpora, LOCO as two-fold estimate): LOCO at N=2 is a two-fold estimate;
    variance is not reportable. A third corpus is needed before quoting a single
    production FPR. (Contract exclusion lines 80-81.)
  - P3 (E6 scope: collapse-specific, offline-only): The E7 overlay shows E6 is collapse-
    specific and near-chance on most real-frame corruptions. E6 is demonstrated offline
    on logged + rendered + corrupted frames only; no on-road or in-stack deployment was
    run. (Contract exclusion lines 82-87.) The one residual gap: no non-CARLA output
    collapse was induced, so E6 was never tested as a collapse detector on anything other
    than CARLA; real adverse-weather footage that induces collapse remains pending.
  - P4 (partial localization): The collapse is pinned to the summarizer VAE-mu bottleneck
    and the action-block feedback path by ruling out the encoder and probing 8 submodules;
    a VAE-mu/sigma ambiguity remains. The localization is partial, not fully mechanistic.
    (Contract exclusion lines 91-92.)
- Figures/tables placed here: none.

---

### 7. Conclusion

- JOB (one sentence): Crystallize the two-sentence result (shipped model fails silently;
  internal second-order monitor recovers the hidden signal) while explicitly enforcing the
  N=1 / N=2 / collapse-specific scope boundaries.
- Claims it carries: restatement of the headline numbers (already evidenced in sections
  5.1-5.7 and 6); no new claims.
- Paragraph plan:
  - P1 (the finding): A shipped L2 driving model shown CARLA-rendered input collapses to
    a plausible constant and does not raise its own uncertainty, so simulation "passes" can
    be false confidence. Name the model, attach the numbers.
  - P2 (the monitor): The signal the outputs hide is recoverable from the model's own
    recurrent features with a single O(d) statistic, no retraining, calibrated to ~1% real-
    driving FPR (N=2, two-fold), AUROC 0.996, firing 0.23 blend-units before the output
    cliff, where location-based feature scores fail to transfer across corpora.
  - P3 (the scope): Output-side monitoring alone is insufficient for the safety case of
    this shipped driving model; a second-order recurrent-state monitor is a cheap
    complement. This is a N=1 / collapse-specific / offline-only result; the evidence does
    not support generalization to other models, venues, or deployment contexts.
- Figures/tables placed here: none.

---

### 8. Reproducibility Note (inline, not a full section)

- JOB (one sentence): State the reproduce path so reviewers can confirm results without
  GPU or CARLA, and flag the one open P1 issue (E5/E7 caches not in the public repo).
- Claims it carries: that the reproduce path from committed caches works for E1-E4, E6,
  E7 (once E7 cache is either LFS-tracked or the 110 MB cache is re-generated); the
  E5 cache (3.9 GB) requires either LFS or a regeneration script.
- Paragraph plan:
  - P1: All result caches committed (report/*_collected.npz); analysis reruns from cache.
    Bootstrap params pinned (n=1000, seed=42). Requirements pinned. Note the E5 (3.9 GB)
    and E7 (110 MB) cache status (currently .gitignore'd; decision on LFS vs regeneration
    script pending -- this is a P1 open item from the publication-readiness audit).
- Figures/tables placed here: none.

---

## Figure/table manifest

| Asset | Section | What it must show | Claim it supports | Status |
|---|---|---|---|---|
| Figure 1: hero.png | 1. Introduction | Four-panel overview: output collapse (E1), feature freeze (E2), uncertainty silence (E3), monitor detection (E6) | Summary of the four contributions; grounds the introduction's contribution list | exists |
| Table RW: competitor contrast | 2. Related Work | Five-axis comparison (substrate, score type, calibration, target model, guarantee) for all six named neighbors + this work | Bounded-novelty claim; why no single neighbor has the full set | to-produce |
| Figure E1: e1_head_collapse.png | 5.1 E1 | CARLA/real activity ratio per output head, 8 collapsed vs 2 alive | "8 of 10 heads collapse to under 1%" (VERIFIED: teardown_results.md) | exists |
| Figure E2: e2_feature_ood.png | 5.2 E2 | Feature-space scatter or distribution, real vs CARLA, showing the freeze point | "CARLA spread is 1e-5 of real; separability 87.9%, d'=2.19" (VERIFIED) | exists |
| Figure E3: e3_confidence.png | 5.3 E3 | Uncertainty distributions real vs CARLA with real p95 line | "0/220 OOD frames above real p95; ratios 1.20x-1.84x" (VERIFIED) | exists |
| Figure E4: e4_interpolation.png + e4_ram_interpolation.png | 5.4 E4 | Activity ratio and feature spread vs alpha for Subaru (cliff width 0.015) and RAM (gradient width 0.274) | Hard-cliff claim and segment-dependence claim (VERIFIED) | both exist |
| Figure E5a: e5_layer_localization.png | 5.5 E5 | Per-stage activity ratio vs alpha showing encoder stages at or above real | "Collapse is downstream of the vision encoder" (VERIFIED) | exists |
| Figure E5b: e5_submodule_localization.png | 5.5 E5 | Cliff-alpha per submodule (summarizer_div, action_block_body, passive relays) | Partial localization: summarizer VAE-mu and action-block feedback (VERIFIED) | exists |
| Table E6: detector comparison | 5.6 E6 | AUROC, AUPR, FPR@95TPR, LOCO FPR for E6/Mahalanobis/RMD/KNN-50/PCA-Mahalanobis at alpha=1.0 with 95% CI | Transfer/calibration claim; E6 LOCO ~1% vs baselines 100% (VERIFIED) | to-produce (values in metrics_results.md) |
| Figure E6a: e6_detector.png | 5.6 E6 | E6 fire rate vs alpha, showing firing before the output cliff | "Fires 0.23 blend-units before cliff" (VERIFIED: e6_results.md) | exists |
| Figure E6b: auroc_vs_alpha.png | 5.6 E6 | AUROC vs alpha for all five detectors | E6 AUROC trajectory and early-warning relative to cliff (VERIFIED) | exists |
| Figure E6c: roc_curves.png + pr_curves.png | 5.6 E6 | ROC and PR curves at alpha=1.0 for all five detectors | Threshold-free comparison; may be combined or moved to appendix | exists |
| Figure E7a: e7_auroc_heatmap.png | 5.7 E7 | 15 x 5 AUROC heatmap for E6 across corruption/severity cells | "Near chance on most; 4 high-AUROC outlier cells" (VERIFIED: e7_results.md) | exists |
| Figure E7b: e7_severity_sweep.png | 5.7 E7 | E6 AUROC vs severity per corruption family | Bounded scope of corruption sensitivity (VERIFIED) | exists |
| Figure E7c: e7_overlay.png | 5.7 E7 | Cell-for-cell E1 collapse count vs E6 AUROC, showing decoupling | "E6 firings are NOT output-collapse firings; collapse is sim-specific" (VERIFIED: e7_overlay_results.md) | exists |

Figures flagged for possible cut or consolidation:
- roc_curves.png and pr_curves.png are informative but may not earn their page budget if
  Table E6 already carries the threshold-free numbers. If space is tight, move both to an
  appendix or supplemental; do NOT cut without replacing the claim evidence in Table E6.
- e7_severity_sweep.png (E7b) and e7_auroc_heatmap.png (E7a) are complementary; if the
  heatmap alone conveys both the "near chance on most" and the "4 outlier cells" findings,
  the severity sweep may be an appendix figure. Both exist and both are load-bearing for
  the E7 claim so neither should be cut from the paper entirely.

---

## Contract-realization check

- Contribution sentence realized by: Section 4 (parity + harness: "parity-exact
  reimplementation, verified to within +/-0.5 m/s^2 of comma's reference on 100% of 1159
  real frames"), Sections 5.1-5.3 (silent failure: "simultaneous output-head collapse,
  recurrent-feature freeze, and a non-responsive uncertainty channel"), Sections 5.4-5.5
  (localization: "localizes the failure downstream of the vision encoder"), Section 5.6
  (monitor + baselines: "rolling temporal spread of the 512-D state detects the collapse
  about 0.23 blend-units before the outputs cliff at about 1% real-driving FPR, where
  location-based scores fail to transfer"), Section 5.7 (bounds: "sim-specific collapse,
  collapse-specific monitor"), Sections 1 and 3 (framing and threat model).
- Boundary check: confirmed no section introduces a claim outside the contract.
  Specifically:
  - Section 2 (Related Work) does NOT write "first to use a second-order hidden-activation
    statistic for OOD" -- EigenTrack pre-dates this on LLMs/VLMs (C2 NEAR-MISS per
    literature_map.md; the correct framing is "first on the recurrent state of a SHIPPED
    end-to-end driving model under cross-corpus LOCO").
  - Section 5.6 does NOT frame the result as "beats baselines" -- KNN-50 ties E6 at
    AUROC 1.000; the honest claim is transfer/calibration (contract exclusion lines
    108-111).
  - Section 5.7 does NOT write "E6 generalizes beyond CARLA" -- no non-CARLA output
    collapse was induced (contract exclusion lines 99-105, e7_overlay_results.md).
  - Sections 1 and 7 do NOT write "this explains openpilot's phantom braking" -- the
    user-reported issue is motivation only; no causal claim (contract exclusion lines
    88-89, 118-119).
  - Section 5.5 states the localization as PARTIAL -- VAE-mu/sigma ambiguity remains
    (contract exclusion lines 91-92).
  - Every FPR number carries the N=2 / LOCO two-fold qualifier (contract exclusion lines
    80-81, 106-107).
  - The contribution is attributed to "openpilot v0.9.7 supercombo" (named version, not
    "production driving models" plural; contract exclusion lines 97-98).
  - The monitor is described as "offline-only" with no on-road or in-stack deployment
    claim (contract exclusion lines 85-87).
  - One citation correction flagged by literature_map.md is carried into Section 2:
    the falsification paper is "von Stein and Elbaum, ASE 2022" (DOI 10.1145/3551349.3559500),
    not "Geretti et al., GPCE/SPLASH 2022." The skeleton_source.md carries the wrong
    attribution; the drafter must use the correct one.

---

## End-of-turn honesty summary

What I VERIFIED in this outline (all numbers traceable to report/*.md):

- Parity: 100% of 1159 frames within +/-0.5 m/s^2, median abs delta 0.04 m/s^2
  (report/parity_results.md, run 2026-05-30).
- E1: 8 of 10 heads collapsed (ratios quoted from report/teardown_results.md).
- E2: CARLA spread 1e-5 of real, separability 87.9%, d'=2.19
  (report/teardown_results.md).
- E3: uncertainty ratios 1.20x-1.84x, 0/220 OOD frames above real p95
  (report/teardown_results.md).
- E4 cliff: width 0.015 on Subaru; gradient width 0.274 on RAM (contract/skeleton_source).
- E5: encoder stages all at or above real (min 0.96); cliff entry at summarizer_div
  (alpha 0.900) and action_block_body (alpha 0.500) (skeleton_source).
- E6: AUROC 0.996 [0.992, 1.000], LOCO mean FPR 1.03% (max 2.07%), fires at alpha=0.550,
  0.23 blend-units before cliff at 0.784 (report/e6_results.md, report/metrics_results.md).
- Baselines: KNN AUROC 1.000 (100% LOCO FPR), RMD AUROC 0.934 (100% LOCO FPR),
  Mahalanobis AUROC 0.159 (100% LOCO FPR), PCA-Mahalanobis AUROC 0.152 (LOCO 11.91%)
  (report/metrics_results.md).
- E7: 0 of 75 corruption cells produce output collapse (max 1/10 heads vs 7/10 under CARLA),
  validation gate reproduced 7/10 CARLA collapse, 4 FP cells named and explained
  (report/e7_overlay_results.md).
- All 15 figures confirmed to exist in report/figures/ (glob verified this run).
- Citation corrections confirmed: von Stein/Elbaum ASE 2022 is the correct falsification
  paper author/venue (sourced from literature_map.md which carries a fetched confirmation).
- EigenTrack (arXiv:2509.15735) is on LLMs/VLMs, not a driving model (sourced from
  literature_map.md, which carries a fetched quote from the arXiv abstract).

What I did NOT verify (may not be treated as cleared):

- The specific cliff-alpha numbers for VAE-mu/sigma submodule ambiguity were sourced from
  the skeleton_source.md description of report/e5_submodule_results.md; I did not read
  report/e5_submodule_results.md directly (it was not listed in the five specified result
  files). ASSUMPTION: the numbers in skeleton_source.md (cliff alpha 0.900 for
  summarizer_div, cliff alpha 0.500 for action_block_body) are correctly transcribed from
  that file.
- The E4-RAM gradient width (0.274) is sourced from the contribution contract; I did not
  find a separate e4_ram_results.md to cross-check the number against. ASSUMPTION: 0.274
  is the committed result as stated in the contract.
- The "mean AUROC 0.52 to 0.74" range for E6 on ImageNet-C families is stated in the
  contract; the per-family means I quoted in E7 (fog 0.55, snow 0.55, jpeg 0.52, blur
  families 0.68-0.71) are sourced from report/e7_results.md directly (VERIFIED). The
  contract's "0.52-0.74" bound is consistent with the per-family means in the file.
- The Guo/Su 2026 venue/DOI (arXiv:2603.14603) is not yet pinned to a final publication;
  the literature_map.md flags this explicitly and states it must be re-pinned at
  camera-ready.
- EigenScore (arXiv:2510.07206) content remains [UNVERIFIED] (title-only per
  literature_map.md); CARLA (arXiv:1711.03938) and the Henriksson id are also
  [UNVERIFIED] and must be fetched before drafting.

THE SINGLE RISKIEST UNVERIFIED ASSUMPTION: the VAE-mu/sigma submodule cliff-alpha values
in Section 5.5 (summarizer_div 0.900, action_block_body 0.500) come from skeleton_source.md
rather than from a direct read of report/e5_submodule_results.md. If those numbers are
wrong or approximate in the skeleton, the localization paragraph will state incorrect
cliff-alpha values. The drafter should cross-check against the actual e5_submodule_results.md
before writing the E5 prose.
