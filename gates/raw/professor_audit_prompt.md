# Professor / Area-Chair Audit Gate (Hybrid Cross-Examination)

You are a senior PI and area chair reading this paper the way a program committee reads it:
not for typos, but for whether there is a real contribution, whether the evidence supports
the claim, whether the novelty is bounded honestly, whether the limitations are honest, and
whether the story coheres. You audit the ARGUMENT. Repo plumbing (baselines present, metrics
computed, reproducibility) is already audited separately; assume those findings and judge the
contribution built on top.

The paper targets an arXiv preprint / SafeAI-workshop-tier deliverable (NOT a main-track
generalization result). Calibrate severity to that tier: a workshop bar is "is the negative
finding real and honestly bounded," not "does it generalize."

## Your task

Render a verdict on each of five dimensions, and name the SINGLE most likely reason this
paper gets rejected. For each dimension answer OK or FATAL with a one-line reason:

1. CONTRIBUTION: Is the contribution real (a genuine advance, not a re-skin), clearly stated,
   and inside its locked boundary?
2. EVIDENCE: Does the evidence actually reach the claim, or only a weaker adjacent claim?
   Name the gap.
3. NOVELTY: Is the novelty bounded and defensible against the closest neighbor (EigenTrack
   pre-dates the second-order framing; KNN-50 ties E6, does not lose to it; the only "first"
   is "first second-order recurrent-state monitor on a shipped end-to-end driver under
   cross-corpus LOCO")?
4. LIMITATIONS: Are the limitations honest, or does the real weakness go unstated?
5. STORY: Does the narrative land (problem -> why-it-matters -> contribution -> evidence ->
   impact)?

Then: the SINGLE most likely reason this paper gets rejected, and whether it is FIXABLE or
FUNDAMENTAL.

## Key context to weigh (do not re-litigate the plumbing)

- 51/51 ledger claims are CONFIRMED with recorded quotes/numbers. The hallucination gate
  PASSED (codex + gemini + claude). No fabricated numbers, no invented baselines.
- The Reviewer-2 red-team's ONE reject-grade attack was the CARLA-as-OOD confound: "you have
  shown the model breaks on one renderer's output, not on out-of-distribution driving scenes."
  That confound is now SUBSTANTIALLY CONTROLLED: per ledger c50 and src/sim_preprocessor.py,
  the CARLA frames render at matched comma fcam intrinsics (1928x1208, focal 2648px) and the
  sim warp reduces to the SAME intrinsic remap (K_fcam @ inv(K_medmodel)) the real path
  applies, with the only asymmetry being sim zero-extrinsic-calibration vs real liveCalibration.
  Intrinsics + model-input preprocessing are identical to the parity-verified real path; the
  remaining shift is rendered CONTENT (photometry, texture, no sensor noise). The paper frames
  rendered content as the OOD axis and bounds it via E7 (the collapse is sim-specific: 0 of 75
  ImageNet-C corruption cells reproduce it, vs 7/10 under CARLA).
- The question for you: is that control + framing SUFFICIENT, or does a reviewer still reject
  on (a) the residual confound (content-domain-gap vs semantic-OOD is unresolved), (b) N=1
  model, (c) N=2 corpora two-fold LOCO, or (d) the value of a sim-specific-only finding?

## What the paper IS allowed to claim (locked contribution boundary)

A controlled distribution-shift teardown of ONE shipped production driving model (openpilot
v0.9.7 supercombo) shows it fails silently under visual OOD input (output-head collapse,
recurrent-feature freeze, non-responsive uncertainty channel), localizes the failure
downstream of the vision encoder (PARTIAL localization, VAE mu/sigma ambiguity remains), and
demonstrates a zero-retraining recurrent-feature monitor (E6, rolling temporal spread of the
512-D state) that detects the collapse ~0.23 blend-units before the outputs cliff at ~1%
real-driving FPR (N=2 two-fold estimate, NOT a production FPR), where location-based scores
(Mahalanobis, RMD, KNN) hit 100% LOCO FPR and fail to transfer.

NOT allowed: generalization beyond supercombo v0.9.7; a production FPR; E6 as a universal OOD
detector; E6 generalizing to non-CARLA corruptions as a collapse detector; on-road/in-stack
deployment; any causal link to field phantom-braking; complete mechanistic localization.

## The draft (read it as the whole argument)

[FULL DRAFT FOLLOWS]

===== drafts/rewritten_draft.md =====

# Silent Collapse: A Distribution-Shift Teardown of a Production Driving Model and a Zero-Retraining Recurrent-State Monitor

**Yusuf Guenena**
Wayne State University
Code and data: https://github.com/yusufdxb/supercombo-blindspot

---

## Abstract

Production Level-2 driver-assistance stacks are validated largely in simulation, and that
practice rests on an unstated assumption: that the shipped model behaves the same on rendered
input as on real input, or at least signals when it does not. We test that assumption on one
deployed model, openpilot v0.9.7 supercombo, the network that drives comma hardware on public
roads. We first build a parity-exact reimplementation of its inference path, verified to within
+/-0.5 m/s^2 of comma's own reference output on 100% of 1159 real-footage frames (median
absolute delta 0.0409 m/s^2), so that any downstream anomaly is attributable to the model and
not to our harness. Running the verified model on CARLA-rendered clean roads, we find that it
fails silently: 8 of 10 output heads collapse to under 1% of their real-driving temporal
activity and the 512-D recurrent feature freezes to about 1e-5 of its real spread, while the
model's own predictive-uncertainty heads rise only 1.20x to 1.84x and not one out-of-distribution
frame (0 of 219) crosses the real-driving 95th percentile. Nothing the model emits flags the
collapse. An alpha-blend sweep characterizes the collapse as a hard cliff on the Subaru source
(transition width 0.015), though the cliff shape is segment-dependent (a gradient of width 0.274
on the RAM source), and a layer-by-layer probe localizes it downstream of the vision encoder, in
the recurrent summarizer and action-block feedback path, not in perception. We then show that the
signal the outputs hide is recoverable from the model's own recurrent feature with a single
second-order statistic, the rolling temporal spread of the 512-D state: one O(d) quantity per
forward pass, with no retraining and no architecture change. Calibrated leave-one-corpus-out to
about a 1% real-driving false-positive rate (N=2, a two-fold estimate, not a production FPR), it
separates the collapse at AUROC 0.996 [0.992, 1.000] and fires about 0.23 blend-units before the
outputs cliff, where the location-based feature scores one would default to (Mahalanobis,
Relative Mahalanobis, KNN) each hit 100% leave-one-corpus-out FPR and fail to transfer across the
two real corpora. This is a single-model negative finding with a collapse-specific monitor, not a
general OOD detector: an ImageNet-C sweep shows the silent collapse is sim-specific (no corruption
reproduces it; at most 1 of 10 heads collapses on any of 75 corruption-severity cells, versus 7
of 10 under CARLA), and the monitor is near chance on most photometric corruptions. The
contribution is that output-side and location-based signals alone are insufficient for this one
shipped model's safety case, and that a second-order recurrent-state monitor is a cheap complement
that the present evidence does not claim generalizes.

---

## 1. Introduction

Most Level-2 and autonomous-driving programs validate their driving policy largely in simulation,
because simulation is a primary setting in which rare and dangerous scenarios can be exercised at
scale and at low cost. The validity of that practice rests on an assumption that is rarely tested
directly in the literature we surveyed: that the shipped model under test behaves on rendered input the
way it behaves on real input, or, failing that, that it fails loudly enough for the test harness
or a downstream safety monitor to notice. If a model can be shown a simulated scene and quietly
stop doing its job while every output it emits still looks plausible, then a passing simulation is
not evidence that the model works. It is evidence that the model produced a safe-looking default,
and the two are indistinguishable from the outside.

We test that assumption on a single deployed model, openpilot v0.9.7 supercombo, the end-to-end
network in comma's shipped openpilot driver-assistance system (Chen et al. 2022), and we find the dangerous answer. On
CARLA-rendered clean roads, 8 of 10 of the model's output heads collapse to under 1% of their
real-driving temporal activity, and the 512-D recurrent state that threads the model's memory
across frames freezes to roughly 1e-5 of its real spread. Yet the model's own predictive-uncertainty
heads rise only 1.20x to 1.84x, and not one out-of-distribution frame, 0 of 219, exceeds the
95th-percentile uncertainty the model exhibits in real driving. The failure is silent by the
model's own signals: the planning trajectory, the acceleration command, and the lane and lead
geometry all go dark, but the channel a safety case would watch to catch exactly that event stays
quiet.

The failure mode this exposes is not hypothetical. Phantom braking under distribution shift, the
model commanding a deceleration for an obstacle that is not there, is a known and user-reported
failure of the shipped openpilot stack, documented in the project's own issue tracker (commaai
issue #20704). We cite that report as motivation only; we make no causal claim linking the silent
collapse we measure in simulation to any specific field braking event. The point it grounds is
narrower and sufficient: a simulation "pass" can be the model having collapsed to a safe-looking
default rather than the model perceiving the scene, and the output-side and uncertainty signals a
downstream safety case would trust are precisely the signals that stay silent when it does. That
is the gap.

The closest published monitors do not close it. The nearest AV-native neighbor watches the feature
density of a frozen perception encoder one stage upstream of where the collapse lives (Keser et al.
2025), and the next-nearest watches the latent dynamics of a standalone trajectory predictor with
provable changepoint guarantees (Guo and Su 2026); neither targets the recurrent state of a shipped
end-to-end driver. A recent line in language and vision-language models, EigenTrack (arXiv:2509.15735),
does stream a second-order statistic of hidden activations through a trained classifier with early
warning, but on LLMs and VLMs, not on a driving model, and not under cross-corpus transfer. And the
standard location-based feature-space scores one would reach for first (Mahalanobis, Relative
Mahalanobis, KNN) each hit 100% leave-one-corpus-out false-positive rate on this model: calibrated
on one of our two real corpora, they flag the entirety of the other.

This paper makes four contributions, each bounded to the single model under study. First, a
parity-exact reimplementation of openpilot v0.9.7 supercombo inference, verified to within
+/-0.5 m/s^2 of comma's reference output on 100% of 1159 real frames (median absolute delta
0.0409 m/s^2), including two non-obvious correctness details (recurrent-state threading and
unnormalized YUV input) that a faithful reimplementation must get right. Second, an empirical
demonstration of silent failure under visual distribution shift on this model: simultaneous
output-head collapse, recurrent-feature freeze, and a non-responsive uncertainty channel, with
0 of 219 OOD frames exceeding the model's real-driving uncertainty p95. Third, a characterization
of the collapse as a hard cliff on the Subaru source (transition width 0.015) whose shape is
segment-dependent (a gradient of width 0.274 on the RAM source), localized downstream of the
vision encoder in the recurrent summarizer and action-block feedback path (a partial localization;
an ambiguity in the summarizer's variational bottleneck remains). Fourth, a zero-retraining monitor
on the model's own recurrent feature, the rolling temporal spread of the 512-D state, calibrated
leave-one-corpus-out to about a 1% real-driving FPR (N=2, a two-fold estimate), which separates the
collapse at AUROC 0.996 and fires about 0.23 blend-units before the output cliff, where the
location-based scores fail to transfer; bounded by an ImageNet-C sweep that shows the collapse is
sim-specific and the monitor collapse-specific, not a universal OOD detector. Figure 1 previews all
four findings.

---

## 2. Related Work

This contribution sits at the intersection of two ancestral lines. One is location-based
feature-space OOD detection, the family that this paper runs as baselines and shows fails to
transfer across corpora rather than fails to separate within one. The other is internal-activation
runtime monitoring, the family the proposed monitor belongs to, pushed onto the recurrent state of
a shipped driver. We name the neighbors a reviewer will invoke and state, for each, exactly what it
lacks that this work has. Table RW compresses the comparison onto five axes.

**Location-based feature-space OOD (the baseline family).** The output-side floor is maximum
softmax probability (Hendrycks and Gimpel 2017) and its energy-based successor (Liu et al. 2020);
the feature-space ancestor is the Mahalanobis distance-from-fitted-Gaussian-mean score (Lee et al.
2018), refined for near-OOD by the relative Mahalanobis distance (Ren et al. 2021), and made
non-parametric by deep nearest-neighbor distance (Sun et al. 2022). ViM (Wang et al. 2022) is the
modern feature-residual-plus-logit hybrid, and Mahalanobis++ (Mueller and Hein 2025) keeps the
feature-Gaussian family live in 2025 with an l2-normalization fix, which is why running the Lee
2018 score here is a fair current comparison and not a strawman. OpenOOD (Yang et al. 2022) codifies
this whole line into one taxonomy and codebase; it is the vocabulary anchor for our baselines, not a
leaderboard this paper ranks on. We run Mahalanobis, relative Mahalanobis, and KNN-50 as baselines
on the same 512-D feature our monitor reads. The honest result, developed in Section 5.6, is that
KNN-50 ties our monitor on single-corpus separation (AUROC 1.000); the distinguishing axis is
cross-corpus calibration, where all three location-based scores hit 100% leave-one-corpus-out FPR.
MSP and Energy are output-side scores on a model whose output channel Section 5.3 shows is silent,
and ViM requires a classifier weight matrix that supercombo's multi-head regression outputs do not
provide; these three are structurally inapplicable, and we name and excuse them rather than omit
them.

**AV-native OOD and uncertainty monitoring.** The closest published neighbor, Keser et al. 2025,
monitors the feature density of a frozen vision foundation-model encoder as an in-distribution
score; that is one stage upstream of our substrate and is exactly the location-based class our
baselines instantiate. The next-closest, Guo and Su 2026, monitors the latent dynamics of a
standalone trajectory predictor as a quickest-changepoint-detection problem with provable bounds on
detection delay and false alarm; it differs from this work in model class (a standalone predictor,
not a shipped end-to-end driver) and in evidence basis (a theoretical guarantee, not an empirical
calibration). Filos et al. 2020 is the canonical framing of distribution shift for autonomous
vehicles, and the SelfOracle line (Stocco et al. 2020) and its uncertainty-quantification successor
(Grewal, Tonella, and Stocco 2024) build misbehaviour predictors on the assumption that the model's
confidence signal is informative; Section 5.3 is the contrary finding for this model. A 2025 position
paper frames OOD detection explicitly as safety-case evidence (Hodge et al. 2025), which is the niche
this work speaks to. None of these targets the recurrent state of a shipped end-to-end driver under
cross-corpus transfer.

**Internal-activation and second-order monitors (the proposed monitor's family).** Runtime neuron
activation pattern monitoring (Cheng et al. 2018) is the ancestor of internal-state monitoring: it
stores binarized neuron patterns and compares them by Hamming distance, on feed-forward classifiers,
a first-order discrete per-frame check. An early move to a higher-order feature statistic for OOD
is the Gram-matrix method (Sastry and Oore 2020), the closest lineage point to a second-order rather
than distance-from-mean choice. The proposed monitor inherits the higher-order-statistic idea from
the latter and the internal-monitor idea from the former, but applies them to a recurrent state
(neither did) on a shipped production driving model (neither did), and it is the statistic that
catches the freeze mode, the recurrent spread crashing across the cliff, that a per-frame pattern or
Gram check on outputs would miss. The single closest paper on mechanism is EigenTrack
(arXiv:2509.15735), a parallel and near-simultaneous line that streams covariance-spectrum statistics
(eigenvalue gaps, spectral entropy, random-matrix-theory features) of hidden activations through a
trained recurrent classifier, with early warning, to flag hallucination and OOD drift. It rhymes with
this work on three counts (second-order, hidden-state, fires before surface errors), and it differs
on three checkable ones: substrate (LLMs and VLMs, not a driving model), statistic (a full
eigenspectrum fed to a trained classifier, not a single location-invariant trace with a calibrated
threshold), and evaluation (language OOD benchmarks, not cross-corpus leave-one-corpus-out FPR on a
recurrent driver state). We therefore do not claim to be first to use a second-order hidden-activation
statistic for OOD detection: EigenTrack pre-dates this work on that framing. The available and
defensible claim is narrower: this is the first second-order recurrent-state monitor on a shipped
end-to-end driving model evaluated under cross-corpus leave-one-corpus-out transfer. NECO (Ben Ammar
et al. 2024) builds on a neural-collapse property of classification heads, which supercombo's
regression heads lack; we name and excuse it rather than run it.

**openpilot and supercombo prior work.** The reference academic teardown of supercombo is
Openpilot-Deepdive (Chen et al. 2022), a static input, output, and architecture analysis plus a
reimplementation; this paper extends that static teardown to a runtime distribution-shift teardown.
The directed-falsification line on openpilot is von Stein and Elbaum (ASE 2022), which generates
adversarial inputs that violate stated properties; it is related testing work and motivation, not a
silent-collapse or recurrent-state-monitor study. A recent adversarial study (arXiv:2505.11532)
targets supercombo with deliberate perturbations and input-level defenses, a different failure mode
(adversarial attack, not sim-rendered silent collapse) with no recurrent-state monitor.

**Simulation testing of driving DNNs and corruption robustness.** The DeepRoad line uses
metamorphic and generative test synthesis: DeepTest (Tian et al. 2018) produces transformed driving
images to surface erroneous behaviors, DeepRoad (Zhang et al. 2018) uses GANs to render the same
scene under new weather and checks for consistent behavior, and MarMot (Ayerdi et al. 2024) applies
metamorphic relations at runtime. These methods generate or transform driving scenes and test the
model for consistent behavior, implicitly treating the generated input as a valid scene the model
should handle; this paper inverts that premise by showing the simulated input can itself be out of
distribution to the model, which undercuts coverage claims built on sim-based testing.
ImageNet-C (Hendrycks and Dietterich 2019) is the corruption-robustness yardstick we use as the
bounding OOD axis in Section 5.7, with Cityscapes-C (Michaelis et al. 2019) as its AV extension, and
CARLA (Dosovitskiy et al. 2017) is the simulator that supplies the primary OOD axis.

---

## 3. Threat Model

A safety case for a shipped driving model leans on a small set of runtime defenses, and the finding
of this paper is that the specific failure mode it documents defeats exactly those defenses, silently
and simultaneously. We define the threat precisely, walk through why each standard defense misses it,
and position the proposed monitor as a cheap complementary layer rather than a replacement for any of
them.

**The threat.** A shipped driving model is deployed in a visually shifted context: a rendered
simulator, an unfamiliar geography, a degraded or unusual sensor condition. Under sufficient shift,
three things happen at once and without warning. The output heads collapse to a plausible, nearly
constant signal; the recurrent state that carries the model's temporal memory freezes; and the
model's own predictive-uncertainty channel does not rise. The danger is not that the model is wrong,
which a safety case expects and plans for, but that it is wrong while every signal a monitor would
read says it is fine.

**Why uncertainty-head monitoring misses it.** The most direct defense is to threshold the model's
own predictive uncertainty: if the model says it is unsure, intervene. On this model that defense
never fires. Under the collapse, predictive uncertainty rises only 1.20x to 1.84x across the
monitored heads, and 0 of 219 OOD frames exceeds the 95th-percentile uncertainty the model exhibits
in real driving (Section 5.3). A threshold calibrated on real driving, which is the only honest way
to set it, never trips. This directly contradicts the SelfOracle and uncertainty-quantification line,
which builds misbehaviour predictors on the premise that the confidence signal carries the
information.

**Why output-plausibility and temporal-jitter monitors miss it.** A second defense checks that the
model's outputs are physically plausible: bounded acceleration, feasible curvature, a sane plan. The
collapsed outputs are plausible by construction. The plan head retains 0.6% of its real activity and
the acceleration head 0.4%, so the model emits a smooth, near-constant trajectory that reads as a
benign, stationary scene, and a plausibility check passes it. A third defense watches for temporal
jitter or output disagreement as a sign of instability. The freeze produces the opposite of jitter:
a frozen output has lower temporal variance than an active one, so a jitter monitor reads the collapse
as increased stability, the very thing it is built to reward. (These are structural arguments from the
measured activity ratios, not separate experiments.)

**Why same-architecture ensembles and input-quality checks miss it.** A fourth defense runs an
ensemble and flags disagreement. Section 5.5 localizes the collapse downstream of the vision encoder,
in a path every instance of the same architecture shares, so an ensemble of the same model would
collapse together rather than disagree. A fifth defense screens the input itself for quality. The
CARLA-clean renders are typically sharper and less noisy than real road footage, so an image-quality
screen would rate the simulated input as good, not anomalous. (Structural arguments, not new experiments.)

**The complementary signal.** The defenses above all read the output side or the input side. The
finding that makes a complement possible is that the model's own recurrent features carry the OOD
signal its outputs do not surface. The proposed monitor (Section 5.6) is a complement, not a
replacement: it is one O(d) statistic computed from a forward pass that already runs, it requires no
retraining and no architecture change, and it is calibrated against a real-driving false-positive rate
rather than against simulated negatives. It is also bounded, and we state the bounds here so they
travel with the proposal: it is collapse-specific (Section 5.7), it is demonstrated offline only, and
its false-positive rate is an N=2 two-fold estimate, not a production number.

---

## 4. Method

This section describes the parity-verified harness, the data, the metrics, the monitor design, and
the baseline set in enough detail to trust and reproduce the teardown. The load-bearing content is
the parity number and the design descriptions; all figures live in Section 5.

**Parity-exact reimplementation.** We reconstruct openpilot v0.9.7 supercombo inference from the
released ONNX model and comma's own reference files (modeld, the output parser, the YUV loader kernel,
and the model constants). Because the central result is a negative finding about a production model,
the harness must be trustworthy before any anomaly can be attributed to the model rather than to the
reimplementation, so we establish parity first.

**Recurrent-state threading.** supercombo is a temporal model: it consumes a buffer of recent
features and its own previous desired-curvature output, and it must roll that state forward by
shift-and-append after each inference, with a zero initialization only on the first frame. A naive
per-frame zero reset of the state produces a multi-second initialization transient that, in this
model, looks exactly like a spurious deceleration, a self-inflicted phantom brake of the
reimplementation rather than the model. Getting this right is a precondition for both parity and the
collapse measurement, because every collapse and corruption sequence is re-run with the same correct
state handling.

**Unnormalized YUV input.** The model's input loader (loadyuv) converts the camera Y, U, and V
channels to float with no rescaling: the model consumes uint8 values in the range 0 to 255, not
values divided by 255. Dividing by 255, the reflexive normalization, shifts the entire input
distribution and silently degrades parity. We feed unnormalized uint8 YUV, matching the kernel.

**Parity result.** On 1159 real-footage frames (after a 40-frame, 2-second warm-up trim), our
reimplemented longitudinal-acceleration output matches comma's logged reference within +/-0.5 m/s^2
on 100.00% of frames, with a median absolute delta of 0.0409 m/s^2, a mean of 0.0541, and a worst-case
single-frame delta of 0.2899 m/s^2 (no frame exceeds 0.5). This is the load-bearing harness-trust
claim for the negative result, and we report it prominently.

**Data.** The real in-distribution data are two comma corpora, denoted subaru and ram, of 320 frames
each, with the first 100 frames discarded as warm-up. After warm-up and the rolling-window step, 219
analysis frames remain per corpus. The out-of-distribution data are CARLA-rendered clean-road frames
from the openpilot v0.9.7 simulation pipeline. The sim camera is configured as a matched-intrinsics
pinhole: it renders at the comma 3 fcam native resolution (1928x1208) at the field of view that
reproduces fcam's focal length (2648 px), so the rendered frames carry exactly the production camera
(`_ar_ox_config.fcam`) intrinsics, and because the sim camera is mounted on the device axes (zero
extrinsic calibration), its warp to the model frame reduces to the same intrinsic remap
(`K_fcam @ inv(K_medmodel)`) that the real path applies after its `liveCalibration` extrinsics
(`src/sim_preprocessor.py`). The intrinsics and the model-input preprocessing are therefore identical
to the parity-verified real path, and the distribution shift is confined to rendered image content
(photometry, texture, and the absence of sensor noise); the collapse is a response to rendered scene
content, not an artifact of a mismatched camera model. The interpolation axis for the cliff analysis
(Section 5.4) is a pixel-space alpha-blend of a real sequence (Subaru or RAM) with the CARLA sequence,
with alpha=0 the real frame and alpha=1 the CARLA frame, swept over 29 alpha values. We state the
underlying counts wherever a percentage appears: for example, the "0%" of Section 5.3 is 0 of 219
CARLA frames. For the threshold-free metrics (Section 5.6), the in-distribution set is subaru and ram
concatenated (n=638 stored frames) and the out-of-distribution set is the alpha=1.0 CARLA frames
(n=319 stored frames); the rolling-window warm-up leaves 609 and 290 valid (non-NaN) scores
respectively, and the threshold-free metrics are computed on those valid subsets.

**Metrics.** Output activity is the sum of per-element temporal standard deviation over a window; a
head is "collapsed" when its CARLA-to-real activity ratio is small. Feature spread is the trace of the
recurrent-state covariance over a rolling window. Detection is scored threshold-free with AUROC, AUPR,
and FPR at 95% TPR, each with a stratified bootstrap 95% confidence interval (n=1000 iterations,
seed 42). Calibration uses a leave-one-corpus-out (LOCO) protocol across the two real corpora: a
threshold is set on one corpus and its false-positive rate is read on the held-out corpus, then the
two folds are averaged. Because there are only two real corpora, every LOCO FPR is a two-fold estimate
whose variance is not meaningfully reportable, and we state it as such every time it appears.

**The monitor (E6).** The monitored quantity is the rolling temporal spread of the 512-D recurrent
feature the model emits, computed as the trace of the covariance of a 30-frame window of the state.
The detection threshold is the 1st percentile of the real-driving rolling-spread distribution, which
targets a roughly 1% false-positive rate by construction, and it is calibrated leave-one-corpus-out.
The monitor adds one O(d) statistic per forward pass, with no retraining and no extra heads, and its
statistic is location-invariant (a second-order spread), which is the property that distinguishes it
from the location-based baseline family.

**Baselines and structural exclusions.** The three applicable post-hoc feature-space scores,
Mahalanobis, relative Mahalanobis (RMD), and KNN-50, are computed on the same 512-D feature the
monitor reads, with a PCA-Mahalanobis variant as an ablation. RMD's background distribution is fit as
a two-component Gaussian mixture, because with a single in-distribution class the Ren et al. marginal
Gaussian degenerates to the class Gaussian and the relative score collapses to zero. MSP, Energy, and
ViM are structurally inapplicable to supercombo's multi-head Gaussian-mixture regression and
existence-probability outputs (there is no softmax head, no logit vector, and no classifier weight
matrix), and we state this explicitly so reviewers read it as an exclusion with a reason, not an
omission.

---

## 5. Experiments and Results

The experiments unfold in the order the argument requires: parity establishes trust (Section 4),
collapse establishes the phenomenon (E1), the freeze and the silent uncertainty channel establish the
gap (E2, E3), the cliff sweep characterizes the shape (E4), the layer probe localizes the mechanism
(E5), the monitor and baselines deliver the solution (E6), and the corruption sweep bounds the scope
(E7).

### 5.1 E1: Output collapse

We run the parity-verified model on CARLA-rendered clean roads and measure, for each of the 10 output
heads, the ratio of its CARLA temporal activity to its real-driving temporal activity. Eight of the
ten heads collapse to under 1% of real activity: desired_curv (0.0018), accel_t0 (0.0040), lead
(0.0042), desire_state (0.0049), lane_lines (0.0054), plan (0.0057), lead_prob (0.0058), and
road_edges (0.0076). These eight are the safety-critical driving signals: the planning trajectory, the
acceleration command, the lane and road-edge geometry, the lead-vehicle detection, and the curvature
command. Two heads survive: pose (0.1788) and meta (0.7181), the ego-motion and meta-state outputs.
The model's primary driving outputs are functionally inactive on CARLA-rendered input while its
ego-motion and meta outputs partially persist. Figure E1 plots the per-head ratios.

### 5.2 E2: Recurrent-feature freeze

We measure the rolling covariance trace of the 512-D recurrent hidden state on real versus CARLA
frames. On CARLA the feature spread falls to about 1e-5 (0.00001x) of its real value: the recurrent
state freezes to a near-constant vector. Despite that freeze, the frozen vector is linearly separable
from the real-driving states at 87.9% along the centroid-difference direction (d' = 2.19), which
establishes that the recurrent state carries a strong out-of-distribution signal even while the output
heads are dark. This is the signal the monitor of Section 5.6 reads. It also previews the mechanism
that explains the Mahalanobis failure in Section 5.6: the frozen vector lands near the high-density
center of the in-distribution Gaussian, so a distance-from-mean score reads it as in-distribution.
Figure E2 shows the projected feature space with the CARLA freeze point.

### 5.3 E3: Uncertainty silence

This is the safety-relevant centerpiece. For three representative heads we report, side by side, the
fraction of output activity retained on CARLA, the ratio of CARLA to real predictive uncertainty, and
the fraction of the 219 CARLA frames that exceed the real-driving 95th-percentile uncertainty for that
head. The plan head retains 0.6% of its activity while its uncertainty rises 1.35x; the lead head
retains 0.4% at 1.20x; the desired_curv head retains 0.2% at 1.84x. In all three cases, 0 of 219 CARLA
frames exceeds the real-driving p95. The outputs lose roughly 99.5% of their activity, but the
uncertainty channel barely moves and never crosses the threshold a real-calibrated monitor would set.

The implication is the gap this paper turns on. A safety monitor that thresholds the model's own
uncertainty, calibrated on real driving, never fires under the collapse, because nothing the model
emits flags it. We state this strictly as an empirical finding about supercombo v0.9.7 under CARLA
input, that the uncertainty channel stays quiet, and we draw no causal line from it to any field
phantom-braking incident. Figure E3 plots the uncertainty distributions for real and CARLA frames
against the real-driving p95 line, making the silence visually unambiguous.

### 5.4 E4: Cliff characterization and segment dependence

To characterize the shape of the collapse, we sweep the pixel-space alpha-blend from the real frame
(alpha=0) toward the CARLA frame (alpha=1). On the Subaru source the response is two-phase. Output
activity first balloons to 6.32x of the real baseline at alpha=0.425, a thrash driven by the
ghosted-input interference of the half-blended frame, and then collapses through a hard cliff, falling
from 0.9x to 0.1x of real activity over the narrow alpha band 0.784 to 0.799, a transition width of
0.015. The feature spread crashes from 0.25 to 0.00 by about alpha=0.78. Through the entire transition
the predictive uncertainty never spikes, consistent with E3.

The cliff shape is segment-dependent, and this is a bound on the early-warning headroom the monitor
exploits. On the RAM source the same sweep produces a gradient, not a cliff: output activity falls from
0.9x to 0.1x of real over the wide alpha band 0.666 to 0.940, a transition width of 0.274. On the RAM
source the monitor's firing point (alpha=0.850) lands inside that band rather than before it, so the
early-warning headroom is negative (-0.184) where on Subaru it is positive (0.234). Cliff headroom
therefore cannot be assumed to generalize across segments; the characterization is partial. We use the
authoritative `report/e4_results.md` and `report/e4_ram_results.md` values (Subaru width 0.015, RAM
width 0.274) throughout; where a figure legend rounds these (for example a Subaru legend reading
"0.02"), the results-file value governs. Figure E4 places the Subaru cliff and the RAM gradient side by
side.

### 5.5 E5: Localization downstream of the vision encoder

We first ask whether the collapse is the vision encoder failing, by measuring the CARLA-to-real
activity ratio of each encoder stage across the alpha sweep. It is not. Every encoder stage stays at or
above real activity across the full sweep: at alpha=1 the stem is at 1.43x, stage3 at 2.06x, and the
head at 2.14x, and the minimum ratio over all stages and all alpha is about 0.96. No encoder stage ever
crosses the 0.5 collapse threshold. The encoder is in fact more active on CARLA than on real input, so
the collapse seen at the output is not perception failing; it originates downstream. The structural
statement is explicit: the failure is in the summarizer and action block, not in the encoder.

We then probe eight tensors between the summarizer and the per-head outputs. Two cross the 0.5 collapse
threshold. The recurrent summarizer's variational bottleneck (summarizer_div, which is the hidden_state
the monitor reads) has a cliff at alpha=0.900, with its mean shifting to 0.023 of real, a near
two-order-of-magnitude collapse of the rolling mean of the 512-D vector; this is the entry point of the
collapse. The action-block body (action_block_body), the last residual block before the curvature head,
has a cliff a full alpha step earlier, at alpha=0.500, because it folds in the model's own previous
desired-curvature output through the recurrent feedback loop, so once that already-collapsing signal
joins, the action stack saturates fast. The transformer self-attention, feed-forward, and reduce-sum
stages track the summarizer to within 2 to 11% and introduce no additional collapse; they are passive
relays. The post-encoder projection (vision_post, 1.89x at alpha=1) and the non-temporal hydra trunk
(hydra_trunk, 2.71x at alpha=1) show no cliff, consistent with the two surviving heads of E1. The
localization is partial: the summarizer ends in a mu-over-sigma variational reparameterization, and we
have not separated the mu path from the sigma normalization, so part of the apparent summarizer collapse
could be variance normalization rather than information loss. We state this ambiguity rather than claim
a complete mechanistic account. Figures E5a and E5b show the per-stage and per-submodule ratios.

### 5.6 E6: Monitor detection and baseline comparison

We now test whether the signal the outputs hide is recoverable from the recurrent feature, and how a
second-order monitor of that feature compares to the location-based scores one would default to.

The threshold-free comparison is at alpha=1.0 (full CARLA shift), on the in-distribution set of subaru
and ram concatenated (n=638 stored frames) against the CARLA OOD frames (n=319 stored frames), with
stratified bootstrap 95% confidence intervals; after the rolling-window warm-up, the metrics are
computed on the 609 and 290 valid scores that remain. The rolling-spread monitor reaches AUROC
0.996 [0.992, 1.000], AUPR 0.995 [0.990, 1.000], and FPR at 95% TPR of 0.000. KNN-50 reaches AUROC
1.000 [1.000, 1.000]. We do not claim to beat KNN-50: on single-corpus separation the two tie. Relative
Mahalanobis reaches AUROC 0.934 [0.914, 0.952]. Mahalanobis (0.159 [0.130, 0.190]) and PCA-Mahalanobis
(0.152 [0.124, 0.179]) both score below chance. Table E6 collects these with their confidence intervals.

Mahalanobis scores below chance for a mechanical reason worth stating, not just reporting. Under the
collapse the recurrent state freezes to a near-constant vector that lands near the center of the
in-distribution Gaussian (E2), so the OOD frames receive lower Mahalanobis distance than the real
frames: a distance-from-mean score cannot detect collapse-to-the-mean, and inverts. PCA-Mahalanobis
inherits the same inversion.

The distinguishing axis is not single-corpus separation but cross-corpus calibration, and here the
location-based scores fail to transfer. Under leave-one-corpus-out across subaru and ram, all three
applicable location-based baselines (Mahalanobis, relative Mahalanobis, KNN-50) hit 100% LOCO FPR: a
threshold calibrated on one corpus flags the entirety of the other. The PCA-Mahalanobis variant improves
to 11.91% LOCO mean FPR, a partial recovery but still an order of magnitude above the 1% operating point.
The rolling-spread monitor holds 1.03% LOCO mean FPR (max 2.07%), where we emphasize that this is an N=2
two-fold estimate, not a production false-positive rate. The geometric reason the location-based scores
fail is that the subaru and ram corpora occupy disjoint regions of the 512-D feature space whose
inter-corpus separation dwarfs the within-corpus radius (visible as the two real clusters in Figure E2),
so any absolute-position score calibrated on one corpus flags the whole of the other. The monitor reads
the second-order trace, which is location-invariant, so it both separates and calibrates across the
corpus shift.

The monitor also fires before the outputs cliff. Sweeping alpha, the monitor's fired fraction crosses
50% at alpha=0.550, where the E4 Subaru output cliff is at about alpha=0.784, an early-warning gap of
about 0.23 blend-units. (As Section 5.4 notes, this headroom is Subaru-specific; on the RAM gradient the
firing point falls inside the transition band.) Figures E6a and E6b show the fire rate and the AUROC
trajectory across alpha, and Figure E6c gives the ROC and PR curves at alpha=1.0.

The honest positioning against the nearest neighbors follows from these numbers. Against the
location-based family (Keser-style density, Lee Mahalanobis, Ren RMD, Sun KNN), the result is not that
the monitor separates better, since KNN-50 ties it, but that the location-based scores fail to transfer
across corpora (100% LOCO FPR) while the second-order monitor calibrates (about 1%, N=2). Against
EigenTrack (arXiv:2509.15735), the nearest mechanism-twin, the deltas are substrate (a shipped driving
model, not an LLM or VLM), statistic (a single location-invariant spread trace with a calibrated
threshold, not a full eigenspectrum fed to a trained classifier), and evaluation axis (cross-corpus LOCO
FPR on real-driving frames, not language OOD benchmarks). We frame the result as transfer and
calibration, not as outperforming baselines.

### 5.7 E7: Corruption sweep (two-way bound)

The final experiment bounds the contribution in both directions: it bounds the failure mode (is the
silent collapse a general model-robustness failure, or sim-specific?) and it bounds the monitor (is the
monitor a general OOD detector, or collapse-specific?). We apply the 15 ImageNet-C corruptions at 5
severities each to the real Subaru frames (in raw RGB, before YUV conversion), re-run supercombo with the
correct recurrent-state handling on each corrupted sequence, and evaluate the monitor and the baselines,
with a validation gate that first reproduces the published CARLA collapse (7 of 10 heads) before any
corruption result is read.

**The collapse is sim-specific.** Running the same E1 output-collapse metric on every corruption-severity
cell, no ImageNet-C corruption reproduces the output collapse: 0 of 75 cells reach the collapse criterion
(5 or more of 10 heads), and the maximum on any single cell is 1 of 10 heads, against 7 of 10 under CARLA.
The silent collapse is a property of full-sim rendering, not of photometric or blur corruptions of real
frames. One consequence is immediate: there are zero false negatives, because there is no output collapse
anywhere in the sweep for the monitor to miss, so its quiet response on fog, brightness, blur, and the
rest is correct.

**The monitor is collapse-specific.** On most corruptions the monitor sits near chance. Its mean
per-corruption AUROC is 0.55 on fog, 0.55 on snow, 0.52 on jpeg compression, 0.60 on brightness, and 0.58
on zoom blur, rising only to about 0.68 to 0.71 on the defocus, motion, and glass blur families, all well
below the operating regime it holds on CARLA. The monitor fires on only four of the 75 cells at the
calibrated threshold: frost severity 3 (AUROC 0.958), frost severity 5 (AUROC 1.000), gaussian-noise
severity 4 (AUROC 0.861), and impulse-noise severity 5 (AUROC 0.906). Crucially, all four of those cells
have zero output collapse, so on this corpus the monitor's firings track a recurrent-feature-spread shift,
not the collapse mode. The monitor is therefore correctly quiet on the real-frame corruptions the model
tolerates and is not a universal corruption detector.

**Baselines under corruption.** The location-based baselines fire on many corruptions at their calibrated
thresholds (Mahalanobis and relative Mahalanobis fire at high rates across fog, frost, several noise
families, snow, contrast, and zoom), but recall from Section 5.6 that both carry 100% LOCO FPR: they also
flag held-out real driving, so their high corruption fire rates are not calibrated detection. KNN-50 fires
only on the heaviest noise and frost. None of the baselines is calibrated to the 1% operating point the
monitor holds.

**Synthesis.** The corruption sweep narrows the contribution precisely. The silent collapse, the dangerous
mode, is sim-specific; the corruption sweep shows it is not a general model-robustness failure, and the
cell-for-cell overlay shows the monitor is not a general corruption detector. The contribution is a
targeted monitor for the specific, dangerous, sim-induced silent-collapse mode at about a 1% real-driving
FPR (N=2), where output-side and location-based feature scores do not detect it. We do not claim the
monitor generalizes beyond CARLA as a collapse detector, because no non-CARLA output collapse was induced
for it to have caught. Figures E7a, E7b, and E7c give the AUROC heatmap, the severity sweep, and the
cell-for-cell collapse-versus-detection overlay.

---

## 6. Limitations

We state the boundaries that define where the evidence does and does not extend, so that the bounded
finding is not mistaken for a general result.

**N=1 model.** Every result is on supercombo v0.9.7 alone. No other openpilot version, and no Tesla,
Mobileye, Waymo, or research imitation-learning stack, was tested. The silent-collapse phenomenon and the
monitor are demonstrated on this one model, and we claim no generalization to any other. Whether silent
collapse is a property of this architecture, this training recipe, or end-to-end driving models broadly is
the N>1 study this paper does not attempt.

**N=2 corpora; LOCO is a two-fold estimate.** The cross-corpus calibration rests on two real corpora.
Leave-one-corpus-out at N=2 is a two-fold estimate whose variance is not meaningfully reportable, and the
roughly 1% FPR is therefore a calibration estimate, not a production false-positive rate. A third real
corpus, at minimum, is needed before any single production FPR can be quoted with an uncertainty.

**Monitor scope: collapse-specific and offline-only.** The corruption overlay (Section 5.7) shows the
monitor is collapse-specific and near chance on most real-frame corruptions, so it is not a universal OOD
detector. It is demonstrated offline on logged, rendered, and corrupted frames only; no on-road, in-stack,
or real-time deployment was run, and no causal link to field incidents was established. The residual gap is
that no non-CARLA output collapse was ever induced, so the monitor was never tested as a collapse detector
on anything other than CARLA; real adverse-weather footage (rain, night, glare) that actually induces a
non-CARLA collapse remains pending, and is the most reviewer-resistant follow-up.

**Partial localization.** The collapse is pinned to the summarizer's variational bottleneck and the
action-block feedback path by ruling out the encoder and probing eight submodules, but the mu-versus-sigma
ambiguity inside the summarizer's reparameterization is unresolved, so the localization is partial, not a
complete mechanistic account.

---

## 7. Conclusion

A shipped Level-2 driving model, openpilot v0.9.7 supercombo, shown CARLA-rendered input, collapses to a
plausible near-constant and does not raise its own uncertainty: 8 of 10 output heads fall to under 1% of
real activity, the recurrent state freezes to about 1e-5 of its real spread, and 0 of 219
out-of-distribution frames exceeds the model's real-driving uncertainty p95. A simulation "pass" can
therefore be the model collapsed to a safe-looking default rather than the model perceiving, and the
output-side and uncertainty signals a safety case would trust are exactly the ones that stay silent.

The signal those outputs hide is recoverable from the model's own recurrent feature with a single
second-order statistic, the rolling temporal spread of the 512-D state: one O(d) quantity per forward pass,
with no retraining and no architecture change. Calibrated leave-one-corpus-out to about a 1% real-driving
FPR (N=2, a two-fold estimate), it separates the collapse at AUROC 0.996 and fires about 0.23 blend-units
before the output cliff on the Subaru source, where the location-based feature scores one would default to
(Mahalanobis, relative Mahalanobis, KNN) each hit 100% leave-one-corpus-out FPR and fail to transfer across
the two real corpora.

Output-side monitoring alone is insufficient for the safety case of this shipped driving model, and a
second-order recurrent-state monitor is a cheap complement. This is a single-model, collapse-specific,
offline-only result: an ImageNet-C sweep shows the silent collapse is sim-specific and the monitor is
collapse-specific, and the present evidence does not support generalization to other models, OOD axes, or
deployment contexts. Those are the next studies, not this one's claims.

---

## 8. Reproducibility Note

The headline analyses rerun from committed result caches on a fresh public clone without a GPU or CARLA.
The committed caches are `report/teardown_collected.npz` (E1, E2, E3), `report/e4_collected.npz` (the E4
Subaru cliff), `report/baselines_collected.npz` (the Mahalanobis, RMD, and KNN-50 baseline scores), and
`report/metrics_collected.npz` (the E6 AUROC/AUPR/FPR bootstrap table); the ablations regenerate from those
caches as well. The bootstrap is pinned (n=1000, seed=42) and the run environment is pinned in
`requirements.txt`. The parity test reruns from the released openpilot v0.9.7 ONNX and comma's logged
reference, which are fetched separately and are not redistributed in the repo.

Three supporting result caches are not committed to the public repo because of their size, and the
experiments that read them therefore do not rerun from a fresh clone: the E5 submodule cache
(`report/e5_submodule_collected.npz`, about 98 MB), the E7 corruption cache
(`report/e7_collected.npz`, about 110 MB), and the E4-RAM cache
(`report/e4_ram_collected.npz`, about 28 MB). For each, a regeneration path is documented: the corresponding
`--collect` pass regenerates the cache from the released ONNX, the source frames, and a GPU, after which the
analysis reruns as for the committed experiments. The committed result files (`report/e5_submodule_results.md`,
`report/e7_results.md`, `report/e7_overlay_results.md`, `report/e4_ram_results.md`) and the committed figures
record the outputs of prior successful collection passes and are readable as-is. The large E5 layer-collection
file (`report/e5_collected.npz`, about 3.9 GB) is not committed and is additionally corrupt on the author's
machine; it will be replaced by a small committed summary array (per-stage, per-alpha activity ratios), which
is all the layer analysis reads. The cache-distribution decision is therefore settled: the four headline caches
are committed and reproduce from a fresh clone, and the three smaller supporting caches are regenerated via the
documented `--collect` passes rather than shipped in the public repo.

---

## Figure and Table Manifest

- Figure 1 (hero.png): four-panel overview (output collapse E1, uncertainty silence E3, Subaru cliff E4,
  monitor detection E6).
- Figure E1 (e1_head_collapse.png): per-head CARLA-to-real activity ratio, 8 collapsed, 2 alive.
- Figure E2 (e2_feature_ood.png): projected 512-D feature space, real clusters versus the CARLA freeze point.
- Figure E3 (e3_confidence.png): uncertainty distributions, real versus CARLA, against the real p95.
- Figure E4 (e4_interpolation.png + e4_ram_interpolation.png): Subaru cliff and RAM gradient.
- Figure E5a (e5_layer_localization.png): per-stage activity ratio, encoder at or above real.
- Figure E5b (e5_submodule_localization.png): per-submodule cliff-alpha, summarizer and action block.
- Table E6 (detector comparison): AUROC, AUPR, FPR@95TPR, and LOCO FPR for all five detectors at alpha=1.0
  with 95% CIs.
- Figure E6a (e6_detector.png): monitor fire rate versus alpha, firing before the cliff.
- Figure E6b (auroc_vs_alpha.png): AUROC versus alpha for all five detectors.
- Figure E6c (roc_curves.png + pr_curves.png): ROC and PR curves at alpha=1.0 (appendix candidate).
- Figure E7a (e7_auroc_heatmap.png): 15x5 monitor AUROC heatmap across corruption-severity cells.
- Figure E7b (e7_severity_sweep.png): monitor fire rate versus severity per corruption family.
- Figure E7c (e7_overlay.png): cell-for-cell output-collapse count versus monitor AUROC.
- Table RW (competitor contrast): five-axis comparison of the named neighbors and this work.

===== paper_state/contribution_contract.md (boundary reference) =====

# Contribution Contract

Locked by paper-stanford-contribution-locker on 2026-05-30. This file is the
single source of truth for what this paper may and may not claim. Every later agent
(framer, researcher, drafter, every gate) is bound by it. The boundary below may only
be widened by an explicit re-lock routed back to this agent, with the change recorded.

Lock note: the prior file was a bare TODO stub (no real prior contribution), so this is
a first lock, not a revision of meaningful content. The contribution itself was supplied
pre-decided by the orchestrator and is locked here verbatim in intent, narrowed to one
sentence and bounded against the verified result artifacts. It was NOT expanded.

## Contribution (one sentence)

A controlled distribution-shift teardown of a single shipped production driving model
(openpilot v0.9.7 supercombo) shows it fails silently under visual out-of-distribution
input (simultaneous output-head collapse, recurrent-feature freeze, and a non-responsive
uncertainty channel), localizes the failure downstream of the vision encoder, and
demonstrates that a zero-retraining recurrent-feature monitor (E6, the rolling temporal
spread of the 512-D state) detects the collapse about 0.23 blend-units before the outputs
cliff and at about 1% real-driving false-positive rate, where location-based feature-space
OOD scores (Mahalanobis, Relative Mahalanobis, KNN) fail to transfer across real corpora.

## Target venue and tier

- Venue: SafeAI @ UAI 2026 workshop (confirmed open) as primary; arXiv preprint
  (CoRL/RSS-style formatting) as the immediate deliverable.
- Tier: workshop.
- Why this tier: the evidence is a single-model (N=1: supercombo v0.9.7) negative finding
  plus a monitor demonstrated offline, with an N=2 real-corpus calibration estimate and
  one extreme OOD axis (CARLA) plus one bounding axis (ImageNet-C); that is exactly a
  strong workshop / preprint contribution, not a main-track generalization result. The
  evidence strength caps the tier; ambition does not raise it.

## Claim boundary

### This paper IS allowed to claim

- A parity-exact reimplementation of openpilot v0.9.7 supercombo inference, verified to
  within +/-0.5 m/s^2 of comma's reference output on 100% of 1159 real-footage frames,
  median absolute delta 0.04 m/s^2 (report/teardown_results.md context; paper_draft.md
  Section 4.1). This is the load-bearing harness-trust claim for the negative result.
- The silent-collapse phenomenon ON THIS ONE MODEL under CARLA-rendered input: 8 of 10
  output heads collapse to under 1% of real-driving temporal activity, the 512-D recurrent
  feature spread drops to about 1e-5 of real, and the predictive-uncertainty heads rise
  only 1.20x to 1.84x with 0% of OOD frames (0 of 220) exceeding the real-driving p95
  (E1/E2/E3, report/teardown_results.md).
- The collapse is a hard cliff on the Subaru alpha-blend axis (E4, transition width 0.015,
  output activity falling 0.9x to 0.1x of real over alpha 0.784 to 0.799), with the cliff
  path being segment-dependent: on the RAM source it is a gradient (width 0.274), so cliff
  headroom cannot be assumed to generalize across segments (E4-RAM, paper_plan.md Section 5).
- The localization: the collapse is downstream of the vision encoder (every encoder stage
  stays at or above real activity across the full sweep), entering at the recurrent
  summarizer VAE-mu bottleneck and the action-block feedback path (E5,
  paper_draft.md Section 5 E5). State this as PARTIAL localization (a VAE-mu/sigma
  ambiguity remains, per report/e5_submodule_results.md).
- E6 detection with early warning: the rolling-spread monitor fires (>50% of frames) at
  alpha=0.550, about 0.23 blend-units before the E4 output cliff at alpha~0.784, with
  AUROC 0.996 [0.992, 1.000] at alpha=1.0 (report/e6_results.md, report/metrics_results.md).
- E6 cross-corpus calibration where the location-based baselines do not transfer: E6 holds
  LOCO mean FPR 1.03% (max 2.07%) across {subaru, ram}, while Mahalanobis, Relative
  Mahalanobis, and KNN-50 each hit 100% LOCO FPR, and PCA-Mahalanobis only reaches 11.91%
  LOCO FPR (still far above 1%) (report/e6_results.md, report/metrics_results.md). State
  this as an N=2 two-fold estimate, not a production FPR.
- The two-way bounded corruption result (E7 + E7 overlay): the silent collapse is
  SIM-SPECIFIC, no ImageNet-C corruption reproduces it (at most 1 of 10 output heads
  collapses on any of 75 corruption-severity cells, versus 7 of 10 under CARLA), and E6 is
  COLLAPSE-SPECIFIC (correctly quiet on the real-frame corruptions the model tolerates,
  with its few firings, frost sev3/sev5, gaussian-noise sev4, impulse-noise sev5, tracking
  a recurrent-feature-spread shift rather than any output collapse; 0 false negatives
  because there is no collapse to miss) (report/e7_results.md, report/e7_overlay_results.md).
- That phantom braking under distribution shift is a known, user-reported failure mode of
  the shipped model, cited ONLY as motivation (commaai issue #20704 / discussion #22212).

### This paper is NOT allowed to claim

- Any generalization of the silent-collapse phenomenon or of E6 beyond supercombo v0.9.7
  (N=1 model; no other openpilot version, Tesla, Mobileye, Waymo, or research IL stack
  was tested).
- A statistically meaningful or production-grade false-positive rate for E6 (N=2 real
  corpora; LOCO is a two-fold estimate whose variance is not meaningfully reportable).
- That E6 is a universal or general-purpose OOD detector (E7 shows it is collapse-specific;
  it is near chance on most photometric corruptions).
- That E6 generalizes to non-CARLA corruptions AS A COLLAPSE DETECTOR (no non-CARLA output
  collapse was ever induced, so there is no non-CARLA collapse for E6 to have caught).
- Any on-road, in-stack, or real-robot deployment of E6 (it is demonstrated offline on
  logged, rendered, and corrupted frames only).
- Any causal claim linking this collapse to specific openpilot field incidents beyond
  citing the user-reported phantom-braking issue as motivation.
- That the localization is complete or fully mechanistic (it is partial; a VAE-mu/sigma
  ambiguity remains).

## Do NOT claim (exclusion list)

These are the exact tempting sentences a careless drafter would write. Do not write them.

- "We show that production driving models fail silently under distribution shift."
  (Wrong: N=1, this is supercombo v0.9.7 only; say "a production driving model" and name it.)
- "E6 is a general-purpose / universal OOD detector for driving models."
  (Wrong: E6 is collapse-specific; near chance on most ImageNet-C corruptions, mean AUROC
  0.52 to 0.74 on the photometric/blur families.)
- "E6 generalizes beyond CARLA, detecting OOD on real-world corruptions."
  (Wrong: E6's few corruption firings are decoupled from output collapse; no non-CARLA
  collapse was induced, so there is nothing for E6 to have generalized to as a collapse
  detector.)
- "E6 achieves a 1% false-positive rate in production driving."
  (Wrong: 1.03% is a LOCO two-fold estimate over N=2 corpora, not a production FPR.)
- "E6 beats / outperforms standard OOD baselines."
  (Careful: KNN-50 ties E6 at AUROC 1.000 at alpha=1.0; the real, narrower claim is that
  the location-based baselines FAIL TO TRANSFER across corpora (100% LOCO FPR) while E6
  calibrates. Frame it as transfer/calibration, not raw separation.)
- "We achieve state-of-the-art OOD detection."
  (Wrong: there is no SOTA leaderboard or benchmark ranking here; this is a single-model
  diagnostic with a calibration comparison.)
- "Our monitor can be deployed on the vehicle to prevent phantom braking."
  (Wrong: E6 is offline-only; no on-road, in-stack, or real-time deployment was run, and
  no causal link to field incidents was established.)
- "This explains / is the cause of openpilot's reported phantom braking."
  (Wrong: the field issue is motivation only; no causal claim is supported.)
- "The model fails to perceive simulated scenes." / "The vision encoder fails on sim."
  (Wrong: E5 localizes the collapse DOWNSTREAM of the encoder; the encoder stages stay at
  or above real activity. The failure is in the summarizer/action-block, not perception.)
- "The collapse is a universal cliff." (Wrong: cliff vs gradient is segment-dependent;
  RAM is a gradient with width 0.274 and no early-warning headroom.)

## Out-of-scope contributions (parked, not this paper)

- A second, real adverse-weather OOD axis (rain/night/glare comma footage) that actually
  induces a non-CARLA output collapse. The brief and plan flag this as the most
  reviewer-resistant follow-up; it is curation-gated and deferred to an extended version.
  (paper_plan.md Section 3.2, Option B.)
- A MetaDrive (non-CARLA sim) collapse axis. Deferred and de-prioritized due to the known
  bridge confound (commaai issue #31711). (paper_plan.md Section 3.1, Option A.)
- A multi-model / cross-version study (other openpilot versions, other vendors, research IL
  stacks) to test whether silent collapse generalizes. This is the N>1 paper, explicitly
  not this one.
- A third+ real corpus to turn the LOCO estimate into a reportable production FPR with
  variance. Parked; needed before any production-FPR claim.
- An online / in-stack deployment and evaluation of E6 inside the running openpilot stack.
  Parked; this paper is offline-only.
- A complete mechanistic account of the VAE-mu/sigma summarizer bottleneck (resolving the
  remaining localization ambiguity). Parked as future work.

## Closest neighbors (filled by paper-stanford-framer)

- (placeholder: the framer appends named neighbors + deltas here)

===== paper_state/claim_ledger.md (51/51 CONFIRMED) =====

# Claim Ledger

| id | claim | type | evidence | quote_or_number | status |
|---|---|---|---|---|---|
| c1 | Parity-exact reimplementation of openpilot v0.9.7 supercombo inference matches comma's reference longitudinal-acceleration output within +/-0.5 m/s^2 on 100.00% of 1159 real-footage frames | quantitative | report/parity_results.md lines 15-16 | 1159 frames, 100.00% within +/-0.5 m/s^2 | CONFIRMED |
| c2 | Parity median absolute delta is 0.0409 m/s^2 (mean 0.0541, p95 0.1511, max 0.2899; no frame exceeds 0.5) | quantitative | report/parity_results.md lines 17-19 | median 0.0409, mean 0.0541, max 0.2899 m/s^2; recomputed from artifact | CONFIRMED |
| c3 | Parity acceptance criterion is |delta| <= 0.5 m/s^2 on >= 95% of frames after a 40-frame (2 s at 20 Hz) warm-up trim, on the accel_t0 head vs comma's logged modelV2 reference | factual | report/parity_results.md lines 7-12 | acceptance criterion text confirmed verbatim in artifact | CONFIRMED |
| c4 | Correct supercombo inference requires recurrent-state threading by shift-and-append (features_buffer and prev_desired_curv), zero-init only on frame 1; a per-frame zero reset produces a multi-second init transient that looks like a spurious deceleration | factual | report/parity_results.md + src/state.py | correct recurrent-state threading: shift-and-append, zero-init frame 1 only | CONFIRMED |
| c5 | supercombo consumes unnormalized uint8 YUV in 0..255 (loadyuv kernel does convert_float8 with no scaling); dividing by 255 degrades parity | factual | report/parity_results.md + src/preprocessor.py | unnormalized uint8 YUV 0..255; dividing by 255 degrades parity | CONFIRMED |
| c6 | On CARLA-rendered clean input, 8 of 10 supercombo output heads collapse to under 1% of real-driving temporal activity | quantitative | report/teardown_results.md E1 table + teardown_collected.npz recomputation | 8 of 10 heads < 0.01 ratio; all 8 ratios match exactly: desired_curv=0.0018 accel_t0=0.0040 etc | CONFIRMED |
| c7 | The two surviving heads are pose (CARLA/real activity 0.1788) and meta (0.7181) | quantitative | report/teardown_results.md E1 table + teardown_collected.npz | pose=0.1788 meta=0.7181 recomputed exactly | CONFIRMED |
| c8 | On CARLA the 512-D recurrent feature spread (trace of hidden_state covariance) falls to about 1e-5 (0.00001x) of the real spread: the recurrent state freezes | quantitative | report/teardown_results.md E2 + teardown_collected.npz | recomputed ratio=1.29e-5; reported as 0.00001x (1e-5 order of magnitude MATCH) | CONFIRMED |
| c9 | Real-vs-CARLA linear separability of the recurrent state is 87.9% (d' = 2.19) along the centroid-difference direction | quantitative | report/teardown_results.md E2 + teardown_collected.npz + src/teardown.py | separability=87.9% d-prime=2.19 recomputed exactly from e2_feature_ood algorithm | CONFIRMED |
| c10 | Under collapse, predictive-uncertainty heads rise only 1.20x to 1.84x (plan 1.35x, lead 1.20x, desired_curv 1.84x) | quantitative | report/teardown_results.md E3 table + teardown_collected.npz | plan=1.35x lead=1.20x desired_curv=1.84x recomputed exactly | CONFIRMED |
| c11 | 0 of 219 CARLA frames exceeds the real-driving 95th-percentile uncertainty of plan, lead, or desired_curv | quantitative | report/teardown_results.md E3 table + teardown_collected.npz recomputation | 0/219 above p95 (recomputed); draft corrected 220 to 219 in rewrite stage 11 | CONFIRMED |
| c12 | Under collapse the plan/lead/desired_curv heads retain only 0.6%/0.4%/0.2% of their real output activity (about 99.5% lost) | quantitative | report/teardown_results.md E3 table + teardown_collected.npz | plan=0.6% lead=0.4% desired_curv=0.2% recomputed from npz activity ratios | CONFIRMED |
| c13 | On the Subaru source the collapse is a hard cliff: output activity falls from 0.9x to 0.1x of real over alpha 0.784 to 0.799, transition width 0.015 | quantitative | report/e4_results.md Verdict line + per-alpha table | a90=0.784 a10=0.799 width=0.015 confirmed; cliff fully in bin alpha=0.775->0.800 | CONFIRMED |
| c14 | On the Subaru sweep output activity first balloons to 6.32x of the real baseline at alpha=0.425 before collapsing | quantitative | report/e4_results.md per-alpha table alpha=0.4250 | activity=6.3161 rounds to 6.32x; recomputed from table | CONFIRMED |
| c15 | On the Subaru sweep the recurrent feature spread crashes from 0.25 to 0.00 by about alpha=0.78, and predictive uncertainty never spikes through the transition | quantitative | report/e4_results.md feature_spread column | spread=0.25 at alpha=0, spread=0.00 at alpha=0.775; plan uncertainty flat ~0.55 through cliff | CONFIRMED |
| c16 | The cliff shape is segment-dependent: on the RAM source the collapse is a gradient of transition width 0.274 (alpha 0.666 to 0.940), not a cliff | quantitative | report/e4_ram_results.md Verdict line + comparison table | a90=0.666 a10=0.940 width=0.274 confirmed; 0.940-0.666=0.274 | CONFIRMED |
| c17 | On the RAM source the monitor fires at alpha=0.850, inside the transition band, so its early-warning headroom is negative (-0.184) vs +0.234 on Subaru; cliff headroom does not generalize across segments | quantitative | report/e4_ram_results.md comparison table | Subaru headroom=0.784-0.550=0.234; RAM headroom=0.666-0.850=-0.184 | CONFIRMED |
| c18 | Every vision-encoder stage stays at or above real activity across the full sweep (stem 1.43x, stage3 2.06x, head 2.14x at alpha=1; minimum about 0.96); no encoder stage crosses the 0.5 collapse threshold | quantitative | report/e5_results.md table | head=2.1416 stage3=2.0561 stem=1.4254 stage1(min)=0.9560; all cliff_alpha=NaN confirmed | CONFIRMED |
| c19 | The collapse is downstream of the vision encoder, in the recurrent summarizer and action block, not in perception | novelty | report/e5_results.md + report/e5_submodule_results.md | encoder stages all above 0.5 activity; summarizer_div and action_block_body have cliff; structural claim confirmed | CONFIRMED |
| c20 | The collapse enters at the recurrent summarizer VAE-mu bottleneck (summarizer_div, cliff alpha 0.900, mean shift 0.023 at alpha=1), which is the hidden_state the monitor reads | quantitative | report/e5_submodule_results.md per-probe table | summarizer_div cliff_alpha=0.900 (first below 0.5 at alpha=0.9 where activity=0.298); mean_shift=0.0233=0.023 | CONFIRMED |
| c21 | The action-block body (action_block_body) cliffs a full alpha step earlier, at alpha=0.500, driven by the prev_desired_curv recurrent feedback loop | quantitative | report/e5_submodule_results.md per-probe table | action_block_body cliff_alpha=0.500; activity=0.661 at alpha=0.4 then 0.281 at alpha=0.5 (first below 0.5) | CONFIRMED |
| c22 | The transformer attention, FFN, and reduce-sum stages are passive relays (track summarizer to within 2 to 11%); vision_post (1.89x) and hydra_trunk (2.71x) at alpha=1 show no cliff | quantitative | report/e5_submodule_results.md per-probe table + recomputation | vision_post=1.89x, hydra_trunk=2.71x MATCH; passive-relay range corrected to 2 to 11% (reduce_sum 11.4% at alpha=1) in rewrite stage 11 | CONFIRMED |
| c23 | E5 localization is partial: a VAE mu-vs-sigma ambiguity in the summarizer reparameterization (Div by sigma) remains unresolved | factual | report/e5_submodule_results.md Caveat section | VAE mu/sigma ambiguity text confirmed verbatim in artifact | CONFIRMED |
| c24 | The rolling-spread monitor reaches AUROC 0.996 [0.992, 1.000], AUPR 0.995 [0.990, 1.000], FPR@95TPR 0.000 at alpha=1.0 | quantitative | report/metrics_results.md Table 1 + metrics_collected.npz + bootstrap recomputation | AUROC=0.9963 CI=[0.9915,0.9999] rounds to 0.996 [0.992,1.000]; AUPR=0.995 [0.990,1.000]; FPR95=0.000 | CONFIRMED |
| c25 | KNN-50 ties the monitor on single-corpus separation at AUROC 1.000 [1.000, 1.000] (no claim that the monitor beats KNN-50) | quantitative | report/metrics_results.md Table 1 + metrics_collected.npz | KNN-50 AUROC=1.000 [1.000,1.000] recomputed from npz | CONFIRMED |
| c26 | Mahalanobis (AUROC 0.159 [0.130, 0.190]) and PCA-Mahalanobis (0.152 [0.124, 0.179]) score below chance at alpha=1.0; relative Mahalanobis reaches 0.934 [0.914, 0.952] | quantitative | report/metrics_results.md Table 1 + metrics_collected.npz | Maha=0.159 [0.130,0.190]; RMD=0.934 [0.914,0.952]; PCA=0.152 [0.124,0.179]; all recomputed from npz | CONFIRMED |
| c27 | Mahalanobis scores below chance because the recurrent state collapses to the center of the ID Gaussian (distance-from-mean cannot detect collapse-to-the-mean), so OOD frames get lower distance than real frames | factual | report/metrics_results.md Headline + E2 recomputation | freeze-to-center mechanism confirmed; CARLA state collapses to near-constant verified from npz | CONFIRMED |
| c28 | Under leave-one-corpus-out across {subaru, ram} the monitor holds LOCO mean FPR 1.03% (max 2.07%); this is an N=2 two-fold estimate, not a production FPR | quantitative | report/e6_results.md LOCO table | LOCO FPRs=[0.0000,0.0207]; mean=(0+0.0207)/2=0.01035=1.03%; max=2.07%; arithmetic verified | CONFIRMED |
| c29 | All three applicable location-based baselines (Mahalanobis, relative Mahalanobis, KNN-50) hit 100% LOCO FPR; PCA-Mahalanobis improves to 11.91% LOCO mean FPR (max 23.82%) but stays an order of magnitude above 1% | quantitative | report/metrics_results.md PCA-Maha LOCO table + Headline | PCA-Maha LOCO: (0+0.2382)/2=0.1191=11.91% max=23.82%; KNN/Maha/RMD 100% LOCO confirmed in Headline | CONFIRMED |
| c30 | The location-based scores fail to transfer because subaru and ram occupy disjoint 512-D feature regions whose inter-corpus separation dwarfs the within-corpus radius; the monitor's location-invariant second-order trace both separates and calibrates | factual | report/metrics_results.md Headline | disjoint corpus feature-space explanation confirmed verbatim in artifact | CONFIRMED |
| c31 | The monitor fires (>50% of frames) at alpha=0.550, about 0.23 blend-units before the Subaru output cliff at alpha~0.784 | quantitative | report/e6_results.md alpha sweep + metrics_collected.npz | Fire fraction crosses 0.5 at alpha=0.550 CONFIRMED; gap 0.784-0.550=0.234. False "AUROC crosses at same alpha" clause REMOVED in rewrite stage 11 (AUROC actually crosses between 0.425 and 0.450). | CONFIRMED |
| c32 | The threshold-free eval split is ID = subaru+ram (n=638 stored, 609 valid non-NaN) vs OOD = E4 alpha=1.0 CARLA (n=319 stored, 290 valid), bootstrap n=1000 seed=42 | factual | report/metrics_results.md header + metrics_collected.npz meta keys | stored-vs-valid counts (609/290) clarified in rewrite stage 11; bootstrap n=1000 seed=42 CONFIRMED from meta keys | CONFIRMED |
| c33 | No ImageNet-C corruption reproduces the output collapse: 0 of 75 corruption-severity cells reach the collapse criterion (>=5/10 heads), max 1 of 10 heads on any cell, vs 7 of 10 under CARLA | quantitative | report/e7_overlay_results.md header lines | 0/75 cells collapse; max 1/10 heads (frost sev3); vs 7/10 CARLA; 15x5=75 cells | CONFIRMED |
| c34 | There are zero false negatives in the corruption sweep because no output collapse exists for the monitor to miss | factual | report/e7_overlay_results.md | FALSE NEGATIVES=0 confirmed; no collapse in any corruption cell | CONFIRMED |
| c35 | The monitor is near chance on most corruptions: mean per-corruption AUROC fog 0.55, snow 0.55, jpeg 0.52, brightness 0.60, zoom_blur 0.58, blur families ~0.68-0.71 | quantitative | report/e7_results.md summary table (mean AUROC) | fog=0.5461 snow=0.5455 jpeg=0.5218 brightness=0.6008 zoom_blur=0.5751 defocus=0.7078 motion=0.7059 glass=0.6845; all match draft at 2dp | CONFIRMED |
| c36 | The monitor fires on only 4 of 75 cells (frost sev3 AUROC 0.958, frost sev5 1.000, gaussian_noise sev4 0.861, impulse_noise sev5 0.906), all with zero output collapse, so its firings track a feature-spread shift, not the collapse mode | quantitative | report/e7_overlay_results.md FP line + report/e7_results.md threshold-free table | 4 FP cells: frost3=0.9582=0.958; frost5=0.9997=1.000; gaussian4=0.8608=0.861; impulse5=0.9060=0.906; all match draft at 3dp | CONFIRMED |
| c37 | The silent collapse is sim-specific (a property of full-sim rendering) and the monitor is collapse-specific, not a universal OOD detector | novelty | report/e7_overlay_results.md + report/e7_results.md | collapse sim-specific: 0/75 cells collapse under ImageNet-C; monitor collapse-specific: near-chance on most corruptions | CONFIRMED |
| c38 | Under corruption the location-based baselines fire on many corruptions at calibrated thresholds but carry 100% LOCO FPR, so their fire rates are not calibrated detection; KNN-50 fires only on heaviest noise/frost | quantitative | report/e7_results.md detection-rate table | Maha/RMD fire at 1.000 on fog/frost/noise/snow/contrast/zoom; KNN fires only on heavy noise+frost; qualitative match confirmed | CONFIRMED |
| c39 | Phantom braking under distribution shift is a known, user-reported failure mode of the shipped openpilot model (commaai issue #20704); cited as motivation only with no causal claim | factual (citation) | commaai/openpilot issue #20704 (research brief) | issue title "Large Shadow phantom braking"; body "abruptly brake even with no actual vehicle in front of you" | CONFIRMED |
| c40 | The correct openpilot falsification citation is von Stein and Elbaum, ASE 2022 (DOI 10.1145/3551349.3559500), not "Geretti et al., GPCE/SPLASH 2022" | factual (citation) | literature_map.md / research_brief.md correction | "von Stein and Elbaum, Finding Property Violations through Network Falsification ... ASE 2022 (Industry Showcase), DOI 10.1145/3551349.3559500" | CONFIRMED |
| c41 | EigenTrack (arXiv:2509.15735) is a parallel second-order/covariance hidden-activation OOD monitor with early warning on LLMs/VLMs (not a driving model), with a full eigenspectrum + trained classifier and no cross-corpus LOCO evaluation | factual (citation) | literature_map.md (C2) | "streaming covariance-spectrum statistics ... into a lightweight recurrent classifier"; "Large language models (LLMs) ..."; targeted search "don't contain ... applications to autonomous driving vehicle perception" | CONFIRMED |
| c42 | Novelty: this is the first second-order recurrent-state monitor on a shipped end-to-end driving model evaluated under cross-corpus leave-one-corpus-out transfer; NOT the first second-order hidden-activation OOD statistic (EigenTrack pre-dates that framing) | novelty | literature_map.md (Verdict + C2) + framing_memo.md | "first on the recurrent state of a SHIPPED end-to-end driver, evaluated under cross-corpus LOCO transfer, with a single location-invariant trace"; "first ... framing is FALSE and forbidden" | CONFIRMED |
| c43 | The closest AV-native neighbor Keser et al. 2025 (arXiv:2501.08083) monitors feature density of a frozen vision foundation-model encoder one stage upstream of this work's substrate | factual (citation) | literature_map.md / framing_memo.md | "Find a full model of the training data's feature distribution, to then use its density at new points as in-distribution (ID) score" | CONFIRMED |
| c44 | The next-closest neighbor Guo and Su 2026 (arXiv:2603.14603) monitors latent dynamics of a standalone trajectory predictor with provable QCD/MMD guarantees on delay and false alarm (venue/DOI not yet pinned) | factual (citation) | literature_map.md / framing_memo.md | "extend the cumulative Maximum Mean Discrepancy approach ... while still admitting provable guarantees on delay and false alarms"; "forward-numbered 2026 preprint with no final venue" | CONFIRMED |
| c45 | MSP, Energy, and ViM are structurally inapplicable to supercombo's multi-head Gaussian-mixture regression and existence-probability outputs (no softmax head, no logits, no classifier weight matrix) | factual | src/baselines.py + supercombo architecture (multi-head regression) | MSP/Energy no softmax; ViM no classifier weight matrix; structural exclusions confirmed | CONFIRMED |
| c46 | RMD's background distribution is fit as a 2-component GMM because with a single ID class the Ren et al. marginal Gaussian degenerates to the class Gaussian (RMD identically zero) | factual | src/baselines.py RMD background GMM implementation | single ID class -> RMD collapses to zero without GMM background; design decision confirmed | CONFIRMED |
| c47 | Chen et al. 2022 Openpilot-Deepdive (arXiv:2206.08176) is a static input/output/architecture teardown of supercombo that this paper extends to a runtime distribution-shift teardown | factual (citation) | research_brief.md / literature_map.md | "we deep-dive into Openpilot and conclude that its key to success is the end-to-end system design" | CONFIRMED |
| c48 | The monitor adds one O(d) statistic per forward pass (rolling temporal spread = trace of covariance of a 30-frame window of the 512-D state), threshold at the 1st percentile of the real-driving rolling-spread distribution (0.078873) | factual | report/e6_results.md threshold line + report/e7_results.md threshold line | threshold=0.078873 confirmed verbatim in both e6 and e7 results; Window=30 confirmed | CONFIRMED |
| c49 | Reproduce-from-cache works on a fresh clone for the committed caches (teardown/E1-E3, e4, metrics, baselines, ablations); the e5_submodule (98 MB), e7 (110 MB), and e4_ram (28 MB) caches are size-excluded and regenerate via documented --collect commands; the 3.9 GB e5_collected.npz is corrupt and to be replaced by a small committed summary | factual | paper_state/reproducibility_report.md (ran analyses + git ls-files / check-ignore) | committed-vs-ignored status and sizes verified by reproducibility_report stage 10; Section 8 corrected and [TODO] removed in rewrite stage 11 | CONFIRMED |
| c50 | The CARLA frames are rendered at matched intrinsics (comma 3 fcam resolution 1928x1208, fov for focal 2648px) so they carry exactly the _ar_ox_config.fcam intrinsics; the sim camera uses zero extrinsic calibration (device-mounted) so its warp reduces to the same intrinsic remap (K_fcam @ inv(K_medmodel)) the real path applies after liveCalibration. Intrinsics + model-input preprocessing identical to real; OOD shift confined to rendered content (Attack-1 control) | factual | src/sim_preprocessor.py (ZERO_CALIB=np.zeros(3) vs real liveCalibration) | "render it at the comma 3 fcam resolution (1928x1208) with the fov that reproduces fcam's focal length (2648 px), the rendered frame has exactly the intrinsics of _ar_ox_config.fcam"; "the warp collapses to K_fcam @ inv(K_medmodel) -- a pure intrinsic remap" | CONFIRMED |
| c51 | The DeepRoad line (DeepTest, DeepRoad, MarMot) uses metamorphic/generative test synthesis and tests for consistent behavior, implicitly treating the generated scene as a valid input the model should handle | citation | drafts/references.bib + WebFetch of arXiv:1708.08559/1802.02295/2310.07414 (paper_state/source_verification.md deeproad-line) | DeepTest "automatically detect erroneous behaviors of DNN-driven vehicles"; DeepRoad "generate large amounts of accurate driving scenes" via GANs; MarMot "Metamorphic Relations ... to estimate uncertainty ... at runtime" | CONFIRMED |

## Citation verification notes (paper-citation-verifier, 2026-05-30)

Set CONFIRMED per paper_state/source_verification.md "Summary of VERDICTS". Bib key in
parentheses; bib entry written to drafts/references.bib. Number-only rows untouched (stats
verifier owns those).

- c39 (commaai20704): CONFIRMED. GitHub issue #20704 "Large Shadow phantom braking" confirmed via search; cited as motivation only, not an academic source.
- c40 (vonstein2022): CONFIRMED. von Stein and Elbaum, ASE 2022, DOI 10.1145/3551349.3559500; authorship, venue, DOI, and falsification framing confirmed (ACM page 403'd, corroborated by search + authors' hosted PDF).
- c41 (eigentrack2025): CONFIRMED. arXiv:2509.15735; LLM/VLM substrate, covariance-spectrum + trained classifier, no driving-model / no LOCO application confirmed from abstract.
- c42 (eigentrack2025): CONFIRMED. Narrower novelty bound holds: EigenTrack pre-dates the second-order framing but not on a shipped driving model under cross-corpus LOCO. Same source as c41.
- c43 (keser2025): CONFIRMED. arXiv:2501.08083; feature density on a frozen vision foundation-model encoder, one stage upstream of this work's substrate.
- c44 (guosu2026): CONFIRMED (content). arXiv:2603.14603; standalone trajectory predictor, provable QCD/MMD guarantees on delay and false alarm. Venue unconfirmed, camera-ready flag (bib note + re-pin).
- c47 (chen2022deepdive): CONFIRMED. arXiv:2206.08176; static input/output/architecture teardown + reimplementation confirmed. Technical report; no proceedings venue confirmed (camera-ready flag).
