# Silent Collapse: A Distribution-Shift Teardown of a Production Driving Model and a Zero-Retraining Recurrent-State Monitor

> Target venue: arXiv preprint (CoRL/RSS-style), written to compress to a 4-page SafeAI-style
> workshop submission without restructuring. Single model under study: openpilot v0.9.7
> supercombo. Numbers are drawn verbatim from the committed `report/*.md` result files; every
> factual, quantitative, and novelty claim is registered in `paper_state/claim_ledger.md`.

---

## Abstract

Production Level-2 driver-assistance stacks are validated, in large part, in simulation, and
that practice rests on an unstated assumption: that the shipped model either behaves the same
on rendered input as on real input, or at least signals when it does not. We test that
assumption on one deployed model, openpilot v0.9.7 supercombo, the network that drives comma
hardware on public roads. We first build a parity-exact reimplementation of its inference path,
verified to within +/-0.5 m/s^2 of comma's own reference output on 100% of 1159 real-footage
frames (median absolute delta 0.0409 m/s^2), so that any downstream anomaly is the model, not
our harness. Running the verified model on CARLA-rendered clean roads, we find it fails
silently: 8 of 10 output heads collapse to under 1% of their real-driving temporal activity and
the 512-D recurrent feature freezes to about 1e-5 of its real spread, while the model's own
predictive-uncertainty heads rise only 1.20x to 1.84x and not one out-of-distribution frame
(0 of 220) crosses the real-driving 95th percentile. Nothing the model emits flags the
collapse. An alpha-blend sweep characterizes the collapse as a hard cliff on the Subaru source
(transition width 0.015), though the cliff shape is segment-dependent (a gradient of width
0.274 on the RAM source), and a layer-by-layer probe localizes it downstream of the vision
encoder, in the recurrent summarizer and action-block feedback path, not in perception. We then
show that the signal the outputs hide is recoverable from the model's own recurrent feature with
a single second-order statistic, the rolling temporal spread of the 512-D state: one O(d)
quantity per forward pass, no retraining, no architecture change. Calibrated leave-one-corpus-out
to about a 1% real-driving false-positive rate (N=2, a two-fold estimate, not a production FPR),
it separates the collapse at AUROC 0.996 [0.992, 1.000] and fires about 0.23 blend-units before
the outputs cliff, where the location-based feature scores one would default to (Mahalanobis,
Relative Mahalanobis, KNN) each hit 100% leave-one-corpus-out FPR and fail to transfer across the
two real corpora. This is a single-model negative finding with a collapse-specific monitor, not a
general OOD detector: an ImageNet-C sweep shows the silent collapse is sim-specific (no corruption
reproduces it; at most 1 of 10 heads collapses on any of 75 corruption-severity cells, versus 7
of 10 under CARLA) and the monitor is near chance on most photometric corruptions. The
contribution is that output-side and location-based signals alone are insufficient for this one
shipped model's safety case, and a second-order recurrent-state monitor is a cheap complement
that the present evidence does not claim generalizes.

---

## 1. Introduction

Every Level-2 and autonomous-driving program validates its driving policy, in large part, in
simulation, because simulation is the only setting in which rare and dangerous scenarios can be
exercised at scale and at low cost. The validity of that practice rests on an assumption that is
rarely stated and almost never tested directly: that the shipped model under test behaves on
rendered input the way it behaves on real input, or, failing that, that it fails loudly enough
for the test harness or a downstream safety monitor to notice. If a model can be shown a
simulated scene and quietly stop doing its job while every output it emits still looks
plausible, then a passing simulation is not evidence that the model works; it is evidence that
the model produced a safe-looking default, and the two are indistinguishable from the outside.

We test that assumption on a single deployed model, openpilot v0.9.7 supercombo, the end-to-end
network that drives comma hardware on public roads, and we find the dangerous answer. On
CARLA-rendered clean roads, 8 of 10 of the model's output heads collapse to under 1% of their
real-driving temporal activity, the 512-D recurrent state that threads the model's memory across
frames freezes to roughly 1e-5 of its real spread, and yet the model's own predictive-uncertainty
heads rise only 1.20x to 1.84x and not one out-of-distribution frame, 0 of 220, exceeds the
95th-percentile uncertainty the model exhibits in real driving. The failure is silent by the
model's own signals: the planning trajectory, the acceleration command, the lane and lead
geometry all go dark, but the channel a safety case would watch to catch exactly that event
stays quiet.

The failure mode this exposes is not hypothetical. Phantom braking under distribution shift, the
model commanding a deceleration for an obstacle that is not there, is a known and user-reported
failure of the shipped openpilot stack, documented in the project's own issue tracker (commaai
issue #20704). We cite that report as motivation only: we make no causal claim linking the silent
collapse we measure in simulation to any specific field braking event. The point it grounds is
narrower and sufficient. A simulation "pass" can be the model having collapsed to a safe-looking
default rather than the model perceiving the scene, and the output-side and uncertainty signals a
downstream safety case would trust are precisely the signals that stay silent when it does. That
is the gap.

The closest published monitors do not close it. The nearest AV-native neighbor watches the
feature density of a frozen perception encoder one stage upstream of where the collapse lives
(Keser et al. 2025), and the next-nearest watches the latent dynamics of a standalone trajectory
predictor with provable changepoint guarantees (Guo and Su 2026); neither targets the recurrent
state of a shipped end-to-end driver. A recent line in language and vision-language models,
EigenTrack (arXiv:2509.15735), does stream a second-order statistic of hidden activations through
a trained classifier with early warning, but on LLMs and VLMs, not on a driving model, and not
under cross-corpus transfer. And the standard location-based feature-space scores one would reach
for first (Mahalanobis, Relative Mahalanobis, KNN) each hit 100% leave-one-corpus-out
false-positive rate on this model: calibrated on one of our two real corpora, they flag the
entirety of the other.

This paper makes four contributions, each bounded to the single model under study. First, a
parity-exact reimplementation of openpilot v0.9.7 supercombo inference, verified to within
+/-0.5 m/s^2 of comma's reference output on 100% of 1159 real frames (median absolute delta
0.0409 m/s^2), including two non-obvious correctness details (recurrent-state threading and
unnormalized YUV input) that a faithful reimplementation must get right. Second, an empirical
demonstration of silent failure under visual distribution shift on this model: simultaneous
output-head collapse, recurrent-feature freeze, and a non-responsive uncertainty channel, with
0 of 220 OOD frames exceeding the model's real-driving uncertainty p95. Third, a characterization
of the collapse as a hard cliff on the Subaru source (transition width 0.015) whose shape is
segment-dependent (a gradient of width 0.274 on the RAM source), localized downstream of the
vision encoder in the recurrent summarizer and action-block feedback path (a partial
localization; an ambiguity in the summarizer's variational bottleneck remains). Fourth, a
zero-retraining monitor on the model's own recurrent feature, the rolling temporal spread of the
512-D state, calibrated leave-one-corpus-out to about a 1% real-driving FPR (N=2, a two-fold
estimate), which separates the collapse at AUROC 0.996 and fires about 0.23 blend-units before
the output cliff, where the location-based scores fail to transfer; bounded by an ImageNet-C
sweep that shows the collapse is sim-specific and the monitor collapse-specific, not a universal
OOD detector. Figure 1 previews all four findings.

---

## 2. Related Work

This contribution sits at the intersection of two ancestral lines. One is location-based
feature-space OOD detection, the family that this paper runs as baselines and shows fails to
transfer across corpora rather than fails to separate within one. The other is
internal-activation runtime monitoring, the family the proposed monitor belongs to, pushed onto
the recurrent state of a shipped driver. We name the neighbors a reviewer will invoke and state,
for each, exactly what it does not have that this work does. Table RW compresses the comparison
onto five axes.

**Location-based feature-space OOD (the baseline family).** The output-side floor is maximum
softmax probability (Hendrycks and Gimpel 2017) and its energy-based successor (Liu et al. 2020);
the feature-space ancestor is the Mahalanobis distance-from-fitted-Gaussian-mean score (Lee et al.
2018), refined for near-OOD by the relative Mahalanobis distance (Ren et al. 2021), and made
non-parametric by deep nearest-neighbor distance (Sun et al. 2022). ViM (Wang et al. 2022) is the
modern feature-residual-plus-logit hybrid, and Mahalanobis++ (Mueller and Hein 2025) keeps the
feature-Gaussian family live in 2025 with an l2-normalization fix, which is why running the Lee
2018 score here is a fair current comparison and not a strawman. OpenOOD (Yang et al. 2022)
codifies this whole line into one taxonomy and codebase; it is the vocabulary anchor for our
baselines, not a leaderboard this paper ranks on. We run Mahalanobis, relative Mahalanobis, and
KNN-50 as baselines on the same 512-D feature our monitor reads. The honest result, developed in
Section 5.6, is that KNN-50 ties our monitor on single-corpus separation (AUROC 1.000); the
distinguishing axis is cross-corpus calibration, where all three location-based scores hit 100%
leave-one-corpus-out FPR. MSP and Energy are output-side scores on a model whose output channel
Section 5.3 shows is silent, and ViM requires a classifier weight matrix that supercombo's
multi-head regression outputs do not provide; these three are structurally inapplicable and we
name and excuse them rather than omit them.

**AV-native OOD and uncertainty monitoring.** The closest published neighbor, Keser et al. 2025,
monitors the feature density of a frozen vision foundation-model encoder as an in-distribution
score; that is one stage upstream of our substrate and is exactly the location-based class our
baselines instantiate. The next-closest, Guo and Su 2026, monitors the latent dynamics of a
standalone trajectory predictor as a quickest-changepoint-detection problem with provable bounds
on detection delay and false alarm; it differs from this work in model class (a standalone
predictor, not a shipped end-to-end driver) and in evidence basis (a theoretical guarantee, not
an empirical calibration). Filos et al. 2020 is the canonical framing of distribution shift for
autonomous vehicles, and the SelfOracle line (Stocco et al. 2020) and its uncertainty-quantification
successor (Grewal, Tonella, and Stocco 2024) build misbehaviour predictors on the assumption that
the model's confidence signal is informative; Section 5.3 is the contrary finding for this model.
A 2025 position paper frames OOD detection explicitly as safety-case evidence (Hodge et al. 2025),
which is the niche this work speaks to. None of these targets the recurrent state of a shipped
end-to-end driver under cross-corpus transfer.

**Internal-activation and second-order monitors (the proposed monitor's family).** Runtime neuron
activation pattern monitoring (Cheng et al. 2018) is the ancestor of internal-state monitoring: it
stores binarized neuron patterns and compares them by Hamming distance, on feed-forward
classifiers, a first-order discrete per-frame check. The first move to a higher-order feature
statistic for OOD is the Gram-matrix method (Sastry and Oore 2020), the closest lineage point to a
second-order rather than distance-from-mean choice. The proposed monitor inherits the
higher-order-statistic idea from the latter and the internal-monitor idea from the former, but
applies them to a recurrent state (neither did) on a shipped production driving model (neither did),
and it is the statistic that catches the freeze mode, the recurrent spread crashing across the
cliff, that a per-frame pattern or Gram check on outputs would miss. The single closest paper on
mechanism is EigenTrack (arXiv:2509.15735), a parallel and near-simultaneous line that streams
covariance-spectrum statistics (eigenvalue gaps, spectral entropy, random-matrix-theory features)
of hidden activations through a trained recurrent classifier, with early warning, to flag
hallucination and OOD drift. It rhymes with this work on three counts (second-order, hidden-state,
fires before surface errors), and it differs on three checkable ones: substrate (LLMs and VLMs,
not a driving model), statistic (a full eigenspectrum fed to a trained classifier, not a single
location-invariant trace with a calibrated threshold), and evaluation (language OOD benchmarks,
not cross-corpus leave-one-corpus-out FPR on a recurrent driver state). We therefore do not claim
to be first to use a second-order hidden-activation statistic for OOD detection: EigenTrack
pre-dates this work on that framing. The available and defensible claim is narrower: this is the
first second-order recurrent-state monitor on a shipped end-to-end driving model evaluated under
cross-corpus leave-one-corpus-out transfer. NECO (Ben Ammar et al. 2024) exploits a
classification-head neural-collapse property that supercombo's regression heads do not have; we
name and excuse it rather than run it.

**openpilot and supercombo prior work.** The reference academic teardown of supercombo is
Openpilot-Deepdive (Chen et al. 2022), a static input, output, and architecture analysis plus a
reimplementation; this paper extends that static teardown to a runtime distribution-shift
teardown. The directed-falsification line on openpilot is von Stein and Elbaum (ASE 2022), which
generates adversarial inputs that violate stated properties; it is related testing work and
motivation, not a silent-collapse or recurrent-state-monitor study. A recent adversarial study
(arXiv:2505.11532) targets supercombo with deliberate perturbations and input-level defenses, a
different failure mode (adversarial attack, not sim-rendered silent collapse) with no
recurrent-state monitor.

**Simulation testing of driving DNNs and corruption robustness.** The DeepRoad line (DeepTest,
DeepRoad, MarMot) generates tests for driving networks on the premise that the simulator renders
in-distribution scenes; this paper inverts that premise by showing the simulated input can itself
be out of distribution to the model, which undercuts coverage claims built on sim-based testing.
ImageNet-C (Hendrycks and Dietterich 2019) is the corruption-robustness yardstick we use as the
bounding OOD axis in Section 5.7, with Cityscapes-C (Michaelis et al. 2019) as its AV extension,
and CARLA (Dosovitskiy et al. 2017) is the simulator that supplies the primary OOD axis.

---

## 3. Threat Model

A safety case for a shipped driving model leans on a small set of runtime defenses, and the
finding of this paper is that the specific failure mode it documents defeats exactly those
defenses, silently and simultaneously. We define the threat precisely, walk through why each
standard defense misses it, and position the proposed monitor as a cheap complementary layer
rather than a replacement for any of them.

**The threat.** A shipped driving model is deployed in a visually shifted context: a rendered
simulator, an unfamiliar geography, a degraded or unusual sensor condition. Under sufficient
shift, three things happen at once and without warning. The output heads collapse to a plausible,
nearly constant signal; the recurrent state that carries the model's temporal memory freezes; and
the model's own predictive-uncertainty channel does not rise. The danger is not that the model is
wrong, which a safety case expects and plans for, but that it is wrong while every signal a
monitor would read says it is fine.

**Why uncertainty-head monitoring misses it.** The most direct defense is to threshold the model's
own predictive uncertainty: if the model says it is unsure, intervene. On this model that defense
never fires. Under the collapse, predictive uncertainty rises only 1.20x to 1.84x across the
monitored heads, and 0 of 220 OOD frames exceeds the 95th-percentile uncertainty the model
exhibits in real driving (Section 5.3). A threshold calibrated on real driving, which is the only
honest way to set it, never trips. This is the direct empirical contradiction of the SelfOracle and
uncertainty-quantification line, which builds misbehaviour predictors on the premise that the
confidence signal carries the information.

**Why output-plausibility and temporal-jitter monitors miss it.** A second defense checks that the
model's outputs are physically plausible: bounded acceleration, feasible curvature, a sane plan. The
collapsed outputs are plausible by construction. The plan head retains 0.6% of its real activity and
the acceleration head 0.4%, so the model emits a smooth, near-constant trajectory that reads as a
benign, stationary scene; a plausibility check passes it. A third defense watches for temporal jitter
or output disagreement as a sign of instability. The freeze produces the opposite of jitter: a frozen
output has lower temporal variance than an active one, so a jitter monitor reads the collapse as
increased stability, the very thing it is built to reward. (These are structural arguments from the
measured activity ratios, not separate experiments.)

**Why same-architecture ensembles and input-quality checks miss it.** A fourth defense runs an
ensemble and flags disagreement. Section 5.5 localizes the collapse downstream of the vision encoder,
in a path every instance of the same architecture shares, so an ensemble of the same model collapses
together and agrees on the wrong answer. A fifth defense screens the input itself for quality. The
CARLA-clean renders are sharper and less noisy than real road footage, so an image-quality screen
rates the simulated input as good, not anomalous. (Structural arguments, not new experiments.)

**The complementary signal.** The defenses above all read the output side or the input side. The
finding that makes a complement possible is that the model's own recurrent features carry the
OOD signal its outputs do not surface. The proposed monitor (Section 5.6) is a complement, not a
replacement: it is one O(d) statistic computed from a forward pass that already runs, it requires
no retraining and no architecture change, and it is calibrated against a real-driving
false-positive rate rather than against simulated negatives. It is also bounded, and we state the
bounds here so they travel with the proposal: it is collapse-specific (Section 5.7), it is
demonstrated offline only, and its false-positive rate is an N=2 two-fold estimate, not a
production number.

---

## 4. Method

This section describes the parity-verified harness, the data, the metrics, the monitor design,
and the baseline set in enough detail to trust and reproduce the teardown. The load-bearing
content is the parity number and the design descriptions; all figures live in Section 5.

**Parity-exact reimplementation.** We reconstruct openpilot v0.9.7 supercombo inference from the
released ONNX model and comma's own reference files (modeld, the output parser, the YUV loader
kernel, and the model constants). Because the central result is a negative finding about a
production model, the harness must be trustworthy before any anomaly can be attributed to the
model rather than to the reimplementation, so we establish parity first.

**Recurrent-state threading.** supercombo is a temporal model: it consumes a buffer of recent
features and its own previous desired-curvature output, and it must roll that state forward by
shift-and-append after each inference, with a zero initialization only on the first frame. A naive
per-frame zero reset of the state produces a multi-second initialization transient that, in this
model, looks exactly like a spurious deceleration, a self-inflicted phantom brake of the
reimplementation rather than the model. Getting this right is a precondition for both parity and
for the collapse measurement, because every collapse and corruption sequence is re-run with the
same correct state handling.

**Unnormalized YUV input.** The model's input loader (loadyuv) converts the camera Y, U, and V
channels to float with no rescaling: the model consumes uint8 values in the range 0 to 255, not
values divided by 255. Dividing by 255, the reflexive normalization, shifts the entire input
distribution and silently degrades parity. We feed unnormalized uint8 YUV, matching the kernel.

**Parity result.** On 1159 real-footage frames (after a 40-frame, 2-second warm-up trim), our
reimplemented longitudinal-acceleration output matches comma's logged reference within
+/-0.5 m/s^2 on 100.00% of frames, with a median absolute delta of 0.0409 m/s^2, a mean of 0.0541,
and a worst-case single-frame delta of 0.2899 m/s^2 (no frame exceeds 0.5). This is the
load-bearing harness-trust claim for the negative result, and we report it prominently.

**Data.** The real in-distribution data are two comma corpora, denoted subaru and ram, of 320
frames each, with the first 100 frames discarded as warm-up, leaving 220 analysis frames per
corpus. The out-of-distribution data are CARLA-rendered clean-road frames from the openpilot
v0.9.7 simulation pipeline. The interpolation axis for the cliff analysis (Section 5.4) is a
pixel-space alpha-blend of a real sequence (Subaru or RAM) with the CARLA sequence, with alpha=0
the real frame and alpha=1 the CARLA frame, swept over 29 alpha values. We state the underlying
counts wherever a percentage appears: for example, the "0%" of Section 5.3 is 0 of 220 CARLA
frames. For the threshold-free metrics (Section 5.6), the in-distribution set is subaru and ram
concatenated (n=638) and the out-of-distribution set is the alpha=1.0 CARLA frames (n=319).

**Metrics.** Output activity is the sum of per-element temporal standard deviation over a window;
a head is "collapsed" when its CARLA-to-real activity ratio is small. Feature spread is the trace
of the recurrent-state covariance over a rolling window. Detection is scored threshold-free with
AUROC, AUPR, and FPR at 95% TPR, each with a stratified bootstrap 95% confidence interval (n=1000
iterations, seed 42). Calibration uses a leave-one-corpus-out (LOCO) protocol across the two real
corpora: a threshold is set on one corpus and its false-positive rate is read on the held-out
corpus, then the two folds are averaged. Because there are only two real corpora, every LOCO FPR
is a two-fold estimate whose variance is not meaningfully reportable, and we state it as such
every time it appears.

**The monitor (E6).** The monitored quantity is the rolling temporal spread of the 512-D recurrent
feature the model emits, computed as the trace of the covariance of a 30-frame window of the state.
The detection threshold is the 1st percentile of the real-driving rolling-spread distribution,
which targets a roughly 1% false-positive rate by construction, and it is calibrated leave-one-
corpus-out. The monitor adds one O(d) statistic per forward pass, with no retraining and no extra
heads, and its statistic is location-invariant (a second-order spread), which is the property that
distinguishes it from the location-based baseline family.

**Baselines and structural exclusions.** The three applicable post-hoc feature-space scores,
Mahalanobis, relative Mahalanobis (RMD), and KNN-50, are computed on the same 512-D feature the
monitor reads, with a PCA-Mahalanobis variant as an ablation. RMD's background distribution is fit
as a two-component Gaussian mixture, because with a single in-distribution class the Ren et al.
marginal Gaussian degenerates to the class Gaussian and the relative score collapses to zero. MSP,
Energy, and ViM are structurally inapplicable to supercombo's multi-head Gaussian-mixture
regression and existence-probability outputs (there is no softmax head, no logit vector, and no
classifier weight matrix), and we state this explicitly so reviewers read it as an exclusion with a
reason, not an omission.

---

## 5. Experiments and Results

The experiments unfold in the order the argument requires: parity establishes trust (Section 4),
then collapse establishes the phenomenon (E1), the freeze and the silent uncertainty channel
establish the gap (E2, E3), the cliff sweep characterizes the shape (E4), the layer probe
localizes the mechanism (E5), the monitor and baselines deliver the solution (E6), and the
corruption sweep bounds the scope (E7).

### 5.1 E1: Output collapse

We run the parity-verified model on CARLA-rendered clean roads and measure, for each of the 10
output heads, the ratio of its CARLA temporal activity to its real-driving temporal activity.
Eight of the ten heads collapse to under 1% of real activity: desired_curv (0.0018), accel_t0
(0.0040), lead (0.0042), desire_state (0.0049), lane_lines (0.0054), plan (0.0057), lead_prob
(0.0058), and road_edges (0.0076). These eight are the safety-critical driving signals, the
planning trajectory, the acceleration command, the lane and road-edge geometry, the lead-vehicle
detection, and the curvature command. Two heads survive: pose (0.1788) and meta (0.7181), the
ego-motion and meta-state outputs. The model's primary driving outputs are functionally inactive
on CARLA-rendered input while its ego-motion and meta outputs partially persist. Figure E1 plots
the per-head ratios.

### 5.2 E2: Recurrent-feature freeze

We measure the rolling covariance trace of the 512-D recurrent hidden state on real versus CARLA
frames. On CARLA the feature spread falls to about 1e-5 (0.00001x) of its real value: the
recurrent state freezes to a near-constant vector. Despite that freeze, the frozen vector is
linearly separable from the real-driving states at 87.9% along the centroid-difference direction
(d' = 2.19), which establishes that the recurrent state carries a strong out-of-distribution
signal even while the output heads are dark. This is the signal the monitor of Section 5.6 reads.
It also previews a mechanism that explains the Mahalanobis failure in Section 5.6: the frozen
vector lands near the high-density center of the in-distribution Gaussian, so a distance-from-mean
score reads it as in-distribution. Figure E2 shows the projected feature space with the CARLA
freeze point.

### 5.3 E3: Uncertainty silence

This is the safety-relevant centerpiece. For three representative heads we report, side by side,
the fraction of output activity retained on CARLA, the ratio of CARLA to real predictive
uncertainty, and the fraction of the 220 CARLA frames that exceed the real-driving 95th-percentile
uncertainty for that head. The plan head retains 0.6% of its activity while its uncertainty rises
1.35x; the lead head retains 0.4% at 1.20x; the desired_curv head retains 0.2% at 1.84x. In all
three cases, 0 of 220 CARLA frames exceeds the real-driving p95. The outputs lose roughly 99.5% of
their activity, but the uncertainty channel barely moves and never crosses the threshold a
real-calibrated monitor would set.

The implication is the gap this paper turns on. A safety monitor that thresholds the model's own
uncertainty, calibrated on real driving, never fires under the collapse, because nothing the model
emits flags it. We state this strictly as an empirical finding about supercombo v0.9.7 under CARLA
input, the uncertainty channel stays quiet, and we draw no causal line from it to any field
phantom-braking incident. Figure E3 plots the uncertainty distributions for real and CARLA frames
against the real-driving p95 line, making the silence visually unambiguous.

### 5.4 E4: Cliff characterization and segment dependence

To characterize the shape of the collapse, we sweep the pixel-space alpha-blend from the real frame
(alpha=0) toward the CARLA frame (alpha=1). On the Subaru source the response is two-phase. Output
activity first balloons to 6.32x of the real baseline at alpha=0.425, a thrash driven by the
ghosted-input interference of the half-blended frame, and then collapses through a hard cliff,
falling from 0.9x to 0.1x of real activity over the narrow alpha band 0.784 to 0.799, a transition
width of 0.015. The feature spread crashes from 0.25 to 0.00 by about alpha=0.78. Through the entire
transition the predictive uncertainty never spikes, consistent with E3.

The cliff shape is segment-dependent, and this is a bound on the early-warning headroom the monitor
exploits. On the RAM source the same sweep produces a gradient, not a cliff: output activity falls
from 0.9x to 0.1x of real over the wide alpha band 0.666 to 0.940, a transition width of 0.274. On
the RAM source the monitor's firing point (alpha=0.850) lands inside that band rather than before it,
so the early-warning headroom is negative (-0.184) where on Subaru it is positive (0.234). Cliff
headroom therefore cannot be assumed to generalize across segments; the characterization is partial.
We use the authoritative `report/e4_results.md` and `report/e4_ram_results.md` values (Subaru width
0.015, RAM width 0.274) throughout; where a figure legend rounds these (for example a Subaru legend
reading "0.02"), the results-file value governs. Figure E4 places the Subaru cliff and the RAM
gradient side by side.

### 5.5 E5: Localization downstream of the vision encoder

We first ask whether the collapse is the vision encoder failing, by measuring the CARLA-to-real
activity ratio of each encoder stage across the alpha sweep. It is not. Every encoder stage stays at
or above real activity across the full sweep: at alpha=1 the stem is at 1.43x, stage3 at 2.06x, and
the head at 2.14x, and the minimum ratio over all stages and all alpha is about 0.96. No encoder stage
ever crosses the 0.5 collapse threshold. The encoder is in fact more active on CARLA than on real
input, so the collapse seen at the output is not perception failing; it originates downstream. The
structural statement is explicit: the failure is in the summarizer and action block, not in the
encoder.

We then probe eight tensors between the summarizer and the per-head outputs. Two cross the 0.5
collapse threshold. The recurrent summarizer's variational bottleneck (summarizer_div, which is the
hidden_state the monitor reads) has a cliff at alpha=0.900, with its mean shifting to 0.023 of real,
a near two-order-of-magnitude collapse of the rolling mean of the 512-D vector; this is the entry
point of the collapse. The action-block body (action_block_body), the last residual block before the
curvature head, has a cliff a full alpha step earlier, at alpha=0.500, because it folds in the
model's own previous desired-curvature output through the recurrent feedback loop, so once that
already-collapsing signal joins, the action stack saturates fast. The transformer self-attention,
feed-forward, and reduce-sum stages track the summarizer to within about 2% and introduce no
additional collapse; they are passive relays. The post-encoder projection (vision_post, 1.89x at
alpha=1) and the non-temporal hydra trunk (hydra_trunk, 2.71x at alpha=1) show no cliff, consistent
with the two surviving heads of E1. The localization is partial: the summarizer ends in a
mu-over-sigma variational reparameterization, and we have not separated the mu path from the sigma
normalization, so part of the apparent summarizer collapse could be variance normalization rather
than information loss. We state this ambiguity rather than claim a complete mechanistic account.
Figures E5a and E5b show the per-stage and per-submodule ratios.

### 5.6 E6: Monitor detection and baseline comparison

We now test whether the signal the outputs hide is recoverable from the recurrent feature, and how a
second-order monitor of that feature compares to the location-based scores one would default to.

The threshold-free comparison is at alpha=1.0 (full CARLA shift), on the in-distribution set of
subaru and ram concatenated (n=638) against the CARLA OOD frames (n=319), with stratified bootstrap
95% confidence intervals. The rolling-spread monitor reaches AUROC 0.996 [0.992, 1.000], AUPR
0.995 [0.990, 1.000], and FPR at 95% TPR of 0.000. KNN-50 reaches AUROC 1.000 [1.000, 1.000].
We do not claim to beat KNN-50: on single-corpus separation the two tie. Relative Mahalanobis
reaches AUROC 0.934 [0.914, 0.952]. Mahalanobis (0.159 [0.130, 0.190]) and PCA-Mahalanobis
(0.152 [0.124, 0.179]) both score below chance. Table E6 collects these with their confidence
intervals.

Mahalanobis scores below chance for a mechanical reason worth stating, not just reporting. Under the
collapse the recurrent state freezes to a near-constant vector that lands near the center of the
in-distribution Gaussian (E2), so the OOD frames receive lower Mahalanobis distance than the real
frames: a distance-from-mean score cannot detect collapse-to-the-mean, and inverts. PCA-Mahalanobis
inherits the same inversion.

The distinguishing axis is not single-corpus separation but cross-corpus calibration, and here the
location-based scores fail to transfer. Under leave-one-corpus-out across subaru and ram, all three
applicable location-based baselines (Mahalanobis, relative Mahalanobis, KNN-50) hit 100% LOCO FPR:
a threshold calibrated on one corpus flags the entirety of the other. The PCA-Mahalanobis variant
improves to 11.91% LOCO mean FPR, a partial recovery but still an order of magnitude above the 1%
operating point. The rolling-spread monitor holds 1.03% LOCO mean FPR (max 2.07%), where we
emphasize that this is an N=2 two-fold estimate, not a production false-positive rate. The geometric
reason the location-based scores fail is that the subaru and ram corpora occupy disjoint regions of
the 512-D feature space whose inter-corpus separation dwarfs the within-corpus radius (visible as the
two real clusters in Figure E2), so any absolute-position score calibrated on one corpus flags the
whole of the other. The monitor reads the second-order trace, which is location-invariant, so it
both separates and calibrates across the corpus shift.

The monitor also fires before the outputs cliff. Sweeping alpha, the monitor's fired fraction crosses
50% at alpha=0.550, where the E4 Subaru output cliff is at about alpha=0.784, an early-warning gap of
about 0.23 blend-units; its AUROC crosses 0.5 at the same alpha=0.550, confirming the early signal is
genuine separation and not a calibration artifact. (As Section 5.4 notes, this headroom is
Subaru-specific; on the RAM gradient the firing point falls inside the transition band.) Figures E6a
and E6b show the fire rate and the AUROC trajectory across alpha, and Figure E6c gives the ROC and PR
curves at alpha=1.0.

The honest positioning against the nearest neighbors follows from these numbers. Against the
location-based family (Keser-style density, Lee Mahalanobis, Ren RMD, Sun KNN), the result is not
that the monitor separates better, since KNN-50 ties it, but that the location-based scores fail to
transfer across corpora (100% LOCO FPR) while the second-order monitor calibrates (about 1%, N=2).
Against EigenTrack (arXiv:2509.15735), the nearest mechanism-twin, the deltas are substrate (a
shipped driving model, not an LLM or VLM), statistic (a single location-invariant spread trace with
a calibrated threshold, not a full eigenspectrum fed to a trained classifier), and evaluation axis
(cross-corpus LOCO FPR on real-driving frames, not language OOD benchmarks). We frame the result as
transfer and calibration, not as outperforming baselines.

### 5.7 E7: Corruption sweep (two-way bound)

The final experiment bounds the contribution in both directions: it bounds the failure mode (is the
silent collapse a general model-robustness failure, or sim-specific?) and it bounds the monitor (is
the monitor a general OOD detector, or collapse-specific?). We apply the 15 ImageNet-C corruptions at
5 severities each to the real Subaru frames (in raw RGB, before YUV conversion), re-run supercombo with
the correct recurrent-state handling on each corrupted sequence, and evaluate the monitor and the
baselines, with a validation gate that first reproduces the published CARLA collapse (7 of 10 heads)
before any corruption result is read.

**The collapse is sim-specific.** Running the same E1 output-collapse metric on every corruption-
severity cell, no ImageNet-C corruption reproduces the output collapse: 0 of 75 cells reach the
collapse criterion (5 or more of 10 heads), and the maximum on any single cell is 1 of 10 heads,
against 7 of 10 under CARLA. The silent collapse is a property of full-sim rendering, not of
photometric or blur corruptions of real frames. One consequence is immediate: there are zero false
negatives, because there is no output collapse anywhere in the sweep for the monitor to miss, so its
quiet response on fog, brightness, blur, and the rest is correct.

**The monitor is collapse-specific.** On most corruptions the monitor sits near chance. Its mean
per-corruption AUROC is 0.55 on fog, 0.55 on snow, 0.52 on jpeg compression, 0.60 on brightness, and
0.58 on zoom blur, rising only to about 0.68 to 0.71 on the defocus, motion, and glass blur families,
all well below the operating regime it holds on CARLA. The monitor fires on only four of the 75 cells
at the calibrated threshold: frost severity 3 (AUROC 0.958), frost severity 5 (AUROC 1.000),
gaussian-noise severity 4 (AUROC 0.861), and impulse-noise severity 5 (AUROC 0.906). Crucially, all
four of those cells have zero output collapse, so on this corpus the monitor's firings track a
recurrent-feature-spread shift, not the collapse mode. The monitor is therefore correctly quiet on
the real-frame corruptions the model tolerates and is not a universal corruption detector.

**Baselines under corruption.** The location-based baselines fire on many corruptions at their
calibrated thresholds (Mahalanobis and relative Mahalanobis fire at high rates across fog, frost,
several noise families, snow, contrast, and zoom), but recall from Section 5.6 that both carry 100%
LOCO FPR: they also flag held-out real driving, so their high corruption fire rates are not calibrated
detection. KNN-50 fires only on the heaviest noise and frost. None of the baselines is calibrated to
the 1% operating point the monitor holds.

**Synthesis.** The corruption sweep narrows the contribution precisely. The silent collapse, the
dangerous mode, is sim-specific; the corruption sweep shows it is not a general model-robustness
failure, and the cell-for-cell overlay shows the monitor is not a general corruption detector. The
contribution is a targeted monitor for the specific, dangerous, sim-induced silent-collapse mode at
about a 1% real-driving FPR (N=2), where output-side and location-based feature scores do not. We do
not claim the monitor generalizes beyond CARLA as a collapse detector, because no non-CARLA output
collapse was induced for it to have caught. Figures E7a, E7b, and E7c give the AUROC heatmap, the
severity sweep, and the cell-for-cell collapse-versus-detection overlay.

---

## 6. Limitations

We state the boundaries that define where the evidence does and does not extend, so that the bounded
finding is not mistaken for a general result.

**N=1 model.** Every result is on supercombo v0.9.7 alone. No other openpilot version, and no Tesla,
Mobileye, Waymo, or research imitation-learning stack, was tested. The silent-collapse phenomenon and
the monitor are demonstrated on this one model, and we claim no generalization to any other. Whether
silent collapse is a property of this architecture, this training recipe, or end-to-end driving
models broadly is the N>1 study this paper does not attempt.

**N=2 corpora; LOCO is a two-fold estimate.** The cross-corpus calibration rests on two real corpora.
Leave-one-corpus-out at N=2 is a two-fold estimate whose variance is not meaningfully reportable, and
the roughly 1% FPR is therefore a calibration estimate, not a production false-positive rate. A third
real corpus, at minimum, is needed before any single production FPR can be quoted with an uncertainty.

**Monitor scope: collapse-specific and offline-only.** The corruption overlay (Section 5.7) shows the
monitor is collapse-specific and near chance on most real-frame corruptions, so it is not a universal
OOD detector. It is demonstrated offline on logged, rendered, and corrupted frames only; no on-road,
in-stack, or real-time deployment was run, and no causal link to field incidents was established. The
residual gap is that no non-CARLA output collapse was ever induced, so the monitor was never tested as
a collapse detector on anything other than CARLA; real adverse-weather footage (rain, night, glare)
that actually induces a non-CARLA collapse remains pending, and is the most reviewer-resistant
follow-up.

**Partial localization.** The collapse is pinned to the summarizer's variational bottleneck and the
action-block feedback path by ruling out the encoder and probing eight submodules, but the
mu-versus-sigma ambiguity inside the summarizer's reparameterization is unresolved, so the
localization is partial, not a complete mechanistic account.

---

## 7. Conclusion

A shipped Level-2 driving model, openpilot v0.9.7 supercombo, shown CARLA-rendered input, collapses
to a plausible near-constant and does not raise its own uncertainty: 8 of 10 output heads fall to
under 1% of real activity, the recurrent state freezes to about 1e-5 of its real spread, and 0 of 220
out-of-distribution frames exceeds the model's real-driving uncertainty p95. A simulation "pass" can
therefore be the model collapsed to a safe-looking default rather than the model perceiving, and the
output-side and uncertainty signals a safety case would trust are exactly the ones that stay silent.

The signal those outputs hide is recoverable from the model's own recurrent feature with a single
second-order statistic, the rolling temporal spread of the 512-D state: one O(d) quantity per forward
pass, no retraining, no architecture change. Calibrated leave-one-corpus-out to about a 1%
real-driving FPR (N=2, a two-fold estimate), it separates the collapse at AUROC 0.996 and fires about
0.23 blend-units before the output cliff on the Subaru source, where the location-based feature scores
one would default to (Mahalanobis, relative Mahalanobis, KNN) each hit 100% leave-one-corpus-out FPR
and fail to transfer across the two real corpora.

Output-side monitoring alone is insufficient for the safety case of this shipped driving model, and a
second-order recurrent-state monitor is a cheap complement. This is a single-model, collapse-specific,
offline-only result: an ImageNet-C sweep shows the silent collapse is sim-specific and the monitor is
collapse-specific, and the present evidence does not support generalization to other models, OOD axes,
or deployment contexts. Those are the next studies, not this one's claims.

---

## 8. Reproducibility Note

The analysis reruns from committed result caches without a GPU or CARLA. Each experiment's collected
tensors are committed (`report/*_collected.npz`) and the analysis regenerates from cache; the parity
test reruns from the released ONNX and comma's logged reference. The bootstrap is pinned (n=1000,
seed=42) and the run environment is pinned in `requirements.txt`. One open issue remains from the
publication-readiness audit: the E5 cache (about 3.9 GB) and the E7 cache (about 110 MB) are currently
excluded from the public repository, so the reproduce-from-cache path for E5 and E7 requires either
Git LFS tracking or a regeneration script with a smaller committed summary; the E1 through E4 and E6
caches reproduce from the committed files as-is. Large experiment caches (E5 submodule, E7, E4-RAM)
are gitignored with `--collect` regeneration instructions in the README; E1–E4 and E6 collected tensors
are committed directly (E4 in git-lfs), and E5-layer uses a small summary cache (e5_summary.npz)
committed after GPU collection for fast analysis reruns.

---

## Figure and Table Manifest

- Figure 1 (hero.png): four-panel overview (output collapse E1, uncertainty silence E3, Subaru cliff
  E4, monitor detection E6).
- Figure E1 (e1_head_collapse.png): per-head CARLA-to-real activity ratio, 8 collapsed, 2 alive.
- Figure E2 (e2_feature_ood.png): projected 512-D feature space, real clusters versus the CARLA
  freeze point.
- Figure E3 (e3_confidence.png): uncertainty distributions, real versus CARLA, against the real p95.
- Figure E4 (e4_interpolation.png + e4_ram_interpolation.png): Subaru cliff and RAM gradient.
- Figure E5a (e5_layer_localization.png): per-stage activity ratio, encoder at or above real.
- Figure E5b (e5_submodule_localization.png): per-submodule cliff-alpha, summarizer and action block.
- Table E6 (detector comparison): AUROC, AUPR, FPR@95TPR, and LOCO FPR for all five detectors at
  alpha=1.0 with 95% CIs.
- Figure E6a (e6_detector.png): monitor fire rate versus alpha, firing before the cliff.
- Figure E6b (auroc_vs_alpha.png): AUROC versus alpha for all five detectors.
- Figure E6c (roc_curves.png + pr_curves.png): ROC and PR curves at alpha=1.0 (appendix candidate).
- Figure E7a (e7_auroc_heatmap.png): 15x5 monitor AUROC heatmap across corruption-severity cells.
- Figure E7b (e7_severity_sweep.png): monitor fire rate versus severity per corruption family.
- Figure E7c (e7_overlay.png): cell-for-cell output-collapse count versus monitor AUROC.
- Table RW (competitor contrast): five-axis comparison of the named neighbors and this work.
