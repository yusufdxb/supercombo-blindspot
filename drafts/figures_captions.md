# Figures and Tables: Captions and Revision Notes

**CRITICAL NOTE on the claim ledger:** `paper_state/claim_ledger.md` contains only column
headers and no rows. There are no registered claim IDs. Every caption below therefore ties
to the named experiment identifier (E1-E7) and the specific claim text in
`paper_state/contribution_contract.md`. The claim-ledger agent must populate the ledger
before any drafter can insert `\cite{claimX}` style pointers. Until that is done, this
document uses the contract section label (e.g., "contract E1 claim") as the claim anchor.

---

## Figure 1 (hero.png)

**Caption:** Openpilot v0.9.7 supercombo shown CARLA-rendered input fails silently across
all four safety-relevant signal layers simultaneously. Top-left (E1): output-head activity
collapses to under 1% of real-driving levels on 8 of 10 heads (log-scale bar chart). Top-
right (E3): the model's own uncertainty heads stay quiet, with 0 of 220 CARLA frames
exceeding the real-driving 95th-percentile threshold (shown as a paired bar). Bottom-left
(E4, Subaru source): the collapse is a hard cliff at approximately alpha = 0.784 on the
real-to-sim blend axis, with a transition width of approximately 0.015 and an initial
output-activity spike of 6.32x at alpha = 0.425 from ghosted-input interference;
predictive uncertainty (lower sub-panel) never spikes through the transition. Bottom-right
(E6): the rolling hidden-state-spread monitor, calibrated on real-driving frames, begins
firing (fraction of frames flagged OOD rises above 0.5) at alpha = 0.55, approximately
0.23 blend-units before the output cliff. Taken together, the four panels establish the
core argument: a shipped driving model can collapse silently under simulated input while
its own output-based safety signals remain uninformative, and the failure is detectable
via a second-order recurrent-state statistic that the output heads do not surface.

- WHAT is measured: Per-head output-activity collapse ratio (E1), fraction of CARLA frames
  flagging abnormal uncertainty vs the real-driving p95 (E3), output activity and feature
  spread versus the real-to-sim blend alpha (E4, Subaru), and the rolling-spread monitor
  fire rate versus alpha (E6).
- HOW it was measured: Parity-verified openpilot v0.9.7 supercombo ONNX with correct
  recurrent-state threading (shift-and-append, no per-frame reset) and unnormalized uint8
  YUV input; real ID corpus: Subaru + RAM footage (220 analysis frames each after 100-frame
  warmup discard); CARLA OOD corpus: 220 analysis frames (clean-road renders); E4 sweep:
  29 alpha values, Subaru source; E6 threshold: 1st percentile of real-driving rolling-
  spread distribution (w=30), calibrated LOCO.
- WHY it matters: Establishes that the three defenses a safety case would rely on
  (uncertainty heads, output plausibility, temporal jitter) are exactly the channels that
  stay silent, motivating a complementary second-order internal monitor.
- WHICH claim: Contract contribution sentence (parity + E1 + E3 + E6 early-warning claims;
  ledger not yet populated).

**Revision notes:** The hero figure is a composite of four panels from E1, E3, E4, and E6.
(1) The E4 panel legend reads "transition (width 0.02)" but the contract and E4 standalone
figure both state width 0.015; check which is correct and harmonize the label. (2) The
composite is small at typical two-column width; verify that the y-axis text in all four
sub-panels is legible at 88 mm column width. (3) No explicit n is annotated on any sub-
panel; add n=220 (real) and n=220 (CARLA) as a subtitle or axis note for reviewers.

**Verdict:** KEEP.

---

## Figure E1 (e1_head_collapse.png)

**Caption:** Output-head activity collapse of openpilot v0.9.7 supercombo on CARLA-
rendered clean-road input relative to real comma driving footage. The x-axis is the ratio
(CARLA temporal activity) / (real temporal activity) on a log scale; the y-axis lists all
10 output heads. Eight heads (desired_curv, accel_t0, lead, desire_state, lane_lines, plan,
lead_prob, road_edges) fall below the 0.01 collapse threshold (ratio range 0.0018 to
0.0076), shown in orange. Two heads survive: pose (ratio 0.1788) and meta (ratio 0.7181),
shown in gray. Activity is the temporal standard deviation summed across all head elements,
computed on 220 analysis frames per condition after 100-frame warmup discard. The dashed
yellow line marks the 0.1 collapse threshold; the dotted gray line marks parity with real
(ratio = 1.0). The finding establishes that the model's primary driving outputs (planning
trajectory, acceleration, lane geometry, lead detection, curvature command) are functionally
inactive on CARLA-rendered input while its ego-motion and meta outputs partially survive.

- WHAT is measured: The ratio of CARLA to real output-head temporal activity (temporal std,
  summed across head elements) for each of the 10 supercombo output heads.
- HOW it was measured: Parity-verified openpilot v0.9.7 ONNX; Subaru real footage
  (n=220 analysis frames) vs CARLA clean-road renders (n=220 analysis frames); 100-frame
  warmup discarded; activity = sum of per-element temporal standard deviations.
- WHY it matters: Demonstrates the output-collapse phenomenon quantitatively; shows which
  heads carry the safety-critical driving signal that goes dark under sim input.
- WHICH claim: Contract E1 claim ("8 of 10 output heads collapse to under 1% of real-
  driving temporal activity").

**Revision notes:** (1) The log-scale x-axis truncates at approximately 10^-2.5; confirm
that desired_curv (ratio 0.0018) is fully visible and not cut off. (2) The two surviving
heads (pose, meta) extend past the rightmost gridline; adding a second dotted line at
ratio = 1.0 (parity) would help readers read the surviving-head values. The legend already
mentions "parity with real" as a dotted line, but the line appears to be at approximately
0.9 rather than 1.0 in the rendered figure; verify placement. (3) n is not annotated on
the figure; add "(n=220 per condition)" to the axis label or as a subtitle.

**Verdict:** KEEP. Fix the n annotation and verify the parity-line placement before
submission.

---

## Figure E2 (e2_feature_ood.png)

**Caption:** PCA projection (first two principal components) of the supercombo 512-D
recurrent hidden state for real comma footage (cyan, n=438) and CARLA-rendered frames
(orange, n=219). All 219 CARLA frames collapse to a single point in the lower-center of
the projection, annotated with an arrow. The real corpus spreads across two distinct
clusters, consistent with the two source segments (Subaru and RAM) occupying disjoint
regions of the feature space. Feature spread, measured as the trace of the hidden-state
covariance, is 0.00001x (approximately 1e-5) of the real-corpus spread. Real-vs-CARLA
linear separability along the centroid-difference direction is 88% (d' = 2.2). The figure
shows that the recurrent state freezes to a near-constant vector under CARLA input and
that the freeze point lies in a region of the ID feature space distinct from both real
clusters, which explains why Mahalanobis distance (calibrated on the ID mean) fails: the
frozen vector falls near the high-density center of the ID Gaussian and therefore scores
as a low-distance, i.e., in-distribution, point.

- WHAT is measured: The 512-D hidden_state of supercombo projected onto its first two
  principal components, with feature-spread ratio (covariance trace) and linear
  separability (d') as summary statistics.
- HOW it was measured: PCA fitted on the real corpus; the same projection applied to CARLA
  frames; trace ratio computed from full 512-D covariance matrices; d' computed along the
  centroid-difference direction; n values from the figure annotation (n=438 real, n=219
  CARLA).
- WHY it matters: Shows that the recurrent state carries a strong OOD signal even when the
  output heads are collapsed, justifying the E6 second-order monitor; also explains the
  Mahalanobis failure mode mechanistically.
- WHICH claim: Contract E2 claim ("CARLA feature spread is 1e-5 of real; real-vs-CARLA
  separability 87.9%, d' = 2.19").

**Revision notes:** (1) The figure title rounds the numbers to "0.0000x", "88%", and
"d'=2.2", whereas the data source (teardown_results.md) states 0.00001x, 87.9%, and 2.19.
The caption should match the data; either the figure annotation or the caption must use the
same precision. (2) The n values shown in the figure (n=438 real, n=219 CARLA) do not
match the teardown_results.md description ("320 frames each, 100 warmup discarded" = 220
analysis frames per corpus, so 440 real total). The metrics_results.md states ID n=638
(inconsistent with both). The figure author must reconcile the analysis-frame counts before
submission; the caption cannot report a definitive n until this is resolved. ASSUMPTION:
the figure n values reflect the actual analysis after warmup, and the teardown_results.md
header may describe per-segment counts before concatenation; the discrepancy needs
explicit confirmation from the analysis script. (3) The two real-corpus clusters are not
labeled by source segment (Subaru vs RAM); labeling them would make the cross-corpus
separation in E6 immediately legible from this figure.

**Verdict:** FIX-THEN-KEEP. Reconcile n annotation and round/precision mismatch before
submission.

---

## Figure E3 (e3_confidence.png)

**Caption:** Supercombo v0.9.7 predictive-uncertainty response to CARLA-rendered input on
three output heads (plan, lead, desired_curv). For each head, the left bar shows the
percentage of temporal output activity lost on CARLA (99.4%, 99.6%, and 99.8% activity
lost, respectively), confirming the collapse from E1. The right bar shows the fraction of
CARLA frames that exceed the real-driving 95th-percentile uncertainty threshold; in every
case this is 0% (0 of 220 CARLA frames). The uncertainty ratios (CARLA / real) are 1.35x
for plan, 1.20x for lead, and 1.84x for desired_curv, far below the real-driving p95.
The finding is that the model's own predictive-uncertainty heads fail to register the
collapse as anomalous: a monitor that sets an uncertainty threshold on real driving would
never fire, making the failure silent by the model's own signals.

- WHAT is measured: Two quantities per output head: (1) percentage of CARLA output activity
  relative to real that is lost, and (2) percentage of 220 CARLA frames exceeding the
  real-driving 95th-percentile of the head's predictive uncertainty.
- HOW it was measured: Parity-verified openpilot v0.9.7 ONNX; real ID frames (Subaru +
  RAM, 220 frames each after warmup) used to set the p95 uncertainty threshold; CARLA OOD
  frames (n=220 analysis frames after warmup) tested against that threshold; paired-bar
  visualization.
- WHY it matters: Establishes the silence of the model's built-in uncertainty channel under
  the collapse condition, showing that output-side monitoring alone cannot detect this
  failure mode.
- WHICH claim: Contract E3 claim ("0/220 OOD frames above real p95; uncertainty ratios
  1.20x-1.84x").

**Revision notes:** (1) The cyan "CARLA frames the model flags abnormal" bars are invisible
because they represent 0% -- this is intentional but makes the legend entry for cyan
visually absent. Adding a text annotation "0%" at the base of the right bar would make it
clearer that the bar exists but is zero, not that it was omitted. The current figure
already annotates "0%" in text above each right bar, so this is adequate, but verify that
the annotation is legible at column width. (2) The legend label reads "CARLA frames the
model flags abnormal (uncertainty > real p95)" -- confirm "real p95" is defined in the
caption body of the paper, as a standalone reader of the figure will not know what p95
means without context. (3) n=220 should be annotated on the figure or in the caption.

**Verdict:** KEEP.

---

## Figure E4 (e4_interpolation.png) -- Subaru panel

**Caption:** Supercombo v0.9.7 response on the real-to-sim alpha-blend interpolation axis,
Subaru source, 29 alpha values from 0 (real Subaru frame) to 1 (CARLA frame). Upper panel:
output activity (cyan, normalized so 1.0 = real level) and feature collapse (orange,
normalized so 1.0 = CARLA centroid) versus alpha. Output activity peaks at approximately
6.32x real at alpha = 0.425 (ghosted-input thrash), then falls through a hard cliff
(transition width approximately 0.02 as marked in the legend, see revision note) between
alpha approximately 0.784 and 0.799, reaching near-zero by alpha = 0.80. Feature spread
(orange curve) converges smoothly to the CARLA centroid across the sweep; no cliff is
visible in the feature curve. Lower panel: predicted plan uncertainty (mean plan_std) rises
from approximately 0.34 at alpha=0 to approximately 0.60 at alpha=0.4, then plateaus near
0.55-0.57 and does not spike at the cliff, confirming uncertainty silence through the
transition. The two-phase response (thrash, then cliff) characterizes the Subaru segment's
collapse geometry and establishes a lower bound on E6 early-warning headroom of
approximately 0.23 blend-units (E6 fires at alpha=0.55, cliff at alpha~0.784).

- WHAT is measured: Normalized output activity (sum of per-head temporal std) and feature-
  collapse progress (distance to CARLA centroid, normalized) versus the real-to-sim blend
  factor alpha, plus predicted plan uncertainty, all on the Subaru-source clip.
- HOW it was measured: 29-point sweep; at each alpha, the pixel-space blend (alpha * CARLA
  + (1-alpha) * real) is fed to the parity-verified supercombo ONNX with rolled recurrent
  state; activity and uncertainty computed on 220 analysis frames per alpha value after
  warmup.
- WHY it matters: Characterizes the collapse as a hard cliff (not a gradual degradation)
  and establishes the early-warning gap that E6 exploits; shows uncertainty is useless as
  a cliff detector.
- WHICH claim: Contract E4 claim ("cliff transition width 0.015 on Subaru; output activity
  falls 0.9x to 0.1x of real over alpha 0.784-0.799; peak 6.32x at alpha=0.425").

**Revision notes:** (1) The legend shows "transition (width 0.02)" but the contribution
contract states the transition width as 0.015. Verify from the analysis script which value
is correct and update the legend and all caption text to match. This is a precision
discrepancy between the figure and the paper's locked contract. (2) The feature-collapse
curve (orange) is labeled "feature collapse (1.0 = CARLA centroid)" but the y-axis is
shared with output activity (1.0 = real). The dual-normalization convention is confusing;
consider a secondary y-axis or separate annotation. (3) The x-axis label "(0 = real Subaru
frame, 1 = CARLA frame)" is clear; keep it. (4) No n annotation; add "(n=220 per alpha)"
to the axis or subtitle.

**Verdict:** FIX-THEN-KEEP. Resolve the width 0.02 vs 0.015 discrepancy.

---

## Figure E4-RAM (e4_ram_interpolation.png) -- RAM panel

**Caption:** Supercombo v0.9.7 response on the real-to-sim alpha-blend interpolation axis,
RAM source, confirming segment-dependence of the collapse geometry. Upper panel: output
activity (cyan, normalized so 1.0 = real level) and feature collapse (orange). Unlike the
Subaru source, the RAM source produces a gradient collapse: output activity falls gradually
across a transition band of approximately 0.27 (shown as the shaded region in the legend,
labeled "RAM transition (0.27)"; the Subaru 0.02 transition is shown for comparison as a
narrower shaded band). No sharp cliff exists; activity reaches near-zero only at the
highest alpha values. The feature curve converges more slowly than in the Subaru case.
Lower panel: predicted plan uncertainty (mean plan_std) rises from approximately 0.50 at
alpha=0 to approximately 0.65 at alpha=0.5, then declines to approximately 0.55 at
alpha=1; again no spike at the transition. The gradient geometry implies no early-warning
headroom: E6 cannot fire substantially before output collapse because the recurrent-feature
shift is spread across the full sweep rather than concentrated before a cliff. Cliff
headroom cannot be assumed to generalize across segments.

- WHAT is measured: Same quantities as E4-Subaru but computed on the RAM-source clip,
  exposing segment-dependent collapse geometry.
- HOW it was measured: Same protocol as E4-Subaru, applied to the RAM real footage as the
  alpha=0 source.
- WHY it matters: Establishes segment-dependence of the cliff shape; prevents
  over-generalization of the early-warning gap measured in E4-Subaru.
- WHICH claim: Contract E4-RAM claim ("on the RAM source the collapse is a gradient with
  width 0.274; cliff headroom cannot be assumed to generalize").

**Revision notes:** (1) The legend reads "RAM transition (0.27)" but the contract states
"width 0.274". Use the three-digit value for consistency with the contract and with
teardown_results.md. (2) The dual-normalization convention on the y-axis (same issue as
E4-Subaru) needs resolution. (3) The two shaded transition bands (RAM broad, Subaru
narrow) are a good visual contrast but the narrow Subaru band is nearly invisible at
journal print resolution; increase its opacity or width for legibility. (4) Add n
annotation.

**Verdict:** FIX-THEN-KEEP. Align transition-width labels with the contract value (0.274).
This figure and e4_interpolation.png should appear as a two-panel figure (E4a/E4b) in the
paper; confirm the drafter places them side by side.

---

## Figure E5a (e5_layer_localization.png)

**Caption:** Per-stage activity ratio (CARLA / real) across the openpilot v0.9.7
supercombo vision-encoder stages as alpha varies from 0 (real) to 1 (CARLA), showing that
the collapse does not originate in the encoder. Stages plotted: stem, stage0, stage1,
stage2, stage3, and head. All six lines remain at or above 1.0 across the full alpha sweep
(minimum ratio approximately 0.96 at alpha = 0.1 for stage0); none crosses the 0.5
collapse threshold (dashed gray line, never crossed). Stage3 reaches approximately 2.06x
and head reaches approximately 2.14x at alpha=1.0, indicating that the encoder is actually
more active on CARLA input than on real input. The collapse observed in E1 at the model
output therefore originates downstream of the vision encoder, in the recurrent summarizer
or action-block stages. This is a PARTIAL localization: the encoder is ruled out, but the
exact entry point within the downstream submodules requires E5b (submodule probing).

- WHAT is measured: The ratio (CARLA temporal activity) / (real temporal activity) for
  the output activations of each encoder stage, as a function of the alpha-blend parameter.
- HOW it was measured: Activation hooks inserted at each named encoder stage; same 29-point
  alpha sweep as E4 on the Subaru source; activity metric identical to E1 (temporal std
  summed over spatial/channel dims).
- WHY it matters: Rules out the vision encoder as the collapse site, redirecting the
  localization search to the recurrent summarizer and action-block feedback path, which is
  a key mechanistic claim of the paper.
- WHICH claim: Contract E5 claim ("every encoder stage stays at or above real activity;
  minimum 0.96; collapse is downstream of the encoder").

**Revision notes:** (1) The figure uses a white-background style inconsistent with the
dark-background style of E1-E4 and E6; standardize the color scheme across all figures
before submission. (2) The y-axis label "activity ratio (CARLA / real)" is clear but
would benefit from a note that values above 1.0 mean "more active on CARLA than real."
(3) n annotation is absent; add. (4) The "head" line (dark purple) is nearly
indistinguishable from "stage3" (dark green) in the upper region; consider distinct
markers or a wider color separation.

**Verdict:** FIX-THEN-KEEP. Standardize background style with the other experiment figures.

---

## Figure E5b (e5_submodule_localization.png)

**Caption:** Per-submodule activity ratio (CARLA / real) for eight downstream components
of openpilot v0.9.7 supercombo, probing the cliff entry point downstream of the vision
encoder. Submodules plotted: vision_post, summarizer_div, attention_block_out,
transformer_block_out, reduce_sum, action_block_body, hydra_trunk, temporal_hydra_trunk.
The recurrent summarizer output (summarizer_div, dark blue) and the action-block body
(action_block_body, salmon/red) are the two submodules whose ratios cross below the 0.5
collapse threshold (dashed gray). The action_block_body drops below 0.5 first, at
approximately alpha = 0.5, driven by the prev_desired_curv recurrent feedback loop.
The summarizer_div drops below 0.5 at approximately alpha = 0.80-0.90. The hydra_trunk
and temporal_hydra_trunk remain above 1.0 or near 1.0 across the sweep, acting as passive
amplifiers. vision_post stays elevated (approximately 1.9 at alpha=1.0), confirming E5a's
conclusion that the encoder pathway is uninvolved in the collapse. Localization is PARTIAL:
a VAE-mu / sigma ambiguity within the summarizer_div block remains unresolved.

- WHAT is measured: Activity ratio (CARLA / real) for eight named submodule outputs as a
  function of the alpha-blend parameter.
- HOW it was measured: Activation hooks at each named submodule; same Subaru-source 29-
  point alpha sweep; same activity metric as E1/E5a.
- WHY it matters: Pins the collapse entry point to the recurrent summarizer and the
  action-block feedback path, providing partial mechanistic localization that guides future
  architectural interventions.
- WHICH claim: Contract E5 claim ("cliff entry at summarizer_div VAE-mu bottleneck and
  action_block_body feedback path; vision_post and hydra_trunk are passive relays;
  localization is PARTIAL -- VAE-mu/sigma ambiguity remains").

**Revision notes:** (1) Same white-background style inconsistency as E5a; harmonize.
(2) Eight lines with similar colors in the mid-range are difficult to distinguish at column
width; consider grouping passive relays (vision_post, hydra_trunk, temporal_hydra_trunk)
into a shaded band and highlighting only the two cliff-crossing lines (summarizer_div,
action_block_body) in distinct saturated colors. (3) The x-axis only has labeled ticks at
0.0, 0.2, 0.4, 0.6, 0.8, 1.0; the cliff crossings are between ticks -- mark the
approximate cliff-alpha values with vertical annotations for clarity. (4) The VAE-mu/sigma
ambiguity is not visually represented; a text annotation on the summarizer_div line noting
"VAE-mu bottleneck (ambiguity: mu vs sigma)" would alert readers.

**Verdict:** FIX-THEN-KEEP. Line-color crowding at column width is a readability hazard.

---

## Figure E6a (e6_detector.png)

**Caption:** Response of the E6 rolling hidden-state-spread detector across the E4 Subaru
alpha sweep. The y-axis is the fraction of frames flagged as OOD at each alpha value; the
x-axis is alpha (0 = real Subaru frame, 1 = CARLA frame). The detector is calibrated on
all real-driving frames at the 1st-percentile threshold of the rolling spread distribution
(threshold = 0.0787, window w=30). The fire fraction remains at 0 for alpha up to
approximately 0.40, rises through a ramp from 0.40 to 0.55, and crosses the 50% fire
threshold (dashed horizontal line) at alpha = 0.55. A vertical solid line marks the
firing point. The fraction continues rising to 0.986 at alpha = 1.0. Because the E4 output
cliff occurs at alpha approximately 0.784-0.799, the detector fires (>50% frames) roughly
0.23 blend-units before the output becomes operationally collapsed. The LOCO-calibrated
FPR is 1.03% mean (max 2.07%) across the N=2 real corpora (Subaru and RAM), a two-fold
estimate whose variance is not meaningfully reportable at N=2.

- WHAT is measured: Fraction of frames per alpha step that the E6 detector flags as OOD,
  showing the detector fire profile along the distribution-shift gradient.
- HOW it was measured: Rolling spread (trace of covariance of a 30-frame window of the
  512-D hidden_state); threshold calibrated at the 1st percentile of the real-driving
  rolling-spread distribution (all real frames); applied to each alpha step of the E4
  Subaru sweep (29 alpha values, n=220 frames per step after warmup).
- WHY it matters: Demonstrates early-warning capability: the monitor signals OOD 0.23
  blend-units before the outputs collapse, providing a monitoring gap that output-side
  checks cannot achieve.
- WHICH claim: Contract E6 claim ("fires at alpha=0.550; AUROC 0.996; LOCO mean FPR 1.03%
  (max 2.07%); fires 0.23 blend-units before cliff at alpha~0.784").

**Revision notes:** (1) The early-warning gap (0.23 blend-units) is not visually annotated;
add a horizontal bracket between the fire line (alpha=0.55) and the cliff position
(alpha~0.784) with the "0.23 gap" label. (2) The cliff position from E4 is not overlaid
on this figure; adding a dashed vertical line at alpha=0.784 would make the gap immediately
legible. (3) The N=2 / LOCO two-fold qualifier for the 1.03% FPR is not on the figure;
add a text note "(LOCO, N=2)" near the detector-fires annotation.

**Verdict:** FIX-THEN-KEEP. Add the E4 cliff overlay and early-warning gap annotation.

---

## Figure E6b (auroc_vs_alpha.png)

**Caption:** AUROC versus alpha-blend (distribution-shift gradient) for all five detectors
evaluated on the 512-D supercombo recurrent feature: E6 rolling-spread (red), Mahalanobis
(blue), Relative Mahalanobis (green), KNN-50 (gold), and PCA-Mahalanobis (purple). The
chance line (AUROC = 0.5) is shown as a dashed horizontal line. At alpha=1.0 (full CARLA),
E6 reaches AUROC 0.996 and KNN-50 reaches 1.000; these two detectors converge to
essentially perfect separation. The distinguishing axis is not single-corpus AUROC but
cross-corpus calibration: KNN-50 achieves 100% LOCO FPR while E6 holds 1.03% mean (max
2.07%), a comparison not visible in this figure (see Table E6). Mahalanobis (blue) inverts
at low alpha, dropping to near 0.15 at alpha=1.0 because the recurrent state collapses to
the ID mean and distance-from-mean cannot detect collapse-to-the-mean. Relative Mahalanobis
maintains AUROC approximately 0.93 across most of the sweep. PCA-Mahalanobis behaves
similarly to Mahalanobis: it falls to near 0.15 at alpha=1.0. E6 crosses the 0.5 AUROC
threshold at approximately alpha=0.55, coinciding with its detector firing point from E6a,
confirming that the early-warning signal is genuine rather than an artifact of calibration.

- WHAT is measured: AUROC (ID = Subaru+RAM, OOD = alpha-blend CARLA frames) as a function
  of alpha, for five detectors on the same 512-D recurrent feature.
- HOW it was measured: At each alpha value in the 29-point E4 sweep, the 220 alpha-blend
  frames are treated as OOD and the 638 real-corpus frames as ID (per metrics_results.md);
  AUROC computed without bootstrap (sweep visualization does not show CIs; full CIs at
  alpha=1.0 are in Table E6).
- WHY it matters: Shows the complete trajectory of each detector's separability as the
  distribution shift increases, contextualizing the alpha=1.0 snapshot in Table E6 and
  illustrating when E6's early-warning signal becomes detectable.
- WHICH claim: Contract E6 claim (AUROC trajectory; early-warning at alpha=0.55;
  Mahalanobis below-chance explanation).

**Revision notes:** (1) No confidence intervals are shown on the sweep; this is acceptable
for a trajectory plot, but a note in the caption (and in the paper body) that CIs are
provided at alpha=1.0 in Table E6 should be included. (2) The KNN-50 and Relative
Mahalanobis curves are nearly coincident in the middle of the alpha range and both gold/
green; consider adding distinct markers (square vs triangle) to disambiguate at print
resolution. (3) The figure title "AUROC vs distribution-shift gradient" is accurate;
consider adding "-- 5 detectors on supercombo 512-D recurrent feature" as a subtitle for
standalone readability.

**Verdict:** KEEP.

---

## Figure E6c-ROC (roc_curves.png)

**Caption:** Receiver-operating-characteristic curves at full CARLA shift (alpha=1.0) for
all five detectors on supercombo v0.9.7 recurrent features (ID: Subaru+RAM concatenated,
n=638; OOD: CARLA frames, n=319). The TPR=0.95 reference line (dashed) is annotated. E6
(red) and KNN-50 (gold) hug the upper-left corner and are nearly coincident: both achieve
essentially zero FPR at 95% TPR (FPR@95TPR = 0.000 for both). Relative Mahalanobis
(green) crosses the TPR=0.95 line at approximately FPR=0.067. Mahalanobis (blue) and PCA-
Mahalanobis (purple) show near-degenerate curves: Mahalanobis performs below chance
(curve below the diagonal at high FPR), and PCA-Mahalanobis reaches TPR=1.0 only at
FPR approximately 0.85, consistent with AUROC 0.152 (below chance). The below-chance
behavior of Mahalanobis (AUROC 0.159) reflects the mechanical collapse: CARLA frames
receive lower Mahalanobis distance than real frames because the frozen recurrent state
lands at the center of the ID Gaussian. Note that single-corpus AUROC parity between E6
and KNN-50 does not imply calibration parity: KNN-50 has 100% LOCO FPR versus E6's 1.03%
mean (max 2.07%, N=2 two-fold estimate); this distinction requires Table E6 and is not
visible here.

- WHAT is measured: ROC curves (TPR vs FPR) at alpha=1.0 for E6, Mahalanobis, Relative
  Mahalanobis, KNN-50, and PCA-Mahalanobis on the 512-D supercombo hidden state.
- HOW it was measured: ID = Subaru + RAM concatenated (n=638 per metrics_results.md); OOD
  = CARLA alpha=1.0 frames (n=319 per metrics_results.md); threshold-free sweep over all
  classifier thresholds; bootstrap CIs not shown on this plot (reported as Table E6).
- WHY it matters: Provides the full threshold-free comparison at the target operating point,
  making the Mahalanobis failure mode and the E6/KNN-50 near-perfect separation visible
  simultaneously.
- WHICH claim: Contract E6 claim (threshold-free metrics; Mahalanobis AUROC 0.159 below
  chance; E6 AUROC 0.996; KNN AUROC 1.000).

**Revision notes:** (1) The Mahalanobis (blue) and diagonal (dotted) are nearly
indistinguishable at low FPR; the PCA-Mahalanobis (purple) curve's abrupt step at
FPR~0.85 is visually dramatic but may be misleading at small print size -- verify it
renders correctly. (2) The n values (n=638 ID, n=319 OOD) should appear in the figure
caption area or subtitle, not just in the title. The current title states
"ROC curves: ID (subaru+ram) vs CARLA alpha=1.0" which is good; add the n values.
(3) Consider moving to a supplemental figure or appendix if Table E6 already covers the
headline numbers; the ROC/PR pair is informative for mechanism but may not earn main-body
space given the two-column budget.

**Verdict:** FIX-THEN-KEEP (or move to appendix if space is tight; DO NOT cut without
ensuring Table E6 carries the FPR@95TPR numbers with CIs).

---

## Figure E6c-PR (pr_curves.png)

**Caption:** Precision-recall curves at alpha=1.0 for all five detectors on supercombo
v0.9.7 recurrent features (same ID/OOD split as ROC figure). E6 (red) and KNN-50 (gold)
both reach precision=1.0 at all recall levels up to approximately 0.98-1.0, confirming
AUPR approximately 1.000 for both. Relative Mahalanobis (green) shows a precision that
degrades from approximately 1.0 at low recall to approximately 0.88 at full recall, with
AUPR 0.732 (per metrics_results.md). Mahalanobis (blue) and PCA-Mahalanobis (purple) are
near the baseline precision curve (proportion of OOD frames = 319/957 approximately 0.33),
consistent with their below-chance AUROC. The PR curves confirm that E6 and KNN-50 achieve
near-perfect separation at alpha=1.0 but do not distinguish their cross-corpus calibration
performance, which requires Table E6.

- WHAT is measured: Precision-recall curves (Precision vs Recall/TPR on OOD) at alpha=1.0
  for the same five detectors as the ROC figure.
- HOW it was measured: Same ID/OOD split; precision = TP/(TP+FP) at each threshold sweep
  point; axes: x = Recall (TPR on OOD), y = Precision.
- WHY it matters: Complements the ROC curve by showing precision degradation at high recall,
  which is the operationally relevant regime for a safety monitor (catching most OOD frames
  without flooding with false positives).
- WHICH claim: Contract E6 claim (AUPR 0.995 for E6; AUPR 1.000 for KNN-50; the
  distinction between single-corpus AUPR and cross-corpus calibration).

**Revision notes:** (1) The E6 (red) and KNN-50 (gold) curves are nearly identical and
overlap; consider adding a small offset or annotation. (2) n values absent; add "(ID n=638,
OOD n=319)" to the title or subtitle. (3) Same appendix / consolidation flag as roc_curves.

**Verdict:** FIX-THEN-KEEP (or move to appendix alongside roc_curves.png; both may be
consolidated into one two-panel figure).

---

## Figure E7a (e7_auroc_heatmap.png)

**Caption:** E6 rolling-spread AUROC across 15 ImageNet-C corruptions at 5 severity levels
(75 cells total), applied to real Subaru frames. The color scale runs from 0.0 (red, chance
level) to 1.0 (dark green). The vast majority of cells are light-green, indicating AUROC
in the 0.50-0.70 range (near chance). Four cells stand out: frost severity-3 (0.96), frost
severity-5 (1.00), gaussian_noise severity-4 (0.86), and impulse_noise severity-5 (0.91).
No corruption cell reaches the output-collapse regime (0 of 75 cells collapse 5 or more of
10 output heads, versus 7 of 10 under CARLA; see E7c for cell-for-cell verification). The
four high-AUROC cells therefore track a recurrent-feature-spread shift that is NOT an
output collapse, confirming that E6 is collapse-specific and near-chance on the corruptions
the model tolerates. Mean AUROC across all corruptions and severities is substantially
below 0.80 for every corruption family except frost (mean 0.71 driven by the two outlier
cells). The heatmap establishes that E6 is not a universal corruption detector.

- WHAT is measured: E6 AUROC (rolling hidden-state-spread, threshold-free, ID = Subaru
  real frames) for each of the 75 (corruption type x severity) cells.
- HOW it was measured: 15 ImageNet-C corruptions applied to raw RGB Subaru frames before
  YUV conversion; supercombo re-run with correct recurrent-state threading (w=30,
  100-frame warmup discard); E6 AUROC computed using the same 512-D hidden_state threshold
  as in E6 main analysis; stratified bootstrap (n=1000, seed=42) available but CIs not
  shown on heatmap.
- WHY it matters: Bounds E6's scope to collapse-specific detection; shows the monitor is
  not firing on most real-frame corruptions, preventing overclaim of generalization.
- WHICH claim: Contract E7 claim ("E6 is near chance on most photometric/blur corruptions;
  4 high-AUROC cells track a feature-spread shift, not output collapse;
  mean AUROC 0.52-0.74 across named families").

**Revision notes:** (1) The heatmap cells show two decimal places (e.g., 0.96, 1.00)
which is appropriate. (2) The four high-AUROC cells (frost sev3, frost sev5, gaussian_noise
sev4, impulse_noise sev5) are visually prominent but not annotated with their cell labels;
adding small "FP" markers or a box around those cells would help readers identify them
without having to scan the full grid. (3) The colorbar label is "AUROC" which is
sufficient; confirm the colorbar extends from 0.0 to 1.0 and that the 0.5 chance level is
visually distinct (consider adding a contour line at AUROC=0.5). (4) frost severity-4
shows AUROC=0.47 (below chance), which is an interesting anomaly not discussed in the
outline; a brief note in the paper body would prevent reviewer confusion.

**Verdict:** KEEP.

---

## Figure E7b (e7_severity_sweep.png)

**Caption:** E6 rolling-spread detector fire rate (fraction of frames flagged OOD at the
calibrated threshold) versus severity level (1-5) for all 15 ImageNet-C corruptions,
grouped into four families: noise (gaussian, shot, impulse), blur (defocus, motion, glass,
zoom), weather (fog, frost, snow, brightness, contrast), and digital (elastic transform,
pixelate, jpeg compression). Within the noise group, impulse_noise severity-5 and
gaussian_noise severity-4 show elevated fire rates (approaching 0.55 and 0.20,
respectively). Within the weather group, frost severity-5 shows a fire rate of 1.00 at
severity-5. All blur corruptions and all digital corruptions show fire rates near 0 across
all severities, confirming that E6 is insensitive to blur and digital artifacts. The figure
shows that E6's few firings are concentrated in the most severe noise and frost conditions,
and none coincides with an output collapse (confirmed by E7c).

- WHAT is measured: Fraction of analysis frames (after warmup) that E6 flags as OOD at
  each severity level, per corruption type, grouped by family; y-axis = fire rate
  (0 = never fires, 1 = always fires).
- HOW it was measured: Same setup as E7a; the calibrated E6 threshold (0.0787, 1st
  percentile of real-driving rolling spread) applied to each corruption-severity sequence.
- WHY it matters: Shows severity dependence within each corruption family, confirming that
  E6's rare firings are monotonically concentrated in the most extreme noise/frost cases
  and absent for all blur/digital corruptions.
- WHICH claim: Contract E7 claim ("E6 fires on 4 cells with NO output collapse; correctly
  quiet on most corruptions; bounded to a few severe noise/frost cases").

**Revision notes:** (1) The four subplots have very small text at typical column width;
increase font sizes or make this a full-column (88 mm) or two-column (180 mm) figure.
(2) The y-axis label "Fraction of Frames Flagged OOD" is correct but cut off in the
rendering; verify full label is visible. (3) In the weather group, the frost severity-5
fire rate of 1.00 appears as a dramatic spike; visually annotate that this is the FP
cell (E6 fires but no output collapse), so readers do not misread it as a true positive.
(4) The legend in each subplot has small text; consider a single shared legend.

**Verdict:** FIX-THEN-KEEP. Font sizes are the primary readability issue at column width.

---

## Figure E7c (e7_overlay.png)

**Caption:** Cell-for-cell scatter of E6 detection AUROC (y-axis) versus the number of
output heads collapsed (x-axis, out of 10) for all 75 corruption-severity cells, with the
CARLA reference (7/10 heads collapsed) shown as an annotated dashed vertical line. No
corruption cell appears to the right of the cutoff line: 0 of 75 cells reach the
output-collapse threshold of 5 or more heads, confirming the validation-gate result that
the silent-collapse failure mode is CARLA/full-sim specific and does not reproduce under
ImageNet-C corruptions of real frames. Points are color-coded by verdict: blue (TN,
correctly quiet, n=53), orange (FP, E6 fires with no output collapse, n=4), and gray
(marginal, n=18). All orange and gray points cluster at x=0 or x=1 (0 or 1 head
collapsed), meaning every high-AUROC E6 response is decoupled from output collapse. Cyan
and blue horizontal dotted lines at AUROC=0.70 and 0.85 mark approximate
"marginal/alert" boundaries. The figure directly answers the false-negative question:
because no non-CARLA corruption induces output collapse, there are zero false negatives.
E6's 4 FP firings track a recurrent-feature-spread shift that is not the collapse mode.

- WHAT is measured: For each of the 75 ImageNet-C corruption-severity cells, two quantities
  are paired: (1) the number of supercombo output heads collapsed at that cell (E1 collapse
  criterion, ratio < 0.10, max heads = 10), and (2) E6 AUROC for that cell.
- HOW it was measured: E1 collapse count read from the e7_overlay_results.md validation
  run (which first reproduced the CARLA 7/10 collapse as a gate); E6 AUROC from e7_results.md;
  verdict labels (TN / FP / marginal) assigned per e7_overlay_results.md.
- WHY it matters: This is the key bounding figure: it separates the scope of the collapse
  (CARLA-specific) from the scope of E6 (collapse-specific), preventing the reader from
  concluding that E6 generalizes as a corruption detector. The right half of the plot is
  empty, which is itself the finding.
- WHICH claim: Contract E7 claim ("0 of 75 corruption cells produce output collapse;
  E6 FP firings are decoupled from collapse; collapse is sim-specific; E6 is collapse-
  specific; 0 false negatives").

**Revision notes:** (1) The x-axis extends to 10 with all points clustering at 0-1; the
region x=2 to x=10 is effectively dead space that makes the left cluster visually cramped.
Consider truncating the x-axis at x=4 and placing the CARLA annotation as an inset arrow
or off-axis label, freeing horizontal space to spread the left-cluster points. (2) The two
horizontal dotted cyan/blue lines (at ~0.70 and ~0.85) are not explained in the figure
legend; add labels or drop them if they are not formally defined thresholds. (3) The
CARLA=7/10 dashed vertical line annotation is the most important visual element; make it
visually dominant (thicker line, saturated color, explicit text label). (4) Consider
plotting the CARLA point itself (x=7, AUROC=0.996 from E6 main) as a large star or
diamond at (7, 0.996) to show where the collapse-plus-detection case lands -- it would
live in the empty upper-right quadrant and powerfully illustrate the claim.

**Verdict:** FIX-THEN-KEEP. X-axis truncation and CARLA annotation are the priority fixes.

---

## Flagged for cut

None of the 15 figures (including e4_ram_interpolation.png as the Subaru-panel companion)
is purely decorative or claim-less. However, two figures have consolidation and space
notes:

- **roc_curves.png + pr_curves.png**: Both earn their place as threshold-free evidence for
  the E6 vs baseline comparison, but both could be moved to supplemental/appendix if Table
  E6 is promoted to the main body with bootstrap CIs. Do NOT cut both without ensuring
  Table E6 (to-produce) covers FPR@95TPR with CIs.

- **e7_severity_sweep.png**: Informative for showing monotonic severity dependence within
  noise/frost families, but e7_auroc_heatmap.png already conveys the "near chance on most,
  4 outlier cells" story. If space forces a cut to 5.7, retain the heatmap (E7a) and move
  the severity sweep (E7b) to supplemental. Do NOT cut E7b from the paper entirely without
  replacing the per-family monotonicity argument in text.

---

## Table RW (competitor contrast) -- to-produce

**Caption (draft for when the table is produced):** Comparison of E6 and six prior
monitoring / OOD-detection approaches on five axes: substrate (what activations are
monitored), score type (first-order location vs second-order spread), cross-corpus
calibration evaluation, target model (shipping end-to-end driver vs research stack), and
formal guarantee. Keser 2025 and Guo/Su 2026 are the closest AV-native neighbors; neither
targets the recurrent state of a shipped end-to-end driver. EigenTrack (arXiv:2509.15735)
uses a second-order covariance-spectrum statistic but on LLMs/VLMs with a trained
recurrent classifier, not on a driving model and without cross-corpus LOCO evaluation.
Location-based scores (Mahalanobis, RMD, KNN) achieve high single-corpus AUROC on this
model but fail at 100% LOCO FPR, a failure mode not visible in single-corpus evaluations.
This work is the first to apply a location-invariant second-order spread monitor to the
recurrent state of a parity-verified shipped end-to-end driving model evaluated under
cross-corpus LOCO transfer.

- WHAT is measured: Five categorical axes per method; no continuous statistics.
- HOW it was measured: Cell values sourced from literature_map.md (verified) and from
  metrics_results.md (E6 and baselines).
- WHY it matters: Lets reviewers audit the novelty claim in one glance.
- WHICH claim: Contract contribution sentence (bounded-novelty claim: "first to use a
  location-invariant second-order spread on the recurrent state of a SHIPPED end-to-end
  driving model, evaluated under cross-corpus LOCO transfer").

**Revision notes:** The table does not exist yet (status: to-produce). Values are
available in literature_map.md and metrics_results.md. The table author must NOT write
"first to use a second-order hidden-activation statistic for OOD" because EigenTrack
pre-dates this work on LLMs/VLMs; the correct defensible framing is in the contract
(pp. 128-130 of outline.md). Bold the "This Work" row. Define all column abbreviations in
a table note. Do not include ViM or Energy (structurally inapplicable; named and excused
in Section 2 text only, not in the contrast table).

**Verdict:** to-produce. Required for Section 2; do not omit.

---

## Table E6 (detector comparison) -- to-produce

**Caption (draft for when the table is produced):** Threshold-free OOD detection metrics
for five detectors on the supercombo v0.9.7 512-D recurrent hidden state at alpha=1.0
(full CARLA shift). ID corpus: Subaru + RAM concatenated (n=638); OOD corpus: CARLA
alpha=1.0 frames (n=319). Bootstrap 95% confidence intervals: stratified by label,
n=1000 iterations, seed=42. LOCO FPR: leave-one-corpus-out mean and max across N=2 real
corpora (Subaru and RAM); this is a two-fold estimate and the variance is not meaningfully
reportable at N=2. E6 (rolling-spread) achieves AUROC 0.996 [0.992, 1.000] and LOCO mean
FPR 1.03% (max 2.07%). KNN-50 achieves AUROC 1.000 [1.000, 1.000] but LOCO mean FPR
100%, illustrating that single-corpus separation does not imply cross-corpus calibration.
Mahalanobis and PCA-Mahalanobis score below chance (AUROC 0.159 and 0.152 respectively)
because the recurrent state collapses to a point near the center of the ID Gaussian, where
distance-from-mean is minimized. Relative Mahalanobis achieves AUROC 0.934 but 100% LOCO
FPR. Bold values indicate the best-calibrated detector per column. All LOCO FPR values
above 1% should be interpreted as "fails to calibrate at the 1% operating point."

- WHAT is measured: AUROC, AUPR, FPR@95TPR (all with 95% bootstrap CI) and LOCO mean/max
  FPR for each of the five detectors.
- HOW it was measured: Bootstrap: n=1000 stratified, seed=42 (per metrics_results.md);
  LOCO: calibrate on one corpus, evaluate on the held-out corpus, average two folds.
- WHY it matters: The definitive quantitative summary of the transfer/calibration claim;
  the distinction between E6 and KNN-50 is only visible in the LOCO column.
- WHICH claim: Contract E6 claim (all numbers; KNN-50 ties E6 on AUROC, fails on LOCO;
  location-based baselines fail to transfer; E6 LOCO ~1% calibrated).

**Revision notes:** The table does not exist yet (status: to-produce; values in
metrics_results.md Table 1 and LOCO sections). When producing: (1) include the LOCO mean
and max FPR columns (not just single-corpus FPR@95TPR); (2) bold the LOCO FPR column for
E6 to highlight the calibration claim; (3) add a table note "LOCO = leave-one-corpus-out
over N=2 corpora; variance not reportable at N=2"; (4) do NOT write "E6 outperforms
baselines" -- the correct framing is "E6 calibrates where location-based baselines do not."

**Verdict:** to-produce. Load-bearing for the E6 section; required in the main body.

---

## End-of-turn honesty summary

**What I verified (source-grounded, with evidence read this session):**

- All 15 PNG files exist in `/home/yusuf/Projects/phantom-braking/report/figures/` (ls
  verified, files read visually).
- All numerical claims in captions are sourced from the following files read this session:
  `report/teardown_results.md`, `report/metrics_results.md`, `report/e6_results.md`,
  `report/e7_results.md`, `report/e7_overlay_results.md`, `report/parity_results.md`.
- The claim ledger (`paper_state/claim_ledger.md`) is empty (only headers, no rows).
  Every caption therefore uses the contract section label as its claim anchor, not a
  ledger ID.
- E4 figure legend reads "transition (width 0.02)" but contract states 0.015; flagged as
  a discrepancy in revision notes for both E4 figures.
- E4-RAM figure legend reads "RAM transition (0.27)"; contract states 0.274; flagged.
- E2 figure shows n=438 real and n=219 CARLA; teardown_results.md states 220 analysis
  frames per corpus (440 total real); metrics_results.md states ID n=638 (inconsistent);
  flagged in E2 revision notes as an n-reconciliation item.
- E5 figures use a white background inconsistent with the dark background of E1-E4, E6-E7;
  flagged.
- The CARLA validation gate in e7_overlay_results.md reproduces 7/10 heads collapsed
  (confirmed from file).
- All four E7 FP cells named in captions (frost sev3, frost sev5, gaussian_noise sev4,
  impulse_noise sev5) match e7_overlay_results.md exactly.

**What I did NOT verify and cannot treat as cleared:**

- The exact cliff-alpha values for summarizer_div (0.900) and action_block_body (0.500)
  in E5b are stated in the contribution contract / outline; I read the figure visually and
  the action_block_body crosses 0.5 somewhere between alpha=0.4 and 0.6, consistent with
  0.500, but I did not read a separate e5_submodule_results.md data file (it is not listed
  among the report/*.md files in the working directory). ASSUMPTION: the contract values
  are correctly transcribed from that file.
- The E4 transition width discrepancy (figure: 0.02, contract: 0.015) is UNRESOLVED;
  I cannot determine from the figure alone which is correct. The analysis script must be
  consulted.

**Riskiest unverified assumption:** The E4 transition-width discrepancy between the figure
("0.02") and the locked contract ("0.015") is the single riskiest item. If the contract
value is correct, the figure label must be corrected before any reviewer sees it; if the
figure value is correct, the contract must be reopened (which requires an explicit re-lock
by the contribution-locker agent). Neither action can be taken from caption-writing alone.
The figure author must resolve this before the drafter writes the E4 prose.
