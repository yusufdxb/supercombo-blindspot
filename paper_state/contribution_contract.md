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
