# Reviewer 2 Red-Team

Adversarial pre-review of `drafts/rewritten_draft.md` against the locked contribution
contract and the three audits (stats, source, reproducibility). Every attack below cites a
quoted draft location, names the evidence, rates reject-probability, and gives the exact
rebuttal or fix the authors must produce. Ordered by reject-probability (most dangerous
first). The target is a workshop / arXiv preprint (SafeAI @ UAI 2026), and severities are
calibrated to that tier: at a main track several of these would be flat rejects; at a
workshop the bar is "is the negative finding real and honestly bounded," which changes the
math on the N=1 and N=2 attacks but NOT on the CARLA confound.

---

## Attack 1: CARLA-clean is a rendering-pipeline confound, not demonstrated "distribution shift," and the parity argument does not cover it

- CLASS: uncontrolled confound (the central one).
- TARGET (draft location): Abstract, "Running the verified model on CARLA-rendered clean
  roads, we find that it fails silently"; Section 1, "On CARLA-rendered clean roads, 8 of 10
  of the model's output heads collapse"; Section 4 Method, "The out-of-distribution data are
  CARLA-rendered clean-road frames from the openpilot v0.9.7 simulation pipeline." The entire
  load-bearing phenomenon (E1, E2, E3) rests on this single OOD source.
- THE ATTACK: The paper's whole negative finding is "a production model fails silently under
  visual OOD," but the only OOD axis that induces the collapse is CARLA-clean frames. The
  paper never rules out the mundane explanation: CARLA renders with different camera
  intrinsics, different field-of-view, different tone-mapping/gamma, no lens distortion, no
  rolling-shutter, no sensor noise, and a synthetic color/texture distribution. A frozen
  network fed input from a pipeline whose geometry and photometry it never saw at train time
  will of course produce garbage. That is not "the model fails to perceive a valid driving
  scene"; it is "the input was fed through the wrong camera model." The parity argument
  (Section 4) proves only that the harness reproduces comma's output on REAL frames; it says
  nothing about whether the CARLA frames were geometrically and photometrically conditioned
  the way comma's own simulation pipeline conditions them. E7 makes this worse, not better:
  Section 5.7 shows that NO photometric or blur corruption of real frames reproduces the
  collapse ("0 of 75 cells"), which means the trigger is something specific to the full-sim
  rendering pipeline, exactly the camera-model/geometry confound, and not a general property
  of "out-of-distribution input." The reviewer's one-line kill: "You have shown the model
  breaks on one renderer's output. You have not shown it breaks on out-of-distribution
  driving scenes."
- EVIDENCE: The contract itself concedes the scope: "one extreme OOD axis (CARLA) plus one
  bounding axis (ImageNet-C)" and parks "a second, real adverse-weather OOD axis ... that
  actually induces a non-CARLA output collapse" as "the most reviewer-resistant follow-up"
  (contribution_contract.md, out-of-scope). The Limitations section concedes "no non-CARLA
  output collapse was ever induced." The draft asserts the collapse is "downstream of the
  vision encoder" (E5), which is the only structural fact that pushes back on the
  camera-intrinsics story, BUT the draft never measures the obvious confound directly: it
  never reports whether the CARLA frames were passed through comma's documented sim camera
  calibration / transform, never compares CARLA-clean to a geometry-and-photometry-matched
  render, and never tests the openpilot stack's OWN known sim-bridge path (the contract flags
  a "known bridge confound, commaai issue #31711" for MetaDrive but does not clear CARLA of
  the same class of issue). The sim-specificity result (c33, c37) is presented as a bound but
  is equally consistent with the null hypothesis "CARLA input is malformed for this model."
- SEVERITY: REJECT (this is the attack that decides the paper). For a workshop it is MAJOR
  but still the single most likely cause of rejection, because it attacks the meaning of the
  headline word "silent failure."
- REJECT-PROBABILITY: high.
- REQUIRED RESPONSE: The authors cannot argue this away with prose; they must either (a) run
  ONE control that separates "model failure on a valid scene" from "malformed input": render
  the SAME CARLA scene through comma's documented openpilot simulation camera transform
  (the calibration the production sim bridge applies) and show the collapse persists, OR show
  that real comma footage geometrically/photometrically matched to CARLA (FOV, intrinsics,
  tone curve) does NOT collapse while CARLA does; or (b) narrow every headline claim from
  "fails silently under visual out-of-distribution input" to the literal, defensible
  statement: "produces a silent collapse when fed clean CARLA renders through this inference
  path," and add an explicit paragraph in Section 5.1 and Limitations stating that the
  collapse has not been separated from a camera-model/rendering-pipeline domain gap, and that
  whether the trigger is semantic OOD or input-format mismatch is unresolved. Option (b) is
  the minimum to survive; option (a) is what makes the paper strong. The E5 localization
  helps but does not substitute: "downstream of the encoder" is consistent with both a
  semantic-OOD story and a malformed-input story (a garbage input also produces a
  near-constant summarizer state).

---

## Attack 2: The E6-over-baselines win is an artifact of the LOCO protocol on N=2 corpora, not a real advantage

- CLASS: unfair comparison / cherry-picked metric (the protocol favors the proposed method).
- TARGET (draft location): Abstract, "where the location-based feature scores one would
  default to (Mahalanobis, Relative Mahalanobis, KNN) each hit 100% leave-one-corpus-out FPR
  and fail to transfer"; Section 5.6, "all three applicable location-based baselines ... hit
  100% LOCO FPR: a threshold calibrated on one corpus flags the entirety of the other"; the
  contribution sentence itself ends on this claim.
- THE ATTACK: On the metric that actually measures detection power, raw single-corpus AUROC,
  KNN-50 TIES the proposed monitor at 1.000 (the draft concedes this: "We do not claim to
  beat KNN-50: on single-corpus separation the two tie"). The ENTIRE distinguishing claim
  rests on one number: location-based baselines get 100% LOCO FPR while E6 gets ~1%. But LOCO
  here is a TWO-FOLD estimate over N=2 corpora (subaru, ram). With two corpora, "leave one
  out" means "calibrate on corpus A, test on corpus B, then swap," and the draft's own
  mechanism explanation (Section 5.6: "subaru and ram corpora occupy disjoint regions of the
  512-D feature space") says the two corpora are simply far apart in feature space. A
  location score calibrated on one absolute region and tested on a disjoint absolute region
  will trivially fire 100%, by construction, for ANY two sufficiently separated corpora. This
  is not evidence that E6 "calibrates" in any general sense; it is evidence that with N=2 you
  have a single train/test pair and the baselines happen to be position-sensitive while E6
  happens to be position-invariant. A reviewer will say: with N=2 this is one coin flip, not
  a measured property. Add a third corpus that happens to overlap subaru's region and the
  baselines' LOCO FPR could drop sharply; add one that is even more disjoint and E6 could rise.
  The "100% vs 1%" gap is the headline differentiator and it is measured on the thinnest
  possible protocol.
- EVIDENCE: c28, c29 confirm the arithmetic but the contract explicitly forbids treating it
  as more than it is: "N=2 real corpora; LOCO is a two-fold estimate whose variance is not
  meaningfully reportable" and "E6 beats / outperforms standard OOD baselines (Careful: KNN-50
  ties E6 at AUROC 1.000)." The reproducibility report confirms KNN is insensitive to k (k=5
  through k=100 all give AUROC 1.000), so the baseline is at full strength on raw separation;
  the only axis on which it loses is the two-fold LOCO. The draft frames this honestly in
  places, but the Abstract and contribution sentence still LEAD with the transfer claim as
  the differentiator, which invites exactly this attack.
- SEVERITY: MAJOR (at a main track, REJECT; at a workshop, this is the strongest "so what"
  objection to the monitor half of the paper).
- REJECT-PROBABILITY: high.
- REQUIRED RESPONSE: Two things. First, the honest rebuttal the authors CAN make and must
  state explicitly: "The LOCO result is a mechanism demonstration, not a generalization
  claim. The point is geometric and a priori: any absolute-position score is corpus-location
  dependent and a second-order trace is corpus-location invariant; the N=2 LOCO is an
  existence proof that the predicted failure occurs, not a measurement of its rate." That
  framing survives. Second, the concrete fix that would make it robust: add at least one more
  real corpus (the contract parks "a third real corpus" precisely for this), OR run a
  synthetic control that demonstrates the location-invariance property holds as corpus
  separation varies (sweep a synthetic shift between two clusters and show E6 FPR flat while
  Mahalanobis FPR climbs with separation). Without one of these, the authors must downgrade
  the Abstract and contribution sentence so the transfer claim reads as "a predicted and
  observed failure of position-based scores under corpus shift (N=2 demonstration)," never as
  a measured advantage.

---

## Attack 3: E7 (sim-specific collapse) quietly undercuts the contribution it is meant to bound

- CLASS: overclaim collides with its own bounding result (the bound eats the headline).
- TARGET (draft location): Section 5.7, "The collapse is sim-specific ... The silent collapse
  is a property of full-sim rendering, not of photometric or blur corruptions of real
  frames"; Abstract, "an ImageNet-C sweep shows the silent collapse is sim-specific (no
  corruption reproduces it ...)"; Section 1 framing, "a simulation 'pass' can be the model
  having collapsed to a safe-looking default."
- THE ATTACK: The paper sells the finding as a SAFETY result: production models validated in
  sim can silently fail, so a sim "pass" is not evidence. But E7 proves the failure mode is
  triggered ONLY by full-sim rendering and by no real-frame corruption whatsoever (0 of 75
  cells). A skeptical reviewer flips this: if the collapse happens only under full simulation
  and never under any corruption of real driving footage, then the real-world safety
  relevance is asserted, not shown. The phantom-braking field issue (commaai #20704) is
  caused by SHADOWS on real roads, a real-frame photometric event, exactly the class E7 shows
  does NOT induce the collapse. So the motivating real-world failure and the measured
  simulated failure may be two different phenomena, and the paper's own bounding experiment is
  the evidence that they are different. The contribution then shrinks to: "when you feed this
  model frames from this simulator, it collapses, and a cheap monitor catches that." That is a
  simulation-tooling caveat (useful to people validating openpilot in CARLA) but not the
  safety-of-deployed-systems result the framing promises.
- EVIDENCE: c33, c37 (CONFIRMED): "0 of 75 cells reach the collapse criterion ... vs 7 of 10
  under CARLA." The contract is explicit that the field issue is "motivation only" and forbids
  "any causal claim linking this collapse to specific openpilot field incidents," but the
  Introduction still uses the field issue to ground the stakes ("The failure mode this exposes
  is not hypothetical"), and E7 is the experiment that severs the link between the simulated
  collapse and the real-frame failure mode the issue describes. The draft does not connect
  these two facts, which is the gap.
- SEVERITY: MAJOR.
- REJECT-PROBABILITY: medium-high.
- REQUIRED RESPONSE: The authors must own the tension in the body, not let the reviewer find
  it. Add a sentence to Section 5.7 Synthesis and to Limitations: "Because the collapse is
  sim-specific and no real-frame corruption (including the shadow/photometric class of the
  motivating field issue) reproduces it, we make no claim that the measured collapse is the
  mechanism of any field phantom-braking event; the contribution is bounded to the
  sim-validation setting, where a simulator-rendered 'pass' can mask a collapsed model." Then
  reframe the stakes paragraph in Section 1 around sim-VALIDATION integrity (which E7 fully
  supports: people DO validate openpilot in CARLA) rather than around field deployment safety.
  This is survivable because the sim-validation framing is genuinely supported; the
  field-safety framing is not, and pretending otherwise is the rejection risk. There is no
  experiment that rescues the field-safety framing short of the parked real-adverse-weather
  collapse axis.

---

## Attack 4: N=1 model carries claims that the Abstract and Introduction phrase as general

- CLASS: N=1 generalization / overclaim by phrasing.
- TARGET (draft location): Abstract opening, "Production Level-2 driver-assistance stacks are
  validated largely in simulation, and that practice rests on an unstated assumption"; Section
  1 opening, "Every Level-2 and autonomous-driving program validates its driving policy
  largely in simulation"; Conclusion, "Output-side monitoring alone is insufficient for the
  safety case of this shipped driving model."
- THE ATTACK: The evidence is a single model (supercombo v0.9.7) on a single simulator. The
  Abstract and Introduction open with sweeping statements about "Production Level-2 stacks"
  and "Every Level-2 and autonomous-driving program," which primes the reader to read the
  finding as general before the N=1 caveat arrives. A reviewer who reads the opening as a
  claim about driving models broadly, then reaches "one deployed model," will feel the bait
  and switch and discount the whole paper. Even though the Limitations section is scrupulous
  (the contract and Limitations both nail N=1), the FRAMING does the overclaiming that the
  body retracts.
- EVIDENCE: Contract forbids "any generalization of the silent-collapse phenomenon or of E6
  beyond supercombo v0.9.7 (N=1 model)" and lists the exact tempting sentence to avoid:
  "We show that production driving models fail silently under distribution shift. (Wrong: N=1
  ... say 'a production driving model' and name it.)" The draft mostly complies in the body
  ("a single deployed model," "this one shipped model"), so the violation is concentrated in
  the framing sentences that generalize the MOTIVATION ("Every Level-2 ... program") even
  while the FINDING is correctly singular.
- SEVERITY: MINOR (the body is correctly bounded; this is a framing-discipline fix, not a
  claim that exceeds evidence in the results). It rises to MAJOR only if a reviewer reads the
  opening as the contribution.
- REJECT-PROBABILITY: medium (low alone, but it compounds Attack 1 by making the paper feel
  overclaimed).
- REQUIRED RESPONSE: Pure framing fix, no new experiment. Tighten the first sentence of the
  Abstract and Section 1 so the general statement is explicitly the MOTIVATING practice and
  the finding is explicitly singular in the same breath: e.g., "Level-2 stacks are validated
  largely in simulation; we test, on one such deployed model (openpilot v0.9.7 supercombo),
  whether that practice's core assumption holds." Keep every general verb attached to the
  practice, never to the finding. State once, early, that this is an N=1 case study whose
  generality is untested.

---

## Attack 5: E6 is offline-only and was never run in the loop, yet the framing implies a safety monitor

- CLASS: unsupported claim / scope (demonstrated artifact does not match implied artifact).
- TARGET (draft location): Section 3 Threat Model, "we position the proposed monitor as a
  cheap complementary layer"; Section 3, "The proposed monitor (Section 5.6) is a complement,
  not a replacement: it is one O(d) statistic computed from a forward pass that already runs";
  Conclusion, "a second-order recurrent-state monitor is a cheap complement."
- THE ATTACK: The Threat Model section is written as if the monitor is a runtime safety layer
  ("complementary layer," "computed from a forward pass that already runs," "calibrated
  against a real-driving false-positive rate"). But the monitor was only ever computed offline
  on logged/rendered/corrupted frames; it was never run inside the openpilot stack, never run
  in real time, and the 30-frame rolling window means it has a 1.5 s (at 20 Hz) detection
  latency that the draft never quantifies as an in-loop cost. A reviewer reading Section 3 as
  a deployment proposal will demand an in-stack latency/overhead measurement and a
  false-trigger-rate-in-the-loop number, find neither, and downgrade. The "fires 0.23
  blend-units before the cliff" early-warning claim is also only meaningful in the alpha-sweep
  construction; in a real drive there is no alpha axis, so the early-warning headroom is not
  demonstrated to exist on any real trajectory (and on the RAM segment it is already negative).
- EVIDENCE: Contract forbids "any on-road, in-stack, or real-robot deployment of E6 (it is
  demonstrated offline ...)" and lists the tempting sentence "Our monitor can be deployed on
  the vehicle to prevent phantom braking. (Wrong: E6 is offline-only.)" Limitations correctly
  says "demonstrated offline ... no on-road, in-stack, or real-time deployment was run." So the
  body is compliant, but Section 3's "complementary layer" language and the early-warning
  framing read as deployment claims. The reproducibility report confirms E6 is computed in
  `scripts/build_metrics.py` from cached scores, never in a closed loop.
- SEVERITY: MINOR to MAJOR depending on read (the Limitations rescue it, but Section 3 invites
  the demand).
- REJECT-PROBABILITY: medium.
- REQUIRED RESPONSE: Framing fix plus one honest concession. In Section 3, replace
  "complementary layer" deployment language with "an offline diagnostic that could in
  principle be computed in-loop, which we do not demonstrate here." Add to Section 5.6 and
  Limitations the concrete in-loop costs the monitor WOULD incur: the 30-frame (about 1.5 s at
  20 Hz) window latency, and the fact that the early-warning headroom is defined on the
  alpha-blend construction and is already negative on the RAM segment, so on a real trajectory
  the lead-time is not established. Do NOT add deployment language anywhere it does not already
  appear. The defensible claim is "the signal is present and cheap to compute"; the
  indefensible one is "the monitor is a runtime safety layer."

---

## Attack 6: Two FLAGGED numerical errors in the draft (n=219 vs 220, and the AUROC/fire-rate "same alpha" claim)

- CLASS: cherry-picked / loose metric reporting (small, but reviewers weaponize sloppiness).
- TARGET (draft location): (a) Abstract, "0 of 219," but Section 4 Method states the warm-up
  arithmetic as "320 frames, 100 discarded"; the draft mixes 219 and the 220 implied by its
  own method text. (b) Section 5.6, "the monitor's fired fraction crosses 50% at alpha=0.550
  ... its AUROC crosses 0.5 at the same alpha=0.550."
- THE ATTACK: A reviewer who recomputes ANY number and finds it off will distrust EVERY
  number. Two are flagged in the stats audit. (a) The draft's frame count is internally
  inconsistent: it says 0 of 219 in the Abstract but the Method section's "320 minus 100"
  arithmetic implies 220, and the npz actually stores 319 (post-warmup 219). (b) The "AUROC
  crosses 0.5 at the same alpha=0.550" claim is simply false: the stats audit recomputed AUROC
  = 0.400 at alpha=0.425 and 0.540 at alpha=0.450, so AUROC crosses 0.5 near alpha=0.45, about
  0.10 alpha-units BEFORE the fire-fraction crossing at 0.550. The draft overstates the
  coherence of the two metrics to make the early-warning story look tighter than it is.
- EVIDENCE: Stats audit c11 / IC-1 (FLAGGED): "The draft's '0 of 220' should read '0 of 219'
  ... the actual npz gives 219." Stats audit c31 / IC-6 (FLAGGED, FAIL): "the 'same alpha'
  claim is incorrect by approximately 0.10 alpha units ... AUROC actually crosses 0.5 ~0.10
  alpha units earlier than the fire-fraction." Both are confirmed by recomputation, not
  opinion. Also FLAGGED: c22/IC-7, the "passive relay within about 2%" claim where reduce_sum
  actually deviates 11.4% at alpha=1; and c32, where the reported n=638/319 are stored counts
  but only 609/290 are valid (non-NaN) scores entering the AUROC.
- SEVERITY: MINOR each, but collectively MAJOR for credibility (these are free points a
  reviewer scores against you).
- REJECT-PROBABILITY: low alone; medium as a credibility multiplier on Attacks 1-3.
- REQUIRED RESPONSE: Pure correction, no argument. (a) Make the frame count consistently 219
  everywhere and fix the Method arithmetic to "319 stored, 100 warm-up discarded, 219
  analysis frames." (b) Delete the "at the same alpha=0.550" claim; state the two crossings
  separately and correctly ("the fired fraction crosses 50% at alpha=0.550; the alpha-swept
  AUROC crosses 0.5 near alpha=0.45"). (c) Change "passive relays ... to within 2%" to "to
  within about 2 to 11% (reduce_sum deviating most, 11% at alpha=1)" or just cite the table.
  (d) In Method, state both the stored count (638/319) and the valid-score count (609/290)
  that actually enters the AUROC. These are non-negotiable: a reviewer who finds them
  un-fixed will assume the headline numbers are equally loose.

---

## Attack 7: Reproducibility gaps in the draft and README contradict the artifacts (gitignored/corrupt caches)

- CLASS: reproducibility gap (claimed-reproducible experiments do not reproduce from a fresh
  clone).
- TARGET (draft location): Section 8 Reproducibility Note, "Three supporting result caches are
  not committed to the public repo because of their size ... For each, a regeneration path is
  documented"; and "The cache-distribution decision is therefore settled." Also the draft's
  earlier TODO-marked version (ledger c49 FLAGGED, "[TODO: verify]").
- THE ATTACK: A reviewer who clones the repo to check E5, E7, or E4-RAM will hit
  FileNotFoundError on every one of them, because all three caches are gitignored, and the
  3.9 GB E5 cache is additionally CORRUPT on the author's own machine (truncated zip, no
  End-of-Central-Directory record). The draft Section 8 now asserts the matter is "settled"
  and that regeneration paths are "documented," but the reproducibility report shows Git LFS
  is NOT configured, so there is no working recovery path on a fresh clone, and the README
  still claims these analyses "run from the cache," which is false. The localization claim
  (E5), which is the structural heart of the "downstream of the encoder" finding that
  partially answers Attack 1, is the LEAST reproducible experiment in the paper: its cache is
  gitignored and corrupt and cannot be re-run on the author's machine either.
- EVIDENCE: Reproducibility report Gap 1 (CRITICAL): "Git LFS is not configured in the repo
  ... A fresh clone therefore cannot recover E5 or E7 caches." Gap 2 (MEDIUM): "README ...
  claims cache-based re-run for E5, E7, and E4-RAM; all three claims are false on a fresh
  clone." Gap 3 (MEDIUM): "e5_collected.npz ... has a valid ZIP header but no valid
  End-of-Central-Directory record. It cannot be opened." The report does confirm the HEADLINE
  caches (E1/E2/E3, E4, E6, ablations) reproduce in under a minute, which is a genuine
  strength to defend; the gap is concentrated in E5/E7/E4-RAM.
- SEVERITY: MAJOR (a workshop reviewer who hits a corrupt cache on the structural-localization
  experiment will not trust the localization claim, which is load-bearing for Attack 1's
  rebuttal).
- REJECT-PROBABILITY: medium.
- REQUIRED RESPONSE: This must be FIXED, not argued. (1) Replace the 3.9 GB E5 layer cache
  with the ~100 KB per-stage-per-alpha summary array the analysis actually reads (the repro
  report names this fix, L3) and commit it. (2) Commit E4-RAM (28 MB, under GitHub's 100 MB
  limit, no LFS needed). (3) Either configure and test Git LFS for the E5-submodule (98 MB)
  and E7 (110 MB) caches and verify `git lfs pull` on a fresh clone, OR state plainly in
  Section 8 and the README that those two experiments require a `--collect` pass (GPU + data)
  and are not cache-reproducible. (4) Remove every "settled" / "documented regeneration path"
  assertion that is not actually true on a fresh clone, and remove the README's "runs from the
  cache" lines for E5/E7/E4-RAM. The honest posture (headline reproduces in under a minute;
  three supporting experiments need a collect pass) is defensible; the current overstated
  posture is a desk-level credibility hit.

---

## Attack 8: No real adverse-weather OOD axis, so the most natural reviewer-requested control is conspicuously absent

- CLASS: missing experiment / weak adverse-condition coverage.
- TARGET (draft location): Limitations, "real adverse-weather footage (rain, night, glare)
  that actually induces a non-CARLA collapse remains pending, and is the most
  reviewer-resistant follow-up"; Section 5.7 is the only non-CARLA OOD axis and it is
  synthetic corruptions, not real adverse conditions.
- THE ATTACK: The single most obvious experiment for "does a driving model fail silently under
  visual OOD" is real night / rain / glare footage, the conditions that actually cause field
  failures (and the motivating shadow issue is exactly this class). The paper does not have
  it. ImageNet-C corruptions of clean daytime frames are a poor proxy for real adverse
  driving conditions (they are synthetic perturbations of in-distribution scenes, not new
  scenes). A reviewer will read "the most reviewer-resistant follow-up remains pending" as the
  author conceding the experiment that would have made the paper, was not done.
- EVIDENCE: Contract parks this as out-of-scope: "A second, real adverse-weather OOD axis
  (rain/night/glare comma footage) that actually induces a non-CARLA output collapse ... is
  curation-gated and deferred." E7 (c33) confirms synthetic corruptions do NOT reproduce the
  collapse, which strengthens the case that the missing axis is the only one that could test
  real-world relevance.
- SEVERITY: MAJOR at a main track (this is THE missing baseline-experiment); MINOR-to-MAJOR at
  a workshop, where a single-axis case study is acceptable if honestly scoped.
- REJECT-PROBABILITY: medium (lower at the target workshop tier, higher if the paper is read
  as a deployment-safety result rather than a sim-validation diagnostic).
- REQUIRED RESPONSE: The authors cannot add the experiment in rebuttal (it is curation-gated),
  so they must pre-empt it: state in the Introduction (not just Limitations) that this is a
  controlled single-axis (CARLA) case study, that the real-adverse-weather axis is explicitly
  deferred, and that the contribution is the existence and mechanism of the silent-collapse
  mode plus a cheap monitor for it, NOT a claim of coverage over real OOD conditions. The
  honest scoping is the only defense. If any reviewer time/budget exists, even a small curated
  real-night-or-glare clip that shows the collapse does or does not occur would convert this
  from a fatal gap to a strength; without it, the framing must not imply real-world OOD
  coverage.

---

## Attack 9: The "partial localization" (E5) leaves the mechanism ambiguous, weakening the one structural answer to the confound

- CLASS: weak-or-absent ablation (the mechanism probe does not fully resolve).
- TARGET (draft location): Section 5.5, "The localization is partial: the summarizer ends in a
  mu-over-sigma variational reparameterization, and we have not separated the mu path from the
  sigma normalization, so part of the apparent summarizer collapse could be variance
  normalization rather than information loss."
- THE ATTACK: E5 is the paper's only structural defense against Attack 1 (it argues the
  failure is "downstream of the encoder," i.e., not just the encoder choking on a malformed
  image). But E5 itself is explicitly incomplete: the authors cannot say whether the
  summarizer "collapse" is information loss or merely variance normalization in the VAE
  reparameterization. If it is variance normalization, then the "collapse" the monitor reads
  (a drop in feature spread) could be a benign artifact of the bottleneck's sigma scaling
  rather than a genuine loss of scene information, which would partially deflate both the
  mechanism story and the monitor's interpretation. A reviewer will press: "Your monitor reads
  feature spread; your own localization says the spread drop might be variance normalization,
  not information loss. So what exactly is your monitor detecting?"
- EVIDENCE: c23 (CONFIRMED), the VAE mu/sigma ambiguity is real and stated. The stats audit
  also flags c22 (the "passive relay within 2%" overstatement) in the same experiment, so the
  one experiment that localizes the mechanism has two flagged soft spots.
- SEVERITY: MINOR-to-MAJOR (it is honestly disclosed, which helps, but it weakens the rebuttal
  to the central attack).
- REJECT-PROBABILITY: low-to-medium.
- REQUIRED RESPONSE: A targeted ablation the authors likely CAN run from existing tensors:
  separate the summarizer mu from the sigma path and report whether the mu (the information
  channel) collapses independently of sigma. If mu collapses, the information-loss
  interpretation holds and the monitor's reading is vindicated; if only sigma moves, the
  authors must reinterpret what the monitor detects. If the ablation cannot be run, the
  required rebuttal is to state plainly that the monitor is an empirical detector of the
  spread shift regardless of its mu/sigma decomposition, and that the spread shift is what
  separates the classes (AUROC 0.996) whether or not it is "information loss," decoupling the
  monitor's validity from the unresolved mechanism. Either path closes the opening.

---

## Summary

- Attacks by severity: 1 REJECT-grade (Attack 1), 4 MAJOR (Attacks 2, 3, 6-as-credibility,
  7), 1 MAJOR-at-main/MINOR-at-workshop (Attack 8), 3 MINOR-to-MAJOR (Attacks 4, 5, 9).
  Counted strictly by the worst plausible read: 1 REJECT, 5 MAJOR, 3 MINOR.
- The single attack most likely to sink this paper: Attack 1. The collapse has not been
  separated from a CARLA camera-model / rendering-pipeline confound, and the paper's own
  sim-specific result (E7) is consistent with that confound rather than refuting it. Every
  headline use of "fails silently under visual out-of-distribution input" is exposed until a
  geometry-and-photometry-matched control is run or the claim is narrowed to "collapses on
  clean CARLA renders through this inference path."
- Attacks with no available rebuttal (must be fixed, cannot be argued away):
  - Attack 6 (the n=219/220 inconsistency and the false "AUROC crosses 0.5 at the same
    alpha=0.550" claim): these are recomputed errors; correct them.
  - Attack 7 (gitignored + corrupt caches, Git LFS not configured, README/Section-8 assert a
    "settled" recovery path that does not exist on a fresh clone): fix the repo or correct the
    prose; cannot be argued.
  - Attack 1's minimum survival bar (narrow the headline claim to the literal evidence) is a
    fix, not an argument; the full rebuttal (a matched-camera control) is an experiment, not a
    sentence.

---

## Honesty summary

What I verified: I read the contribution contract, the claim ledger, the full rewritten
draft, and all three audits (stats, source, reproducibility) end to end. Every attack is
anchored to a quoted draft location and cross-checked against a ledger row or an audit
finding. The FLAGGED numbers (c11 n=219/220, c31 AUROC-vs-fire-rate same-alpha, c22 passive
relay 2% vs 11%, c32 638/319 vs 609/290) are taken directly from the stats audit's
recomputations, not from my own recomputation. The reproducibility gaps (gitignored E4-RAM/
E5-submodule/E7 caches, corrupt 3.9 GB E5 cache, Git LFS not configured) are taken directly
from the reproducibility report's run logs. The citation claims I did not re-verify; I relied
on the source_verification.md CONFIRMED verdicts and did not independently fetch any source.

What I did not verify: I did not run any code, open any npz, or fetch any URL myself; I did
not inspect the actual CARLA frames or the openpilot sim-bridge camera transform to confirm
the intrinsics/photometry confound concretely (Attack 1 is argued from the absence of a
control in the draft and from E7's sim-specificity, not from a measurement I performed). I
did not confirm whether the summarizer mu/sigma ablation (Attack 9) is runnable from existing
tensors; I inferred it is likely runnable from the E5 submodule cache description.

Riskiest unverified assumption in my output: that CARLA-clean input to supercombo is in fact
materially mismatched in camera intrinsics/photometry from the model's training distribution
(the premise of Attack 1). I did not measure this; it is the standard, well-grounded concern
for sim-as-OOD work, and the draft's own E7 sim-specificity result is consistent with it, but
if the authors can show the CARLA frames were rendered through comma's documented production
sim camera transform (matching what openpilot itself uses to validate in CARLA), then Attack 1
weakens from "uncontrolled confound" to "a confound the authors have already controlled," and
the required response shrinks to citing that control. That single fact is the hinge of the
whole review and I could not check it from the files provided.
