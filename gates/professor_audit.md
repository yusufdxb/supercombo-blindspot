# Professor Audit (Hybrid)

Stage 15 of the paper-council, the PI / area-chair gate. Owner: this agent. Run date 2026-05-30.
Paper dir: /home/yusuf/Projects/phantom-braking. Draft audited: `drafts/rewritten_draft.md`.

Engines: codex (read-only sandbox) + gemini (plan / read-only) + Claude (this agent).
Runner status this run: `codex_STATUS=ok`, `gemini_STATUS=ok` (both cross-examiners live, both
returned non-empty output; no engine outage, no re-run needed). Raw outputs:
`gates/raw/codex.txt`, `gates/raw/gemini.txt`. Prompt: `gates/raw/professor_audit_prompt.md`
(the full draft + contribution boundary + 51/51-CONFIRMED ledger + the specific five-dimension
PI question, including the explicit instruction to weigh whether the c50 CARLA-confound control
is sufficient).

This audit judges the ARGUMENT, not the plumbing. The repo readiness (baselines present, metrics
computed, caches reproduce) is owned by the reproducibility report and is assumed, not redone. The
adversarial attacks are owned by `gates/reviewer2_redteam.md`; this gate is the constructive
complement that asks whether the paper as a whole earns publication at its locked tier (arXiv
preprint / SafeAI-workshop-tier teardown + monitor).

Reconciliation rule (conservative): a dimension passes only if NONE of codex, gemini, or Claude
marks it FATAL. Any single FATAL blocks the gate. A failed or empty engine is itself blocking.

---

## Contribution

- Is the contribution REAL? YES. This is a genuine advance for its tier, not a re-skin. The
  load-bearing move is a parity-exact reimplementation of a SHIPPED production model (openpilot
  v0.9.7 supercombo), verified to within +/-0.5 m/s^2 of comma's own reference on 100% of 1159
  real frames (median abs delta 0.0409 m/s^2, ledger c1/c2 CONFIRMED, reproducibility report
  REPRODUCED in under a minute). That harness-trust step is what converts a "the model does
  something weird on sim" anecdote into a defensible negative result: any downstream anomaly is
  attributable to the model, not the reimplementation. On top of it sits a real, non-obvious
  phenomenon (simultaneous output-head collapse + recurrent-feature freeze + non-responsive
  uncertainty channel, c6/c8/c10/c11), a localization downstream of the encoder (c18/c19), and a
  cheap zero-retraining monitor that recovers the hidden signal (c24/c28). The contribution is the
  bounded claim that output-side and location-based signals alone are insufficient for THIS one
  shipped model's safety case, and that a second-order recurrent-state trace is a cheap complement.
  That is a real, useful, falsifiable contribution at workshop / preprint tier.
- Does it match the locked sentence and stay inside the boundary? YES. The abstract, the four
  enumerated contributions in Section 1, and the conclusion all carry the N=1 / N=2 / sim-specific
  / collapse-specific / offline-only / partial-localization qualifiers verbatim from the contract.
  The hallucination gate (stage 14) confirmed no claim exceeds the boundary and the only "first" is
  the contract-allowed narrow LOCO claim. I re-verified the intro opening: the general verb ("Most
  Level-2 ... programs validate ... in simulation") is attached to the MOTIVATING practice, and the
  finding is made singular in the very next sentence ("We test that assumption on a single deployed
  model, openpilot v0.9.7 supercombo"). Inside the boundary.

## Evidence

- Does the evidence reach the contribution, or only a weaker adjacent claim? It reaches the
  contribution AS LOCKED, because the locked contribution is already narrowed to exactly what the
  evidence supports. The honest gap (named, not hidden): the evidence proves silent collapse on
  CARLA-rendered CONTENT through a now-intrinsics-matched inference path. It does NOT prove the
  collapse is triggered by semantic out-of-distribution (unfamiliar road geometry / scene meaning)
  rather than by the residual rendered-content domain gap (synthetic photometry, texture, absence of
  sensor noise). The c50 control (matched fcam intrinsics, warp collapsing to K_fcam @ inv(K_medmodel),
  sim ZERO_CALIB vs real liveCalibration as the only asymmetry) closes the camera-MODEL confound that
  was Reviewer-2's reject-grade attack, and E5 (collapse downstream of the encoder, c18/c19) rules out
  "the encoder choked on a malformed image." What remains open is content-domain-gap vs semantic-OOD,
  and the draft does NOT overclaim past it: it says "the collapse is a response to rendered scene
  content" and bounds real-world transfer via E7. So the evidence supports the locked claim; it does
  not support the unlocked stronger claim, and the paper does not assert the stronger one. That is the
  correct posture for this tier.
- Which load-bearing claims are not yet CONFIRMED? NONE. The ledger is 51/51 CONFIRMED, each with a
  recorded quote/number; the integrity check returns `OK: all claims confirmed`; the headline caches
  (E1/E2/E3, E4, E6, ablations) reproduce from a fresh clone. The four numbers the stats audit once
  FLAGGED (c11 0/219, c22 2-to-11%, c31 fires-at-0.550 with the false same-alpha-AUROC clause removed,
  c32 638/319 stored vs 609/290 valid) are all now corrected in the rewritten draft and re-set to
  CONFIRMED. The argument has a floor.

## Novelty

- Is the novelty bounded and defensible against the closest neighbor? YES, defensible. The litmap's
  collision hunt found no blocking pre-emption on any of the four attack surfaces. The single closest
  mechanism-twin, EigenTrack (arXiv:2509.15735), is correctly conceded to PRE-DATE the second-order
  hidden-activation framing, and the draft never writes the forbidden "first to use a second-order
  hidden-activation statistic." The only "first" claimed is the bounded one: first second-order
  recurrent-state monitor on a SHIPPED end-to-end driver under cross-corpus leave-one-corpus-out
  transfer, with substrate (driving vs LLM/VLM), statistic (single location-invariant trace vs full
  eigenspectrum + trained classifier), and evaluation-axis (LOCO FPR vs language benchmarks) deltas all
  stated. Against the strongest baseline, the draft concedes KNN-50 TIES E6 at AUROC 1.000 and never
  claims to beat it; the real, narrower claim is transfer/calibration (100% LOCO FPR for location-based
  scores vs ~1% for E6). The novelty is the union of the contrast-table deltas, and no single named
  neighbor has all of them. Bounded and honest.

## Limitations

- Are the limitations honest, or does the real weakness go unstated? HONEST. Section 6 states the four
  real weaknesses explicitly: N=1 model (no generalization claimed), N=2 corpora (LOCO is a two-fold
  estimate, ~1% is a calibration estimate not a production FPR), collapse-specific + offline-only
  monitor (near chance on most corruptions; never run in-stack; the most reviewer-resistant follow-up,
  a real adverse-weather collapse axis, explicitly deferred), and partial localization (the VAE
  mu/sigma ambiguity unresolved). The one weakness a careless paper would bury, that the collapse is
  sim-specific and no real-frame corruption (including the shadow/photometric class of the motivating
  field issue) reproduces it, IS surfaced in both Section 5.7 Synthesis and Limitations, and the
  phantom-braking field issue is held to motivation-only with no causal claim. The residual confound
  (content-domain-gap vs semantic-OOD) is the one weakness stated more softly than a pure skeptic would
  write it: the draft frames the shift as "rendered scene content" but does not add a flat sentence
  saying "whether the trigger is semantic OOD or a rendered-content domain gap is unresolved." That is
  a sharpening opportunity, not a dishonesty (the c50 control and E5 are reported, and no stronger
  claim is made), so it does not rise to FATAL.

## Story

- Does the narrative land? YES. The arc is coherent and well-ordered: simulation is the primary
  validation setting -> the shipped model fails silently on simulated input -> the output-side and
  uncertainty monitors a safety case would trust all miss it (the Threat Model walks each defense and
  shows why) -> the signal is recoverable from the model's own recurrent feature with one cheap
  statistic -> the result is useful but explicitly narrow. Parity establishes trust before any anomaly
  is attributed (the right dependency order), and the experiments unfold in the order the argument
  requires (E1 phenomenon, E2/E3 the gap, E4 shape, E5 mechanism, E6 solution, E7 bound). The one place
  the story has tension is that E7 (the bounding result) partially undercuts the field-safety framing:
  if the collapse is sim-specific and never reproduces under real-frame corruption, the real-world
  deployment stakes are asserted more than shown. The draft now owns this tension by reframing the
  stakes around SIM-VALIDATION integrity ("a simulation pass can be the model collapsed to a
  safe-looking default"), which E7 fully supports, rather than around field deployment. The story
  coheres; it does not break.

## Per-engine verdict table

| dimension | codex | gemini | claude | reconciled |
|---|---|---|---|---|
| contribution | OK (real, bounded, single-model teardown + cheap monitor) | OK (rare valuable negative result on shipped code) | OK (parity-anchored real advance, inside boundary) | **OK** |
| evidence | OK (reaches CARLA-content OOD claim, not semantic-OOD generalization) | OK (parity is gold standard; gap = sim-pixels vs semantic-OOD) | OK (reaches locked claim; content-vs-semantic gap named, not overclaimed; 51/51 floor) | **OK** |
| novelty | OK (correctly "first ... under cross-corpus LOCO," not "first second-order") | OK (bounded vs EigenTrack and KNN-50) | OK (union-of-deltas, no blocking collision, KNN tie conceded) | **OK** |
| limitations | OK (N=1, N=2, offline, CARLA-only, no production FPR all stated) | OK (refreshingly honest on N=1/N=2/sim-specificity) | OK (all four stated; residual content-vs-semantic stated softly, not buried) | **OK** |
| story | OK (arc lands: sim assumption -> silent collapse -> monitors fail -> spread recovers) | OK (narrative arc bulletproof) | OK (coheres; E7-vs-field-stakes tension owned via sim-validation reframing) | **OK** |

Engine summary. Codex returned five OKs and named the single most likely reject reason as "the
finding may be too CARLA/rendered-content-specific to matter beyond 'this model dislikes one
simulator's visual domain,'" graded FIXABLE (value framing, not plumbing; sharpen the title/abstract/
intro to say renderer-content-domain silent collapse). Gemini returned an OVERALL ACCEPT with five
OKs and named the single most likely reject reason as "the N=2 / two-fold LOCO statistical thinness"
(a ~1% FPR from a single leave-one-out fold over two corpora reads as an anecdote, not a calibration),
graded FIXABLE (add a third/fourth real corpus). Claude (this agent) independently read the draft,
the contract, the 51/51 ledger, the c50 control in src/sim_preprocessor.py context, and the intro
framing sentences, and concurs OK on all five dimensions. No engine marked any dimension FATAL.

## Gate decision

PASS.

No dimension is marked FATAL by codex, gemini, or Claude. The argument is real, the evidence reaches
the locked (already-narrowed) claim with a 51/51-CONFIRMED floor, the novelty is bounded honestly
against EigenTrack and the KNN-50 tie, the limitations are honest, and the story coheres. The one
reject-grade adversarial attack on record (Reviewer-2 Attack 1, the CARLA camera-model confound) is
substantially controlled by c50 (matched fcam intrinsics, warp collapse to K_fcam @ inv(K_medmodel),
sim-zero-vs-real-liveCalibration asymmetry stated) and structurally backstopped by E5 (collapse
downstream of the encoder), and the paper does not overclaim past what that control buys.

No fatal flaws to list. The three engines split on which non-fatal reason a committee would most
return to, and BOTH named reasons are FIXABLE, neither FUNDAMENTAL:
- codex: the finding may read as too CARLA/rendered-content-specific (value framing). Owner:
  paper-stanford-framer (sharpen title/abstract/intro to "renderer-content-domain silent collapse"
  and lead the stakes with sim-validation integrity, which E7 supports, not field deployment).
- gemini: the N=2 two-fold LOCO reads as an anecdote not a calibration. Owner: contribution-locker /
  the verification agents (the contract already parks "a third real corpus" for exactly this; until
  then the draft's "N=2 two-fold estimate, not a production FPR" qualifier is the honest mitigation
  and is present everywhere the 1% appears).
- Claude (additive sharpening, non-blocking): add one flat sentence to Section 5.1 and Limitations
  stating that whether the trigger is semantic OOD or a residual rendered-content domain gap is
  unresolved. The c50 control and E5 already make this defensible; the sentence converts a soft
  framing into an explicit bound. Owner: paper-stanford-framer.

THE SINGLE MOST LIKELY REASON THIS PAPER GETS REJECTED: a skeptical reviewer decides the finding is
too narrow to matter, a silent collapse triggered ONLY by one simulator's rendered content (E7 shows
no real-frame corruption reproduces it) and a ~1% FPR resting on a two-fold N=2 estimate, so the
contribution shrinks from "production models can fail silently" to "this one model dislikes CARLA's
visual domain, and a cheap monitor catches that"; this is the value/scope objection (codex's
framing axis fused with gemini's statistical-thinness axis), and it is FIXABLE (framing + one more
corpus or the parked real-adverse-weather axis), not fundamental, at the locked workshop / preprint
tier where a single-axis honestly-bounded negative finding clears the bar.

---

## Honesty summary

- What I VERIFIED this run: (1) Both cross-examiners ran LIVE (`codex_STATUS=ok`, `gemini_STATUS=ok`,
  both non-empty); I read both raw outputs in full (`gates/raw/codex.txt`, `gates/raw/gemini.txt`).
  (2) The ledger is 51/51 CONFIRMED with recorded quotes and the prior FLAGGED numbers (c11, c22, c31,
  c32) are corrected in the rewritten draft (cross-checked against the stats audit and hallucination
  gate). (3) The c50 CARLA-confound control wording matches src/sim_preprocessor.py per the
  hallucination gate's verbatim code read, which I relied on. (4) I independently read the intro
  framing sentences (the general verb is on the practice, the finding is singular), confirmed no TODO
  markers remain in the draft, and confirmed Section 3's "complementary layer" language is immediately
  bounded as "demonstrated offline only" with Limitations explicit. (5) I synthesized all prior
  reports (framing memo, literature map, stats audit, source verification, reproducibility report,
  reviewer2 red-team, hallucination gate) and judged the argument built on their established findings.
- What I did NOT verify: I did not re-run any analysis or re-open any npz (the reproducibility report
  owns that and ran the headline caches; I trusted its REPRODUCED verdicts). I did not re-fetch the
  external citation sources (the source-verification report owns that; I relied on its CONFIRMED
  quotes). I did not myself inspect the raw CARLA frames or the openpilot sim-bridge transform beyond
  the c50 ledger row and the hallucination gate's code read.
- THE SINGLE RISKIEST UNVERIFIED ASSUMPTION: that the c50 control is as tight as the ledger and the
  hallucination gate record it, i.e., that the CARLA frames genuinely render at the matched fcam
  intrinsics with the warp collapsing to K_fcam @ inv(K_medmodel) and no other geometric/photometric
  mismatch sneaks in upstream of the model input. If that control is weaker than recorded, Reviewer-2
  Attack 1 reopens from "controlled confound" back toward "uncontrolled confound," and the
  evidence-dimension verdict would need re-grading toward the value/scope rejection becoming harder to
  fix by framing alone. I read the ledger row and relied on the stage-14 verbatim code read rather than
  re-reading src/sim_preprocessor.py myself this run; that single fact is the hinge of the central
  review and is the one I leaned on another stage's verification for.
