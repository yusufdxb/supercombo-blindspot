# Hallucination Gate (Hybrid)

Stage 14 of the paper-council, RE-RUN after the orchestrator applied the prior-run fixes.
Owner: paper-hallucination-flagger. Run date: 2026-05-30.
Paper dir: /home/yusuf/Projects/phantom-braking.

Engines: codex (read-only sandbox) + gemini (plan / read-only) + Claude (this agent).
Runner status this run: `codex_STATUS=ok`, `gemini_STATUS=ok` (both cross-examiners live; no
engine outage, no re-run needed). Raw outputs: gates/raw/codex.txt, gates/raw/gemini.txt.
Ledger integrity: `python3 ledger.py check paper_state/claim_ledger.md` returns
`OK: all claims confirmed` (51/51 CONFIRMED, each with a recorded quote_or_number).

Reconciliation rule (conservative): a claim PASSES only if (a) none of codex, gemini, or Claude
flags it AND (b) it carries a CONFIRMED ledger row with a recorded supporting quote/number. Any
single engine flag, or any missing quote, BLOCKS that claim. Disagreement resolves toward BLOCKED.

---

## Outcome at a glance

- All nine prior flags (D1..D9) now CLEAR. All three engines agree D1..D9 PASS, and I
  independently confirmed each fix is present in the draft and grounded (citation fixes against
  the recorded source_verification quotes; the c51 DeepRoad-line row CONFIRMED; structural-argument
  disclosures intact).
- All 51 ledger-tied claims (c1..c51) carry CONFIRMED status with a recorded quote/number, and the
  recently changed numbers (c11, c22, c31, c32) re-verify against the artifacts.
- ONE NEW flag appeared: c50's calibration-identical wording. Codex flagged it (pattern 1, a fact
  the recorded artifact does not support); I independently concur after reading
  src/sim_preprocessor.py; gemini passed it. Per the conservative single-flag rule this BLOCKS c50.

The gate is BLOCKED on exactly one narrow, one-clause overclaim (c50). It is clearable without any
new artifact by deleting/qualifying the word "calibration" in one sentence.

---

## Claim inventory

Claims c1..c51 are the ledger-tied factual/quantitative/citation/novelty claims that appear in the
draft, all CONFIRMED with a recorded quote_or_number. The D1..D9 rows are the nine NON-ledger prose
sentences flagged in the prior run; each is re-inventoried with its current (post-fix) draft text
and location. Only the rows that changed this run, plus the newly flagged c50 clause, are spelled
out; the unchanged c1..c49/c51 rows are summarized.

| id | claim (verbatim, trimmed) | type | ledger status | recorded quote present? |
|---|---|---|---|---|
| c1..c49 | parity, collapse, freeze, uncertainty-silence, cliff, RAM-gradient, localization, monitor, baselines, corruption-sweep, citation rows (all unchanged this run) | fact/number/citation/novelty | CONFIRMED (each) | yes (each) |
| **c11** | "0 of 219 ... exceeds the real-driving 95th percentile" (220->219) | number | CONFIRMED | yes (0/219) |
| **c22** | "transformer ... and reduce-sum stages track the summarizer to within 2 to 11% ... passive relays" | number | CONFIRMED | yes (att 3.2%, trans 2.7%, reduce_sum 11.4%) |
| **c31** | "fired fraction crosses 50% at alpha=0.550 ... cliff ... about alpha=0.784 ... gap ... about 0.23" (false same-alpha-AUROC clause removed) | number | CONFIRMED | yes (fired 0.517 @ 0.550; gap 0.234; no same-alpha clause) |
| **c32** | "in-distribution set ... (n=638 stored) ... OOD (n=319 stored); ... 609 and 290 valid (non-NaN)" | fact | CONFIRMED | yes (638/319 stored; 609/290 valid; n=1000 seed=42) |
| **c50** | "the warp ... reduces to the same intrinsic remap applied to real footage ... camera geometry, **calibration**, and model-input preprocessing are held identical to the parity-verified real path" (Section 4, Data, L290-294) | fact | CONFIRMED (but for INTRINSICS + warp-collapse, NOT for "calibration held identical") | partial: the quote backs matched fcam intrinsics + the `K_fcam @ inv(K_medmodel)` warp, NOT "calibration held identical" |
| **c51** | "The DeepRoad line (DeepTest, DeepRoad, MarMot) uses metamorphic/generative test synthesis ... implicitly treating the generated scene as a valid input" (Related Work, "Simulation testing", L184-191) | citation | CONFIRMED | yes (DeepTest/DeepRoad/MarMot fetched quotes in source_verification.md [deeproad-line]) |
| D1 | "the end-to-end network in comma's shipped openpilot driver-assistance system (Chen et al. 2022)" (intro, L56-57) | fact (motivation) | n/a (anchored to c47) | n/a (now citation-anchored) |
| D2 | "Most Level-2 and autonomous-driving programs validate their driving policy largely in simulation" (intro, L46) | fact (motivation) | n/a (hedged) | n/a |
| D3 | "simulation is a primary setting in which rare and dangerous scenarios can be exercised at scale and at low cost" (intro, L46-47) | fact (motivation) | n/a (hedged) | n/a |
| D4 | "an assumption that is rarely tested directly in the literature we surveyed" (intro, L48) | fact (motivation) | n/a (scoped to surveyed lit) | n/a |
| D5 | "an ensemble of the same model would collapse together rather than disagree" (threat model, L235-236) | fact (structural argument) | n/a (disclosed) | derived from c18/c19 |
| D6 | "CARLA-clean renders are typically sharper and less noisy ... an image-quality screen would rate the simulated input as good" (threat model, L236-238) | fact (structural argument) | n/a (disclosed) | n/a |
| D7 | "These methods generate or transform driving scenes and test the model for consistent behavior, implicitly treating the generated input as a valid scene" (related work, L189-191) | citation | CONFIRMED via c51 | yes (now quote-backed) |
| D8 | "An early move to a higher-order feature statistic for OOD is the Gram-matrix method (Sastry and Oore 2020)" (related work, L154-155) | novelty/lineage | source CONFIRMED; superlative removed | yes (no "first") |
| D9 | "NECO (Ben Ammar et al. 2024) builds on a neural-collapse property of classification heads, which supercombo's regression heads lack" (related work, L171-173) | citation | source CONFIRMED; reframed as structural exclusion | yes (structural exclusion, not quoted property) |

---

## Per-engine verdict table

| id | codex | gemini | claude | reconciled |
|---|---|---|---|---|
| c1..c49 | pass | pass | pass | PASS |
| c11 (0/219) | pass | pass | pass | PASS |
| c22 (2 to 11%) | pass | pass | pass | PASS |
| c31 (fires 0.550; no same-alpha clause) | pass | pass | pass | PASS |
| c32 (638/319 stored; 609/290 valid) | pass | pass | pass | PASS |
| **c50 (camera-intrinsics control: "calibration ... held identical")** | **FLAG** | pass | **FLAG** | **BLOCKED** |
| c51 (DeepRoad-line, was D7) | pass | pass | pass | PASS |
| D1 (comma shipped-system phrasing) | pass | pass | pass | PASS |
| D2 (Most Level-2 hedge) | pass | pass | pass | PASS |
| D3 (a primary setting hedge) | pass | pass | pass | PASS |
| D4 (rarely-tested-in-surveyed-lit hedge) | pass | pass | pass | PASS |
| D5 (ensemble "would collapse together") | pass | pass | pass | PASS |
| D6 (image-quality "would rate ... good") | pass | pass | pass | PASS |
| D7 (DeepRoad-line citation = c51) | pass | pass | pass | PASS |
| D8 (Sastry "An early move") | pass | pass | pass | PASS |
| D9 (NECO structural exclusion) | pass | pass | pass | PASS |
| novelty boundary (EigenTrack pre-date, KNN tie, single "first") | pass | pass | pass | PASS |
| baselines (Maha/RMD/KNN/PCA run; MSP/Energy/ViM excluded) | pass | pass | pass | PASS |

Engine summary. Codex returned OVERALL: FLAG with exactly one flag, the c50 calibration-identical
wording, and explicitly PASSED all of D1..D9, c11, c22, c31, c32, the novelty boundary, and the
baselines. Gemini returned OVERALL: PASS, confirming all of D1..D9 and the spot-checked numbers
including c50. Claude (this agent) independently confirmed D1..D9 clear and independently concurs
with codex on c50 after reading src/sim_preprocessor.py (see the blocking flag). Per the
conservative rule, codex's single flag on c50 blocks that claim regardless of gemini's pass; and
here Claude concurs, so c50 is a two-engine flag, not a lone-engine conservative block.

---

## Blocking flags

All nine prior flags (D1..D9) CLEAR this run. The one new blocking flag is below.

### c50 (NEW this run; HARD block, calibration-identical overclaim). Camera-intrinsics control.

- Claim (verbatim, Section 4 Data, L290-294): "with the camera mounted on the device axes the warp
  to the model frame reduces to the same intrinsic remap applied to real footage
  (`src/sim_preprocessor.py`). The distribution shift is therefore confined to rendered image
  content (photometry, texture, and the absence of sensor noise), while camera geometry,
  **calibration**, and model-input preprocessing are held identical to the parity-verified real
  path; the collapse is a response to rendered scene content, not an artifact of a mismatched
  camera model."
- Draft location: drafts/rewritten_draft.md, Section 4 (Method), the Data paragraph, lines 290-294.
- WHY BLOCKED: unsupported fact (fabrication pattern 1). The ledger c50 recorded quote and the
  src/sim_preprocessor.py docstring support matched fcam INTRINSICS and the warp collapsing to a
  pure intrinsic remap, but they do NOT support "calibration ... held identical to the
  parity-verified real path." The artifact is explicit that calibration is precisely NOT identical:
  real openpilot feeds `get_warp_matrix(device_from_calib_euler, fcam.intrinsics)` where the euler
  comes from `liveCalibration` (a per-run, non-zero measured extrinsic), whereas in sim
  `ZERO_CALIB = np.zeros(3)` and "the calibration euler is exactly zero and the warp collapses to
  `K_fcam @ inv(K_medmodel)` -- a pure intrinsic remap, no rotation. That is why no per-run
  calibration is needed in sim." So sim uses ZERO calibration by construction (correct, because the
  camera is mounted on device axes), which is a DIFFERENT calibration value than the real path, not
  an identical one. The control is sound (zero extrinsic is the right thing for a device-mounted
  sim camera), but the word "calibration" in the "held identical" conjunction overstates it.
  RAISED BY: codex and claude (independent concurrence after reading the artifact). Gemini passed
  it, reading "camera geometry ... held identical (Attack-1 control)" as the matched-intrinsics
  setup; the codex/Claude reading is stricter and correct on the artifact text.
- REQUIRED FIX (route to paper-results-stats-verifier / drafter, low severity, no new artifact):
  narrow the conjunction so it claims only what the artifact supports. Replace "while camera
  geometry, calibration, and model-input preprocessing are held identical to the parity-verified
  real path" with wording that states matched intrinsics + identical preprocessing, and that the
  sim camera uses ZERO calibration (mounted on device axes) so the warp reduces to the same
  intrinsic remap, e.g.: "while the camera intrinsics and the model-input preprocessing are held
  identical to the parity-verified real path, with the sim camera using zero calibration (mounted
  on device axes) so the warp reduces to the same intrinsic remap; the collapse is a response to
  rendered scene content, not an artifact of a mismatched camera model." This keeps the Attack-1
  control intact while removing the unsupported "calibration held identical" assertion. Once the
  prose matches src/sim_preprocessor.py, update the c50 ledger evidence note if needed and this
  clears to PASS.

---

## Gate decision

BLOCKED.

All nine prior flags (D1..D9) CLEAR: D7's DeepRoad-line citation is now quote-backed
(source_verification.md [deeproad-line / c51], ledger c51 CONFIRMED) with the "valid-scene premise"
framed as the paper's own interpretation; D8's superlative ("The first move") is narrowed to "An
early move"; D9's NECO claim is reframed as a structural exclusion; D1 is citation-anchored to Chen
et al. 2022; D2-D4 are hedged; D5-D6 are disclosed as structural arguments. All three engines agree
on D1..D9, and I independently confirmed each fix is present and grounded.

All 51 ledger-tied claims (c1..c51) carry a CONFIRMED status with a recorded quote/number, the
ledger integrity check passes (`OK: all claims confirmed`), and the recently changed numbers (c11
0/219, c22 2-to-11%, c31 fires-at-0.550-no-same-alpha-clause, c32 638/319 stored 609/290 valid)
re-verify against the artifacts. The four fabrication patterns are otherwise absent: no number
without an artifact, no invented baseline (Mahalanobis/RMD/KNN-50/PCA-Maha all run and recorded;
MSP/Energy/ViM excluded with stated structural reasons), no over-broad novelty (EigenTrack
pre-dating and the KNN-50 tie are conceded; the only "first" is the contract-allowed narrow LOCO
claim), and no contribution-boundary violation (N=1, collapse-specific, N=2 two-fold FPR,
downstream-of-encoder, segment-dependent cliff all held).

The gate is BLOCKED on exactly ONE new, narrow overclaim that the prior run did not catch:

- c50: the Data paragraph asserts "camera geometry, calibration, and model-input preprocessing are
  held identical to the parity-verified real path." The recorded artifact supports matched
  intrinsics and the warp-collapse, but NOT "calibration held identical": real uses liveCalibration,
  sim uses zero calibration. Raised by codex, independently concurred by Claude. Fix is a one-clause
  narrowing, no new artifact. Owner: paper-results-stats-verifier / drafter.

No D-row flag survives, and no flag touches a numeric result, a baseline, the novelty boundary, or
the load-bearing negative finding. The gate clears to PASS the moment the c50 calibration clause is
narrowed to what src/sim_preprocessor.py supports.

---

## Honesty summary

- What I verified: the ledger is 51/51 CONFIRMED (`ledger.py check paper_state/claim_ledger.md` =
  `OK: all claims confirmed`). I confirmed all nine prior-run fixes are present in
  drafts/rewritten_draft.md and grounded: D7/c51 against the fetched DeepTest/DeepRoad/MarMot quotes
  in source_verification.md [deeproad-line / c51]; D8 "An early move" (L154); D9 NECO structural
  exclusion (L171-173); D1 citation-anchored (L57); D2-D4 hedged (L46-48); D5-D6 disclosed as
  structural arguments (L236-238). Both cross-examiners ran LIVE (codex_STATUS=ok, gemini_STATUS=ok;
  raw outputs saved); cross-examination is complete, not owed. I independently read
  src/sim_preprocessor.py and confirmed codex's c50 flag is correct on the artifact: real path uses
  liveCalibration-derived euler, sim uses `ZERO_CALIB = np.zeros(3)`, so "calibration held identical"
  is not artifact-supported.
- What I did NOT verify: I did not re-fetch the external citation sources myself this run (I relied
  on the source_verification.md recorded quotes, which is this gate's recorded-evidence input). I
  did not re-derive every one of c1..c49/c51 from raw npz this run; I re-verified the changed
  numbers (c11, c22, c31, c32) and the c50 artifact, and trusted the ledger's CONFIRMED+quote rows
  for the rest. I did not read the full e2/e4/e7 npz arrays this run.
- Riskiest unverified assumption: that gemini's c50 PASS reflects a genuine inspection that read the
  word "calibration" in the conjunction and judged it acceptable, rather than pattern-matching the
  "matched-intrinsics / Attack-1 control" setup and missing the liveCalibration-vs-zero-calibration
  distinction. I did not rely on gemini to confirm anything; the c50 block stands on codex's flag
  plus my own read of src/sim_preprocessor.py. The single concrete open issue is the c50 calibration
  clause, which is a one-clause narrowing away from PASS.
