# Hallucination Gate (Hybrid)

Stage 14 of the paper-council, FINAL RE-RUN after the orchestrator applied the prior-run c50 fix.
Owner: paper-hallucination-flagger. Run date: 2026-05-30.
Paper dir: /home/yusuf/Projects/phantom-braking.

Engines: codex (read-only sandbox) + gemini (plan / read-only) + Claude (this agent).
Runner status this run: `codex_STATUS=ok`, `gemini_STATUS=ok` (both cross-examiners live, both
returned non-empty output; no engine outage, no re-run needed). Raw outputs:
gates/raw/codex.txt, gates/raw/gemini.txt. Prompt: gates/raw/hallucination_gate_prompt.md.
Ledger integrity: `python3 ~/.claude/skills/paper-council/scripts/ledger.py check
paper_state/claim_ledger.md` returns `OK: all claims confirmed` (exit 0; 51/51 CONFIRMED, each
with a recorded quote_or_number; no UNVERIFIED / FLAGGED / REFUTED rows).

Reconciliation rule (conservative): a claim PASSES only if (a) none of codex, gemini, or Claude
flags it AND (b) it carries a CONFIRMED ledger row with a recorded supporting quote/number. Any
single engine flag, or any missing quote, BLOCKS that claim. Disagreement resolves toward BLOCKED.

---

## Outcome at a glance

- The one blocking flag from the prior re-run (c50, the CARLA camera-control sentence overstating
  "calibration ... held identical") is now FIXED and CLEARS. The rewritten Section 4 Data sentence
  no longer claims calibration is held identical; it correctly states the sim camera uses ZERO
  extrinsic calibration (device-mounted) while the real path uses `liveCalibration`, and that only
  the intrinsics and model-input preprocessing are identical, with the warp collapsing to
  `K_fcam @ inv(K_medmodel)`. I verified this verbatim against src/sim_preprocessor.py. Both codex
  and gemini independently confirmed c50 PASS against the same artifact.
- All nine prior prose flags (D1..D9) remain CLEAR. All three engines agree D1..D9 PASS; I
  independently confirmed each fix is still present in the draft and grounded.
- All 51 ledger-tied claims (c1..c51) carry CONFIRMED status with a recorded quote/number, the
  ledger integrity check passes, and the changed numbers (c11, c22, c31, c32) re-verify in the draft.
- No new substantive flag appeared. The four fabrication patterns are absent: no number without an
  artifact, no invented baseline, no over-broad novelty, no contribution-boundary violation.

GATE DECISION: PASS. Every inventoried claim is PASS; no engine flagged anything; the ledger is
51/51 CONFIRMED with recorded quotes.

---

## Claim inventory

Claims c1..c51 are the ledger-tied factual/quantitative/citation/novelty claims that appear in the
draft, all CONFIRMED with a recorded quote_or_number. The D1..D9 rows are the nine NON-ledger prose
sentences flagged two runs ago; each is re-inventoried with its current (post-fix) draft text and
location. The c50 row is spelled out in full because it is the focus of this final re-run; the
other changed rows (c11, c22, c31, c32, c51) and the D1..D9 rows are summarized with their current
draft text; the unchanged c1..c49 rows are summarized.

| id | claim (verbatim, trimmed) | type | ledger status | recorded quote present? |
|---|---|---|---|---|
| c1..c49 | parity, collapse, freeze, uncertainty-silence, cliff, RAM-gradient, localization, monitor, baselines, corruption-sweep, citation rows (all unchanged this run) | fact/number/citation/novelty | CONFIRMED (each) | yes (each) |
| **c11** | "the '0%' of Section 5.3 is 0 of 219 CARLA frames" (220 -> 219) | number | CONFIRMED | yes (0/219) |
| **c22** | "track the summarizer to within 2 to 11% ... they are passive [relays]" (Section 5.5, L421) | number | CONFIRMED | yes (att 3.2%, trans 2.7%, reduce_sum 11.4%) |
| **c31** | "fired fraction crosses 50% at alpha=0.550, where the E4 Subaru output cliff is at about alpha=0.784, an early-warning gap of about 0.23 blend-units" (L463-464; false same-alpha-AUROC clause removed; Subaru-specific caveat present) | number | CONFIRMED | yes (fires 0.550; gap 0.234; no same-alpha clause) |
| **c32** | "in-distribution set is subaru and ram concatenated (n=638 stored frames) and the out-of-distribution set is the alpha=1.0 CARLA frames (n=319 stored frames); ... 609 and 290 valid (non-NaN) scores ... (n=1000 iterations, seed 42)" (L299-302, L307) | fact | CONFIRMED | yes (638/319 stored; 609/290 valid; n=1000 seed=42) |
| **c50** | "because the sim camera is mounted on the device axes (zero extrinsic calibration), its warp to the model frame reduces to the same intrinsic remap (`K_fcam @ inv(K_medmodel)`) that the real path applies after its `liveCalibration` extrinsics (`src/sim_preprocessor.py`). The intrinsics and the model-input preprocessing are therefore identical to the parity-verified real path ... shift is confined to rendered image content" (Section 4 Data, L286-294) | fact | CONFIRMED (matched fcam intrinsics + warp collapse + correct sim-zero-vs-real-liveCalibration asymmetry) | yes (the quote backs matched fcam intrinsics, the `K_fcam @ inv(K_medmodel)` warp collapse, and the ZERO_CALIB-vs-liveCalibration distinction; src/sim_preprocessor.py L10-14, L38-39, L69-71) |
| **c51** | "The DeepRoad line (DeepTest, DeepRoad, MarMot) uses metamorphic/generative test synthesis ... implicitly treating the generated scene as a valid input" (Related Work, "Simulation testing", L184-188) | citation | CONFIRMED | yes (DeepTest/DeepRoad/MarMot fetched quotes in source_verification.md [deeproad-line]) |
| D1 | comma's shipped openpilot driver-assistance system, citation-anchored to Chen et al. 2022 (intro, ~L82) | fact (motivation) | n/a (anchored to c47) | n/a (citation-anchored) |
| D2 | "Most Level-2 and autonomous-driving programs validate their driving policy largely in simulation" (intro) | fact (motivation) | n/a (hedged) | n/a |
| D3 | "simulation is a primary setting in which rare and dangerous scenarios can be exercised at scale and at low cost" (intro) | fact (motivation) | n/a (hedged) | n/a |
| D4 | "an assumption that is rarely tested directly in the literature we surveyed" (intro) | fact (motivation) | n/a (scoped to surveyed lit) | n/a |
| D5 | "an ensemble of the same model would collapse together rather than disagree" (threat model, ~L236) | fact (structural argument) | n/a (disclosed) | derived from c18/c19 |
| D6 | image-quality screen "would rate the simulated input as good" (threat model, ~L240) | fact (structural argument) | n/a (disclosed) | n/a |
| D7 | DeepRoad-line "implicitly treating the generated input as a valid scene" (related work, L184-188) | citation | CONFIRMED via c51 | yes (now quote-backed) |
| D8 | "An early move to a higher-order feature statistic for OOD is the Gram-matrix method (Sastry and Oore 2020)" (related work, L154) | novelty/lineage | source CONFIRMED; superlative removed | yes (no "first") |
| D9 | "NECO (Ben Ammar et al. 2024) builds on a neural-collapse property of classification heads, which supercombo's regression heads lack" (related work, L171-173) | citation | source CONFIRMED; reframed as structural exclusion | yes (structural exclusion, not quoted property) |

---

## Per-engine verdict table

| id | codex | gemini | claude | reconciled |
|---|---|---|---|---|
| c1..c49 | pass | pass | pass | PASS |
| c11 (0/219) | pass | pass | pass | PASS |
| c22 (2 to 11%) | pass | pass | pass | PASS |
| c31 (fires 0.550; gap 0.23; no same-alpha clause) | pass | pass | pass | PASS |
| c32 (638/319 stored; 609/290 valid; n=1000 seed=42) | pass | pass | pass | PASS |
| **c50 (camera control: sim ZERO_CALIB vs real liveCalibration; intrinsics+preproc identical; warp = K_fcam @ inv(K_medmodel))** | **pass** | **pass** | **pass** | **PASS** |
| c51 (DeepRoad-line, was D7) | pass | pass | pass | PASS |
| D1 (comma shipped-system, anchored to Chen 2022) | pass | pass | pass | PASS |
| D2 (Most Level-2 hedge) | pass | pass | pass | PASS |
| D3 (a primary setting hedge) | pass | pass | pass | PASS |
| D4 (rarely-tested-in-surveyed-lit hedge) | pass | pass | pass | PASS |
| D5 (ensemble "would collapse together") | pass | pass | pass | PASS |
| D6 (image-quality "would rate ... good") | pass | pass | pass | PASS |
| D7 (DeepRoad-line citation = c51) | pass | pass | pass | PASS |
| D8 (Sastry "An early move") | pass | pass | pass | PASS |
| D9 (NECO structural exclusion) | pass | pass | pass | PASS |
| novelty boundary (EigenTrack pre-date, KNN-50 tie, single narrow "first") | pass | pass | pass | PASS |
| baselines (Maha/RMD/KNN-50/PCA-Maha run; MSP/Energy/ViM structurally excluded) | pass | pass | pass | PASS |

Engine summary. Codex returned OVERALL: PASS, explicitly passing c50 ("real path uses
`liveCalibration`, sim uses `ZERO_CALIB`, and the sim warp collapses to `K_fcam @ inv(K_medmodel)`
... the draft now claims only intrinsics and preprocessing are identical, not calibration"), all of
D1..D9, the spot-checked numbers, the novelty boundary, and the baselines (confirming Maha/RMD/KNN
in src/baselines.py and PCA in src/pca_mahalanobis.py, with MSP/Energy/ViM excluded for stated
structural reasons). Gemini returned OVERALL: PASS, confirming c50 ("correctly limits the 'identical'
claim to intrinsics and model-input preprocessing, distinguishing them from the zero-extrinsic
calibration ... matches the implementation in src/sim_preprocessor.py"), all of D1..D9, the
spot-checked numbers, the novelty boundary, and the baselines. Claude (this agent) independently
read src/sim_preprocessor.py and the relevant draft lines and concurs on every row. All three
engines agree on PASS with no flag; there is no conservative block to apply.

---

## Blocking flags

None. The single prior-run blocking flag (c50) is resolved.

Resolution record for the prior c50 block (calibration-identical overclaim):
- The prior-run draft asserted "camera geometry, calibration, and model-input preprocessing are
  held identical to the parity-verified real path," which the artifact did not support because the
  sim path uses ZERO extrinsic calibration while the real path uses `liveCalibration`.
- The rewritten draft (drafts/rewritten_draft.md, Section 4 Data, L286-294) now reads: the sim
  camera "is mounted on the device axes (zero extrinsic calibration), its warp to the model frame
  reduces to the same intrinsic remap (`K_fcam @ inv(K_medmodel)`) that the real path applies after
  its `liveCalibration` extrinsics," and only "the intrinsics and the model-input preprocessing are
  therefore identical to the parity-verified real path."
- This matches src/sim_preprocessor.py verbatim: docstring L10-14 ("Real openpilot feeds
  `get_warp_matrix(device_from_calib_euler, fcam.intrinsics)` where the euler comes from
  `liveCalibration`. In sim ... the calibration euler is exactly zero and the warp collapses to
  `K_fcam @ inv(K_medmodel)` -- a pure intrinsic remap"); `ZERO_CALIB = np.zeros(3, dtype=np.float64)`
  (L39, comment "the sim camera is mounted exactly on the device axes"); and
  `warp_y = get_warp_matrix(ZERO_CALIB, K, ...)` with `K = _ar_ox_config.fcam.intrinsics` (L69-71).
  The asymmetry (sim zero calibration vs real liveCalibration) is now stated, not papered over, and
  the "identical" conjunction is correctly narrowed to intrinsics + preprocessing. CLEARS to PASS.
  Ledger c50 row already reflects this (evidence note "src/sim_preprocessor.py
  (ZERO_CALIB=np.zeros(3) vs real liveCalibration)").

---

## Gate decision

PASS.

Every inventoried claim is PASS under the conservative reconciliation rule:
- The c50 camera-control sentence, the sole blocker from the prior re-run, is fixed and clears. The
  rewrite correctly states the sim uses ZERO_CALIB (device-mounted) while real uses liveCalibration,
  claims only intrinsics + model-input preprocessing identical, and gives the warp collapse
  `K_fcam @ inv(K_medmodel)`, all matching src/sim_preprocessor.py. Codex, gemini, and Claude all
  independently pass it against that artifact.
- All nine prior prose flags (D1..D9) remain CLEAR: D1 citation-anchored to Chen et al. 2022;
  D2-D4 hedged; D5-D6 disclosed as structural arguments; D7/c51 DeepRoad-line quote-backed with the
  valid-scene premise framed as the paper's interpretation; D8 superlative narrowed to "An early
  move"; D9 NECO reframed as a structural exclusion.
- All 51 ledger-tied claims (c1..c51) carry CONFIRMED with a recorded quote/number; the ledger
  integrity check returns `OK: all claims confirmed`; and the changed numbers (c11 0/219, c22
  2-to-11%, c31 fires-at-0.550 / gap-0.23 / no-same-alpha-clause, c32 638/319 stored, 609/290 valid,
  n=1000 seed=42) re-verify in the draft.
- The four fabrication patterns are absent: no number lacks an artifact; the baselines
  (Mahalanobis, Relative Mahalanobis, KNN-50, PCA-Mahalanobis) are all actually implemented and run
  in src/baselines.py and src/pca_mahalanobis.py, with MSP/Energy/ViM excluded for stated structural
  reasons (no softmax head, no logits, no classifier weight matrix); the only "first" is the
  contract-allowed narrow LOCO claim (EigenTrack pre-dating and the KNN-50 AUROC-1.000 tie are both
  conceded); and no claim exceeds the contribution boundary (N=1 model, collapse-specific monitor,
  N=2 two-fold FPR stated as not a production FPR, downstream-of-encoder partial localization,
  segment-dependent cliff all held).

No engine flagged any claim. No new substantive flag appeared. The gate passes.

---

## Honesty summary

- What I verified: (1) The ledger is 51/51 CONFIRMED, integrity check `OK: all claims confirmed`
  (exit 0), no non-CONFIRMED rows. (2) The c50 rewrite (drafts/rewritten_draft.md L286-294) matches
  src/sim_preprocessor.py verbatim on all three load-bearing points: sim ZERO_CALIB (np.zeros(3),
  device-mounted) vs real liveCalibration extrinsics; the "identical" claim correctly narrowed to
  intrinsics + model-input preprocessing; the warp collapse to `K_fcam @ inv(K_medmodel)`. (3) All
  nine prior fixes (D1..D9) are still present and grounded in the draft (D8 "An early move" L154;
  D9 NECO structural exclusion L171-173; D7/c51 quote-backed; novelty bounded at L169-171 with the
  EigenTrack and KNN-50 concessions at L129/L470). (4) The changed numbers (c11, c22 L421, c31
  L463-464, c32 L299-302/L307) are present in the draft as the ledger records them. (5) The
  baselines are implemented in src/baselines.py (Maha/RMD/KNN on the 512-D recurrent feature;
  MSP/Energy/ViM marked not-applicable with the no-softmax-logits structural reason) and
  src/pca_mahalanobis.py. (6) Both cross-examiners ran LIVE (codex_STATUS=ok, gemini_STATUS=ok, both
  non-empty), both returned OVERALL: PASS with explicit c50 confirmation; cross-examination is
  complete, not owed.
- What I did NOT verify this run: I did not re-fetch the external citation sources myself (I relied
  on the recorded source_verification.md quotes, which is this gate's recorded-evidence input). I did
  not re-derive every one of c1..c49/c51 from the raw npz; I re-verified the changed numbers
  (c11, c22, c31, c32), the c50 artifact, the novelty/baseline boundary, and trusted the ledger's
  CONFIRMED+quote rows for the rest. I did not read the full e2/e4/e7 npz arrays.
- Riskiest unverified assumption: that the recorded quote_or_number values in the ledger for the
  unchanged c1..c49/c51 rows still faithfully match their npz artifacts (the stats verifier owns
  that recomputation, and I trusted the CONFIRMED status + recorded quote rather than recomputing
  each from raw npz this run). For the c50 fix that is the focus of this re-run, there is no residual
  assumption: I read the source code directly and the draft wording matches it line for line.
