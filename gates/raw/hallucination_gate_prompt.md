# Hallucination Gate (Hybrid) -- FINAL re-run, Stage 14

You are an adversarial hallucination auditor for an academic paper. You are READ-ONLY.
Do not edit any file. Read the repo by relative path from the paper root.

Your job: enumerate the factual, quantitative, citation, and novelty claims in the draft
and decide, source-grounded (NOT from memory), whether each is actually supported by the
recorded evidence in the claim ledger, the source-verification file, and the cited code
artifacts in `src/`. Plausibility is NOT support. A claim passes only if a recorded quote
or number backs it AND it stays inside the contribution boundary.

## Context: what this re-run is checking

This is the FINAL re-run. A prior re-run cleared nine prose flags (D1..D9) and BLOCKED on
exactly ONE flag: claim c50, the CARLA camera-control sentence in Section 4 (Data), which
previously overstated that "camera geometry, calibration, and model-input preprocessing are
held identical to the parity-verified real path." That was wrong because the SIM path uses
ZERO extrinsic calibration while the REAL path uses `liveCalibration` extrinsics; calibration
is therefore NOT identical. That sentence has now been REWRITTEN. Your primary task is to
confirm the rewrite is now correct against the source code, and that all D1..D9 fixes still
hold, and to flag any genuinely NEW substantive problem (not a wording nit).

## The rewritten c50 sentence (drafts/rewritten_draft.md, Section 4 Data, lines 286-294)

"The sim camera is configured as a matched-intrinsics pinhole: it renders at the comma 3
fcam native resolution (1928x1208) at the field of view that reproduces fcam's focal length
(2648 px), so the rendered frames carry exactly the production camera (`_ar_ox_config.fcam`)
intrinsics, and because the sim camera is mounted on the device axes (zero extrinsic
calibration), its warp to the model frame reduces to the same intrinsic remap
(`K_fcam @ inv(K_medmodel)`) that the real path applies after its `liveCalibration`
extrinsics (`src/sim_preprocessor.py`). The intrinsics and the model-input preprocessing are
therefore identical to the parity-verified real path, and the distribution shift is confined
to rendered image content (photometry, texture, and the absence of sensor noise); the
collapse is a response to rendered scene content, not an artifact of a mismatched camera
model."

## The grounding artifact (src/sim_preprocessor.py)

Docstring + constants (lines 10-14, 38-39, 69-71):
- "Real openpilot feeds `get_warp_matrix(device_from_calib_euler, fcam.intrinsics)` where the
  euler comes from `liveCalibration`. In sim the camera is mounted perfectly (we control the
  transform), so the calibration euler is exactly zero and the warp collapses to
  `K_fcam @ inv(K_medmodel)` -- a pure intrinsic remap, no rotation. That is why no per-run
  calibration is needed in sim."
- `ZERO_CALIB = np.zeros(3, dtype=np.float64)`  with comment "Zero calibration: the sim camera
  is mounted exactly on the device axes."
- `warp_y = get_warp_matrix(ZERO_CALIB, K, bigmodel_frame=False)` where `K =
  _ar_ox_config.fcam.intrinsics`.

## Specific questions for THIS gate

1. c50: Does the REWRITTEN sentence accurately match src/sim_preprocessor.py? Specifically:
   (a) Does it correctly state the SIM camera uses ZERO extrinsic calibration (device-mounted)
   while the REAL path uses `liveCalibration`? (b) Does it correctly claim only INTRINSICS +
   model-input preprocessing are identical (NOT calibration)? (c) Does it correctly state the
   warp collapses to `K_fcam @ inv(K_medmodel)`? If all three are accurate, c50 PASSES. If the
   sentence still overstates calibration as "held identical," FLAG it.

2. D1..D9: Confirm the nine prior prose fixes still hold (no reintroduced overclaim):
   D1 comma-shipped-system phrasing citation-anchored to Chen et al. 2022; D2/D3/D4 motivation
   hedges intact; D5/D6 structural-argument disclosures intact ("would collapse together",
   image-quality "would rate good"); D7 DeepRoad-line citation quote-backed (c51) with the
   valid-scene premise framed as the paper's interpretation; D8 Sastry superlative removed
   ("An early move", not "The first"); D9 NECO reframed as a structural exclusion, not a quoted
   property.

3. Spot-check the changed numbers: c11 ("0 of 219" exceeds real p95), c22 (transformer/FFN/
   reduce-sum passive relays "to within 2 to 11%"), c31 (monitor fires at alpha=0.550, about
   0.23 before the cliff at ~0.784, with NO false "AUROC crosses at the same alpha" clause),
   c32 (ID n=638 stored / 609 valid; OOD n=319 stored / 290 valid; bootstrap n=1000 seed=42).

4. Novelty boundary: the only "first" claim must be the narrow contract-allowed one (first
   second-order recurrent-state monitor on a SHIPPED end-to-end driving model under cross-corpus
   LOCO). EigenTrack pre-dating the second-order framing must be conceded; the KNN-50 tie at
   AUROC 1.000 must be conceded; no universal-OOD claim; no production-FPR claim (LOCO is N=2).

5. Baselines: Mahalanobis, Relative Mahalanobis, KNN-50, PCA-Mahalanobis must all be actually
   RUN (not invented), and MSP/Energy/ViM excluded with stated STRUCTURAL reasons (no softmax /
   no logits / no classifier weight matrix). Flag any invented baseline.

6. Any other unsupported fact, fabricated citation, invented baseline, or contribution-boundary
   violation anywhere in the draft.

## Output format

Give a per-claim PASS/FLAG verdict for c50, each of D1..D9, the spot-checked numbers, the
novelty boundary, and the baselines. For any FLAG, quote the offending draft sentence and
state exactly why the recorded evidence does not support it. End with one line:
OVERALL: PASS  (if nothing flagged)  or  OVERALL: FLAG  (if anything flagged), listing the
flagged ids. Be conservative: if a claim is not backed by a recorded quote/number/code
artifact, FLAG it. Do not invent issues that are mere wording preferences.

## Files you may read (relative to paper root)

- drafts/rewritten_draft.md  (the prose under audit; Section 4 Data is lines ~283-302)
- paper_state/claim_ledger.md  (51 claims c1..c51, all marked CONFIRMED with a quote/number)
- paper_state/source_verification.md  (per-citation fetched quotes + verdicts)
- paper_state/contribution_contract.md  (the locked claim boundary and exclusion list)
- src/sim_preprocessor.py  (the c50 grounding artifact: ZERO_CALIB, K_fcam @ inv(K_medmodel))
- src/baselines.py, src/pca_mahalanobis.py  (baseline implementations, for question 5)
