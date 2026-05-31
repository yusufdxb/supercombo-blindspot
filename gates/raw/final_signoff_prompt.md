# Final Sign-Off Cross-Examination (Stage 16, terminal go/no-go)

You are an independent cross-examiner for the TERMINAL sign-off gate of an academic paper.
The decision is a single binary: SUBMIT or DO NOT SUBMIT. The target is an **arXiv preprint**
(primary category cs.LG). The default is DO NOT SUBMIT; it flips to SUBMIT only when every
blocking issue from every prior gate is explicitly closed, every ledger claim is CONFIRMED,
venue compliance passes for arXiv, and the prose-level numbers/citations are intact.

IMPORTANT SCOPE NOTE you must respect:
- The humanizer-academic-pro de-AI prose pass was DEFERRED (not run). There is therefore NO
  `drafts/rewritten_draft.humanized.md`. The final prose artifact is `drafts/rewritten_draft.md`,
  and a PDF was built at `drafts/paper.pdf` (23 pages). Because the humanizer never ran, no
  equation, citation, or number was ever humanizer-masked, so there is no "humanizer altered a
  number" risk to diff. Treat the missing humanized draft as a DEFERRED-POLISH item, not a
  fabrication risk. arXiv does not gate on AI-prose styling.
- Distinguish HARD blockers (things that block an arXiv preprint upload: a non-CONFIRMED ledger
  row, a still-open REJECT-grade gate issue, an arXiv format/metadata violation, a fabricated or
  altered number) from SOFT items (recommended before a peer-reviewed venue or camera-ready: a
  de-AI prose pass, adding a 3rd corpus, pinning camera-ready venues, sharpening framing).

## Your task

Read these files in the repo (paper dir is your cwd):
- `paper_state/contribution_contract.md` (the locked claim boundary)
- `paper_state/claim_ledger.md` (51 claims; the maintainers report 51/51 CONFIRMED)
- `drafts/rewritten_draft.md` (the final prose draft)
- `paper_state/arxiv_metadata.md` (author block + 1598-char metadata abstract)
- `paper_state/venue_compliance.md` (arXiv compliance; NOTE: it lists 3 FAILs that were fixed
  AFTER it was written: author block added, abstract shortened to <1920 chars, editorial
  scaffolding stripped. Re-confirm those fixes against the actual draft, not the stale report.)
- `paper_state/reproducibility_report.md` (4 headline caches reproduce on a fresh clone; the
  e5/e7/e4_ram supporting caches are size-excluded and regenerate via --collect; the 3.9GB e5
  cache is corrupt; Section 8 of the draft states this honestly)
- `gates/hallucination_gate.md` (reported PASS, 3-engine)
- `gates/professor_audit.md` (reported PASS, no fatal flaw)
- `gates/reviewer2_redteam.md` (adversarial attacks; check whether any REJECT/MAJOR attack is
  unresolved vs already answered/narrowed in the draft)

## Answer these questions, each with evidence quoted from the files

1. PRESERVATION (numbers/citations intact): Since the humanizer never ran, confirm there is no
   humanizer-masking risk. Spot-check that the headline numbers in `drafts/rewritten_draft.md`
   (parity 1159 frames / median 0.0409; 8 of 10 heads; 1e-5 spread; 0 of 219 uncertainty; AUROC
   0.996; cliff width 0.015; RAM gradient 0.274; 0.23 blend-units; 100% LOCO FPR baselines; 0 of
   75 corruption cells) match the ledger CONFIRMED values. Any mismatch is BLOCK.

2. GATE CLOSURE: For hallucination_gate, professor_audit, and reviewer2_redteam, is every
   BLOCKING / REJECT-grade issue explicitly CLOSED in the current draft (not merely stamped
   "pass")? In particular: is the prior c50 camera-confound flag fixed? Are the 4 prior FLAGGED
   numbers (c11 0/219, c22 2-to-11%, c31 fires-at-0.550 with the false same-alpha clause removed,
   c32 638/319 stored vs 609/290 valid) corrected in the draft? Is reviewer2 Attack 1 (CARLA
   confound) answered to at least the "minimum survival bar" (narrowed claim + intrinsics control)
   for a preprint? List any issue that is still OPEN.

3. LEDGER: Are all 51 claims CONFIRMED with no UNVERIFIED / FLAGGED / REFUTED row? Any non-
   CONFIRMED row is an automatic DO NOT SUBMIT.

4. COMPLIANCE (arXiv preprint, cs.LG): Re-confirm against the ACTUAL draft + metadata file that
   (a) an author block is present, (b) the metadata abstract is <1920 chars, (c) no editorial
   scaffolding (blockquote header / Handoff section) remains, (d) no [TODO]/[DRAFT] markers, (e)
   all in-text citations have bib entries. Flag any HARD arXiv violation. Camera-ready-only items
   (venue pins, bib @inproceedings->@misc fixes) are SOFT, not blocking for a preprint.

## Output format

For each of the four checks output: OK or BLOCK, with the one-line evidence.
Then a final line: `OVERALL: SUBMIT` or `OVERALL: DO NOT SUBMIT`, and if DO NOT SUBMIT, the list
of HARD blocking issues (not soft/recommended items). Be conservative: if a real HARD blocker
exists, say DO NOT SUBMIT. If only soft/recommended items remain for an arXiv preprint, you may
say SUBMIT and list the soft items as recommended follow-ups. Do not use em dashes.
