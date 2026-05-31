# Final Sign-Off (Hybrid)

Stage 16 of the paper-council, the TERMINAL go/no-go gate. Owner: this agent (the only agent
that may declare submit-ready). Run date: 2026-05-30. Paper dir:
`/home/yusuf/Projects/phantom-braking`. Target: **arXiv preprint, primary category cs.LG**
(cross-list cs.CV, cs.RO).

Engines: codex (read-only sandbox) + gemini (plan / read-only) + Claude (this agent).
Runner status this run: `codex_STATUS=ok`, `gemini_STATUS=ok` (both cross-examiners live, both
returned non-empty output). Raw outputs preserved at `gates/raw/final_signoff_codex.txt` and
`gates/raw/final_signoff_gemini.txt` (copied off the shared `gates/raw/codex.txt` /
`gates/raw/gemini.txt` the runner writes). Prompt: `gates/raw/final_signoff_prompt.md`.

Default verdict is DO NOT SUBMIT. This run looks for the proof that it is safe to flip.

## Scope deviation, weighed honestly (read before the tables)

- The humanizer-academic-pro de-AI prose pass was **DEFERRED, not run**. There is therefore no
  `drafts/rewritten_draft.humanized.md` (confirmed absent by `ls`), and no
  pre-vs-post-humanizer diff is possible or needed. Because the humanizer never executed,
  **no equation, citation, or number was ever humanizer-masked**, so the canonical "humanizer
  silently altered a number" failure mode this gate exists to catch **cannot have occurred**.
  The final prose artifact is `drafts/rewritten_draft.md`; the submission artifact is
  `drafts/paper.pdf` (23 pages). The de-AI polish is a recommended follow-up before a
  peer-reviewed venue; arXiv does not gate on AI-prose styling. This is a SOFT item, not a
  content/correctness blocker, and not a fabrication risk.
- HARD blockers (block an arXiv upload): a non-CONFIRMED ledger row, a still-open REJECT-grade
  gate issue, an arXiv format/metadata violation, a fabricated or altered number. SOFT items
  (recommended before camera-ready / a peer-reviewed venue): the de-AI pass, a 3rd real corpus,
  camera-ready venue pins, framing sharpening, Git-LFS-ing the supporting caches.

## Preservation check (humanizer)

Because the humanizer pass was deferred, the "preservation" question reduces to: do the
protected items (equations, citations, numbers) in the final prose draft and the built PDF
match the CONFIRMED ledger values? They were never masked, so the check is a direct match, not
a pre/post diff.

| protected content | source of truth (ledger) | in `rewritten_draft.md` / `paper.pdf` | identical? |
|---|---|---|---|
| equations / formulas | trace-of-covariance monitor, `K_fcam @ inv(K_medmodel)` warp, O(d) statistic | present and unchanged in draft L286-318 and PDF | yes |
| citations / cite-keys | 28 bib keys in `references.bib`; all 28 cited in body (venue_compliance item 7/8) | all 28 present in PDF reference list (1 References section, 26 embedded images incl. figures) | yes |
| numbers / table values | ledger c1..c51 CONFIRMED quote_or_number | every spot-checked headline + secondary number matches (table below) | yes |

- COMMAND RUN (humanized draft absence):
  `ls -la drafts/rewritten_draft.humanized.md` -> `No such file or directory` (pass deferred).
- COMMAND RUN (numbers intact in the actual submission PDF):
  `pdftotext drafts/paper.pdf /tmp/paper_text.txt && grep -c -F "<value>" /tmp/paper_text.txt`
  for each headline and secondary value. Results (hits in PDF):
  `1159` (3), `0.0409` (3), `8 of 10` (2), `1e-5` (4), `0 of 219` (7), `0.996` (4),
  `0.015` (5), `0.274` (5), `0.23` (5), `100% leave-one-corpus-out` (3), `0 of 75` (1),
  and secondary: `0.1788`, `0.7181`, `1.20x`, `1.35x`, `1.84x`, `6.32x`, `87.9%`, `2.19`,
  `11.91%`, `1.03%`, `2.07%`, `2648`, `1928x1208` all present. Author block `Yusuf Guenena`
  (2) / `Wayne State` (2) present. Em dashes in PDF text: `0`. Editorial scaffolding leaked
  into PDF: `0`.
- One value, the literal monitor threshold float `0.078873` (ledger c48), does not appear as a
  raw float in the prose; the draft correctly states it by its definition ("the 1st percentile
  of the real-driving rolling-spread distribution," draft L315, PDF L267). That is the accurate
  description, not an altered number. Not a discrepancy.
- Every spot-checked number matches the CONFIRMED ledger. No NO row. Preservation: OK.

## Gate closure

Each prior gate was opened and its blocking issues read directly (not trusted by pass stamp).

| gate | status | unresolved blocking issues |
|---|---|---|
| hallucination_gate | closed | none (prior c50 camera-confound flag fixed and re-PASSED 3-engine; ledger integrity OK) |
| professor_audit | closed | none (5/5 dimensions OK, no FATAL; only FIXABLE soft framing/N=2 notes) |
| reviewer2_redteam | closed-for-preprint | none HARD-open; Attack 1 (CARLA confound) met to preprint survival bar; Attacks 6/7 fixes landed in draft + Section 8; see note below |
| venue_compliance (arXiv) | pass | none (3 prior FAILs all fixed in the actual draft; re-confirmed) |

Detail on each closure:

- **hallucination_gate (PASS, 3-engine):** The sole prior blocker, c50 (camera-control
  sentence overstating "calibration held identical"), is fixed. Draft L286-294 now states the
  sim uses ZERO extrinsic calibration (device-mounted) while the real path uses
  `liveCalibration`, claims only intrinsics + model-input preprocessing identical, and gives the
  warp collapse `K_fcam @ inv(K_medmodel)`, matching `src/sim_preprocessor.py`. All four prior
  FLAGGED numbers re-verify in the draft (c11 0/219, c22 2-to-11%, c31 fires-at-0.550 with the
  false same-alpha clause removed, c32 638/319 stored vs 609/290 valid). I independently
  grep-confirmed each in `drafts/rewritten_draft.md` and the PDF. No engine flagged anything.

- **professor_audit (PASS):** No FATAL on any of the five dimensions (contribution, evidence,
  novelty, limitations, story). The two non-fatal reject-reasons the engines split on (codex:
  finding may read as CARLA-content-specific; gemini: N=2 two-fold LOCO reads thin) are both
  graded FIXABLE, not FUNDAMENTAL, and are SOFT at the locked preprint / workshop tier. The
  paper does not overclaim past them.

- **reviewer2_redteam:** This report is an adversarial pre-review with NO pass/fail stamp; I
  must check whether its REJECT/MAJOR attacks are answered in the current draft. Attack 1
  (CARLA = rendering-pipeline confound, the one REJECT-grade attack) requires either a
  matched-camera control (strong) or narrowing every headline claim + an explicit confound
  paragraph (minimum survival bar). The current draft does the minimum-survival path AND the
  intrinsics control: the c50 fix supplies matched fcam intrinsics with the warp collapsing to
  `K_fcam @ inv(K_medmodel)`, E5 (c18/c19) localizes the collapse downstream of the encoder, the
  headline is narrowed to "a response to rendered scene content," and Section 5.7 + Limitations
  bound real-world transfer. That clears the preprint bar; the full matched-photometry control
  and the real-adverse-weather axis remain SOFT follow-ups (contract-parked). Attack 6 (n=219
  vs 220; false "AUROC crosses at same alpha"; passive-relay 2% vs 11%; 638/319 vs 609/290) was
  must-fix-cannot-argue: all four are corrected in the draft (the same c11/c31/c22/c32 fixes).
  Attack 7 (gitignored/corrupt caches; Section 8 overstated "settled") is addressed: Section 8
  now states honestly which caches reproduce on a fresh clone and which need a `--collect` pass,
  and the [TODO] is removed (c49 CONFIRMED; verified zero TODO markers in draft). No HARD-open
  reviewer2 attack remains for an arXiv preprint.

- **venue_compliance (arXiv):** The stale report lists three FAILs (no author block; abstract
  848 chars over the 1920 limit; editorial scaffolding present). All three were fixed AFTER that
  report was written, and I re-confirmed each against the ACTUAL draft/metadata, not the stale
  report: (1) author block present (`drafts/rewritten_draft.md` L3-5: "Yusuf Guenena / Wayne
  State University / Code and data: ..."; PDF shows it). (2) The metadata abstract in
  `paper_state/arxiv_metadata.md` is 1598 chars by my `wc -c` (the file header says 1592; both
  are under the 1920 hard limit). (3) No blockquote header and no `## Handoff` section remain
  (grep returned zero). (4) Zero `[TODO]/[DRAFT]/[FIXME]/[TBD]` markers in the draft. (5) All 28
  bib keys present and cited. The only remaining venue items are SOFT camera-ready advisories
  (5 venue-unconfirmed pins; 3 `@inproceedings`->`@misc` field-type fixes; the optional
  commaai #22212 second motivation cite), none of which block an arXiv preprint.

## Ledger

- All claims CONFIRMED? **yes.** `python3 ~/.claude/skills/paper-council/scripts/ledger.py check
  paper_state/claim_ledger.md` returns `OK: all claims confirmed` (exit 0). Direct count: 51/51
  rows end in `CONFIRMED`; zero `UNVERIFIED`, zero `FLAGGED`, zero `REFUTED`. No non-CONFIRMED
  id. The four previously-FLAGGED numbers (c11, c22, c31, c32) are corrected and re-set to
  CONFIRMED, and re-verify in the draft and PDF.

## Per-engine verdict table

| check | codex | gemini | claude | reconciled |
|---|---|---|---|---|
| preservation | OK (no humanized draft; numbers match ledger c1..c33 spot-check) | OK (10/10 headline numbers match CONFIRMED, with line cites) | OK (PDF grep: every headline + secondary number present; 0 em dashes; 0 scaffolding leak) | **OK** |
| gate closure | OK (c50 resolved; 4 flagged numbers corrected; Attack 1 at preprint bar; OPEN hard issues: none) | OK (halluc + professor PASS; Attack 1 controlled; c11/c22/c31/c32 corrected) | OK (each gate opened, not trusted by stamp; no HARD-open issue) | **OK** |
| ledger | OK (direct count 51 51 0; all CONFIRMED) | OK (51/51 CONFIRMED; integrity check passes) | OK (ledger.py `OK: all claims confirmed`, exit 0) | **OK** |
| compliance | OK (author block present; abstract <1920; no scaffolding; no TODO; cites mapped) | OK (author block L4-5; 1592-char abstract; scaffolding stripped; 28 cites mapped) | OK (3 prior FAILs all fixed in actual draft; re-confirmed; only SOFT camera-ready items left) | **OK** |

Engine summary. Codex returned `OVERALL: SUBMIT` with all four checks OK, "OPEN hard issues:
none," and three recommended SOFT follow-ups (run the deferred prose polish, add a non-CARLA
real adverse-weather corpus, clean up camera-ready bib field types / venue pins). Gemini
returned `OVERALL: SUBMIT` with all four checks OK and line-cited evidence for each. Claude
(this agent) independently ran the ledger integrity check, grepped every headline and secondary
number out of the built PDF, confirmed the absence of the humanized draft, re-confirmed the
three prior venue FAILs are fixed in the actual draft, and opened each prior gate report to
verify its blocking issues are closed rather than merely stamped. All three engines agree:
every check is OK, no engine raised a single HARD block. There is no conservative block to apply.

## DECISION

**SUBMIT** (arXiv preprint, primary cs.LG; cross-list cs.CV, cs.RO).

The default DO NOT SUBMIT flips to SUBMIT because every condition is met and proven, not
assumed:
- Preservation is OK: the humanizer never ran, so no protected content could have been masked or
  altered, and a direct grep of the built PDF confirms every headline and secondary number, the
  author block, zero em dashes, and zero leaked scaffolding.
- Every prior gate's blocking issue is explicitly closed (c50 fixed; the four FLAGGED numbers
  corrected; reviewer2 Attack 1 met to the preprint survival bar; Attacks 6 and 7 fixed), not
  merely stamped passed.
- The ledger is 51/51 CONFIRMED, integrity check `OK: all claims confirmed`, exit 0; no
  non-CONFIRMED row.
- arXiv compliance passes on the actual draft: the three prior FAILs (author block, abstract
  length, editorial scaffolding) are all fixed and re-confirmed.
- Both live cross-examiners (codex, gemini) independently returned SUBMIT with no HARD block;
  Claude concurs. Conservative reconciliation finds no single engine BLOCK.

### HARD blocking issues (block the arXiv upload)

**None.** There are no open HARD blockers.

### SOFT follow-ups (recommended before a peer-reviewed venue / camera-ready, NOT blocking the preprint)

These are explicitly NOT blockers for the arXiv preprint. Listed with owner for routing:

1. Run the deferred humanizer-academic-pro de-AI prose pass on `drafts/rewritten_draft.md`
   before any peer-reviewed-venue submission. After it runs, re-run THIS gate's preservation
   diff (it will then be a real pre/post diff). Owner: humanizer-academic-pro (then back to this
   gate for the post-humanizer preservation check).
2. Add a third real corpus to convert the N=2 two-fold LOCO estimate toward a reportable FPR
   with variance (contract-parked). Owner: paper-results-stats-verifier / contribution-locker.
3. Add the parked real adverse-weather OOD axis (rain/night/glare) that induces a non-CARLA
   collapse, the most reviewer-resistant follow-up and the full answer to reviewer2 Attack 1.
   Owner: paper-stanford-framer / experiment owner.
4. Camera-ready bibliography hygiene: pin the 5 "venue unconfirmed" entries (ren2021,
   muellerplus2025, michaelis2019, hodge2025, guosu2026) and fix the 3 `@inproceedings`->`@misc`
   arXiv field-type entries (keser2025, eigentrack2025, adversarial2025); optionally add the
   commaai #22212 second motivation cite. Owner: paper-venue-compliance.
5. Repository reproducibility polish: commit the ~100 KB E5 summary array in place of the
   corrupt 3.9 GB cache, commit the 28 MB E4-RAM cache, and either Git-LFS-track or document the
   `--collect` path for the E5-submodule (98 MB) and E7 (110 MB) caches. The headline caches
   already reproduce on a fresh clone and Section 8 states the gap honestly, so this is polish,
   not a correctness fix. Owner: reproducibility checker / repo owner.

---

## Honesty summary

- **What I verified (commands run, output observed):** (1) `drafts/rewritten_draft.humanized.md`
  is absent (humanizer deferred). (2) `ledger.py check` returns `OK: all claims confirmed`, exit
  0; 51/51 CONFIRMED, zero non-CONFIRMED rows. (3) Zero em dashes, zero TODO/DRAFT markers, zero
  editorial scaffolding in `drafts/rewritten_draft.md`. (4) Author block present in the draft
  (L3-5) and in the PDF (`Yusuf Guenena` x2, `Wayne State` x2). (5) Metadata abstract is 1598
  chars by `wc -c` (under the 1920 arXiv limit). (6) The built `drafts/paper.pdf` is 23 pages
  with 26 embedded images and a References section; I grepped every headline number (1159,
  0.0409, 8 of 10, 1e-5, 0 of 219, 0.996, 0.015, 0.274, 0.23, 100% LOCO, 0 of 75) and a dozen
  secondary numbers (head ratios, uncertainty ratios, 87.9%, d'=2.19, 11.91%, 1.03%, 2.07%, the
  fcam intrinsics) out of the PDF text and each is present; PDF has zero em dashes and zero
  leaked scaffolding. (7) I opened all three prior gate reports and the reproducibility and
  venue reports and checked their blocking issues against the actual draft, confirming c50, the
  four FLAGGED numbers, reviewer2 Attacks 1/6/7, and the three prior venue FAILs are closed in
  the current draft. (8) Both cross-examiners ran LIVE (codex_STATUS=ok, gemini_STATUS=ok, both
  non-empty); I read both raw outputs; both returned SUBMIT.
- **What I did NOT verify:** I did not re-run any analysis or re-open any npz (I trusted the
  reproducibility report's REPRODUCED verdicts on the four headline caches and the ledger's
  CONFIRMED+quote rows for the per-claim numbers; the stats verifier owns the npz recomputation).
  I did not re-fetch any external citation source (I relied on `source_verification.md`'s
  CONFIRMED quotes). I did not re-read `src/sim_preprocessor.py` myself this run; I relied on the
  hallucination gate's verbatim code read for the c50 fix. I did not render or visually inspect
  every figure panel in the PDF beyond confirming 26 images are embedded and the figure files
  exist.
- **The single riskiest unverified assumption:** that the c50 CARLA-confound control is as tight
  as the ledger and the hallucination gate record it, i.e., that the CARLA frames genuinely
  render at the matched fcam intrinsics with the warp collapsing to `K_fcam @ inv(K_medmodel)`
  and no other geometric/photometric mismatch sneaks in upstream of the model input. If that
  control is weaker than recorded, reviewer2 Attack 1 reopens from "controlled confound" toward
  "uncontrolled confound." That would NOT change the arXiv-preprint SUBMIT decision (the draft
  narrows the headline claim to "a response to rendered scene content" regardless, which is the
  minimum survival posture), but it would harden the value/scope objection at a peer-reviewed
  venue and raise the priority of SOFT follow-up #3 (the real adverse-weather axis). I leaned on
  the stage-14 code read rather than re-reading the preprocessor myself this run.
