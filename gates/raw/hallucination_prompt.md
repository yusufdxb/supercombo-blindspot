# Hybrid Hallucination Gate -- RE-RUN (Stage 14) -- Phantom-Braking paper

You are an adversarial hallucination auditor cross-examining a research paper draft. Your single
job: find FABRICATION. Do NOT assess writing quality, story, or contribution strength. Find claims
that are not grounded in a recorded artifact or source.

## The four fabrication patterns you hunt (and ONLY these)

1. A NUMBER or FACT with no artifact or source behind it (invented statistic).
2. A CITATION to a paper that does not exist or does not say what the prose attaches to it
   (fabricated / misattributed citation).
3. An INVENTED BASELINE, dataset, or method the work claims to compare against but never actually
   ran, or that does not exist.
4. A NOVELTY claim ("first", "novel", "unlike prior work") broader than the evidence and the locked
   contribution boundary support (unjustified novelty).

A claim is SUPPORTED only if it is backed by a recorded quote/number in the ledger or
source_verification, OR is uncontroversial domain context / an explicitly-disclosed structural
argument (the draft labels it "structural argument, not a new experiment"). Plausibility is NOT
support. But conservatism cuts both ways: do NOT invent a flag for defensible domain context or a
disclosed structural argument. Flag genuine ungrounded claims only.

## This is a RE-RUN. A prior run BLOCKED on nine prose flags (D1-D9). Fixes applied:

- D7 (DeepRoad-line citation -- the one HARD fabrication block last run): the three sources
  (DeepTest arXiv:1708.08559, DeepRoad arXiv:1802.02295, MarMot arXiv:2310.07414) now carry FETCHED
  quotes in source_verification.md section "[deeproad-line / c51]", a ledger row c51 set CONFIRMED,
  and the draft prose was NARROWED so the "implicitly treating the generated input as a valid scene"
  framing is stated as THIS PAPER'S INTERPRETATION, not as a quoted property of the cited works.
- D8: "The first move" -> "An early move" (Sastry and Oore 2020 lineage).
- D9: NECO reframed as "builds on a neural-collapse property of classification heads, which
  supercombo's regression heads lack" (a structural exclusion reason, not a quoted property).
- D1: supercombo described as "the end-to-end network in comma's shipped openpilot driver-assistance
  system (Chen et al. 2022)".
- D2-D4: universals hedged ("Most Level-2 ... programs", "a primary setting", "rarely tested directly
  in the literature we surveyed").
- D5-D6: hedged to "would collapse together rather than disagree", "typically sharper ... would rate
  ... as good", still disclosed as "(Structural arguments, not new experiments.)".

## Your task

Read these files (all under the paper dir you are rooted at):
- drafts/rewritten_draft.md (the prose to audit)
- paper_state/claim_ledger.md (51 rows; every claim + recorded quote_or_number + status; all CONFIRMED)
- paper_state/source_verification.md (per-citation FETCHED quotes and verdicts; includes the
  "[deeproad-line / c51]" section)
- paper_state/contribution_contract.md (the LOCKED claim boundary; nothing may exceed it)

Then answer:

1. Does EVERY prior flag D1-D9 now clear? For each, say PASS or still-FLAG and why. Ground citation
   judgments in the recorded quote; ground structural-argument judgments in the disclosure. Pay
   special attention to D7: do the three DeepRoad-line sources carry recorded supporting quotes, and
   does the draft prose stay within what those quotes support (metamorphic / GAN test synthesis +
   consistency testing), with the "in-distribution / valid-scene premise" framing presented as the
   paper's own interpretation rather than attributed to the cited papers?

2. Is there any NEW fabrication anywhere in the draft that the prior run missed? Check specifically:
   any number not traceable to a ledger row; any citation whose prose claim exceeds its recorded
   quote; any baseline named but not run; any novelty/scope claim exceeding the contract boundary.
   The contract FORBIDS: generalization beyond supercombo v0.9.7; a production-grade FPR; "universal
   / general OOD detector"; "E6 beats/outperforms baselines" (it TIES KNN-50); on-road deployment;
   a causal link to field incidents; complete localization; "universal cliff"; "the vision encoder
   fails on sim". Confirm the draft's novelty claims stay inside the boundary (EigenTrack pre-dating
   conceded; KNN-50 tie conceded; "first second-order recurrent-state monitor on a shipped driver
   under cross-corpus LOCO" is the only novelty asserted).

3. Spot-check these recently changed numbers are consistent with the ledger and not overclaimed:
   c11 (0 of 219 above p95, was 220); c22 (passive relays "track the summarizer to within 2 to 11%");
   c31 (monitor fires at alpha=0.550, gap ~0.23, NO surviving same-alpha-AUROC clause); c32 (ID n=638
   stored / 609 valid, OOD n=319 stored / 290 valid); c50 (camera-intrinsics Attack-1 control: is
   "camera geometry held identical" overclaimed vs the src/sim_preprocessor.py docstring?).

4. Final verdict: PASS (no fabrication remains) or BLOCKED (list every remaining flag with the exact
   verbatim draft sentence, why it is fabrication, the recorded quote it contradicts or lacks, and
   the required fix).

Output a terse per-claim line `CLAIM <id/quote> : PASS` or `CLAIM <id/quote> : FLAG -- <reason +
which of the 4 patterns>`, then `OVERALL: PASS` or `OVERALL: FLAG` with the flagged list. Do not
edit any files.
