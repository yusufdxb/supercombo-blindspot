# Venue Compliance: arXiv Preprint (cs.LG primary), 2026

## Guidelines source

- FETCHED FROM: https://info.arxiv.org/help/submit/index.html (submission overview)
- FETCHED FROM: https://info.arxiv.org/help/prep.html (metadata / abstract preparation)
- FETCHED FROM: https://info.arxiv.org/help/policies/format_requirements.html (format requirements)
- FETCHED FROM: https://info.arxiv.org/help/policies/content-types.html (content type policies)
- FETCHED FROM: https://arxiv.org/category_taxonomy (category descriptions)

KEY QUOTES (each with its source):

- Abstract length: "arXiv limits abstracts to 1920 characters and will not accept abstracts longer than this; authors must abridge their abstract if necessary." (info.arxiv.org/help/prep.html)
- No anonymous submissions: "Title and authorship (no anonymous submissions)" listed under Required Elements. (info.arxiv.org/help/policies/format_requirements.html)
- File formats accepted for figures: "PostScript (PS, EPS), JPEG, GIF, PNG, or PDF formats are accepted." (info.arxiv.org/help/submit/index.html)
- Preferred source format: "(La)TeX, AMS(La)TeX, PDFLaTeX" are preferred; PDF is also accepted. (info.arxiv.org/help/submit/index.html)
- Completeness requirement: "Articles should be complete final drafts" and must demonstrate "novel results and be of plausible interest to professional researchers." (info.arxiv.org/help/policies/content-types.html)
- Prohibited content: "should not have: Line numbers, Watermarks that obstruct the text, Advertisements of any kind, Highlighted text, Margin notes, Referee remarks." (info.arxiv.org/help/policies/format_requirements.html)
- License: Authors must "grant arXiv.org an irrevocable license to distribute the work." (info.arxiv.org/help/submit/index.html) Default license is arXiv's non-exclusive distribution license; authors may opt up to CC-BY 4.0 at submission time.
- Endorsement / moderation: "We only accept submissions from registered authors. New users or those submitting to unfamiliar categories may require endorsements." (info.arxiv.org/help/submit/index.html)
- cs.LG: "Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology." (arxiv.org/category_taxonomy)
- cs.CV: "image processing, computer vision, pattern recognition, and scene understanding." (arxiv.org/category_taxonomy)
- cs.RO: "Robotic systems and their applications." (arxiv.org/category_taxonomy, paraphrased from ACM I.2.9 alignment)
- cs.SE: "design tools, software metrics, testing and debugging, programming environments, etc." (arxiv.org/category_taxonomy)

## Workshop / page-limit note

The SafeAI@UAI 2026 deadline has passed. This submission is an arXiv preprint only. arXiv imposes NO page limit and NO anonymization requirement. The 4-page workshop limit and double-blind rules from SafeAI are therefore not applicable and are NOT enforced here.

---

## Checklist

| # | Rule (quoted) | Check | Status | Fix if FAIL |
|---|---|---|---|---|
| 1 | Category selection: primary must be one of the accepted cs subtrees; cross-lists allowed | Draft targets cs.LG primary. Justification: the core contribution is an ML monitoring method (a second-order recurrent-state OOD monitor with AUROC/FPR evaluation) evaluated on a production neural network. cs.LG covers "robustness" explicitly. Cross-list cs.CV (visual distribution shift, CARLA rendering, corruption sweeps) is strongly justified. Cross-list cs.RO (deployed driving model, Level-2 safety) is appropriate. cs.SE is a reasonable fourth cross-list (DeepTest/DeepRoad testing context, software falsification neighbor), though optional. | PASS | N/A |
| 2 | Abstract present and within the 1920-character limit | Abstract is present (lines 12-41 of rewritten_draft.md). Measured character count (non-whitespace-collapsed, as arXiv metadata field): 2768 characters with newlines stripped. That is **848 characters over the 1920-char limit**. | **FAIL** | The abstract metadata field must be shortened to at most 1920 characters before submission. The body abstract can remain long; only the metadata text field is capped. Cut to roughly the first six sentences (through "Nothing the model emits flags the collapse") plus a compressed two-sentence summary of E6 and E7. Target: retain the parity number, the collapse statistic (8/10 heads, 1e-5 spread, 0/219 uncertainty), the monitor result (AUROC 0.996, ~1% FPR, 0.23 blend-unit early warning), and the bounding statement (collapse sim-specific, monitor collapse-specific). Drop the internal detail on the alpha-blend, the cliff shape, the localization specifics, and the ImageNet-C cell counts from the metadata abstract; those details remain in the body. |
| 3 | No undefined macros in the abstract metadata | Draft is Markdown; no LaTeX macros appear in the abstract text itself. The abstract does use "+/-" (ASCII) and numeric notation throughout, which is acceptable in the arXiv metadata field. No TeX commands of the form \cmd appear. | PASS | N/A |
| 4 | No placeholder markers ([DRAFT], [TODO], [AUTHOR TODO], [TBD], [FIXME]) in the body | grep found zero hits for any of these markers in rewritten_draft.md. | PASS | N/A |
| 5 | Editorial meta-content must be stripped before submission: the blockquote header block (lines 3-6) and the "## Handoff" section (lines 617-622) are internal council artifacts not intended for the arXiv submission. | Both are present in the current draft file. The blockquote block ("> Target venue: arXiv preprint...") and the Handoff section ("Rewrite complete. Facts, numbers...NEXT: run humanizer...") are editorial scaffolding. | **FAIL** | Strip lines 3-6 (the blockquote block) and lines 617-622 (the ## Handoff section and its three lines) before converting to LaTeX or PDF for submission. The "## Figure and Table Manifest" section (lines 595-613) is also an editorial inventory; it may be retained as an appendix or stripped depending on whether a formal LaTeX figure list replaces it, but it should not appear as a top-level section in the submitted PDF. |
| 6 | Author name / affiliation block required (arXiv does not accept anonymous submissions) | No author block is present in the draft. The draft header contains only a title and an editorial blockquote. arXiv explicitly requires "Title and authorship (no anonymous submissions)." | **FAIL** | Add an author block below the title before LaTeX conversion. Include: full author name(s), institutional affiliation(s) in the format "(Institution, City, Country)" per arXiv metadata norms. Also populate the arXiv metadata author field at submission time with "Firstname Lastname" format, no honorifics, no "et al." truncation. |
| 7 | Every in-text citation has a corresponding bib entry | The draft uses author-year prose citations throughout (no LaTeX \cite{} commands, as this is a Markdown draft). Mapping verified: Hendrycks and Gimpel 2017 -> hendrycks2017msp (PRESENT); Liu et al. 2020 -> liu2020energy (PRESENT); Lee et al. 2018 -> lee2018 (PRESENT); Ren et al. 2021 -> ren2021 (PRESENT); Sun et al. 2022 -> sun2022 (PRESENT); Wang et al. 2022 / ViM -> vim2022 (PRESENT); Mueller and Hein 2025 -> muellerplus2025 (PRESENT, line 124); Yang et al. 2022 / OpenOOD -> yang2022openood (PRESENT); Keser et al. 2025 -> keser2025 (PRESENT); Guo and Su 2026 -> guosu2026 (PRESENT); Filos et al. 2020 -> filos2020 (PRESENT); Stocco et al. 2020 -> stocco2020 (PRESENT); Grewal, Tonella, and Stocco 2024 -> grewal2024 (PRESENT); Hodge et al. 2025 -> hodge2025 (PRESENT); Cheng et al. 2018 -> cheng2018 (PRESENT); Sastry and Oore 2020 -> sastry2020 (PRESENT); EigenTrack / arXiv:2509.15735 -> eigentrack2025 (PRESENT); Ben Ammar et al. 2024 / NECO -> neco2024 (PRESENT); Hendrycks and Dietterich 2019 -> hendrycks2019imgnetc (PRESENT); Michaelis et al. 2019 -> michaelis2019 (PRESENT); Dosovitskiy et al. 2017 / CARLA -> dosovitskiy2017carla (PRESENT); DeepTest -> deeptest2018 (PRESENT); DeepRoad -> deeproad2018 (PRESENT); MarMot -> marmot2024 (PRESENT); von Stein and Elbaum 2022 -> vonstein2022 (PRESENT); Chen et al. 2022 / Openpilot-Deepdive -> chen2022deepdive (PRESENT); arXiv:2505.11532 adversarial study -> adversarial2025 (PRESENT); commaai issue #20704 -> commaai20704 (PRESENT). | PASS | N/A |
| 8 | Every bib entry is cited in the body | All 28 bib keys verified cited in body text above. No orphan bib entries found. One note: the contribution contract (paper_state/contribution_contract.md) also mentions commaai discussion #22212 as a second motivation citation alongside issue #20704, but #22212 does not appear in the draft body and has no bib entry. This is a minor gap: either add a second bib entry for #22212 and cite it, or omit it (the contract says both are cited "as motivation only," but the draft currently cites only #20704). | PASS (with advisory) | Advisory: add commaai discussion #22212 as a second motivation bib entry and cite it alongside #20704 in the Introduction, or document the deliberate omission. Not a blocking item. |
| 9 | Figures: all manifest-listed figures exist in report/figures/ in arXiv-accepted PNG format | All 15 PNG files listed in the Figure and Table Manifest section are present in /home/yusuf/Projects/phantom-braking/report/figures/ and confirmed valid PNG image data by the `file` command. Format is PNG (arXiv-accepted per guidelines). | PASS | N/A |
| 10 | Figure resolution adequate for arXiv rendering | DPI metadata ranges from 140 dpi (auroc_vs_alpha.png, e5_layer_localization.png, e5_submodule_localization.png, e6_detector.png, pr_curves.png, roc_curves.png) to 150 dpi (all others). Pixel dimensions range from 910x770 to 2100x1500. arXiv does not specify a minimum DPI for PNG figures; the figures will render at their native pixel dimensions in the compiled PDF. At typical single-column (~3.5 inch) or double-column (~7 inch) figure widths, 140-150 dpi with 910-2100 px width is within acceptable range for screen and print. The 910 px wide figures (pr_curves, roc_curves) at 140 dpi render at ~6.5 inches, adequate for a single column. | PASS | N/A |
| 11 | No prohibited content in body (line numbers, watermarks, margin notes, referee remarks, highlighted text, advertisements) | No such content found in the Markdown draft. The blockquote meta-header and Handoff section (flagged under item 5) are editorial notes, not arXiv-prohibited content classes; they are a submission-readiness issue rather than a format violation per se. | PASS | N/A |
| 12 | Bibliography entries have required fields | All 28 bib entries examined. Entries use a mix of @inproceedings, @article, and @misc. Each has: author, title, year. Conference proceedings have booktitle; journal articles have journal; the misc entry (commaai20704) has howpublished. Entries citing arXiv preprints include eprint and archivePrefix. Three entries note "venue unconfirmed; pin at camera-ready" (ren2021, muellerplus2025, michaelis2019, hodge2025, guosu2026). keser2025 uses booktitle for what is an arXiv preprint ("booktitle = {arXiv preprint arXiv:2501.08083}"), which is technically incorrect field use but not arXiv-submission-blocking. eigentrack2025 and adversarial2025 have the same structural issue (arXiv preprint cited as @inproceedings with booktitle). | PASS (with advisory) | Advisory: at camera-ready, change keser2025, eigentrack2025, and adversarial2025 from @inproceedings to @article or @misc with correct fields, and resolve the five "venue unconfirmed" notes by confirming or updating the venue. Neither issue blocks arXiv preprint submission. |
| 13 | License: arXiv irrevocable distribution license required; CC-BY is an upgrade option | arXiv requires an irrevocable distribution license granted at submission time. The default is arXiv's own non-exclusive distribution license. Authors may optionally upgrade to CC-BY 4.0, CC-BY-SA 4.0, or CC0 at submission. This is a submission-time selection, not a file-level requirement; no change to the draft file is needed. The repo is public (github.com/yusufdxb/supercombo-blindspot per memory context), which is compatible with any of these license choices. | PASS | N/A (choose license at submission time; CC-BY 4.0 is recommended for a public research preprint with a public repo, but arXiv's default license is also acceptable). |
| 14 | Endorsement / moderation requirement for cs.LG | arXiv requires endorsement for new submitters to cs categories. Specifically: "New users or those submitting to unfamiliar categories may require endorsements." An existing cs.LG endorsement from any registered user who has previously published in the category is sufficient. If this is a first-time cs.LG submission, plan for a 1-3 day moderation delay. The content (runtime OOD monitoring for production ML systems) is clearly within cs.LG scope. | PASS (conditional) | If the submitting account has not previously submitted to cs.LG or an adjacent cs category, obtain an endorsement from a cs.LG regular before submitting. Not a document issue; a registration / account issue. |
| 15 | Anonymization requirement | arXiv explicitly does NOT permit anonymous submissions: "no anonymous submissions." The draft currently has NO author block (see item 6 above). This is a FAIL on the opposite axis from workshop double-blind: arXiv requires named authorship. | **FAIL** | Same fix as item 6: add full author name(s) and affiliation(s) to the draft before LaTeX conversion. |

---

## Verdict

**NOT COMPLIANT** as currently drafted. Three FAILs block arXiv submission readiness:

**Priority 1 (required before submission):**

1. FAIL item 6 / item 15: No author block. arXiv prohibits anonymous submissions. Add author name(s), affiliation(s), and contact email to the draft header before LaTeX conversion and to the arXiv metadata form at submission time.

2. FAIL item 2: Abstract is 2768 characters, 848 characters over the 1920-char hard limit. The arXiv submission interface will reject the metadata field. Shorten to under 1920 characters. Retain the parity result, the 8/10 head collapse and 0/219 uncertainty statistic, the AUROC 0.996 and ~1% FPR monitor result, and the bounding sentence. Cut the cliff shape detail, localization specifics, and ImageNet-C cell counts from the metadata text.

3. FAIL item 5: The draft file contains a four-line editorial blockquote at lines 3-6 ("> Target venue: arXiv preprint...claim_ledger.md") and a "## Handoff" section at lines 617-622 that are internal council scaffolding and must be stripped from the submitted LaTeX/PDF. The "## Figure and Table Manifest" section is also an editorial artifact and should be removed or replaced by proper LaTeX figure environments.

**Priority 2 (advisories, not blocking):**

- commaai discussion #22212 is named in the contribution contract as a motivation citation but is absent from both the draft body and the bib file. Either cite it alongside #20704 in the Introduction or document the deliberate omission.
- Three bib entries (keser2025, eigentrack2025, adversarial2025) use @inproceedings with "booktitle = {arXiv preprint ...}", which is incorrect field usage. Acceptable for a preprint; fix at camera-ready.
- Five bib entries carry "venue unconfirmed; pin at camera-ready" notes. Acceptable for a preprint; resolve before any journal or conference submission.

**Note on workshop concerns:** Because the SafeAI@UAI 2026 deadline has passed and this is an arXiv preprint, the 4-page workshop limit and double-blind anonymization rules from SafeAI do NOT apply. There is no page limit on arXiv, and the anonymization concern for arXiv is the reverse of double-blind: arXiv requires named authorship.

---

## Honesty summary

**What was verified directly:**
- arXiv submission rules fetched from four official info.arxiv.org pages and quoted above; rules not recalled from memory.
- Category descriptions fetched from arxiv.org/category_taxonomy.
- Abstract character count measured via bash (awk extraction + wc -c): 2768 characters.
- All 28 bib keys listed in references.bib; all 28 verified cited in rewritten_draft.md by grep.
- All 15 manifest-listed figure files confirmed present in report/figures/ by ls and `file` command.
- Figure formats confirmed PNG (arXiv-accepted) with pixel dimensions and DPI metadata read via PIL.
- Placeholder marker grep returned zero hits.
- Editorial meta-content (blockquote header, Handoff section) confirmed present at specific lines.
- No LaTeX \cite{} or \ref{} commands are used (draft is Markdown), confirmed by grep.

**What was NOT verified:**
- The arXiv submission interface behavior was not tested live; the 1920-char limit is taken from the info.arxiv.org/help/prep.html page fetched in this session (consistent with the well-known constraint).
- Figure rendering quality in a compiled LaTeX PDF was not tested; the DPI/pixel assessment is an estimate.
- Whether the submitting account has an existing cs.LG endorsement was not checked (account-level information not accessible).
- The "## Figure and Table Manifest" section disposition (strip vs retain as appendix) depends on the LaTeX conversion step not yet performed.

**Riskiest unverified assumption:** The abstract character count is measured on the Markdown prose extracted from the draft. When the draft is converted to LaTeX and the abstract is typed into the arXiv submission form, the count may differ slightly if special characters ("+/-", brackets, numbers) are handled differently. The measured 2768 chars is far enough above 1920 that this uncertainty does not change the FAIL verdict: even a 10% measurement error leaves the abstract 630+ chars over the limit.
