# Literature Map

Written by paper-litmapper on 2026-05-30. Bound by
`paper_state/contribution_contract.md` (locked 2026-05-30), and built on
`paper_state/framing_memo.md` and `paper_state/research_brief.md`. Every external
collision judgment below rests on a quote I FETCHED this run (WebFetch / direct-link
WebSearch) with the URL or arXiv id, never on model memory. Where I could not fetch a
source, it is marked `[UNVERIFIED]` and is not allowed to clear a collision.

The locked contribution sentence this map is judged against:

> A controlled distribution-shift teardown of a single shipped production driving model
> (openpilot v0.9.7 supercombo) shows it fails silently under visual out-of-distribution
> input (simultaneous output-head collapse, recurrent-feature freeze, and a non-responsive
> uncertainty channel), localizes the failure downstream of the vision encoder, and
> demonstrates that a zero-retraining recurrent-feature monitor (E6, the rolling temporal
> spread of the 512-D state) detects the collapse about 0.23 blend-units before the outputs
> cliff and at about 1% real-driving false-positive rate, where location-based feature-space
> OOD scores (Mahalanobis, Relative Mahalanobis, KNN) fail to transfer across real corpora.

The four pre-emption surfaces a reviewer would attack, and what this map establishes:
1. A prior silent-collapse-on-supercombo teardown. NOT FOUND (collision hunt below).
2. A prior recurrent-spread / second-order-state OOD monitor on a shipped driving model.
   NOT FOUND; the nearest mechanism-twin (EigenTrack) is on LLMs/VLMs. NEAR-MISS, must cite.
3. The location-based-scores-fail-to-transfer claim being already published. Not found as a
   transfer/LOCO result on a recurrent driver state.
4. The uncertainty-channel-stays-silent claim being already published for this model. Not
   found; the UQ-misbehaviour line (Grewal/Stocco) assumes the opposite.

---

## Lineage tree

Two ancestral lines converge on this paper. The contribution sits at the intersection: a
SECOND-ORDER statistic (line B) on the RECURRENT STATE of a SHIPPED end-to-end driver
(neither line's home turf), evaluated against the LOCATION-BASED family (line A) under
cross-corpus transfer.

- LINE A: location-based feature-space OOD (the family this paper runs as baselines and
  shows fails to TRANSFER, not fails to separate)
  - Hendrycks and Gimpel 2017, MSP (arXiv:1610.02136). Output-side softmax-confidence
    floor. ANCESTOR of the output-side scores; E3 is precisely its failure mode on
    supercombo (uncertainty rises only 1.20x-1.84x; 0/220 OOD frames over real p95).
    - Liu et al. 2020, Energy (arXiv:2010.03759). Output-side energy score; same family,
      same blind spot to a collapse that keeps outputs in range.
  - Lee et al. 2018, Mahalanobis (arXiv:1807.03888, NeurIPS 2018). Distance-from-fitted-
    Gaussian-mean on features. THE first-order-distance ancestor and the closest
    methodological neighbor by substrate (same 512-D feature, opposite statistic choice).
    - Ren et al. 2021, Relative Mahalanobis / RMD (arXiv:2106.09022). The canonical
      near-OOD fix to Lee 2018; still location-based, still 100% LOCO FPR here.
    - Sun et al. 2022, KNN (arXiv:2204.06507, ICML 2022). Non-parametric absolute
      nearest-neighbor distance; the strongest applicable baseline (TIES E6 at AUROC 1.000),
      and the one whose 100% LOCO FPR makes the transfer/calibration delta the real claim.
    - Wang et al. 2022, ViM (arXiv:2203.10807). Feature-residual + logit hybrid; modern
      location-based baseline cited to pre-empt "no recent baseline."
    - Mueller and Hein 2025, Mahalanobis++ (arXiv:2505.18032). Keeps the Gaussian-feature
      family LIVE in 2025 (l2-normalization fix); proves running Lee 2018 here is a fair
      current comparison, not a strawman.
  - Yang et al. 2022, OpenOOD (arXiv:2210.07242, NeurIPS D&B). Codifies the whole line A
    into one taxonomy/codebase; the vocabulary anchor, not a leaderboard this paper ranks on.

- LINE B: internal-activation / higher-order-statistic runtime monitoring (the family this
  paper's monitor BELONGS to, pushed onto a recurrent state)
  - Cheng et al. 2018, Runtime Neuron Activation Patterns (arXiv:1809.06573). ANCESTOR of
    internal-state runtime monitoring: stores BINARIZED neuron patterns, compares by Hamming
    distance, on CLASSIFIERS. First-order, per-frame, discrete.
    - Sastry and Oore 2020, Gram-matrix OOD (arXiv:1912.12510, ICML 2020 sastry20a). First
      move to a HIGHER-ORDER feature statistic (Gram / pairwise correlations) for OOD.
      THE closest lineage point to "second-order, not distance-from-mean."
      - THIS PAPER: rolling temporal spread (a second-order trace) of the 512-D RECURRENT
        state of a SHIPPED end-to-end driver. Inherits the higher-order-statistic idea from
        Sastry/Oore and the internal-monitor idea from Cheng, but applies it to a recurrent
        state (neither did) and to a shipped production driving model (neither did), and is
        the statistic that catches the FREEZE mode (spread crashing 0.25 -> 0.00 across the
        cliff) a per-frame pattern/Gram check on outputs would miss.
    - PARALLEL MODERN TWIN (post-dates the family, converges on the mechanism, different
      substrate): EigenTrack 2025 (arXiv:2509.15735), covariance-SPECTRUM of hidden
      activations + recurrent classifier, on LLMs/VLMs. See Collision watch C2: NEAR-MISS,
      not pre-empting (wrong substrate, full eigenspectrum/RMT vs single-trace spread, and
      independent line on language models).

- CROSS-CUTTING (the AV-safety / driving-specific frontier this paper enters, none of which
  monitor a recurrent shipped-driver state)
  - Stocco et al. 2020, SelfOracle (ICSE 2020) -> Grewal/Tonella/Stocco 2024
    (arXiv:2404.18573). Predict misbehaviour from model confidence / Bayesian UQ. ASSUMES the
    confidence signal is informative; E3 is direct evidence it is silent for supercombo.
  - Keser et al. 2025 (arXiv:2501.08083). Density (location-based) on a FROZEN vision
    foundation-model encoder. CLOSEST AV-native neighbor; one stage upstream of E6's substrate.
  - Guo and Su 2025 (arXiv:2509.13577) / 2026 (arXiv:2603.14603). Latent-dynamics QCD/MMD
    OOD on a STANDALONE trajectory predictor, with PROVABLE guarantees. Second-closest
    neighbor; different model class and guarantee type.
  - Chen et al. 2022, Openpilot-Deepdive (arXiv:2206.08176). Static input/output/architecture
    teardown of supercombo. THE anchor this paper extends to a RUNTIME distribution-shift
    teardown.

A rendered Graphviz version of this two-line convergence can be produced for the camera-ready
(the project has `dot` + `rsvg-convert` available); the nested list is the load-bearing form
for the .md. ASSUMPTION: a figure is not required for the arXiv preprint draft; flagged so the
drafter can request one.

---

## Closest-competitor contrast table

Axes are concrete and checkable. Every cell is a value a reviewer could verify against the
cited source or the committed `report/*.md`. Cells that cannot be filled from a fetched source
are `[UNVERIFIED]`.

Axes:
- SUBSTRATE: what activations the score reads.
- SCORE TYPE: the statistic (first-order distance / density vs second-order spread/trace).
- CALIBRATION: how the threshold/FPR is set and on what.
- TARGET MODEL: research IL vs frozen encoder vs standalone predictor vs shipped end-to-end.
- GUARANTEE: none / empirical / provable.

| Method (author, year) | Substrate | Score type | Calibration | Target model | Guarantee |
|---|---|---|---|---|---|
| Keser et al. 2025 (arXiv:2501.08083) | Frozen vision foundation-model encoder features | Feature-DENSITY (location-based) ID score | Density model fit on training features (no cross-corpus / LOCO FPR reported) | Frozen perception/foundation-model ENCODER | None stated |
| Guo and Su 2026 (arXiv:2603.14603) | Latent dynamics of a trajectory predictor | Cumulative MMD / QCD changepoint statistic | Provable bounds on delay and false alarm | STANDALONE trajectory predictor | PROVABLE (QCD/MMD guarantees) |
| Cheng et al. 2018 (arXiv:1809.06573) | Hidden-layer neuron activations | BINARIZED activation pattern, Hamming distance (first-order, discrete) | In-pattern vs out-of-pattern set membership from training pass | Feed-forward CLASSIFIER | None |
| Lee et al. 2018 (arXiv:1807.03888) | Deep features (here: the 512-D recurrent state) | Distance-from-fitted-Gaussian-MEAN (first-order, location) | Threshold on Mahalanobis distance; here 100% LOCO FPR, AUROC 0.159 below chance | Research/feed-forward DNN (here: run on supercombo) | None |
| Sun et al. 2022 (arXiv:2204.06507) | Deep features (here: the 512-D recurrent state) | Absolute nearest-neighbor DISTANCE (first-order, location, non-parametric) | Threshold on k-NN distance; here TIES E6 at AUROC 1.000 but 100% LOCO FPR | Research DNN (here: run on supercombo) | None |
| EigenTrack 2025 (arXiv:2509.15735) | Hidden activations of an LLM/VLM (sliding window) | Covariance-SPECTRUM stats (eigenvalues, spectral gap, entropy, RMT/Marchenko-Pastur); second-order, multi-stat | Trained lightweight recurrent classifier on the spectral stream | LLM / VLM (NOT a driving model) | None |
| THIS WORK (E6, report/e6_results.md) | RECURRENT state (512-D) of a SHIPPED end-to-end driver | Rolling temporal SPREAD (single second-order trace; location-INVARIANT) | Empirical leave-one-corpus-out real-driving FPR ~1.03% (N=2 two-fold), AUROC 0.996 | SHIPPED production end-to-end driver (openpilot v0.9.7 supercombo, parity-verified) | None (empirical LOCO estimate, NOT provable, NOT a production FPR) |

Reading of the table (the bounded novelty as the set of cells true here and false in every row
above): SUBSTRATE = recurrent state of a shipped end-to-end driver (Keser is the encoder;
Cheng/Lee/Sun are not recurrent; Guo-Su is a standalone predictor; EigenTrack is an LLM/VLM).
SCORE TYPE = a single location-invariant second-order trace (Lee/Sun are first-order location;
Keser is density; EigenTrack is a multi-statistic eigenspectrum). CALIBRATION = leave-one-
corpus-out real-driving FPR where the location-based scores hit 100% LOCO FPR (no neighbor
reports cross-corpus LOCO on a recurrent driver state). No single row matches on all three.

Note enforced from the contract (exclusion line 108-111): the table does NOT claim E6 "beats"
Sun 2022. Sun's KNN-50 TIES E6 at AUROC 1.000 at alpha=1.0. The only axis on which they
differ is CALIBRATION/transfer (100% LOCO FPR vs ~1%). That is the honest claim and the table
encodes it.

---

## Collision watch

Ordered most-threatening first. A candidate clears only on a FETCHED quote of what it
actually does. No candidate reached BLOCKING.

### C1. A prior silent-collapse-on-supercombo teardown (the headline-killer)

- Candidate class: any paper that already reports openpilot/supercombo failing silently
  (outputs collapsing while uncertainty stays quiet) on simulated / rendered input.
- Adversarial searches run: "openpilot supercombo silent failure simulation distribution
  shift teardown out-of-distribution"; "openpilot CARLA simulation rendered input model
  collapse perception failure"; "openpilot comma supercombo academic analysis parity
  reimplementation 2024 2025 distribution shift evaluation arxiv".
- What surfaced and its FETCHED disposition:
  - Chen et al. 2022, Openpilot-Deepdive (arXiv:2206.08176). FETCHED QUOTE (research brief,
    re-confirmed via search this run): "we deep-dive into Openpilot and conclude that its key
    to success is the end-to-end system design." This is a STATIC input/output/architecture
    teardown and a reimplementation; it does NOT induce or report a sim-rendered silent
    collapse, a recurrent-state freeze, or a non-responsive uncertainty channel. It is the
    anchor this paper EXTENDS, not a pre-emption.
  - "Revisiting Adversarial Perception Attacks and Defense Methods on Autonomous Driving
    Systems" (arXiv:2505.11532). FETCHED QUOTE: "for the regression task, we use the
    Supercombo model, an end-to-end model used on a production ADS" and the failure mode is
    "deliberate perturbations designed to deceive perception systems." Pre-emption analysis:
    targets supercombo but via ADVERSARIAL perturbations with INPUT-LEVEL defenses, and
    "does not propose any runtime out-of-distribution monitor examining recurrent or hidden
    states" (FETCHED). Different failure mode (adversarial attack, not sim-rendered silent
    collapse), no recurrent monitor, no uncertainty-silence finding.
  - The CARLA+openpilot hits (commaai wiki, UVA-DSA/openpilot-CARLA, setup issues) are
    engineering integration guides, not academic teardowns of model collapse.
- Severity: NONE. No prior teardown reports the silent-collapse phenomenon on this model.

### C2. A prior second-order / spread / covariance OOD monitor on a recurrent state (the mechanism-killer)

- Candidate: EigenTrack, "Spectral Activation Feature Tracking for Hallucination and
  Out-of-Distribution Detection in LLMs and VLMs," 2025 (arXiv:2509.15735, v1 Sep 2025; later
  v4 retitled "Temporal Spectral Analysis of Hidden Activations").
- FETCHED QUOTE of what it does: "By streaming covariance-spectrum statistics such as
  entropy, eigenvalue gaps, and KL divergence from random baselines into a lightweight
  recurrent classifier" and "EigenTrack tracks temporal shifts in representation structure
  that signal hallucination and OOD drift before surface errors appear." On substrate:
  "Large language models (LLMs) offer broad utility but remain prone to hallucination and
  out-of-distribution (OOD) errors." Source: https://arxiv.org/abs/2509.15735.
- Pre-emption analysis: this is the single closest paper on MECHANISM. It is a SECOND-ORDER
  statistic (covariance spectrum) of HIDDEN ACTIVATIONS, single forward pass, with EARLY
  WARNING ("before surface errors appear") streamed through a recurrent classifier. That
  rhymes hard with E6 (a second-order spread of a recurrent state, single forward pass, fires
  before the outputs cliff). It differs on three checkable axes: (a) SUBSTRATE: LLMs/VLMs, NOT
  a driving model. A targeted search ("EigenTrack ... autonomous driving vehicle perception
  applied") returned, FETCHED: "the search results don't contain specific information about
  EigenTrack applications to autonomous driving vehicle perception systems." (b) STATISTIC:
  the FULL covariance EIGENSPECTRUM plus RMT/Marchenko-Pastur features fed to a TRAINED
  classifier, vs E6's single location-invariant TRACE (the rolling spread) with a calibrated
  threshold and no trained classifier. (c) EVALUATION AXIS: hallucination/OOD on language
  benchmarks, vs cross-corpus LOCO FPR on real-driving frames of a shipped driver. It is also
  a PARALLEL, near-simultaneous discovery (Sep 2025) on a different domain, not prior work that
  did THIS contribution.
- Severity: NEAR-MISS. It does not pre-empt the locked claim (wrong substrate, wrong model
  class, different statistic and evaluation), but it is the paper a reviewer WILL name as
  "someone already monitors a second-order hidden-activation statistic for OOD with early
  warning." The draft MUST cite it and state the substrate/statistic/evaluation delta
  explicitly. Treated as a hard citation-gap (see list). Not BLOCKING.

### C3. A prior latent-state OOD runtime monitor presented as the same contribution

- Candidate: Guo and Su, "Latent Dynamics-Aware OOD Monitoring for Trajectory Prediction with
  Provable Guarantees," 2026 (arXiv:2603.14603).
- FETCHED QUOTE: "by leveraging this structure we extend the cumulative Maximum Mean
  Discrepancy approach to enable detection without requiring explicit knowledge of the
  post-change distribution while still admitting provable guarantees on delay and false
  alarms." Source: arXiv:2603.14603 (re-confirmed via framing memo, fetched twice that run).
- Pre-emption analysis: monitors a LATENT STATE for OOD, but on a STANDALONE TRAJECTORY
  PREDICTOR, with PROVABLE QCD/MMD guarantees. This work targets a SHIPPED production
  end-to-end model, makes NO provable guarantee (a calibrated empirical LOCO FPR), and pairs
  the monitor with a collapse teardown + localization. Different model class, different
  evidence basis. CAVEAT: 2603.14603 is a forward-numbered 2026 preprint with no final venue;
  the title/target are fetched-confirmed but venue/DOI must be re-pinned at camera-ready.
- Severity: NEAR-MISS. Closest on the "monitor a latent/recurrent state for OOD" idea, but a
  different model class and a guarantee type this paper explicitly does not claim. Not BLOCKING.

### C4. A prior result that the model's own uncertainty channel is silent (the E3-killer)

- Candidate class: any paper showing a shipped driver's predictive-uncertainty / output
  confidence stays quiet under OOD (which would pre-empt E3, the load-bearing safety hook).
- FETCHED disposition: the closest line, Grewal/Tonella/Stocco 2024 (arXiv:2404.18573),
  FETCHED QUOTE: "evaluates different Bayesian uncertainty quantification methods ... for the
  anticipatory testing of safety-critical misbehaviours." This ASSUMES the confidence signal
  is informative and BUILDS a misbehaviour predictor on it. It does not report the opposite
  (a non-responsive uncertainty channel on a shipped model); E3 (0/220 OOD frames over real
  p95) is the contrary finding, on the gap this line assumes away.
- Severity: NONE. No prior work pre-empts the uncertainty-silence finding for this model.

### C5. A prior Gram-matrix / higher-order-statistic OOD monitor on a driving/temporal model

- Candidate: Sastry and Oore, "Detecting Out-of-Distribution Examples with Gram Matrices,"
  ICML 2020 (arXiv:1912.12510).
- FETCHED QUOTE (research brief): "We find that characterizing activity patterns by Gram
  matrices and identifying anomalies in gram matrix values can yield high OOD detection rates."
- Pre-emption analysis: the closest LINEAGE point to a higher-order feature statistic, but it
  is per-frame Gram correlations on feed-forward CLASSIFIER features (CIFAR/ImageNet-style),
  not a temporal spread of a RECURRENT state, and not on a driving model. It is the ancestor
  E6 generalizes, not a pre-emption.
- Severity: NONE (lineage ancestor, must cite).

### Collision-hunt completeness note

The two pre-emption surfaces the orchestrator named (prior silent-collapse-on-supercombo
teardown; prior recurrent-spread OOD monitor on a shipped driving model) were each hunted
with at least two adversarial queries and every surfaced candidate fetched. NO BLOCKING
collision was found. The closest live threat is EigenTrack (C2, NEAR-MISS): a parallel
second-order-hidden-activation OOD monitor with early warning, on LLMs/VLMs rather than a
driving model. The contribution survives as locked, contingent on the draft citing EigenTrack
and stating the substrate/statistic/evaluation delta (the union novelty is not weakened, but
the "we are the first to use a second-order hidden-activation statistic for OOD" framing is
NOT available and must never be written).

---

## Citation-gap list

Papers a reviewer will expect, that are not yet captured (or are mis-captured) and a one-line
reason each.

- EigenTrack, 2025 (arXiv:2509.15735). HARD GAP. The nearest mechanism-twin (second-order /
  covariance-spectrum of hidden activations, single pass, early warning); a reviewer will say
  "this already exists." Cite and state the substrate (LLM/VLM vs shipped driver), statistic
  (full eigenspectrum/RMT vs single spread-trace), and evaluation (language OOD vs cross-corpus
  LOCO FPR) deltas. Absent from the brief and framing memo; surfaced this run.
- EigenScore, 2025 (arXiv:2510.07206), "OOD Detection using Covariance in Diffusion Models."
  SOFT GAP. Reinforces that the covariance-statistic OOD family is active in 2025; one cite to
  show E6's second-order choice is current, not idiosyncratic. Surfaced this run, FETCHED title
  only; mark `[UNVERIFIED]` content until the citation-verifier fetches the abstract.
- CARLA (Dosovitskiy et al. 2017, arXiv:1711.03938). HARD GAP. The OOD axis is CARLA-rendered;
  the simulator must be cited by its source paper. Listed `[UNVERIFIED]` in the brief
  (correction item 6); the citation-verifier must fetch and pin it.
- Henriksson et al., OOD-on-AV-datasets. HARD GAP and a BLOCKER per the brief: at least three
  distinct real papers exist (arXiv:2103.15580; 2204.12378; 2401.17013) and the existing
  "RefSQ 2023" attribution is unconfirmed. Pin ONE id (arXiv:2401.17013 is the most on-point
  for AV datasets) before drafting; do not cite the ambiguous label.
- Filos et al. 2020 (arXiv:2006.14911), "Can Autonomous Vehicles Identify, Recover From, and
  Adapt to Distribution Shifts?" ICML 2020. EXPECTED. The canonical AV-distribution-shift
  framing paper; confirmed off-[UNVERIFIED] in the brief, must appear in related work.
- Stocco et al. 2020, SelfOracle (ICSE 2020). EXPECTED. The origin of the misbehaviour-
  prediction line that Grewal 2024 continues; cite as the lineage root for the UQ-monitor
  comparison (and the assumption E3 falsifies).
- NECO, ICLR 2024 (arXiv:2310.06823). EXPECTED (cite-only, do NOT run). Pre-empts the "why not
  the newest OOD method" reviewer; it is a classification-head property and supercombo is
  multi-head regression, so naming-and-excusing is the right move.
- The dozen secondary leads still `[UNVERIFIED]` in the brief (ODIN, deep ensembles,
  McAllister, Bogdoll survey, Codevilla, Norden, DeepXplore/DeepTest/DeepRoad, MarMot,
  Parallel-Activations-Drift arXiv:2404.07776, Topological Uncertainty arXiv:2105.04404,
  the 3D-object-detection and adversarial-AV 2025 papers, OpenOOD-VLM). The citation-verifier
  must fetch each before it enters the draft; none may be cited from memory.

### REQUIRED citation correction (carried forward from the research brief, re-verified this run)

- WRONG AUTHOR AND VENUE for the openpilot falsification paper. `docs/related_work.md` (line
  ~72) and `docs/paper_plan.md` (line ~58) cite "Geretti et al., ... GPCE/SPLASH 2022." This is
  WRONG. The actual paper is von Stein and Elbaum, "Finding Property Violations through Network
  Falsification: Challenges, Adaptations and Lessons Learned from OpenPilot," ASE 2022 (Industry
  Showcase), DOI 10.1145/3551349.3559500. FETCHED QUOTE confirming authorship this run:
  "Authors: Meriel von Stein and Sebastian Elbaum, both from the University of Virginia" and
  "presented at the 37th IEEE/ACM International Conference on Automated Software Engineering
  (ASE '22) in October 2022." Source: https://dl.acm.org/doi/10.1145/3551349.3559500 (ACM
  landing returned 403 on direct WebFetch this run; authorship/venue confirmed via the ACM
  full-HTML and conf.researchr.org search hits and the authors' own UVA copy at
  https://missmeriel.github.io/files/publications/ASE2022-OpenPilot_industry_track.pdf). This
  is a fabricated-looking author/venue and MUST be replaced before drafting; it is exactly the
  error a software-engineering-literate reviewer catches. Note: this paper studies FALSIFICATION
  (directed adversarial input generation), so it is MOTIVATION/related-testing-work only, NOT a
  pre-emption of the silent-collapse contribution (see C1 reasoning).

### Other citation-metadata corrections carried from the brief (lower stakes, still required)

- Sastry and Oore: use the ICML 2020 proceedings title "Detecting Out-of-Distribution Examples
  with Gram Matrices" with arXiv id 1912.12510 (whose arXiv title is the longer
  "...with In-distribution Examples and Gram Matrices"). Venue ICML 2020 is CORRECT.
- ImageNet-C: pin arXiv:1903.12261 (Hendrycks and Dietterich, ICLR 2019), not the older 2018
  preprint id.
- Michaelis et al.: use the full title "Benchmarking Robustness in Object Detection: Autonomous
  Driving when Winter is Coming" (arXiv:1907.07484).

---

## Verdict

NOVELTY DEFENSIBLE AS LOCKED. No re-lock required.

The collision hunt found NO BLOCKING pre-emption of the locked contribution sentence on any of
its four attack surfaces:
- No prior silent-collapse-on-supercombo teardown exists (C1: NONE).
- No prior recurrent-spread / second-order-state OOD monitor on a SHIPPED DRIVING MODEL exists
  (C2 EigenTrack is the nearest mechanism-twin but on LLMs/VLMs, NEAR-MISS).
- No prior cross-corpus-LOCO-transfer result on a recurrent driver state pre-empts the
  baseline-transfer claim (Lee/Sun/Keser are location-based and not LOCO-on-recurrent-driver).
- No prior uncertainty-channel-is-silent finding for this model exists (C4: NONE; the UQ line
  assumes the opposite).

The bounded novelty (the union of the contrast-table deltas) holds exactly as the framing memo
states: a zero-retraining, location-invariant second-order (rolling-spread) monitor on the
recurrent state of a parity-verified SHIPPED end-to-end driving model, paired with a
silent-collapse teardown localized downstream of the encoder, evaluated under LOCO FPR where the
named location-based scores fail to transfer. No single neighbor has all of those.

TWO BINDING CONDITIONS on the drafter (not re-locks, enforcement of what this hunt revealed):
1. EigenTrack (arXiv:2509.15735) MUST be cited and dispositioned (C2). The framing "first to
   use a second-order hidden-activation statistic for OOD" is FALSE and forbidden; the available,
   true framing is "first on the recurrent state of a SHIPPED end-to-end driver, evaluated under
   cross-corpus LOCO transfer, with a single location-invariant trace." Stay inside that.
2. The von Stein and Elbaum / ASE 2022 citation correction MUST be applied; the existing
   "Geretti et al., GPCE/SPLASH 2022" is wrong and reviewer-catchable.

VENUE NOTE carried forward (decision-relevant, outside this agent's authority to set): the
research brief's strongest finding is that the SafeAI@UAI 2026 submission deadline (May 28,
2026) appears to be PAST as of today (2026-05-30), with no extension shown on the workshop page.
Per the orchestrator's instruction this run, the target venue is now the ARXIV PREPRINT, with
SafeAI (or a successor venue) as a later, non-archival resubmission ("there will be no
proceedings, so authors are free to submit their work elsewhere," fetched from the workshop CFP).
This does not affect the novelty verdict; it only changes the immediate deliverable to the
preprint. ROUTE to the orchestrator (not to paper-contribution-locker, since novelty is not
blocked): confirm the live SafeAI deadline before any SafeAI-specific formatting, and treat the
arXiv preprint as the primary target.

---

## Honesty summary (end of turn)

What I VERIFIED this run (each with a fetched quote + URL/arXiv id in the body):
- The von Stein and Elbaum ASE 2022 authorship/venue, confirming the required correction away
  from the wrong "Geretti et al., GPCE/SPLASH 2022."
- EigenTrack (arXiv:2509.15735) is on LLMs/VLMs (not a driving model), uses a covariance-
  SPECTRUM + RMT statistic through a trained recurrent classifier, and a targeted search
  confirmed no autonomous-driving application: graded NEAR-MISS, not BLOCKING.
- The adversarial-perception openpilot paper (arXiv:2505.11532) targets supercombo via
  ADVERSARIAL attacks with input-level defenses and proposes NO recurrent/hidden-state monitor:
  different failure mode, NONE.
- The GPSSM OOD monitor (arXiv:2309.06655) is on a quadruped/terrains, not driving: NONE.
- No prior silent-collapse-on-supercombo teardown and no prior recurrent-spread OOD monitor on
  a shipped driving model surfaced across six adversarial queries.
- I carried both required findings forward: (1) the von Stein/Elbaum ASE 2022 correction, and
  (2) the venue shift to the arXiv preprint (SafeAI@UAI 2026 deadline appears past).

What I did NOT verify (and must not be treated as cleared):
- I did NOT independently re-fetch the five framing-memo neighbor quotes (Keser, Guo-Su, Cheng,
  Lee, Sun); I reused the framing memo's FETCHED quotes, which were obtained that run. They are
  fetched-grounded but at one remove in this artifact.
- EigenScore (arXiv:2510.07206) content is `[UNVERIFIED]` (title-only); CARLA (arXiv:1711.03938)
  and the Henriksson id are `[UNVERIFIED]` and routed to the citation-verifier.
- The ACM landing page for the falsification paper returned 403 on direct WebFetch; authorship
  was confirmed via the ACM full-HTML/conf-page search hits and the authors' own UVA-hosted PDF,
  not a direct fetch of the canonical DOI page.

THE SINGLE RISKIEST UNVERIFIED ASSUMPTION: that EigenTrack (arXiv:2509.15735) is the FULL extent
of the parallel second-order-hidden-activation-OOD line and that none of that line has been
applied to a shipped driving model. I confirmed EigenTrack itself is LLM/VLM-only via a targeted
search, but the covariance/spectral-OOD family is clearly active in 2025 (EigenScore, the RMT
"Spectral Geometry for Deep Learning" hit, Dynamic Covariance Calibration arXiv:2506.09399), and
I did not exhaustively fetch each to rule out a driving-model application. The citation-verifier
should fetch EigenScore and the RMT/covariance-OOD 2025 cluster before camera-ready to confirm
none lands on a recurrent shipped-driver state. If one does, C2 would need re-grading.
