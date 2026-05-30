# Framing Memo

Written by paper-stanford-framer on 2026-05-30. Bound by
`paper_state/contribution_contract.md` (locked 2026-05-30). Every sentence below lives
inside that contract's claim boundary. Headline numbers are quoted from the committed
`report/*.md` files (E6, metrics, teardown, e7). Neighbor quotes are FETCHED, not recalled.

This memo is the single home for the closest-neighbors list. The orchestrator should have
`paper-contribution-locker` backfill the contract's "Closest neighbors (filled by
paper-stanford-framer)" placeholder (line 144-146) from the table below.

---

## Hook (one paragraph)

Production L2 driving stacks are validated, in large part, in simulation, and that
practice rests on an unstated assumption: that the shipped model either behaves the same
on rendered input as on real input, or at least fails loudly when it does not. We test
that assumption on one real, deployed model (openpilot v0.9.7 supercombo) and find the
dangerous answer. On CARLA-rendered clean roads the model fails silently: 8 of 10 output
heads collapse to under 1% of their real-driving temporal activity and the 512-D recurrent
state freezes to about 1e-5 of its real spread, yet the model's own predictive-uncertainty
heads rise only 1.20x to 1.84x and not one out-of-distribution frame (0 of 220) crosses
the real-driving 95th percentile. Nothing the model emits flags the collapse, so the gap
is precise: a sim "pass" can be the model having collapsed to a safe-looking default rather
than the model perceiving, and the output-side and uncertainty signals a downstream safety
case would trust are exactly the signals that stay quiet. The closest published monitors
either watch a frozen perception encoder's feature density (Keser et al. 2025) or a
standalone trajectory predictor's latent dynamics (Guo and Su 2026); neither targets the
recurrent state of a shipped end-to-end driving model, and the standard location-based
feature scores that one would reach for first (Mahalanobis, Relative Mahalanobis, KNN) do
not transfer across our two real corpora (100% leave-one-corpus-out FPR each). We show the
hidden signal is recoverable from the model's own recurrent feature with a single
second-order statistic, the rolling temporal spread of the 512-D state, calibrated
leave-one-corpus-out to about a 1% real-driving false-positive rate (N=2, a two-fold
estimate), which separates the collapse (AUROC 0.996) about 0.23 blend-units before the
outputs cliff. This is a bounded negative result on N=1 model with a collapse-specific
monitor: a corruption sweep shows the collapse is sim-specific (no ImageNet-C corruption
reproduces it) and the monitor is collapse-specific, not a universal OOD detector.

---

## Candidate titles (3)

1. Silent Collapse: A Production Driving Model Fails on Simulated Input Without Raising Its
   Own Uncertainty, and a Recurrent-Feature Monitor Recovers the Signal
2. Does openpilot Know When It Is Blind? A Distribution-Shift Teardown of supercombo v0.9.7
   and a Zero-Retraining Recurrent-State Monitor
3. The Uncertainty Channel Stays Quiet: Localizing and Monitoring Silent Output Collapse in
   a Shipped End-to-End Driving Model

(Title 1 is the mechanism-forward option and my recommendation: it names the failure mode
"silent collapse," states it is a production model, and names the monitor's substrate.
Title 2 is the punchier, venue-friendly option that opens on the user-facing question;
keep "supercombo v0.9.7" in it so the N=1 scope is on the cover. Title 3 leads with the
safety-relevant centerpiece (E3, the non-responsive uncertainty head). None overclaim: all
three say "a"/"a shipped"/named-version, never "production driving models" plural.)

---

## Abstract skeleton (5-6 sentence arc)

1. **Context / stakes.** Shipped L2 driving models are validated largely in simulation, a
   practice that silently assumes the model behaves the same on rendered input as on real
   input, or at least signals when it does not.
2. **The gap / failure mode.** We instrument one deployed model (openpilot v0.9.7
   supercombo), parity-verified to within +/-0.5 m/s^2 of comma's reference on 100% of 1159
   real frames (median abs delta 0.04 m/s^2), and find that on CARLA-rendered roads it
   fails silently: 8 of 10 output heads collapse to under 1% of real temporal activity and
   the 512-D recurrent state freezes, while its own uncertainty heads rise only 1.20x to
   1.84x and 0 of 220 OOD frames exceed the real-driving p95.
3. **This paper's move.** We characterize the collapse as a hard cliff on a real-to-sim
   blend axis (transition width 0.015), localize it downstream of the vision encoder (every
   encoder stage stays at or above real activity; the cliff enters at the recurrent
   summarizer and action-block feedback path), and ask whether the model's own internal
   features carry the signal its outputs hide.
4. **How it works.** A zero-retraining monitor on the rolling temporal spread of the 512-D
   recurrent state (one O(d) statistic per forward pass, no architecture change) is
   calibrated leave-one-corpus-out to about a 1% real-driving false-positive rate.
5. **Headline result.** The monitor separates the collapse at AUROC 0.996 [0.992, 1.000]
   and fires about 0.23 blend-units before the output cliff, where the location-based
   feature scores one would default to (Mahalanobis, Relative Mahalanobis, KNN) each hit
   100% leave-one-corpus-out FPR and fail to transfer across the two real corpora.
6. **Bounded takeaway.** This is a single-model negative finding with a collapse-specific
   monitor, not a general OOD detector: an ImageNet-C sweep shows the silent collapse is
   sim-specific (no corruption reproduces it) and the monitor is near chance on most
   photometric corruptions, so the contribution is that output-side and location-based
   signals alone are insufficient for this one shipped model's safety case, and a
   second-order recurrent-state monitor is a cheap complement that the present evidence
   does not claim generalizes.

(N=2 LOCO must be stated as a two-fold estimate, not a production FPR, in the abstract per
contract exclusion list line 106-107. The word "complement" not "replacement," and "this
one shipped model" not "production driving models," are load-bearing against the exclusion
list lines 97-98 and 115.)

---

## Closest neighbors (named, quoted, delta)

Every row is a named, published work with a fetched verbatim quote and a real arXiv id. No
row rests on model memory. The union of these deltas IS the bounded novelty: a
zero-retraining second-order (spread, location-invariant) monitor on the recurrent state of
a parity-verified SHIPPED end-to-end driving model, paired with a collapse teardown,
evaluated under leave-one-corpus-out FPR, where the named location-based scores fail to
transfer. No single neighbor has all of those true.

| Paper (author, year, venue) | What it actually does (FETCHED QUOTE) | Source | Delta vs this contribution |
|---|---|---|---|
| Keser, Orhan, Amini-Naieni, Schwalbe, Knoll, Rottmann, "Benchmarking Vision Foundation Models for Input Monitoring in Autonomous Driving," 2025 | "Find a full model of the training data's feature distribution, to then use its density at new points as in-distribution (ID) score." | arXiv:2501.08083 | Closest neighbor. They monitor the **input/feature density of a frozen vision foundation-model encoder** as a density (location-based) ID score. We monitor the **rolling spread (a second-order, location-invariant statistic) of the recurrent state of a shipped end-to-end driving model**, one stage downstream of the encoder. Their score is exactly the location-based class our baselines instantiate (Mahalanobis/KNN), which here hit 100% LOCO FPR; ours survives LOCO at ~1%. Different substrate (recurrent state vs encoder features), different statistic (spread vs density), different model class (shipped end-to-end vs foundation-model perception). |
| Guo and Su, "Latent Dynamics-Aware OOD Monitoring for Trajectory Prediction with Provable Guarantees," 2026 | "by leveraging this structure we extend the cumulative Maximum Mean Discrepancy approach to enable detection without requiring explicit knowledge of the post-change distribution while still admitting provable guarantees on delay and false alarms" | arXiv:2603.14603 | Also monitors a latent state for OOD, but on a **standalone trajectory predictor** ("accurate trajectory prediction provides vital guidance for downstream planning and control"), with **provable QCD/MMD guarantees on delay and false alarm**. We target a **shipped production end-to-end model (supercombo v0.9.7)** with a parity-verified harness, make **no provable guarantee** (a calibrated empirical LOCO FPR instead), and pair the monitor with a collapse teardown + localization. Different model (predictor vs shipped E2E driver), different evidence basis (theoretical guarantee vs N=1 empirical negative + N=2 calibration). |
| Cheng, Nuhrenberg, Yasuoka, "Runtime Monitoring Neuron Activation Patterns," 2018 | "We propose runtime neuron activation pattern monitoring - after the standard training process, one creates a monitor by feeding the training data to the network again in order to store the neuron activation patterns in abstract form." | arXiv:1809.06573 | Intellectual ancestor of internal-state runtime monitoring. They store **binarized neuron activation patterns** and compare via Hamming distance on **classifiers**. We monitor a **continuous second-order spread of a recurrent feature** on a **temporal driving model**, sensitive to the freeze mode (spread crashes 0.25 to 0.00 across the cliff) that a per-frame activation-pattern comparison would not naturally capture. Different monitored quantity (binary pattern vs covariance trace), different model (feed-forward classifier vs recurrent driver), no temporal/recurrent state in theirs. |
| Lee, K., Lee, K., Lee, H., Shin, "A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks," NeurIPS 2018 | "We obtain the class conditional Gaussian distributions with respect to (low- and upper-level) features of the deep models under Gaussian discriminant analysis, which result in a confidence score based on the Mahalanobis distance." | arXiv:1807.03888 | Canonical location-based feature OOD (the Mahalanobis ancestor). Their score is **distance-from-a-fitted-Gaussian-mean on feed-forward features**, which we run as a baseline: it scores **below chance (AUROC 0.159)** here because the recurrent state collapses TO the ID mean, and hits **100% LOCO FPR** across corpora. Our monitor uses a **location-invariant second-order statistic**, which is precisely what fixes the collapse-to-the-mean and cross-corpus-drift failures their distance-from-mean score exhibits on this recurrent feature. Same lineage, opposite design choice (second-order spread vs first-order distance), and we show theirs fails on the exact substrate we monitor. |
| Sun, Ming, Zhu, Li, "Out-of-Distribution Detection with Deep Nearest Neighbors," ICML 2022 | "non-parametric nearest-neighbor distance for OOD detection" that "does not impose any distributional assumption" | arXiv:2204.06507 | The strongest applicable baseline: KNN-50 **ties E6 at AUROC 1.000** at alpha=1.0, so we do NOT claim to beat it on raw separation (contract exclusion line 108-111). The delta is **transfer/calibration**: KNN is still an **absolute-position score in feature space**, so it hits **100% LOCO FPR** (the subaru and ram corpora sit in disjoint feature regions whose separation dwarfs within-corpus radius), whereas our spread monitor calibrates LOCO to ~1%. Different geometry used (absolute nearest-neighbor distance vs second-order spread), different evaluation axis on which the difference shows (cross-corpus calibration, not single-corpus AUROC). |

Bounded novelty (the union of the deltas above, stated as the set true here and false in
every named neighbor): a **zero-retraining, location-invariant second-order (rolling-spread)
monitor on the recurrent state of a parity-verified SHIPPED end-to-end driving model**,
introduced alongside a **silent-collapse teardown (output collapse + frozen state +
non-responsive uncertainty) localized downstream of the encoder**, and evaluated under
**leave-one-corpus-out FPR where the named location-based scores (Keser-style density,
Lee Mahalanobis, RMD, Sun KNN) fail to transfer**. No named neighbor has all of these:
Keser and Lee and Sun are location-based and not on a recurrent shipped-driver state; Guo
and Su is a guaranteed monitor on a standalone predictor; Cheng is binarized patterns on a
classifier. Do NOT write "we are not aware of prior work"; the novelty is this bounded set.

---

## Framing risk note

1. **The strongest honest framing fits inside the contract; no re-lock is needed.** The
   hook leans on the silent-failure phenomenon (E1/E2/E3), the localization (E5, stated as
   partial), the early-warning monitor (E6), and the transfer-vs-baselines result, all of
   which are inside the "IS allowed to claim" list. I did not need any sentence on the
   exclusion list to make it sharp. The gap is sharpened via the *mechanism* (uncertainty
   channel stays quiet; location-based scores cannot detect collapse-to-the-mean or survive
   cross-corpus drift), not via inflating the number.

2. **Three guardrails the drafter must not relax when turning this skeleton into prose:**
   - Never write "outperforms/beats baselines." KNN-50 ties E6 at AUROC 1.000. The honest
     claim is **transfer/calibration** (100% LOCO FPR for the location-based scores vs ~1%
     for E6), per contract exclusion line 108-111. The Sun 2022 row above is written to
     enforce this.
   - Always carry the N=2 / two-fold qualifier with the 1% FPR, and "a production driving
     model"/named-version (not the plural). The abstract skeleton sentence 6 and the
     parenthetical after it encode this.
   - The E7 bound must ship in the abstract's last sentence, not be buried. Omitting it
     would let a reader infer "general OOD detector," which the contract forbids (exclusion
     lines 99-105). Skeleton sentence 6 carries it.

3. **One framing tension worth flagging to the orchestrator (not a re-lock request).** The
   single sharpest sentence a reviewer remembers would be "the uncertainty head a safety
   monitor trusts is exactly the one that stays silent." That is fully supported by E3 (0 of
   220 OOD frames above real p95) and is inside the boundary, BUT it sits one inferential
   step from the forbidden causal claim about field phantom-braking incidents (exclusion
   lines 88-89, 118-119). Keep the phantom-braking issue as *motivation only* in the intro
   and never let the E3 sentence acquire a "which is why openpilot brakes for shadows"
   clause. This is a drafting risk, not a contract gap.

4. **No neighbor row is `[UNVERIFIED]`.** All five quotes were fetched this run (WebFetch on
   the arXiv abstract pages) with real arXiv ids. The novelty does not rest on any
   un-fetched paper. The Guo and Su id (2603.14603) is a forward-style arXiv identifier;
   I confirmed the title and the trajectory-prediction target in a second fetch, so the
   delta ("standalone predictor, provable guarantee" vs "shipped E2E, empirical") is
   checkable, but the drafter should re-pin the venue/DOI before camera-ready since a
   2026-numbered preprint may not yet have a final venue.
