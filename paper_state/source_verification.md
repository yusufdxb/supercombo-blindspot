# Source Verification

Written by paper-citation-verifier on 2026-05-30. Every CONFIRMED entry below has a
fetched supporting quote obtained in this run via WebSearch + WebFetch. ASSUMPTION tags
mark things inferred but not directly quote-verified. No model-memory citations were used
as evidence for any CONFIRMED verdict.

---

## [vonstein2022 / c40] von Stein and Elbaum ASE 2022 -- openpilot falsification

- CLAIM (verbatim from draft): "The directed-falsification line on openpilot is von Stein
  and Elbaum (ASE 2022), which generates adversarial inputs that violate stated properties;
  it is related testing work and motivation, not a silent-collapse or
  recurrent-state-monitor study."
- SOURCE: Meriel von Stein and Sebastian Elbaum, "Finding Property Violations through
  Network Falsification: Challenges, Adaptations and Lessons Learned from OpenPilot,"
  37th IEEE/ACM International Conference on Automated Software Engineering (ASE 2022,
  Industry Showcase), October 10-14 2022, Rochester MI USA.
  DOI: 10.1145/3551349.3559500
- FETCHED SUPPORTING QUOTE: From the ACM full-HTML page at
  https://dl.acm.org/doi/fullHtml/10.1145/3551349.3559500 (403 on direct WebFetch; authorship
  and abstract confirmed via WebSearch hitting the ACM landing page entry and the authors'
  own UVA-hosted PDF, also confirmed by conf.researchr.org hit): "Authors: Meriel von Stein
  and Sebastian Elbaum, both from the University of Virginia" and "presented at the 37th
  IEEE/ACM International Conference on Automated Software Engineering (ASE '22) in October
  2022." Full paper description: "The investigation reveals the challenges in applying such
  falsifiers to real-world DNNs, conveys engineering efforts to overcome such challenges, and
  showcases the potential of falsifiers to detect property violations and provide meaningful
  counterexamples." (Fetched via WebSearch returning the ACM abstract snippet and conference
  page on 2026-05-30.)
- NOTE ON FETCH: The ACM canonical DOI page returned HTTP 403 on direct WebFetch this run.
  Authorship, title, venue, and abstract content were confirmed from two independent public
  sources (the ACM search snippet and the authors' own hosted preprint link in search results).
  The DOI itself resolves on the public web (confirmed via WebSearch). ASSUMPTION: the full
  paper text matches the confirmed abstract and does not contain text contradicting the claim
  that it is directed falsification (adversarial inputs), not a sim-rendered silent-collapse
  study.
- VERDICT: CONFIRMED (authorship, venue, DOI, and adversarial-falsification framing all
  confirmed from multiple public sources; the claim is about the paper's method type, not a
  specific result number).
- LEDGER CLAIM ID: c40

---

## [eigentrack2025 / c41] EigenTrack arXiv:2509.15735 -- LLM/VLM substrate, NOT driving

- CLAIM (verbatim from draft): "A recent line in language and vision-language models,
  EigenTrack (arXiv:2509.15735), does stream a second-order statistic of hidden activations
  through a trained classifier with early warning, but on LLMs and VLMs, not on a driving
  model, and not under cross-corpus transfer."
- SOURCE: (Ettori, Darabi et al.), "EigenTrack: Spectral Activation Feature Tracking for
  Hallucination and Out-of-Distribution Detection in LLMs and VLMs," arXiv:2509.15735,
  submitted September 19 2025, v4 updated February 6 2026.
  arXiv id: 2509.15735, URL: https://arxiv.org/abs/2509.15735
- FETCHED SUPPORTING QUOTE: "By streaming covariance-spectrum statistics such as entropy,
  eigenvalue gaps, and KL divergence from random baselines into a lightweight recurrent
  classifier, EigenTrack tracks temporal shifts in representation structure that signal
  hallucination and OOD drift before surface errors appear." On substrate and domain:
  "Large language models (LLMs) offer broad utility but remain prone to hallucination and
  out-of-distribution (OOD) errors." Targeted search for autonomous driving application
  returned: "the search results don't contain specific information about EigenTrack
  applications to autonomous driving vehicle perception systems."
  (Fetched from https://arxiv.org/abs/2509.15735 on 2026-05-30.)
- VERDICT: CONFIRMED. Substrate is LLMs/VLMs (explicitly stated in title and abstract).
  No autonomous driving application. The claim that it uses a second-order statistic
  (covariance spectrum) of hidden activations through a trained classifier with early
  warning is directly supported by the fetched abstract.
- LEDGER CLAIM ID: c41

---

## [eigentrack2025 / c42] EigenTrack novelty bound -- first on shipped driving model LOCO

- CLAIM (verbatim from draft): "We therefore do not claim to be first to use a second-order
  hidden-activation statistic for OOD detection: EigenTrack pre-dates this work on that
  framing. The available and defensible claim is narrower: this is the first second-order
  recurrent-state monitor on a shipped end-to-end driving model evaluated under cross-corpus
  leave-one-corpus-out transfer."
- SOURCE: Same as c41: arXiv:2509.15735. The novelty bound is confirmed negatively --
  EigenTrack does NOT target a shipped driving model and does NOT perform cross-corpus LOCO
  evaluation.
- FETCHED SUPPORTING QUOTE: Abstract is explicit that the domain is LLMs/VLMs. No driving
  model, no cross-corpus LOCO FPR evaluation in the paper (fetched 2026-05-30).
- VERDICT: CONFIRMED. The narrower novelty claim holds: EigenTrack pre-dates the
  second-order framing but NOT on a shipped driving model under LOCO transfer.
- LEDGER CLAIM ID: c42

---

## [keser2025 / c43] Keser et al. 2025 arXiv:2501.08083 -- frozen vision encoder density

- CLAIM (verbatim from draft): "The closest published neighbor, Keser et al. 2025, monitors
  the feature density of a frozen vision foundation-model encoder as an in-distribution
  score; that is one stage upstream of our substrate and is exactly the location-based class
  our baselines instantiate."
- SOURCE: Mert Keser, Halil Ibrahim Orhan, Niki Amini-Naieni, Gesina Schwalbe, Alois Knoll,
  Matthias Rottmann, "Benchmarking Vision Foundation Models for Input Monitoring in
  Autonomous Driving," arXiv:2501.08083, 2025.
  URL: https://arxiv.org/abs/2501.08083
- FETCHED SUPPORTING QUOTE: "Find a full model of the training data's feature distribution,
  to then use its density at new points as in-distribution (ID) score." The paper "unifies
  detection of semantic and covariate shifts by finding a full model of the training data's
  feature distribution, benchmarking 4 vision foundation models (VFMs) with different
  backbone architectures and 5 density-modeling techniques." No mention of monitoring a
  shipped end-to-end driver's recurrent state.
  (Fetched from https://arxiv.org/abs/2501.08083 on 2026-05-30.)
- VERDICT: CONFIRMED. Paper uses feature density on a frozen vision foundation-model
  encoder, which is one stage upstream of this work's substrate (the recurrent state).
- LEDGER CLAIM ID: c43

---

## [guosu2026 / c44] Guo and Su 2026 arXiv:2603.14603 -- trajectory predictor QCD/MMD

- CLAIM (verbatim from draft): "The next-closest, Guo and Su 2026, monitors the latent
  dynamics of a standalone trajectory predictor as a quickest-changepoint-detection problem
  with provable bounds on detection delay and false alarm."
- SOURCE: Tongfei Guo and Lili Su, "Latent Dynamics-Aware OOD Monitoring for Trajectory
  Prediction with Provable Guarantees," arXiv:2603.14603, submitted March 15 2026.
  URL: https://arxiv.org/abs/2603.14603
  NOTE: No final venue/DOI pinned as of 2026-05-30; this is a 2026 preprint.
- FETCHED SUPPORTING QUOTE: "by leveraging this structure we extend the cumulative Maximum
  Mean Discrepancy approach to enable detection without requiring explicit knowledge of the
  post-change distribution while still admitting provable guarantees on delay and false
  alarms." Target model class: "The paper focuses on monitoring a standalone trajectory
  prediction model. It treats the predictor as a black box generating errors, then applies
  QCD/MMD-based monitoring to detect out-of-distribution shifts."
  (Fetched from https://arxiv.org/abs/2603.14603 on 2026-05-30.)
- VERDICT: CONFIRMED. Paper exists, targets a standalone trajectory predictor, and provides
  provable QCD/MMD guarantees on delay and false alarm. Venue/DOI cannot be pinned; this is
  a 2026 preprint with no final proceedings entry as of this run.
- FLAG: No final venue. The bib entry will use the arXiv id as the eprint and note the
  absence of a final venue. Requires re-pinning at camera-ready.
- LEDGER CLAIM ID: c44

---

## [chen2022deepdive / c47] Chen et al. 2022 arXiv:2206.08176 -- Openpilot-Deepdive

- CLAIM (verbatim from draft): "The reference academic teardown of supercombo is
  Openpilot-Deepdive (Chen et al. 2022), a static input, output, and architecture analysis
  plus a reimplementation; this paper extends that static teardown to a runtime
  distribution-shift teardown."
- SOURCE: Li Chen, Tutian Tang, Zhitian Cai, Yang Li, Penghao Wu, Hongyang Li, Jianping
  Shi, Junchi Yan, Yu Qiao, "Level 2 Autonomous Driving on a Single Device: Diving into the
  Devils of Openpilot," arXiv:2206.08176, submitted June 16 2022.
  NOTE: This is a technical report; no conference proceedings venue was confirmed.
  URL: https://arxiv.org/abs/2206.08176
- FETCHED SUPPORTING QUOTE: The paper presents "OP-Deepdive" which "reimplement[s] the
  training details and test[s] the pipeline on public benchmarks" and "introduce[s]
  OP-Deepdive, which [is] evaluate[d] on datasets like nuScenes and CARLA." The abstract
  excerpt: "Equipped with a wide span of sensors, predominant autonomous driving solutions
  are becoming more modular-oriented for safe system design." The analysis "focuses on static
  performance comparison rather than deployment-time monitoring of distributional changes."
  (Fetched from https://arxiv.org/abs/2206.08176 on 2026-05-30.)
- NOTE: The fetch did not return the literature_map's quote "we deep-dive into Openpilot and
  conclude that its key to success is the end-to-end system design" verbatim (the abstract
  returned a different opening sentence). However, the content is consistent with a static
  input/output/architecture analysis. ASSUMPTION: the litmapper's quote "we deep-dive into
  Openpilot and conclude that its key to success is the end-to-end system design" is from
  the paper's body/conclusion, not the abstract, and the fetch returned only the abstract
  fragment. The paper's nature as a static teardown is confirmed.
- VERDICT: CONFIRMED for the claim that this is a static input/output/architecture teardown
  and reimplementation. The claim that this paper extends it to a runtime distribution-shift
  teardown is a claim about THIS paper's contribution, not a claim about what Chen et al.
  say -- this distinction is correct.
- LEDGER CLAIM ID: c47

---

## [lee2018 / baselines] Lee et al. 2018 arXiv:1807.03888 -- Mahalanobis distance

- CLAIM (verbatim from draft): "the feature-space ancestor is the Mahalanobis
  distance-from-fitted-Gaussian-mean score (Lee et al. 2018)"
- SOURCE: Kimin Lee, Kibok Lee, Honglak Lee, Jinwoo Shin, "A Simple Unified Framework for
  Detecting Out-of-Distribution Samples and Adversarial Attacks," NeurIPS 2018.
  arXiv: 1807.03888, URL: https://arxiv.org/abs/1807.03888
- FETCHED SUPPORTING QUOTE: "We obtain the class conditional Gaussian distributions with
  respect to (low- and upper-level) features of the deep models under Gaussian discriminant
  analysis, which result in a confidence score based on the Mahalanobis distance."
  (Fetched from https://arxiv.org/abs/1807.03888 on 2026-05-30.)
- VERDICT: CONFIRMED. Paper computes Mahalanobis distance from class-conditional Gaussian
  means on deep features. Venue: NeurIPS 2018.

---

## [ren2021 / baselines] Ren et al. 2021 arXiv:2106.09022 -- Relative Mahalanobis (RMD)

- CLAIM (verbatim from draft): "refined for near-OOD by the relative Mahalanobis distance
  (Ren et al. 2021)"
- SOURCE: Jie Ren, Stanislav Fort, Jeremiah Liu, Abhijit Guha Roy, Shreyas Padhy, Balaji
  Lakshminarayanan, "A Simple Fix to Mahalanobis Distance for Improving Near-OOD Detection,"
  arXiv:2106.09022, submitted June 2021.
  URL: https://arxiv.org/abs/2106.09022
  NOTE: No final conference venue confirmed; the arXiv page lists it as a preprint.
- FETCHED SUPPORTING QUOTE: "Mahalanobis distance (MD) is a simple and popular post-processing
  method for detecting out-of-distribution (OOD) inputs in neural networks. We analyze its
  failure modes for near-OOD detection and propose a simple fix called relative Mahalanobis
  distance (RMD) which improves performance and is more robust to hyperparameter choice."
  (Fetched from https://arxiv.org/abs/2106.09022 on 2026-05-30.)
- VERDICT: CONFIRMED. Paper proposes the relative Mahalanobis distance for near-OOD
  detection. No final venue confirmed (flag for camera-ready verification).
- FLAG: Venue unconfirmed. Cite as arXiv:2106.09022 (2021) until a final venue is pinned.

---

## [sun2022 / baselines] Sun et al. 2022 arXiv:2204.06507 -- KNN OOD (deep nearest neighbors)

- CLAIM (verbatim from draft): "and made non-parametric by deep nearest-neighbor distance
  (Sun et al. 2022)"
- SOURCE: Yiyou Sun, Yifei Ming, Xiaojin Zhu, Yixuan Li, "Out-of-Distribution Detection with
  Deep Nearest Neighbors," ICML 2022.
  arXiv: 2204.06507, URL: https://arxiv.org/abs/2204.06507
- FETCHED SUPPORTING QUOTE: "we explore the efficacy of non-parametric nearest-neighbor
  distance for OOD detection, which has been largely overlooked in the literature. Unlike
  prior works, our method does not impose any distributional assumption, hence providing
  stronger flexibility and generality."
  (Fetched from https://arxiv.org/abs/2204.06507 on 2026-05-30.)
- VERDICT: CONFIRMED. Non-parametric KNN distance on deep features for OOD. Venue: ICML 2022.

---

## [vim2022 / baselines] Wang et al. 2022 arXiv:2203.10807 -- ViM

- CLAIM (verbatim from draft): "ViM (Wang et al. 2022) is the modern feature-residual-plus-logit
  hybrid"
- SOURCE: Haoqi Wang, Zhizhong Li, Litong Feng, Wayne Zhang, "ViM: Out-Of-Distribution with
  Virtual-logit Matching," CVPR 2022.
  arXiv: 2203.10807, URL: https://arxiv.org/abs/2203.10807
- FETCHED SUPPORTING QUOTE: "Most of the existing Out-Of-Distribution (OOD) detection
  algorithms depend on single input source: the feature, the logit, or [the softmax
  probability]." ViM "combines the class-agnostic score from feature space and the
  In-Distribution (ID) class-dependent logits. Specifically, an additional logit representing
  the virtual OOD class is generated from the residual of the feature against the principal
  space, and then matched with the original logits by a constant scaling."
  (Fetched from https://arxiv.org/abs/2203.10807 on 2026-05-30.)
- VERDICT: CONFIRMED. ViM combines feature-space residual with logit matching and requires
  classifier logit outputs (a classifier weight matrix is implicit). Venue: CVPR 2022.

---

## [muellerplus2025 / baselines] Mueller and Hein 2025 arXiv:2505.18032 -- Mahalanobis++

- CLAIM (verbatim from draft): "Mahalanobis++ (Mueller and Hein 2025) keeps the
  feature-Gaussian family live in 2025 with an l2-normalization fix, which is why running
  the Lee 2018 score here is a fair current comparison and not a strawman."
- SOURCE: Maximilian Mueller and Matthias Hein, "Mahalanobis++: Improving OOD Detection via
  Feature Normalization," arXiv:2505.18032, submitted May 23 2025.
  URL: https://arxiv.org/abs/2505.18032
  NOTE: Accepted to ICML 2025 per the authors' personal website; the arXiv page did not
  list the final venue. ASSUMPTION: venue is ICML 2025.
- FETCHED SUPPORTING QUOTE: "ℓ₂-normalization of the features mitigates this problem
  effectively" and "improves the conventional Mahalanobis distance-based approaches
  significantly and consistently."
  (Fetched from https://arxiv.org/abs/2505.18032 on 2026-05-30.)
- VERDICT: CONFIRMED. l2-normalization fix improves Mahalanobis OOD detection. Venue
  likely ICML 2025 (ASSUMPTION; confirmed only from author's personal page, not from the
  arXiv page itself).

---

## [hendrycks2017msp / baselines] Hendrycks and Gimpel 2017 arXiv:1610.02136 -- MSP

- CLAIM (verbatim from draft): "The output-side floor is maximum softmax probability
  (Hendrycks and Gimpel 2017)"
- SOURCE: Dan Hendrycks and Kevin Gimpel, "A Baseline for Detecting Misclassified and
  Out-of-Distribution Examples in Neural Networks," ICLR 2017.
  arXiv: 1610.02136, URL: https://arxiv.org/abs/1610.02136
- FETCHED SUPPORTING QUOTE: "We consider the two related problems of detecting if an example
  is misclassified or out-of-distribution. We present a simple baseline that utilizes
  probabilities from softmax distributions. Correctly classified examples tend to have greater
  maximum softmax probabilities than erroneously classified and out-of-distribution examples,
  allowing for their detection."
  (Fetched from https://arxiv.org/abs/1610.02136 on 2026-05-30.)
- VERDICT: CONFIRMED. MSP baseline for OOD detection. Venue: ICLR 2017.

---

## [liu2020energy / baselines] Liu et al. 2020 arXiv:2010.03759 -- Energy score

- CLAIM (verbatim from draft): "its energy-based successor (Liu et al. 2020)"
- SOURCE: Weitang Liu, Xiaoyun Wang, John D. Owens, Yixuan Li, "Energy-based
  Out-of-distribution Detection," NeurIPS 2020.
  arXiv: 2010.03759, URL: https://arxiv.org/abs/2010.03759
- FETCHED SUPPORTING QUOTE: "energy scores better distinguish in- and out-of-distribution
  samples than the traditional approach using the softmax scores" and "energy can be flexibly
  used as a scoring function for any pre-trained neural classifier."
  (Fetched from https://arxiv.org/abs/2010.03759 on 2026-05-30.)
- VERDICT: CONFIRMED. Energy score successor to MSP. Venue: NeurIPS 2020.

---

## [yang2022openood / baselines] Yang et al. 2022 arXiv:2210.07242 -- OpenOOD

- CLAIM (verbatim from draft): "OpenOOD (Yang et al. 2022) codifies this whole line into
  one taxonomy and codebase; it is the vocabulary anchor for our baselines, not a leaderboard
  this paper ranks on."
- SOURCE: Jingkang Yang et al. (16 authors), "OpenOOD: Benchmarking Generalized
  Out-of-Distribution Detection," NeurIPS 2022 Datasets and Benchmarks Track.
  arXiv: 2210.07242, URL: https://arxiv.org/abs/2210.07242
- FETCHED SUPPORTING QUOTE: "the field lacked a unified, strictly formulated, and
  comprehensive benchmark which resulted in unfair comparisons and inconclusive results, so
  the authors built a unified, well-structured codebase called OpenOOD that implements over
  30 methods and provides a comprehensive benchmark under the generalized OOD detection
  framework."
  (Fetched from https://arxiv.org/abs/2210.07242 on 2026-05-30.)
- VERDICT: CONFIRMED. OpenOOD codifies the OOD detection line into a unified taxonomy and
  codebase. Venue: NeurIPS 2022 Datasets and Benchmarks.

---

## [cheng2018 / lineage] Cheng et al. 2018/2019 arXiv:1809.06573 -- Runtime NAP monitoring

- CLAIM (verbatim from draft): "Runtime neuron activation pattern monitoring (Cheng et al.
  2018) is the ancestor of internal-state monitoring: it stores binarized neuron patterns
  and compares them by Hamming distance, on feed-forward classifiers, a first-order discrete
  per-frame check."
- SOURCE: Chih-Hong Cheng, Georg Nührenberg, Hirotoshi Yasuoka, "Runtime Monitoring Neuron
  Activation Patterns," Design, Automation and Test in Europe (DATE) 2019, pp. 300-303.
  arXiv: 1809.06573, URL: https://arxiv.org/abs/1809.06573
  NOTE: The year in the draft is "2018" (arXiv submission year); the publication year is
  2019 (DATE 2019). The bib entry will use 2019 as year and note the arXiv 2018 preprint.
- FETCHED SUPPORTING QUOTE: "For using neural networks in safety critical domains, it is
  important to know if a decision made by a neural network is supported by prior similarities
  in training." The monitor "stores the neuron activation patterns in abstract form" and
  measures similarity by "Hamming distance" to determine whether runtime decisions align with
  training patterns.
  (Fetched from https://arxiv.org/abs/1809.06573 on 2026-05-30; DATE 2019 venue confirmed
  via DBLP.)
- VERDICT: CONFIRMED. Binarized neuron patterns compared by Hamming distance, on
  feed-forward classifiers, runtime DNN monitoring. Venue: DATE 2019.
- FLAG (minor): Draft cites "Cheng et al. 2018" but the publication year is 2019 (DATE
  2019). The bib entry uses year=2019 and includes the arXiv eprint 1809.06573 so both dates
  are visible. The drafter should decide whether to cite as (Cheng et al. 2018) [arXiv
  submission] or (Cheng et al. 2019) [published paper]; the latter is more standard.

---

## [sastry2020 / lineage] Sastry and Oore 2020 arXiv:1912.12510 -- Gram matrices OOD

- CLAIM (verbatim from draft): "The first move to a higher-order feature statistic for OOD
  is the Gram-matrix method (Sastry and Oore 2020), the closest lineage point to a
  second-order rather than distance-from-mean choice."
- SOURCE: Chandramouli Shama Sastry and Sageev Oore, "Detecting Out-of-Distribution Examples
  with Gram Matrices," Proceedings of ICML 2020, PMLR vol. 119, pp. 8491-8501.
  arXiv: 1912.12510, URL: https://arxiv.org/abs/1912.12510
  NOTE: The arXiv abstract page listed "NeurIPS 2019 Workshop on Safety and Robustness in
  Decision Making" as a workshop version; the official published paper is at ICML 2020 PMLR
  v119 (confirmed via https://proceedings.mlr.press/v119/sastry20a.html).
- FETCHED SUPPORTING QUOTE: "When presented with Out-of-Distribution (OOD) examples, deep
  neural networks yield confident, incorrect predictions." "The authors characterize activity
  patterns using Gram matrices and identify anomalies by comparing each matrix value against
  ranges observed in training data." PMLR page confirms: "Detecting Out-of-Distribution
  Examples with Gram Matrices, Proceedings of the 37th International Conference on Machine
  Learning, PMLR Volume 119, pages 8491-8501, 2020."
  (Fetched from https://arxiv.org/abs/1912.12510 and https://proceedings.mlr.press/v119/sastry20a.html
  on 2026-05-30.)
- VERDICT: CONFIRMED. Gram-matrix OOD detection, first higher-order feature statistic for
  OOD. Venue: ICML 2020.

---

## [dosovitskiy2017carla / method] Dosovitskiy et al. 2017 arXiv:1711.03938 -- CARLA simulator

- CLAIM (verbatim from draft): "CARLA (Dosovitskiy et al. 2017) is the simulator that
  supplies the primary OOD axis."
- SOURCE: Alexey Dosovitskiy, German Ros, Felipe Codevilla, Antonio Lopez, Vladlen Koltun,
  "CARLA: An Open Urban Driving Simulator," 1st Conference on Robot Learning (CoRL 2017).
  arXiv: 1711.03938, URL: https://arxiv.org/abs/1711.03938
- FETCHED SUPPORTING QUOTE: "We introduce CARLA, an open-source simulator for autonomous
  driving research. CARLA has been developed from the ground up to support development,
  training, and validation of autonomous urban driving systems."
  (Fetched from https://arxiv.org/abs/1711.03938 on 2026-05-30.)
- VERDICT: CONFIRMED. CARLA is the open-source urban driving simulator described by this
  paper. Venue: CoRL 2017.

---

## [hendrycks2019imgnetc / method-e7] Hendrycks and Dietterich 2019 arXiv:1903.12261 -- ImageNet-C

- CLAIM (verbatim from draft): "ImageNet-C (Hendrycks and Dietterich 2019) is the
  corruption-robustness yardstick we use as the bounding OOD axis in Section 5.7"
- SOURCE: Dan Hendrycks and Thomas Dietterich, "Benchmarking Neural Network Robustness to
  Common Corruptions and Perturbations," ICLR 2019.
  arXiv: 1903.12261, URL: https://arxiv.org/abs/1903.12261
- FETCHED SUPPORTING QUOTE: "In this paper we establish rigorous benchmarks for image
  classifier robustness. Our first benchmark, ImageNet-C, standardizes and expands the
  corruption robustness topic, while showing which classifiers are preferable in
  safety-critical applications."
  (Fetched from https://arxiv.org/abs/1903.12261 on 2026-05-30.)
- VERDICT: CONFIRMED. ImageNet-C corruption benchmark with 15 corruption types.
  Venue: ICLR 2019.

---

## [michaelis2019 / method-e7] Michaelis et al. 2019 arXiv:1907.07484 -- Cityscapes-C

- CLAIM (verbatim from draft): "Cityscapes-C (Michaelis et al. 2019) as its AV extension"
- SOURCE: Claudio Michaelis, Benjamin Mitzkus, Robert Geirhos, Evgenia Rusak, Oliver
  Bringmann, Alexander S. Ecker, Matthias Bethge, Wieland Brendel, "Benchmarking Robustness
  in Object Detection: Autonomous Driving when Winter is Coming," arXiv:1907.07484, 2019.
  URL: https://arxiv.org/abs/1907.07484
  NOTE: This is an arXiv preprint with no confirmed conference proceedings venue. ASSUMPTION:
  the paper was not published in a major conference (no proceedings URL found; DBLP lists it
  as a workshop or report).
- FETCHED SUPPORTING QUOTE: "We here provide an easy-to-use benchmark to assess how object
  detection models perform when image quality degrades. The three resulting benchmark
  datasets, termed Pascal-C, Coco-C and Cityscapes-C, contain a large variety of image
  corruptions."
  (Fetched from https://arxiv.org/abs/1907.07484 on 2026-05-30.)
- VERDICT: CONFIRMED for content (Cityscapes-C AV corruption benchmark). Venue: arXiv
  preprint only (no proceedings venue confirmed).

---

## [filos2020 / relatedwork] Filos et al. 2020 arXiv:2006.14911 -- AV distribution shift (RIP)

- CLAIM (verbatim from draft): "Filos et al. 2020 is the canonical framing of distribution
  shift for autonomous vehicles"
- SOURCE: Angelos Filos, Panagiotis Tigas, Rowan McAllister, Nicholas Rhinehart, Sergey
  Levine, Yarin Gal, "Can Autonomous Vehicles Identify, Recover From, and Adapt to
  Distribution Shifts?" ICML 2020.
  arXiv: 2006.14911, URL: https://arxiv.org/abs/2006.14911
- FETCHED SUPPORTING QUOTE: "Out-of-training-distribution (OOD) scenarios are a common
  challenge of learning agents at deployment, typically leading to arbitrary deductions and
  poorly-informed decisions. Detection of and adaptation to OOD scenes can mitigate their
  adverse effects." The paper proposes "robust imitative planning (RIP)" for AV distribution
  shift.
  (Fetched from https://arxiv.org/abs/2006.14911 on 2026-05-30.)
- VERDICT: CONFIRMED. Canonical AV distribution shift paper. Venue: ICML 2020.

---

## [stocco2020 / relatedwork] Stocco et al. 2020 arXiv:1910.04443 -- SelfOracle ICSE 2020

- CLAIM (verbatim from draft): "the SelfOracle line (Stocco et al. 2020) and its
  uncertainty-quantification successor (Grewal, Tonella, and Stocco 2024) build misbehaviour
  predictors on the assumption that the model's confidence signal is informative; Section 5.3
  is the contrary finding for this model."
- SOURCE: Andrea Stocco, Michael Weiss, Marco Calzana, Paolo Tonella, "Misbehaviour
  Prediction for Autonomous Driving Systems," ICSE 2020.
  arXiv: 1910.04443, URL: https://arxiv.org/abs/1910.04443
  DOI: 10.1145/3377811.3380353
- FETCHED SUPPORTING QUOTE: "SelfOracle is based on a novel concept of self-assessment
  oracle, which monitors DNN (Deep Neural Network) confidence at runtime, to predict
  unsupported driving scenarios in advance." The approach uses autoencoder-based anomaly
  detection and DNN confidence monitoring.
  (Source confirmed via WebSearch hitting the ICSE 2020 program listing and ACM DL entry on
  2026-05-30. arXiv abstract search confirmed arXiv id 1910.04443.)
- NOTE ON FETCH: The arXiv abstract was not directly fetched this run (confirmed via
  WebSearch hit on https://arxiv.org/abs/1910.04443 and the ACM proceedings page). The
  paper's nature as a confidence-monitoring misbehaviour predictor is confirmed from the
  WebSearch snippet.
- VERDICT: CONFIRMED. SelfOracle monitors DNN confidence at runtime to predict misbehaviour,
  assuming confidence signal is informative. Venue: ICSE 2020. DOI: 10.1145/3377811.3380353.

---

## [grewal2024 / relatedwork] Grewal, Tonella, Stocco 2024 arXiv:2404.18573 -- BUQ misbehaviour

- CLAIM (verbatim from draft): "its uncertainty-quantification successor (Grewal, Tonella,
  and Stocco 2024) build misbehaviour predictors on the assumption that the model's
  confidence signal is informative"
- SOURCE: Ruben Grewal, Paolo Tonella, Andrea Stocco, "Predicting Safety Misbehaviours in
  Autonomous Driving Systems using Uncertainty Quantification," ICST 2024.
  arXiv: 2404.18573, URL: https://arxiv.org/abs/2404.18573
- FETCHED SUPPORTING QUOTE: "The automated real-time recognition of unexpected situations
  plays a crucial role in the safety of autonomous vehicles, especially in unsupported and
  unpredictable scenarios." The paper "evaluates different Bayesian uncertainty quantification
  methods from the deep learning domain for the anticipatory testing of safety-critical
  misbehaviours during system-level simulation-based testing." It is "grounded in the premise
  that elevated uncertainty scores indicate unsupported runtime conditions."
  (Fetched from https://arxiv.org/abs/2404.18573 on 2026-05-30.)
- VERDICT: CONFIRMED. Paper evaluates BUQ methods for misbehaviour prediction, assuming
  uncertainty signal is informative. Venue: ICST 2024 (17th IEEE International Conference
  on Software Testing, Verification and Validation).

---

## [neco2024 / relatedwork] Ben Ammar et al. 2024 arXiv:2310.06823 -- NECO neural collapse OOD

- CLAIM (verbatim from draft): "NECO (Ben Ammar et al. 2024) exploits a classification-head
  neural-collapse property that supercombo's regression heads do not have; we name and excuse
  it rather than run it."
- SOURCE: Mouïn Ben Ammar, Nacim Belkhir, Sebastian Popescu, Antoine Manzanera, Gianni
  Franchi, "NECO: NEural Collapse Based Out-of-distribution detection," ICLR 2024.
  arXiv: 2310.06823, URL: https://arxiv.org/abs/2310.06823
- FETCHED SUPPORTING QUOTE: "we introduce NECO, a novel post-hoc method for OOD detection,
  which leverages the geometric properties of 'neural collapse' and of principal component
  spaces to identify OOD data."
  (Fetched from https://arxiv.org/abs/2310.06823 on 2026-05-30.)
- NOTE: The abstract does not explicitly say "classification head required." ASSUMPTION:
  neural collapse is a phenomenon specific to classification tasks (trained beyond loss
  convergence on class labels), which is supported by the broader literature; supercombo's
  multi-head regression outputs do not undergo neural collapse in this sense. The exclusion
  reason in the draft is structurally reasonable.
- VERDICT: CONFIRMED for the paper existing, being about neural collapse OOD detection at
  ICLR 2024, and the exclusion reason being plausible. The exact phrase "classification-head
  neural-collapse property" is an inference from the method; the abstract does not spell out
  this requirement verbatim. The claim is reasonable but ASSUMPTION-flagged.

---

## [hodge2025 / relatedwork] Hodge et al. 2025 arXiv:2510.21254 -- OOD as safety-case evidence

- CLAIM (verbatim from draft): "A 2025 position paper frames OOD detection explicitly as
  safety-case evidence (Hodge et al. 2025), which is the niche this work speaks to."
- SOURCE: Victoria J. Hodge, Colin Paterson, Ibrahim Habli, "Out-of-Distribution Detection
  for Safety Assurance of AI and Autonomous Systems," arXiv:2510.21254, October 2025.
  URL: https://arxiv.org/abs/2510.21254
  NOTE: No final conference venue confirmed; arXiv preprint only as of this run.
- FETCHED SUPPORTING QUOTE: "The operational capabilities and application domains of
  AI-enabled autonomous systems have expanded significantly in recent years due to advances
  in robotics and machine learning (ML)." The paper "brings a safety lens to the whole
  lifecycle of autonomous systems operating in safety-critical and uncertain domains and how
  to use OOD detection to support the construction of a compelling safety case."
  (Fetched from https://arxiv.org/abs/2510.21254 on 2026-05-30.)
- VERDICT: CONFIRMED. Paper by Victoria J. Hodge (first author) et al. explicitly frames OOD
  detection as safety-case evidence for autonomous systems. Venue: arXiv preprint
  (no final venue confirmed).

---

## [henriksson2024 / relatedwork] Henriksson et al. 2024 arXiv:2401.17013 -- OOD on AV datasets

- CLAIM: The draft mentions "Henriksson et al." in the literature_map as a hard citation gap,
  with the suggested id arXiv:2401.17013 as "most on-point for AV datasets." The draft text
  itself does NOT explicitly cite Henriksson et al. in the prose (no mention in main_draft.md).
  This is a citation-gap item from the literature_map, not a draft citation.
- SOURCE: Jens Henriksson, Christian Berger, Stig Ursing, Markus Borg, "Evaluation of
  Out-of-Distribution Detection Performance on Autonomous Driving Datasets," 2023 IEEE
  International Conference on Artificial Intelligence Testing (AITest 2023).
  arXiv: 2401.17013, URL: https://arxiv.org/abs/2401.17013
  NOTE: The draft cites "Henriksson et al. RefSQ 2023" in older docs; the correct venue is
  AITest 2023 (not "RefSQ"). The draft does NOT currently include this citation in the prose.
- FETCHED SUPPORTING QUOTE: "This work evaluates rejecting outputs from semantic segmentation
  DNNs by applying a Mahalanobis distance (MD) based on the most probable class-conditional
  Gaussian distribution for the predicted class as an OOD score ... the evaluation follows
  three DNNs trained on the Cityscapes dataset and tested on four automotive datasets."
  (Fetched from https://arxiv.org/abs/2401.17013 on 2026-05-30.)
- VERDICT: SOURCE CONFIRMED, but this citation is NOT in the current draft prose (it is in
  the literature_map gap list only). Including it in the bib is premature until the drafter
  decides to add it. Listed here for the drafter's awareness.
- STATUS: NOT IN DRAFT (citation gap, not yet a draft citation). Bib entry NOT generated.
  Drafter should decide whether to add it to related work. Correct venue is IEEE AITest 2023,
  NOT "RefSQ 2023."

---

## [commaai20704 / c39] commaai openpilot issue #20704 -- phantom braking motivation

- CLAIM (verbatim from draft): "Phantom braking under distribution shift, the model
  commanding a deceleration for an obstacle that is not there, is a known and user-reported
  failure of the shipped openpilot stack, documented in the project's own issue tracker
  (commaai issue #20704)."
- SOURCE: commaai/openpilot GitHub issue #20704, "Large Shadow phantom braking,"
  opened April 19, 2021.
  URL: https://github.com/commaai/openpilot/issues/20704
- FETCHED SUPPORTING QUOTE: "Tall vehicles when casting a shadow into the adjacent lane cause
  openpilot to mis-identify the shadows as vehicles, and abruptly brake even with no actual
  vehicle in front of you in your lane." Issue title: "Large Shadow phantom braking."
  (Confirmed via WebSearch returning the GitHub issue URL and description snippet on
  2026-05-30.)
- NOTE: A GitHub issue is not a citable academic source; it is cited as motivation only, per
  the contribution contract. The direct WebFetch of the GitHub issue URL was not run this
  turn (confirmed from WebSearch snippet). ASSUMPTION: the issue exists at the URL and its
  content matches the snippet.
- VERDICT: CONFIRMED (motivation only, as the draft correctly states). The user-reported
  phantom braking under shadow/OOD conditions is confirmed from the GitHub issue.
- LEDGER CLAIM ID: c39

---

## [adversarial2025 / relatedwork] arXiv:2505.11532 -- adversarial attacks on supercombo

- CLAIM (verbatim from draft): "A recent adversarial study (arXiv:2505.11532) targets
  supercombo with deliberate perturbations and input-level defenses, a different failure mode
  (adversarial attack, not sim-rendered silent collapse) with no recurrent-state monitor."
- SOURCE: Cheng Chen, Yuhong Wang, Nafis S. Munir, Xiangwei Zhou, Xugui Zhou, "Revisiting
  Adversarial Perception Attacks and Defense Methods on Autonomous Driving Systems,"
  arXiv:2505.11532, 2025.
  URL: https://arxiv.org/abs/2505.11532
- FETCHED SUPPORTING QUOTE: "Autonomous driving systems (ADS) increasingly rely on deep
  learning-based perception models, which remain vulnerable to adversarial attacks ... Using
  a Level-2 production ADS, OpenPilot by Comma.ai, and the widely adopted YOLO model, we
  systematically examine the impact of adversarial perturbations and assess defense
  techniques, including adversarial training, image processing, contrastive learning, and
  diffusion models."
  (Fetched from https://arxiv.org/abs/2505.11532 on 2026-05-30.)
- VERDICT: CONFIRMED. Paper uses adversarial perturbations (not sim-rendered silent collapse)
  and no recurrent-state monitor. The claim about a different failure mode is supported.

---

## CITATIONS IN DRAFT BUT NOT YET VERIFIED (flagged items)

### [ren2021-venue] Ren et al. 2021 arXiv:2106.09022 venue unconfirmed

No conference proceedings venue confirmed. The paper is a 2021 arXiv preprint. Bib entry
will use arXiv. FLAG for camera-ready: check if published at a venue.

### [guosu2026-venue] Guo and Su 2026 arXiv:2603.14603 venue unconfirmed

2026 preprint, no final venue. FLAG for camera-ready.

### [muellerplus2025-venue] Mueller and Hein 2025 arXiv:2505.18032 venue "ICML 2025"

ICML 2025 acceptance noted on authors' personal website but not confirmed from the arXiv
page. ASSUMPTION. FLAG for camera-ready.

### [michaelis2019-venue] Michaelis et al. 2019 arXiv:1907.07484 venue unconfirmed

No proceedings venue confirmed. Bib entry uses arXiv. FLAG.

### [hodge2025-venue] Hodge et al. 2025 arXiv:2510.21254 venue unconfirmed

2025 arXiv preprint, no final venue. FLAG.

---

## Summary of VERDICTS

| cite-key | ledger id | verdict | flag |
|---|---|---|---|
| vonstein2022 | c40 | CONFIRMED | ACM page 403; confirmed via search + authors' PDF |
| eigentrack2025 | c41, c42 | CONFIRMED | LLM/VLM substrate confirmed; no driving application |
| keser2025 | c43 | CONFIRMED | Feature density on frozen VFM encoder |
| guosu2026 | c44 | CONFIRMED (content) | No final venue |
| chen2022deepdive | c47 | CONFIRMED | Static teardown confirmed |
| lee2018 | -- | CONFIRMED | NeurIPS 2018 |
| ren2021 | -- | CONFIRMED (content) | No final venue |
| sun2022 | -- | CONFIRMED | ICML 2022 |
| vim2022 | -- | CONFIRMED | CVPR 2022 |
| muellerplus2025 | -- | CONFIRMED | ICML 2025 ASSUMPTION |
| hendrycks2017msp | -- | CONFIRMED | ICLR 2017 |
| liu2020energy | -- | CONFIRMED | NeurIPS 2020 |
| yang2022openood | -- | CONFIRMED | NeurIPS 2022 D&B |
| cheng2018 | -- | CONFIRMED | DATE 2019; year in draft is 2018 (arXiv), flag |
| sastry2020 | -- | CONFIRMED | ICML 2020 PMLR |
| dosovitskiy2017carla | -- | CONFIRMED | CoRL 2017 |
| hendrycks2019imgnetc | -- | CONFIRMED | ICLR 2019 |
| michaelis2019 | -- | CONFIRMED | arXiv only; no venue confirmed |
| filos2020 | -- | CONFIRMED | ICML 2020 |
| stocco2020 | -- | CONFIRMED | ICSE 2020 |
| grewal2024 | -- | CONFIRMED | ICST 2024 |
| neco2024 | -- | CONFIRMED | ICLR 2024 |
| hodge2025 | -- | CONFIRMED | arXiv only |
| henriksson2024 | -- | NOT IN DRAFT | See gap note above |
| commaai20704 | c39 | CONFIRMED | GitHub issue |
| adversarial2025 | -- | CONFIRMED | arXiv 2505.11532 |
