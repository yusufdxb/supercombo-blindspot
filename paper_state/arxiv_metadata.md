# arXiv submission metadata

**Title:** Silent Collapse: A Distribution-Shift Teardown of a Production Driving Model and a Zero-Retraining Recurrent-State Monitor

**Author:** Yusuf Guenena (Wayne State University)

**Primary category:** cs.LG
**Cross-list:** cs.CV, cs.RO

**License:** arXiv non-exclusive (default); CC-BY 4.0 optional.

**Code:** https://github.com/yusufdxb/supercombo-blindspot

---

## Metadata abstract (1592 chars, under the arXiv 1920 hard limit)

Production Level-2 driving stacks are validated largely in simulation, which assumes the shipped model behaves on rendered input as on real input, or fails loudly otherwise. We test this on openpilot v0.9.7's supercombo, the model that drives comma hardware on public roads. We build a parity-exact reimplementation of its inference, matched to comma's reference longitudinal output within 0.5 m/s^2 on 100% of 1159 real frames (median 0.04). Run on CARLA-rendered clean roads at matched camera intrinsics, 8 of 10 output heads collapse to under 1% of their real temporal activity and the 512-D recurrent state freezes to 1e-5 of its real spread, while the model's predictive-uncertainty heads rise only 1.2 to 1.8x and exceed their real-driving 95th percentile on 0 of 219 frames: the failure is silent. An alpha-blend sweep shows a hard cliff localized downstream of the vision encoder, in the recurrent summarizer and action block. A zero-retraining monitor on the rolling temporal spread of the recurrent state detects the condition (AUROC 0.996) about 0.23 blend-units before the outputs cliff, at a roughly 1% real-driving false-positive rate calibrated leave-one-corpus-out, where location-based feature scores (Mahalanobis, KNN) fail to transfer across real corpora. An ImageNet-C corruption sweep bounds the result: the collapse is sim-specific and the monitor is collapse-specific. Findings are limited to this one model and two real corpora. A simulation pass can be the model collapsed to a safe-looking default, not the model perceiving, and output-side monitors alone do not catch it.
