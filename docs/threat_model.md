# Threat Model

This document states the runtime failure mode Phantom-Braking targets, the defenses an openpilot-class stack actually deploys against that failure, why each of those defenses misses the mode we observe, and the role of E6 as a complementary monitor.

## 1. Threat

The measured threat is narrower than visual distribution shift in general. On the tested CARLA corpus,
openpilot v0.9.7 supercombo exhibits three conditions simultaneously and without a reliable alert from
the monitored exported uncertainty channels:

- **Output collapse to a constant.** Per E1, on CARLA-clean inputs 8 of 10 supercombo output heads drop to under 1% of their real-driving temporal activity, with CARLA-to-real activity ratios between 0.0018 and 0.0076 across plan, lane_lines, road_edges, lead, lead_prob, desired_curv, accel_t0, and desire_state. The outputs become a frozen, plausible-looking constant.
- **No monitored uncertainty alert.** Per E3, predictive-uncertainty ratios are between 1.20x and 1.84x of real (plan, lead, desired_curv), and 0 of 219 CARLA analysis frames exceed the real-driving p95 of any monitored head.
- **Recurrent state contracts.** Per E4, feature spread (trace of recurrent-state covariance over a window) crashes from 0.25 at alpha=0.0 to approximately zero by alpha=0.78. This establishes temporal contraction, not a causal proof that the model has stopped integrating all information.

The centerpiece is E3: the in-model uncertainty channel that downstream code would use to detect this condition does not move. This is silent failure, not a noisy failure.

## 2. Existing defenses, and why they miss this mode

The following output-side checks miss or may miss the measured failure mode. Only the uncertainty result
is directly tested as a detector; the other bullets are scoped engineering implications.

- **Predictive-uncertainty heads.** Direct E3 evidence: ratios 1.20-1.84x of real, 0% of CARLA frames above real p95. A threshold tuned on real driving will not fire. This is the defense closest to a learned OOD detector inside supercombo, and it is silent.
- **Plan-feasibility and plausibility limits (steering rate, accel bounds, lateral acceleration caps).** Per E1, the plan and accel_t0 outputs collapse toward zero (plan activity 0.0057x real, accel_t0 activity 0.0040x real). A zero-magnitude plan does not violate plausibility limits; it looks like a stationary scene with no required action. Plausibility passes.
- **Output-disagreement / temporal-consistency checks (frame-to-frame jitter, output variance).** Once outputs freeze (E1 activity ratios under 1%), temporal variance drops, not rises. A jitter monitor would interpret the freeze as the system being more stable, not less.
- **Same-architecture ensembles remain untested.** E5 places the contraction downstream of the vision encoder and selected submodule probes implicate `summarizer_div` and `action_block_body`. It is plausible, but not demonstrated, that independently trained replicas would share this behavior.
- **Input-side image quality checks (blur, exposure, sensor noise).** CARLA-clean is sharper, more uniform, and less noisy than real driving footage. Input-quality monitors will rate it as good.

The evidence directly rules out the monitored exported uncertainty thresholds for this corpus. It does
not establish that every possible output-side or system-level safety monitor would fail.

## 3. Our claim (E6)

An internal-feature monitor over the recurrent state detects the tested collapse. Its early-warning
behavior is source-dependent.

- Monitored quantity: rolling temporal spread of the 512-D recurrent feature vector emitted by supercombo (`src/e6_detector.py`).
- Threshold calibration: collapse-unaware leave-one-corpus-out across four real corpora. LOCO mean FPR is 2.41%, segment-bootstrap 95% CI [0%, 5.17%], and worst-fold FPR is 6.90%. The earlier two-corpus estimate of 1.03% was optimistic.
- Detector response on the E4 sweep: fires (>50% of frames flagged) at alpha = 0.550.
- Output cliff on the same sweep (E4): output activity falls from 0.9x to 0.1x of real over alpha 0.784 to 0.799.
- Gap: detector at alpha=0.55, cliff at alpha=0.78 on the Subaru overlay. The detector fires roughly 0.23 alpha units before that source's cliff. On RAM, it fires inside the transition band and has no early-warning headroom.

The claim is not that E6 replaces output-side monitors. It is that the model's own internal features carry an OOD signal that the output heads do not surface, and that an internal-feature monitor is a complementary safety layer: cheap (one forward pass already happening, one O(d) statistic per frame), shipped-model compatible (no retraining, no architectural change), and calibrated against real-driving FPR rather than against simulated negatives.

## 4. Limitations

- **Two model versions.** v0.9.6 also reacts abnormally to CARLA, but through output amplification rather than the v0.9.7 freeze, and the v0.9.7 monitor does not transfer (33% LOCO FPR). No other driving architecture was tested.
- **One synthetic collapse axis.** CARLA-clean is an extreme distribution shift. ImageNet-C corruptions and real night/glare do not reproduce the collapse. The monitor is not a general OOD detector.
- **Limited real validation.** A daytime segment contains an unexplained near-zero recurrent attractor, but no known-cause real output collapse or validated field alert has been demonstrated.
- **Partial localization.** Selected probes place the contraction downstream of the encoder, at the summarizer and action-block feedback path. The summarizer's mu-versus-sigma ambiguity prevents a complete causal account.
- **Four corpora are not fleet scale.** The 2.41% mean LOCO FPR and [0%, 5.17%] segment-bootstrap interval quantify this small corpus set only. A larger, diverse corpus is required before a production false-positive rate can be quoted.

## 5. Safety implication

Simulation validation can provide false confidence when the simulator itself induces unmonitored model
collapse. E1-E4 demonstrate that risk for one released model and CARLA corpus. The result motivates
internal-state instrumentation as a complement to output-side checks, but it does not validate E6 as an
on-road safety mechanism. The measured software cost is one O(d) statistic per inference. The current
calibration evidence is 2.41% mean LOCO FPR across four corpora, with positive warning lead only on the
Subaru overlay and no physical deployment evidence.
