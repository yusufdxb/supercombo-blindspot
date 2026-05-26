# Threat Model

This document states the runtime failure mode Phantom-Braking targets, the defenses an openpilot-class stack actually deploys against that failure, why each of those defenses misses the mode we observe, and the role of E6 as a complementary monitor.

## 1. Threat

A production driving model is deployed in a visual context shifted from its training distribution (rendered simulation, heavy weather, glare, novel geography, sensor degradation). Under sufficient shift, three things happen simultaneously and silently:

- **Output collapse to a constant.** Per E1, on CARLA-clean inputs 8 of 10 supercombo output heads drop to under 1% of their real-driving temporal activity, with CARLA-to-real activity ratios between 0.0018 and 0.0076 across plan, lane_lines, road_edges, lead, lead_prob, desired_curv, accel_t0, and desire_state. The outputs become a frozen, plausible-looking constant.
- **No uncertainty signal.** Per E3, predictive-uncertainty heads ratio between 1.20x and 1.84x of real (plan, lead, desired_curv), and 0% of CARLA frames exceed the real-driving p95 of any of these heads. The model is wrong, confident, and silent.
- **Recurrent state freezes.** Per E4, feature spread (trace of recurrent-state covariance over a window) crashes from 0.25 at alpha=0.0 to 0.00 by alpha=0.78. The model stops integrating new information; it has stopped seeing.

The centerpiece is E3: the in-model uncertainty channel that downstream code would use to detect this condition does not move. This is silent failure, not a noisy failure.

## 2. Existing defenses, and why they miss this mode

Output-side runtime monitors are the standard defense in shipped driving stacks. Each one fails for the failure mode above.

- **Predictive-uncertainty heads.** Direct E3 evidence: ratios 1.20-1.84x of real, 0% of CARLA frames above real p95. A threshold tuned on real driving will not fire. This is the defense closest to a learned OOD detector inside supercombo, and it is silent.
- **Plan-feasibility and plausibility limits (steering rate, accel bounds, lateral acceleration caps).** Per E1, the plan and accel_t0 outputs collapse toward zero (plan activity 0.0057x real, accel_t0 activity 0.0040x real). A zero-magnitude plan does not violate plausibility limits; it looks like a stationary scene with no required action. Plausibility passes.
- **Output-disagreement / temporal-consistency checks (frame-to-frame jitter, output variance).** Once outputs freeze (E1 activity ratios under 1%), temporal variance drops, not rises. A jitter monitor would interpret the freeze as the system being more stable, not less.
- **Ensembles on the same architecture.** E5 localizes the collapse downstream of the vision encoder (per-stage activity ratios at alpha=1: head 2.14, stem 1.43, stage3 2.06; the visible drop is downstream of the recurrent / policy stack, not at the encoder). [UNVERIFIED interpretation of E5; the report explicitly says collapse is downstream of the encoder but layer-by-layer cliff-alpha values are nan, so we infer rather than measure the exact downstream layer.] An ensemble of the same architecture trained on the same data would share the same downstream collapse path.
- **Input-side image quality checks (blur, exposure, sensor noise).** CARLA-clean is sharper, more uniform, and less noisy than real driving footage. Input-quality monitors will rate it as good.

Every output-side defense we are aware of either does not fire (uncertainty, jitter) or treats the collapse as a benign quiet scene (plausibility, image quality).

## 3. Our claim (E6)

An internal-feature monitor over the recurrent state catches the OOD condition before outputs collapse.

- Monitored quantity: rolling temporal spread of the 512-D recurrent feature vector emitted by supercombo (`src/e6_detector.py`).
- Threshold calibration: leave-one-corpus-out across {subaru, ram}. LOCO mean FPR 1.03%, LOCO max FPR 2.07%.
- Detector response on the E4 sweep: fires (>50% of frames flagged) at alpha = 0.550.
- Output cliff on the same sweep (E4): output activity falls from 0.9x to 0.1x of real over alpha 0.784 to 0.799.
- Gap: detector at alpha=0.55, cliff at alpha=0.78. The detector fires roughly 0.23 in alpha-units before the cliff.

The claim is not that E6 replaces output-side monitors. It is that the model's own internal features carry an OOD signal that the output heads do not surface, and that an internal-feature monitor is a complementary safety layer: cheap (one forward pass already happening, one O(d) statistic per frame), shipped-model compatible (no retraining, no architectural change), and calibrated against real-driving FPR rather than against simulated negatives.

## 4. Limitations

- **N=1 model.** supercombo v0.9.7 only. We have not tested earlier or later openpilot models, Tesla, Mobileye, Waymo, or any research IL stack.
- **N=1 OOD axis.** CARLA-clean is a known extreme distribution shift (synthetic rendering, simplified textures, no real sensor noise, no weather). It is the easiest case in which to find a cliff. We do not claim E6 generalizes to harder real-world shifts on present evidence.
- **No real-world OOD validation.** Rain, night, glare, fog, novel geography, sensor degradation, and adversarial perturbations on real footage are project-pending (see `docs/paper_plan.md` once Agent A writes it; the project memory lists "generalize E6 across sim engines + real-world OOD stimuli" as the next step).
- **E5 layer localization is partial.** Per-stage cliff-alpha values are nan in `report/e5_results.md`; we conclude the collapse is downstream of the encoder by ruling out the encoder, not by pinning the collapse to a specific recurrent submodule. Pinning is also project-pending.
- **In-sample vs LOCO FPR.** In-sample FPR is 1.15% (definitional, not a generalization claim). LOCO max is 2.07% on the ram corpus. With only two real corpora the LOCO estimate has high variance; a third corpus is needed before quoting a single FPR.

## 5. Safety implication

Simulation-based validation of openpilot-class driving models, including coverage-guided testing (DeepXplore, DeepTest, DeepRoad), gives false confidence when the simulator is OOD to the model under test. E1-E4 show that supercombo's outputs collapse on the cleanest possible CARLA input, and E3 shows the in-model uncertainty channel does not flag the condition. Any safety case for a shipped driving model that relies on output-side monitors alone is missing this failure mode. Industry practice should add a runtime monitor on the model's own internal features, of the kind E6 demonstrates, calibrated under leave-one-corpus-out FPR against real-driving negatives. The cost is one O(d) statistic per inference; the benefit is catching the OOD condition roughly 0.23 alpha-units before outputs cliff, with sub-2% real-driving false-positive rate on the corpora tested.
