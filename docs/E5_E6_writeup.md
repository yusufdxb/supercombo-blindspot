# E5 + E6: Where does openpilot collapse, and can we see it coming?

## Motivation

E1 through E4 established that openpilot's supercombo model (v0.9.7) degrades
predictably as the input distribution drifts away from its training regime,
with output behaviour cliffing around alpha ~= 0.78 on a controlled CARLA
sweep. That left two questions open. First, *where* in the network does the
collapse actually happen: is the vision encoder going blind, or is the damage
further downstream? Second, can we detect the OOD condition from internal
activations *before* the outputs visibly fail, which is the only useful
regime for a runtime monitor on a real car?

E5 and E6 answer both questions.

## E5: the encoder is not the failure point

Method. We instrumented the supercombo ONNX graph and pulled activations
from every vision-encoder stage along the same alpha sweep used in E4. For
each stage we computed an "activity ratio" against the in-distribution
baseline, so a stage that has gone silent under OOD input shows up as a
ratio near zero.

Headline number. No vision-encoder stage drops below **0.96 activity** at
any point in the sweep, including the alpha values where the model's
trajectory outputs have already cliffed in E4. The encoder keeps producing
features with comparable activity statistics through the entire range we
tested.

Implication. The collapse observed in E4 is **downstream of the
vision encoder**, in the recurrent / policy stack. The blind-driving
behaviour is not the network failing to see; it is the network failing to
*use* what it sees. This rules out the most intuitive failure model (encoder
saturates on OOD pixels) and redirects the search to a much smaller part of
the network.

## E6: a monitor on internals fires before the outputs cliff

Method. We computed a rolling spread statistic on the 512-D recurrent
feature vector that feeds the policy head. Threshold calibration was done
against real-driving traces so we could pin a concrete false-positive rate
rather than report an in-sample number. We then ran the calibrated detector
along the same E4 alpha sweep and recorded the alpha at which it first
fires.

Headline numbers.

- Detector fires at **alpha = 0.55** (the point where >50% of blended
  frames are flagged, not the point of perfect separation; full
  separation is at alpha=1.0, AUROC 0.996).
- E4's output-collapse cliff sits at **alpha ~= 0.78**.
- Threshold calibrated to **1.15% FPR** on real-driving traces (see the
  Limitations note below on how this number is validated).

Implication. A monitor that watches the recurrent feature vector flags the
OOD condition *before* the outputs visibly fail. That is the regime an
on-car monitor needs to live in: the model has not produced a bad
trajectory yet, but the internal state already looks unlike anything from
the training distribution. This is meaningfully different from a monitor
that watches the model's outputs, which by construction can only react
*after* the failure is already in the actuator path.

## What this implies for AV monitoring

Two practical takeaways for anyone building runtime safety on top of a
black-box driving model:

1. Vision-encoder health checks are not sufficient. In this model, on this
   shift, the encoder looks fine the whole way through the cliff. A
   monitor that only watches encoder statistics would have missed every
   failure E4 documented.
2. A cheap statistic on the recurrent feature vector (a single rolling
   spread, no learned components) gave us roughly 0.23 alpha of headroom
   between "detector fires" and "outputs collapse." That is the kind of
   margin a supervisor or fallback policy can actually act on.

## Open questions

- **PARTIALLY RESOLVED:** E5 submodule probing (`report/e5_submodule_results.md`)
  localized the cliff entry to `summarizer_div` (the VAE-mu / normalized
  hidden_state bottleneck, cliff alpha 0.900) with amplification at
  `action_block_body` (cliff alpha 0.500), which folds in the model's
  own collapsed `prev_desired_curv` output in a feedback loop. The
  transformer + reduce-sum stage is a passive relay. Caveat: the
  summarizer ends with `mu / sigma` (VAE-style reparameterization), so
  part of the apparent collapse in `summarizer_div` could be variance
  normalization rather than information loss; probing the pre-Div mu
  tensor alone would split that further.
- **PARTIALLY RESOLVED (E7):** E7 tested E6 across 15 ImageNet-C corruptions
  x 5 severities on real frames. E6 mostly fails on photometric corruptions
  (mean AUROC 0.52-0.74) but catches extreme noise/frost that freeze the
  recurrent state (frost sev 5: AUROC 1.000). E6 is a collapse detector,
  not a universal OOD detector. Feature-space baselines (Mahalanobis)
  complement E6 on corruption-type OOD. Other sim engines (MetaDrive) and
  real adverse weather remain OPEN.
- **RESOLVED:** The baseline comparison is complete
  (`report/baselines_results.md`, `report/metrics_results.md`). Three
  applicable feature-space baselines (Mahalanobis, Relative Mahalanobis,
  KNN-50) were evaluated on the same 512-D recurrent feature under
  the same LOCO protocol. All three hit 100% LOCO held-out FPR because
  the subaru and ram corpora occupy disjoint regions of the feature
  space. E6's rolling-spread monitor is location-invariant (second-order
  trace, not absolute position) and is the only detector that both
  separates ID/OOD (AUROC 0.996 at alpha=1.0) and calibrates across
  corpora (LOCO mean FPR 1.03%). MSP, Energy, and ViM are structurally
  not applicable to supercombo's regression-head design.

## Reproducing

The full E5 sweep, the E6 calibration, and the alpha-sweep evaluation are
all driven by CLIs under `src/` with tests under `tests/`. See the project
README for the exact invocations and the dataset layout.
