# supercombo-blindspot

**Does a production L2 self-driving model know when it's blind? It doesn't, and it fails silently.**

[![CI](https://github.com/yusufdxb/supercombo-blindspot/actions/workflows/ci.yml/badge.svg)](https://github.com/yusufdxb/supercombo-blindspot/actions/workflows/ci.yml)
&nbsp;[![openpilot](https://img.shields.io/badge/openpilot-v0.9.7%20%2F%20v0.9.6-1f6feb)](https://github.com/commaai/openpilot)
&nbsp;[![reproducible](https://img.shields.io/badge/reproducible-from%20cache-2da44e)](#reproducibility)

`supercombo` is the end-to-end neural network that drives [openpilot](https://github.com/commaai/openpilot),
the L2 driver-assistance system running on comma hardware on public roads today. This project
instruments openpilot's shipped `supercombo` model and answers one safety question:

> When the model is shown input it was never trained on, does it fail loudly, or silently?

**The answer, measured at the model's own output channels: silently, and completely.** On
CARLA-rendered driving scenes, openpilot v0.9.7's outputs collapse to a near-constant default
across 8 of its 10 output heads, its internal recurrent state freezes to a single point, and its
exported predictive-uncertainty heads rise so little they never leave the model's normal
real-driving range. Nothing the model emits would tell a downstream monitor it has stopped
perceiving. An internal recurrent signal *does* carry the failure and is detectable (E6), but the
model never surfaces it.

A full writeup is in [`drafts/paper.pdf`](drafts/paper.pdf).

---

## Contents

- [Findings](#findings)
- [Scope of claims](#scope-of-claims)
- [Significance](#significance)
- [Controls and validity](#controls-and-validity)
- [Experiments](#experiments)
- [Generalization and deployment](#generalization-and-deployment)
- [Limitations](#limitations)
- [Reproducibility](#reproducibility)
- [Repository layout](#repository-layout)
- [Environment](#environment)
- [Attribution and disclaimer](#attribution-and-disclaimer)

---

## Findings

![The four findings at a glance](report/figures/hero.png)

A **parity-exact** reimplementation of openpilot v0.9.7 `supercombo` inference, verified to **100% of
1159 frames within ±0.5 m/s²** of comma's own reference output on real footage (median abs delta
0.04 m/s²), is the foundation: the negative result below is the model, not the harness. With that
control in place, the model was run on real comma footage versus CARLA renders, instrumenting every
output head, every predicted uncertainty, and the internal feature vector.

| | Experiment | Finding |
|---|---|---|
| **E1** | output collapse | 8 / 10 output heads (plan, lane lines, road edges, lead, curvature, ...) collapse to **< 1%** of their real-footage temporal activity on sim input |
| **E2** | internal OOD | the 512-D recurrent feature vector collapses to **0.00001×** the real spread: 219 distinct sim frames map to one frozen point |
| **E3** | silent failure | outputs lose **~99.5%** of their activity, yet predicted uncertainty rises only 1.2-1.8×, and **0%** of sim frames exceed the model's normal real-driving uncertainty |
| **E4** | cliff, not gradient | blending CARLA into a real frame, output activity first balloons to **6.3×** baseline (ghosted-input thrash), then collapses in a **hard cliff** at ~78% CARLA (transition width **0.015**); uncertainty never spikes through it |
| **E5** | encoder is fine | per-stage temporal activity (CARLA / real) stays at or above baseline through the full sweep (min 0.96, several layers amplify 1.4-2.1×). The collapse is **downstream** of the encoder, in the recurrent / policy stack |
| **E6** | a monitor catches it | a 1st-percentile threshold on the rolling spread of the model's own 512-D recurrent vector fires on >50% of CARLA-blended frames at alpha 0.550, well before the E4 cliff at ~0.78. Leave-one-corpus-out over four real corpora: **2.41%** mean false-positive rate (95% CI [0%, 5.17%]) |
| **E7** | not a universal detector | 15 ImageNet-C corruptions × 5 severities: E6 (a collapse detector) mostly misses photometric corruption (mean AUROC 0.52-0.74); feature-space baselines (Mahalanobis) catch what E6 misses |

Every claim is registered with its supporting evidence in [`paper_state/claim_ledger.md`](paper_state/claim_ledger.md).

## Scope of claims

What this project does and does not claim, by confidence bucket:

| Bucket | Claims |
|---|---|
| **VERIFIED** (v0.9.7, CARLA, Subaru/RAM) | E1: 8/10 output heads collapse below 1% of real activity. E2: recurrent feature separates from real at 87.9% (d'=2.19). E3: exported uncertainty heads rise only 1.20-1.84×; 0/219 CARLA frames exceed real p95. E4: collapse is a hard cliff on Subaru (width 0.015) and a gradient on RAM (width 0.274). |
| **REPLICATED on v0.9.6** | v0.9.6 is also out-of-distribution-blind in feature space (d'=6.8, 100% linear separability). |
| **DIFFERS on v0.9.6** | Silent freeze does not replicate (1/10 heads collapse vs 8/10); the model fails by chaotic amplification instead. The E6 monitor does not transfer (33% LOCO FPR vs 2.4% on v0.9.7). |
| **MONITOR-ONLY (E6)** | The rolling recurrent-spread detector is a collapse detector, not a general OOD detector. E7 shows photometric corruptions evade it (mean AUROC 0.52-0.74). |
| **DEPLOYMENT-UNSUPPORTED** | Scaling clean-real calibration from N=2 to N=4 raised LOCO mean FPR from 1.03% to 2.41% (95% CI [0%, 5.17%], 6.90% max on the ram fold). Fleet-scale FPR is unproven and likely higher. |
| **OPEN** | One real daytime-dry segment intermittently enters a near-zero recurrent attractor (E6 fires 58% of frames) on clean, correctly-warped input. The trigger is unexplained; an initial steer/speed hypothesis was falsified. |

## Significance

Every L2 and autonomous-driving program validates in simulation. If a production driving model is
out-of-distribution-blind to your simulator, sim "passes" are false confidence: the car looks like
it drives (stable, benign, plausible outputs) because the model has **collapsed to a safe-looking
default**, not because it perceives anything in the scene. And because the model's own uncertainty
heads do not flag the collapse (E3), you cannot catch this from model outputs alone. You need an
external distribution-shift detector.

This is consistent with comma's own experience: openpilot's official simulator bridge (MetaDrive)
is [reported to drive erratically](https://github.com/commaai/openpilot/issues/31711), and comma
uses sim for integration and CI testing, not for trusting model behavior.

**Provenance.** The project began as an attempt to reproduce a documented openpilot
failure, [phantom braking at highway overpass shadows](https://github.com/commaai/openpilot/issues/20704),
inside CARLA. The reproduction harness was built and works (`src/scenario.py`), but the model did
not respond to the simulated scenes. Chasing *why* produced the teardown above. The project pivoted
from "reproduce a known bug" to "rigorously characterize a silent failure mode," and the
phantom-braking harness stayed as the control that exposed the real result.

## Controls and validity

- **Parity control.** `src/run_parity.py` reproduces comma's v0.9.7 reference output on a real
  segment to 100% within ±0.5 m/s². A skeptic's first objection ("your reimplementation is buggy")
  is ruled out before any claim is made.
- **The model is alive on real data.** Every output head has substantial frame-to-frame activity on
  real footage (E1, "real activity" column). The collapse is sim-specific.
- **Two real segments, two vehicles** (Subaru highway, RAM), each warmed from an independent
  recurrent state with the warmup transient discarded, so the "real" baseline is neither one
  recording nor contaminated by initialization.
- **Honest negative on the original goal.** `src/scout_phantom.py` scanned v0.9.7's output on real
  drives for phantom brakes; it found legitimate curve and intersection braking and **no confirmed
  phantom brake** in the sample. Phantom braking is rare and the easily accessible data is
  failure-poor: a real finding about the difficulty of the original problem, reported rather than
  hidden.

## Experiments

<details>
<summary><b>E1: Output collapse map</b></summary>

Per output head, the temporal activity (mean per-element standard deviation across frames) on CARLA
versus real footage. 8 of 10 heads collapse below 1% of real activity, including every perception
head (`lane_lines`, `road_edges`, `lead`) and every planning head (`plan`, `accel`, `desired_curv`,
`desire_state`). `pose` (ego-motion) partially survives at 18%, plausibly because it is driven by
frame-to-frame optical flow, which retains some signal even in sim. `meta` (disengage / blinker
probabilities) is low-activity on real footage too.

Full table: [`report/teardown_results.md`](report/teardown_results.md).
</details>

<details>
<summary><b>E2: Out-of-distribution inside the model</b></summary>

![E2 feature space](report/figures/e2_feature_ood.png)

`supercombo` carries a 512-D `hidden_state` feature vector. Projected to 2-D (PCA fit on real
features), real driving spreads across the feature space while **219 distinct CARLA frames collapse
to a single point** (feature spread 0.00001× of real). The model's internal representation of the
sim world is frozen and degenerate.
</details>

<details>
<summary><b>E3: The silent failure</b></summary>

![E3 silent failure](report/figures/e3_confidence.png)

`supercombo`'s plan / lead / curvature heads emit predicted uncertainties (MDN standard
deviations). If the model "knew" it was out of distribution, those would spike. They do not: outputs
lose ~99.5% of their activity, predicted uncertainty rises only 1.2-1.8×, and **not one CARLA
frame's uncertainty exceeds the model's 95th-percentile uncertainty on normal real driving.** Any
monitor thresholded to not false-alarm on real driving would never fire. The exported uncertainty
channel is confidently silent about the collapse, even though (E6) an internal recurrent-spread
signal does carry the information.
</details>

<details>
<summary><b>E4: Cliff, not gradient</b></summary>

![E4 interpolation](report/figures/e4_interpolation.png)

E1-E3 show the model collapses on CARLA; E4 asks *how the collapse arrives*. Each real Subaru
model-frame is blended with a CARLA frame, `X(alpha) = (1-alpha)·real + alpha·CARLA`, swept across
29 auto-refined alpha points. Across the first ~78% of the blend the model never degrades
gracefully: output activity instead *balloons*, peaking at **6.3× baseline** near alpha 0.42 as the
ghosted double-exposure makes it thrash. Then, inside a **0.015-wide window near alpha 0.79**,
activity falls off a cliff from 1.4× to 0.03× of real. The 512-D recurrent vector, by contrast,
slides smoothly to the CARLA centroid and is saturated by alpha 0.47, so the internal representation
gives up well before the outputs do. Predicted uncertainty stays flat through the cliff: E3's silent
failure holds across the whole interpolation.

The blend overlays two scenes, so intermediate frames are a double-exposure, not a
content-preserving morph: E4 is an overlay-interference probe along a monotone real-to-sim axis.
Full table: [`report/e4_results.md`](report/e4_results.md).
</details>

<details>
<summary><b>E5: Is the collapse in the encoder, or downstream?</b></summary>

![E5 layer localization](report/figures/e5_layer_localization.png)

Adding one intermediate tensor per vision-encoder stage (stem, stages.0-3, post-pool flatten) to the
ONNX graph and re-running the E4 sweep inverts the naive expectation: across all six encoder layers,
temporal activity on CARLA stays at or above baseline, and several stages amplify (stage3 2.06×,
head 2.14×). Nothing in the encoder collapses. Absolute mean magnitudes *do* shift (stem 1.24×, head
1.33×), so the encoder produces differently-distributed but fully temporally-active features. The
collapse in E1/E2 is therefore not "the encoder went quiet" but the recurrent / policy stack
collapsing the encoder's variation-rich features into a degenerate hidden state. The OOD failure
mode is temporal-aggregation, not perception.

Full table: [`report/e5_results.md`](report/e5_results.md).
</details>

<details>
<summary><b>E6: Could a downstream monitor have caught this?</b></summary>

![E6 detector](report/figures/e6_detector.png)

Instead of trusting any output head, watch the rolling spread of `supercombo`'s own 512-D recurrent
vector. Calibrating the fire threshold at the 1st percentile of the rolling spread on real driving
and evaluating by leave-one-corpus-out over four real corpora (subaru, ram, ev6_night, bronco_night)
gives **2.41% mean held-out FPR** (segment-level bootstrap 95% CI [0%, 5.17%], 6.90% max on the ram
fold). The initial two-corpus estimate of 1.03% was optimistic; the corpora have meaningfully
different rolling-spread distributions (subaru median 0.12 vs ram median 0.19), which is exactly why
the generalization gap matters. On the E4 sweep the detector fires on >50% of frames at alpha 0.550,
while the output-collapse cliff does not arrive until ~0.78: a tiny external monitor watching
internals could flag the OOD condition before the model's own outputs gave it away.

Full table: [`report/e6_results.md`](report/e6_results.md) and [`report/corpus_scaling_results.md`](report/corpus_scaling_results.md).
</details>

<details>
<summary><b>E4-RAM: Vehicle invariance</b></summary>

![E4-RAM interpolation](report/figures/e4_ram_interpolation.png)

Re-running the E4 sweep with a RAM real-driving source: the collapse endpoint is identical (activity
< 1% at alpha 1.0, the feature vector freezes the same way), but the *path* differs.

| Source | Transition width | E6 fires-at-alpha | E6 headroom | Verdict |
|---|---|---|---|---|
| Subaru | 0.015 | 0.550 | 0.234 | cliff |
| RAM | 0.274 | 0.850 | -0.184 | gradient |

E6 fires much later on RAM, providing no early warning. The cliff-versus-gradient distinction is
segment-dependent, so E6's headroom cannot be assumed to generalize across real-driving sources
without re-calibration. Full table: [`report/e4_ram_results.md`](report/e4_ram_results.md).
</details>

<details>
<summary><b>E7: ImageNet-C corruption sweep</b></summary>

![E7 severity sweep](report/figures/e7_severity_sweep.png)
![E7 AUROC heatmap](report/figures/e7_auroc_heatmap.png)

The 15 Hendrycks & Dietterich (ICLR 2019) ImageNet-C corruptions at 5 severities, applied to real
comma frames. E6 mostly fails on photometric corruptions (mean AUROC 0.52-0.74), catching only
extreme corruptions that actually freeze the recurrent state (frost severity 5: AUROC 1.000, impulse
noise severity 5: 0.906). Feature-space baselines (Mahalanobis, Relative Mahalanobis) detect what E6
misses, firing >95% on noise, weather, and compression at moderate severity. E6 monitors temporal
dynamics and fires when the state freezes; photometric corruptions still produce temporally varying
sequences. A production system would need both a temporal monitor (E6) and a feature-space detector.

Full table: [`report/e7_results.md`](report/e7_results.md).
</details>

<details>
<summary><b>Hyperparameter ablations</b></summary>

- **KNN k**: AUROC = 1.000 for all k in {5, 10, 20, 50, 100}; insensitive to neighbour count.
- **E6 window size**: AUROC 0.957 (window=10) to 1.000 (window=50); default window=30 (AUROC 0.996)
  best balances detection power and early warning (fires-at-alpha 0.550).

Full table: [`report/ablations_results.md`](report/ablations_results.md).
</details>

## Generalization and deployment

Four additions test how far the finding travels and make the monitor deployable. Each new number was
independently re-verified by a separate agent and registered in the claim ledger (c52-c61).

- **Second model (openpilot v0.9.6).** The full teardown was re-run on the immediately preceding
  shipped version. Parity holds (100% within ±0.5 m/s² vs comma's v0.9.6 reference, n=560), but the
  failure mode differs: only 1 of 10 heads collapse, the sweep is a gradient of chaotic
  amplification (peaks 14.6× real, stays 3.3× at full CARLA), and the v0.9.7-calibrated monitor does
  not transfer (33% LOCO FPR). Adjacent shipped versions fail OOD in qualitatively different ways.
  ([v0.9.6 teardown](report/teardown_v096_results.md), [E4](report/e4_v096_results.md), [E6](report/e6_v096_results.md), [parity](report/parity_v096_results.md))
- **Real adverse weather.** Real comma-3 night plus headlight/tail-light glare at matched intrinsics
  does **not** collapse v0.9.7 (0/10 heads, E6 fires 0%, vs CARLA 8/10 and 100%). The silent
  collapse is predominantly sim-induced, not a real low-light phenomenon (one daytime segment is the
  open exception above). ([real_weather_results.md](report/real_weather_results.md))
- **Conformal baseline plus lead time.** A split-conformal detector on the KNN-50 score ties KNN on
  single-corpus AUROC (1.000) but also fails cross-corpus (100% LOCO FPR). A lead-time table shows
  E6 is the only detector with both a calibrated cross-corpus threshold and positive detection lead
  (+0.234 blend-units): high single-corpus AUROC does not imply early warning.
  ([conformal](report/conformal_results.md), [lead_time](report/lead_time_results.md))
- **Deployable monitor.** E6 is one O(d) statistic per frame. A portable C++17 implementation
  matches the Python reference to 3.4e-13 and runs in ~0.4 µs per frame on x86 (0.0008% of a 20 Hz
  control budget). The in-the-loop ROS 2 node lives in `policy-health-monitor`; Jetson Orin NX
  on-device timing is HW-pending. ([deployment](report/deployment_results.md), [`deploy/cpp/`](deploy/cpp/))

## Limitations

- Two model versions tested (v0.9.7, v0.9.6); v0.9.6 fails by chaotic amplification, not collapse,
  and the monitor does not transfer. No Tesla, Mobileye, Waymo, or research stack tested.
- "Real" is a small set of calibration segments; a larger real corpus is owed before a production
  FPR can be quoted (N=4 LOCO is honest progress, not a fleet number).
- CARLA only. comma's MetaDrive sim shows consistent erratic behavior (#31711) but is not
  instrumented here.
- E5 localizes the collapse downstream of the encoder; submodule probing pins cliff entry to
  `summarizer_div` (the VAE-mu bottleneck, cliff alpha 0.900) with amplification at
  `action_block_body` (cliff alpha 0.500) via the `prev_desired_curv` feedback loop. The
  summarizer's `mu / sigma` division means part of the apparent collapse could be variance
  normalization rather than information loss.
- E4-RAM shows the cliff/gradient distinction is segment-dependent; E6's early-warning headroom does
  not generalize without re-calibration per source.
- E7 shows E6 is a collapse detector, not a universal OOD detector; a production system needs
  complementary detectors.
- E4's interpolation overlays two scenes (a double-exposure), so it is an overlay-interference
  probe, not a photometric sim-to-real morph. Its 0.015 transition width is a linear-interpolation
  estimate within a single 0.025-wide alpha step.

## Reproducibility

**The teardown runs from a fresh clone**, with no model, no CARLA, and no multi-GB raw frames. It
re-derives every E1 / E2 / E3 / E4 table and figure from the committed output caches
(`report/teardown_collected.npz`, `report/e4_collected.npz`).

```bash
pip install -r requirements-ci.txt matplotlib
python -m src.teardown      # E1/E2/E3 tables + figures, from the cache
python -m src.e4_interp     # E4 interpolation sweep, from its cache
python -m pytest -q         # unit tests + teardown and E4 regression tests
```

<details>
<summary><b>Full end-to-end run (model + CARLA + parity control)</b></summary>

Requires the full stack: Python 3.10 (for the CARLA 0.9.15 client), `supercombo.onnx`, the real
comma segments, and a CARLA frame capture.

- `supercombo.onnx` (51 MB): from the [openpilot v0.9.7 release](https://github.com/commaai/openpilot/releases/tag/v0.9.7), placed at `models/supercombo.onnx`.
- Real comma driving segments: `python -m scripts.fetch_upgrade_data` (also fetches v0.9.6 reference outputs).

```bash
uv venv --python 3.10 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt

env -u PYTHONPATH .venv/bin/python -m src.run_parity         # parity control vs comma's reference
env -u PYTHONPATH .venv/bin/python -m src.teardown --collect # re-collect, then re-derive
```

`env -u PYTHONPATH` is used because a sourced ROS 2 environment otherwise shadows packages; the
project venv is self-contained.

**E5 / E7 / E4-RAM** collection requires GPU + CARLA / real data (`python -m src.e5_submodule
--collect`, `src.e7_corruption --collect`, `src.e4_ram --collect`); analysis runs from cache without
either dependency. These large caches are regenerated via `--collect` rather than shipped.

**2026 upgrade (v0.9.6, real weather, conformal, deployment).** `python -m scripts.fetch_upgrade_data`
re-fetches the v0.9.6 ONNX, comma's v0.9.6 reference, and the real night/glare/control segments
(none redistributed). Then:

```bash
env -u PYTHONPATH .venv/bin/python -m src.run_parity --model v096   # v0.9.6 parity gate
python -m src.conformal_results && python -m src.lead_time          # conformal + lead-time, from caches
( cd deploy/cpp && make )                                           # C++ E6 latency microbenchmark
```
</details>

## Repository layout

| Path | What |
|---|---|
| `src/state.py`, `src/parser.py`, `src/constants.py` | parity-exact `supercombo` inference + recurrent state |
| `src/run_parity.py`, `src/warped_preprocessor.py`, `src/transformations.py` | real-footage parity pipeline (calibrated warp) |
| `src/probe_model.py`, `src/teardown.py` | the E1 / E2 / E3 distribution-shift teardown |
| `src/e4_interp.py`, `report/e4_collected.npz` | E4 real-to-sim interpolation sweep + cached outputs |
| `src/e4_ram.py` | E4-RAM vehicle-invariance sweep |
| `src/e7_corruption.py` | E7 ImageNet-C corruption sweep |
| `src/ablations.py` | KNN k-sensitivity and E6 window-size sweeps |
| `src/scenario.py`, `src/sim_preprocessor.py`, `src/path_sampling.py` | CARLA reproduction harness (the control) |
| `src/scout_phantom.py` | phantom-brake scout for real comma drives |
| `report/teardown_collected.npz` | cached per-frame model outputs (E1/E2/E3 re-derive from this) |
| `report/figures/`, `report/*.md` | figures and per-experiment result tables |
| `deploy/cpp/` | portable C++17 E6 monitor + latency microbenchmark |
| `references/openpilot-v0.9.7/` | vendored openpilot v0.9.7 source (parity reference) |
| `drafts/paper.pdf` | full writeup |

## Environment

openpilot **v0.9.7** (`supercombo.onnx`, 51 MB, from the v0.9.7 tag) and **v0.9.6** (upgrade
section); onnxruntime-gpu **1.23.2** with `ORT_DISABLE_ALL` graph optimization; Python **3.10**;
CARLA **0.9.15**. Runs on an RTX 5070 (Blackwell sm_120): first inference pays a ~28 s PTX JIT, then
~2 ms/frame.

## Attribution and disclaimer

openpilot and `supercombo` are property of comma.ai and vendored here under their respective terms
for parity-reference purposes only. This is independent research and is not affiliated with or
endorsed by comma.ai.
