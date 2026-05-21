# Phantom-Braking Trigger Reproduction Pipeline (MVP)

Research benchmark for identifying visual conditions that trigger false-positive braking in vision-based L2 driving stacks. MVP reproduces one scenario (highway overpass shadow) and runs openpilot's supercombo against it in CARLA.

## Status

- **Step 2 (model load + GPU inference benchmark):** done 2026-05-17. Median 1.96 ms on RTX 5070 (Blackwell sm_120, ORT 1.23.2 + cuDNN 9 pip wheels).
- **Step 3 (preprocessing + recurrent state):** done 2026-05-17. Byte budget 6504 from .onnx metadata, ModelStateMirror mirrors `openpilot.modeld.ModelState` line-for-line, frame-1-vs-100 divergence demo passed.
- **Step 3.5 (real-frame parity vs v0.9.7 reference):** PASS 2026-05-17. 100% of 1159 frames within ±0.5 m/s² on Subaru regen segment (median |delta| 0.040, max 0.335). Bit-identical between Py 3.11/ORT 1.26 and Py 3.10/ORT 1.23.2.
- **Step 4a (overpass survey):** done 2026-05-17. Committed to Town05 op01.
- **Step 4b (Town05 op01 two-phase harness):** done 2026-05-21. `src/scenario.py`
  drives the ego kinematically through the op01 viaduct with a dual camera and
  runs supercombo per frame; warmup settles (std 0.018), trace smooth (jerk
  0.004 m/s²). Two open findings: supercombo emits its zero-input default on
  clean CARLA road (sim domain gap), and the hard-brake excursion (to -1.9 m/s²)
  onsets in the urban-clutter zone, not at the shadow peak. See the vault for
  the direction call before Step 4c.

## Prerequisites

- **Python 3.10** (NOT 3.11). CARLA 0.9.15's bundled Python client ships cp37 and cp310 wheels only. We use Python 3.10 to keep the carla client install simple; this drove a small ORT downgrade (1.26.0 → 1.23.2) since 1.26 dropped cp310. No behavior regression — parity is bit-identical between the two configurations.
- Ubuntu 22.04 LTS, NVIDIA driver ≥ 570, CUDA toolkit 12.x system install.
- CARLA 0.9.15 binary at `~/Sim/CARLA_0.9.15/` (launch with `./CarlaUE4.sh -RenderOffScreen -quality-level=Epic -carla-rpc-port=2000`).

## Pinned versions

- **openpilot:** v0.9.7 (tag)
- **supercombo.onnx:** v0.9.7, fetched from `github.com/commaai/openpilot/raw/v0.9.7/selfdrive/modeld/models/supercombo.onnx`, 51.45 MB, exported from PyTorch 2.2.2
- **parser + state source:** `references/openpilot-v0.9.7/` (parse_model_outputs.py, constants.py, fill_model_msg.py, modeld.py, get_model_metadata.py, common/transformations/*.py, tools_lib/*)
- **onnxruntime-gpu:** 1.23.2 with `ORT_DISABLE_ALL` graph optimization (Level2 SimplifiedLayerNormFusion is incompatible with this export at this ORT version family)
- **NVIDIA runtime:** cuDNN 9.22 + CUDA 12.9 via pip wheels, preloaded with `ort.preload_dlls()`
- **CARLA Python client:** 0.9.15 (PyPI wheel, cp310)

## CARLA town overpass methodology

Canonical "does this town have real overpasses" check (built 2026-05-17 after Town06 turned out to be flat):
1. `cm.generate_waypoints(distance=2.0)` filtered to `LaneType.Driving`.
2. Spatial-bucket by `(int(x/R), int(y/R))` with `R=20m`.
3. For each waypoint, scan own cell + 8 neighbors. If any neighbor's Z exceeds `self.Z + 4m`, it's an under-passage point.
4. Cluster-merge by 60m proximity.
5. Walk `wp.next(2)` / `wp.previous(2)` from each candidate until junction, fork, dead-end, or accumulated bend >3°/m. Runway = max(fwd, bwd).
6. Cross-check `world.get_environment_objects(CityObjectLabel.Bridge)` — count ≠ 0 with Z-spread ≈ 0 means visual bridges with flat OpenDRIVE topology (Town01, Town10HD).

**Do not use absolute Z thresholds** — Town13's whole map sits at Z=144–192m, an absolute-Z detector returns 100% of waypoints as "elevated" and finds nothing. Always relative.

Census results for CARLA 0.9.15:

| town | best under-road runway | overpass count | notes |
|---|---|---|---|
| Town01–02, Town06, Town10HD, Town12 | 0 (flat) | 0 | no real road-over-road |
| Town04 | 84 m | 2 | figure-8 loop, 11m overhead |
| Town05 | 108 m | 5 | ring around grid town |
| Town13 (HD) | 784 m | 11 | requires AdditionalMaps; UE4 segfaults on sensors on Blackwell |

See `scripts/survey_town13.py` for the canonical implementation.

## Camera config (Step 4b, as built)

- Narrow road cam: rectilinear, 1928×1208, **40.05° horizontal FOV** — the fov
  that reproduces comma 3 fcam intrinsics (focal 2648 px). The earlier "~73°"
  note was wrong; fcam is ~40° HFOV. Rendering at fcam intrinsics lets the
  verified Step 3.5 warp map the frame into the 512×256 medmodel frame with a
  zero calibration euler (sim camera is mounted perfectly).
- Wide road cam: rectilinear, 1928×1208, ~119° HFOV (comma 3 ecam focal). The
  real ecam is a fisheye a CARLA pinhole cannot reproduce — recorded for
  overlays; the model's `big_input_imgs` is fed the narrow frame (Step 3.5
  showed accel@t0 is narrow-dominated).
- Rate: 20 Hz, synchronous mode, fixed_delta 0.05.
- Ego playback is kinematic (physics off, `apply_batch_sync(ApplyTransform)`
  one arc-length per tick) — fully deterministic.

## Plan tensor — longitudinal accel @ t≈0

Verified against `references/openpilot-v0.9.7/`:
- `parsed_outs['plan']` shape `(1, 33, 15)`
- `Plan.ACCELERATION = slice(6, 9)` (ax, ay, az)
- **`long_accel_t0 = parsed_outs['plan'][0, 0, 6]`**

## Quickstart

```bash
cd ~/Projects/phantom-braking
uv venv --python 3.10 --seed .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python src/model_runner.py    # Step 2 — load + GPU benchmark
.venv/bin/python -m src.run_parity      # Step 3.5 — parity vs comma reference

# requires CARLA server on localhost:2000:
#   cd ~/Sim/CARLA_0.9.15 && ./CarlaUE4.sh -RenderOffScreen -quality-level=Epic -carla-rpc-port=2000
.venv/bin/python -m pytest tests/test_carla_connection.py -v
```
