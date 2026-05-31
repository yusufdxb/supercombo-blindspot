# Daytime attractor analysis (P2b): why does a clean ID segment fire E6?

**Question.** A real, in-distribution daytime-dry comma-3 segment (dongle
`376bf99325883932`, seg 1) intermittently enters a near-zero recurrent
attractor and fires the E6 collapse detector at ~57.9% of frames, mirroring
the CARLA synthetic collapse. The earlier "high steer + low speed" trigger is
**falsified** (EV6 night reaches a higher peak steer at the same speed without
collapsing). Trigger is currently unexplained.

## 1. Evidence available (no GPU needed)

- **Cached hidden states**: `report/real_weather_collected.npz`, key
  `daytime_control__hidden_state` (320 frames x 512-D), plus all output heads
  (`plan`, `lane_lines`, `desired_curv`, etc.) and predicted-uncertainty arrays
  for the same frames. Night and CARLA references in the same file +
  `report/teardown_collected.npz` (`subaru__`, `ram__`, `carla__`).
- **Detector code**: `src/e6_detector.py::rolling_spread` (window=30,
  threshold=0.078873) — operates purely on the cached hidden_state, so every
  hypothesis below is testable offline by slicing the npz.
- **State threading**: `src/state.py::ModelStateMirror.run` — `features_buffer`
  and `prev_desired_curv` shift-and-append from the model's own outputs each
  frame; zero-init only on frame 1.
- **Input parity (load-bearing)**: `src/probe_model.py::collect` calls
  `state.run(inp, inp)` with **no desire / traffic_convention /
  lateral_control_params** for *all* segments. So those inputs are identically
  zeroed everywhere — they cannot be the differentiator. The only per-segment
  recurrent input is `prev_desired_curv`, which is fed from the model's own
  predicted curvature, plus the image stream itself.
- **Documented observations** (`report/real_weather_results.md` NOTE,
  2026-05-30): hidden-state L2 norm is **bimodal** (no mass between 0.05 and
  0.5), toggles from frame ~16, pops back to ~1.0 near frame 200; in the
  low-norm regime output heads are suppressed ~45x (desired_curv 0.003 vs 0.137).

## 2. Hypotheses

**H1 — prev_desired_curv positive-feedback latch.** `prev_desired_curv` is the
sole per-segment scalar fed back into the recurrent core. If a near-straight
daytime road drives predicted curvature toward ~0, the model feeds ~0 back in,
which can self-reinforce a low-curvature / low-activity fixed point. This is the
one input that differs from the (non-collapsing) night segments and is a
genuine recurrent loop, unlike the zeroed control inputs.

**H2 — multi-modal attractor in the GRU itself (architectural).** The bimodal
norm with no intermediate mass is the signature of a discrete basin, not a
gradual drift. The GRU may have two stable points (active ~1.0, near-zero
~0.004); specific benign image sequences nudge it across the separatrix. CARLA
sits permanently in the near-zero basin; this segment toggles.

**H3 — image-content trigger (data-driven).** Something visual in *this* clip
(low-texture road, sky/overpass, lead-vehicle occlusion, lens flare) repeatedly
pushes the feature extractor toward a degenerate embedding the recurrent core
collapses on. The toggling and recovery near frame 200 fit a transient scene
feature, not a global property.

**H4 — fp16 / numerical underflow in the low-norm regime.** State buffers feed
the ONNX session as float16 (`state.py::_build_feed`). A state near 0.004 is
close to fp16 granularity; rounding could pin it, explaining the hard bimodality
(no intermediate values) better than a smooth dynamical basin.

**H5 — warmup / init coupling.** Norm toggles from frame ~16, well inside the
100-frame warmup that is normally discarded. The collapse may be seeded by the
zero-init transient interacting with this segment's early frames specifically.

## 3. Testable predictions (offline, from the cached npz)

- **H1**: per-frame correlation of `desired_curv` with hidden-norm regime;
  low-norm frames should cluster at near-zero predicted curvature, and the
  toggle should *lag* a curvature collapse by ~1 frame. Night segments (higher
  sustained curvature) should not enter low-norm. → slice `daytime_control__*`.
- **H2**: cluster the 512-D states (k=2). Predict clean bimodal membership;
  measure whether CARLA's mean state and the daytime low-norm cluster centroid
  coincide (cosine ~1) — same basin vs merely both small.
- **H3**: align low-norm frame indices to decoded frames (re-decode HEVC, CPU
  only) and inspect what is on screen during low-norm vs high-norm windows.
- **H4**: recompute `rolling_spread` on the **fp32** cached states (already
  fp32 in the npz). If the bimodality persists in fp32 the model dynamics cause
  it; if it softens, fp16 feedback is implicated (would need a re-run to confirm).
- **H5**: check whether the first low-norm onset (frame ~16) is reproducible
  from a cold start vs after a longer warmup — needs a short re-run, defer.

## 4. Recommended next diagnostic

Run **H1 + H2 together, purely on the cached arrays** (no GPU): for
`daytime_control`, (a) compute per-frame hidden-norm, label low/high regime by
the 0.05–0.5 gap; (b) regress regime against the lagged `desired_curv` scalar
and against the night/CARLA references; (c) cluster states k=2 and test cosine
similarity of the low-norm centroid to the CARLA collapse mean. This
discriminates H1 (curvature-latched), H2 (shared CARLA basin), and rules them in
or out in one offline script before spending any GPU on H3/H5. If the low-norm
centroid is cosine ~1 with CARLA and lags a curvature dip, H1+H2 is the
mechanism and the open caveat closes; if not, escalate to H3 image inspection.
