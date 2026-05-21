# E4: real-to-sim interpolation sweep

Date: 2026-05-21
Project: supercombo-blindspot (phantom-braking)
Status: approved, ready for implementation plan

## Question

The E1/E2/E3 teardown showed supercombo's outputs collapse on CARLA imagery.
E4 asks one follow-up: is that collapse a sharp cliff or a smooth gradient
along the real-to-sim axis?

## Method

Build a monotone interpolation path from a known-alive input (real comma
footage) to a known-collapsed input (CARLA renders), and measure the model
along it.

1. Load the Subaru real model-frame sequence `R = [R_0 .. R_{N-1}]` and the
   CARLA sequence `C = [C_0 .. C_{N-1}]`. These are the `(6,128,256)`
   medmodel-frame tensors produced by `load_real_six` / `load_carla_six` in
   `src/probe_model.py`, the exact inputs E1 uses. `N = 320`.
2. For each blend weight alpha, build the pairwise-blended sequence
   `X(alpha)_k = (1 - alpha) * R_k + alpha * C_k`.
3. Run the existing `collect()` (`src/probe_model.py`) on each blended
   sequence: a fresh `ModelStateMirror`, warmed over the sequence, with the
   first `WARMUP = 100` frames discarded. Identical methodology to E1/E2/E3.
4. Sweep alpha at 11 uniform steps `0.0, 0.1, ..., 1.0`. If the output-activity
   curve shows a sharp transition, run one refinement pass with finer steps
   bracketing it.

The blend is a linear combination in the model-frame tensor space. Inputs are
unnormalized YUV (0..255); a convex blend of two such tensors stays in range.

## Endpoints as a consistency check

alpha = 0 is the real Subaru condition and must reproduce E1's real activity.
alpha = 1 is the CARLA condition and must reproduce E1's CARLA activity (8/10
heads collapsed). If either endpoint disagrees with E1, the sweep is wrong.

## Measured per alpha

- **Output activity**: the E1-style aggregate temporal activity (sum of
  per-element std across the tracked heads), normalized so alpha = 0 equals
  1.0. This is the headline curve.
- **Feature collapse**: the `hidden_state` centroid at each alpha, projected
  onto the real-to-CARLA centroid axis from E2, expressed as the fraction of
  the way from the real centroid to the CARLA centroid. Plus feature spread
  (trace of covariance) at each alpha.

## Verdict metric

The transition width: the alpha-range over which normalized output activity
falls from 0.9 to 0.1 of the real baseline, measured by linear interpolation
between sweep points. Width below ~0.2 reads as a cliff; a width spanning most
of [0, 1] reads as a gradient. The spec does not assume the outcome; the
report states the measured width and classifies it.

## Components

- `src/e4_interp.py` (new): the sweep. Reuses `collect`, `load_real_six`,
  `load_carla_six` from `src/probe_model.py` and `build_session`,
  `load_output_slices` from `src/state.py`. Mirrors `src/teardown.py`'s
  structure:
  - Heavy imports (onnxruntime, OpenCV) sit behind the `--collect` path so the
    cached path is numpy-only.
  - `--collect` runs the model over every blended sequence and writes the
    cache; default mode re-derives the analysis and figure from the cache.
- `report/e4_collected.npz` (new, committed): the per-alpha collected model
  outputs. Same reproducible-from-a-clone pattern as
  `report/teardown_collected.npz`. Expected size a few MB.
- `report/figures/e4_interpolation.png` (new): output activity and feature
  collapse versus alpha, dark theme matching the existing figures.
- `report/teardown_results.md`: gains an E4 section with the per-alpha table
  and the measured transition width.
- `tests/test_e4.py` (new): numpy-only regression test. Loads
  `report/e4_collected.npz`, recomputes the sweep analysis, asserts the
  endpoints match E1 (alpha=0 alive, alpha=1 collapsed) and asserts the
  measured transition width matches the reported verdict. Runs in CI.
- `README.md`: new `### E4` subsection under "The experiments", a repo-map
  row, and the headline table gains an E4 row.

## Data flow

```
load_real_six(Subaru) ---.
                          >-- blend at alpha --> collect() --> per-alpha dict
load_carla_six(CARLA) ---'                                          |
                                                                    v
                                          report/e4_collected.npz (cache)
                                                                    |
                  .-------------------------------------------------'
                  v
   analysis: activity curve + feature-collapse curve + transition width
                  |
                  +--> report/figures/e4_interpolation.png
                  +--> report/teardown_results.md (E4 section)
                  +--> tests/test_e4.py (regression)
```

## Error handling

- `--collect` missing input data (model, Subaru HEVC/rlog, `carla_rgb.npy`):
  fail loudly with the offending path, as `teardown.py` already does.
- Default mode with no `report/e4_collected.npz`: fall back to `--collect`
  (same behavior as `teardown.py`).
- `test_e4.py` skips if the cache is absent (matches `test_teardown.py`).

## Testing

- `tests/test_e4.py`: endpoint consistency (alpha=0 vs alpha=1) and the
  transition-width verdict, from the committed cache, numpy only.
- Manual verification: a fresh-clone run of `python -m src.e4_interp`
  reproduces the figure and table with minimal deps, as done for the teardown.

## Out of scope (YAGNI)

- No statistic-matching interpolation axis (brightness/contrast/sharpness).
- No multi-segment real anchor: Subaru alone is enough to characterize the
  transition shape. RAM stays an E1/E2/E3-only baseline.
- No localization of the collapse to a model layer: that is the separate P2
  item (E5).
