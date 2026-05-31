# E6 Monitor Deployment Results

## 1. What the E6 monitor is

The E6 detector is a **single O(D) statistic** computed once per forward pass
of openpilot's supercombo model.  It measures the rolling temporal spread of
the 512-D recurrent hidden state over the last W=30 frames:

```
spread(t) = trace( Cov(hidden[t-W+1 .. t]) )
           = sum_{d=0..D-1}  Var_d   (population variance, ddof=0)
```

Calibrated threshold: **0.078873** (1st percentile of the real-driving spread
distribution on the subaru + ram corpus).  A spread below this threshold fires
an OOD alert; it is not a neural net.

Source: `src/e6_detector.py::rolling_spread` (lines 16-23).

---

## 2. Build and run

```bash
cd deploy/cpp
make          # compiles e6_bench and runs it immediately
# or: g++ -std=c++17 -O2 -o e6_bench bench_main.cpp
```

**Verified compile + run output (g++ 11.4.0, Ubuntu 22.04, x86):**

```
g++ -std=c++17 -O2 -Wall -Wextra -pedantic -o e6_bench bench_main.cpp
./e6_bench

=== Numerical agreement (Python-faithful batch vs C++ ring-buffer) ===
  Frames checked (post warm-up): 171 / 200
  Max absolute difference:       3.4106e-13
  Max relative difference:       5.2225e-16
  PASS (< 1e-4 threshold)

=== Latency microbenchmark (W=30, D=512, N=150000) ===
  Platform: x86 -- AMD Ryzen 9 9900X (mewtwo)
  NOTE: Jetson Orin NX 16 GB latency is HW-UNVERIFIED,
        pending a CaresLab session. Do NOT extrapolate from these numbers.

  Mean   latency per frame:    0.405 us
  Median latency per frame:    0.400 us
  p99    latency per frame:    0.410 us

  20 Hz control budget  : 50000 us/tick
  supercombo inference  : ~2000 us (reference, Jetson Orin NX)
  Monitor / budget      : 0.0008% (mean)
  Monitor / supercombo  : 0.0202% (mean, x86 vs Jetson -- not apples-to-apples)
```

---

## 3. Numerical agreement: Python vs C++

| Metric | Value |
|---|---|
| Frames compared (post warm-up) | 171 / 200 |
| Max absolute difference | 3.41e-13 |
| Max relative difference | 5.22e-16 |
| PASS threshold | < 1e-4 |
| Verdict | PASS |

The difference is pure floating-point rounding from reordering the summation
(the ring-buffer accumulates incrementally; the Python reference recomputes
batch).  It is 9 orders of magnitude below the 1e-4 tolerance, well within
any operational budget.

The Python-vs-C++ comparison is also exercised by three CI-safe pytest tests
in `tests/test_e6_cpp.py` that skip cleanly when g++ is not on PATH.

---

## 4. Latency: control-budget comparison (x86 -- AMD Ryzen 9 9900X)

| Stat | Latency (us) | % of 20 Hz budget (50 ms) |
|---|---|---|
| Mean | 0.405 | 0.0008% |
| Median | 0.400 | 0.0008% |
| p99 | 0.410 | 0.0008% |

The monitor adds **negligible overhead**: even the worst observed frame
(p99 = 0.41 us) consumes 0.0008% of the 50 ms control-loop budget.  It is
also immaterial relative to supercombo's own inference time (approx. 2 ms on
Jetson Orin NX), which would make the monitor roughly 0.02% of that reference
figure (not a fair comparison -- see caveat below).

**HARDWARE CAVEAT: All latency numbers above are x86 measurements on mewtwo
(AMD Ryzen 9 9900X, Ubuntu 22.04).  Jetson Orin NX 16 GB (Cortex-A78AE cores,
no SIMD equivalent to AVX2) is HW-UNVERIFIED.  Real Orin latency has NOT been
measured; a CaresLab session is required before quoting any Orin number.  The
O(D) ring-buffer algorithm has no platform-specific intrinsics and should be
portable, but actual timing on Orin may differ substantially.**

---

## 5. ROS2 deployment path

The E6 monitor is already registered as a production ROS2 adapter in
`~/Projects/policy-health-monitor` (phm_detectors package):

```
policy-health-monitor/src/phm_detectors/phm_detectors/_core.py
  class RecurrentTemporalSpreadAdapter(Detector)
```

That adapter (lines 453-645) wraps the same rolling_spread / calibrate_threshold
math from `phm_core.calibration` (itself ported from `src/e6_detector.py`).
It accepts `RecurrentSpreadSample(topic, embedding)` objects, maintains its own
ring buffer of embedding frames, applies hysteresis (min_consecutive=2), and
emits `DetectorVerdictData` to `/phm/verdicts` on every inference tick.

For an on-device C++ modeld consumer, the slot-in path is:

1. modeld publishes `hidden_state` (512-D float32) after each supercombo
   inference.
2. A C++ ROS2 subscriber (or a zero-copy shared-memory consumer) receives the
   vector and calls `monitor.update(hidden_state_ptr)` once per frame.
3. The returned `double` is compared against `e6::E6_THRESHOLD` (0.078873).
4. If `spread < E6_THRESHOLD` for `min_consecutive` consecutive frames, the
   node publishes an OOD warning to `/phm/verdicts` or triggers an estop.

The C++ header `deploy/cpp/e6_monitor.hpp` is self-contained (no external
dependencies beyond the C++17 STL) and can be included directly in any
ROS2 C++ package.

**Do not duplicate the detector node**: the Python `RecurrentTemporalSpreadAdapter`
in phm_detectors is the canonical in-the-loop node for the current ROS2 stack.
The C++ header is provided for latency-critical C++ consumers or future porting.

---

## 6. Regression test tally

```
pytest -q
...
215 passed, 5 skipped, 2 warnings in 206.77s (0:03:26)
```

Zero failures.  The 5 skipped tests require live CARLA or hardware that is
absent in CI (unchanged from before this task).  The 3 new tests in
`tests/test_e6_cpp.py` all pass when g++ is available and skip cleanly otherwise.

---

## Summary

| Item | Result |
|---|---|
| Numerical max-abs-diff (Python vs C++) | 3.41e-13 (PASS) |
| Latency mean / median / p99 (x86) | 0.405 / 0.400 / 0.410 us |
| % of 20 Hz control budget (mean) | 0.0008% |
| Jetson Orin NX latency | HW-UNVERIFIED (CaresLab pending) |
| Regression suite | 215 passed, 5 skipped, 0 failed |

**Deployability verdict (honest):** The C++ E6 monitor is correct (bit-faithful
to the Python reference within floating-point rounding), portable (C++17 STL
only), and confirmed negligible on x86.  It is ready to embed in a C++ ROS2
node or modeld consumer.  Real-time suitability on Jetson Orin NX 16 GB has
NOT been verified; a CaresLab hardware session is required before any on-device
latency claim can be made.
