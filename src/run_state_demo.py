"""Gate 3 demo: show that recurrent state actually threads forward.

If state threading is correct, plan[0, 0, 6] after a single zero-input call
on a freshly-initialized ModelStateMirror must DIFFER from plan[0, 0, 6] on
the same ModelStateMirror after 100 zero-input frames of rollforward.

Identical values would mean features_buffer is being re-zeroed each frame
(the project-killing bug Yusuf flagged)."""

from __future__ import annotations

import sys
import time

import numpy as np

from src.state import ModelStateMirror, long_accel_t0


def zero_imgs() -> tuple[np.ndarray, np.ndarray]:
    z = np.zeros((1, 12, 128, 256), dtype=np.float32)
    return z, z.copy()


def main() -> int:
    print("Loading session + state (cold)...")
    t0 = time.perf_counter()
    state = ModelStateMirror()
    t_load = time.perf_counter() - t0
    print(f"  session ready in {t_load:.2f} s")

    img, wide = zero_imgs()

    # frame 1: zero state
    print("\n=== run #1: ZERO STATE ===")
    t0 = time.perf_counter()
    out1 = state.run(img, wide)
    print(f"  inference {1000*(time.perf_counter()-t0):.1f} ms (cold-incl)")
    a1 = long_accel_t0(out1)
    print(f"  plan[0,0,6] (long accel @ t≈0) = {a1:+.6f} m/s^2")
    print(f"  features_buffer sample (first 8 of last row): "
          f"{state.state['features_buffer'][-512:-504]}")
    print(f"  prev_desired_curv sample (last 4): {state.state['prev_desired_curv'][-4:]}")

    # roll forward 99 more zero-input frames (total 100 from init)
    print("\n=== rolling 99 more zero-input frames ===")
    accels = [a1]
    for i in range(99):
        out = state.run(img, wide)
        accels.append(long_accel_t0(out))
    a100 = accels[-1]

    # final state snapshot
    print(f"\n=== run #100: AFTER 99 ROLLS ===")
    print(f"  plan[0,0,6] (long accel @ t≈0) = {a100:+.6f} m/s^2")
    print(f"  features_buffer sample (first 8 of last row): "
          f"{state.state['features_buffer'][-512:-504]}")
    print(f"  prev_desired_curv sample (last 4): {state.state['prev_desired_curv'][-4:]}")
    print(f"  ||features_buffer||_max = {np.abs(state.state['features_buffer']).max():.6f}")
    print(f"  ||prev_desired_curv||_max = {np.abs(state.state['prev_desired_curv']).max():.6f}")

    # divergence check
    print(f"\n=== GATE 3 CHECK ===")
    delta = a100 - a1
    print(f"  accel@t0 frame 1   : {a1:+.6f}")
    print(f"  accel@t0 frame 100 : {a100:+.6f}")
    print(f"  delta              : {delta:+.6f}")
    print(f"  trajectory range   : [{min(accels):+.4f}, {max(accels):+.4f}]")
    print(f"  unique values seen : {len(set(round(x, 6) for x in accels))}")

    if abs(delta) < 1e-6:
        print("\nFAIL: state did NOT change predicted accel. Threading is broken.")
        return 1
    print("\nOK: state threading produces a different prediction at frame 100 vs frame 1.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
