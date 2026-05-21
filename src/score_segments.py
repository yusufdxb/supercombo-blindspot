"""Score candidate regen rlogs for highway clarity to pick the parity segment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.rlog import iter_events

PROBES = Path(__file__).resolve().parents[1] / "data" / "probes"


def score(path: Path) -> dict:
    speeds, accels, lead_probs = [], [], []
    for ev in iter_events(path):
        try:
            w = ev.which()
        except Exception:
            continue
        if w == "carState":
            speeds.append(ev.carState.vEgo)
        elif w == "modelV2":
            mv = ev.modelV2
            if len(mv.acceleration.x) >= 33:
                accels.append(mv.acceleration.x[0])
            if len(mv.leadsV3) > 0:
                lead_probs.append(float(mv.leadsV3[0].prob))
    speeds = np.array(speeds)
    accels = np.array(accels)
    lead_probs = np.array(lead_probs) if lead_probs else np.array([])

    n = min(len(speeds), len(accels), len(lead_probs)) if len(lead_probs) else min(len(speeds), len(accels))
    if n == 0:
        return {"label": path.stem, "n": 0}
    s = speeds[:n] * 2.237  # mph
    a = accels[:n]
    if len(lead_probs):
        lp = lead_probs[:n]
        is_clear = (s > 45.0) & (np.abs(a) < 1.0) & (lp < 0.3)
    else:
        is_clear = (s > 45.0) & (np.abs(a) < 1.0)

    # longest contiguous clear-highway run
    longest = 0
    cur = 0
    cur_start = 0
    longest_start = 0
    for i, v in enumerate(is_clear):
        if v:
            if cur == 0:
                cur_start = i
            cur += 1
            if cur > longest:
                longest = cur
                longest_start = cur_start
        else:
            cur = 0

    return {
        "label": path.stem,
        "n": n,
        "mph_med": float(np.median(s)),
        "mph_range": (float(s.min()), float(s.max())),
        "accel_std": float(a.std()),
        "lead_pct": float(100 * (lp > 0.5).mean()) if len(lead_probs) else None,
        "clear_pct": float(100 * is_clear.mean()),
        "longest_run_s": longest / 20.0,
        "longest_run_start_idx": longest_start,
    }


def main() -> int:
    files = sorted(PROBES.glob("*_regen.rlog.bz2"))
    print(f"{'label':<28s}  {'n':>5s}  {'mph_med':>8s}  {'lead%':>6s}  {'clr%':>6s}  {'longest_s':>10s}  {'start_idx':>10s}")
    for p in files:
        r = score(p)
        if r["n"] == 0:
            print(f"  {r['label']}: no modelV2")
            continue
        print(f"{r['label']:<28s}  {r['n']:>5d}  {r['mph_med']:>8.1f}  "
              f"{(r['lead_pct'] or 0.0):>6.1f}  {r['clear_pct']:>6.1f}  "
              f"{r['longest_run_s']:>10.1f}  {r['longest_run_start_idx']:>10d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
