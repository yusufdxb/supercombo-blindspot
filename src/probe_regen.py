"""Probe the regen rlog for scene characteristics: speed profile, lead presence,
lane curvature. Tells us whether this segment is suitable for the 'clear highway'
parity criterion."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.rlog import iter_events

DATA = Path(__file__).resolve().parents[1] / "data"
REGEN_RLOG = DATA / "regen" / "rlog.bz2"


def main() -> int:
    speeds = []
    lead_probs = []
    accel_t0 = []
    accel_t10 = []  # at last time index, for trajectory shape
    lane_line_probs = []
    laneline_y_at_t0 = []  # ego-frame lateral pos of left/right inner lanes
    frame_ids = []

    for ev in iter_events(REGEN_RLOG):
        which = ev.which()
        if which == "carState":
            speeds.append(ev.carState.vEgo)
        elif which == "modelV2":
            mv = ev.modelV2
            if len(mv.acceleration.x) >= 33:
                accel_t0.append(mv.acceleration.x[0])
                accel_t10.append(mv.acceleration.x[-1])
                frame_ids.append(mv.frameId)
            if hasattr(mv, "leadsV3") and len(mv.leadsV3) > 0:
                lead_probs.append(float(mv.leadsV3[0].prob))
            if hasattr(mv, "laneLineProbs"):
                lane_line_probs.append(list(mv.laneLineProbs))

    speeds = np.array(speeds)
    accel_t0 = np.array(accel_t0)
    accel_t10 = np.array(accel_t10)
    lead_probs = np.array(lead_probs)
    lane_line_probs = np.array(lane_line_probs) if lane_line_probs else None

    print(f"=== Regen segment scene profile ===")
    print(f"  modelV2 frames     : {len(accel_t0)}")
    print(f"  duration           : {len(accel_t0) / 20:.1f} s @ 20 Hz")
    print(f"\n  vEgo m/s    median {np.median(speeds):.2f}  "
          f"mean {speeds.mean():.2f}  range [{speeds.min():.2f}, {speeds.max():.2f}]")
    print(f"  vEgo mph    median {np.median(speeds)*2.237:.1f}  "
          f"mean {speeds.mean()*2.237:.1f}  range [{speeds.min()*2.237:.1f}, {speeds.max()*2.237:.1f}]")

    print(f"\n  accel.x[0]  median {np.median(accel_t0):+.3f}  "
          f"mean {accel_t0.mean():+.3f}  std {accel_t0.std():.3f}")
    print(f"  accel.x[-1] median {np.median(accel_t10):+.3f}  "
          f"mean {accel_t10.mean():+.3f}  std {accel_t10.std():.3f}")
    print(f"  |accel.x[0]|>1 m/s^2  : {(np.abs(accel_t0)>1.0).sum()} / {len(accel_t0)} "
          f"({100*(np.abs(accel_t0)>1.0).mean():.1f}%)")

    if len(lead_probs):
        print(f"\n  lead0.prob  median {np.median(lead_probs):.3f}  "
              f"max {lead_probs.max():.3f}")
        print(f"  lead0.prob > 0.5 : {(lead_probs > 0.5).sum()} / {len(lead_probs)} "
              f"({100*(lead_probs > 0.5).mean():.1f}%)")

    if lane_line_probs is not None:
        avg_lane_prob = lane_line_probs.mean(axis=0)
        print(f"\n  lane line probs (avg over frames, 4 lanes): {avg_lane_prob}")

    # find the longest contiguous "clear highway" window
    # criteria: vEgo > 22 m/s (~50 mph), |accel.x[0]| < 1, lead0.prob < 0.5
    if len(speeds) >= len(accel_t0):
        speeds_aligned = speeds[:len(accel_t0)]
    else:
        speeds_aligned = np.pad(speeds, (0, len(accel_t0) - len(speeds)),
                                constant_values=speeds[-1] if len(speeds) else 0)
    # lead_probs may be shorter / different cadence; use min of lengths
    n = min(len(speeds_aligned), len(accel_t0))
    if len(lead_probs):
        nlead = min(n, len(lead_probs))
        is_clear = (speeds_aligned[:nlead] > 22.0) & (np.abs(accel_t0[:nlead]) < 1.0) & (lead_probs[:nlead] < 0.3)
    else:
        is_clear = (speeds_aligned[:n] > 22.0) & (np.abs(accel_t0[:n]) < 1.0)

    print(f"\n  'clear highway' frames: {is_clear.sum()} / {len(is_clear)} "
          f"({100*is_clear.mean():.1f}%)")
    # longest run of True
    longest = 0
    cur = 0
    longest_start = 0
    cur_start = 0
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
    print(f"  longest clear-highway run: {longest} frames "
          f"({longest/20:.1f}s) starting at modelV2 idx {longest_start} "
          f"(frame_id {frame_ids[longest_start] if longest_start < len(frame_ids) else '-'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
