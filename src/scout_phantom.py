"""Path (a) scouting — scan v0.9.7-regenerated rlogs of real comma drives for
phantom-brake-shaped events.

A regen rlog is openpilot v0.9.7's modeld re-run on a real recorded drive, so
`modelV2.acceleration.x[0]` is supercombo's planned accel@t0 on real imagery and
`modelV2.leadsV3[0].prob` is its own lead belief. carState carries the real car
(vEgo, steeringAngleDeg).

A phantom brake = the model commands a sustained deceleration with (a) no lead it
believes in and (b) a roughly straight road (so it is not braking for a curve).
This script reports every such contiguous window. It does not yet rule out speed
limits / map data, so a window is a *candidate* to confirm against video.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from src.rlog import iter_events

PROBES = Path(__file__).resolve().parents[1] / "data" / "probes"

BRAKE_MS2 = -1.0        # accel@t0 at/below this is a brake command
LEAD_MAX = 0.30         # model lead probability below this = "no lead"
STRAIGHT_DEG = 5.0      # |steering angle| below this = roughly straight
MIN_WIN_FRAMES = 8      # >= 0.4 s, ignore single-frame blips
SPEED_MIN_MS = 10.0


def _aligned_tracks(path: Path) -> dict:
    """Return per-modelV2-frame arrays: accel@t0, lead prob, vEgo, |steer|, frameId.
    carState (~100 Hz) is nearest-time matched to each modelV2 sample."""
    cs_t, cs_v, cs_s = [], [], []
    mv_t, accel, lead, fid = [], [], [], []
    for ev in iter_events(path):
        try:
            w = ev.which()
        except Exception:
            continue
        t = ev.logMonoTime
        if w == "carState":
            cs_t.append(t)
            cs_v.append(float(ev.carState.vEgo))
            cs_s.append(abs(float(ev.carState.steeringAngleDeg)))
        elif w == "modelV2":
            mv = ev.modelV2
            if len(mv.acceleration.x) >= 33:
                mv_t.append(t)
                accel.append(float(mv.acceleration.x[0]))
                lead.append(float(mv.leadsV3[0].prob) if len(mv.leadsV3) else 0.0)
                fid.append(int(mv.frameId))

    accel = np.array(accel)
    if len(accel) == 0:
        return {"n": 0}
    cs_t = np.array(cs_t)
    mv_t = np.array(mv_t)
    if len(cs_t):
        j = np.searchsorted(cs_t, mv_t).clip(0, len(cs_t) - 1)
        vEgo = np.array(cs_v)[j]
        steer = np.array(cs_s)[j]
    else:
        vEgo = np.full(len(accel), SPEED_MIN_MS + 1.0)
        steer = np.zeros(len(accel))
    return {"n": len(accel), "accel": accel, "lead": np.array(lead),
            "vEgo": vEgo, "steer": steer, "frame_id": np.array(fid)}


def find_windows(tr: dict) -> list[dict]:
    """Contiguous runs of: braking, no lead, moving. Reported with curve context."""
    accel, lead, vEgo, steer = tr["accel"], tr["lead"], tr["vEgo"], tr["steer"]
    fid = tr["frame_id"]
    flag = (accel <= BRAKE_MS2) & (lead < LEAD_MAX) & (vEgo > SPEED_MIN_MS)
    windows = []
    i = 0
    n = len(flag)
    while i < n:
        if not flag[i]:
            i += 1
            continue
        j = i
        while j < n and flag[j]:
            j += 1
        if j - i >= MIN_WIN_FRAMES:
            sl = slice(i, j)
            windows.append({
                "i0": i, "i1": j, "frames": j - i, "dur_s": (j - i) / 20.0,
                "frame_id0": int(fid[i]), "frame_id1": int(fid[j - 1]),
                "accel_min": float(accel[sl].min()),
                "accel_mean": float(accel[sl].mean()),
                "lead_max": float(lead[sl].max()),
                "steer_mean": float(steer[sl].mean()),
                "steer_max": float(steer[sl].max()),
                "mph0": float(vEgo[i] * 2.237),
                "mph1": float(vEgo[j - 1] * 2.237),
            })
        i = j
    return windows


def main() -> int:
    files = sorted(PROBES.glob("*_regen.rlog.bz2"))
    if not files:
        print(f"no regen rlogs in {PROBES}")
        return 1

    print(f"Phantom-brake window scout: accel@t0 <= {BRAKE_MS2}, lead < {LEAD_MAX}, "
          f"moving, >= {MIN_WIN_FRAMES} frames\n")
    all_windows = []
    for p in files:
        tr = _aligned_tracks(p)
        label = p.stem.replace("_regen.rlog", "")
        if tr["n"] == 0:
            print(f"  {label:<10} (no modelV2)")
            continue
        wins = find_windows(tr)
        nolead = int(((tr["accel"] <= BRAKE_MS2) & (tr["lead"] < LEAD_MAX)).sum())
        print(f"  {label:<10} {tr['n']:>4} frames  accel_min {tr['accel'].min():>7.3f}  "
              f"no-lead brake frames {nolead:>4}  windows {len(wins)}")
        for w in wins:
            all_windows.append((label, w))

    print(f"\n=== SUSTAINED NO-LEAD BRAKE WINDOWS ({len(all_windows)}) ===")
    if not all_windows:
        print("  none.")
        return 0
    print(f"  {'seg':<8} {'frameId':>14} {'dur_s':>6} {'accel_min':>10} "
          f"{'accel_mean':>11} {'steer_mean':>11} {'mph 0->1':>12} {'verdict':>16}")
    for label, w in sorted(all_windows, key=lambda x: x[1]["accel_min"]):
        straight = w["steer_max"] < STRAIGHT_DEG
        verdict = "straight-road!" if straight else "curve (steer)"
        print(f"  {label:<8} {w['frame_id0']:>6}-{w['frame_id1']:<7} {w['dur_s']:>6.1f} "
              f"{w['accel_min']:>10.3f} {w['accel_mean']:>11.3f} "
              f"{w['steer_mean']:>11.2f} {w['mph0']:>5.0f}->{w['mph1']:<5.0f} "
              f"{verdict:>16}")
    straight = [w for _, w in all_windows if w["steer_max"] < STRAIGHT_DEG]
    print(f"\n  {len(straight)} of {len(all_windows)} windows are on a straight road "
          f"(steer < {STRAIGHT_DEG} deg) — those are the phantom-brake candidates to "
          f"confirm against video.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
