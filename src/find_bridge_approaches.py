"""Better overpass detection: for each road_id+lane_id pair, walk waypoints
sequentially. Find where Z transitions from low (<2m) to high (>5m). The
waypoints leading up to the transition are the ego's approach.

The under-road at a phantom-brake scenario is: the LOWER road at the
transition point, with the elevated continuation casting a shadow as the
overpass crosses overhead — but in CARLA towns this often is the same loop
continuing on. The metric we care about is just: how much LOW continuous road
exists before the Z rise."""

from __future__ import annotations

import sys
from collections import defaultdict

import carla
import numpy as np


def scan_town(client: carla.Client, town: str) -> None:
    print(f"\n=== {town} ===")
    world = client.load_world(town)
    cm = world.get_map()
    wps = cm.generate_waypoints(distance=2.0)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]

    # group by (road_id, lane_id). Sort by s (longitudinal pos along road).
    groups = defaultdict(list)
    for w in driving:
        groups[(w.road_id, w.lane_id)].append(w)
    for k in groups:
        groups[k].sort(key=lambda w: w.s)

    candidates = []
    for (rid, lid), seq in groups.items():
        if len(seq) < 30:
            continue
        zs = np.array([w.transform.location.z for w in seq])
        # find indices where z transitions from <2 to >5 within next ~30m
        for i in range(len(zs) - 1):
            if zs[i] < 2.0 and zs[i + 1] > 2.0:
                # find peak height ahead
                ahead = zs[i:i + 30]
                if len(ahead) == 0:
                    continue
                peak = ahead.max()
                if peak < 5.0:
                    continue
                # walk BACKWARD on same lane to find how much low road exists
                back = 0
                j = i
                while j > 0 and zs[j - 1] < 2.0:
                    back += 1
                    j -= 1
                low_runway_m = back * 2.0
                start_wp = seq[j]
                approach_wp = seq[i]
                candidates.append({
                    "road_id": rid,
                    "lane_id": lid,
                    "runway_m": low_runway_m,
                    "start_xy": (start_wp.transform.location.x, start_wp.transform.location.y),
                    "approach_xy": (approach_wp.transform.location.x, approach_wp.transform.location.y),
                    "approach_yaw": approach_wp.transform.rotation.yaw,
                    "peak_z": float(peak),
                    "n_low_before": back,
                })
                break  # first transition per lane

    candidates.sort(key=lambda c: c["runway_m"], reverse=True)
    print(f"  found {len(candidates)} lane-level z-transitions")
    print(f"\n  {'#':<3} {'road':<5} {'lane':<5} {'runway_m':>8} "
          f"{'start_xy':<22} {'approach_xy':<22} {'yaw':>5} {'peak_z':>6}")
    print("  " + "-" * 94)
    for i, c in enumerate(candidates[:15]):
        sxy = f"({c['start_xy'][0]:+.1f},{c['start_xy'][1]:+.1f})"
        axy = f"({c['approach_xy'][0]:+.1f},{c['approach_xy'][1]:+.1f})"
        print(f"  {i:<3} {c['road_id']:<5} {c['lane_id']:<5} "
              f"{c['runway_m']:>7.0f}  {sxy:<22} {axy:<22} "
              f"{c['approach_yaw']:>+5.0f} {c['peak_z']:>5.1f}")


def main() -> int:
    c = carla.Client("localhost", 2000); c.set_timeout(20)
    for t in ("Town04", "Town05", "Town12", "Town13", "Town07"):
        try:
            scan_town(c, t)
        except Exception as e:
            print(f"\n=== {t} ===\n  ERROR: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
