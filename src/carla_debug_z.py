"""Debug: report driving-waypoint Z + XY distribution per town, and inspect what
generate_waypoints actually returns."""

from __future__ import annotations

import sys

import carla
import numpy as np


def report(client: carla.Client, town: str) -> None:
    print(f"\n=== {town} ===")
    world = client.load_world(town)
    cm = world.get_map()
    for d in (4.0, 2.0):
        wps = cm.generate_waypoints(distance=d)
        driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
        print(f"\n  generate_waypoints(distance={d}): {len(wps)} total, {len(driving)} driving")
        if not driving:
            continue
        xs = np.array([w.transform.location.x for w in driving])
        ys = np.array([w.transform.location.y for w in driving])
        zs = np.array([w.transform.location.z for w in driving])
        print(f"  X range: [{xs.min():+.1f}, {xs.max():+.1f}]")
        print(f"  Y range: [{ys.min():+.1f}, {ys.max():+.1f}]")
        print(f"  Z range: [{zs.min():+.2f}, {zs.max():+.2f}]")
        print(f"  Z pct  : 50%={np.percentile(zs, 50):+.2f}  "
              f"90%={np.percentile(zs, 90):+.2f}  "
              f"95%={np.percentile(zs, 95):+.2f}  "
              f"99%={np.percentile(zs, 99):+.2f}")
        # how many are "elevated" by various thresholds vs the median Z (relative, not absolute)
        z_med = float(np.median(zs))
        for thr in (3.0, 5.0, 8.0, 10.0):
            n = int(((zs - z_med) > thr).sum())
            print(f"  z > median+{thr:.0f}m : {n} waypoints "
                  f"({100*n/len(driving):.2f}%)")

    # also list the road IDs to see how the map is segmented
    wps = cm.generate_waypoints(distance=4.0)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    road_ids = set(w.road_id for w in driving)
    print(f"\n  distinct driving road_ids: {len(road_ids)}")


def main() -> int:
    c = carla.Client("localhost", 2000)
    c.set_timeout(20.0)
    for t in ("Town04", "Town06"):
        report(c, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
