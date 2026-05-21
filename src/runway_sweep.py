"""Walk waypoints from every detected overpass under-road in Town04 + Town05.
Report runway in each direction. Pick the overpass with longest continuous
straight under-road."""

from __future__ import annotations

import math
import sys

import carla
import numpy as np

# Overpasses found earlier (high_xy, under_xy, under_yaw_deg)
OVERPASSES = {
    "Town04": [
        {"name": "op01", "high": (+40.4, +6.2), "under": (+15.5, +6.0), "under_yaw": -90.0, "z_top": 11.0},
        {"name": "op02", "high": (-11.8, +37.5), "under": (-12.4, +38.2), "under_yaw": +90.0, "z_top": 10.8},
    ],
    "Town05": [
        {"name": "op01", "high": (-224.6, -119.5), "under": (-224.5, -95.2), "under_yaw": -180.0, "z_top": 10.0},
        {"name": "op02", "high": (-247.3, -62.4),  "under": (-261.4, -69.8), "under_yaw": -61.0,  "z_top": 10.0},
        {"name": "op03", "high": (-229.0, -4.6),   "under": (-228.4, -3.9),  "under_yaw": +180.0, "z_top": 10.0},
        {"name": "op04", "high": (-247.1, +53.4),  "under": (-265.1, +53.8), "under_yaw": -91.0,  "z_top": 10.0},
        {"name": "op05", "high": (-226.4, +110.4), "under": (-226.1, +94.8), "under_yaw": +0.0,   "z_top": 9.9},
    ],
}

STEP_M = 2.0
MAX_STEPS = 500
YAW_PER_M_LIMIT_DEG = 3.0


def walk(start: carla.Waypoint, direction: str) -> dict:
    cur = start
    distance = 0.0
    last_yaw = cur.transform.rotation.yaw
    for _ in range(MAX_STEPS):
        nexts = cur.next(STEP_M) if direction == "next" else cur.previous(STEP_M)
        if not nexts:
            return {"distance": distance, "reason": "dead_end"}
        if len(nexts) > 1:
            return {"distance": distance, "reason": "fork"}
        nxt = nexts[0]
        if nxt.is_junction:
            return {"distance": distance, "reason": "junction"}
        yaw = nxt.transform.rotation.yaw
        d_yaw = ((yaw - last_yaw + 180.0) % 360.0) - 180.0
        if abs(d_yaw) / STEP_M > YAW_PER_M_LIMIT_DEG:
            return {"distance": distance, "reason": "bend"}
        distance += STEP_M
        cur = nxt
        last_yaw = yaw
    return {"distance": distance, "reason": "max"}


def probe_town(client: carla.Client, town: str) -> None:
    print(f"\n=== {town} ===")
    world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    for _ in range(6):
        world.tick()
    cm = world.get_map()

    print(f"  {'name':<5} {'under(x,y)':<22} {'fwd_m':>7} {'fwd_end':<10} "
          f"{'bwd_m':>7} {'bwd_end':<10} {'best_m':>7}")
    print("  " + "-" * 78)
    for op in OVERPASSES[town]:
        ux, uy = op["under"]
        wp = cm.get_waypoint(carla.Location(x=ux, y=uy, z=0.0),
                             project_to_road=True, lane_type=carla.LaneType.Driving)
        f = walk(wp, "next")
        b = walk(wp, "previous")
        best = max(f["distance"], b["distance"])
        print(f"  {op['name']:<5} ({ux:+7.1f}, {uy:+7.1f})   "
              f"{f['distance']:>6.0f}  {f['reason']:<10} "
              f"{b['distance']:>6.0f}  {b['reason']:<10} {best:>6.0f}")

    settings.synchronous_mode = False
    world.apply_settings(settings)


def main() -> int:
    c = carla.Client("localhost", 2000)
    c.set_timeout(20.0)
    for t in ("Town04", "Town05"):
        probe_town(c, t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
