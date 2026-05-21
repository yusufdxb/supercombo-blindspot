"""Probe what ray-cast actually hits overhead along the op01 corridor, so the
overpass detector can filter the real bridge deck from urban clutter (traffic
gantries, signs, building eaves)."""

from __future__ import annotations

import sys

import carla
import numpy as np

from src.scenario import OVERHEAD_MAX_M, OVERHEAD_MIN_M, build_drive_path


def main() -> int:
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.load_world("Town05")
    geom = build_drive_path(world.get_map())
    path = geom["path"]
    z = geom["vertex_z"]

    print(f"{'arc_m':>7}  overhead hits (label @ clearance_m)")
    for i in range(0, len(path.xy), 3):
        x, y = path.xy[i]
        road_z = float(z[i])
        try:
            hits = world.cast_ray(
                carla.Location(x=float(x), y=float(y), z=road_z + 1.0),
                carla.Location(x=float(x), y=float(y), z=road_z + OVERHEAD_MAX_M))
        except Exception as e:
            print(f"{path.arc_length_at(i):7.1f}  cast_ray error: {e}")
            continue
        over = [(str(h.label), h.location.z - road_z) for h in hits
                if OVERHEAD_MIN_M < (h.location.z - road_z) < OVERHEAD_MAX_M]
        desc = "  ".join(f"{lab}@{c:.1f}" for lab, c in over) if over else "-"
        print(f"{path.arc_length_at(i):7.1f}  {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
