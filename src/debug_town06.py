"""Town06 has flat OpenDRIVE Z. Probe alternative methods:
  1. Spawn points Z distribution (CARLA places these at actual road heights)
  2. cast_ray on a vertical line through known overpass spots
  3. Labelled mesh enumeration
"""

import sys
import carla
import numpy as np


def main() -> int:
    c = carla.Client("localhost", 2000); c.set_timeout(20)
    world = c.load_world("Town06")
    cm = world.get_map()

    # (1) spawn points
    spawns = cm.get_spawn_points()
    zs = np.array([sp.location.z for sp in spawns])
    xs = np.array([sp.location.x for sp in spawns])
    ys = np.array([sp.location.y for sp in spawns])
    print(f"spawn points: {len(spawns)}")
    print(f"  Z range : [{zs.min():+.2f}, {zs.max():+.2f}]  "
          f"median {np.median(zs):+.2f}  90%={np.percentile(zs, 90):+.2f}  "
          f"99%={np.percentile(zs, 99):+.2f}")
    elevated_spawns = [sp for sp in spawns if sp.location.z > 5.0]
    print(f"  elevated (Z>5m): {len(elevated_spawns)}")
    for sp in elevated_spawns[:10]:
        loc = sp.location
        print(f"    ({loc.x:+.1f}, {loc.y:+.1f}, {loc.z:.2f})  yaw={sp.rotation.yaw:+.0f}")

    # (2) cast_ray test
    print("\ncast_ray probe at a few coords...")
    test_pts = [
        (0.0, 0.0),
        (100.0, -20.0),
        (200.0, 50.0),
        (300.0, 50.0),
        (-50.0, 100.0),
        (400.0, 100.0),
    ]
    for x, y in test_pts:
        start = carla.Location(x=x, y=y, z=100.0)
        end = carla.Location(x=x, y=y, z=-10.0)
        try:
            hits = world.cast_ray(start, end)
        except Exception as e:
            print(f"  ({x:+.0f}, {y:+.0f}): cast_ray ERR: {e}")
            continue
        if not hits:
            print(f"  ({x:+.0f}, {y:+.0f}): no hits")
            continue
        print(f"  ({x:+.0f}, {y:+.0f}): {len(hits)} hits")
        for h in hits[:5]:
            print(f"      z={h.location.z:+.2f}  label={h.label!r}")

    # (3) labelled environment objects
    print("\nenv objects (sample of bridge-like labels)...")
    for lbl_name in ("Bridge", "Roads"):
        lbl = getattr(carla.CityObjectLabel, lbl_name, None)
        if lbl is None:
            print(f"  no enum member: {lbl_name}")
            continue
        objs = world.get_environment_objects(lbl)
        print(f"  {lbl_name}: {len(objs)} objects")
        if objs and lbl_name == "Bridge":
            for o in objs[:5]:
                bb = o.bounding_box
                loc = bb.location
                ext = bb.extent
                print(f"    bridge at ({loc.x:+.1f}, {loc.y:+.1f}, {loc.z:+.2f}) "
                      f"size ({2*ext.x:.0f}x{2*ext.y:.0f}x{2*ext.z:.1f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
