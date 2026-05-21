"""Quick overpass census across every available town."""

import sys
import carla
import numpy as np


def probe(client, town):
    world = client.load_world(town)
    cm = world.get_map()
    wps = cm.generate_waypoints(distance=4.0)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    zs = np.array([w.transform.location.z for w in driving])
    xs = np.array([w.transform.location.x for w in driving])
    ys = np.array([w.transform.location.y for w in driving])
    bridges = world.get_environment_objects(carla.CityObjectLabel.Bridge)
    n_elevated = int((zs > 5.0).sum())
    return {
        "town": town,
        "n_wps": len(driving),
        "z_min": float(zs.min()),
        "z_max": float(zs.max()),
        "z_p95": float(np.percentile(zs, 95)),
        "n_elevated_wps": n_elevated,
        "pct_elevated": 100 * n_elevated / max(1, len(driving)),
        "x_span": float(xs.max() - xs.min()),
        "y_span": float(ys.max() - ys.min()),
        "n_bridges": len(bridges),
    }


def main():
    c = carla.Client("localhost", 2000); c.set_timeout(20)
    base_towns = ["Town01", "Town02", "Town03", "Town04", "Town05",
                  "Town06", "Town07", "Town10HD"]
    print(f"{'town':<10s} {'wp':>5s} {'z_min':>6s} {'z_max':>6s} {'p95':>5s} "
          f"{'elev%':>5s} {'X_m':>6s} {'Y_m':>6s} {'bridges':>7s}")
    print("-" * 70)
    results = []
    for t in base_towns:
        try:
            r = probe(c, t)
            results.append(r)
            print(f"{r['town']:<10s} {r['n_wps']:>5d} {r['z_min']:>+6.1f} {r['z_max']:>+6.1f} "
                  f"{r['z_p95']:>5.1f} {r['pct_elevated']:>5.1f} "
                  f"{r['x_span']:>6.0f} {r['y_span']:>6.0f} {r['n_bridges']:>7d}")
        except Exception as e:
            print(f"  {t}: ERROR {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
