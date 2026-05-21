"""Step 4a runway survey for Town13 (and Town07 as fallback).

Two detectors run side by side:
  (1) lane Z-transitions: where a single lane's own Z rises from <2m to >5m.
      That's a bridge approach (ego climbs onto a bridge).
  (2) cross-Z underpasses: for each LOW (Z<2) waypoint, check whether there's
      another DRIVING waypoint within an XY radius at Z>5m. That's a true
      underpass — ego stays low, another road crosses overhead.

For each (2) candidate we walk the under-lane both directions and report
runway = max distance to junction, fork, dead-end, or bend (>3°/m cumulative).

Time-boxed: waypoint sampling at 2 m, hard wall-clock cap of 10 min per town.

Usage: env -u PYTHONPATH .venv/bin/python scripts/survey_town13.py
"""

from __future__ import annotations

import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import carla
import numpy as np

# ---- params ----
LOAD_TIMEOUT_S = 120.0
WALL_BUDGET_S = 600.0           # per-town hard cap
WP_DISTANCE_M = 2.0
GROUND_Z_THRESHOLD = 5.0
OVERPASS_XY_RADIUS = 20.0
OVERPASS_Z_DELTA = 4.0
CLUSTER_MERGE_DIST = 60.0
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


def find_lane_z_transitions(driving: list[carla.Waypoint]) -> list[dict]:
    """Detector (1): per-lane RELATIVE Z rise (works regardless of map origin)."""
    groups = defaultdict(list)
    for w in driving:
        groups[(w.road_id, w.lane_id)].append(w)
    for k in groups:
        groups[k].sort(key=lambda w: w.s)
    out = []
    for (rid, lid), seq in groups.items():
        if len(seq) < 30:
            continue
        zs = np.array([w.transform.location.z for w in seq])
        z_base = float(np.median(zs[:min(50, len(zs))]))  # local "ground"
        # find first index where the lane has risen >= OVERPASS_Z_DELTA above its baseline
        rel = zs - z_base
        rising = np.where(rel >= OVERPASS_Z_DELTA)[0]
        if len(rising) == 0:
            continue
        first_high = int(rising[0])
        # walk back to find where the rel-Z was last near 0 (within 1m)
        j = first_high
        while j > 0 and rel[j - 1] > 1.0:
            j -= 1
        if j == first_high:
            continue
        peak_z = float(rel.max())
        out.append({
            "road_id": rid, "lane_id": lid,
            "runway_m": (first_high - j) * WP_DISTANCE_M,
            "approach_xy": (seq[first_high].transform.location.x,
                            seq[first_high].transform.location.y),
            "approach_yaw": seq[first_high].transform.rotation.yaw,
            "peak_rel_z": peak_z,
            "z_base": z_base,
        })
    out.sort(key=lambda c: c["runway_m"], reverse=True)
    return out


def find_underpasses(driving: list[carla.Waypoint], world: carla.World,
                     cm: carla.Map) -> list[dict]:
    """Detector (2): for each waypoint, check whether ANOTHER waypoint exists
    within OVERPASS_XY_RADIUS in XY at Z >= self.Z + OVERPASS_Z_DELTA. The point
    that has elevated waypoints overhead is an under-passage point. Spatial
    bucketing keeps this tractable on the 454k-waypoint Town13."""
    locs = np.array([(w.transform.location.x, w.transform.location.y,
                      w.transform.location.z) for w in driving], dtype=np.float32)
    n = len(driving)
    print(f"     bucketing {n} waypoints...")
    cell = OVERPASS_XY_RADIUS
    buckets = defaultdict(list)
    cells_x = np.floor(locs[:, 0] / cell).astype(np.int32)
    cells_y = np.floor(locs[:, 1] / cell).astype(np.int32)
    for i in range(n):
        buckets[(int(cells_x[i]), int(cells_y[i]))].append(i)

    print(f"     {len(buckets)} occupied cells, searching neighborhoods...")
    R2 = OVERPASS_XY_RADIUS ** 2
    candidates = []
    for i in range(n):
        cx, cy = int(cells_x[i]), int(cells_y[i])
        zi = locs[i, 2]
        # check this cell + 8 neighbors
        nearby_z_top = []
        n_above = 0
        for ddx in (-1, 0, 1):
            for ddy in (-1, 0, 1):
                key = (cx + ddx, cy + ddy)
                if key not in buckets:
                    continue
                for j in buckets[key]:
                    if j == i:
                        continue
                    dx = locs[j, 0] - locs[i, 0]
                    dy = locs[j, 1] - locs[i, 1]
                    if dx * dx + dy * dy >= R2:
                        continue
                    if locs[j, 2] - zi >= OVERPASS_Z_DELTA:
                        nearby_z_top.append(float(locs[j, 2]))
                        n_above += 1
        if n_above == 0:
            continue
        wp = driving[i]
        candidates.append({
            "under_xy": (float(locs[i, 0]), float(locs[i, 1])),
            "under_z": float(zi),
            "under_yaw": float(wp.transform.rotation.yaw),
            "z_top": float(np.median(nearby_z_top)),
            "rel_z_top": float(np.median(nearby_z_top)) - float(zi),
            "wp": wp,
            "n_above": n_above,
        })

    print(f"     {len(candidates)} raw candidates; clustering...")
    # sort by relative elevation × density (prefer big overpasses with road continuation)
    candidates.sort(key=lambda c: c["rel_z_top"] * c["n_above"], reverse=True)
    merged = []
    for c in candidates:
        ux, uy = c["under_xy"]
        if any(math.hypot(ux - m["under_xy"][0], uy - m["under_xy"][1]) < CLUSTER_MERGE_DIST
               for m in merged):
            continue
        merged.append(c)
        if len(merged) >= 50:  # cap cluster count, we only need top few
            break

    print(f"     {len(merged)} merged clusters; walking runways...")
    for m in merged:
        f = walk(m["wp"], "next")
        b = walk(m["wp"], "previous")
        m["runway_fwd"] = f
        m["runway_bwd"] = b
        m["runway_best"] = max(f["distance"], b["distance"])

    merged.sort(key=lambda m: m["runway_best"], reverse=True)
    return merged


def n_lanes_per_road(driving: list[carla.Waypoint]) -> dict[int, int]:
    lanes_by_road = defaultdict(set)
    for w in driving:
        lanes_by_road[w.road_id].add(w.lane_id)
    return {rid: len(lanes_by_road[rid]) for rid in lanes_by_road}


def survey(client: carla.Client, town: str) -> dict | None:
    print(f"\n{'=' * 70}\n=== {town} ===\n{'=' * 70}")
    t_total = time.perf_counter()
    print(f"  loading world (timeout {LOAD_TIMEOUT_S:.0f}s)...")
    client.set_timeout(LOAD_TIMEOUT_S)
    t0 = time.perf_counter()
    try:
        world = client.load_world(town)
    except RuntimeError as e:
        print(f"  LOAD FAILED: {e}")
        return None
    print(f"  loaded in {time.perf_counter() - t0:.1f}s")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    for _ in range(6):
        world.tick()

    cm = world.get_map()
    print(f"  generating waypoints at {WP_DISTANCE_M}m spacing...")
    t0 = time.perf_counter()
    wps = cm.generate_waypoints(distance=WP_DISTANCE_M)
    print(f"  generated {len(wps)} waypoints in {time.perf_counter() - t0:.1f}s")
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    zs = np.array([w.transform.location.z for w in driving])
    xs = np.array([w.transform.location.x for w in driving])
    ys = np.array([w.transform.location.y for w in driving])
    print(f"  driving waypoints: {len(driving)}")
    print(f"  X span : {xs.max()-xs.min():.0f} m  ({xs.min():+.0f} .. {xs.max():+.0f})")
    print(f"  Y span : {ys.max()-ys.min():.0f} m  ({ys.min():+.0f} .. {ys.max():+.0f})")
    z_med = float(np.median(zs))
    print(f"  Z range: [{zs.min():+.2f}, {zs.max():+.2f}]  "
          f"median={z_med:+.2f}  p95={np.percentile(zs, 95):+.2f}  p99={np.percentile(zs, 99):+.2f}")
    z_spread = float(zs.max() - zs.min())
    print(f"  Z spread: {z_spread:.2f}m (relative elevation range — if <5m, no overpasses)")

    if z_spread < OVERPASS_Z_DELTA:
        print("\n  no elevated waypoints — town has no overpasses in OpenDRIVE topology.")
        bridges = world.get_environment_objects(carla.CityObjectLabel.Bridge)
        print(f"  Bridge env objects: {len(bridges)} (if non-zero, visual bridges exist but topology is flat)")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        return {"town": town, "elev_wps": 0, "underpasses": [], "transitions": []}

    # lane-count summary for highway test
    lpr = n_lanes_per_road(driving)
    max_lanes = max(lpr.values()) if lpr else 0
    n_roads_3lane = sum(1 for n in lpr.values() if n >= 3)
    print(f"  multi-lane: max lanes-per-road {max_lanes}; roads with >=3 lanes: {n_roads_3lane}")

    # (1) lane Z-transitions
    print("\n  -- detector (1): lane Z-transitions (bridge approaches) --")
    transitions = find_lane_z_transitions(driving)
    print(f"     {len(transitions)} transitions")
    for c in transitions[:10]:
        print(f"     road={c['road_id']:<4} lane={c['lane_id']:<3} runway={c['runway_m']:>4.0f}m "
              f"at ({c['approach_xy'][0]:+7.1f}, {c['approach_xy'][1]:+7.1f}) "
              f"yaw={c['approach_yaw']:>+4.0f}  peak_rel_z={c['peak_rel_z']:.1f}")

    # (2) true underpasses
    print("\n  -- detector (2): true underpasses (low road + high road overhead) --")
    if time.perf_counter() - t_total > WALL_BUDGET_S - 60:
        print(f"     SKIPPED — wall budget exhausted")
        underpasses = []
    else:
        underpasses = find_underpasses(driving, world, cm)
    print(f"     {len(underpasses)} merged underpasses (top 10 by best runway):")
    print(f"     {'#':<3} {'under_xy':<24} {'z_top':>5} {'fwd':>5} {'fwd_end':<10} "
          f"{'bwd':>5} {'bwd_end':<10} {'best':>5}")
    print(f"     {'-' * 76}")
    for i, m in enumerate(underpasses[:10]):
        ux, uy = m["under_xy"]
        f = m["runway_fwd"]
        b = m["runway_bwd"]
        print(f"     {i:<3} ({ux:+7.1f},{uy:+7.1f})        "
              f"{m['z_top']:>5.1f} {f['distance']:>5.0f} {f['reason']:<10} "
              f"{b['distance']:>5.0f} {b['reason']:<10} {m['runway_best']:>5.0f}")

    settings.synchronous_mode = False
    world.apply_settings(settings)

    return {
        "town": town,
        "elev_wps": int((zs > GROUND_Z_THRESHOLD).sum()),
        "max_lanes": max_lanes,
        "underpasses": underpasses,
        "transitions": transitions,
    }


def main() -> int:
    client = carla.Client("localhost", 2000)
    client.set_timeout(LOAD_TIMEOUT_S)
    print(f"server: {client.get_server_version()}")

    results = []
    # skip Town07 (load failed previously) — focus the budget on Town13 with the fixed detector
    for town in ("Town13",):
        r = survey(client, town)
        if r is not None:
            results.append(r)

    print("\n" + "=" * 70)
    print("=== DECISION SUMMARY ===")
    print("=" * 70)
    if not results:
        print("  No towns surveyed successfully.")
        print("  Fallback: two-phase init on Town05 op01 (108 m, fwd-to-junction).")
        return 0

    for r in results:
        best = (r["underpasses"][0]["runway_best"] if r["underpasses"] else 0)
        verdict = (
            ">=200m: USE DIRECTLY" if best >= 200 else
            "150-199m: maybe two-phase" if best >= 150 else
            "<150m: two-phase init on Town05 op01"
        )
        print(f"  {r['town']:<8} max_lanes={r.get('max_lanes', '-')}  "
              f"underpasses={len(r['underpasses'])}  "
              f"best_runway={best:.0f}m  -> {verdict}")
        # top-3 candidates with location
        for i, m in enumerate(r["underpasses"][:3]):
            ux, uy = m["under_xy"]
            print(f"            #{i}: ({ux:+.0f}, {uy:+.0f}) rel_z_top={m['rel_z_top']:.1f}m "
                  f"runway={m['runway_best']:.0f}m  yaw={m['under_yaw']:+.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
