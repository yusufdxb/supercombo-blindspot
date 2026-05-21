"""Step 4a — survey Town04 and Town06 for clean multi-lane highway + overpass locations.

Procedure per town:
  1. Load town
  2. Render a top-down overview (high spectator, looking straight down)
  3. Find overpass candidates programmatically: sample all 'Driving' waypoints,
     find clusters where Z > GROUND_Z_THRESHOLD AND there's another driving waypoint
     directly below within OVERPASS_XY_RADIUS at a much lower Z (the underpass road).
     These are the spots where one road crosses over another — the classic
     overpass-shadow scenario.
  4. Pick the top-5 cluster centers ranked by (lanes_on_overpass * (z_top - z_bottom))
  5. For each: render a ground-level approach shot from ~150 m back along the
     approaching lane.

Outputs PNGs to ~/Projects/phantom-braking/data/survey/{town}/{kind}_{N}.png.
"""

from __future__ import annotations

import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import carla
import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "survey"

# survey constants
GROUND_Z_THRESHOLD = 5.0   # meters — anything above this is potentially overpass
OVERPASS_XY_RADIUS = 25.0  # meters — under/over road must be this close in XY (was 6, too tight for angled overpasses)
OVERPASS_Z_DELTA = 4.0     # meters — required height diff to count as overpass
CLUSTER_MERGE_DIST = 60.0  # meters — merge nearby high-cluster centers

TOP_DOWN_MARGIN = 0.10     # 10% padding around bbox
TOP_DOWN_RES = (1600, 1600)

GROUND_RES = (1280, 720)
GROUND_FOV = 90.0          # human-perspective survey shot, wider than model FOV
GROUND_BACK_DIST = 150.0   # meters back from overpass along approach lane


# --- camera capture plumbing ---

class CameraGrabber:
    """Spawn one carla.sensor.camera.rgb, capture a single frame, save PNG."""

    def __init__(self, world: carla.World, width: int, height: int, fov: float = 90.0):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(width))
        bp.set_attribute("image_size_y", str(height))
        bp.set_attribute("fov", str(fov))
        bp.set_attribute("sensor_tick", "0.0")
        self.bp = bp
        self.world = world
        self.actor: carla.Actor | None = None
        self._image: carla.Image | None = None

    def _on_image(self, image: carla.Image) -> None:
        self._image = image

    def capture(self, transform: carla.Transform, out_path: Path,
                settle_ticks: int = 6, timeout_s: float = 5.0) -> bool:
        """Place camera at transform, tick the world a few times, save frame."""
        # detached spawn (no parent)
        self.actor = self.world.spawn_actor(self.bp, transform)
        try:
            self._image = None
            self.actor.listen(self._on_image)
            for _ in range(settle_ticks):
                self.world.tick()
            t0 = time.time()
            while self._image is None and time.time() - t0 < timeout_s:
                self.world.tick()
            if self._image is None:
                print(f"  WARN: no image at {out_path}")
                return False
            img = self._image
            buf = np.frombuffer(img.raw_data, dtype=np.uint8)
            buf = buf.reshape((img.height, img.width, 4))  # BGRA
            rgb = buf[:, :, :3][:, :, ::-1]  # to RGB
            out_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(rgb).save(out_path)
            print(f"  saved {out_path.relative_to(OUT_DIR.parent.parent)}")
            return True
        finally:
            self.actor.stop()
            self.actor.destroy()
            self.actor = None


# --- overpass discovery ---

def find_overpasses_via_waypoint_z(carla_map: carla.Map) -> list[dict]:
    """Detect overpasses where OpenDRIVE encodes elevation properly (works for
    Town04/Town05). For each high waypoint, find a low waypoint within
    OVERPASS_XY_RADIUS — that's the under-road. Capture both the overpass
    yaw (high-road heading) AND the approach yaw (low-road heading, the
    direction ego drives toward the overpass)."""
    wps = carla_map.generate_waypoints(distance=2.0)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    if not driving:
        return []

    locs = np.array([(w.transform.location.x, w.transform.location.y,
                      w.transform.location.z) for w in driving])
    yaws = np.array([w.transform.rotation.yaw for w in driving])
    is_high = locs[:, 2] > GROUND_Z_THRESHOLD
    is_low = ~is_high

    high_idx = np.where(is_high)[0]
    low_idx = np.where(is_low)[0]
    if len(high_idx) == 0 or len(low_idx) == 0:
        return []
    low_xy = locs[low_idx, :2]
    low_yaws = yaws[low_idx]
    low_z = locs[low_idx, 2]

    candidates = []
    R2 = OVERPASS_XY_RADIUS ** 2
    for i in high_idx:
        dx = low_xy[:, 0] - locs[i, 0]
        dy = low_xy[:, 1] - locs[i, 1]
        d2 = dx * dx + dy * dy
        nearby_mask = d2 < R2
        if not nearby_mask.any():
            continue
        z_low_med = float(np.median(low_z[nearby_mask]))
        if locs[i, 2] - z_low_med < OVERPASS_Z_DELTA:
            continue

        # pick the nearest low waypoint as the "ego start" surrogate, take its yaw
        nearby_low_idx = np.where(nearby_mask)[0]
        nearest = nearby_low_idx[np.argmin(d2[nearby_mask])]
        under_yaw = float(low_yaws[nearest])
        under_xy = (float(low_xy[nearest, 0]), float(low_xy[nearest, 1]))
        under_z = float(low_z[nearest])

        candidates.append({
            "xy": (float(locs[i, 0]), float(locs[i, 1])),    # high-road crossing point
            "z_top": float(locs[i, 2]),
            "z_bottom": z_low_med,
            "high_yaw": float(yaws[i]),
            "under_xy": under_xy,
            "under_yaw": under_yaw,
            "under_z": under_z,
            "n_low_nearby": int(nearby_mask.sum()),
        })

    if not candidates:
        return []

    # cluster merge by proximity
    candidates.sort(key=lambda c: c["z_top"] - c["z_bottom"], reverse=True)
    merged = []
    for c in candidates:
        cx, cy = c["xy"]
        if any(math.hypot(cx - m["center_xy"][0], cy - m["center_xy"][1]) < CLUSTER_MERGE_DIST
               for m in merged):
            continue
        merged.append({
            "center_xy": c["xy"],
            "z_top": c["z_top"],
            "z_bottom": c["z_bottom"],
            "high_yaw_deg": c["high_yaw"],
            "under_xy": c["under_xy"],
            "under_yaw_deg": c["under_yaw"],
            "under_z": c["under_z"],
            "score": (c["z_top"] - c["z_bottom"]) * c["n_low_nearby"],
            "source": "waypoint_z",
        })
    return merged


def find_overpasses_via_raycast(world: carla.World, carla_map: carla.Map,
                                grid_step: float = 8.0) -> list[dict]:
    """Fallback for towns whose OpenDRIVE topology has flat Z (Town06).

    Walk all driving waypoints, raycast downward from 50 m above each, look for
    cases where there are TWO road-surface hits (top and bottom road) along the
    same vertical line. Those are overpasses regardless of OpenDRIVE Z."""
    wps = carla_map.generate_waypoints(distance=grid_step)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    if not driving:
        return []

    candidates = []
    for wp in driving:
        loc = wp.transform.location
        start = carla.Location(x=loc.x, y=loc.y, z=loc.z + 50.0)
        end = carla.Location(x=loc.x, y=loc.y, z=loc.z - 5.0)
        try:
            hits = world.cast_ray(start, end)
        except Exception:
            continue
        # cast_ray returns hits sorted by distance from start. Filter for road-like surfaces.
        road_zs = []
        for h in hits:
            if h.label in (
                carla.CityObjectLabel.Roads,
                carla.CityObjectLabel.RoadLines,
                carla.CityObjectLabel.Sidewalks,
                carla.CityObjectLabel.Bridge,
                carla.CityObjectLabel.GuardRail,
            ):
                road_zs.append(h.location.z)
        # need at least two road-like surfaces vertically separated
        if len(road_zs) < 2:
            continue
        z_top = max(road_zs)
        z_bot = min(road_zs)
        if z_top - z_bot < OVERPASS_Z_DELTA:
            continue
        candidates.append({
            "xy": (loc.x, loc.y),
            "z_top": z_top,
            "z_bottom": z_bot,
            "yaw": float(wp.transform.rotation.yaw),
            "score": z_top - z_bot,
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    merged = []
    for c in candidates:
        cx, cy = c["xy"]
        if any(math.hypot(cx - m["center_xy"][0], cy - m["center_xy"][1]) < CLUSTER_MERGE_DIST
               for m in merged):
            continue
        merged.append({
            "center_xy": c["xy"],
            "z_top": c["z_top"],
            "z_bottom": c["z_bottom"],
            "approach_yaw_deg": c["yaw"],
            "score": c["score"],
            "n_high": 1,
            "source": "raycast",
        })
    return merged


def find_overpasses(world: carla.World, carla_map: carla.Map) -> list[dict]:
    """Use waypoint-Z method first; if that finds nothing, fall back to raycast."""
    via_z = find_overpasses_via_waypoint_z(carla_map)
    if via_z:
        return via_z
    print("    waypoint-Z method found nothing — falling back to raycast")
    return find_overpasses_via_raycast(world, carla_map)


def overpass_top_down_transform(o: dict, height: float = 80.0) -> carla.Transform:
    cx, cy = o["center_xy"]
    z = o["z_top"] + height
    return carla.Transform(
        location=carla.Location(x=cx, y=cy, z=z),
        rotation=carla.Rotation(pitch=-90.0, yaw=0.0, roll=0.0),
    )


def overpass_ground_approach_transform(o: dict, back_dist: float = GROUND_BACK_DIST,
                                       eye_height: float = 1.6) -> carla.Transform:
    """Place an eye-level camera on the UNDER-road (where ego drives), `back_dist`
    meters back from the overpass center, looking forward toward the overpass.

    The under-road's nearest waypoint to the overpass center sits effectively at
    the overpass crossing. We step back from that point along the under-road's
    heading, opposite the direction of travel."""
    ux, uy = o["under_xy"]
    yaw_rad = math.radians(o["under_yaw_deg"])
    # step BACKWARD along the under-road's direction-of-travel (negate cos/sin)
    bx = ux - back_dist * math.cos(yaw_rad)
    by = uy - back_dist * math.sin(yaw_rad)
    bz = o["under_z"] + eye_height

    # aim the camera at the overpass center (3D), letting the pitch find the
    # angle that puts the elevated road in frame
    cx, cy = o["center_xy"]
    cz = o["z_top"]
    dx = cx - bx
    dy = cy - by
    dz = cz - bz
    horiz = math.hypot(dx, dy)
    look_yaw = math.degrees(math.atan2(dy, dx))
    look_pitch = math.degrees(math.atan2(dz, horiz))
    return carla.Transform(
        location=carla.Location(x=bx, y=by, z=bz),
        rotation=carla.Rotation(pitch=look_pitch, yaw=look_yaw, roll=0.0),
    )


def town_full_map_top_down(carla_map: carla.Map, fov_deg: float = 90.0) -> carla.Transform:
    """Frame the entire driving graph from straight above, at altitude chosen so the
    map fills the FOV with a small margin."""
    wps = carla_map.generate_waypoints(distance=8.0)
    driving = [w for w in wps if w.lane_type == carla.LaneType.Driving]
    xs = np.array([w.transform.location.x for w in driving])
    ys = np.array([w.transform.location.y for w in driving])
    cx, cy = float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2)
    span_x = (xs.max() - xs.min()) * (1.0 + 2 * TOP_DOWN_MARGIN)
    span_y = (ys.max() - ys.min()) * (1.0 + 2 * TOP_DOWN_MARGIN)
    span = max(span_x, span_y)
    # for a square aspect, altitude = (span/2) / tan(fov/2)
    alt = (span / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    return carla.Transform(
        location=carla.Location(x=cx, y=cy, z=alt),
        rotation=carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
    )


# --- per-town survey ---

def survey_town(client: carla.Client, town: str, n_overpass: int = 5) -> dict:
    print(f"\n=== {town} ===")
    world = client.load_world(town)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # set permanent fair weather for reproducible screenshots
    weather = carla.WeatherParameters(
        cloudiness=0.0, precipitation=0.0, sun_altitude_angle=60.0,
        sun_azimuth_angle=0.0, fog_density=0.0,
    )
    world.set_weather(weather)
    # warm-up ticks (UE often takes a couple frames to render correctly)
    for _ in range(10):
        world.tick()

    out_dir = OUT_DIR / town
    out_dir.mkdir(parents=True, exist_ok=True)

    # discover overpasses
    cm = world.get_map()
    overpasses = find_overpasses(world, cm)
    print(f"  {len(overpasses)} overpass clusters found "
          f"(source: {overpasses[0]['source'] if overpasses else 'none'})")
    for i, o in enumerate(overpasses[:10]):
        cx, cy = o["center_xy"]
        ux, uy = o.get("under_xy", ("-", "-"))
        uyaw = o.get("under_yaw_deg", "-")
        print(f"    #{i}: high@({cx:+.1f}, {cy:+.1f}) z_top={o['z_top']:.1f} "
              f"dz={o['z_top']-o['z_bottom']:.1f} "
              f"under@({ux if isinstance(ux,str) else f'{ux:+.1f}'}, "
              f"{uy if isinstance(uy,str) else f'{uy:+.1f}'}) "
              f"under_yaw={uyaw if isinstance(uyaw,str) else f'{uyaw:+.0f}'} "
              f"score={o['score']:.1f}")

    # top-down overview — frame full map
    tdg = CameraGrabber(world, TOP_DOWN_RES[0], TOP_DOWN_RES[1], fov=90.0)
    tdg.capture(town_full_map_top_down(cm), out_dir / "00_overview_topdown.png")

    # per-overpass top-down + ground
    gg = CameraGrabber(world, GROUND_RES[0], GROUND_RES[1], fov=GROUND_FOV)
    for i, o in enumerate(overpasses[:n_overpass]):
        tdg.capture(overpass_top_down_transform(o), out_dir / f"op{i+1:02d}_topdown.png")
        gg.capture(overpass_ground_approach_transform(o),
                   out_dir / f"op{i+1:02d}_ground_approach.png")

    return {
        "town": town,
        "n_overpasses_total": len(overpasses),
        "top5": overpasses[:n_overpass],
        "out_dir": str(out_dir),
    }


def main() -> int:
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUT_DIR}")
    print(f"server: {client.get_server_version()}, client: {client.get_client_version()}")
    print(f"available maps: "
          f"{sorted(m.rsplit('/', 1)[-1] for m in client.get_available_maps())}")

    results = []
    try:
        for town in ("Town04", "Town05"):
            results.append(survey_town(client, town, n_overpass=5))
    finally:
        # back to async mode so other clients aren't surprised
        world = client.get_world()
        settings = world.get_settings()
        settings.synchronous_mode = False
        world.apply_settings(settings)

    print("\n=== SURVEY SUMMARY ===")
    for r in results:
        print(f"  {r['town']}: {r['n_overpasses_total']} overpasses found, "
              f"top-5 saved to {r['out_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
