"""Town13 pilot location: underpass #0 at (+2418, +4570), 15.1m overhead, 784m runway, under_yaw=-54°.

Capture:
  - top-down overview of the underpass and approach corridor
  - ground approach view from 150m back, eye-level, looking at overpass
  - shadow audit: sun_alt=30°, sun_az perpendicular to ego heading, narrow-cam 50m before overpass
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image

TOWN = "Town13"
UNDER_XY = (2417.6, 4570.2)
UNDER_YAW = -54.0
REL_Z_TOP = 15.1

OUT = Path(__file__).resolve().parents[1] / "data" / "survey" / TOWN / "pilot"

CLIENT_TIMEOUT = 120.0
NARROW_W, NARROW_H = 1928, 1208
NARROW_FOV = 73.0
GROUND_W, GROUND_H = 1280, 720
GROUND_FOV = 90.0

EGO_AUDIT_DIST = 50.0       # meters before overpass for shadow audit
EGO_APPROACH_DIST = 150.0   # meters before for the "ground approach" reference frame
EGO_EYE_HEIGHT = 1.4


def capture(world: carla.World, transform: carla.Transform, w: int, h: int, fov: float,
            out_path: Path) -> None:
    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(w))
    bp.set_attribute("image_size_y", str(h))
    bp.set_attribute("fov", str(fov))
    bp.set_attribute("sensor_tick", "0.0")
    cam = world.spawn_actor(bp, transform)
    ref = {}
    cam.listen(lambda im: ref.setdefault("im", im))
    try:
        for _ in range(12):
            world.tick()
        t0 = time.time()
        while "im" not in ref and time.time() - t0 < 6.0:
            world.tick()
        if "im" not in ref:
            print(f"  WARN: no frame at {out_path.name}")
            return
        im = ref["im"]
        arr = np.frombuffer(im.raw_data, dtype=np.uint8).reshape(im.height, im.width, 4)
        Image.fromarray(arr[:, :, :3][:, :, ::-1]).save(out_path)
        print(f"  saved: {out_path.relative_to(out_path.parents[3])}")
    finally:
        cam.stop()
        cam.destroy()


def step_back(xy, yaw_deg: float, dist: float) -> tuple[float, float]:
    """move `dist` meters opposite the direction-of-travel."""
    r = math.radians(yaw_deg)
    return xy[0] - dist * math.cos(r), xy[1] - dist * math.sin(r)


def find_ego_z(world, x, y) -> float:
    wp = world.get_map().get_waypoint(carla.Location(x=x, y=y, z=200.0),
                                      project_to_road=True,
                                      lane_type=carla.LaneType.Driving)
    return wp.transform.location.z


def main() -> int:
    c = carla.Client("localhost", 2000)
    c.set_timeout(CLIENT_TIMEOUT)
    print(f"server: {c.get_server_version()}")
    print(f"loading {TOWN} (this map is large, ~25s load)...")
    t0 = time.perf_counter()
    world = c.load_world(TOWN)
    print(f"loaded in {time.perf_counter()-t0:.1f}s")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    for _ in range(10):
        world.tick()

    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. Top-down overview at the pilot underpass ----
    print("\n--- top-down overview ---")
    ego_z = find_ego_z(world, *UNDER_XY)
    print(f"  under-road Z at ({UNDER_XY[0]:+.1f}, {UNDER_XY[1]:+.1f}): {ego_z:.2f} m")
    weather_neutral = carla.WeatherParameters(
        cloudiness=0.0, precipitation=0.0, sun_altitude_angle=60.0, sun_azimuth_angle=0.0,
        fog_density=0.0,
    )
    world.set_weather(weather_neutral)
    capture(world, carla.Transform(
        location=carla.Location(x=UNDER_XY[0], y=UNDER_XY[1], z=ego_z + 120.0),
        rotation=carla.Rotation(pitch=-90.0, yaw=-90.0, roll=0.0),
    ), 1024, 1024, 90.0, OUT / "00_topdown.png")

    # ---- 2. Ground approach reference, neutral weather ----
    print("\n--- ground approach reference (150m back, neutral midday) ---")
    bx, by = step_back(UNDER_XY, UNDER_YAW, EGO_APPROACH_DIST)
    bz = find_ego_z(world, bx, by) + EGO_EYE_HEIGHT
    cx, cy = UNDER_XY
    dx, dy = cx - bx, cy - by
    dz = ego_z + REL_Z_TOP - bz
    look_yaw = math.degrees(math.atan2(dy, dx))
    look_pitch = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    print(f"  ego pos: ({bx:+.1f}, {by:+.1f}, {bz:.1f})  look yaw={look_yaw:+.0f} pitch={look_pitch:+.1f}")
    capture(world, carla.Transform(
        location=carla.Location(x=bx, y=by, z=bz),
        rotation=carla.Rotation(pitch=look_pitch, yaw=look_yaw, roll=0.0),
    ), GROUND_W, GROUND_H, GROUND_FOV, OUT / "01_ground_approach_150m.png")

    # ---- 3. Shadow audit: narrow cam, sun_alt=30°, sun_az perpendicular ----
    print("\n--- shadow audit (narrow cam @ 50m, sun_alt=30°, sun_az perpendicular) ---")
    # under_yaw = -54° means ego faces -54°. Perpendicular = -54° + 90° = +36°
    sun_az_perpendicular = (UNDER_YAW + 90.0) % 360.0
    weather_audit = carla.WeatherParameters(
        cloudiness=0.0, precipitation=0.0,
        sun_altitude_angle=30.0, sun_azimuth_angle=sun_az_perpendicular,
        fog_density=0.0,
    )
    world.set_weather(weather_audit)
    bx, by = step_back(UNDER_XY, UNDER_YAW, EGO_AUDIT_DIST)
    bz = find_ego_z(world, bx, by) + EGO_EYE_HEIGHT
    print(f"  ego pos: ({bx:+.1f}, {by:+.1f}, {bz:.1f})  facing yaw={UNDER_YAW:+.0f}")
    print(f"  sun: altitude=30°, azimuth={sun_az_perpendicular:.0f}° (perpendicular to ego)")
    capture(world, carla.Transform(
        location=carla.Location(x=bx, y=by, z=bz),
        rotation=carla.Rotation(pitch=0.0, yaw=UNDER_YAW, roll=0.0),
    ), NARROW_W, NARROW_H, NARROW_FOV, OUT / "02_shadow_audit_alt30_az_perp.png")

    # back to async
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print("\ndone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
