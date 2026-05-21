"""Step 4a-final pre-checks:
  A. Runway verification — walk waypoints back from op03 under-road, report
     max continuous straight road in each direction (yaw-change + is_junction
     terminate the walk).
  B. Building shadow audit — render the narrow-camera view 50 m before op03
     at sun_altitude=30°, sun_azimuth perpendicular to ego heading, save to
     data/survey/Town05/audit/ for visual inspection."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image

TOWN = "Town05"
OP_CENTER = (-229.0, -4.6, 10.0)
OP_UNDER = (-228.4, -3.9)        # under-road point
OP_UNDER_YAW_DEG = 180.0          # direction-of-travel of under-road

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "survey" / TOWN / "audit"

STEP_M = 2.0
MAX_STEPS = 500                   # 1 km max
YAW_PER_M_LIMIT_DEG = 3.0        # >3°/m of cumulative bend ends "straight"

# narrow road cam (matches comma 3 fcam intrinsics, downstreamed to model later)
NARROW_FOV = 73.0
NARROW_W, NARROW_H = 1928, 1208
EGO_BEFORE_OVERPASS_M = 50.0
EGO_EYE_HEIGHT = 1.4


def walk_straight(world: carla.World, start: carla.Waypoint, direction: str) -> dict:
    """direction in {'next', 'previous'}. Walks STEP_M increments, terminates on
    junction, lane change, or accumulated bend > YAW_PER_M_LIMIT_DEG."""
    cur = start
    distance = 0.0
    yaw0 = cur.transform.rotation.yaw
    last_yaw = yaw0
    history = [(cur.transform.location.x, cur.transform.location.y, last_yaw)]

    for _ in range(MAX_STEPS):
        nexts = cur.next(STEP_M) if direction == "next" else cur.previous(STEP_M)
        if not nexts:
            return {"distance": distance, "reason": "no_next_waypoint", "history": history}
        if len(nexts) > 1:
            return {"distance": distance, "reason": "fork", "history": history}
        nxt = nexts[0]
        if nxt.is_junction:
            return {"distance": distance, "reason": "junction", "history": history}
        # accumulated bend check
        yaw = nxt.transform.rotation.yaw
        d_yaw = ((yaw - last_yaw + 180.0) % 360.0) - 180.0
        if abs(d_yaw) / STEP_M > YAW_PER_M_LIMIT_DEG:
            return {"distance": distance, "reason": f"bend ({abs(d_yaw):.1f}°/{STEP_M:.0f}m)", "history": history}
        distance += STEP_M
        history.append((nxt.transform.location.x, nxt.transform.location.y, yaw))
        cur = nxt
        last_yaw = yaw
    return {"distance": distance, "reason": "max_steps", "history": history}


def capture_audit_frame(world: carla.World) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # weather: worst-case shadow contrast
    # under_yaw = 180° means ego faces -X. perpendicular azimuth = +90° (sun from -Y / south side)
    weather = carla.WeatherParameters(
        cloudiness=0.0, precipitation=0.0,
        sun_altitude_angle=30.0,
        sun_azimuth_angle=90.0,
        fog_density=0.0,
    )
    world.set_weather(weather)

    # ego position: 50 m UPSTREAM of overpass along under_yaw=180° direction
    # going against direction-of-travel (yaw 180) = +X
    cx, cy, _ = OP_CENTER
    ux, uy = OP_UNDER
    yaw_rad = math.radians(OP_UNDER_YAW_DEG)
    ego_x = ux - EGO_BEFORE_OVERPASS_M * math.cos(yaw_rad)  # cos(180°) = -1, so this is +50
    ego_y = uy - EGO_BEFORE_OVERPASS_M * math.sin(yaw_rad)  # sin(180°) ≈ 0
    # find road Z at ego position
    proj = world.get_map().get_waypoint(carla.Location(x=ego_x, y=ego_y, z=0.0),
                                        project_to_road=True, lane_type=carla.LaneType.Driving)
    ego_z = proj.transform.location.z + EGO_EYE_HEIGHT
    print(f"  ego pos    : ({ego_x:+.2f}, {ego_y:+.2f}, {ego_z:.2f})")
    print(f"  facing yaw : {OP_UNDER_YAW_DEG:+.0f}°  (toward overpass at ({cx:+.1f}, {cy:+.1f}))")
    print(f"  sun        : altitude=30°, azimuth=90° (perpendicular, worst-case)")

    bp = world.get_blueprint_library().find("sensor.camera.rgb")
    bp.set_attribute("image_size_x", str(NARROW_W))
    bp.set_attribute("image_size_y", str(NARROW_H))
    bp.set_attribute("fov", str(NARROW_FOV))
    bp.set_attribute("sensor_tick", "0.0")
    cam = world.spawn_actor(bp, carla.Transform(
        location=carla.Location(x=ego_x, y=ego_y, z=ego_z),
        rotation=carla.Rotation(pitch=0.0, yaw=OP_UNDER_YAW_DEG, roll=0.0),
    ))

    img_ref = {}
    cam.listen(lambda im: img_ref.setdefault("image", im))
    try:
        for _ in range(12):
            world.tick()
        t0 = time.time()
        while "image" not in img_ref and time.time() - t0 < 5.0:
            world.tick()
        if "image" not in img_ref:
            raise RuntimeError("camera did not return a frame")
        img = img_ref["image"]
        arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
        rgb = arr[:, :, :3][:, :, ::-1]
        out_path = OUT_DIR / "op03_shadow_audit_alt30_az90.png"
        Image.fromarray(rgb).save(out_path)
        print(f"  saved      : {out_path.relative_to(Path.cwd())}")
        return out_path
    finally:
        cam.stop()
        cam.destroy()


def main() -> int:
    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.load_world(TOWN)
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    for _ in range(10):
        world.tick()

    cm = world.get_map()
    start_wp = cm.get_waypoint(carla.Location(x=OP_UNDER[0], y=OP_UNDER[1], z=0.0),
                               project_to_road=True, lane_type=carla.LaneType.Driving)
    print(f"=== op03 under-road waypoint ===")
    print(f"  resolved  : ({start_wp.transform.location.x:+.2f}, "
          f"{start_wp.transform.location.y:+.2f}, {start_wp.transform.location.z:.2f})  "
          f"yaw={start_wp.transform.rotation.yaw:+.1f}")
    print(f"  road_id={start_wp.road_id}  lane_id={start_wp.lane_id}  "
          f"is_junction={start_wp.is_junction}")

    print("\n=== A. RUNWAY ===")
    forward = walk_straight(world, start_wp, "next")
    backward = walk_straight(world, start_wp, "previous")
    print(f"  next()-direction (along under_yaw): {forward['distance']:.1f} m  "
          f"end reason: {forward['reason']}")
    print(f"  prev()-direction (against under_yaw): {backward['distance']:.1f} m  "
          f"end reason: {backward['reason']}")
    print(f"  longer side: {max(forward['distance'], backward['distance']):.1f} m")
    REQUIRED_M = 300.0
    longer = max(forward['distance'], backward['distance'])
    if longer >= REQUIRED_M:
        print(f"  VERDICT: PASS (have {longer:.0f} m, need >= {REQUIRED_M:.0f} m)")
    else:
        print(f"  VERDICT: FAIL (have {longer:.0f} m, need >= {REQUIRED_M:.0f} m)")

    print("\n=== B. SHADOW AUDIT ===")
    capture_audit_frame(world)

    # back to async
    settings.synchronous_mode = False
    world.apply_settings(settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
