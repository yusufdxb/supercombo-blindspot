"""Town13 stability test. Assumes CARLA server is already running on localhost:2000.

Loads Town13, spawns a stationary Tesla Model 3 at a real spawn point, attaches a
single RGB camera at the windshield position, ticks the world 10 times, then
captures 50 frames. Reports success/failure via exit codes:

  0 = full success (50 frames captured)
  1 = sensor tick crashed the server (client-visible: connection lost / RPC timeout)
  2 = ran but failed to capture 50 frames
  3 = setup failure (load_world / spawn / etc)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import carla
import numpy as np
from PIL import Image

CLIENT_TIMEOUT_S = 120.0
N_TICKS_BEFORE_CAPTURE = 10
N_FRAMES_TO_CAPTURE = 50
PILOT_XY = (2417.6, 4570.2)

OUT = Path(__file__).resolve().parents[1] / "data" / "stability"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam-w", type=int, default=1164)
    ap.add_argument("--cam-h", type=int, default=874)
    ap.add_argument("--quality-tag", default="?")
    ap.add_argument("--vulkan-tag", default="?")
    ap.add_argument("--tag", default="t13")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[{args.tag}] config: q={args.quality_tag} vulkan={args.vulkan_tag} cam={args.cam_w}x{args.cam_h}")

    client = carla.Client("localhost", 2000)
    client.set_timeout(CLIENT_TIMEOUT_S)
    try:
        print(f"[{args.tag}] server={client.get_server_version()} client={client.get_client_version()}")
    except Exception as e:
        print(f"[{args.tag}] FAIL: cannot reach server: {e}")
        return 3

    print(f"[{args.tag}] loading Town13...")
    t0 = time.perf_counter()
    try:
        world = client.load_world("Town13")
    except RuntimeError as e:
        print(f"[{args.tag}] CRASH: load_world: {e}")
        return 1
    print(f"[{args.tag}] loaded in {time.perf_counter()-t0:.1f}s")

    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    spawns = world.get_map().get_spawn_points()
    if not spawns:
        print(f"[{args.tag}] FAIL: no spawn points")
        return 3
    spawns = sorted(spawns, key=lambda sp: (sp.location.x - PILOT_XY[0])**2 + (sp.location.y - PILOT_XY[1])**2)
    sp = spawns[0]
    print(f"[{args.tag}] spawn at ({sp.location.x:+.0f}, {sp.location.y:+.0f}, {sp.location.z:+.0f})")

    bp_lib = world.get_blueprint_library()
    veh_bp = bp_lib.find("vehicle.tesla.model3")
    try:
        vehicle = world.spawn_actor(veh_bp, sp)
    except RuntimeError as e:
        print(f"[{args.tag}] FAIL: vehicle spawn: {e}")
        return 3
    vehicle.set_simulate_physics(False)
    print(f"[{args.tag}] vehicle spawned (stationary)")

    try:
        for _ in range(3):
            world.tick()
        print(f"[{args.tag}] 3 pre-sensor ticks survived")
    except RuntimeError as e:
        print(f"[{args.tag}] CRASH pre-sensor: {e}")
        return 1

    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(args.cam_w))
    cam_bp.set_attribute("image_size_y", str(args.cam_h))
    cam_bp.set_attribute("fov", "90.0")
    cam_bp.set_attribute("sensor_tick", "0.0")
    cam_xform = carla.Transform(carla.Location(x=0.8, z=1.4))
    try:
        cam = world.spawn_actor(cam_bp, cam_xform, attach_to=vehicle)
    except RuntimeError as e:
        print(f"[{args.tag}] CRASH camera spawn: {e}")
        return 1
    print(f"[{args.tag}] camera attached")

    received = {"n": 0, "last": None}
    def on_image(im):
        received["n"] += 1
        received["last"] = im
    cam.listen(on_image)

    print(f"[{args.tag}] ticking {N_TICKS_BEFORE_CAPTURE} times with sensor active...")
    i = -1
    try:
        for i in range(N_TICKS_BEFORE_CAPTURE):
            world.tick()
    except RuntimeError as e:
        print(f"[{args.tag}] CRASH at sensor tick #{i}: {e}")
        return 1
    print(f"[{args.tag}] {N_TICKS_BEFORE_CAPTURE} sensor ticks survived (got {received['n']} frames)")

    print(f"[{args.tag}] capturing target {N_FRAMES_TO_CAPTURE} frames...")
    target = received["n"] + N_FRAMES_TO_CAPTURE
    try:
        while received["n"] < target:
            world.tick()
    except RuntimeError as e:
        print(f"[{args.tag}] CRASH capture phase (had {received['n']}/{target}): {e}")
        return 1
    print(f"[{args.tag}] captured {received['n']} frames total")

    if received["n"] < N_FRAMES_TO_CAPTURE:
        return 2

    img = received["last"]
    arr = np.frombuffer(img.raw_data, dtype=np.uint8).reshape(img.height, img.width, 4)
    rgb = arr[:, :, :3][:, :, ::-1]
    out_path = OUT / f"{args.tag}_q{args.quality_tag}_vk{args.vulkan_tag}_{args.cam_w}x{args.cam_h}.png"
    Image.fromarray(rgb).save(out_path)
    print(f"[{args.tag}] saved sanity PNG: {out_path}")

    try:
        cam.stop(); cam.destroy(); vehicle.destroy()
    except Exception:
        pass
    settings.synchronous_mode = False
    world.apply_settings(settings)
    print(f"[{args.tag}] SUCCESS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
