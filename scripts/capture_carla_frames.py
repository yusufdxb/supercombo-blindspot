"""Capture a run of consecutive CARLA narrow-camera frames (clean op01 approach,
before the overpass) for the offline domain-gap study. Saves raw RGB .npy so the
study can re-warp and perturb them without a CARLA server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import carla
import numpy as np

from src.path_sampling import PolylinePath
from src.scenario import (
    CAM_X, CAM_Z, EGO_Z_LIFT, SETTLE_EPS_M, TICK_DT, CameraSensor, build_drive_path)
from src.sim_preprocessor import fcam_fov_deg

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "data" / "domain_gap" / "carla_rgb.npy"


def main() -> int:
    ap = argparse.ArgumentParser(description="Capture consecutive CARLA narrow frames")
    ap.add_argument("--n", type=int, default=170, help="frames to capture")
    ap.add_argument("--speed", type=float, default=8.0,
                    help="ego speed m/s (lower = more frames stay on clean road)")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    args = ap.parse_args()
    N_CAPTURE = args.n
    SPEED = args.speed
    OUT = Path(args.out)

    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.load_world("Town05")
    geom = build_drive_path(world.get_map())
    path: PolylinePath = geom["path"]
    step_m = SPEED * TICK_DT

    orig = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK_DT
    world.apply_settings(settings)

    ego = None
    narrow = None
    frames = []
    try:
        x0, y0, yaw0 = path.sample(0.0)
        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        ego = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=x0, y=y0, z=float(geom["vertex_z"][0]) + EGO_Z_LIFT),
            carla.Rotation(yaw=yaw0)))
        ego.set_simulate_physics(False)
        narrow = CameraSensor(world, ego, fcam_fov_deg(), "narrow")
        cum = path.vertex_arc_lengths

        for k in range(N_CAPTURE):
            s = k * step_m
            x, y, yaw = path.sample(s)
            road_z = float(np.interp(s, cum, geom["vertex_z"]))
            client.apply_batch_sync([carla.command.ApplyTransform(
                ego.id, carla.Transform(carla.Location(x=x, y=y, z=road_z + EGO_Z_LIFT),
                                        carla.Rotation(yaw=yaw)))], False)
            while True:
                wfrm = world.tick()
                loc = ego.get_transform().location
                if abs(loc.x - x) < SETTLE_EPS_M and abs(loc.y - y) < SETTLE_EPS_M:
                    break
            frames.append(narrow.grab(wfrm))
            if k % 40 == 0:
                print(f"  captured {k}/{N_CAPTURE}")
    finally:
        if narrow is not None:
            narrow.destroy()
        if ego is not None and ego.is_alive:
            ego.destroy()
        world.apply_settings(orig)

    arr = np.stack(frames)  # (N, 1208, 1928, 3) uint8
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUT, arr)
    print(f"saved {arr.shape} -> {OUT}  ({arr.nbytes / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
