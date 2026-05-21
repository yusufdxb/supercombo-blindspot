"""Diagnostic for the Step 4b accel oscillation.

Hypotheses for the frame-to-frame +-1.5 m/s^2 jitter in the scenario trace:
  H1  camera desync — grabbed image.frame != world tick frame, so prev/curr are
      not consecutive (or are duplicates).
  H2  the model is unstable on out-of-distribution CARLA imagery.

This run isolates them. Per frame it logs: world frame id, narrow image.frame id,
queue depth, whether prev==curr, input tensor stats, and accel@t0. If image.frame
tracks the world frame and the input is sane but accel still jumps -> H2. If the
frame ids or the prev==curr flag misbehave -> H1.
"""

from __future__ import annotations

import sys
from queue import Queue

import carla
import numpy as np

from src.path_sampling import PolylinePath
from src.scenario import CAM_X, CAM_Z, TICK_DT, build_drive_path
from src.sim_preprocessor import (
    CARLA_CAM_H,
    CARLA_CAM_W,
    build_sim_warps,
    fcam_fov_deg,
    rgb_to_model_input,
)

N_FRAMES = 45
SPEED = 8.0


def main() -> int:
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    world = client.load_world("Town05")
    geom = build_drive_path(world.get_map())
    path: PolylinePath = geom["path"]
    step_m = SPEED * TICK_DT

    settings = world.get_settings()
    orig = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK_DT
    world.apply_settings(settings)

    ego = None
    cam = None
    try:
        x0, y0, yaw0 = path.sample(0.0)
        z0 = float(geom["vertex_z"][0])
        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        ego = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=x0, y=y0, z=z0 + 0.2), carla.Rotation(yaw=yaw0)))
        ego.set_simulate_physics(False)

        cbp = world.get_blueprint_library().find("sensor.camera.rgb")
        cbp.set_attribute("image_size_x", str(CARLA_CAM_W))
        cbp.set_attribute("image_size_y", str(CARLA_CAM_H))
        cbp.set_attribute("fov", f"{fcam_fov_deg():.4f}")
        cbp.set_attribute("sensor_tick", "0.0")
        cam = world.spawn_actor(cbp, carla.Transform(carla.Location(x=CAM_X, z=CAM_Z)),
                                attach_to=ego)
        q: Queue = Queue()
        cam.listen(q.put)

        from src.state import ModelStateMirror, long_accel_t0
        warp_y, warp_uv = build_sim_warps()
        state = ModelStateMirror()

        print(f"{'k':>3} {'wfrm':>6} {'ifrm':>6} {'rtk':>3} {'dup':>4} "
              f"{'in_min':>7} {'in_max':>7} {'in_mean':>8} {'pc_diff':>8} {'accel':>9}")
        prev = None
        for k in range(N_FRAMES):
            s = k * step_m
            x, y, yaw = path.sample(s)
            road_z = float(np.interp(s, path.vertex_arc_lengths, geom["vertex_z"]))
            target = carla.Transform(
                carla.Location(x=x, y=y, z=road_z + 0.2), carla.Rotation(yaw=yaw))
            # synchronous transform apply, then tick until the server confirms
            # the ego actually sits at the target before we keep a rendered frame
            client.apply_batch_sync([carla.command.ApplyTransform(ego.id, target)], False)
            reticks = 0
            while True:
                wfrm = world.tick()
                loc = ego.get_transform().location
                if abs(loc.x - x) < 0.05 and abs(loc.y - y) < 0.05:
                    break
                reticks += 1
                if reticks > 6:
                    break

            # drain the queue to the frame we actually kept
            image = q.get(timeout=10.0)
            while image.frame != wfrm:
                image = q.get(timeout=10.0)
            qd_before = reticks
            buf = np.frombuffer(image.raw_data, dtype=np.uint8)
            rgb = np.ascontiguousarray(
                buf.reshape((image.height, image.width, 4))[:, :, :3][:, :, ::-1])

            dup = "-"
            pc_diff = float("nan")
            accel = float("nan")
            in_min = in_max = in_mean = float("nan")
            if prev is not None:
                dup = "DUP!" if np.array_equal(prev, rgb) else "ok"
                pc_diff = float(np.mean(np.abs(prev.astype(np.int16) - rgb.astype(np.int16))))
                inp = rgb_to_model_input(prev, rgb, warp_y, warp_uv)
                in_min, in_max, in_mean = float(inp.min()), float(inp.max()), float(inp.mean())
                parsed = state.run(inp, inp)
                accel = long_accel_t0(parsed)
            print(f"{k:>3} {wfrm:>6} {image.frame:>6} {qd_before:>3} {dup:>4} "
                  f"{in_min:>7.1f} {in_max:>7.1f} {in_mean:>8.2f} {pc_diff:>8.3f} {accel:>9.4f}")
            prev = rgb
    finally:
        if cam is not None:
            cam.stop()
            cam.destroy()
        if ego is not None and ego.is_alive:
            ego.destroy()
        world.apply_settings(orig)
    return 0


if __name__ == "__main__":
    sys.exit(main())
