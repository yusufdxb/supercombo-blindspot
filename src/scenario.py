"""Step 4b — Town05 op01 dual-camera two-phase scenario harness.

Drives an ego vehicle kinematically along the op01 under-road corridor, through
the overpass, while running openpilot v0.9.7 supercombo frame-by-frame and
recording longitudinal accel @ t=0.

Why two-phase
-------------
supercombo carries a 99-frame recurrent state (`features_buffer`). A cold start
produces a multi-second init transient that looks exactly like a phantom brake
(see the project's `recurrent-state-must-roll` note). So the run is split:

  * Phase 1 "warmup"  — the first `--warmup` frames (default 100 = 5 s @ 20 Hz).
                        The ego drives clean road; the recurrent state fills.
                        accel readings here are discarded.
  * Phase 2 "measure" — the rest of the drive: clean approach, the overpass
                        crossing, and the road beyond. accel readings here are
                        the signal Step 5 will scan for anomalies.

It is one continuous kinematic drive; "two-phase" is a labelling of the trace.

Kinematic playback
------------------
The ego has physics disabled and is teleported one fixed arc-length per tick.
`set_transform` alone races the render — intermittently the camera renders the
*previous* pose (proven in scripts/diag_scenario.py: pixel-delta collapses and
accel snaps to the model's null value). The fix is `apply_batch_sync` with an
`ApplyTransform` command, which blocks until the server has applied the pose,
then a post-tick verify that re-ticks if the ego still has not settled.

Camera faithfulness
-------------------
The narrow camera is rendered at comma 3 fcam native resolution (1928x1208) with
the fov that reproduces fcam's focal length, so its intrinsics match
`_ar_ox_config.fcam` exactly. The verified Step 3.5 warp then maps it to the
512x256 medmodel frame with zero calibration (sim camera mounted perfectly).
The wide camera is recorded for overlays but, per the Step 3.5 finding that
accel@t0 is narrow-dominated, the model's `big_input_imgs` is fed the narrow
frame too (a documented, parity-verified approximation).

Usage
-----
    # CARLA must be running:
    #   cd ~/Sim/CARLA_0.9.15 && ./CarlaUE4.sh -RenderOffScreen \\
    #       -quality-level=Epic -carla-rpc-port=2000
    env -u PYTHONPATH .venv/bin/python -m src.scenario --dry-run
    env -u PYTHONPATH .venv/bin/python -m src.scenario --speed 8 --save-frames
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from queue import Queue

import carla
import numpy as np

from src.path_sampling import PolylinePath
from src.sim_preprocessor import (
    CARLA_CAM_H,
    CARLA_CAM_W,
    build_sim_warps,
    ecam_fov_deg,
    fcam_fov_deg,
    rgb_to_model_input,
)

# --- scenario constants ---
TOWN = "Town05"
OP01_SEED_XY = (-224.5, -95.2)   # under-road waypoint nearest the op01 overpass
WALK_STEP_M = 1.0                # waypoint walk granularity
APPROACH_MAX_M = 130             # cap on the corridor walk past the seed
BEYOND_MAX_M = 60                # cap on the corridor walk before the seed
TICK_DT = 0.05                   # 20 Hz, matches ModelConstants.MODEL_FREQ
CAM_X = 0.5                      # camera mount: metres forward of ego origin
CAM_Z = 1.3                      # camera mount: metres above ego origin (windshield)
EGO_Z_LIFT = 0.2                 # ego placed this far above the road surface
SETTLE_EPS_M = 0.05              # ego-pose verify tolerance
MAX_RETICKS = 6
OVERHEAD_MIN_M = 3.0             # a hit this far above the road counts as overhead
OVERHEAD_MAX_M = 25.0
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "scenario" / "op01"


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _walk(seed: carla.Waypoint, direction: str, max_m: float) -> list[carla.Waypoint]:
    """Walk WALK_STEP_M increments from `seed`. Stops at a junction or a dead end;
    on a fork, takes branch 0. Returns the waypoints walked (excluding seed)."""
    out: list[carla.Waypoint] = []
    cur = seed
    walked = 0.0
    while walked < max_m:
        nxt = cur.next(WALK_STEP_M) if direction == "next" else cur.previous(WALK_STEP_M)
        if not nxt:
            break
        cur = nxt[0]
        if cur.is_junction:
            break
        out.append(cur)
        walked += WALK_STEP_M
    return out


def build_drive_path(carla_map: carla.Map) -> dict:
    """Resolve the op01 corridor into a drive path.

    The lane's travel direction at the seed runs `.next()` along the 108 m runway
    and `.previous()` the other way. The ego drives from the far `.next()` end,
    back through the seed, and out the `.previous()` end — one continuous
    straight corridor that passes under the overpass.
    """
    seed = carla_map.get_waypoint(
        carla.Location(x=OP01_SEED_XY[0], y=OP01_SEED_XY[1], z=0.0),
        project_to_road=True, lane_type=carla.LaneType.Driving,
    )
    fwd = _walk(seed, "next", APPROACH_MAX_M)
    bwd = _walk(seed, "previous", BEYOND_MAX_M)

    wps = list(reversed(fwd)) + [seed] + bwd
    xy = np.array([[w.transform.location.x, w.transform.location.y] for w in wps])
    z = np.array([w.transform.location.z for w in wps])
    return {
        "path": PolylinePath(xy),
        "vertex_z": z,
        "wps": wps,
        "seed_idx": len(fwd),
        "fwd_m": float(len(fwd) * WALK_STEP_M),
        "bwd_m": float(len(bwd) * WALK_STEP_M),
    }


def detect_overpass_span(world: carla.World, geom: dict) -> dict:
    """Cast a ray straight up at every path vertex; the longest contiguous run of
    vertices with structure overhead is the overpass. Returns its entry / mid /
    exit arc-lengths and median clearance (or an empty result if none found)."""
    path: PolylinePath = geom["path"]
    z = geom["vertex_z"]
    overhead = np.zeros(len(path.xy), dtype=bool)
    clearance: list[float] = []
    for i, (x, y) in enumerate(path.xy):
        road_z = float(z[i])
        try:
            hits = world.cast_ray(
                carla.Location(x=float(x), y=float(y), z=road_z + 1.0),
                carla.Location(x=float(x), y=float(y), z=road_z + OVERHEAD_MAX_M),
            )
        except Exception:
            continue
        above = [h.location.z - road_z for h in hits
                 if OVERHEAD_MIN_M < (h.location.z - road_z) < OVERHEAD_MAX_M]
        if above:
            overhead[i] = True
            clearance.append(min(above))

    # longest contiguous True run
    best_lo = best_hi = -1
    lo = None
    for i, v in enumerate(np.append(overhead, False)):
        if v and lo is None:
            lo = i
        elif not v and lo is not None:
            if best_lo < 0 or (i - lo) > (best_hi - best_lo):
                best_lo, best_hi = lo, i
            lo = None

    if best_lo < 0:
        return {"found": False}
    mid = (best_lo + best_hi - 1) // 2
    return {
        "found": True,
        "entry_arc_m": path.arc_length_at(best_lo),
        "exit_arc_m": path.arc_length_at(best_hi - 1),
        "mid_arc_m": path.arc_length_at(mid),
        "clearance_m": float(np.median(clearance)) if clearance else float("nan"),
        "span_vertices": best_hi - best_lo,
    }


# --------------------------------------------------------------------------
# CARLA camera
# --------------------------------------------------------------------------

class CameraSensor:
    """An RGB camera attached to the ego, read through a frame queue. `grab`
    drains the queue to the exact world frame requested, so re-ticked frames
    (from the pose-settle retry) are discarded cleanly."""

    def __init__(self, world: carla.World, parent: carla.Actor, fov_deg: float, role: str):
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(CARLA_CAM_W))
        bp.set_attribute("image_size_y", str(CARLA_CAM_H))
        bp.set_attribute("fov", f"{fov_deg:.4f}")
        bp.set_attribute("sensor_tick", "0.0")
        mount = carla.Transform(carla.Location(x=CAM_X, z=CAM_Z))
        self.sensor = world.spawn_actor(bp, mount, attach_to=parent)
        self.role = role
        self._queue: Queue = Queue()
        self.sensor.listen(self._queue.put)

    def grab(self, world_frame: int, timeout_s: float = 12.0) -> np.ndarray:
        """Return the RGB frame rendered for `world_frame`, dropping any older."""
        image = self._queue.get(timeout=timeout_s)
        while image.frame < world_frame:
            image = self._queue.get(timeout=timeout_s)
        if image.frame != world_frame:
            raise RuntimeError(
                f"{self.role}: wanted frame {world_frame}, overshot to {image.frame}")
        buf = np.frombuffer(image.raw_data, dtype=np.uint8)
        bgra = buf.reshape((image.height, image.width, 4))
        return np.ascontiguousarray(bgra[:, :, :3][:, :, ::-1])  # BGRA -> RGB

    def destroy(self) -> None:
        if self.sensor.is_alive:
            self.sensor.stop()
            self.sensor.destroy()


# --------------------------------------------------------------------------
# scenario run
# --------------------------------------------------------------------------

def _save_png(rgb: np.ndarray, path: Path) -> None:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path)


def run_scenario(args: argparse.Namespace) -> int:
    client = carla.Client("localhost", 2000)
    client.set_timeout(30.0)
    print(f"CARLA server {client.get_server_version()}  client {client.get_client_version()}")

    print(f"Loading {TOWN} ...")
    world = client.load_world(TOWN)
    geom = build_drive_path(world.get_map())
    path: PolylinePath = geom["path"]

    print("Detecting overpass (ray-cast up at every path vertex) ...")
    op = detect_overpass_span(world, geom)

    step_m = args.speed * TICK_DT
    n_frames = int(math.ceil(path.length / step_m)) + 1
    if args.max_frames:
        n_frames = min(n_frames, args.max_frames)

    print("\n=== DRIVE PATH ===")
    print(f"  corridor           : {geom['fwd_m']:.0f} m + {geom['bwd_m']:.0f} m "
          f"= {path.length:.1f} m total")
    print(f"  speed / step       : {args.speed:.1f} m/s -> {step_m:.3f} m/frame")
    print(f"  total frames       : {n_frames}  ({n_frames * TICK_DT:.1f} s)")
    print(f"  warmup frames      : {args.warmup}  (measurement starts frame {args.warmup})")
    if op["found"]:
        op_entry_f = int(round(op["entry_arc_m"] / step_m))
        op_mid_f = int(round(op["mid_arc_m"] / step_m))
        op_exit_f = int(round(op["exit_arc_m"] / step_m))
        print(f"  overpass span      : arc {op['entry_arc_m']:.1f}-{op['exit_arc_m']:.1f} m "
              f"({op['span_vertices']} m), clearance {op['clearance_m']:.1f} m")
        print(f"  overpass frames    : entry {op_entry_f}  mid {op_mid_f}  exit {op_exit_f}")
        if op_entry_f <= args.warmup:
            print(f"  WARN: overpass entry frame {op_entry_f} is inside the warmup "
                  f"window ({args.warmup}) — lower --speed")
    else:
        op_entry_f = op_mid_f = op_exit_f = -1
        print("  overpass span      : NOT DETECTED by ray-cast")

    if args.dry_run:
        print("\n--dry-run: geometry only, no ego/cameras/model. Done.")
        return 0

    world.set_weather(carla.WeatherParameters(
        cloudiness=0.0, precipitation=0.0, fog_density=0.0,
        sun_altitude_angle=args.sun_altitude, sun_azimuth_angle=args.sun_azimuth,
    ))

    orig_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = TICK_DT
    world.apply_settings(settings)

    ego = None
    narrow = wide = None
    rows: list[dict] = []
    total_reticks = 0
    try:
        x0, y0, yaw0 = path.sample(0.0)
        z0 = float(geom["vertex_z"][0])
        bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
        ego = world.spawn_actor(bp, carla.Transform(
            carla.Location(x=x0, y=y0, z=z0 + EGO_Z_LIFT),
            carla.Rotation(yaw=yaw0)))
        ego.set_simulate_physics(False)
        narrow = CameraSensor(world, ego, fcam_fov_deg(), "narrow")
        wide = CameraSensor(world, ego, ecam_fov_deg(), "wide")
        print(f"\nego={ego.type_id}  narrow_fov={fcam_fov_deg():.2f}  "
              f"wide_fov={ecam_fov_deg():.2f}")

        from src.state import ModelStateMirror, long_accel_t0
        warp_y, warp_uv = build_sim_warps()
        print("Building ModelStateMirror (ORT session) ...")
        state = ModelStateMirror()
        print("  first inference triggers ~28 s PTX JIT for sm_120 — expected.")

        cum = path.vertex_arc_lengths
        prev_rgb = None
        t_start = time.perf_counter()

        for k in range(n_frames):
            s = k * step_m
            x, y, yaw = path.sample(s)
            road_z = float(np.interp(s, cum, geom["vertex_z"]))
            target = carla.Transform(
                carla.Location(x=x, y=y, z=road_z + EGO_Z_LIFT),
                carla.Rotation(yaw=yaw))

            # synchronous pose apply + verify (see module docstring)
            client.apply_batch_sync(
                [carla.command.ApplyTransform(ego.id, target)], False)
            reticks = 0
            while True:
                wfrm = world.tick()
                loc = ego.get_transform().location
                if abs(loc.x - x) < SETTLE_EPS_M and abs(loc.y - y) < SETTLE_EPS_M:
                    break
                reticks += 1
                if reticks > MAX_RETICKS:
                    print(f"  WARN frame {k}: ego not settled after {MAX_RETICKS} reticks")
                    break
            total_reticks += reticks

            narrow_rgb = narrow.grab(wfrm)
            wide_rgb = wide.grab(wfrm)

            phase = "warmup" if k < args.warmup else "measure"
            accel = float("nan")
            y_mean = float("nan")
            if prev_rgb is not None:
                input_imgs = rgb_to_model_input(prev_rgb, narrow_rgb, warp_y, warp_uv)
                parsed = state.run(input_imgs, input_imgs)  # narrow -> both (Step 3.5)
                accel = long_accel_t0(parsed)
                y_mean = float(input_imgs[0, 6:10].mean())  # curr-frame luma proxy

            under_op = (op["found"] and op["entry_arc_m"] <= s <= op["exit_arc_m"])
            rows.append({
                "frame": k, "phase": phase, "t_s": round(k * TICK_DT, 3),
                "arc_m": round(s, 3), "x": round(x, 3), "y": round(y, 3),
                "yaw_deg": round(yaw, 2), "speed_mps": args.speed,
                "under_overpass": int(under_op),
                "narrow_y_mean": round(y_mean, 3),
                "accel_t0_mps2": round(accel, 5),
            })

            if args.save_frames and (k % 40 == 0 or k in (op_entry_f, op_mid_f, op_exit_f)):
                _save_png(narrow_rgb, OUT_DIR / f"narrow_{k:04d}.png")
                _save_png(wide_rgb, OUT_DIR / f"wide_{k:04d}.png")

            prev_rgb = narrow_rgb
            if k == 1:
                print(f"  frame 1 done in {time.perf_counter() - t_start:.1f} s "
                      f"(includes ~28 s cold start)")
            elif k % 50 == 0 and k > 0:
                print(f"  frame {k:4d}/{n_frames}  arc {s:6.1f} m  accel {accel:+.3f}")

        dt = time.perf_counter() - t_start
        print(f"\nDrive complete: {n_frames} frames in {dt:.1f} s wall  "
              f"({total_reticks} pose-settle reticks total).")
    finally:
        for actor in (narrow, wide):
            if actor is not None:
                actor.destroy()
        if ego is not None and ego.is_alive:
            ego.destroy()
        world.apply_settings(orig_settings)

    _write_trace(rows, args)
    return _summarize(rows, args, op, op_entry_f, op_mid_f, op_exit_f)


def _write_trace(rows: list[dict], args: argparse.Namespace) -> None:
    out_csv = Path(args.out)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"trace -> {out_csv}  ({len(rows)} rows)")


def _summarize(rows: list[dict], args: argparse.Namespace, op: dict,
               op_entry_f: int, op_mid_f: int, op_exit_f: int) -> int:
    """Print warmup-settling + measurement stats and judge the run usable."""
    accel = np.array([r["accel_t0_mps2"] for r in rows], dtype=np.float64)
    valid = ~np.isnan(accel)
    warm = np.array([r["phase"] == "warmup" for r in rows]) & valid
    meas = np.array([r["phase"] == "measure" for r in rows]) & valid

    print("\n=== WARMUP SETTLING (proves two-phase init works) ===")
    warm_idx = np.where(warm)[0]
    settled = False
    if len(warm_idx) >= 12:
        first = accel[warm_idx[0]]
        last10 = accel[warm_idx[-10:]]
        settle = float(np.std(last10))
        settled = settle < 0.10
        print(f"  accel frame {warm_idx[0]:3d} (cold)   : {first:+.4f} m/s^2")
        print(f"  last 10 warmup frames       : mean {last10.mean():+.4f}  std {settle:.4f}")
        print(f"  settled before measurement  : {'YES' if settled else 'NO'} "
              f"(std {'<' if settled else '>='} 0.10)")
    else:
        print("  too few warmup frames to judge settling")

    print("\n=== MEASUREMENT PHASE accel@t0 ===")
    smooth = False
    if meas.any():
        m = accel[meas]
        jerk = np.abs(np.diff(accel[valid]))
        smooth = bool(np.median(jerk) < 0.15)
        print(f"  frames                      : {int(meas.sum())}")
        print(f"  mean / std                  : {m.mean():+.4f} / {m.std():.4f} m/s^2")
        print(f"  min / max                   : {m.min():+.4f} / {m.max():+.4f} m/s^2")
        print(f"  median frame-to-frame jerk  : {np.median(jerk):.4f} m/s^2 "
              f"({'smooth' if smooth else 'JITTERY'})")

    if op["found"]:
        print("\n=== accel@t0 ACROSS THE OVERPASS ===")
        baseline = [r["accel_t0_mps2"] for r in rows
                    if r["phase"] == "measure" and not r["under_overpass"]
                    and r["frame"] < op_entry_f and not math.isnan(r["accel_t0_mps2"])]
        under = [r["accel_t0_mps2"] for r in rows
                 if r["under_overpass"] and not math.isnan(r["accel_t0_mps2"])]
        if baseline and under:
            b_mean = float(np.mean(baseline))
            u_min = float(np.min(under))
            print(f"  clean-approach baseline     : {b_mean:+.4f} m/s^2 "
                  f"(n={len(baseline)})")
            print(f"  most negative under overpass: {u_min:+.4f} m/s^2 (n={len(under)})")
            print(f"  brake excursion vs baseline : {u_min - b_mean:+.4f} m/s^2")
        for r in rows:
            if (op_entry_f - 6) <= r["frame"] <= (op_exit_f + 6) and r["frame"] % 4 == 0 \
                    and not math.isnan(r["accel_t0_mps2"]):
                tag = "  [under]" if r["under_overpass"] else ""
                print(f"  f{r['frame']:4d} {r['phase']:7s} arc {r['arc_m']:6.1f} m  "
                      f"y_mean {r['narrow_y_mean']:6.1f}  "
                      f"accel {r['accel_t0_mps2']:+.4f}{tag}")

    ok = bool(meas.sum() >= 20 and settled and smooth)
    print(f"\n=== VERDICT ===\n  {'PASS' if ok else 'REVIEW'}: "
          + ("two-phase harness produced a smooth, settled measured trace"
             if ok else "see stats above"))
    return 0 if ok else 1


def main() -> int:
    p = argparse.ArgumentParser(description="Step 4b Town05 op01 scenario harness")
    p.add_argument("--speed", type=float, default=8.0, help="ego speed m/s (default 8)")
    p.add_argument("--warmup", type=int, default=100,
                   help="warmup frame count, recurrent-state fill (default 100)")
    p.add_argument("--sun-altitude", type=float, default=45.0)
    p.add_argument("--sun-azimuth", type=float, default=0.0)
    p.add_argument("--save-frames", action="store_true",
                   help="dump narrow+wide PNGs every 40 frames and at the overpass")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve geometry + overpass and print, no ego/cameras/model")
    p.add_argument("--max-frames", type=int, default=0,
                   help="cap the frame count (0 = full path), for quick iteration")
    p.add_argument("--out", type=str, default=str(OUT_DIR / "trace.csv"))
    args = p.parse_args()
    return run_scenario(args)


if __name__ == "__main__":
    sys.exit(main())
