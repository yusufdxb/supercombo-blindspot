"""Step 3.5 parity pipeline: run our preprocessor + ModelStateMirror on
Subaru source hevc; compare per-frame accel@t0 to regen modelV2 reference.

Acceptance: |delta| <= 0.5 m/s^2 for >=95% of frames after 2s warm-up trim."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from src.decode_hevc import yuv_frame_iter
from src.rlog import iter_events
from src.state import ModelStateMirror, long_accel_t0
from src.transformations import get_warp_matrix, scaled_intrinsics, _ar_ox_config

DATA = Path(__file__).resolve().parents[1] / "data"
SRC_RLOG = DATA / "subaru_source" / "rlog.bz2"
SRC_HEVC = DATA / "subaru_source" / "fcamera.hevc"
REG_RLOG = DATA / "subaru_regen" / "rlog.bz2"

CAMERA_W = 1928
CAMERA_H = 1208
WARMUP_FRAMES = 40  # 2 s at 20 Hz
ACCEPT_THRESHOLD_MS2 = 0.5
ACCEPT_PCT = 95.0


def load_regen_reference(rlog_path: Path) -> dict[int, float]:
    """frameId -> accel.x[0] (m/s^2 in device frame)."""
    ref = {}
    for ev in iter_events(rlog_path):
        if ev.which() == "modelV2":
            mv = ev.modelV2
            if len(mv.acceleration.x) >= 33:
                ref[mv.frameId] = float(mv.acceleration.x[0])
    return ref


def latest_rpy_calib(rlog_path: Path) -> np.ndarray:
    """Return the last liveCalibration.rpyCalib found in the rlog."""
    rpy = None
    for ev in iter_events(rlog_path):
        if ev.which() == "liveCalibration":
            rpy = np.array(list(ev.liveCalibration.rpyCalib), dtype=np.float64)
    if rpy is None:
        raise RuntimeError("no liveCalibration in source rlog")
    return rpy


def build_warps(rpy_calib: np.ndarray):
    fcam_intrinsics = _ar_ox_config.fcam.intrinsics
    warp_y = get_warp_matrix(rpy_calib, fcam_intrinsics, bigmodel_frame=False)
    warp_uv = get_warp_matrix(rpy_calib, scaled_intrinsics(fcam_intrinsics, 0.5),
                              bigmodel_frame=False)
    return warp_y, warp_uv


def main() -> int:
    print("Loading regen reference modelV2 trace...")
    ref = load_regen_reference(REG_RLOG)
    print(f"  {len(ref)} reference frames (frameId range {min(ref)}..{max(ref)})")

    print("Loading liveCalibration...")
    rpy = latest_rpy_calib(SRC_RLOG)
    print(f"  rpyCalib: [{rpy[0]:+.5f}, {rpy[1]:+.5f}, {rpy[2]:+.5f}] rad")

    print("Building warp matrices (comma 3 / _ar_ox_config narrow road cam)...")
    warp_y, warp_uv = build_warps(rpy)
    print(f"  warp_y: \n{warp_y}")
    print(f"  warp_uv: \n{warp_uv}")

    # Lazy import to keep import-time light
    import cv2
    from src.warped_preprocessor import (
        warp_yuv_to_model, yuv_to_6ch, stack_pair, MODEL_W, MODEL_H,
    )

    print("\nInitializing ModelStateMirror (with zero state — frame 0 will warm up)...")
    state = ModelStateMirror()

    print("\nDecoding HEVC + running pipeline frame-by-frame...")
    prev6 = None
    ours = []   # (regen_frame_id, our_accel)
    deltas = []
    frame_id_offset = 3602  # source roadCameraState frameId at hevc index 0 is 3603; regen modelV2 starts at 1
    # so hevc-index K -> source frameId 3603+K -> regen frameId 1+K-1 = K (approx)
    # Empirically: regen frameId range is 1..1199 for 1200 hevc frames. We map hevc K -> regen K.
    t0 = time.perf_counter()
    for k, (Y, U, V) in enumerate(yuv_frame_iter(SRC_HEVC, CAMERA_W, CAMERA_H)):
        # warp + 6-channel split
        Y_m, U_m, V_m = warp_yuv_to_model(Y, U, V, warp_y, warp_uv)
        curr6 = yuv_to_6ch(Y_m, U_m, V_m)

        if prev6 is None:
            prev6 = curr6
            continue  # need 2 frames before any prediction

        input_imgs = stack_pair(prev6, curr6)
        # use the same wide-camera input for big_input_imgs (no wide ecamera in this rlog).
        # the model still expects a tensor; we feed the same warped frame. This is a known
        # corner — modeld normally takes ecamera here. We document the mismatch in the report.
        big = input_imgs

        parsed = state.run(input_imgs, big)
        our_accel = long_accel_t0(parsed)

        # match by regen frameId
        regen_fid = k  # hevc-K maps to regen frameId K (1..1199)
        if regen_fid in ref:
            deltas.append(our_accel - ref[regen_fid])
            ours.append((regen_fid, our_accel, ref[regen_fid]))

        prev6 = curr6
        if k % 100 == 0 and k > 0:
            elapsed = time.perf_counter() - t0
            print(f"  frame {k:4d}/{1200}  elapsed {elapsed:6.1f}s  "
                  f"running delta median: {np.median(np.abs(deltas)) if deltas else 0:.3f} m/s^2")

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {len(ours)} matched frames in {elapsed:.1f}s.")

    # trim warmup
    if len(deltas) > WARMUP_FRAMES:
        kept = deltas[WARMUP_FRAMES:]
        kept_pairs = ours[WARMUP_FRAMES:]
    else:
        kept = deltas
        kept_pairs = ours
    kept = np.array(kept)
    print(f"\nAfter {WARMUP_FRAMES}-frame warm-up trim: {len(kept)} frames")

    abs_d = np.abs(kept)
    pct_within = 100 * (abs_d <= ACCEPT_THRESHOLD_MS2).mean()
    print(f"\n=== HISTOGRAM of |delta| (m/s^2) ===")
    bins = [0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, np.inf]
    hist, _ = np.histogram(abs_d, bins=bins)
    for lo, hi, c in zip(bins[:-1], bins[1:], hist):
        bar = "#" * int(50 * c / max(1, hist.max()))
        print(f"  [{lo:>5.2f}, {hi:>5.2f}) : {c:>5d}  {bar}")

    print(f"\n=== STATS ===")
    print(f"  mean |delta|        : {abs_d.mean():.4f}")
    print(f"  median |delta|      : {np.median(abs_d):.4f}")
    print(f"  p95 |delta|         : {np.percentile(abs_d, 95):.4f}")
    print(f"  p99 |delta|         : {np.percentile(abs_d, 99):.4f}")
    print(f"  max |delta|         : {abs_d.max():.4f}")
    print(f"  signed bias (mean)  : {kept.mean():+.4f}")
    print(f"  % within +/- {ACCEPT_THRESHOLD_MS2} : {pct_within:.2f}%")

    # worst 5 frames
    print(f"\n=== WORST 5 FRAMES ===")
    worst_idx = np.argsort(abs_d)[::-1][:5]
    for i in worst_idx:
        fid, ours_, ref_ = kept_pairs[i]
        print(f"  frame {fid:4d}: ours={ours_:+.4f}  ref={ref_:+.4f}  delta={ours_-ref_:+.4f}")

    print(f"\n=== VERDICT ===")
    if pct_within >= ACCEPT_PCT:
        print(f"  PASS: {pct_within:.2f}% within {ACCEPT_THRESHOLD_MS2} m/s^2 (need >= {ACCEPT_PCT}%)")
        return 0
    print(f"  FAIL: {pct_within:.2f}% within {ACCEPT_THRESHOLD_MS2} m/s^2 (need >= {ACCEPT_PCT}%)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
