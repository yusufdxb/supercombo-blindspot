"""Step 3.5 parity pipeline: run our preprocessor + ModelStateMirror on
source hevc; compare per-frame accel@t0 to the comma modelV2 reference.

Two parity targets:
  - default (v0.9.7): Subaru source hevc vs regen modelV2 reference.
  - --model v096    : CI TEST_ROUTE seg-6 hevc vs comma's v0.9.6 model_replay
                      reference (the reference is matched to THIS route, not Subaru).

Acceptance: |delta| <= 0.5 m/s^2 for >=95% of frames after 2s warm-up trim."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from src.decode_hevc import yuv_frame_iter
from src.rlog import iter_events
from src.state import ModelStateMirror, build_mirror, long_accel_t0
from src.transformations import get_warp_matrix, scaled_intrinsics, _ar_ox_config

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SRC_RLOG = DATA / "subaru_source" / "rlog.bz2"
SRC_HEVC = DATA / "subaru_source" / "fcamera.hevc"
REG_RLOG = DATA / "subaru_regen" / "rlog.bz2"

# v0.9.6 second-model parity route (CI TEST_ROUTE seg-6; fetch_upgrade_data.py).
V096_MODEL = REPO / "models" / "supercombo_v096.onnx"
V096_SRC_HEVC = DATA / "ci_v096_source" / "fcamera.hevc"
V096_SRC_RLOG = DATA / "ci_v096_source" / "rlog.bz2"
V096_REF = DATA / "ci_v096_ref" / "model_ref.bz2"

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


def road_camera_frame_ids(rlog_path: Path) -> list[int]:
    """roadCameraState.frameId in stream order (one per decoded hevc frame)."""
    fids = []
    for ev in iter_events(rlog_path):
        if ev.which() == "roadCameraState":
            fids.append(int(ev.roadCameraState.frameId))
    return fids


def derive_frame_offset(src_rlog: Path, ref: dict[int, float]) -> int:
    """Derive the hevc-index -> reference-frameId offset STRUCTURALLY from metadata.

    comma's model_replay re-bases modelV2.frameId to a 1-based sequential index over
    the camera frames it processes (verified on the Subaru route: source roadCameraState
    frameIds 3603..4802, regen modelV2 frameIds 1..1199 -> hevc index K maps to reference
    frameId K, the positional rule documented at run_parity.py:88-90).

    The offset is therefore the difference between the reference's first frameId and the
    hevc index it corresponds to. Both routes re-base to 1 and the first prediction is at
    hevc index 1 (index 0 is consumed as the warm-up prev-frame), so offset == 0:
    reference frameId R == hevc index R. This is read off the frameId ranges, NOT tuned to
    maximise parity."""
    ref_first = min(ref)
    src_fids = road_camera_frame_ids(src_rlog)
    src_first = src_fids[0] if src_fids else None
    # Positional rule: reference frameId R == hevc index R. offset := R - K = 0.
    offset = 0
    print(f"  reference frameId range: {min(ref)}..{max(ref)} ({len(ref)} frames)")
    print(f"  source roadCameraState frameId range: "
          f"{src_first}..{src_fids[-1] if src_fids else None} ({len(src_fids)} frames)")
    print(f"  structural mapping: reference frameId R == hevc index R "
          f"(ref_first={ref_first}, offset={offset}, not tuned)")
    return offset


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


def run_parity(state: ModelStateMirror, src_hevc: Path, ref: dict[int, float],
               warp_y: np.ndarray, warp_uv: np.ndarray, offset: int,
               n_total: int) -> tuple[np.ndarray, list]:
    """Decode `src_hevc`, thread state through `state`, match accel@t0 to `ref`.

    Returns (deltas array, list of (frameId, ours, ref)). `offset` is the
    hevc-index -> reference-frameId offset derived structurally upstream."""
    import cv2  # noqa: F401  (lazy: keeps import-time light, matches repo CI convention)
    from src.warped_preprocessor import warp_yuv_to_model, yuv_to_6ch, stack_pair

    print("\nDecoding HEVC + running pipeline frame-by-frame...")
    prev6 = None
    ours = []
    deltas = []
    t0 = time.perf_counter()
    for k, (Y, U, V) in enumerate(yuv_frame_iter(src_hevc, CAMERA_W, CAMERA_H)):
        Y_m, U_m, V_m = warp_yuv_to_model(Y, U, V, warp_y, warp_uv)
        curr6 = yuv_to_6ch(Y_m, U_m, V_m)

        if prev6 is None:
            prev6 = curr6
            continue  # need 2 frames before any prediction

        input_imgs = stack_pair(prev6, curr6)
        # same warped frame for big_input_imgs (no wide ecamera in this rlog). Documented corner.
        big = input_imgs

        parsed = state.run(input_imgs, big)
        our_accel = long_accel_t0(parsed)

        # structural mapping: reference frameId R == hevc index k + offset.
        ref_fid = k + offset
        if ref_fid in ref:
            deltas.append(our_accel - ref[ref_fid])
            ours.append((ref_fid, our_accel, ref[ref_fid]))

        prev6 = curr6
        if k % 100 == 0 and k > 0:
            elapsed = time.perf_counter() - t0
            print(f"  frame {k:4d}/{n_total}  elapsed {elapsed:6.1f}s  "
                  f"running delta median: {np.median(np.abs(deltas)) if deltas else 0:.3f} m/s^2")

    elapsed = time.perf_counter() - t0
    print(f"\nDone. {len(ours)} matched frames in {elapsed:.1f}s.")
    return np.array(deltas), ours


def report_parity(deltas: np.ndarray, ours: list) -> float:
    """Trim warm-up, print histogram + stats + worst frames + verdict.
    Returns pct_within (% of post-trim frames within +/-0.5 m/s^2)."""
    if len(deltas) > WARMUP_FRAMES:
        kept = deltas[WARMUP_FRAMES:]
        kept_pairs = ours[WARMUP_FRAMES:]
    else:
        kept = deltas
        kept_pairs = ours
    kept = np.asarray(kept)
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

    print(f"\n=== WORST 5 FRAMES ===")
    worst_idx = np.argsort(abs_d)[::-1][:5]
    for i in worst_idx:
        fid, ours_, ref_ = kept_pairs[i]
        print(f"  frame {fid:4d}: ours={ours_:+.4f}  ref={ref_:+.4f}  delta={ours_-ref_:+.4f}")

    print(f"\n=== VERDICT ===")
    if pct_within >= ACCEPT_PCT:
        print(f"  PASS: {pct_within:.2f}% within {ACCEPT_THRESHOLD_MS2} m/s^2 (need >= {ACCEPT_PCT}%)")
    else:
        print(f"  FAIL: {pct_within:.2f}% within {ACCEPT_THRESHOLD_MS2} m/s^2 (need >= {ACCEPT_PCT}%)")
    return pct_within


def run_v097() -> int:
    """v0.9.7 parity: Subaru source hevc vs regen modelV2 reference (default)."""
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

    print("\nDeriving hevc-index -> reference-frameId mapping (structural)...")
    offset = derive_frame_offset(SRC_RLOG, ref)

    print("\nInitializing ModelStateMirror (zero state — frame 0 warms up)...")
    state = ModelStateMirror()
    print(f"  EP in use: {state.session.get_providers()[0]}")

    deltas, ours = run_parity(state, SRC_HEVC, ref, warp_y, warp_uv, offset, n_total=1200)
    pct = report_parity(deltas, ours)
    return 0 if pct >= ACCEPT_PCT else 1


def run_v096() -> int:
    """v0.9.6 parity: CI TEST_ROUTE seg-6 hevc vs comma's v0.9.6 model_replay reference.

    The reference is matched to THIS route (dongle 2f4452b03ccb98f0, seg 6), so the source
    is data/ci_v096_source, NOT Subaru. Inputs fetched by scripts/fetch_upgrade_data.py."""
    for p in (V096_MODEL, V096_SRC_HEVC, V096_SRC_RLOG, V096_REF):
        if not p.exists():
            raise FileNotFoundError(
                f"missing {p} -- run: env -u PYTHONPATH .venv/bin/python -m scripts.fetch_upgrade_data"
            )

    print("Loading v0.9.6 model_replay reference modelV2 trace...")
    ref = load_regen_reference(V096_REF)
    print(f"  {len(ref)} reference frames (frameId range {min(ref)}..{max(ref)})")

    print("Loading liveCalibration (CI seg-6 source rlog)...")
    rpy = latest_rpy_calib(V096_SRC_RLOG)
    print(f"  rpyCalib: [{rpy[0]:+.5f}, {rpy[1]:+.5f}, {rpy[2]:+.5f}] rad")

    print("Building warp matrices (comma 3 / _ar_ox_config narrow road cam)...")
    warp_y, warp_uv = build_warps(rpy)
    print(f"  warp_y: \n{warp_y}")
    print(f"  warp_uv: \n{warp_uv}")

    print("\nDeriving hevc-index -> reference-frameId mapping (structural)...")
    offset = derive_frame_offset(V096_SRC_RLOG, ref)

    print(f"\nInitializing v0.9.6 ModelStateMirror from {V096_MODEL.name} (zero state)...")
    state = build_mirror(V096_MODEL)
    print(f"  nav inputs exposed + fed (zeroed fp16): {state._has_nav}")
    print(f"  EP in use: {state.session.get_providers()[0]}")

    deltas, ours = run_parity(state, V096_SRC_HEVC, ref, warp_y, warp_uv, offset, n_total=1200)
    pct = report_parity(deltas, ours)
    return 0 if pct >= ACCEPT_PCT else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="supercombo parity vs comma reference")
    ap.add_argument("--model", choices=["v097", "v096"], default="v097",
                    help="v097 (default, Subaru) or v096 (CI seg-6 second-model route)")
    args = ap.parse_args(argv)
    if args.model == "v096":
        return run_v096()
    return run_v097()


if __name__ == "__main__":
    sys.exit(main())
