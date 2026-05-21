"""Domain-gap study: does openpilot supercombo meaningfully see CARLA imagery?

Step 4b found the model emits a flat +0.53 m/s2 (its zero-INPUT default) on
clean CARLA road, vs +0.06 on real comma footage. This script quantifies the
gap and tests whether cheap sim-to-real preprocessing closes it.

Part A  image statistics of the warped model frame: luma, contrast, sharpness.
Part B  model response: warm a fresh ModelStateMirror on each domain, read the
        steady-state accel@t0 of the last 30 frames.
Part C  perturb the CARLA frames (sensor noise, blur, both) and re-measure —
        if a perturbation pulls CARLA off +0.53 the gap is "sim too clean".

No CARLA server needed: real frames decode from the Subaru HEVC, CARLA frames
load from data/domain_gap/carla_rgb.npy (run scripts/capture_carla_frames.py
first).
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from src.decode_hevc import yuv_frame_iter
from src.preprocessor import rgb_to_yuv420
from src.run_parity import SRC_HEVC, SRC_RLOG, build_warps, latest_rpy_calib
from src.sim_preprocessor import build_sim_warps
from src.state import build_session, load_output_slices, long_accel_t0
from src.warped_preprocessor import stack_pair, warp_yuv_to_model, yuv_to_6ch

CARLA_NPY = Path(__file__).resolve().parents[1] / "data" / "domain_gap" / "carla_rgb.npy"
N = 170
LAST = 30  # steady-state window
CROP = (slice(32, 96), slice(64, 192))  # central 64x128 of a 128x256 plane (skip warp borders)


def model_y(six: np.ndarray) -> np.ndarray:
    """Reassemble a 128x256 luma image from the 4 Y quarter-planes of a 6ch frame."""
    return six[0]  # ch0 = Y[0::2,0::2]; one quarter-plane is fine for relative stats


def frame_stats(six_list: list[np.ndarray]) -> dict:
    luma, contrast, sharp = [], [], []
    for six in six_list:
        y = model_y(six)[CROP].astype(np.float64)
        luma.append(float(y.mean()))
        contrast.append(float(y.std()))
        sharp.append(float(cv2.Laplacian(y, cv2.CV_64F).var()))
    return {"luma": np.mean(luma), "contrast": np.mean(contrast), "sharp": np.mean(sharp)}


def warm_and_read(six_list: list[np.ndarray], sess, slices) -> tuple[float, float]:
    """Fresh ModelStateMirror, warm on the whole list, return (mean, std) accel of
    the last LAST frames."""
    from src.state import ModelStateMirror
    state = ModelStateMirror(session=sess, output_slices=slices)
    accels = []
    prev = None
    for six in six_list:
        if prev is not None:
            inp = stack_pair(prev, six)
            accels.append(long_accel_t0(state.run(inp, inp)))
        prev = six
    tail = np.array(accels[-LAST:])
    return float(tail.mean()), float(tail.std())


def load_real() -> list[np.ndarray]:
    rpy = latest_rpy_calib(SRC_RLOG)
    warp_y, warp_uv = build_warps(rpy)
    out = []
    for k, (Y, U, V) in enumerate(yuv_frame_iter(SRC_HEVC, 1928, 1208)):
        if k >= N:
            break
        Ym, Um, Vm = warp_yuv_to_model(Y, U, V, warp_y, warp_uv)
        out.append(yuv_to_6ch(Ym, Um, Vm))
    return out


def carla_six(rgb_frames: np.ndarray, perturb=None) -> list[np.ndarray]:
    warp_y, warp_uv = build_sim_warps()
    out = []
    for rgb in rgb_frames:
        img = perturb(rgb) if perturb else rgb
        Y, U, V = rgb_to_yuv420(np.ascontiguousarray(img))
        Ym, Um, Vm = warp_yuv_to_model(Y, U, V, warp_y, warp_uv)
        out.append(yuv_to_6ch(Ym, Um, Vm))
    return out


# --- perturbations (applied to CARLA RGB, mimicking real-camera characteristics) ---

def add_noise(sigma: float):
    rng = np.random.default_rng(0)
    def f(rgb):
        n = rng.normal(0, sigma, rgb.shape)
        return np.clip(rgb.astype(np.float32) + n, 0, 255).astype(np.uint8)
    return f


def add_blur(ksize: int):
    def f(rgb):
        return cv2.GaussianBlur(rgb, (ksize, ksize), 0)
    return f


def add_noise_blur(sigma: float, ksize: int):
    nf, bf = add_noise(sigma), add_blur(ksize)
    return lambda rgb: nf(bf(rgb))


def photometric_match(six_list: list[np.ndarray], src: dict, dst: dict) -> list[np.ndarray]:
    """Rescale the Y channels (0-3) of each 6ch frame from `src` luma/contrast to
    `dst` — a global brightness + contrast match, the cheapest possible
    sim-to-real photometric correction."""
    gain = dst["contrast"] / src["contrast"]
    out = []
    for six in six_list:
        m = six.copy()
        m[0:4] = np.clip((m[0:4] - src["luma"]) * gain + dst["luma"], 0, 255)
        out.append(m.astype(np.float32))
    return out


def main() -> int:
    if not CARLA_NPY.exists():
        print(f"missing {CARLA_NPY} — run scripts/capture_carla_frames.py first")
        return 1

    print("Loading real Subaru frames (decode + calib warp) ...")
    real = load_real()
    print(f"  {len(real)} real model frames")

    print("Loading CARLA frames ...")
    carla_rgb = np.load(CARLA_NPY)[:N]
    carla = carla_six(carla_rgb)
    print(f"  {len(carla)} CARLA model frames")

    print("\n=== PART A — warped model-frame image statistics (central crop) ===")
    rs, cs = frame_stats(real), frame_stats(carla)
    print(f"  {'metric':<12} {'real':>10} {'CARLA':>10} {'CARLA/real':>12}")
    for key, label in [("luma", "luma mean"), ("contrast", "contrast(std)"),
                       ("sharp", "sharpness")]:
        ratio = cs[key] / rs[key] if rs[key] else float("nan")
        print(f"  {label:<12} {rs[key]:>10.2f} {cs[key]:>10.2f} {ratio:>11.2f}x")

    print("\n=== PART B — supercombo steady-state accel@t0 (fresh warm, last 30) ===")
    sess, slices = build_session(), load_output_slices()
    print("  (first run triggers ~28 s PTX JIT) ...")
    r_mean, r_std = warm_and_read(real, sess, slices)
    c_mean, c_std = warm_and_read(carla, sess, slices)
    print(f"  real footage   : {r_mean:+.4f} +- {r_std:.4f} m/s^2")
    print(f"  CARLA (raw)    : {c_mean:+.4f} +- {c_std:.4f} m/s^2")
    print(f"  zero-input default is ~+0.535 (Gate-3). CARLA raw == default? "
          f"{'YES' if abs(c_mean - 0.535) < 0.03 else 'no'}")

    print("\n=== PART C — does sim-to-real preprocessing move CARLA off the default? ===")
    variants = [
        ("noise sigma=10", add_noise(10.0)),
        ("noise sigma=25", add_noise(25.0)),
        ("blur k=7", add_blur(7)),
        ("noise25 + blur7", add_noise_blur(25.0, 7)),
    ]
    print(f"  {'variant':<20} {'accel mean':>12} {'std':>9}  {'vs raw':>9}")
    for label, fn in variants:
        m, s = warm_and_read(carla_six(carla_rgb, fn), sess, slices)
        print(f"  {label:<20} {m:>+12.4f} {s:>9.4f}  {m - c_mean:>+9.4f}")
    # photometric match (operates on the 6ch frame, not RGB)
    matched = photometric_match(carla, cs, rs)
    m, s = warm_and_read(matched, sess, slices)
    print(f"  {'luma+contrast match':<20} {m:>+12.4f} {s:>9.4f}  {m - c_mean:>+9.4f}")

    print("\n=== VERDICT ===")
    gap = abs(c_mean - r_mean)
    is_default = abs(c_mean - 0.535) < 0.03
    print(f"  real vs CARLA accel gap : {gap:.3f} m/s^2")
    if is_default:
        print("  CARLA raw sits on the model's zero-input default — supercombo is")
        print("  effectively blind to clean CARLA imagery. See Part C for whether")
        print("  noise/blur augmentation recovers a response.")
    else:
        print("  CARLA raw is NOT the bare default — the model does respond to sim.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
