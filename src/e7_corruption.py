"""E7: ImageNet-C corruption sweep on real driving frames.

E6 calibrated a rolling-spread detector on the 512-D recurrent feature that
fires at alpha=0.55 on the CARLA interpolation sweep. Reviewers will ask
whether this monitor is fitted to CARLA. E7 answers: apply 15 standard
ImageNet-C corruptions (Hendrycks & Dietterich, ICLR 2019) at 5 severity
levels to the existing real comma driving frames, re-run supercombo with
proper recurrent state handling, and evaluate E6 + all baselines.

The corruptions are applied to the RAW RGB image *before* any preprocessing
(before YUV conversion). The full pipeline is:

    RGB -> corruption(severity) -> YUV420 BT.601 -> warp -> 6ch -> 2-frame stack -> model

This module implements all 15 corruptions using only numpy, scipy, PIL, and
cv2 (no imagecorruptions dependency).

    python -m src.e7_corruption --collect   # re-run model on corrupted frames
    python -m src.e7_corruption             # analysis from cache
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "report" / "e7_collected.npz"
FIG_DIR = ROOT / "report" / "figures"
RESULTS_MD = ROOT / "report" / "e7_results.md"

# Use the same frame budget and warmup as the main teardown.
from src.teardown import N, WARMUP

SUBARU_HEVC = DATA / "subaru_source" / "fcamera.hevc"
SUBARU_RLOG = DATA / "subaru_source" / "rlog.bz2"

SEVERITIES = [1, 2, 3, 4, 5]

# -----------------------------------------------------------------------
# ImageNet-C corruptions (Hendrycks & Dietterich, ICLR 2019)
#
# Each corruption function takes (img: np.ndarray HxWx3 uint8, severity: int)
# and returns np.ndarray HxWx3 uint8.
# -----------------------------------------------------------------------


def _clamp(img: np.ndarray) -> np.ndarray:
    """Clip to [0, 255] and cast to uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)


def gaussian_noise(img: np.ndarray, severity: int) -> np.ndarray:
    """Additive Gaussian noise. Severity scales the standard deviation."""
    sigma = [0.08, 0.12, 0.18, 0.26, 0.38][severity - 1]
    noise = np.random.RandomState(42).randn(*img.shape) * (sigma * 255)
    return _clamp(img.astype(np.float32) + noise)


def shot_noise(img: np.ndarray, severity: int) -> np.ndarray:
    """Poisson (shot) noise. Severity scales the lambda divisor."""
    lam = [60, 25, 12, 5, 3][severity - 1]
    noisy = np.random.RandomState(42).poisson(
        img.astype(np.float32) / 255.0 * lam
    ) / lam * 255.0
    return _clamp(noisy)


def impulse_noise(img: np.ndarray, severity: int) -> np.ndarray:
    """Salt-and-pepper noise. Severity scales the fraction of affected pixels."""
    amount = [0.03, 0.06, 0.09, 0.17, 0.27][severity - 1]
    rng = np.random.RandomState(42)
    out = img.copy()
    # Salt
    n_salt = int(np.ceil(amount * img.size / 2))
    coords = tuple(rng.randint(0, d, n_salt) for d in img.shape)
    out[coords] = 255
    # Pepper
    n_pepper = int(np.ceil(amount * img.size / 2))
    coords = tuple(rng.randint(0, d, n_pepper) for d in img.shape)
    out[coords] = 0
    return out


def defocus_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Disk (defocus) blur via a circular kernel."""
    radius = [3, 4, 6, 8, 10][severity - 1]
    kernel_size = 2 * radius + 1
    # Create circular kernel
    Y, X = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    mask = (X * X + Y * Y <= radius * radius).astype(np.float32)
    mask /= mask.sum()
    out = cv2.filter2D(img, -1, mask)
    return out.astype(np.uint8)


def motion_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Linear motion blur. Severity scales kernel size."""
    ksize = [10, 15, 20, 25, 30][severity - 1]
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(img, -1, kernel).astype(np.uint8)


def glass_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Glass blur: Gaussian blur + vectorized pixel-shuffle.

    The original Hendrycks implementation uses a nested Python loop over
    every pixel. We vectorize the shuffle with numpy index arrays for
    acceptable speed on 1928x1208 frames.
    """
    sigma = [0.7, 0.9, 1.0, 1.1, 1.5][severity - 1]
    iters = [1, 1, 1, 2, 2][severity - 1]
    delta = [1, 2, 2, 3, 4][severity - 1]
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    rng = np.random.RandomState(42)
    h, w = blurred.shape[:2]
    for _ in range(iters):
        # Generate random offsets for the interior region
        inner_h = h - 2 * delta
        inner_w = w - 2 * delta
        di = rng.randint(-delta, delta + 1, size=(inner_h, inner_w))
        dj = rng.randint(-delta, delta + 1, size=(inner_h, inner_w))
        # Build source index arrays
        rows = np.arange(delta, h - delta)[:, None] + di
        cols = np.arange(delta, w - delta)[None, :] + dj
        out = blurred.copy()
        out[delta:h - delta, delta:w - delta] = blurred[rows, cols]
        blurred = out
    return out


def zoom_blur(img: np.ndarray, severity: int) -> np.ndarray:
    """Zoom blur: average of zoomed-in versions of the image."""
    zoom_factors = [
        np.arange(1, 1.11, 0.01),
        np.arange(1, 1.16, 0.01),
        np.arange(1, 1.21, 0.02),
        np.arange(1, 1.26, 0.02),
        np.arange(1, 1.31, 0.03),
    ][severity - 1]
    h, w, c = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for zf in zoom_factors:
        zh, zw = int(h * zf), int(w * zf)
        zoomed = cv2.resize(img, (zw, zh), interpolation=cv2.INTER_LINEAR)
        # Center crop back to original size
        top = (zh - h) // 2
        left = (zw - w) // 2
        out += zoomed[top:top + h, left:left + w].astype(np.float32)
    out /= len(zoom_factors)
    return _clamp(out)


def fog(img: np.ndarray, severity: int) -> np.ndarray:
    """Fog: diamond-square plasma fractal blended over the image.

    We use a simpler approximation: multi-octave Perlin-like noise via
    Gaussian-filtered random fields, blended per the Hendrycks formula
    c = (1 - t) * img + t * 255, where t is the fog density map.
    """
    strength = [1.5, 2.0, 2.5, 2.8, 3.0][severity - 1]
    thickness = [2.0, 2.0, 1.75, 1.5, 1.4][severity - 1]
    h, w = img.shape[:2]
    rng = np.random.RandomState(42)
    fog_layer = np.zeros((h, w), dtype=np.float32)
    for octave in range(4):
        scale = 2 ** octave
        noise = rng.randn(h // (2 * scale) + 1, w // (2 * scale) + 1).astype(np.float32)
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        fog_layer += noise / (scale + 1)
    # Normalize to [0, 1]
    fog_layer = (fog_layer - fog_layer.min()) / (fog_layer.max() - fog_layer.min() + 1e-8)
    fog_layer = fog_layer ** thickness
    t = np.clip(fog_layer * strength / 5.0, 0, 1)[:, :, None]
    out = (1 - t) * img.astype(np.float32) + t * 255.0
    return _clamp(out)


def snow(img: np.ndarray, severity: int) -> np.ndarray:
    """Snow overlay: brightness shift + additive noise pattern."""
    brightness = [0.1, 0.15, 0.2, 0.25, 0.3][severity - 1]
    flake_strength = [0.3, 0.5, 0.6, 0.7, 0.8][severity - 1]
    h, w = img.shape[:2]
    rng = np.random.RandomState(42)
    # Brightness increase (overcast sky)
    out = img.astype(np.float32) + brightness * 255
    # Snowflake layer: sparse white dots + motion blur
    flakes = rng.uniform(0, 1, (h, w)) < 0.01 * (severity + 1)
    flake_img = (flakes * 255).astype(np.float32)
    ksize = 5 + severity * 2
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    # Diagonal motion blur to simulate falling snow
    for i in range(ksize):
        kernel[i, min(i, ksize - 1)] = 1.0 / ksize
    flake_img = cv2.filter2D(flake_img, -1, kernel)
    out += flake_img[:, :, None] * flake_strength
    return _clamp(out)


def frost(img: np.ndarray, severity: int) -> np.ndarray:
    """Frost: crystalline overlay pattern blended with the image.

    Instead of loading an external frost image (as the original does),
    we generate a procedural frost pattern using Gaussian noise at
    multiple scales.
    """
    alpha = [0.2, 0.3, 0.4, 0.5, 0.6][severity - 1]
    h, w = img.shape[:2]
    rng = np.random.RandomState(42)
    # Procedural frost: multi-scale Gaussian noise, thresholded
    frost_layer = np.zeros((h, w), dtype=np.float32)
    for s in [64, 32, 16, 8]:
        small = rng.randn(h // s + 1, w // s + 1).astype(np.float32)
        up = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        frost_layer += up
    frost_layer = (frost_layer - frost_layer.min()) / (frost_layer.max() - frost_layer.min() + 1e-8)
    frost_layer = (frost_layer * 255).astype(np.float32)
    frost_rgb = np.stack([frost_layer] * 3, axis=-1)
    out = (1 - alpha) * img.astype(np.float32) + alpha * frost_rgb
    return _clamp(out)


def brightness(img: np.ndarray, severity: int) -> np.ndarray:
    """Increase brightness in HSV space."""
    delta = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + delta * 255, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def contrast(img: np.ndarray, severity: int) -> np.ndarray:
    """Reduce contrast by blending toward the mean gray."""
    factor = [0.4, 0.3, 0.2, 0.15, 0.1][severity - 1]
    mean = img.astype(np.float32).mean()
    out = factor * img.astype(np.float32) + (1 - factor) * mean
    return _clamp(out)


def elastic_transform(img: np.ndarray, severity: int) -> np.ndarray:
    """Elastic deformation (Simard et al. 2003), ImageNet-C parameterization."""
    alpha_val = [244 * 2, 244 * 2, 244 * 2, 244 * 2, 244 * 2][severity - 1]
    sigma_val = [244 * 0.21, 244 * 0.19, 244 * 0.17, 244 * 0.15, 244 * 0.13][severity - 1]
    h, w = img.shape[:2]
    rng = np.random.RandomState(42)
    dx = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma_val
    ) * alpha_val
    dy = cv2.GaussianBlur(
        rng.uniform(-1, 1, (h, w)).astype(np.float32),
        (0, 0), sigma_val
    ) * alpha_val
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


def pixelate(img: np.ndarray, severity: int) -> np.ndarray:
    """Pixelate by downscaling and upscaling."""
    factor = [0.6, 0.5, 0.4, 0.3, 0.25][severity - 1]
    h, w = img.shape[:2]
    small_h, small_w = max(1, int(h * factor)), max(1, int(w * factor))
    small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def jpeg_compression(img: np.ndarray, severity: int) -> np.ndarray:
    """JPEG compression artifacts. Severity decreases quality."""
    quality = [25, 18, 15, 10, 7][severity - 1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    # cv2 uses BGR internally but encode/decode is format-agnostic for quality
    _, buf = cv2.imencode('.jpg', img, encode_param)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # If input was RGB, cv2.imencode treats it as BGR -> decode gives BGR
    # We need to be consistent: treat img as RGB throughout, so convert
    # both ways through BGR for the JPEG round-trip.
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode('.jpg', bgr, encode_param)
    decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


# Registry of all 15 corruptions in the canonical ImageNet-C order.
CORRUPTIONS: dict[str, Callable[[np.ndarray, int], np.ndarray]] = {
    "gaussian_noise": gaussian_noise,
    "shot_noise": shot_noise,
    "impulse_noise": impulse_noise,
    "defocus_blur": defocus_blur,
    "motion_blur": motion_blur,
    "glass_blur": glass_blur,
    "zoom_blur": zoom_blur,
    "fog": fog,
    "snow": snow,
    "frost": frost,
    "brightness": brightness,
    "contrast": contrast,
    "elastic_transform": elastic_transform,
    "pixelate": pixelate,
    "jpeg_compression": jpeg_compression,
}

CORRUPTION_NAMES = list(CORRUPTIONS.keys())


# -----------------------------------------------------------------------
# Real-frame loading: decode HEVC -> YUV -> invert to RGB for corruption
# -----------------------------------------------------------------------

def _load_real_rgb_frames(hevc_path: Path, rlog_path: Path,
                          n: int) -> list[np.ndarray]:
    """Load n real-driving frames as RGB uint8 (1208x1928x3).

    We need the raw RGB *before* any YUV conversion so corruptions apply
    to the perceptual domain. The comma HEVC is natively YUV420, so we
    decode YUV and convert to RGB via BT.601. This is not a round-trip
    loss because the corruption pipeline will convert back to YUV420
    through the same BT.601 path the real pipeline uses.
    """
    from src.decode_hevc import yuv_frame_iter

    frames: list[np.ndarray] = []
    for k, (Y, U, V) in enumerate(yuv_frame_iter(hevc_path)):
        if k >= n:
            break
        # Upsample U, V to full resolution
        U_full = cv2.resize(U, (Y.shape[1], Y.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        V_full = cv2.resize(V, (Y.shape[1], Y.shape[0]),
                            interpolation=cv2.INTER_LINEAR)
        # BT.601 limited range YUV -> RGB
        y = Y.astype(np.float32) - 16.0
        u = U_full.astype(np.float32) - 128.0
        v = V_full.astype(np.float32) - 128.0
        r = 1.164 * y + 1.596 * v
        g = 1.164 * y - 0.392 * u - 0.813 * v
        b = 1.164 * y + 2.017 * u
        rgb = np.stack([r, g, b], axis=-1)
        frames.append(np.clip(rgb, 0, 255).astype(np.uint8))
    return frames


def _rgb_to_model_six(rgb: np.ndarray, warp_y: np.ndarray,
                      warp_uv: np.ndarray) -> np.ndarray:
    """Single RGB frame (HxWx3 uint8) -> model 6-channel input (6, 128, 256).

    Pipeline: RGB -> YUV420 BT.601 -> warp to medmodel -> 6ch split.
    """
    from src.preprocessor import rgb_to_yuv420
    from src.warped_preprocessor import warp_yuv_to_model, yuv_to_6ch

    Y, U, V = rgb_to_yuv420(rgb)
    Y_m, U_m, V_m = warp_yuv_to_model(Y, U, V, warp_y, warp_uv)
    return yuv_to_6ch(Y_m, U_m, V_m)


# -----------------------------------------------------------------------
# Model collection
# -----------------------------------------------------------------------

def _corrupt_one_frame(args: tuple) -> np.ndarray:
    """Worker: corrupt one RGB frame and convert to model 6-ch input.
    Runs in a subprocess to parallelize across CPU cores."""
    rgb, corruption_name, severity, warp_y, warp_uv = args
    corrupt_fn = CORRUPTIONS[corruption_name]
    corrupted_rgb = corrupt_fn(rgb, severity)
    return _rgb_to_model_six(corrupted_rgb, warp_y, warp_uv)


_POOL_WORKERS = min(16, __import__("os").cpu_count() or 4)


def _collect_corrupted(
    rgb_frames: list[np.ndarray],
    corruption_name: str,
    severity: int,
    sess,
    slices,
    warp_y: np.ndarray,
    warp_uv: np.ndarray,
) -> dict[str, np.ndarray]:
    """Apply a single corruption at a single severity to all RGB frames,
    run supercombo with proper recurrent-state handling, return collected
    outputs (same schema as probe_model.collect)."""
    from multiprocessing import Pool
    from src.probe_model import collect

    work = [(rgb, corruption_name, severity, warp_y, warp_uv)
            for rgb in rgb_frames]
    with Pool(_POOL_WORKERS) as pool:
        six_list = pool.map(_corrupt_one_frame, work)

    return collect(six_list, sess, slices)


def collect_all(
    corruption_names: list[str] | None = None,
    severities: list[int] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """Run supercombo on every (corruption, severity) combination.

    Returns {f"{corruption}__{severity}": collected_dict, ...}
    Also includes "clean__0" for the uncorrupted baseline.
    """
    from src.probe_model import collect, _calib_warps
    from src.state import build_session, load_output_slices

    if corruption_names is None:
        corruption_names = CORRUPTION_NAMES
    if severities is None:
        severities = SEVERITIES

    sess = build_session()
    slices = load_output_slices()
    warp_y, warp_uv = _calib_warps(SUBARU_RLOG)

    print(f"Loading {N} real RGB frames from {SUBARU_HEVC.name} ...")
    rgb_frames = _load_real_rgb_frames(SUBARU_HEVC, SUBARU_RLOG, N)
    print(f"  loaded {len(rgb_frames)} frames")

    results: dict[str, dict[str, np.ndarray]] = {}

    # Clean baseline (uncorrupted)
    print("Collecting clean baseline ...")
    six_clean = [_rgb_to_model_six(f, warp_y, warp_uv) for f in rgb_frames]
    results["clean__0"] = collect(six_clean, sess, slices)
    print("  clean__0 done")

    # Corrupted conditions
    total = len(corruption_names) * len(severities)
    done = 0
    for cname in corruption_names:
        for sev in severities:
            done += 1
            key = f"{cname}__{sev}"
            print(f"  [{done}/{total}] {key} ...")
            results[key] = _collect_corrupted(
                rgb_frames, cname, sev, sess, slices, warp_y, warp_uv
            )

    return results


def save_cache(path: Path, collected: dict[str, dict[str, np.ndarray]]) -> None:
    """Persist collected outputs as one compressed .npz."""
    flat: dict[str, np.ndarray] = {}
    for condition_key, d in collected.items():
        for array_key, arr in d.items():
            flat[f"{condition_key}__{array_key}"] = arr
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_cache(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """Inverse of save_cache."""
    z = np.load(path)
    out: dict[str, dict[str, np.ndarray]] = {}
    for full_key in z.files:
        # Keys are "corruption__severity__array_name"
        parts = full_key.split("__")
        # First two parts are the condition key, rest is array name
        condition = f"{parts[0]}__{parts[1]}"
        array_name = "__".join(parts[2:])
        out.setdefault(condition, {})[array_name] = z[full_key]
    return out


# -----------------------------------------------------------------------
# Analysis: E6 rolling-spread detector on corrupted conditions
# -----------------------------------------------------------------------

def _id_hidden() -> np.ndarray:
    """Concatenate subaru + ram hidden_state from teardown_collected.npz."""
    d = np.load(str(ROOT / "report" / "teardown_collected.npz"))
    return np.concatenate([d["subaru__hidden_state"],
                           d["ram__hidden_state"]], axis=0)


def _id_hidden_by_corpus() -> dict[str, np.ndarray]:
    d = np.load(str(ROOT / "report" / "teardown_collected.npz"))
    return {"subaru": d["subaru__hidden_state"],
            "ram": d["ram__hidden_state"]}


def analyze(collected: dict[str, dict[str, np.ndarray]],
            window: int = 30, e6_percentile: float = 1.0,
            baseline_percentile: float = 99.0,
            n_bootstrap: int = 1000) -> dict:
    """Compute E6 + baseline detection metrics for each condition.

    Returns a dict with per-condition results, ready for figure + report.
    """
    from src.e6_detector import rolling_spread, calibrate_threshold
    from src.baselines import (APPLICABLE_BASELINES, _score,
                               calibrate_threshold_high)
    from src import metrics

    # ID calibration data
    id_hidden = _id_hidden()
    id_spreads = rolling_spread(id_hidden, window)
    e6_thr = calibrate_threshold(id_spreads, e6_percentile)

    # Baseline thresholds calibrated on ID data
    baseline_thrs: dict[str, float] = {}
    for bname in APPLICABLE_BASELINES:
        id_scores = _score(bname, id_hidden, id_hidden)
        baseline_thrs[bname] = calibrate_threshold_high(id_scores, baseline_percentile)

    # Parse conditions
    conditions: list[tuple[str, int]] = []
    for key in sorted(collected.keys()):
        parts = key.split("__")
        cname, sev = parts[0], int(parts[1])
        conditions.append((cname, sev))

    results: dict[str, dict] = {}

    for cname, sev in conditions:
        key = f"{cname}__{sev}"
        d = collected[key]

        # Discard warmup frames
        hidden = d["hidden_state"][WARMUP:]
        T = len(hidden)
        if T < window:
            results[key] = {"corruption": cname, "severity": sev,
                            "n_frames": T, "skip": True}
            continue

        cond_result: dict = {
            "corruption": cname, "severity": sev,
            "n_frames": T, "skip": False,
        }

        # --- E6: rolling-spread detector ---
        spreads = rolling_spread(hidden, window)
        valid_spreads = spreads[~np.isnan(spreads)]
        if len(valid_spreads) > 0:
            # E6 uses lower-is-OOD convention
            e6_fired_frac = float(np.mean(valid_spreads < e6_thr))
        else:
            e6_fired_frac = float("nan")
        cond_result["e6_fired_fraction"] = e6_fired_frac
        cond_result["e6_threshold"] = e6_thr

        # --- Threshold-free metrics for E6 ---
        # For threshold-free: need ID scores + OOD scores with labels.
        # E6 is lower-is-OOD, so negate spread for "higher=more OOD".
        id_valid = id_spreads[~np.isnan(id_spreads)]
        ood_scores_e6 = -valid_spreads
        id_scores_e6 = -id_valid
        if len(ood_scores_e6) > 0 and len(id_scores_e6) > 0:
            scores_e6 = np.concatenate([id_scores_e6, ood_scores_e6])
            labels_e6 = np.concatenate([
                np.zeros(len(id_scores_e6), dtype=np.int64),
                np.ones(len(ood_scores_e6), dtype=np.int64),
            ])
            cond_result["e6_auroc"] = metrics.auroc(scores_e6, labels_e6)
            cond_result["e6_aupr"] = metrics.aupr(scores_e6, labels_e6)
            cond_result["e6_fpr95"] = metrics.fpr_at_tpr(scores_e6, labels_e6, 0.95)
            # Bootstrap CIs
            for mname, mfn in [("auroc", metrics.auroc), ("aupr", metrics.aupr),
                               ("fpr95", lambda s, l: metrics.fpr_at_tpr(s, l, 0.95))]:
                mean, lo, hi = metrics.bootstrap_ci(
                    mfn, scores_e6, labels_e6, n_bootstrap=n_bootstrap, seed=42
                )
                cond_result[f"e6_{mname}_ci"] = (lo, hi)
        else:
            for k in ["e6_auroc", "e6_aupr", "e6_fpr95"]:
                cond_result[k] = float("nan")

        # --- Feature-space baselines ---
        for bname in APPLICABLE_BASELINES:
            b_scores_ood = _score(bname, id_hidden, hidden)
            b_scores_id = _score(bname, id_hidden, id_hidden)
            thr = baseline_thrs[bname]

            # Fired fraction at calibrated threshold
            valid_b = b_scores_ood[~np.isnan(b_scores_ood)]
            if len(valid_b) > 0:
                fired = float(np.mean(valid_b > thr))
            else:
                fired = float("nan")
            cond_result[f"{bname}_fired_fraction"] = fired
            cond_result[f"{bname}_threshold"] = thr

            # Threshold-free metrics
            if len(valid_b) > 0 and len(b_scores_id) > 0:
                scores_b = np.concatenate([b_scores_id, valid_b])
                labels_b = np.concatenate([
                    np.zeros(len(b_scores_id), dtype=np.int64),
                    np.ones(len(valid_b), dtype=np.int64),
                ])
                cond_result[f"{bname}_auroc"] = metrics.auroc(scores_b, labels_b)
                cond_result[f"{bname}_aupr"] = metrics.aupr(scores_b, labels_b)
                cond_result[f"{bname}_fpr95"] = metrics.fpr_at_tpr(scores_b, labels_b, 0.95)
                for mname, mfn in [("auroc", metrics.auroc), ("aupr", metrics.aupr),
                                   ("fpr95", lambda s, l: metrics.fpr_at_tpr(s, l, 0.95))]:
                    mean, lo, hi = metrics.bootstrap_ci(
                        mfn, scores_b, labels_b, n_bootstrap=n_bootstrap, seed=42
                    )
                    cond_result[f"{bname}_{mname}_ci"] = (lo, hi)
            else:
                for k in [f"{bname}_auroc", f"{bname}_aupr", f"{bname}_fpr95"]:
                    cond_result[k] = float("nan")

        results[key] = cond_result

    return {"conditions": results, "e6_threshold": e6_thr,
            "baseline_thresholds": baseline_thrs,
            "window": window, "e6_percentile": e6_percentile,
            "baseline_percentile": baseline_percentile}


# -----------------------------------------------------------------------
# Output: figures + markdown report
# -----------------------------------------------------------------------

def _fig_severity_sweep(results: dict, out_dir: Path) -> None:
    """Per-corruption severity-vs-detection-rate figure for E6."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figure_style as _figure_style  # editorial-print theme
    _figure_style.apply()
    conditions = results["conditions"]

    # Group by corruption type
    by_corruption: dict[str, dict[int, dict]] = defaultdict(dict)
    for key, cond in conditions.items():
        if cond.get("skip"):
            continue
        by_corruption[cond["corruption"]][cond["severity"]] = cond

    # E6 detection rate vs severity, one subplot per corruption category
    noise_corruptions = ["gaussian_noise", "shot_noise", "impulse_noise"]
    blur_corruptions = ["defocus_blur", "motion_blur", "glass_blur", "zoom_blur"]
    weather_corruptions = ["fog", "snow", "frost", "brightness", "contrast"]
    digital_corruptions = ["elastic_transform", "pixelate", "jpeg_compression"]
    categories = [
        ("Noise", noise_corruptions),
        ("Blur", blur_corruptions),
        ("Weather", weather_corruptions),
        ("Digital", digital_corruptions),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.ravel()

    for ax, (cat_name, cat_corruptions) in zip(axes, categories):
        for cname in cat_corruptions:
            if cname not in by_corruption:
                continue
            sevs = sorted(by_corruption[cname].keys())
            # Skip severity 0 (clean) for the per-corruption plot
            sevs = [s for s in sevs if s > 0]
            if not sevs:
                continue
            rates = [by_corruption[cname][s].get("e6_fired_fraction", float("nan"))
                     for s in sevs]
            ax.plot(sevs, rates, "o-", label=cname.replace("_", " "), lw=1.5)

        ax.set_title(f"E6 Detection Rate: {cat_name} Corruptions")
        ax.set_xlabel("Severity")
        ax.set_ylabel("Fraction of frames flagged OOD")
        ax.set_xticks(SEVERITIES)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="grey", lw=0.7, ls="--", alpha=0.5)
        if ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("E7: E6 Rolling-Spread Detector on ImageNet-C Corruptions",
                 fontsize=13)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e7_severity_sweep.png", dpi=150)
    plt.close(fig)


def _fig_auroc_heatmap(results: dict, out_dir: Path) -> None:
    """Heatmap: corruption x severity, cell = E6 AUROC."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figure_style as _figure_style  # editorial-print theme
    _figure_style.apply()

    conditions = results["conditions"]
    corruption_order = [c for c in CORRUPTION_NAMES
                        if any(f"{c}__{s}" in conditions for s in SEVERITIES)]
    n_corr = len(corruption_order)
    n_sev = len(SEVERITIES)

    matrix = np.full((n_corr, n_sev), np.nan)
    for i, cname in enumerate(corruption_order):
        for j, sev in enumerate(SEVERITIES):
            key = f"{cname}__{sev}"
            if key in conditions and not conditions[key].get("skip"):
                matrix[i, j] = conditions[key].get("e6_auroc", float("nan"))

    fig, ax = plt.subplots(figsize=(8, 10), dpi=140)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_sev))
    ax.set_xticklabels([str(s) for s in SEVERITIES])
    ax.set_yticks(range(n_corr))
    ax.set_yticklabels([c.replace("_", " ") for c in corruption_order])
    ax.set_xlabel("Severity")
    ax.set_title("E7: E6 AUROC per Corruption x Severity")

    # Annotate cells
    for i in range(n_corr):
        for j in range(n_sev):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if 0.3 < val < 0.8 else "white")

    fig.colorbar(im, ax=ax, shrink=0.6, label="AUROC")
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "e7_auroc_heatmap.png", dpi=150)
    plt.close(fig)


def _write_results(results: dict, out: Path) -> None:
    """Write report/e7_results.md."""
    from src.baselines import APPLICABLE_BASELINES

    conditions = results["conditions"]
    e6_thr = results["e6_threshold"]

    lines = [
        "# E7 Results: ImageNet-C Corruption Sweep",
        "",
        "Applies 15 ImageNet-C corruptions (Hendrycks & Dietterich, ICLR 2019) "
        "at 5 severity levels to real comma driving frames. Corruptions are "
        "applied to raw RGB before YUV conversion. Supercombo is re-run with "
        "proper recurrent state handling on each corrupted sequence.",
        "",
        f"- E6 rolling-spread threshold (calibrated p={results['e6_percentile']}): "
        f"{e6_thr:.6f}",
        f"- Baseline thresholds (calibrated p={results['baseline_percentile']}): "
        + ", ".join(f"{b}={results['baseline_thresholds'][b]:.4f}"
                    for b in APPLICABLE_BASELINES),
        f"- Window: {results['window']}",
        f"- Warmup frames discarded: {WARMUP}",
        "",
        "## Detection rate at calibrated thresholds",
        "",
    ]

    # Table: corruption, severity, E6 fired fraction, baseline fired fractions
    header = "| Corruption | Severity | E6 fired |"
    sep = "|---|---|---|"
    for bname in APPLICABLE_BASELINES:
        header += f" {bname} fired |"
        sep += "---|"
    lines.append(header)
    lines.append(sep)

    for key in sorted(conditions.keys()):
        cond = conditions[key]
        if cond.get("skip"):
            continue
        row = f"| {cond['corruption']} | {cond['severity']} | "
        row += f"{cond['e6_fired_fraction']:.3f} |"
        for bname in APPLICABLE_BASELINES:
            val = cond.get(f"{bname}_fired_fraction", float("nan"))
            row += f" {val:.3f} |"
        lines.append(row)

    lines.extend(["", "## Threshold-free metrics (E6 rolling-spread)", ""])

    header2 = "| Corruption | Severity | AUROC | AUROC 95% CI | AUPR | FPR@95TPR |"
    sep2 = "|---|---|---|---|---|---|"
    lines.append(header2)
    lines.append(sep2)

    for key in sorted(conditions.keys()):
        cond = conditions[key]
        if cond.get("skip"):
            continue
        auroc_val = cond.get("e6_auroc", float("nan"))
        aupr_val = cond.get("e6_aupr", float("nan"))
        fpr95_val = cond.get("e6_fpr95", float("nan"))
        ci = cond.get("e6_auroc_ci", (float("nan"), float("nan")))
        row = (f"| {cond['corruption']} | {cond['severity']} | "
               f"{auroc_val:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
               f"{aupr_val:.4f} | {fpr95_val:.4f} |")
        lines.append(row)

    # Summary: per-corruption mean AUROC across severities
    lines.extend(["", "## Summary: mean AUROC across severities", ""])

    by_corruption: dict[str, list[float]] = defaultdict(list)
    for key, cond in conditions.items():
        if cond.get("skip") or cond["severity"] == 0:
            continue
        auroc_val = cond.get("e6_auroc", float("nan"))
        if np.isfinite(auroc_val):
            by_corruption[cond["corruption"]].append(auroc_val)

    lines.append("| Corruption | Mean AUROC | Min AUROC | Max AUROC |")
    lines.append("|---|---|---|---|")
    for cname in CORRUPTION_NAMES:
        vals = by_corruption.get(cname, [])
        if vals:
            lines.append(f"| {cname} | {np.mean(vals):.4f} | "
                         f"{min(vals):.4f} | {max(vals):.4f} |")

    lines.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="E7: ImageNet-C corruption sweep on real driving frames")
    ap.add_argument("--collect", action="store_true",
                    help="Re-run the model instead of using cached results")
    ap.add_argument("--corruptions", nargs="*", default=None,
                    help=f"Subset of corruptions to run (default: all 15)")
    ap.add_argument("--severities", nargs="*", type=int, default=None,
                    help=f"Subset of severities (default: 1-5)")
    ap.add_argument("--window", type=int, default=30,
                    help="Rolling-spread window size (default: 30)")
    ap.add_argument("--e6-percentile", type=float, default=1.0,
                    help="E6 threshold percentile (default: 1.0)")
    ap.add_argument("--baseline-percentile", type=float, default=99.0,
                    help="Baseline threshold percentile (default: 99.0)")
    ap.add_argument("--n-bootstrap", type=int, default=1000,
                    help="Bootstrap replicates for CIs (default: 1000)")
    args = ap.parse_args(argv)

    if args.collect or not CACHE.exists():
        collected = collect_all(
            corruption_names=args.corruptions,
            severities=args.severities,
        )
        save_cache(CACHE, collected)
        print(f"  cached -> {CACHE.relative_to(ROOT)}")
    else:
        print(f"Loading {CACHE.relative_to(ROOT)} ...")
        collected = load_cache(CACHE)

    # Filter to requested corruptions/severities if specified
    if args.corruptions or args.severities:
        cnames = args.corruptions or CORRUPTION_NAMES
        sevs = args.severities or SEVERITIES
        collected = {
            k: v for k, v in collected.items()
            if any(k.startswith(f"{c}__") for c in cnames + ["clean"])
            and int(k.split("__")[1]) in sevs + [0]
        }

    results = analyze(
        collected, window=args.window,
        e6_percentile=args.e6_percentile,
        baseline_percentile=args.baseline_percentile,
        n_bootstrap=args.n_bootstrap,
    )

    _fig_severity_sweep(results, FIG_DIR)
    _fig_auroc_heatmap(results, FIG_DIR)
    _write_results(results, RESULTS_MD)

    # Print summary
    print(f"\nE7 corruption sweep complete.")
    print(f"  E6 threshold: {results['e6_threshold']:.6f}")
    print(f"  Conditions analyzed: {len(results['conditions'])}")

    # Quick summary table
    print(f"\n  {'Corruption':>22s} {'Sev':>4s} {'E6 fired':>10s} {'AUROC':>8s}")
    for key in sorted(results["conditions"].keys()):
        cond = results["conditions"][key]
        if cond.get("skip"):
            continue
        print(f"  {cond['corruption']:>22s} {cond['severity']:>4d} "
              f"{cond['e6_fired_fraction']:>10.3f} "
              f"{cond.get('e6_auroc', float('nan')):>8.4f}")

    print(f"\n  figures -> {FIG_DIR.relative_to(ROOT)}/e7_*.png")
    print(f"  results -> {RESULTS_MD.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
