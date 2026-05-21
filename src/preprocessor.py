"""RGB → YUV420 → 6-channel split → two-frame stack.

Per openpilot/selfdrive/modeld/models/README.md (v0.9.7), supercombo expects
12-channel images shaped (1, 12, 128, 256), where channels 0..5 are the
previous frame and 6..11 are the current frame. Each 6-channel block is:
  0: Y[::2, ::2]
  1: Y[::2, 1::2]
  2: Y[1::2, ::2]
  3: Y[1::2, 1::2]
  4: U at half-res
  5: V at half-res

The source image is 256H × 512W RGB. After downsampling, each plane is
128 × 256.

Uses BT.601 limited-range matrix (matches what comma's camera ISP emits,
which is what the model was trained on). If color drift causes parity-check
issues at Step 3.5, the conversion matrix is the first thing to revisit.
"""

from __future__ import annotations

import numpy as np

EXPECTED_H = 256
EXPECTED_W = 512


def rgb_to_yuv420(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RGB[H, W, 3] uint8 -> (Y[H, W], U[H/2, W/2], V[H/2, W/2]) uint8, BT.601 limited."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError(f"expected RGB uint8 HxWx3, got {rgb.shape} {rgb.dtype}")
    H, W = rgb.shape[:2]
    if H % 2 or W % 2:
        raise ValueError(f"H,W must be even (got {H},{W})")

    r = rgb[:, :, 0].astype(np.float32)
    g = rgb[:, :, 1].astype(np.float32)
    b = rgb[:, :, 2].astype(np.float32)

    # BT.601 limited-range (Y in [16,235], UV in [16,240])
    Y = 0.257 * r + 0.504 * g + 0.098 * b + 16.0
    U = -0.148 * r - 0.291 * g + 0.439 * b + 128.0
    V = 0.439 * r - 0.368 * g - 0.071 * b + 128.0

    Y = np.clip(Y, 0, 255).astype(np.uint8)
    U_full = np.clip(U, 0, 255)
    V_full = np.clip(V, 0, 255)

    # 4:2:0 chroma subsample by 2x2 averaging
    U_half = ((U_full[0::2, 0::2] + U_full[0::2, 1::2] +
               U_full[1::2, 0::2] + U_full[1::2, 1::2]) / 4.0).astype(np.uint8)
    V_half = ((V_full[0::2, 0::2] + V_full[0::2, 1::2] +
               V_full[1::2, 0::2] + V_full[1::2, 1::2]) / 4.0).astype(np.uint8)
    return Y, U_half, V_half


def yuv420_to_6ch(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """(Y[H, W], U[H/2, W/2], V[H/2, W/2]) -> (6, H/2, W/2) float32 in [0, 1]."""
    out = np.stack([
        Y[0::2, 0::2],
        Y[0::2, 1::2],
        Y[1::2, 0::2],
        Y[1::2, 1::2],
        U,
        V,
    ], axis=0).astype(np.float32) / 255.0
    return out


def rgb_pair_to_input(prev_rgb: np.ndarray, curr_rgb: np.ndarray) -> np.ndarray:
    """Two RGB frames (256 x 512 x 3 uint8) -> (1, 12, 128, 256) float32."""
    if prev_rgb.shape != (EXPECTED_H, EXPECTED_W, 3):
        raise ValueError(f"prev_rgb shape {prev_rgb.shape} != ({EXPECTED_H}, {EXPECTED_W}, 3)")
    if curr_rgb.shape != (EXPECTED_H, EXPECTED_W, 3):
        raise ValueError(f"curr_rgb shape {curr_rgb.shape} != ({EXPECTED_H}, {EXPECTED_W}, 3)")
    prev_6 = yuv420_to_6ch(*rgb_to_yuv420(prev_rgb))
    curr_6 = yuv420_to_6ch(*rgb_to_yuv420(curr_rgb))
    return np.concatenate([prev_6, curr_6], axis=0)[np.newaxis, ...]
