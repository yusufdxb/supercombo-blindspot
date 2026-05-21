"""RGB -> YUV420 conversion (BT.601 limited range).

Per openpilot/selfdrive/modeld/models/README.md (v0.9.7), supercombo expects
12-channel images shaped (1, 12, 128, 256). The 6-channel split, the warp into
the medmodel frame, and the two-frame stack live in `warped_preprocessor`
(the parity-verified Step 3.5 path). This module owns only the colour-space
conversion that feeds that path.

Uses BT.601 limited-range matrix (matches what comma's camera ISP emits,
which is what the model was trained on).

IMPORTANT: supercombo's image input is UNNORMALIZED. `loadyuv.cl` does
`convert_float8()` with no scaling, so the model frame is raw uint8 cast to
float in [0, 255]. Do NOT divide by 255 anywhere on this path. The 6-channel
packing that enforces this is `warped_preprocessor.yuv_to_6ch` -- that is the
single source of truth for the model-input layout.
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
