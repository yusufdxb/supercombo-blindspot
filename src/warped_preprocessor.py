"""Calibrated warp + YUV420 six-channel split for one camera frame.

Mirrors what `selfdrive/modeld/models/commonmodel.cc::ModelFrame::prepare` does
on the device side (OpenCL kernel applies the warp_matrix then splits Y into
four quarter-planes). Here we do it with cv2 + numpy.

Model input layout (1, 6, 128, 256) per frame, channels:
  0: Y[::2, ::2]   1: Y[::2, 1::2]
  2: Y[1::2, ::2]  3: Y[1::2, 1::2]
  4: U (half-res of model Y)
  5: V
"""

from __future__ import annotations

import numpy as np
import cv2

MODEL_W = 512
MODEL_H = 256
MODEL_UV_W = MODEL_W // 2  # 256
MODEL_UV_H = MODEL_H // 2  # 128


def warp_yuv_to_model(
    Y: np.ndarray, U: np.ndarray, V: np.ndarray,
    warp_y: np.ndarray, warp_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply per-plane warp to camera-frame YUV, return model-frame Y(256x512),
    U(128x256), V(128x256). warp_y is the full-res warp; warp_uv is the half-res
    warp (intrinsics scaled by 0.5)."""
    # cv2.warpPerspective with WARP_INVERSE_MAP: for each pixel in dst, sample src at M @ dst.
    # warp_matrix as defined by openpilot maps model->camera, so we use it with INVERSE_MAP
    # to fill the model from the camera.
    Y_m = cv2.warpPerspective(
        Y, warp_y, (MODEL_W, MODEL_H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    U_m = cv2.warpPerspective(
        U, warp_uv, (MODEL_UV_W, MODEL_UV_H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=128,
    )
    V_m = cv2.warpPerspective(
        V, warp_uv, (MODEL_UV_W, MODEL_UV_H),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=128,
    )
    return Y_m, U_m, V_m


def yuv_to_6ch(Y_m: np.ndarray, U_m: np.ndarray, V_m: np.ndarray) -> np.ndarray:
    """Pack model-frame Y/U/V into supercombo's 6-channel layout.

    Channel order is derived from openpilot/selfdrive/modeld/transforms/loadyuv.cl
    (the README has channels 1 and 2 swapped — the kernel is authoritative):

      ch 0: Y[even rows, even cols]   = Y[0::2, 0::2]
      ch 1: Y[odd  rows, even cols]   = Y[1::2, 0::2]
      ch 2: Y[even rows, odd  cols]   = Y[0::2, 1::2]
      ch 3: Y[odd  rows, odd  cols]   = Y[1::2, 1::2]
      ch 4: U (half-res of model Y)
      ch 5: V (half-res of model Y)

    Values are raw uint8 cast to float in [0, 255] — loadyuv.cl does
    convert_float8() with no scaling. Returns (6, 128, 256) float32.
    """
    out = np.empty((6, MODEL_UV_H, MODEL_UV_W), dtype=np.float32)
    out[0] = Y_m[0::2, 0::2]
    out[1] = Y_m[1::2, 0::2]
    out[2] = Y_m[0::2, 1::2]
    out[3] = Y_m[1::2, 1::2]
    out[4] = U_m
    out[5] = V_m
    return out


def stack_pair(prev6: np.ndarray, curr6: np.ndarray) -> np.ndarray:
    """Stack two 6-channel planes -> (1, 12, 128, 256)."""
    return np.concatenate([prev6, curr6], axis=0)[np.newaxis, ...]
