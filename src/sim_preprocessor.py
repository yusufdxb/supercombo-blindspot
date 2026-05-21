"""CARLA RGB frame -> supercombo 12-channel input (Step 4 sim path).

The key idea that keeps this faithful: a CARLA `sensor.camera.rgb` is an ideal
pinhole. If we render it at the comma 3 fcam resolution (1928x1208) with the fov
that reproduces fcam's focal length (2648 px), the rendered frame has *exactly*
the intrinsics of `_ar_ox_config.fcam`. We can then reuse the verified Step 3.5
warp (`get_warp_matrix` + `cv2.warpPerspective`) to map it into the 512x256
medmodel frame.

Real openpilot feeds `get_warp_matrix(device_from_calib_euler, fcam.intrinsics)`
where the euler comes from `liveCalibration`. In sim the camera is mounted
perfectly (we control the transform), so the calibration euler is exactly zero
and the warp collapses to `K_fcam @ inv(K_medmodel)` — a pure intrinsic remap,
no rotation. That is why no per-run calibration is needed in sim.

Pipeline: RGB(1928x1208) --BT.601--> YUV420 --warp--> model YUV(256x512 / 128x256)
--6ch split--> (6,128,256), then two frames stacked -> (1,12,128,256).

The 6-channel split and the no-normalization rule are inherited verbatim from
`warped_preprocessor` (the Step 3.5 path that hit 100% parity).
"""

from __future__ import annotations

import math

import numpy as np

from src.preprocessor import rgb_to_yuv420
from src.transformations import _ar_ox_config, get_warp_matrix, scaled_intrinsics
from src.warped_preprocessor import stack_pair, warp_yuv_to_model, yuv_to_6ch

# CARLA narrow camera is rendered at comma 3 fcam native resolution so that the
# rendered intrinsics match `_ar_ox_config.fcam` exactly.
CARLA_CAM_W = 1928
CARLA_CAM_H = 1208

# Zero calibration: the sim camera is mounted exactly on the device axes.
ZERO_CALIB = np.zeros(3, dtype=np.float64)


def _fov_deg_for_focal(focal_px: float, width_px: int) -> float:
    """Horizontal fov (degrees) a pinhole of `focal_px` has at `width_px` wide.
    CARLA's camera `fov` attribute is this horizontal fov."""
    return math.degrees(2.0 * math.atan((width_px / 2.0) / focal_px))


def fcam_fov_deg() -> float:
    """fov to give a CARLA narrow camera so it reproduces comma 3 fcam intrinsics."""
    fcam = _ar_ox_config.fcam
    return _fov_deg_for_focal(fcam.focal_length, fcam.width)


def ecam_fov_deg() -> float:
    """fov for the wide camera, matching comma 3 ecam focal length.

    NOTE: the real ecam is a fisheye; a CARLA pinhole at this fov is only a rough
    stand-in. The supercombo `big_input_imgs` feed is handled in `scenario.py`,
    which (per the verified Step 3.5 finding) feeds the narrow model frame to
    both inputs. This wide camera exists for recording / overlays, not as a
    faithful ecam."""
    ecam = _ar_ox_config.ecam
    return _fov_deg_for_focal(ecam.focal_length, ecam.width)


def build_sim_warps() -> tuple[np.ndarray, np.ndarray]:
    """(warp_y, warp_uv) homographies mapping the medmodel frame back into the
    CARLA fcam-intrinsics camera frame, for full-res Y and half-res U/V."""
    K = _ar_ox_config.fcam.intrinsics
    warp_y = get_warp_matrix(ZERO_CALIB, K, bigmodel_frame=False)
    warp_uv = get_warp_matrix(ZERO_CALIB, scaled_intrinsics(K, 0.5), bigmodel_frame=False)
    return warp_y, warp_uv


def rgb_to_model_6ch(rgb: np.ndarray, warp_y: np.ndarray, warp_uv: np.ndarray) -> np.ndarray:
    """One CARLA RGB frame (1208x1928x3 uint8) -> (6, 128, 256) float32, no /255."""
    if rgb.shape != (CARLA_CAM_H, CARLA_CAM_W, 3):
        raise ValueError(f"expected RGB {(CARLA_CAM_H, CARLA_CAM_W, 3)}, got {rgb.shape}")
    Y, U, V = rgb_to_yuv420(rgb)  # Y full-res, U/V half-res
    Y_m, U_m, V_m = warp_yuv_to_model(Y, U, V, warp_y, warp_uv)
    return yuv_to_6ch(Y_m, U_m, V_m)


def rgb_to_model_input(prev_rgb: np.ndarray, curr_rgb: np.ndarray,
                       warp_y: np.ndarray, warp_uv: np.ndarray) -> np.ndarray:
    """Two consecutive CARLA RGB frames -> (1, 12, 128, 256) float32."""
    prev6 = rgb_to_model_6ch(prev_rgb, warp_y, warp_uv)
    curr6 = rgb_to_model_6ch(curr_rgb, warp_y, warp_uv)
    return stack_pair(prev6, curr6)
