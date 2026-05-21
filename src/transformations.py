"""Vendored from openpilot/common/transformations/{camera,model,orientation}.py at v0.9.7.

Contains the bits needed to compute the calibration-dependent perspective warp
that modeld applies to camera frames before feeding supercombo. We avoid the
full openpilot install by porting rot_from_euler in pure numpy (the upstream
version delegates to a C++ extension)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- orientation ---

def rot_from_euler(eul) -> np.ndarray:
    """Equivalent to openpilot.common.transformations.orientation.euler2rot_single.

    Convention from openpilot/orientation.cc: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Eul is [roll, pitch, yaw] in radians."""
    roll, pitch, yaw = float(eul[0]), float(eul[1]), float(eul[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return Rz @ Ry @ Rx


# --- camera ---

@dataclass(frozen=True)
class CameraConfig:
    width: int
    height: int
    focal_length: float

    @property
    def intrinsics(self) -> np.ndarray:
        return np.array([
            [self.focal_length, 0.0, self.width / 2.0],
            [0.0, self.focal_length, self.height / 2.0],
            [0.0, 0.0, 1.0],
        ])


@dataclass(frozen=True)
class DeviceCameraConfig:
    fcam: CameraConfig
    dcam: CameraConfig
    ecam: CameraConfig


# comma 3 (tici) cameras (camera.py:49-51 at v0.9.7)
_ar_ox_fisheye = CameraConfig(1928, 1208, 567.0)
_ar_ox_config = DeviceCameraConfig(CameraConfig(1928, 1208, 2648.0), _ar_ox_fisheye, _ar_ox_fisheye)
_neo_config = DeviceCameraConfig(CameraConfig(1164, 874, 910.0), CameraConfig(816, 612, 650.0), CameraConfig(0, 0, 0))

DEVICE_CAMERAS = {
    ("neo", "unknown"): _neo_config,
    ("tici", "unknown"): _ar_ox_config,
    ("unknown", "ar0231"): _ar_ox_config,
    ("unknown", "ox03c10"): _ar_ox_config,
    ("pc", "unknown"): _ar_ox_config,
    ("tici", "ar0231"): _ar_ox_config,
    ("tici", "ox03c10"): _ar_ox_config,
    ("tizi", "ar0231"): _ar_ox_config,
    ("tizi", "ox03c10"): _ar_ox_config,
    ("mici", "ar0231"): _ar_ox_config,
    ("mici", "ox03c10"): _ar_ox_config,
}

device_frame_from_view_frame = np.array([
    [0., 0., 1.],
    [1., 0., 0.],
    [0., 1., 0.],
])
view_frame_from_device_frame = device_frame_from_view_frame.T


# --- model ---

MEDMODEL_INPUT_SIZE = (512, 256)
MEDMODEL_CY = 47.6
medmodel_fl = 910.0
medmodel_intrinsics = np.array([
    [medmodel_fl, 0.0, 0.5 * MEDMODEL_INPUT_SIZE[0]],
    [0.0, medmodel_fl, MEDMODEL_CY],
    [0.0, 0.0, 1.0],
])

SBIGMODEL_INPUT_SIZE = (512, 256)
sbigmodel_fl = 455.0
sbigmodel_intrinsics = np.array([
    [sbigmodel_fl, 0.0, 0.5 * SBIGMODEL_INPUT_SIZE[0]],
    [0.0, sbigmodel_fl, 0.5 * (256 + MEDMODEL_CY)],
    [0.0, 0.0, 1.0],
])


def _get_view_frame_from_calib_frame(roll: float, pitch: float, yaw: float, height: float) -> np.ndarray:
    device_from_calib = rot_from_euler([roll, pitch, yaw])
    view_from_calib = view_frame_from_device_frame @ device_from_calib
    return np.hstack((view_from_calib, [[0.0], [height], [0.0]]))


medmodel_frame_from_calib_frame = medmodel_intrinsics @ _get_view_frame_from_calib_frame(0, 0, 0, 0)
sbigmodel_frame_from_calib_frame = sbigmodel_intrinsics @ _get_view_frame_from_calib_frame(0, 0, 0, 0)

calib_from_medmodel = np.linalg.inv(medmodel_frame_from_calib_frame[:, :3])
calib_from_sbigmodel = np.linalg.inv(sbigmodel_frame_from_calib_frame[:, :3])


def get_warp_matrix(device_from_calib_euler: np.ndarray, intrinsics: np.ndarray, bigmodel_frame: bool = False) -> np.ndarray:
    """Verbatim from openpilot/common/transformations/model.py:58-63 at v0.9.7."""
    calib_from_model = calib_from_sbigmodel if bigmodel_frame else calib_from_medmodel
    device_from_calib = rot_from_euler(device_from_calib_euler)
    camera_from_calib = intrinsics @ view_frame_from_device_frame @ device_from_calib
    warp_matrix = camera_from_calib @ calib_from_model
    return warp_matrix


def scaled_intrinsics(intrinsics: np.ndarray, scale: float) -> np.ndarray:
    """Adjust intrinsics for a frame that's been resized by `scale` (e.g. 0.5 for half-res U/V planes)."""
    K = intrinsics.copy()
    K[0, 0] *= scale
    K[1, 1] *= scale
    K[0, 2] *= scale
    K[1, 2] *= scale
    return K
