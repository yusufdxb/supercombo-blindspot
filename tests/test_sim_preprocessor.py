"""Unit tests for src/sim_preprocessor — the CARLA-RGB to supercombo-input path.

These tests need no CARLA server; they exercise the pure-numpy preprocessing.
The warp itself is the verified Step 3.5 path (warped_preprocessor); what is new
here is (a) the zero-calibration warp construction for sim and (b) the RGB->YUV
glue. So that is what we test.
"""

from __future__ import annotations

import numpy as np

from src.sim_preprocessor import (
    CARLA_CAM_H,
    CARLA_CAM_W,
    build_sim_warps,
    ecam_fov_deg,
    fcam_fov_deg,
    rgb_to_model_6ch,
    rgb_to_model_input,
)
from src.transformations import _ar_ox_config, medmodel_intrinsics


def test_fcam_fov_matches_focal_2648():
    """CARLA fov attribute that reproduces comma 3 fcam intrinsics (focal 2648,
    width 1928). 2*atan(964/2648) = 40.05 deg."""
    assert abs(fcam_fov_deg() - 40.05) < 0.05


def test_ecam_fov_matches_focal_567():
    """comma 3 wide ecam: focal 567, width 1928 -> 2*atan(964/567) = 119.04 deg."""
    assert abs(ecam_fov_deg() - 119.04) < 0.1


def test_zero_calib_warp_is_pure_intrinsic_rescale():
    """With zero calibration euler the warp collapses to K_fcam @ inv(K_medmodel)
    — a pure intrinsic remap, no rotation. This is the whole reason a CARLA
    camera with fcam intrinsics needs no calibration warp."""
    warp_y, _warp_uv = build_sim_warps()
    expected = _ar_ox_config.fcam.intrinsics @ np.linalg.inv(medmodel_intrinsics)
    assert np.allclose(warp_y, expected, rtol=1e-9, atol=1e-9)


def test_rgb_to_model_6ch_shape_dtype():
    warp_y, warp_uv = build_sim_warps()
    rgb = np.random.randint(0, 256, (CARLA_CAM_H, CARLA_CAM_W, 3), dtype=np.uint8)
    six = rgb_to_model_6ch(rgb, warp_y, warp_uv)
    assert six.shape == (6, 128, 256)
    assert six.dtype == np.float32


def test_no_normalization_values_stay_in_0_255():
    """loadyuv.cl casts uint8->float with NO /255. A white frame must land near
    BT.601 limited-range white (Y~235), not near 1.0."""
    warp_y, warp_uv = build_sim_warps()
    white = np.full((CARLA_CAM_H, CARLA_CAM_W, 3), 255, dtype=np.uint8)
    six = rgb_to_model_6ch(white, warp_y, warp_uv)
    # central pixels are inside the warp footprint -> real luma, not border 0
    cy, cx = six.shape[1] // 2, six.shape[2] // 2
    assert 230 <= six[0, cy, cx] <= 240
    assert six.min() >= 0.0 and six.max() <= 255.0


def test_rgb_to_model_input_is_12ch_pair():
    """Two RGB frames -> (1, 12, 128, 256), prev block then curr block."""
    warp_y, warp_uv = build_sim_warps()
    prev = np.random.randint(0, 256, (CARLA_CAM_H, CARLA_CAM_W, 3), dtype=np.uint8)
    curr = np.random.randint(0, 256, (CARLA_CAM_H, CARLA_CAM_W, 3), dtype=np.uint8)
    inp = rgb_to_model_input(prev, curr, warp_y, warp_uv)
    assert inp.shape == (1, 12, 128, 256)
    # channels 0..5 are prev, 6..11 are curr
    six_prev = rgb_to_model_6ch(prev, warp_y, warp_uv)
    six_curr = rgb_to_model_6ch(curr, warp_y, warp_uv)
    assert np.array_equal(inp[0, :6], six_prev)
    assert np.array_equal(inp[0, 6:], six_curr)
