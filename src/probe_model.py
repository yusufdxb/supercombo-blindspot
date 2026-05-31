"""Collect full per-frame supercombo outputs for the distribution-shift teardown.

`collect()` warms a fresh ModelStateMirror on a list of model-frame inputs and
records every output head plus the model's internal recurrent feature vector.
The teardown analysis (src/teardown.py) runs on what this returns.

Loaders produce the 512x256 medmodel-frame 6-channel input from either real
comma HEVC (calibrated warp from the segment's liveCalibration) or CARLA RGB
(zero-calibration warp).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from src.decode_hevc import yuv_frame_iter
from src.preprocessor import rgb_to_yuv420
from src.rlog import iter_events
from src.sim_preprocessor import build_sim_warps
from src.state import ModelStateMirror
from src.transformations import _ar_ox_config, get_warp_matrix, scaled_intrinsics
from src.warped_preprocessor import stack_pair, warp_yuv_to_model, yuv_to_6ch


# --------------------------------------------------------------------------
# loaders -> list of (6, 128, 256) model-frame inputs
# --------------------------------------------------------------------------

def _calib_warps(rlog_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Build the (warp_y, warp_uv) pair from a real segment's liveCalibration."""
    rpy = None
    for ev in iter_events(rlog_path):
        if ev.which() == "liveCalibration":
            rpy = np.array(list(ev.liveCalibration.rpyCalib), dtype=np.float64)
    if rpy is None:
        raise RuntimeError(f"no liveCalibration in {rlog_path}")
    K = _ar_ox_config.fcam.intrinsics
    return (get_warp_matrix(rpy, K, False),
            get_warp_matrix(rpy, scaled_intrinsics(K, 0.5), False))


def load_real_six(hevc_path: Path, rlog_path: Path, n: int) -> list[np.ndarray]:
    """First `n` frames of a real comma segment, warped to the medmodel frame."""
    warp_y, warp_uv = _calib_warps(rlog_path)
    out: list[np.ndarray] = []
    for k, (Y, U, V) in enumerate(yuv_frame_iter(hevc_path, 1928, 1208)):
        if k >= n:
            break
        out.append(yuv_to_6ch(*warp_yuv_to_model(Y, U, V, warp_y, warp_uv)))
    return out


def load_carla_six(npy_path: Path, n: int) -> list[np.ndarray]:
    """First `n` captured CARLA RGB frames, warped to the medmodel frame."""
    rgb = np.load(npy_path)[:n]
    warp_y, warp_uv = build_sim_warps()
    out = []
    for frame in rgb:
        Y, U, V = rgb_to_yuv420(np.ascontiguousarray(frame))
        out.append(yuv_to_6ch(*warp_yuv_to_model(Y, U, V, warp_y, warp_uv)))
    return out


# --------------------------------------------------------------------------
# collector
# --------------------------------------------------------------------------

# heads tracked for the collapse map: name -> (parsed key, slice into [0])
_HEADS = {
    "plan": "plan",                       # (33,15) full longitudinal+lateral plan
    "lane_lines": "lane_lines",           # lane line geometry
    "road_edges": "road_edges",           # road edge geometry
    "lead": "lead",                       # lead vehicle hypotheses
    "pose": "pose",                       # ego-motion estimate
    "desire_state": "desire_state",       # maneuver intent
    "meta": "meta",                       # disengage / blinker / etc probabilities
}


def collect(six_list: list[np.ndarray], sess=None, slices=None) -> dict[str, np.ndarray]:
    """Warm a fresh ModelStateMirror on `six_list`; return per-frame stacked
    arrays for every tracked head, the model's predicted uncertainties, and its
    internal recurrent feature vector.

    `sess`/`slices` may be a passed-in v0.9.6 session + its output_slices (built
    via src.state.build_mirror / build_session + load_output_slices on the v0.9.6
    path). ModelStateMirror auto-detects the nav_features/nav_instructions inputs
    from the session, so v0.9.7 behaviour is unchanged. When both are None the
    default v0.9.7 session is used."""
    if sess is None and slices is None:
        state = ModelStateMirror()
    else:
        state = ModelStateMirror(session=sess, output_slices=slices)
    rec: dict[str, list] = defaultdict(list)
    prev = None
    for six in six_list:
        if prev is not None:
            inp = stack_pair(prev, six)
            p = state.run(inp, inp)
            # scalar readouts
            rec["accel_t0"].append(float(p["plan"][0, 0, 6]))
            rec["desired_curv"].append(float(p["desired_curvature"][0, 0]))
            rec["lead_prob"].append(float(p["lead_prob"][0, 0]))
            # full heads (for the collapse map)
            for name, key in _HEADS.items():
                rec[name].append(np.asarray(p[key][0], dtype=np.float32).ravel())
            rec["lane_lines_prob"].append(
                np.asarray(p["lane_lines_prob"][0], dtype=np.float32))
            # the model's own predicted uncertainty
            rec["plan_std"].append(np.asarray(p["plan_stds"][0], dtype=np.float32).ravel())
            rec["lead_std"].append(np.asarray(p["lead_stds"][0], dtype=np.float32).ravel())
            rec["desired_curv_std"].append(float(p["desired_curvature_stds"][0, 0]))
            # internal recurrent feature vector
            rec["hidden_state"].append(
                np.asarray(p["hidden_state"][0], dtype=np.float32))
        prev = six
    return {k: np.array(v) for k, v in rec.items()}


HEAD_NAMES = list(_HEADS.keys())
