"""Regression tests for the E9b geometry-control aggregation (no model needed)."""

from pathlib import Path

import numpy as np
import pytest

from src.e9b_geomwarp import _counts
from src.teardown import HEAD_NAMES, SCALARS

ROOT = Path(__file__).resolve().parents[1]


def _segment(n, activity, spread, seed=0):
    """Synthetic collected-outputs dict with controllable per-frame activity and
    hidden-state spread, holding every key _counts / e1 / e2 / e3 consume."""
    rng = np.random.default_rng(seed)
    d = {}
    for key in SCALARS:
        d[key] = (rng.standard_normal(n) * activity).astype(np.float32)
    for key in HEAD_NAMES:
        d[key] = (rng.standard_normal((n, 8)) * activity).astype(np.float32)
    d["hidden_state"] = (rng.standard_normal((n, 16)) * spread).astype(np.float32)
    for key in ("plan_std", "lead_std"):
        d[key] = np.abs(rng.standard_normal((n, 4))).astype(np.float32)
    d["desired_curv_std"] = np.abs(rng.standard_normal(n)).astype(np.float32)
    return d


def test_counts_flags_a_frozen_probe():
    base = _segment(60, activity=1.0, spread=1.0, seed=1)
    frozen = _segment(60, activity=0.001, spread=0.001, seed=2)
    r = _counts(base, frozen)
    assert r["n_readouts"] == len(SCALARS) + len(HEAD_NAMES) == 10
    assert r["below01"] >= 8          # nearly all readouts below 1% of baseline
    assert r["spread_ratio"] < 0.01   # feature cluster contracted


def test_counts_passes_an_active_probe():
    base = _segment(60, activity=1.0, spread=1.0, seed=1)
    active = _segment(60, activity=1.0, spread=1.0, seed=9)
    r = _counts(base, active)
    assert r["below10"] == 0          # nothing collapsed
    assert r["spread_ratio"] > 0.1    # spread comparable to baseline


def test_zerowarp_loader_uses_zero_calibration():
    # the zero-warp real loader must build its warp from the zero euler, not from
    # a segment's liveCalibration.
    import inspect

    from src import e9b_geomwarp
    src = inspect.getsource(e9b_geomwarp.load_real_six_zerowarp)
    assert "_warps(ZERO_EULER)" in src   # zero-calibration euler
    assert "_calib_warps" not in src     # not the liveCalibration path


def test_zero_warp_equals_carla_sim_warp():
    # Part B claims CARLA and real-zero share the IDENTICAL preprocessing warp.
    # Prove it: _warps(0) is byte-for-byte the CARLA path's build_sim_warps.
    from src.e9b_geomwarp import ZERO_EULER, _warps
    from src.sim_preprocessor import build_sim_warps
    wy, wuv = _warps(ZERO_EULER)
    by, buv = build_sim_warps()
    assert np.array_equal(wy, by)
    assert np.array_equal(wuv, buv)


def test_euler_actually_changes_the_warp():
    # sanity: the euler is a live variable, not ignored by the builder.
    from src.e9b_geomwarp import ZERO_EULER, _warps
    live = _warps(np.array([0.01, -0.02, 0.03]))
    zero = _warps(ZERO_EULER)
    assert not np.array_equal(live[0], zero[0])
    assert zero[0].shape == live[0].shape == (3, 3)


def test_carla_loader_really_uses_the_zero_calibration_warp():
    # Part B's "identical warp" claim depends on load_carla_six actually building
    # its warp from build_sim_warps. Read the loader source as text rather than
    # importing probe_model, which pulls in the onnx-dependent state module not
    # present in the CI image (repo convention: importorskip("onnx") elsewhere).
    src = (ROOT / "src" / "probe_model.py").read_text(encoding="utf-8")
    assert "def load_carla_six" in src
    assert "build_sim_warps()" in src


@pytest.mark.skipif(not (ROOT / "data" / "subaru_source" / "rlog.bz2").exists(),
                    reason="subaru rlog not present")
def test_calibrated_path_differs_from_zero_only_by_euler():
    """The strong form of the isolation claim, against the ACTUAL calibrated
    loader: rebuilding _calib_warps' euler through _warps reproduces it exactly,
    so the calibrated and zero conditions differ only in the euler."""
    pytest.importorskip("onnx")  # probe_model -> state -> onnx (absent in CI)
    from src.e9b_geomwarp import _warps
    from src.probe_model import _calib_warps
    from src.rlog import iter_events

    rlog = ROOT / "data" / "subaru_source" / "rlog.bz2"
    rpy = None
    for ev in iter_events(rlog):
        if ev.which() == "liveCalibration":
            rpy = np.array(list(ev.liveCalibration.rpyCalib), dtype=np.float64)
    assert rpy is not None
    live_y, live_uv = _calib_warps(rlog)
    mine_y, mine_uv = _warps(rpy)
    # same intrinsics + same builder: feeding the segment's euler reproduces it
    assert np.allclose(live_y, mine_y)
    assert np.allclose(live_uv, mine_uv)
