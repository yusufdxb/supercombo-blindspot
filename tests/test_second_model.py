"""Second-model foundation tests for the v0.9.6 supercombo.

Asserts the v0.9.6 model loads, exposes the two extra nav inputs, keeps the
v0.9.7 output contract (15 slices summing to 6504, hidden_state == slice(5992,
None)), and that an >=8-frame state-threaded run rolls features_buffer with no
NaN/Inf. Skips cleanly when onnx/onnxruntime or the model file are absent (repo
CI convention)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
V096 = REPO / "models" / "supercombo_v096.onnx"


def _require_model():
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    if not V096.exists():
        pytest.skip(f"{V096} not present (run scripts.fetch_upgrade_data)")


def test_v096_loads_in_onnxruntime():
    _require_model()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(V096), so, providers=["CPUExecutionProvider"])
    assert sess is not None
    # output is the single flat [1, 6504] tensor, same as v0.9.7
    outs = sess.get_outputs()
    assert [o.name for o in outs] == ["outputs"]
    assert list(outs[0].shape) == [1, 6504]


def test_v096_exposes_nav_inputs():
    _require_model()
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(V096), so, providers=["CPUExecutionProvider"])
    by_name = {i.name: i for i in sess.get_inputs()}
    # the two extra inputs vs v0.9.7
    assert "nav_features" in by_name
    assert "nav_instructions" in by_name
    assert list(by_name["nav_features"].shape) == [1, 256]
    assert list(by_name["nav_instructions"].shape) == [1, 150]
    assert by_name["nav_features"].type == "tensor(float16)"
    assert by_name["nav_instructions"].type == "tensor(float16)"
    # the v0.9.7 seven inputs all survive
    for name in (
        "input_imgs", "big_input_imgs", "desire", "traffic_convention",
        "lateral_control_params", "prev_desired_curv", "features_buffer",
    ):
        assert name in by_name, f"{name} missing from v0.9.6 inputs"


def test_v096_output_slices_contract():
    _require_model()
    from src.state import load_output_slices

    slices = load_output_slices(V096)
    assert len(slices) == 15, f"expected 15 output slices, got {len(slices)}"

    # spans sum to 6504 (hidden_state has stop=None -> spans 5992..6504)
    total = 0
    for k, sl in slices.items():
        start = sl.start or 0
        stop = sl.stop if sl.stop is not None else 6504
        total += stop - start
    assert total == 6504, f"slice spans sum to {total}, expected 6504"

    # hidden_state is the recurrent feature tail
    assert slices["hidden_state"] == slice(5992, None), slices["hidden_state"]


def test_v096_state_threaded_run_rolls_features_buffer():
    _require_model()
    from src.constants import ModelConstants
    from src.state import build_mirror

    state = build_mirror(V096)
    # the mirror must have detected + wired the nav inputs
    assert state._has_nav is True
    assert "nav_features" in state.state
    assert "nav_instructions" in state.state
    assert state.state["nav_features"].shape == (256,)
    assert state.state["nav_instructions"].shape == (150,)

    rng = np.random.RandomState(0)
    n_frames = 8
    # 6-channel medmodel frames; stack consecutive pairs into the (1,12,128,256) input
    sixes = [rng.rand(6, 128, 256).astype(np.float32) for _ in range(n_frames)]

    last_row_l2 = None
    prev = None
    n_run = 0
    for six in sixes:
        if prev is not None:
            inp = np.concatenate([prev, six], axis=0)[np.newaxis].astype(np.float32)
            parsed = state.run(inp, inp)
            n_run += 1
            accel = float(parsed["plan"][0, 0, 6])
            assert np.isfinite(accel), "accel@t0 is NaN/Inf"
            # no NaN/Inf anywhere in the recurrent feature vector
            hs = parsed["hidden_state"]
            assert np.all(np.isfinite(hs)), "hidden_state has NaN/Inf"
            fb = state.state["features_buffer"].reshape(
                ModelConstants.HISTORY_BUFFER_LEN, ModelConstants.FEATURE_LEN)
            last_row_l2 = float(np.linalg.norm(fb[-1]))
        prev = six

    assert n_run >= 7, f"expected >=7 threaded predictions, ran {n_run}"
    # features_buffer rolled: the newest row is the just-produced hidden_state, non-zero
    assert last_row_l2 is not None
    assert last_row_l2 > 0.0, "features_buffer last row is all zeros (state did not roll)"
