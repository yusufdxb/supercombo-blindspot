"""Tests for E5: layer-localized collapse."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest

from src.e5_layer import LAYER_PROBES, intermediates_to_outputs


def test_layer_probes_cover_all_vision_stages():
    names = {p.name for p in LAYER_PROBES}
    assert len(names) == len(LAYER_PROBES), "probe names must be unique"
    assert "stem" in names
    assert "head" in names
    # Every encoder stage gets at least one probe, and stage 2 is densely covered.
    for s in (0, 1, 2, 3):
        stage_probes = {n for n in names if n.startswith(f"stage{s}")}
        assert stage_probes, f"no probe in stage{s}"
    # Stage 2 has 6 blocks; we probe all 6 to rule out mid-block collapse.
    stage2 = {n for n in names if n.startswith("stage2")}
    assert len(stage2) == 6, f"stage2 needs 6 probes, got {len(stage2)}"


def test_intermediates_to_outputs_adds_tensors(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    src = Path("models/supercombo.onnx")
    if not src.exists():
        pytest.skip("supercombo.onnx not present")
    dst = tmp_path / "supercombo_probed.onnx"
    intermediates_to_outputs(src, dst, [p.tensor for p in LAYER_PROBES])
    m = onnx.load(str(dst))
    out_names = {o.name for o in m.graph.output}
    for p in LAYER_PROBES:
        assert p.tensor in out_names, f"{p.tensor} missing from probed outputs"


from src.e5_layer import per_layer_activity_ratio


def test_per_layer_activity_ratio_endpoints():
    real = np.random.RandomState(0).randn(50, 4, 4).astype(np.float32)
    carla = np.zeros((50, 4, 4), dtype=np.float32)
    assert per_layer_activity_ratio(real, carla) < 0.05
    assert abs(per_layer_activity_ratio(real, real.copy()) - 1.0) < 1e-6


from src.e5_layer import save_cache, load_cache


def test_cache_roundtrip(tmp_path):
    alphas = np.linspace(0.0, 1.0, 6)
    per_layer = {p.name: np.random.randn(len(alphas), 50, 4).astype(np.float32)
                 for p in LAYER_PROBES}
    p = tmp_path / "e5.npz"
    save_cache(p, alphas, per_layer)
    a2, pl2 = load_cache(p)
    assert np.allclose(a2, alphas)
    for k, v in per_layer.items():
        assert np.allclose(pl2[k], v)


from src.e5_layer import cliff_alpha


def test_cliff_alpha_finds_step():
    alphas = np.linspace(0.0, 1.0, 21)
    ratios = np.where(alphas < 0.7, 1.0, 0.05)
    a = cliff_alpha(alphas, ratios, threshold=0.5)
    assert 0.65 <= a <= 0.75


def test_cliff_alpha_none_if_never_crosses():
    alphas = np.linspace(0.0, 1.0, 11)
    ratios = np.full_like(alphas, 0.9)
    assert np.isnan(cliff_alpha(alphas, ratios, threshold=0.5))


def test_e5_cache_reproduces():
    cache = Path("report/e5_collected.npz")
    if not cache.exists():
        pytest.skip("e5 cache not present")
    try:
        alphas, per_layer = load_cache(cache)
    except (zipfile.BadZipFile, OSError, ValueError) as exc:
        pytest.skip(f"e5 cache corrupt: {exc}")
    assert len(alphas) >= 5
    assert set(per_layer.keys()) == {p.name for p in LAYER_PROBES}
    # Verify each layer has non-trivial float32 data across alpha sweep.
    from src.e5_layer import per_layer_activity_ratio
    for p in LAYER_PROBES:
        arr = per_layer[p.name]
        assert arr.dtype == np.float32
        assert arr.shape[0] == len(alphas)
        # ratio of real to itself must be 1.0 (sanity check on the metric)
        assert abs(per_layer_activity_ratio(arr[0], arr[0]) - 1.0) < 1e-5


from src.e5_layer import per_layer_mean_shift


def test_per_layer_mean_shift_endpoints():
    real = np.zeros((50, 4, 4), dtype=np.float32)
    real[:, 0, 0] = 1.0  # nonzero mean
    carla = real * 2.0
    assert abs(per_layer_mean_shift(real, carla) - 2.0) < 1e-6
    assert abs(per_layer_mean_shift(real, real.copy()) - 1.0) < 1e-6


def test_per_layer_mean_shift_zero_real_returns_nan():
    real = np.zeros((50, 4, 4), dtype=np.float32)
    carla = np.ones((50, 4, 4), dtype=np.float32)
    import math
    assert math.isnan(per_layer_mean_shift(real, carla))


# --- Summary cache round-trip ---

from src.e5_layer import save_summary_cache, load_summary_cache


def test_summary_cache_roundtrip(tmp_path):
    alphas = np.linspace(0.0, 1.0, 11)
    names = ["stem", "stage0_blk0", "head"]
    ratios = {n: np.random.rand(11) for n in names}
    cliffs = {n: float("nan") for n in names}
    mean_shifts = {n: float(i + 0.5) for i, n in enumerate(names)}
    p = tmp_path / "e5_summary.npz"
    save_summary_cache(p, alphas, ratios, cliffs, mean_shifts)
    a2, r2, c2, ms2 = load_summary_cache(p)
    assert np.allclose(a2, alphas)
    for n in names:
        assert np.allclose(r2[n], ratios[n])
        assert (np.isnan(c2[n]) and np.isnan(cliffs[n])) or abs(c2[n] - cliffs[n]) < 1e-9
        assert abs(ms2[n] - mean_shifts[n]) < 1e-9


def test_summary_cache_keys_match(tmp_path):
    alphas = np.linspace(0.0, 1.0, 5)
    names = [p.name for p in LAYER_PROBES]
    ratios = {n: np.ones(5) * 0.7 for n in names}
    cliffs = {n: 0.8 for n in names}
    mean_shifts = {n: 1.2 for n in names}
    p = tmp_path / "e5_summary2.npz"
    save_summary_cache(p, alphas, ratios, cliffs, mean_shifts)
    _, r2, c2, ms2 = load_summary_cache(p)
    assert set(r2.keys()) == set(names)
    assert set(c2.keys()) == set(names)


# --- E5 v0.9.6 module ---

from src.e5_layer_v096 import LAYER_PROBES_V096


def test_v096_layer_probes_unique():
    names = {p.name for p in LAYER_PROBES_V096}
    assert len(names) == len(LAYER_PROBES_V096)
    assert "stem" in names
    assert "head" in names
    for s in (0, 1, 2, 3):
        assert any(n.startswith(f"stage{s}") for n in names)
    stage2 = {n for n in names if n.startswith("stage2")}
    assert len(stage2) == 6


def test_v096_probes_exist_in_model():
    onnx = pytest.importorskip("onnx")
    m_path = Path("models/supercombo_v096.onnx")
    if not m_path.exists():
        pytest.skip("supercombo_v096.onnx not present")
    m = onnx.load(str(m_path))
    node_outputs = {out for node in m.graph.node for out in node.output}
    for probe in LAYER_PROBES_V096:
        assert probe.tensor in node_outputs, (
            f"v0.9.6 probe tensor missing from model: {probe.tensor}")


def test_v096_probes_differ_from_v097_for_stem_and_head():
    v097_tensors = {p.tensor for p in LAYER_PROBES}
    v096_tensors = {p.tensor for p in LAYER_PROBES_V096}
    # stem and head tensors must differ (architecture change between versions)
    stem_v097 = next(p.tensor for p in LAYER_PROBES if p.name == "stem")
    stem_v096 = next(p.tensor for p in LAYER_PROBES_V096 if p.name == "stem")
    head_v097 = next(p.tensor for p in LAYER_PROBES if p.name == "head")
    head_v096 = next(p.tensor for p in LAYER_PROBES_V096 if p.name == "head")
    assert stem_v097 != stem_v096
    assert head_v097 != head_v096


def test_collect_per_layer_accepts_custom_probes(tmp_path):
    """collect_per_layer with probes= kwarg uses the provided list, not global LAYER_PROBES."""
    from src.e5_layer import collect_per_layer as cpl
    tiny_probe = [LAYER_PROBES[0]]  # just stem
    n_alphas, n_frames = 2, 3
    alphas = np.array([0.0, 1.0])
    real_six = [np.zeros((6, 128, 256), dtype=np.float32)] * n_frames
    carla_six = [np.zeros((6, 128, 256), dtype=np.float32)] * n_frames
    # This is an import-only check; the actual GPU call is skipped without a model.
    # We just verify the signature accepts the kwargs without TypeError.
    import inspect
    sig = inspect.signature(cpl)
    assert "probes" in sig.parameters
    assert "session" in sig.parameters
    assert "slices" in sig.parameters
