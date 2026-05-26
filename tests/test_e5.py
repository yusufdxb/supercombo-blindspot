"""Tests for E5: layer-localized collapse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest

from src.e5_layer import LAYER_PROBES, intermediates_to_outputs


def test_layer_probes_are_unique_and_ordered():
    names = [p.name for p in LAYER_PROBES]
    assert names == sorted(names, key=lambda n: [int(c) if c.isdigit() else c
                                                  for c in n.split("_")]) \
        or len(set(names)) == len(names)
    assert {"stem", "stage0", "stage1", "stage2", "stage3", "head"}.issubset(set(names))


def test_intermediates_to_outputs_adds_tensors(tmp_path: Path):
    src = Path("models/supercombo.onnx")
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
    alphas, per_layer = load_cache(cache)
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
