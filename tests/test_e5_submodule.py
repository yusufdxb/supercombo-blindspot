"""Tests for E5 submodule localization."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.e5_layer import LAYER_PROBES
from src.e5_submodule import (
    SUBMODULE_PROBES,
    CACHE,
    RESULTS,
    analyse,
    load_cache,
    save_cache,
)


def test_submodule_probes_unique_and_non_empty():
    assert len(SUBMODULE_PROBES) >= 4
    names = [p.name for p in SUBMODULE_PROBES]
    tensors = [p.tensor for p in SUBMODULE_PROBES]
    assert len(set(names)) == len(names), "probe names must be unique"
    assert len(set(tensors)) == len(tensors), "tensor names must be unique"


def test_submodule_probes_disjoint_from_encoder_probes():
    """E5 submodule must not re-probe the encoder tensors that e5_layer
    already covers. Probing the same tensor twice would be wasted work."""
    e5_tensors = {p.tensor for p in LAYER_PROBES}
    sub_tensors = {p.tensor for p in SUBMODULE_PROBES}
    assert e5_tensors.isdisjoint(sub_tensors)


def test_submodule_probes_in_enumeration_file():
    """Probe targets must be a documented subset of the enumeration markdown
    (so we cannot quietly probe something we never justified)."""
    enum_path = Path("report/e5_submodule_enumeration.md")
    assert enum_path.exists(), "enumeration md must exist before probing"
    text = enum_path.read_text()
    for p in SUBMODULE_PROBES:
        assert p.tensor in text, (
            f"probe tensor `{p.tensor}` is not mentioned in the "
            f"enumeration file: probe targets must be justified there first"
        )


def test_intermediates_to_outputs_adds_submodule_tensors(tmp_path: Path):
    onnx = pytest.importorskip("onnx")
    from src.e5_layer import intermediates_to_outputs

    src = Path("models/supercombo.onnx")
    if not src.exists():
        pytest.skip("supercombo.onnx not present")
    dst = tmp_path / "submodule_probed.onnx"
    intermediates_to_outputs(src, dst, [p.tensor for p in SUBMODULE_PROBES])
    m = onnx.load(str(dst))
    out_names = {o.name for o in m.graph.output}
    for p in SUBMODULE_PROBES:
        assert p.tensor in out_names, f"{p.tensor} missing from probed outputs"


def test_save_load_cache_roundtrip(tmp_path: Path):
    alphas = np.linspace(0.0, 1.0, 5)
    fake = {
        p.name: np.random.RandomState(0)
                 .randn(len(alphas), 20, 8)
                 .astype(np.float32)
        for p in SUBMODULE_PROBES
    }
    f = tmp_path / "e5sub.npz"
    save_cache(f, alphas, fake)
    a2, p2 = load_cache(f)
    assert np.allclose(a2, alphas)
    assert set(p2.keys()) == {p.name for p in SUBMODULE_PROBES}
    for k, v in fake.items():
        assert np.allclose(p2[k], v)


def test_analyse_gives_sensible_endpoints():
    """At alpha=0 the probe equals the real baseline so activity ratio must
    be 1.0; at alpha=1 it must be the explicit value we feed in."""
    alphas = np.linspace(0.0, 1.0, 6)
    per_probe = {}
    rng = np.random.RandomState(0)
    for p in SUBMODULE_PROBES:
        real = rng.randn(20, 16).astype(np.float32)
        # Construct alpha sweep where alpha=0 is real, alpha=1 is zeros
        # (so activity ratio drops to ~0 at alpha=1 -> the cliff must be found)
        per_probe[p.name] = np.stack(
            [real * (1.0 - float(a)) for a in alphas], axis=0
        ).astype(np.float32)
    res = analyse(alphas, per_probe)
    for p in SUBMODULE_PROBES:
        assert abs(res["ratios"][p.name][0] - 1.0) < 1e-5
        assert res["ratios"][p.name][-1] < 0.05
        assert 0.0 <= res["cliffs"][p.name] <= 1.0


def test_cache_loads_and_has_expected_shape():
    if not CACHE.exists():
        pytest.skip("e5_submodule cache not present yet")
    alphas, per_probe = load_cache(CACHE)
    assert len(alphas) >= 5
    assert set(per_probe.keys()) == {p.name for p in SUBMODULE_PROBES}
    for p in SUBMODULE_PROBES:
        arr = per_probe[p.name]
        assert arr.ndim == 3
        assert arr.shape[0] == len(alphas)
        assert arr.dtype == np.float32


def test_results_md_has_expected_rows():
    text = RESULTS.read_text()
    # Every probe must appear as a row in the per-probe table.
    for p in SUBMODULE_PROBES:
        assert f"`{p.name}`" in text, (
            f"probe `{p.name}` missing from results markdown"
        )
    # And every probe tensor must be cited (the probe-list table)
    for p in SUBMODULE_PROBES:
        assert p.tensor in text, (
            f"tensor `{p.tensor}` missing from results markdown"
        )
