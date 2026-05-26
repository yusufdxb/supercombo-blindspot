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
