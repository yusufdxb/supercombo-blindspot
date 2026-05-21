"""Tests for E4: real-to-sim interpolation sweep."""

from __future__ import annotations

import numpy as np
import pytest

from src.e4_interp import blend


def test_blend_endpoints_and_midpoint():
    r = np.full((6, 4, 4), 10.0, dtype=np.float32)
    c = np.full((6, 4, 4), 50.0, dtype=np.float32)
    assert np.allclose(blend(r, c, 0.0), 10.0)
    assert np.allclose(blend(r, c, 1.0), 50.0)
    assert np.allclose(blend(r, c, 0.5), 30.0)


def test_blend_is_float32_and_in_range():
    r = np.zeros((6, 2, 2), dtype=np.uint8)
    c = np.full((6, 2, 2), 255, dtype=np.uint8)
    out = blend(r, c, 0.25)
    assert out.dtype == np.float32
    assert out.min() >= 0.0 and out.max() <= 255.0
