"""Unit tests for src/path_sampling — pure polyline arc-length resampling.

No CARLA needed. This is the kinematic-playback math: given a polyline of road
waypoints, advance the ego a fixed arc-length per tick and read off an
interpolated pose.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.path_sampling import PolylinePath


def test_straight_line_length():
    path = PolylinePath(np.array([[0.0, 0.0], [10.0, 0.0]]))
    assert path.length == pytest.approx(10.0)


def test_straight_line_midpoint():
    path = PolylinePath(np.array([[0.0, 0.0], [10.0, 0.0]]))
    x, y, yaw = path.sample(5.0)
    assert (x, y) == pytest.approx((5.0, 0.0))
    assert yaw == pytest.approx(0.0)


def test_l_shape_length_and_corner():
    path = PolylinePath(np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]))
    assert path.length == pytest.approx(20.0)
    # 5 m up the second leg -> (10, 5), heading +y = 90 deg
    x, y, yaw = path.sample(15.0)
    assert (x, y) == pytest.approx((10.0, 5.0))
    assert yaw == pytest.approx(90.0)


def test_sample_clamps_below_zero():
    path = PolylinePath(np.array([[0.0, 0.0], [10.0, 0.0]]))
    x, y, _ = path.sample(-5.0)
    assert (x, y) == pytest.approx((0.0, 0.0))


def test_sample_clamps_past_end():
    path = PolylinePath(np.array([[0.0, 0.0], [10.0, 0.0]]))
    x, y, _ = path.sample(999.0)
    assert (x, y) == pytest.approx((10.0, 0.0))


def test_negative_x_heading_is_180():
    path = PolylinePath(np.array([[0.0, 0.0], [-10.0, 0.0]]))
    _, _, yaw = path.sample(5.0)
    assert abs(((yaw - 180.0 + 180.0) % 360.0) - 180.0) == pytest.approx(0.0)


def test_arc_length_of_overpass_index():
    """arc_length_at(i) gives the cumulative distance to polyline vertex i —
    used to locate the overpass crossing on the drive path."""
    path = PolylinePath(np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0]]))
    assert path.arc_length_at(0) == pytest.approx(0.0)
    assert path.arc_length_at(1) == pytest.approx(3.0)
    assert path.arc_length_at(2) == pytest.approx(7.0)


def test_two_point_degenerate_zero_length():
    """A degenerate single-vertex path has zero length and samples that point."""
    path = PolylinePath(np.array([[5.0, 5.0]]))
    assert path.length == pytest.approx(0.0)
    x, y, _ = path.sample(3.0)
    assert (x, y) == pytest.approx((5.0, 5.0))
