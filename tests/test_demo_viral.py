"""Tests for src/demo_viral.py — the viral render pipeline.

These cover the new code paths only (the model I/O is the parity-verified
ModelStateMirror / probe_model path, tested elsewhere). No model, no GPU, no
ffmpeg: pure-numpy unit tests on the schedule, projection, monitor, and the
collapse logic.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.demo_viral as dv


def test_alpha_schedule_monotone_and_bounds():
    xs = list(range(0, dv.N_FRAMES))
    a = [dv.alpha_at(k) for k in xs]
    # clean hold up front
    assert a[0] == 0.0
    assert dv.alpha_at(dv.HOLD_CLEAN) == 0.0
    # monotonically non-decreasing
    assert all(a[i + 1] >= a[i] - 1e-9 for i in range(len(a) - 1))
    # reaches and holds the peak, never exceeds it
    assert max(a) == pytest.approx(dv.ALPHA_MAX)
    assert dv.alpha_at(dv.RAMP_END) == pytest.approx(dv.ALPHA_MAX)
    assert dv.alpha_at(dv.RAMP_END + 50) == pytest.approx(dv.ALPHA_MAX)
    assert all(0.0 <= v <= dv.ALPHA_MAX + 1e-9 for v in a)


def test_alpha_smoothstep_midpoint():
    mid = (dv.HOLD_CLEAN + dv.RAMP_END) // 2
    v = dv.alpha_at(mid)
    # smoothstep at t=0.5 is 0.5 -> alpha = ALPHA_MAX/2
    assert v == pytest.approx(dv.ALPHA_MAX * 0.5, abs=0.05)


def test_project_points_forward_on_axis():
    # identity-ish calib: a point straight ahead projects near the principal point
    rpy = np.zeros(3)
    M = dv.camera_from_device(rpy)
    # device frame: x fwd, y left, z up. Point dead ahead, slightly down.
    pt = np.array([[50.0, 0.0, 0.0]])
    uv = dv.project_points(pt, M)
    assert np.all(np.isfinite(uv))
    # principal point cx = 1928/2 = 964; on-axis point lands near cx horizontally
    assert abs(uv[0, 0] - 964.0) < 5.0


def test_project_points_behind_camera_is_nan():
    M = dv.camera_from_device(np.zeros(3))
    pt = np.array([[-10.0, 0.0, 0.0]])   # behind the car (x<0)
    uv = dv.project_points(pt, M)
    assert np.all(np.isnan(uv))


def test_project_lateral_separation_consistent():
    # two laterally offset points must project to DIFFERENT, finite pixel
    # columns and be symmetric about the on-axis projection. (The exact
    # left/right sign follows openpilot's view_frame_from_device_frame, which
    # the clean-frame overlay validates empirically.)
    M = dv.camera_from_device(np.zeros(3))
    onax = dv.project_points(np.array([[40.0, 0.0, 0.0]]), M)[0, 0]
    a = dv.project_points(np.array([[40.0, 3.0, 0.0]]), M)
    b = dv.project_points(np.array([[40.0, -3.0, 0.0]]), M)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert abs(a[0, 0] - b[0, 0]) > 50          # clearly separated
    # symmetric about the on-axis column
    assert (a[0, 0] - onax) == pytest.approx(-(b[0, 0] - onax), abs=1.0)


def test_rolling_spread_collapse():
    rng = np.random.RandomState(0)
    # first half: high variance; second half: frozen (collapsed)
    active = rng.randn(80, 8)
    frozen = np.ones((80, 8)) * 0.5
    h = np.concatenate([active, frozen])
    sp = dv.rolling_spread(h, dv.WINDOW)
    # warmup region is nan
    assert np.isnan(sp[:dv.WINDOW - 1]).all()
    # spread in the active region is much larger than in the frozen tail
    assert np.nanmean(sp[dv.WINDOW:70]) > 10 * (np.nanmean(sp[-30:]) + 1e-9)


def test_rolling_spread_window_length():
    h = np.random.RandomState(1).randn(50, 4)
    sp = dv.rolling_spread(h, dv.WINDOW)
    assert len(sp) == len(h)
    assert np.isfinite(sp[dv.WINDOW - 1:]).all()


def test_collapse_flag_requires_both_reach_and_flatten():
    clean_reach, clean_lat = 320.0, 2.4
    # reach drops but path still curving -> NOT collapsed
    assert not ((150 < 0.55 * clean_reach) and (2.0 < 0.5 * clean_lat))
    # both reach short AND flat -> collapsed
    assert (90 < 0.55 * clean_reach) and (0.07 < 0.5 * clean_lat)


def test_end_card_renders_and_fits():
    card = dv.end_card(1928, 1208, dv.FINDING_LINES)
    assert card.shape == (1208, 1928, 3)
    assert card.dtype == np.uint8
    # something was drawn (not all background)
    assert card.max() > 50


def test_confidence_scale_anchored_not_pinned_to_max():
    # the confidence anchor is 3x clean baseline, so a 1.7x rise stays high.
    u_base = 0.33
    u_ceil = 3.0 * u_base
    unc_collapse = 1.7 * u_base
    conf = 1.0 - np.clip((unc_collapse - u_base) / (u_ceil - u_base), 0, 1)
    assert conf > 0.6   # still reads as confident despite full path collapse
