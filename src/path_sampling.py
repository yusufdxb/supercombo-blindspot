"""Pure arc-length resampling of a polyline road path.

The Step 4b harness drives the ego kinematically: it precomputes a polyline of
road waypoints, then each 20 Hz tick advances a fixed arc-length (speed * dt)
and reads off an interpolated pose. Kinematic playback is deterministic — the
same scenario produces byte-identical camera framing every run, which is what a
reproduction benchmark needs. No CARLA dependency here so it is unit-testable.
"""

from __future__ import annotations

import math

import numpy as np


class PolylinePath:
    """A 2D polyline you can sample by arc-length.

    `sample(s)` returns `(x, y, yaw_deg)`: position `s` metres along the path and
    the heading of the segment it lands on. `s` is clamped to `[0, length]`.
    """

    def __init__(self, xy: np.ndarray):
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError(f"xy must be (N, 2), got {xy.shape}")
        if len(xy) == 0:
            raise ValueError("xy must have at least one vertex")
        self.xy = xy
        seg = np.diff(xy, axis=0)  # (N-1, 2)
        seg_len = np.hypot(seg[:, 0], seg[:, 1]) if len(seg) else np.array([])
        # cumulative arc length at each vertex
        self._cum = np.concatenate([[0.0], np.cumsum(seg_len)])
        self._seg_len = seg_len
        # per-segment heading in degrees; empty for a single-vertex path
        if len(seg):
            self._seg_yaw = np.degrees(np.arctan2(seg[:, 1], seg[:, 0]))
        else:
            self._seg_yaw = np.array([0.0])

    @property
    def length(self) -> float:
        """Total arc length in metres."""
        return float(self._cum[-1])

    @property
    def vertex_arc_lengths(self) -> np.ndarray:
        """Cumulative arc length at every polyline vertex — handy for `np.interp`
        of a per-vertex quantity (e.g. road elevation) against arc position."""
        return self._cum

    def arc_length_at(self, vertex_index: int) -> float:
        """Cumulative arc length from the start to polyline vertex `vertex_index`."""
        return float(self._cum[vertex_index])

    def sample(self, s: float) -> tuple[float, float, float]:
        """Pose `s` metres along the path: `(x, y, yaw_deg)`."""
        if len(self.xy) == 1 or self.length == 0.0:
            x, y = self.xy[0]
            return float(x), float(y), float(self._seg_yaw[0])

        s = min(max(s, 0.0), self.length)
        # segment index: last segment whose start cum-length <= s
        seg_i = int(np.searchsorted(self._cum, s, side="right") - 1)
        seg_i = min(max(seg_i, 0), len(self._seg_len) - 1)

        s0 = self._cum[seg_i]
        frac = (s - s0) / self._seg_len[seg_i] if self._seg_len[seg_i] > 0 else 0.0
        p0 = self.xy[seg_i]
        p1 = self.xy[seg_i + 1]
        x = p0[0] + frac * (p1[0] - p0[0])
        y = p0[1] + frac * (p1[1] - p0[1])
        return float(x), float(y), float(self._seg_yaw[seg_i])


def heading_deg(dx: float, dy: float) -> float:
    """Heading of a 2D vector in degrees, atan2 convention."""
    return math.degrees(math.atan2(dy, dx))
