"""E6: a self-aware OOD detector from supercombo's own internal state.

E2 showed the model's 512-D recurrent feature vector freezes to a single
point on CARLA. E6 asks whether a downstream monitor watching the rolling
spread of that vector, calibrated on real driving, would catch the
collapse, and at what alpha along the E4 sweep it fires.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def rolling_spread(hidden: np.ndarray, window: int) -> np.ndarray:
    """Per-frame trace of the rolling covariance of the hidden state.
    `hidden` is shape (T, D). Returns shape (T,), NaN before the window fills."""
    T, D = hidden.shape
    out = np.full(T, np.nan, dtype=np.float64)
    for t in range(window, T + 1):
        out[t - 1] = float(np.var(hidden[t - window:t], axis=0).sum())
    return out


def calibrate_threshold(real_spreads: np.ndarray, percentile: float = 1.0) -> float:
    """Below this spread = OOD. Pick the `percentile`-th percentile of the
    real-driving spread distribution so real drives stay above it ~99% of
    the time."""
    s = real_spreads[~np.isnan(real_spreads)]
    return float(np.percentile(s, percentile))
