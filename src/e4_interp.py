"""E4: real-to-sim interpolation sweep.

Blends real Subaru model-frame inputs toward CARLA inputs and measures
supercombo along the path, to test whether the output collapse from the
E1/E2/E3 teardown is a sharp cliff or a smooth gradient.

    python -m src.e4_interp              # analysis + figure, from the cache
    env -u PYTHONPATH .venv/bin/python -m src.e4_interp --collect  # re-run model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from src.teardown import (CARLA_C, HEAD_NAMES, N, REAL_C, SCALARS, WARMUP,
                          WARN_C, _flat, _plt, _post)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / "report" / "e4_collected.npz"
FIG = ROOT / "report" / "figures" / "e4_interpolation.png"
RESULTS = ROOT / "report" / "e4_results.md"
SUBARU_HEVC = DATA / "subaru_source" / "fcamera.hevc"
SUBARU_RLOG = DATA / "subaru_source" / "rlog.bz2"
CARLA_NPY = DATA / "domain_gap" / "carla_rgb.npy"

BASE_ALPHAS = [round(0.1 * i, 4) for i in range(11)]  # 0.0, 0.1, ... 1.0
REFINE_GAP = 0.2      # insert a midpoint when adjacent activity differs by > this
REFINE_ROUNDS = 2     # refinement passes after the first sweep
CLIFF_WIDTH = 0.2     # transition width below this reads as a cliff


def blend(real_six: np.ndarray, carla_six: np.ndarray, alpha: float) -> np.ndarray:
    """Convex blend (1-alpha)*real + alpha*carla in float32."""
    r = np.asarray(real_six, dtype=np.float32)
    c = np.asarray(carla_six, dtype=np.float32)
    return (1.0 - alpha) * r + alpha * c


def _crossing(xs: list[float], ys: list[float], level: float) -> float:
    """First x where the piecewise-linear curve (xs, ys) crosses `level`."""
    for i in range(1, len(xs)):
        y0, y1 = ys[i - 1], ys[i]
        if (y0 - level) * (y1 - level) <= 0 and y0 != y1:
            x0, x1 = xs[i - 1], xs[i]
            return x0 + (level - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")


def transition_width(alphas, norm_activity: dict) -> tuple[float, float]:
    """Return (alpha@0.9, alpha@0.1): the alphas where normalized activity
    crosses 0.9 and 0.1. The transition width is the difference."""
    xs = sorted(alphas)
    ys = [norm_activity[a] for a in xs]
    return _crossing(xs, ys, 0.9), _crossing(xs, ys, 0.1)
