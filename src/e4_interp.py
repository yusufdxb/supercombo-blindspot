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
