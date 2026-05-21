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


def save_cache(path: Path, collected: dict[float, dict]) -> None:
    """Persist {alpha: {head: array}} as one compressed .npz."""
    flat = {f"{a:.4f}__{k}": v
            for a, d in collected.items() for k, v in d.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **flat)


def load_cache(path: Path) -> dict[float, dict]:
    """Inverse of `save_cache`."""
    z = np.load(path)
    out: dict[float, dict] = {}
    for key in z.files:
        a_str, _, name = key.partition("__")
        out.setdefault(float(a_str), {})[name] = z[key]
    return out


def activity_per_head(d: dict) -> dict[str, float]:
    """Per-head temporal activity: sum of per-element std over the frames."""
    return {name: float(_flat(d[name]).std(axis=0).sum())
            for name in SCALARS + HEAD_NAMES}


def normalized_activity(per_head_by_alpha: dict[float, dict]) -> dict[float, float]:
    """Mean over heads of (head activity / head activity at alpha 0).
    Equals 1.0 at alpha 0 by construction."""
    base = per_head_by_alpha[0.0]
    out: dict[float, float] = {}
    for a, ph in per_head_by_alpha.items():
        ratios = [ph[h] / base[h] for h in base if base[h] > 1e-12]
        out[a] = float(np.mean(ratios))
    return out


def feature_centroid(d: dict) -> np.ndarray:
    """Mean hidden_state vector (512,) over the frames."""
    return d["hidden_state"].astype(np.float64).mean(axis=0)


def feature_projection(centroid_by_alpha: dict[float, np.ndarray]) -> dict[float, float]:
    """Project each centroid onto the real->CARLA axis w = mu_c - mu_r:
    f(alpha) = ((mu_a - mu_r) . w) / (w . w). f(0)=0, f(1)=1."""
    mu_r = centroid_by_alpha[0.0]
    mu_c = centroid_by_alpha[1.0]
    w = mu_c - mu_r
    denom = float(w @ w) + 1e-12
    return {a: float((mu - mu_r) @ w / denom)
            for a, mu in centroid_by_alpha.items()}


def feature_spread(d: dict) -> float:
    """Trace of the hidden_state covariance (total variance)."""
    return float(np.var(d["hidden_state"].astype(np.float64), axis=0).sum())


def mean_uncertainty(d: dict) -> float:
    """Mean predicted plan uncertainty over the frames."""
    return float(_flat(d["plan_std"]).mean())
