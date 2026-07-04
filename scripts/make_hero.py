"""Stitch four headline figures into a 2x2 hero composite for the README."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import physx_style as _physx_style  # editorial-print theme
_physx_style.apply()
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"

PANELS = [
    ("E1: 8 of 10 output heads collapse on CARLA", FIG / "e1_head_collapse.png"),
    ("E3: confidence does not flag it",            FIG / "e3_confidence.png"),
    ("E4: cliff, not gradient",                    FIG / "e4_interpolation.png"),
    ("E6: a monitor on internals catches it",      FIG / "e6_detector.png"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    for ax, (title, path) in zip(axes.flat, PANELS):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Does openpilot's driving model know when it's blind?  No (but a monitor on internals does).",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG / "hero.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
