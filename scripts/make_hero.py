"""Stitch four headline figures into a 2x2 hero composite for the README."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import figure_style as _figure_style  # editorial-print theme
_figure_style.apply()
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figures"

PANELS = [
    ("E1: 8 of 10 output heads collapse on CARLA", FIG / "e1_head_collapse.png"),
    ("E3: monitored uncertainty does not flag the collapse", FIG / "e3_confidence.png"),
    ("E4: transition shape is source-dependent", FIG / "e4_interpolation.png"),
    ("E6: recurrent spread detects the CARLA collapse", FIG / "e6_detector.png"),
]


def main() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    for ax, (title, path) in zip(axes.flat, PANELS):
        ax.imshow(mpimg.imread(path))
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Silent output collapse under CARLA shift and recurrent-state instrumentation",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    out = FIG / "hero.png"
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
