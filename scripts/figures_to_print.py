"""Convert the repo's dark-theme figures to a white-background print theme.

Journals print on white. The figures in report/figures were authored on a dark
background (#0d1017 family). This does a per-pixel HSV pass:

  * near-neutral pixels (background, axes, ticks, gridlines, all text) get their
    value inverted, so dark background -> white and light text -> black, with
    antialiasing preserved because the inversion is continuous. Neutrality is
    judged on absolute chroma (max channel minus min channel), not on HSV
    saturation: the near-black #0d1017 background is only 10 levels of blue off
    grey, which is a large *relative* saturation and a tiny absolute chroma;
  * chromatic pixels (the data series) keep hue and saturation, and are only
    darkened enough to hold contrast on white.

Output: report/figures_print/<same names>. Originals are not modified.

    python3 scripts/figures_to_print.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "report" / "figures"
DST = ROOT / "report" / "figures_print"

NEUTRAL_CHROMA = 0.12  # below this max-min channel spread a pixel is ink/background
MAX_VALUE = 0.82       # cap brightness of coloured series so they read on white
WHITE_POINT = 0.07     # levels stretch: inverted background snaps to pure white


DARK_BG = (13, 16, 23, 255)  # #0d1017, the theme these figures were authored on


def convert(img: Image.Image) -> Image.Image:
    # some figures were saved with a transparent background; flatten onto the
    # dark theme colour first so those regions invert to white like the rest
    src = img.convert("RGBA")
    src = Image.alpha_composite(Image.new("RGBA", src.size, DARK_BG), src)
    rgba = np.asarray(src).astype(np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3:]

    mx = rgb.max(axis=-1)
    mn = rgb.min(axis=-1)
    val = mx
    chroma = mx - mn

    neutral = chroma < NEUTRAL_CHROMA
    out = rgb.copy()

    # neutral pixels: invert luminance, keep them grey
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # levels stretch so the page background lands on true white and the ink on
    # true black instead of the 6%-off greys the inversion alone would give
    inv = np.clip((1.0 - lum - WHITE_POINT) / (1.0 - 2.0 * WHITE_POINT), 0.0, 1.0)
    inv = inv[..., None]
    out = np.where(neutral[..., None], np.repeat(inv, 3, axis=-1), out)

    # coloured pixels: scale value down if too bright for a white page
    scale = np.minimum(1.0, MAX_VALUE / np.maximum(val, 1e-6))[..., None]
    out = np.where(neutral[..., None], out, rgb * scale)

    out = np.clip(out, 0.0, 1.0)
    res = np.concatenate([out, alpha], axis=-1)
    return Image.fromarray((res * 255.0).round().astype(np.uint8), mode="RGBA")


def is_dark_theme(img: Image.Image) -> bool:
    """Median luminance of the opaque pixels separates the two families cleanly:
    the dark-theme figures sit at 0.06, the light ones at 0.87 and above."""
    a = np.asarray(img.convert("RGBA")).astype(np.float32) / 255.0
    rgb, alpha = a[..., :3], a[..., 3]
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    opaque = alpha > 0.5
    return bool(opaque.any() and np.median(lum[opaque]) < 0.30)


def flatten_white(img: Image.Image) -> Image.Image:
    src = img.convert("RGBA")
    bg = Image.new("RGBA", src.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, src).convert("RGB")


# the panels carry their own titles, so the composite only adds subfigure letters
HERO_PANELS = [
    ("(a)", "e1_head_collapse.png"),
    ("(b)", "e3_confidence.png"),
    ("(c)", "e4_interpolation.png"),
    ("(d)", "e6_detector.png"),
]


def build_hero() -> None:
    """Rebuild the 2x2 teaser from the print-theme panels, on white and with
    lettered subfigure labels instead of the README's headline question."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor("white")
    for ax, (title, name) in zip(axes.flat, HERO_PANELS):
        ax.imshow(mpimg.imread(DST / name))
        ax.set_title(title, fontsize=11, color="black", loc="left")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(DST / "hero.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    if not SRC.is_dir():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    converted, copied = [], []
    for p in sorted(SRC.glob("*.png")):
        if p.name == "hero.png":
            continue  # composite, rebuilt from the converted panels below
        img = Image.open(p)
        if is_dark_theme(img):
            convert(img).convert("RGB").save(DST / p.name)
            converted.append(p.name)
        else:
            flatten_white(img).save(DST / p.name)
            copied.append(p.name)
    build_hero()
    print(f"{DST}: {len(converted)} inverted from dark theme, "
          f"{len(copied)} already light (flattened onto white), hero rebuilt")
    print("  inverted:", ", ".join(converted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
