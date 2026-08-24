"""Production styling for the viral demo: real-typography text (Pillow + Inter /
Space Grotesk) and a clean minimal HUD. Replaces the cv2 Hershey rendering.

All public functions take/return BGR uint8 numpy frames (cv2 convention).
"""
from __future__ import annotations

from functools import lru_cache

import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = os.environ.get(
    "DEMO_FONT_DIR", str(Path.home() / ".local/share/fonts/demo")
)
GROTESK = f"{FONT_DIR}/SpaceGrotesk.ttf"   # headers / display
INTER = f"{FONT_DIR}/Inter.ttf"            # HUD / body

# palette (RGB)
INK = (236, 238, 241)
DIM = (150, 156, 166)
GREEN = (70, 214, 130)
RED = (242, 88, 88)
AMBER = (240, 190, 90)
PANEL = (12, 14, 18)


@lru_cache(maxsize=64)
def _font(path: str, size: int, weight: str):
    f = ImageFont.truetype(path, size)
    try:
        f.set_variation_by_name(weight)
    except Exception:
        pass
    return f


def _pil(bgr):
    return Image.fromarray(bgr[:, :, ::-1].copy())


def _bgr(img):
    return np.asarray(img)[:, :, ::-1].copy()


def _text(d, xy, s, font, fill, anchor="la", spacing=0):
    if spacing:
        # manual letter-spacing for small-caps labels
        x, y = xy
        for ch in s:
            d.text((x, y), ch, font=font, fill=fill, anchor="la")
            x += d.textlength(ch, font=font) + spacing
    else:
        d.text(xy, s, font=font, fill=fill, anchor=anchor)


# --------------------------------------------------------------------------
# displayed-image degradation: morph the shown frame toward the real CARLA
# render by the same alpha the model sees, so "blinding" is visible on screen.
# --------------------------------------------------------------------------

def degrade_display(real_bgr, carla_bgr, alpha):
    a = float(np.clip(alpha, 0, 1))
    if a <= 0:
        return real_bgr
    out = (1.0 - a) * real_bgr.astype(np.float32) + a * carla_bgr.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# HUD
# --------------------------------------------------------------------------

def _scrim(img):
    """Top + bottom dark gradients so HUD text reads over any image."""
    arr = np.array(img).astype(np.float32)
    h, w = arr.shape[:2]
    top_h, bot_h = 150, 240
    top = np.linspace(0.80, 0.0, top_h)[:, None, None]
    arr[:top_h] *= (1 - top)
    bot = np.linspace(0.0, 0.86, bot_h)[:, None, None]
    arr[h - bot_h:] *= (1 - bot)
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def _gauge(d, x, y, w, label, value_txt, frac, color, anchor_l="la"):
    _text(d, (x, y), label.upper(), _font(INTER, 24, "SemiBold"), DIM, spacing=2)
    d.text((x, y + 28), value_txt, font=_font(GROTESK, 42, "Bold"), fill=color, anchor="la")
    by = y + 88
    d.rounded_rectangle([x, by, x + w, by + 9], radius=4, fill=(46, 50, 58))
    fw = max(int(w * float(np.clip(frac, 0, 1))), 6)
    d.rounded_rectangle([x, by, x + fw, by + 9], radius=4, fill=color)


def draw_frame(bgr, alpha, conf_frac, ood_frac, ood_fires, collapsed, reach_m):
    h, w = bgr.shape[:2]
    img = _scrim(_pil(bgr))
    d = ImageDraw.Draw(img, "RGBA")

    # top-left source tag
    d.ellipse([40, 48, 58, 66], fill=RED if collapsed else GREEN)
    d.text((72, 42), "openpilot supercombo", font=_font(INTER, 30, "SemiBold"), fill=INK, anchor="la")
    d.text((72, 78), "live predicted path  /  real dashcam", font=_font(INTER, 22, "Regular"),
           fill=DIM, anchor="la")

    # top-right input-shift readout
    shift = int(round(100 * float(np.clip(alpha / 0.85, 0, 1))))
    d.text((w - 40, 40), f"{shift}%", font=_font(GROTESK, 44, "Bold"), fill=AMBER, anchor="ra")
    d.text((w - 40, 92), "INPUT SHIFTED TOWARD SIMULATOR", font=_font(INTER, 19, "Medium"),
           fill=DIM, anchor="ra")

    # collapse banner (sits above the gauges, no overlap)
    if collapsed:
        d.text((w // 2, h - 196), "PREDICTED PATH COLLAPSED", font=_font(GROTESK, 50, "Bold"),
               fill=RED, anchor="ma")
        d.text((w // 2, h - 150), "model still reports high confidence",
               font=_font(INTER, 25, "Medium"), fill=(210, 214, 220), anchor="ma")

    # bottom gauges
    gy = h - 124
    _gauge(d, 40, gy, 300, "model confidence", f"{round(100*conf_frac)}%",
           conf_frac, GREEN if conf_frac > 0.5 else AMBER)
    _gauge(d, w - 340, gy, 300, "internal OOD monitor",
           "ALARM" if ood_fires else "normal", ood_frac, RED if ood_fires else GREEN)
    return _bgr(img)


# --------------------------------------------------------------------------
# cards
# --------------------------------------------------------------------------

def _card(w, h):
    img = Image.new("RGB", (w, h), (9, 10, 13))
    return img, ImageDraw.Draw(img, "RGBA")


def title_card(w, h):
    img, d = _card(w, h)
    cx = w // 2
    d.text((cx, h // 2 - 70), "Does a self-driving car", font=_font(GROTESK, 92, "Bold"),
           fill=INK, anchor="mm")
    d.text((cx, h // 2 + 36), "know when it's blind?", font=_font(GROTESK, 92, "Bold"),
           fill=INK, anchor="mm")
    d.text((cx, h // 2 + 150), "openpilot supercombo, running live on real dashcam footage",
           font=_font(INTER, 32, "Regular"), fill=DIM, anchor="mm")
    return _bgr(img)


def end_card(w, h):
    img, d = _card(w, h)
    cx = w // 2
    d.text((cx, h // 2 - 150), "DISTRIBUTION-SHIFT TEARDOWN", font=_font(INTER, 26, "SemiBold"),
           fill=DIM, anchor="mm")
    lines = [("The path collapsed. The model's confidence did not.", INK),
             ("Output-side monitors miss it.", DIM),
             ("A monitor on the internal features catches it.", DIM)]
    y = h // 2 - 60
    for s, col in lines:
        d.text((cx, y), s, font=_font(GROTESK, 50, "Medium"), fill=col, anchor="mm")
        y += 70
    d.text((cx, h // 2 + 180), "github.com/yusufdxb/supercombo-blindspot",
           font=_font(INTER, 30, "Medium"), fill=AMBER, anchor="mm")
    return _bgr(img)
