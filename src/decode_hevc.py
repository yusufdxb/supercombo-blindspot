"""Stream-decode a comma HEVC file to yuv420p frames via ffmpeg subprocess."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np


def yuv_frame_iter(hevc_path: Path, width: int = 1928, height: int = 1208) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield (Y, U, V) plane arrays per frame. Y is HxW uint8; U/V are H/2 x W/2 uint8."""
    y_size = width * height
    uv_size = (width // 2) * (height // 2)
    frame_size = y_size + 2 * uv_size

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-i", str(hevc_path),
        "-f", "rawvideo",
        "-pix_fmt", "yuv420p",
        "-",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 * frame_size)
    try:
        while True:
            buf = proc.stdout.read(frame_size)
            if len(buf) < frame_size:
                break
            arr = np.frombuffer(buf, dtype=np.uint8)
            Y = arr[:y_size].reshape(height, width)
            U = arr[y_size:y_size + uv_size].reshape(height // 2, width // 2)
            V = arr[y_size + uv_size:].reshape(height // 2, width // 2)
            yield Y, U, V
    finally:
        proc.stdout.close()
        proc.wait()
