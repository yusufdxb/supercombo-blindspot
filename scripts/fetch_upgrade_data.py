"""Reproducible fetch for the v0.9.6 second-model parity foundation
AND the real-weather OOD axis segments.

`data/` and `models/*.onnx` are gitignored, so THIS SCRIPT is the
reproducibility artifact for the second-model upgrade. It installs:

  - the v0.9.6 supercombo ONNX            -> models/supercombo_v096.onnx
  - the CI TEST_ROUTE seg-6 source        -> data/ci_v096_source/{fcamera.hevc, rlog.bz2}
  - comma's v0.9.6 model_replay reference -> data/ci_v096_ref/model_ref.bz2

Real-weather segments (added for the adverse-weather OOD axis):
  - EV6 night residential + headlight glare   -> data/ev6_night_source/{fcamera.hevc, rlog.bz2}
  - Ford Bronco night highway + glare          -> data/bronco_night_source/{fcamera.hevc, rlog.bz2}
  - Daytime-dry C3 control (in-distribution)  -> data/daytime_control_source/{fcamera.hevc, rlog.bz2}

All three are comma-3 (tici) at 1928x1208 yuv420p with liveCalibration records,
so load_real_six applies identically (no intrinsics confound).

Each artifact is reused from models/candidates/ when a valid cached copy is
present (the scout already downloaded them); only the seg-6 fcamera.hevc is
fetched over HTTP. The ONNX is sanity-checked (>40 MB, loads in onnxruntime,
not a 130-byte LFS pointer stub) before it is accepted.

Run:  env -u PYTHONPATH .venv/bin/python -m scripts.fetch_upgrade_data
      env -u PYTHONPATH .venv/bin/python -m scripts.fetch_upgrade_data --weather
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CANDIDATES = REPO / "models" / "candidates"

# Destinations
MODEL_DST = REPO / "models" / "supercombo_v096.onnx"
SRC_DIR = REPO / "data" / "ci_v096_source"
SRC_HEVC = SRC_DIR / "fcamera.hevc"
SRC_RLOG = SRC_DIR / "rlog.bz2"
REF_DIR = REPO / "data" / "ci_v096_ref"
REF_DST = REF_DIR / "model_ref.bz2"

# Cached scout copies (reused, never re-downloaded)
CACHED_MODEL = CANDIDATES / "supercombo_v096.onnx"
CACHED_RLOG = CANDIDATES / "seg6_rlog.bz2"
CACHED_REF = CANDIDATES / "v096_model_ref.bz2"

# CI TEST_ROUTE seg-6 fcamera.hevc (dongle 2f4452b03ccb98f0, seg 6).
# The v0.9.6 model_replay reference is matched to THIS route, so parity uses it.
HEVC_URL = (
    "https://commadataci.blob.core.windows.net/openpilotci/"
    "2f4452b03ccb98f0/2022-12-03--13-45-30/6/fcamera.hevc"
)

MIN_MODEL_BYTES = 40 * 1024 * 1024  # reject LFS pointer stubs / truncated downloads


def _copy_cached(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        raise FileNotFoundError(f"cached {label} missing at {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"  [{label}] copied cached {src.name} -> {dst} ({dst.stat().st_size/1e6:.2f} MB)")


def _verify_model(path: Path) -> None:
    """Reject stubs; confirm the ONNX loads in onnxruntime."""
    size = path.stat().st_size
    if size < MIN_MODEL_BYTES:
        raise RuntimeError(
            f"model at {path} is only {size} bytes (<40 MB) -- looks like an LFS "
            f"pointer stub or a truncated download, not the real ONNX"
        )
    # Lazy import: heavy, and not needed for the HTTP-only path.
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    print(f"  [model] onnxruntime load OK; {len(names)} inputs: {names}")


def _fetch_http(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [hevc] GET {url}")
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req) as resp:  # noqa: S310 (trusted comma CI bucket)
        status = resp.status
        clen = resp.headers.get("Content-Length")
        print(f"  [hevc] HTTP {status}  Content-Length={clen}")
        if status != 200:
            raise RuntimeError(f"HEVC fetch returned HTTP {status}")
        with open(dst, "wb") as f:
            shutil.copyfileobj(resp, f, length=8 * 1024 * 1024)
    print(f"  [hevc] saved -> {dst} ({dst.stat().st_size/1e6:.2f} MB)")


CI_BASE = "https://commadataci.blob.core.windows.net/openpilotci"

# Real-weather adverse-condition segments (all comma-3 / tici, 1928x1208).
# URL pattern: {CI_BASE}/{dongle}/{time}/{seg}/{file}
WEATHER_SEGMENTS: dict[str, dict] = {
    "ev6_night_source": {
        "dongle": "d545129f3ca90f28",
        "time": "2022-11-07--20-43-08",
        "seg": "3",
        "label": "EV6 night residential + oncoming-headlight glare",
    },
    "bronco_night_source": {
        "dongle": "54827bf84c38b14f",
        "time": "2023-01-26--21-59-07",
        "seg": "4",
        "label": "Ford Bronco night highway + tail-light/sign glare",
    },
    "daytime_control_source": {
        "dongle": "376bf99325883932",
        "time": "2022-10-27--13-41-22",
        "seg": "1",
        "label": "Daytime-dry C3 in-distribution control",
    },
}

MIN_HEVC_BYTES = 2 * 1024 * 1024  # any real hevc segment is >2 MB


def _fetch_weather_segment(name: str, info: dict) -> tuple[Path, Path]:
    """Fetch one weather segment's fcamera.hevc + rlog.bz2. Returns (hevc_path, rlog_path)."""
    base = CI_BASE + f"/{info['dongle']}/{info['time']}/{info['seg']}"
    dst_dir = REPO / "data" / name
    dst_dir.mkdir(parents=True, exist_ok=True)
    hevc_dst = dst_dir / "fcamera.hevc"
    rlog_dst = dst_dir / "rlog.bz2"

    for fname, dst, min_bytes in [
        ("fcamera.hevc", hevc_dst, MIN_HEVC_BYTES),
        ("rlog.bz2", rlog_dst, 1024),
    ]:
        url = f"{base}/{fname}"
        if dst.exists() and dst.stat().st_size >= min_bytes:
            print(f"  [{name}/{fname}] cached ({dst.stat().st_size/1e6:.2f} MB)")
            continue
        print(f"  [{name}/{fname}] GET {url}")
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req) as resp:  # noqa: S310
            status = resp.status
            clen = resp.headers.get("Content-Length")
            print(f"  [{name}/{fname}] HTTP {status}  Content-Length={clen}")
            if status != 200:
                raise RuntimeError(f"fetch {fname} returned HTTP {status}")
            with open(dst, "wb") as f:
                shutil.copyfileobj(resp, f, length=8 * 1024 * 1024)
        print(f"  [{name}/{fname}] saved -> {dst} ({dst.stat().st_size/1e6:.2f} MB)")
    return hevc_dst, rlog_dst


def _verify_hevc(hevc: Path, label: str) -> None:
    """Run ffprobe to confirm 1928x1208 yuv420p HEVC; paste the stream info."""
    import subprocess
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_streams", "-select_streams", "v:0",
         "-print_format", "compact", str(hevc)],
        capture_output=True, text=True,
    )
    line = result.stdout.strip()
    print(f"  [ffprobe {label}] {line[:200]}")
    if "codec_name=hevc" not in line:
        raise RuntimeError(f"{hevc}: ffprobe did not report codec=hevc; got: {line}")
    if "1928" not in line or "1208" not in line:
        raise RuntimeError(f"{hevc}: ffprobe did not report 1928x1208; got: {line}")
    if "yuv420p" not in line:
        raise RuntimeError(f"{hevc}: ffprobe did not report yuv420p; got: {line}")
    print(f"  [ffprobe {label}] OK: hevc 1928x1208 yuv420p confirmed")


def _verify_tici(rlog: Path, label: str) -> None:
    """Read rlog deviceType and assert it equals 'tici'."""
    from src.rlog import iter_events
    device_type = None
    for ev in iter_events(rlog):
        try:
            w = ev.which()
        except Exception:
            continue
        if w == "initData":
            try:
                device_type = ev.initData.deviceType
            except Exception:
                pass
            break
    if device_type is None:
        # Some rlogs omit initData; try deviceState
        for ev in iter_events(rlog):
            try:
                if ev.which() == "deviceState":
                    pass  # deviceState doesn't carry deviceType, skip
            except Exception:
                continue
        print(f"  [rlog {label}] WARNING: no initData/deviceType found; "
              f"intrinsics assumed tici based on segment metadata")
        return
    print(f"  [rlog {label}] deviceType = {device_type!r}")
    if str(device_type) not in ("tici", "DeviceType.tici"):
        raise RuntimeError(
            f"Expected tici device, got {device_type!r} in {rlog}"
        )
    print(f"  [rlog {label}] tici confirmed -> no intrinsics confound")


def fetch_weather_segments() -> None:
    print("\n=== real-weather segment fetch ===\n")
    for name, info in WEATHER_SEGMENTS.items():
        print(f"-- {name}: {info['label']}")
        hevc, rlog = _fetch_weather_segment(name, info)
        _verify_hevc(hevc, name)
        _verify_tici(rlog, name)
        print()
    print("All weather segments fetched and verified.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weather", action="store_true",
                    help="also fetch (or verify) the real-weather OOD segments")
    args, _ = ap.parse_known_args()

    if args.weather:
        fetch_weather_segments()
        return 0

    print("=== v0.9.6 second-model fetch ===\n")

    print("1) v0.9.6 supercombo model")
    if MODEL_DST.exists() and MODEL_DST.stat().st_size >= MIN_MODEL_BYTES:
        print(f"  [model] already present at {MODEL_DST} ({MODEL_DST.stat().st_size/1e6:.2f} MB)")
    else:
        _copy_cached(CACHED_MODEL, MODEL_DST, "model")
    _verify_model(MODEL_DST)

    print("\n2) CI TEST_ROUTE seg-6 source (fcamera.hevc + rlog.bz2)")
    # rlog: reuse the cached seg-6 copy.
    if SRC_RLOG.exists() and SRC_RLOG.stat().st_size > 0:
        print(f"  [rlog] already present at {SRC_RLOG} ({SRC_RLOG.stat().st_size/1e6:.2f} MB)")
    else:
        _copy_cached(CACHED_RLOG, SRC_RLOG, "rlog")
    # hevc: must be fetched over HTTP (not cached on disk).
    if SRC_HEVC.exists() and SRC_HEVC.stat().st_size > 40 * 1024 * 1024:
        print(f"  [hevc] already present at {SRC_HEVC} ({SRC_HEVC.stat().st_size/1e6:.2f} MB)")
    else:
        _fetch_http(HEVC_URL, SRC_HEVC)

    print("\n3) v0.9.6 model_replay reference")
    if REF_DST.exists() and REF_DST.stat().st_size > 0:
        print(f"  [ref] already present at {REF_DST} ({REF_DST.stat().st_size/1e6:.2f} MB)")
    else:
        _copy_cached(CACHED_REF, REF_DST, "ref")

    print("\n=== final sizes ===")
    for label, p in [
        ("model", MODEL_DST), ("hevc", SRC_HEVC), ("rlog", SRC_RLOG), ("ref", REF_DST),
    ]:
        ok = p.exists()
        sz = f"{p.stat().st_size/1e6:.2f} MB" if ok else "MISSING"
        print(f"  {label:6s} {str(p):60s} {sz}")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
