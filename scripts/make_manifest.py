"""Generate report/MANIFEST.json — provenance record for phantom-braking.

Captures:
  - SHA-256 + size + mtime for every models/*.onnx and report/*.npz present locally.
  - Git state: commit, branch, dirty flag.
  - Python version and key package versions.
  - Data provenance: WEATHER_SEGMENTS and CI parity route parsed from
    scripts/fetch_upgrade_data.py (the authoritative source; never hardcoded here).
  - Openpilot model tags per known filename.

Stdlib only: hashlib, json, subprocess, sys, pathlib, importlib.metadata.

Usage:
    .venv/bin/python scripts/make_manifest.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

KEY_PACKAGES = [
    "onnxruntime-gpu",
    "onnxruntime",
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "carla",
]

MODEL_TAGS: dict[str, str] = {
    "supercombo.onnx": "openpilot v0.9.7",
    "supercombo_v096.onnx": "openpilot v0.9.6",
    "supercombo_probed.onnx": "openpilot v0.9.7 (E6 probed variant)",
    "supercombo_submodule_probed.onnx": "openpilot v0.9.7 (E5 submodule probed variant)",
}


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    """Stream a file in 1 MiB chunks to avoid loading it whole into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _file_record(path: Path) -> dict:
    st = path.stat()
    rec: dict = {
        "filename": path.name,
        "path_relative": str(path.relative_to(REPO)),
        "sha256": _sha256(path),
        "size_bytes": st.st_size,
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
    }
    tag = MODEL_TAGS.get(path.name)
    if tag:
        rec["model_tag"] = tag
    return rec


def _git_info() -> dict:
    def _run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    commit = _run("rev-parse", "HEAD")
    branch = _run("rev-parse", "--abbrev-ref", "HEAD")
    dirty_output = _run("status", "--porcelain")
    dirty = bool(dirty_output)
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "absent"


def _env_info() -> dict:
    return {
        "python_version": sys.version,
        "packages": {pkg: _pkg_version(pkg) for pkg in KEY_PACKAGES},
    }


def _parse_data_provenance() -> dict:
    """Parse WEATHER_SEGMENTS dict and CI parity route from fetch_upgrade_data.py.

    We exec the relevant lines by importing the module directly, relying on
    stdlib ast to extract the dict literal safely. Fallback: import the module
    and read its attributes.
    """
    fetch_mod_path = REPO / "scripts" / "fetch_upgrade_data.py"

    # Strategy: use importlib + spec to import the module in isolation
    # without running main(), then read WEATHER_SEGMENTS and HEVC_URL.
    import importlib.util as _ilu
    spec = _ilu.spec_from_file_location("_fetch_upgrade_data", fetch_mod_path)
    mod = _ilu.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]

    weather: dict = {}
    for seg_name, info in mod.WEATHER_SEGMENTS.items():
        weather[seg_name] = {
            "dongle": info["dongle"],
            "time": info["time"],
            "seg": info["seg"],
            "label": info["label"],
        }

    # CI parity route: dongle and seg are embedded in HEVC_URL
    # e.g. .../2f4452b03ccb98f0/2022-12-03--13-45-30/6/fcamera.hevc
    hevc_url: str = mod.HEVC_URL
    parts = hevc_url.rstrip("/").split("/")
    # Find the dongle in the URL (the part after /openpilotci/)
    ci_dongle = ""
    ci_seg = ""
    ci_time = ""
    for i, part in enumerate(parts):
        if part == "openpilotci" and i + 3 < len(parts):
            ci_dongle = parts[i + 1]
            ci_time = parts[i + 2]
            ci_seg = parts[i + 3]
            break

    return {
        "weather_segments": weather,
        "ci_parity_route": {
            "dongle": ci_dongle,
            "time": ci_time,
            "seg": ci_seg,
            "description": "v0.9.6 parity reference (model_replay TEST_ROUTE seg-6)",
            "hevc_url": hevc_url,
        },
    }


def main() -> int:
    print("Scanning models/*.onnx ...")
    models_dir = REPO / "models"
    model_files = sorted(models_dir.glob("*.onnx")) if models_dir.exists() else []
    model_records = []
    for p in model_files:
        print(f"  hashing {p.name} ({p.stat().st_size / 1e6:.1f} MB) ...")
        model_records.append(_file_record(p))

    print("Scanning report/*.npz ...")
    report_dir = REPO / "report"
    npz_files = sorted(report_dir.glob("*.npz")) if report_dir.exists() else []
    npz_records = []
    for p in npz_files:
        print(f"  hashing {p.name} ({p.stat().st_size / 1e6:.1f} MB) ...")
        npz_records.append(_file_record(p))

    print("Gathering git state ...")
    git = _git_info()

    print("Gathering environment info ...")
    env = _env_info()

    print("Parsing data provenance from scripts/fetch_upgrade_data.py ...")
    provenance = _parse_data_provenance()

    manifest = {
        "generated_iso": datetime.now(tz=timezone.utc).isoformat(),
        "git": git,
        "env": env,
        "models": model_records,
        "caches": npz_records,
        "data_provenance": provenance,
    }

    out = report_dir / "MANIFEST.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"\nWrote {out.relative_to(REPO)}  ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
