"""Generate the paper artifact hash manifest from public evidence files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "artifact_manifest.json"

RECOMPUTABLE = [
    "report/teardown_collected.npz",
    "report/teardown_results.md",
    "report/e4_collected.npz",
    "report/e4_results.md",
    "report/teardown_v096_collected.npz",
    "report/teardown_v096_results.md",
    "report/e4_v096_collected.npz",
    "report/e4_v096_results.md",
    "report/real_weather_collected.npz",
    "report/real_weather_results.md",
    "report/baselines_collected.npz",
    "report/baselines_results.md",
    "report/metrics_collected.npz",
    "report/metrics_results.md",
    "report/corpus_scaling_results.md",
    "report/loco_threshold_free_results.md",
    "report/e9_collected.npz",
    "report/e9_pixelstat_results.md",
    "report/e9b_collected.npz",
    "report/e9b_geomwarp_results.md",
]

INSPECT_ONLY = [
    "report/parity_results.md",
    "report/e4_ram_results.md",
    "report/e5_results.md",
    "report/e5_submodule_results.md",
    "report/e7_results.md",
    "report/e7_overlay_results.md",
    "report/e6_v096_results.md",
    "report/deployment_results.md",
    "report/figures/hero.png",
    "report/figures/e1_head_collapse.png",
    "report/figures/e2_feature_ood.png",
    "report/figures/e3_confidence.png",
    "report/figures/e4_interpolation.png",
    "report/figures/e4_ram_interpolation.png",
    "report/figures/e5_layer_localization.png",
    "report/figures/e5_submodule_localization.png",
    "report/figures/e6_detector.png",
    "report/figures/auroc_vs_alpha.png",
    "report/figures/e7_auroc_heatmap.png",
    "report/figures/e7_severity_sweep.png",
    "report/figures/e7_overlay.png",
    "report/figures/e9_pixelstat.png",
    "report/figures/e9b_geomwarp.png",
]


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> int:
    entries = []
    for level, paths in (("recomputable", RECOMPUTABLE), ("inspect_only", INSPECT_ONLY)):
        for relative in paths:
            path = ROOT / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            entries.append(
                {
                    "path": relative,
                    "sha256": digest(path),
                    "bytes": path.stat().st_size,
                    "reproduction_level": level,
                }
            )
    payload = {
        "schema_version": 1,
        "generated_by": "python -m scripts.update_artifact_manifest",
        "boundary": "inspect_only files require undistributed inputs or activation caches for recomputation",
        "artifacts": entries,
    }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} with {len(entries)} artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
