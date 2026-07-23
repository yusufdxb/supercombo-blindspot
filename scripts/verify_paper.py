"""Verify paper claims, evidence hashes, and release-language boundaries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "paper" / "manuscript.md"
MANIFEST = ROOT / "paper" / "artifact_manifest.json"


@dataclass(frozen=True)
class Claim:
    identifier: str
    evidence: str
    evidence_markers: tuple[str, ...]
    manuscript_markers: tuple[str, ...]


CLAIMS = (
    Claim("C1", "report/parity_results.md", ("Frames compared (post warm-up trim): **1159**", "median `|delta|`: **0.0409 m/s^2**"), ("all 1,159 evaluated real-footage frames", "0.0409 m/s^2")),
    Claim("C2", "report/teardown_results.md", ("| desired_curv | 0.1318 | 0.0002 | 0.0018", "| road_edges | 280.3987 | 2.1272 | 0.0076"), ("8 of 10 tracked output readouts", "less than 1%")),
    Claim("C3", "report/teardown_results.md", ("**0.00001x**",), ("about 1e-5 of its real spread",)),
    Claim("C4", "report/teardown_results.md", ("CARLA above real p95", "| desired_curv | 0.2%"), ("0 of 219 CARLA analysis frames",)),
    Claim("C5", "report/e4_results.md", ("transition width 0.015",), ("0.015-wide cliff",)),
    Claim("C6", "report/e4_ram_results.md", ("transition width 0.274", "| RAM | 0.666 | 0.940 | 0.274"), ("0.274-wide gradient",)),
    Claim("C7", "report/e5_submodule_results.md", ("`summarizer_div` | 0.900", "`action_block_body` | 0.500"), ("downstream of the vision encoder", "mu-versus-sigma")),
    Claim("C8", "report/corpus_scaling_results.md", ("**LOCO mean FPR: 2.41%**", "LOCO max 6.90%"), ("2.41% mean leave-one-corpus-out", "worst fold 6.90%")),
    Claim("C9", "report/e4_ram_results.md", ("| Subaru | 0.784 | 0.799 | 0.015 | 0.550 | 0.234",), ("positive only on the Subaru overlay",)),
    Claim("C10", "report/metrics_results.md", ("0.996 [0.992, 1.000]",), ("AUROC 0.996 [0.992, 1.000]",)),
    Claim("C11", "report/loco_threshold_free_results.md", ("| e6 | 0.00%", "| knn50 | 60.82%", "| mahalanobis | 95.14%"), ("secondary collapse-aware analysis", "60.82%, 95.14%, and 99.69%")),
    Claim("C12", "report/e7_overlay_results.md", ("corruption cells evaluated: **75**", "output-collapsed cells (>= 5/10 heads): **0**"), ("No ImageNet-C corruption reproduces",)),
    Claim("C13", "report/e6_v096_results.md", ("**LOCO mean FPR: 0.3328 (33.28%)**",), ("v0.9.7 monitor does not transfer",)),
    Claim("C14", "report/real_weather_results.md", ("EV6 night + headlight glare | 0/10", "Bronco night + tail-light/sign glare | 0/10"), ("Real night and glare sequences do not induce",)),
    Claim("C15", "report/corpus_scaling_results.md", ("| daytime_control | 0.6034 |",), ("60.34% of analyzed frames",)),
    Claim("C16", "report/deployment_results.md", ("Mean   latency per frame:    0.405 us", "Max absolute difference:       3.4106e-13"), ("about 0.4", "target embedded-platform")),
    Claim("C17", "report/e9_pixelstat_results.md", ("| CARLA + histogram match | 2/10 |", "| CARLA (raw) | 8/10 |"), ("1, 2 and 3 of 10 under the moment, histogram and Fourier interventions", "was therefore sufficient to lift the recurrent freeze")),
    Claim("C18", "report/e9b_geomwarp_results.md", ("| A: real zero-warp vs real calibrated | 0/10 ", "| B: CARLA vs real zero-warp (identical warp) | 5/10 "), ("0 of 10 readouts fall below either the 1% or the 10% threshold", "not sufficient by itself to explain the freeze")),
)

BANNED = (
    "parity-exact",
    "first second-order",
    "the model exposes no signal whatsoever",
    "monitor firing on 58%",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_claims(manuscript: str) -> list[str]:
    failures: list[str] = []
    for claim in CLAIMS:
        path = ROOT / claim.evidence
        if not path.is_file():
            failures.append(f"{claim.identifier}: missing {claim.evidence}")
            continue
        evidence = path.read_text(encoding="utf-8")
        for marker in claim.evidence_markers:
            if marker not in evidence:
                failures.append(f"{claim.identifier}: evidence marker missing: {marker}")
        for marker in claim.manuscript_markers:
            if marker not in manuscript:
                failures.append(f"{claim.identifier}: manuscript marker missing: {marker}")
    return failures


def verify_manifest() -> list[str]:
    failures: list[str] = []
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in payload["artifacts"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            failures.append(f"manifest missing file: {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            failures.append(f"manifest hash mismatch: {entry['path']}")
    return failures


def main() -> int:
    manuscript = MANUSCRIPT.read_text(encoding="utf-8")
    failures = verify_claims(manuscript)
    failures.extend(verify_manifest())
    lowered = manuscript.lower()
    failures.extend(f"banned manuscript wording: {token}" for token in BANNED if token in lowered)
    if chr(0x2014) in manuscript:
        failures.append("manuscript contains U+2014 em dash")
    if failures:
        print("paper verification: FAIL")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print(f"paper verification: PASS ({len(CLAIMS)} claims, artifact hashes intact)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
