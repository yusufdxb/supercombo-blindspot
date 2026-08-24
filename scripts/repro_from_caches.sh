#!/usr/bin/env bash
# Recompute public-cache results and verify inspect-only paper artifacts.

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-.venv/bin/python}"
MODE="${1:-core}"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: $PYTHON not found. Create the pinned environment first." >&2
    exit 1
fi
if [[ "$MODE" != "core" && "$MODE" != "--extended" ]]; then
    echo "Usage: bash scripts/repro_from_caches.sh [--extended]" >&2
    exit 2
fi

TRACKED_CACHES=(
    report/teardown_collected.npz
    report/teardown_v096_collected.npz
    report/e4_collected.npz
    report/e4_v096_collected.npz
    report/real_weather_collected.npz
)
for path in "${TRACKED_CACHES[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "ERROR: tracked cache missing: $path" >&2
        exit 1
    fi
done

run_module() {
    echo "RUN $*"
    env -u PYTHONPATH "$PYTHON" -m "$@"
}

echo "=== Core fresh-clone recomputation ==="
run_module src.teardown
run_module src.teardown_v096
run_module src.e4_interp
run_module src.e4_interp_v096
run_module src.real_weather
run_module src.baselines
run_module scripts.build_metrics
run_module src.corpus_scaling
run_module src.loco_threshold_free
run_module src.conformal_results
run_module src.lead_time
run_module scripts.ablations

if [[ "$MODE" == "--extended" ]]; then
    echo "=== Extended local-cache recomputation ==="
    declare -A EXTENDED=(
        [report/e4_ram_collected.npz]=src.e4_ram
        [report/e5_submodule_collected.npz]=src.e5_submodule
        [report/e7_collected.npz]=src.e7_corruption
    )
    for cache in "${!EXTENDED[@]}"; do
        if [[ ! -f "$cache" ]]; then
            echo "ERROR: extended cache missing: $cache" >&2
            echo "Use the corresponding --collect path documented in docs/REPRODUCIBILITY.md." >&2
            exit 1
        fi
        run_module "${EXTENDED[$cache]}"
    done
fi

echo "=== Paper claim and artifact verification ==="
env -u PYTHONPATH "$PYTHON" -m scripts.verify_paper
env -u PYTHONPATH "$PYTHON" -m scripts.build_pdf --check
echo "PASS: recomputation completed for mode '$MODE'"
