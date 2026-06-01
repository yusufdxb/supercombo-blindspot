#!/usr/bin/env bash
# One-command cache-only reproduction for phantom-braking.
#
# Regenerates all tables, figures, and result markdown files from the
# committed .npz caches in report/. No ONNX model, no CARLA, no GPU required.
#
# Usage:
#   cd /path/to/phantom-braking
#   bash scripts/repro_from_caches.sh
#
# Output files regenerated:
#   report/teardown_results.md          report/figures/e1_*.png e2_*.png e3_*.png
#   report/teardown_v096_results.md     report/figures/teardown_v096_*.png
#   report/e4_results.md                report/figures/e4_interpolation.png
#   report/e4_v096_results.md           report/figures/e4_v096_interpolation.png
#   report/e4_ram_results.md            report/figures/e4_ram_interpolation.png
#   report/e5_submodule_results.md      report/figures/e5_submodule.png
#   report/e5_submodule_enumeration.md
#   report/e7_results.md                report/figures/e7_*.png
#   report/baselines_results.md         report/baselines_collected.npz  (recomputed)
#   report/metrics_results.md           report/metrics_collected.npz    (recomputed)
#   report/figures/roc_curves.png  pr_curves.png  auroc_vs_alpha.png
#   report/conformal_results.md
#   report/lead_time_results.md         report/figures/lead_time.png
#   report/ablations_results.md         report/ablations_collected.npz  (recomputed)
#
# RAW-DATA PATHS (not run here — need CUDA GPU + raw segments + model):
#   src.teardown        --collect    (requires supercombo.onnx + data/)
#   src.teardown_v096   --collect    (requires supercombo_v096.onnx + data/)
#   src.e4_interp       --collect    (requires supercombo.onnx + data/)
#   src.e4_interp_v096  --collect    (requires supercombo_v096.onnx + data/)
#   src.e4_ram          --collect    (requires supercombo.onnx + data/)
#   src.e5_submodule    --collect    (requires supercombo_submodule_probed.onnx + data/)
#   src.e7_corruption   --collect    (requires supercombo.onnx + data/)
#   src.real_weather    --collect    (requires supercombo.onnx + real-weather data/)
#   scripts/fetch_upgrade_data.py         (fetches from comma CI / network)
#   scripts/fetch_upgrade_data.py --weather (fetches from comma CI / network)

set -euo pipefail

VENV_PYTHON=".venv/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: $VENV_PYTHON not found. Run from the repo root." >&2
    exit 1
fi

# Verify all required caches are present before starting
REQUIRED_CACHES=(
    "report/teardown_collected.npz"
    "report/teardown_v096_collected.npz"
    "report/e4_collected.npz"
    "report/e4_v096_collected.npz"
    "report/e4_ram_collected.npz"
    "report/e5_submodule_collected.npz"
    "report/e7_collected.npz"
    "report/real_weather_collected.npz"
    "report/ablations_collected.npz"
)
echo "=== Verifying required caches ==="
for f in "${REQUIRED_CACHES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "MISSING: $f" >&2
        echo "  -> Run the corresponding --collect path to generate this cache." >&2
        exit 1
    fi
    echo "  OK: $f"
done
echo ""

# ---------------------------------------------------------------------------
# E1/E2/E3 teardown (v0.9.7)
# ---------------------------------------------------------------------------
echo "=== [1/10] src.teardown (E1/E2/E3, v0.9.7, from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.teardown
echo ""

# ---------------------------------------------------------------------------
# E1/E2/E3 teardown (v0.9.6)
# ---------------------------------------------------------------------------
echo "=== [2/10] src.teardown_v096 (E1/E2/E3, v0.9.6, from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.teardown_v096
echo ""

# ---------------------------------------------------------------------------
# E4 interpolation sweep (v0.9.7 Subaru source)
# ---------------------------------------------------------------------------
echo "=== [3/10] src.e4_interp (v0.9.7 Subaru, from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.e4_interp
echo ""

# ---------------------------------------------------------------------------
# E4 interpolation sweep (v0.9.6 Subaru source)
# ---------------------------------------------------------------------------
echo "=== [4/10] src.e4_interp_v096 (v0.9.6 Subaru, from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.e4_interp_v096
echo ""

# ---------------------------------------------------------------------------
# E4-RAM interpolation sweep (v0.9.7 RAM source)
# ---------------------------------------------------------------------------
echo "=== [5/10] src.e4_ram (RAM source, from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.e4_ram
echo ""

# ---------------------------------------------------------------------------
# E5 submodule enumeration
# ---------------------------------------------------------------------------
echo "=== [6/10] src.e5_submodule (from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.e5_submodule
echo ""

# ---------------------------------------------------------------------------
# E7 ImageNet-C corruption sweep
# ---------------------------------------------------------------------------
echo "=== [7/10] src.e7_corruption (from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.e7_corruption
echo ""

# ---------------------------------------------------------------------------
# Real-weather OOD axis
# ---------------------------------------------------------------------------
echo "=== [8/10] src.real_weather (from cache) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.real_weather
echo ""

# ---------------------------------------------------------------------------
# Baselines + metrics (pure-numpy/sklearn; reads from teardown + e4 caches)
# ---------------------------------------------------------------------------
echo "=== [9/10] src.baselines + scripts/build_metrics.py (from caches) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.baselines
env -u PYTHONPATH "$VENV_PYTHON" -m scripts.build_metrics
echo ""

# ---------------------------------------------------------------------------
# Conformal detector + lead-time analysis (reads baselines/metrics caches)
# ---------------------------------------------------------------------------
echo "=== [10/10] src.conformal_results + src.lead_time (from caches) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m src.conformal_results
env -u PYTHONPATH "$VENV_PYTHON" -m src.lead_time
echo ""

# ---------------------------------------------------------------------------
# Ablations (reads teardown + e4 caches via scripts/ablations.py)
# ---------------------------------------------------------------------------
echo "=== [11/11] scripts/ablations.py (from caches) ==="
env -u PYTHONPATH "$VENV_PYTHON" -m scripts.ablations
echo ""

echo "=== repro_from_caches.sh COMPLETE ==="
echo "Result files regenerated under report/ and report/figures/"
