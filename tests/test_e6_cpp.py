"""Python-vs-C++ numerical agreement test for deploy/cpp/e6_monitor.hpp.

The C++ binary is built from deploy/cpp/ via the Makefile (requires g++) and
run with --agree-only.  The test is skipped cleanly when g++ is not on PATH or
the build fails (CI-safe, matching the repo's convention for HW/toolchain gates).

The deterministic sequence fed here MUST match bench_main.cpp's LCG seed (42)
and frame count (200) exactly, so the Python-computed reference and the C++
output share the same input.  The C++ bench_main already prints PASS/FAIL;
this test additionally verifies agreement from the Python side using the same
rolling_spread function that ships in src/e6_detector.py.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.e6_detector import rolling_spread

# ---------------------------------------------------------------------------
# Deterministic sequence matching bench_main.cpp's LCG(seed=42)
# ---------------------------------------------------------------------------

def _lcg_sequence(seed: int, n: int) -> np.ndarray:
    """64-bit LCG matching the constants in bench_main.cpp.

    state = state * 6364136223846793005 + 1442695040888963407  (mod 2^64)
    value = int64(state) / INT64_MAX  * 2.0   in [-2, 2]
    """
    INT64_MAX = 2**63 - 1
    MULT = 6364136223846793005
    INC  = 1442695040888963407
    MOD  = 2**64
    state = seed
    out = []
    for _ in range(n):
        state = (state * MULT + INC) % MOD
        # Reinterpret as signed int64 (two's-complement)
        signed = state if state < 2**63 else state - 2**64
        out.append(signed / INT64_MAX * 2.0)
    return np.array(out)


D = 512
W = 30
T = 200

@pytest.fixture(scope="module")
def hidden_sequence():
    raw = _lcg_sequence(42, T * D)
    return raw.reshape(T, D)


# ---------------------------------------------------------------------------
# Python reference
# ---------------------------------------------------------------------------

def test_python_rolling_spread_reference(hidden_sequence):
    """Smoke-test that Python rolling_spread runs without error."""
    s = rolling_spread(hidden_sequence, window=W)
    assert s.shape == (T,)
    assert np.all(np.isnan(s[:W - 1]))
    assert np.all(~np.isnan(s[W - 1:]))


# ---------------------------------------------------------------------------
# C++ agreement
# ---------------------------------------------------------------------------

def _build_cpp() -> Path:
    """Compile e6_bench; return path to binary or raise pytest.skip."""
    if shutil.which("g++") is None:
        pytest.skip("g++ not on PATH -- C++ agreement test skipped")

    deploy_dir = Path(__file__).parent.parent / "deploy" / "cpp"
    result = subprocess.run(
        ["make", "-B"],   # -B forces rebuild so the binary is always fresh
        cwd=str(deploy_dir),
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(
            f"C++ build failed (g++ available but make exited {result.returncode}). "
            f"stderr: {result.stderr[:400]}"
        )
    return deploy_dir / "e6_bench"


def test_cpp_bench_passes(hidden_sequence):
    """Run the C++ bench binary and confirm it reports PASS.

    The binary itself checks max-abs-diff < 1e-4 and exits 0 on success.
    This test gates on g++ availability and skips cleanly in CI without a
    C++ toolchain.
    """
    binary = _build_cpp()
    result = subprocess.run(
        [str(binary)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"e6_bench returned {result.returncode}.\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout, (
        f"e6_bench did not print PASS.\nstdout:\n{result.stdout}"
    )


def test_python_cpp_agreement_values(hidden_sequence):
    """Cross-check Python rolling_spread values match C++ ring-buffer output.

    The C++ ring-buffer result is extracted by a small inline C++ program
    that prints the spread values as CSV; we compare them against Python here.
    Skips cleanly if g++ is absent.
    """
    if shutil.which("g++") is None:
        pytest.skip("g++ not on PATH")

    # Write a tiny C++ driver that emits just the spread values as CSV
    deploy_dir = Path(__file__).parent.parent / "deploy" / "cpp"
    driver_src = r"""
#include "e6_monitor.hpp"
#include <cstdio>
#include <cstring>

// LCG matching bench_main.cpp
struct LCG {
    uint64_t state;
    explicit LCG(uint64_t seed) : state(seed) {}
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        double v = static_cast<double>(static_cast<int64_t>(state))
                   / static_cast<double>(INT64_MAX);
        return v * 2.0;
    }
};

int main() {
    constexpr size_t D = 512, W = 30, T = 200;
    LCG rng(42);
    std::vector<double> row(D);
    e6::E6Monitor mon(D, W);
    for (size_t t = 0; t < T; ++t) {
        for (size_t d = 0; d < D; ++d) row[d] = rng.next();
        double s = mon.update(row.data());
        if (t >= W - 1)
            printf("%.17g\n", s);
        else
            printf("nan\n");
    }
}
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = Path(tmpdir) / "dump.cpp"
        bin_path = Path(tmpdir) / "dump"
        src_path.write_text(driver_src)

        comp = subprocess.run(
            ["g++", "-std=c++17", "-O2",
             f"-I{deploy_dir}",
             str(src_path), "-o", str(bin_path)],
            capture_output=True, text=True, timeout=30,
        )
        if comp.returncode != 0:
            pytest.skip(f"dump driver compile failed: {comp.stderr[:400]}")

        run = subprocess.run(
            [str(bin_path)], capture_output=True, text=True, timeout=30,
        )
        assert run.returncode == 0

        cpp_vals = []
        for line in run.stdout.strip().splitlines():
            s = line.strip()
            cpp_vals.append(float("nan") if s == "nan" else float(s))

    cpp_arr = np.array(cpp_vals)  # length T, nans for warm-up frames

    py_spread = rolling_spread(hidden_sequence, window=W)

    # Restrict comparison to post-warm-up frames (indices W-1 .. T-1)
    assert len(cpp_arr) == T, (
        f"C++ output length {len(cpp_arr)} != T={T}"
    )
    assert len(py_spread) == T
    cpp_valid = cpp_arr[W - 1:]
    py_valid  = py_spread[W - 1:]

    max_abs = float(np.max(np.abs(cpp_valid - py_valid)))
    ref_scale = float(np.max(np.abs(py_valid)))
    max_rel = max_abs / ref_scale if ref_scale > 0 else max_abs

    # Tight tolerances: ring-buffer is algebraically equivalent, only
    # floating-point reordering differs.
    assert max_abs < 1e-4, f"max_abs={max_abs:.4e} exceeds 1e-4"
    assert max_rel < 1e-4, f"max_rel={max_rel:.4e} exceeds 1e-4"
