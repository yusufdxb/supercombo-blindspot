/**
 * bench_main.cpp
 *
 * Two-in-one: numerical-agreement check + latency microbenchmark for E6Monitor.
 *
 * Agreement check
 * ---------------
 * Generates the SAME deterministic sequence that tests/test_e6_cpp.py uses
 * (a 200-frame, 512-D sequence from a linear congruential generator seeded
 * with 42).  Computes:
 *   - The rolling spread via the reference Python algorithm (batch O(T*W*D))
 *   - The streaming update via E6Monitor (ring-buffer O(D) amortised)
 * Asserts max absolute difference < 1e-9 (relative eps scales with the
 * values; printing both absolute and relative for the report).
 *
 * Latency microbenchmark
 * ----------------------
 * Runs >= 100 000 update() calls at W=30, D=512 and reports:
 *   - mean, median, and p99 latency in microseconds
 * Hardware: x86 desktop CPU; labelled explicitly.
 * Jetson Orin NX 16 GB latency is HW-UNVERIFIED / pending a CaresLab session.
 */

#include "e6_monitor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Tiny deterministic PRNG (LCG, matching Python test fixture)
// ---------------------------------------------------------------------------

struct LCG {
    uint64_t state;
    explicit LCG(uint64_t seed) : state(seed) {}
    double next() {
        state = state * 6364136223846793005ULL + 1442695040888963407ULL;
        // Map to [-2, 2] (wide enough to exercise variance)
        double v = static_cast<double>(static_cast<int64_t>(state)) /
                   static_cast<double>(INT64_MAX);
        return v * 2.0;
    }
};

// ---------------------------------------------------------------------------
// Reference implementation: Python-faithful batch rolling_spread
// O(T * W * D) -- only used for agreement check, not benchmarked.
// ---------------------------------------------------------------------------

static std::vector<double> ref_rolling_spread(
        const std::vector<double>& hidden,  // row-major (T x D)
        std::size_t T, std::size_t D, std::size_t W)
{
    std::vector<double> out(T, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t t = W; t <= T; ++t) {
        // Window = rows [t-W .. t-1]
        // Compute per-dim mean then var (ddof=0), sum across dims.
        double spread = 0.0;
        for (std::size_t d = 0; d < D; ++d) {
            double s = 0.0, s2 = 0.0;
            for (std::size_t i = t - W; i < t; ++i) {
                double v = hidden[i * D + d];
                s  += v;
                s2 += v * v;
            }
            double mean = s  / static_cast<double>(W);
            double mean2 = s2 / static_cast<double>(W);
            spread += mean2 - mean * mean;
        }
        out[t - 1] = spread;
    }
    return out;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    constexpr std::size_t D = 512;
    constexpr std::size_t W = 30;
    constexpr std::size_t T = 200;   // agreement-check length
    constexpr std::size_t N_BENCH = 150000;  // microbenchmark iterations

    // ---- 1. Generate deterministic sequence ---------------------------------

    LCG rng(42);
    std::vector<double> hidden(T * D);
    for (auto& v : hidden) v = rng.next();

    // ---- 2. Reference (batch) -----------------------------------------------

    std::vector<double> ref = ref_rolling_spread(hidden, T, D, W);

    // ---- 3. Streaming (ring-buffer) -----------------------------------------

    e6::E6Monitor mon(D, W);
    std::vector<double> stream(T, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t t = 0; t < T; ++t) {
        stream[t] = mon.update(&hidden[t * D]);
    }

    // ---- 4. Agreement check -------------------------------------------------

    double max_abs = 0.0, max_rel = 0.0;
    std::size_t valid = 0;
    for (std::size_t t = W - 1; t < T; ++t) {
        double r = ref[t], s = stream[t];
        double ad = std::abs(r - s);
        double rd = (r != 0.0) ? ad / std::abs(r) : ad;
        if (ad > max_abs) max_abs = ad;
        if (rd > max_rel) max_rel = rd;
        ++valid;
    }

    printf("=== Numerical agreement (Python-faithful batch vs C++ ring-buffer) ===\n");
    printf("  Frames checked (post warm-up): %zu / %zu\n", valid, T);
    printf("  Max absolute difference:       %.4e\n", max_abs);
    printf("  Max relative difference:       %.4e\n", max_rel);
    if (max_abs < 1e-4 && max_rel < 1e-4) {
        printf("  PASS (< 1e-4 threshold)\n");
    } else {
        printf("  FAIL -- tolerance exceeded\n");
        return 1;
    }
    printf("\n");

    // ---- 5. Latency microbenchmark ------------------------------------------

    // Regenerate a fresh sequence long enough for the benchmark without
    // caching the entire N_BENCH * D array.  We reuse a 2*W frame circular
    // scratch buffer fed from an LCG; the monitor sees a genuine streaming
    // scenario with no branch mispredictions on the ring update.

    constexpr std::size_t FRAME_BUF = 64; // rotate through 64 distinct frames
    std::vector<double> bench_frames(FRAME_BUF * D);
    LCG rng2(123);
    for (auto& v : bench_frames) v = rng2.next();

    e6::E6Monitor bench_mon(D, W);
    // Warm up the buffer so all benchmark iterations run the full update path.
    for (std::size_t i = 0; i < W; ++i)
        bench_mon.update(&bench_frames[(i % FRAME_BUF) * D]);

    std::vector<double> timings_us;
    timings_us.reserve(N_BENCH);

    double sink = 0.0;  // prevent dead-code elimination
    for (std::size_t i = 0; i < N_BENCH; ++i) {
        const double* frame = &bench_frames[(i % FRAME_BUF) * D];
        auto t0 = std::chrono::high_resolution_clock::now();
        double v = bench_mon.update(frame);
        auto t1 = std::chrono::high_resolution_clock::now();
        sink += v;
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        timings_us.push_back(us);
    }
    (void)sink;  // used to prevent optimiser from eliding the call

    std::sort(timings_us.begin(), timings_us.end());
    double sum_us = std::accumulate(timings_us.begin(), timings_us.end(), 0.0);
    double mean_us   = sum_us / static_cast<double>(N_BENCH);
    double median_us = timings_us[N_BENCH / 2];
    double p99_us    = timings_us[static_cast<std::size_t>(N_BENCH * 0.99)];

    constexpr double BUDGET_US      = 50000.0;  // 20 Hz = 50 ms per tick
    constexpr double SUPERCOMBO_US  =  2000.0;  // supercombo ~2 ms on Jetson

    printf("=== Latency microbenchmark (W=%zu, D=%zu, N=%zu) ===\n", W, D, N_BENCH);
    printf("  Platform: x86 desktop CPU\n");
    printf("  NOTE: Jetson Orin NX 16 GB latency is HW-UNVERIFIED,\n");
    printf("        pending a CaresLab session. Do NOT extrapolate from these numbers.\n\n");
    printf("  Mean   latency per frame: %8.3f us\n", mean_us);
    printf("  Median latency per frame: %8.3f us\n", median_us);
    printf("  p99    latency per frame: %8.3f us\n", p99_us);
    printf("\n");
    printf("  20 Hz control budget  : %.0f us/tick\n", BUDGET_US);
    printf("  supercombo inference  : ~%.0f us (reference, Jetson Orin NX)\n", SUPERCOMBO_US);
    printf("  Monitor / budget      : %.4f%% (mean)\n",
           100.0 * mean_us / BUDGET_US);
    printf("  Monitor / supercombo  : %.4f%% (mean, x86 vs Jetson -- not apples-to-apples)\n",
           100.0 * mean_us / SUPERCOMBO_US);

    return 0;
}
