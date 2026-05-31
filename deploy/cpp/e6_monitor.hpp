/**
 * e6_monitor.hpp
 *
 * Portable, header-only rolling-spread monitor for the E6 OOD detector.
 *
 * Computes, per frame, the trace of the rolling covariance of a W-frame window
 * of a D-dimensional recurrent state vector.  Concretely:
 *
 *   spread(t) = sum_{d=0..D-1}  Var(x[t-W+1..t], dim=d)
 *
 * where Var uses the SAME population-variance convention as NumPy's np.var
 * (ddof=0, i.e. divide by N=W, not N-1).  This exactly matches the Python
 * reference in src/e6_detector.py::rolling_spread():
 *
 *   np.var(hidden[t-window:t], axis=0).sum()   # ddof=0 by default
 *
 * The ring-buffer streaming update is O(D) amortized per frame:
 *   - maintain a running sum  S[d]   = sum_i x[i,d]
 *   - maintain a running sum  S2[d]  = sum_i x[i,d]^2
 *   - variance of d-th dim  = S2[d]/W  -  (S[d]/W)^2
 *   - rolling spread        = sum_d variance[d]
 *
 * When the buffer has fewer than W frames, spread() returns NaN (warm-up),
 * matching the Python semantics exactly.
 *
 * Reference threshold from E6 calibration (1st percentile of real-driving
 * spread on subaru+ram corpora, W=30, D=512):
 *   E6_THRESHOLD = 0.078873
 *
 * Intended use on-device (Jetson Orin NX):
 *   modeld publishes hidden_state -> ROS2 topic -> phm_detectors node
 *   (RecurrentTemporalSpreadAdapter) which wraps this exact math in Python
 *   today.  For a latency-critical C++ subscriber, construct one E6Monitor and
 *   call update() once per supercombo inference frame (20 Hz).  No heap
 *   allocation occurs after construction.
 */

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace e6 {

/**
 * Rolling-spread monitor.
 *
 * Template parameter D: state dimension (compile-time constant optional).
 * All arithmetic is double-precision to match the Python reference.
 */
class E6Monitor {
public:
    /**
     * Construct a monitor.
     *
     * @param dim    Dimension of the state vector (D=512 for supercombo).
     * @param window Rolling covariance window length W (default 30, matching E6).
     *
     * Throws std::invalid_argument if window < 2 or dim == 0.
     */
    explicit E6Monitor(std::size_t dim, std::size_t window = 30)
        : dim_(dim), window_(window), head_(0), count_(0),
          sum_(dim, 0.0), sum2_(dim, 0.0),
          buf_(dim * window, 0.0)
    {
        if (dim == 0)
            throw std::invalid_argument("dim must be > 0");
        if (window < 2)
            throw std::invalid_argument("window must be >= 2");
    }

    /** Reset the monitor to its initial (warm-up) state. */
    void reset() {
        head_ = 0;
        count_ = 0;
        std::fill(sum_.begin(), sum_.end(), 0.0);
        std::fill(sum2_.begin(), sum2_.end(), 0.0);
        std::fill(buf_.begin(), buf_.end(), 0.0);
    }

    /**
     * Ingest one new state vector and return the rolling spread.
     *
     * @param x  Pointer to the first element of a D-dimensional vector.
     *           Values should be raw (not normalised) -- matches openpilot's
     *           unnormalised hidden_state.
     *
     * @return  Rolling spread (double) if the window is full; NaN during warm-up.
     *          Complexity: O(D) per call.
     */
    double update(const double* x) {
        // 1. Evict the oldest frame if the ring is full.
        if (count_ == window_) {
            const double* old_frame = &buf_[head_ * dim_];
            for (std::size_t d = 0; d < dim_; ++d) {
                sum_[d]  -= old_frame[d];
                sum2_[d] -= old_frame[d] * old_frame[d];
            }
        } else {
            ++count_;
        }

        // 2. Write new frame at head_ and advance.
        double* slot = &buf_[head_ * dim_];
        for (std::size_t d = 0; d < dim_; ++d) {
            slot[d]   = x[d];
            sum_[d]  += x[d];
            sum2_[d] += x[d] * x[d];
        }
        head_ = (head_ + 1) % window_;

        // 3. Warm-up: fewer than window frames seen.
        if (count_ < window_)
            return std::numeric_limits<double>::quiet_NaN();

        // 4. Compute trace of population covariance (ddof=0).
        //    Var_d = E[x_d^2] - E[x_d]^2 = sum2_d/W - (sum_d/W)^2
        double spread = 0.0;
        const double inv_w = 1.0 / static_cast<double>(window_);
        for (std::size_t d = 0; d < dim_; ++d) {
            const double mean_d  = sum_[d]  * inv_w;
            const double mean2_d = sum2_[d] * inv_w;
            spread += mean2_d - mean_d * mean_d;
        }
        return spread;
    }

    /** Convenience overload accepting a std::vector<double>. */
    double update(const std::vector<double>& x) {
        if (x.size() != dim_)
            throw std::invalid_argument("x.size() != dim");
        return update(x.data());
    }

    /** Number of frames seen so far (saturates at window_). */
    std::size_t count() const { return count_; }

    /** True once the window is full and spread() is meaningful. */
    bool ready() const { return count_ == window_; }

    std::size_t dim()    const { return dim_; }
    std::size_t window() const { return window_; }

private:
    std::size_t dim_;
    std::size_t window_;
    std::size_t head_;    // next write position in the ring
    std::size_t count_;   // frames in buffer (capped at window_)

    std::vector<double> sum_;   // sum_[d] = sum of x[d] over current window
    std::vector<double> sum2_;  // sum2_[d] = sum of x[d]^2 over current window
    std::vector<double> buf_;   // ring buffer, shape (window_, dim_) row-major
};

/** Reference E6 threshold (1st percentile, W=30, D=512, subaru+ram corpus). */
constexpr double E6_THRESHOLD = 0.078873;

} // namespace e6
