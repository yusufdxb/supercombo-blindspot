"""Unit tests for the E9 pixel-statistic interventions (pure numpy, no model)."""

import numpy as np

from src.e9_pixelstat import (_channel_stats, fda_match, hist_match,
                              moment_match)


def _fake(n=8, C=6, H=16, W=24, seed=0):
    rng = np.random.default_rng(seed)
    # CARLA-like: shifted mean, compressed std vs the "real" reference
    carla = (rng.standard_normal((n, C, H, W)) * 4 + 40).astype(np.float32)
    real = (rng.standard_normal((n + 3, C, H, W)) * 20 + 90).astype(np.float32)
    return carla, real


def test_moment_match_aligns_mean_and_std():
    carla, real = _fake()
    out = moment_match(carla, real)
    r_mu, r_sd, _, _ = _channel_stats(real)
    o_mu, o_sd, _, _ = _channel_stats(out)
    # clipping to real [0.5, 99.5] pct keeps this from being exact, but close
    assert np.abs(o_mu - r_mu).mean() < 1.0
    assert np.abs(o_sd - r_sd).mean() < 2.0


def test_hist_match_aligns_full_distribution():
    carla, real = _fake()
    out = hist_match(carla, real)
    # per channel, sorted values should track the real quantiles closely
    for ch in range(carla.shape[1]):
        oq = np.percentile(out[:, ch], [10, 50, 90])
        rq = np.percentile(real[:, ch], [10, 50, 90])
        assert np.abs(oq - rq).max() < 3.0


def test_interventions_preserve_shape_and_are_finite():
    carla, real = _fake()
    for fn in (moment_match, hist_match, fda_match):
        out = fn(carla, real)
        assert out.shape == carla.shape
        assert out.dtype == np.float32
        assert np.isfinite(out).all()


def test_interventions_change_the_input():
    carla, real = _fake()
    for fn in (moment_match, hist_match, fda_match):
        out = fn(carla, real)
        # a real intervention must move the pixels off the raw CARLA values
        assert not np.allclose(out, carla)


def test_hist_match_is_monotonic_per_frame_channel():
    carla, real = _fake()
    out = hist_match(carla, real)
    # a CDF remap preserves the rank order of pixels within each frame/channel
    for i in range(len(carla)):
        for ch in range(carla.shape[1]):
            a = carla[i, ch].ravel()
            b = out[i, ch].ravel()
            order = np.argsort(a, kind="stable")
            assert np.all(np.diff(b[order]) >= -1e-4)
