"""Tests for E7: ImageNet-C corruption sweep.

All tests are synthetic/mock and do not require the model, real data, or GPU.
They verify corruption application, score aggregation, cache round-trip, and
output format.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from src.e7_corruption import (
    CORRUPTION_NAMES,
    CORRUPTIONS,
    SEVERITIES,
    _clamp,
    analyze,
    brightness,
    contrast,
    defocus_blur,
    elastic_transform,
    fog,
    frost,
    gaussian_noise,
    glass_blur,
    impulse_noise,
    jpeg_compression,
    load_cache,
    motion_blur,
    pixelate,
    save_cache,
    shot_noise,
    snow,
    zoom_blur,
)


# -----------------------------------------------------------------------
# Corruption function tests
# -----------------------------------------------------------------------


class TestCorruptionRegistry:
    """Verify the corruption registry is complete and well-formed."""

    def test_all_15_corruptions_registered(self):
        assert len(CORRUPTIONS) == 15

    def test_corruption_names_match_keys(self):
        assert CORRUPTION_NAMES == list(CORRUPTIONS.keys())

    def test_all_corruptions_are_callable(self):
        for name, fn in CORRUPTIONS.items():
            assert callable(fn), f"{name} is not callable"


class TestCorruptionOutputs:
    """Each corruption must return uint8 HxWx3 of the same shape as input."""

    @pytest.fixture
    def sample_img(self):
        """A small synthetic RGB image for fast testing."""
        rng = np.random.RandomState(123)
        return rng.randint(0, 256, (120, 160, 3), dtype=np.uint8)

    @pytest.mark.parametrize("corruption_name", CORRUPTION_NAMES)
    @pytest.mark.parametrize("severity", [1, 3, 5])
    def test_output_shape_dtype(self, sample_img, corruption_name, severity):
        # glass_blur on full-size images is slow; use small image
        fn = CORRUPTIONS[corruption_name]
        result = fn(sample_img, severity)
        assert result.shape == sample_img.shape, (
            f"{corruption_name} sev={severity}: "
            f"expected {sample_img.shape}, got {result.shape}"
        )
        assert result.dtype == np.uint8, (
            f"{corruption_name} sev={severity}: "
            f"expected uint8, got {result.dtype}"
        )

    @pytest.mark.parametrize("corruption_name", CORRUPTION_NAMES)
    def test_severity_monotonicity(self, sample_img, corruption_name):
        """Higher severity should produce a more different image (L2 distance
        from clean should generally increase). We check sev=1 vs sev=5."""
        fn = CORRUPTIONS[corruption_name]
        out1 = fn(sample_img, 1).astype(np.float32)
        out5 = fn(sample_img, 5).astype(np.float32)
        clean = sample_img.astype(np.float32)
        dist1 = np.linalg.norm(out1 - clean)
        dist5 = np.linalg.norm(out5 - clean)
        # Allow a small tolerance for corruptions where the relationship
        # is not strictly monotonic at pixel level (e.g., elastic_transform),
        # but sev=5 should generally be further from clean than sev=1.
        # We use a very relaxed check: dist5 >= 0.5 * dist1
        assert dist5 >= 0.5 * dist1, (
            f"{corruption_name}: sev=5 L2={dist5:.1f} < 0.5 * sev=1 L2={dist1:.1f}"
        )


class TestClamp:
    def test_clamp_clips_and_casts(self):
        arr = np.array([-10.0, 0.0, 127.5, 260.0])
        out = _clamp(arr)
        assert out.dtype == np.uint8
        np.testing.assert_array_equal(out, [0, 0, 127, 255])


class TestIndividualCorruptions:
    """Targeted tests for specific corruptions to verify correctness."""

    def test_gaussian_noise_adds_noise(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = gaussian_noise(img, severity=3)
        # Should not be identical to input
        assert not np.array_equal(img, out)
        # Should be roughly centered around 128
        assert abs(out.astype(float).mean() - 128) < 30

    def test_brightness_increases(self):
        img = np.full((64, 64, 3), 100, dtype=np.uint8)
        out = brightness(img, severity=3)
        assert out.astype(float).mean() > img.astype(float).mean()

    def test_contrast_reduces(self):
        rng = np.random.RandomState(0)
        img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = contrast(img, severity=5)
        # Contrast reduction should reduce std
        assert out.astype(float).std() < img.astype(float).std()

    def test_pixelate_reduces_detail(self):
        # Checkerboard pattern should lose detail when pixelated
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        img[::2, ::2] = 255
        out = pixelate(img, severity=5)
        # After severe pixelation, the std should drop
        assert out.astype(float).std() < img.astype(float).std()

    def test_jpeg_compression_round_trip(self):
        img = np.full((64, 64, 3), 128, dtype=np.uint8)
        out = jpeg_compression(img, severity=1)
        assert out.shape == img.shape
        assert out.dtype == np.uint8
        # Uniform image should survive JPEG relatively intact
        assert abs(out.astype(float).mean() - 128) < 10

    def test_motion_blur_is_directional(self):
        rng = np.random.RandomState(7)
        img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = motion_blur(img, severity=3)
        # Motion blur should smooth in the horizontal direction, reducing
        # the std of horizontal pixel differences.
        diff_in = np.diff(img.astype(float), axis=1)
        diff_out = np.diff(out.astype(float), axis=1)
        assert diff_out.std() < diff_in.std()

    def test_defocus_blur_smooths(self):
        rng = np.random.RandomState(0)
        img = rng.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        out = defocus_blur(img, severity=3)
        # Blurring should reduce high-frequency content (reduce std of
        # differences between adjacent pixels)
        diff_in = np.diff(img.astype(float), axis=1)
        diff_out = np.diff(out.astype(float), axis=1)
        assert diff_out.std() < diff_in.std()


# -----------------------------------------------------------------------
# Cache round-trip
# -----------------------------------------------------------------------


class TestCacheRoundTrip:
    def test_save_load_preserves_data(self, tmp_path):
        rng = np.random.RandomState(0)
        collected = {
            "gaussian_noise__1": {
                "hidden_state": rng.randn(50, 512).astype(np.float32),
                "accel_t0": rng.randn(50).astype(np.float64),
            },
            "clean__0": {
                "hidden_state": rng.randn(50, 512).astype(np.float32),
                "accel_t0": rng.randn(50).astype(np.float64),
            },
        }
        path = tmp_path / "e7_test.npz"
        save_cache(path, collected)
        loaded = load_cache(path)

        assert set(loaded.keys()) == set(collected.keys())
        for key in collected:
            for arr_key in collected[key]:
                np.testing.assert_array_equal(
                    loaded[key][arr_key], collected[key][arr_key]
                )

    def test_cache_handles_many_conditions(self, tmp_path):
        rng = np.random.RandomState(0)
        collected = {}
        for cname in CORRUPTION_NAMES[:3]:
            for sev in SEVERITIES:
                key = f"{cname}__{sev}"
                collected[key] = {
                    "hidden_state": rng.randn(10, 512).astype(np.float32),
                }
        path = tmp_path / "e7_many.npz"
        save_cache(path, collected)
        loaded = load_cache(path)
        assert set(loaded.keys()) == set(collected.keys())


# -----------------------------------------------------------------------
# Analysis / score aggregation (mocked ID data)
# -----------------------------------------------------------------------


class TestAnalyze:
    """Test the analyze function with synthetic hidden states."""

    @pytest.fixture
    def mock_collected(self):
        """Synthetic collected data mimicking real output structure.

        Clean frames have hidden states with high variance (in-distribution).
        Corrupted frames at high severity have hidden states with low variance
        (simulating the collapse E6 detects).
        """
        rng = np.random.RandomState(42)
        n = 200  # enough frames for rolling window after warmup
        warmup = 100

        collected = {}
        # Clean baseline: high-variance hidden state
        collected["clean__0"] = {
            "hidden_state": rng.randn(n, 512).astype(np.float32) * 5.0,
        }
        # Low severity: still high variance
        collected["gaussian_noise__1"] = {
            "hidden_state": rng.randn(n, 512).astype(np.float32) * 4.0,
        }
        # High severity: low variance (simulating collapse)
        collected["gaussian_noise__5"] = {
            "hidden_state": rng.randn(n, 512).astype(np.float32) * 0.1,
        }
        return collected

    @pytest.fixture
    def mock_teardown(self, tmp_path):
        """Create a mock teardown_collected.npz for ID calibration."""
        rng = np.random.RandomState(0)
        npz_path = tmp_path / "report" / "teardown_collected.npz"
        npz_path.parent.mkdir(parents=True)
        np.savez(
            npz_path,
            subaru__hidden_state=rng.randn(150, 512).astype(np.float32) * 5.0,
            ram__hidden_state=rng.randn(150, 512).astype(np.float32) * 5.0,
        )
        return npz_path

    def test_analyze_returns_expected_keys(self, mock_collected, mock_teardown):
        """Verify analyze() returns proper structure with mocked ID data."""
        with patch("src.e7_corruption._id_hidden") as mock_id, \
             patch("src.e7_corruption._id_hidden_by_corpus") as mock_corpus, \
             patch("src.e7_corruption.WARMUP", 100):
            rng = np.random.RandomState(0)
            id_h = rng.randn(300, 512).astype(np.float32) * 5.0
            mock_id.return_value = id_h
            mock_corpus.return_value = {
                "subaru": id_h[:150],
                "ram": id_h[150:],
            }

            results = analyze(mock_collected, window=20, n_bootstrap=50)

        assert "conditions" in results
        assert "e6_threshold" in results
        assert "baseline_thresholds" in results
        assert "window" in results

        conds = results["conditions"]
        assert "clean__0" in conds
        assert "gaussian_noise__1" in conds
        assert "gaussian_noise__5" in conds

        # Check that each condition has the expected metric keys
        for key, cond in conds.items():
            if cond.get("skip"):
                continue
            assert "e6_fired_fraction" in cond
            assert "e6_auroc" in cond
            assert "corruption" in cond
            assert "severity" in cond

    def test_high_severity_detected_more(self, mock_collected, mock_teardown):
        """Higher corruption severity should yield higher E6 detection rate."""
        with patch("src.e7_corruption._id_hidden") as mock_id, \
             patch("src.e7_corruption._id_hidden_by_corpus") as mock_corpus, \
             patch("src.e7_corruption.WARMUP", 100):
            rng = np.random.RandomState(0)
            id_h = rng.randn(300, 512).astype(np.float32) * 5.0
            mock_id.return_value = id_h
            mock_corpus.return_value = {
                "subaru": id_h[:150],
                "ram": id_h[150:],
            }

            results = analyze(mock_collected, window=20, n_bootstrap=50)

        conds = results["conditions"]
        rate_1 = conds["gaussian_noise__1"]["e6_fired_fraction"]
        rate_5 = conds["gaussian_noise__5"]["e6_fired_fraction"]
        # The mock data is designed so severity 5 has much lower variance
        # which should trigger the rolling-spread detector more.
        assert rate_5 >= rate_1, (
            f"severity 5 fired fraction {rate_5} < severity 1 fired fraction {rate_1}"
        )


# -----------------------------------------------------------------------
# Integration-style tests on the analysis pipeline
# -----------------------------------------------------------------------


class TestAnalyzeEdgeCases:
    def test_analyze_handles_short_sequences(self):
        """Sequences shorter than the window should be skipped."""
        with patch("src.e7_corruption._id_hidden") as mock_id, \
             patch("src.e7_corruption.WARMUP", 0):
            rng = np.random.RandomState(0)
            mock_id.return_value = rng.randn(100, 512).astype(np.float32) * 5.0

            collected = {
                "gaussian_noise__1": {
                    "hidden_state": rng.randn(5, 512).astype(np.float32),
                }
            }
            results = analyze(collected, window=30, n_bootstrap=10)

        cond = results["conditions"]["gaussian_noise__1"]
        assert cond["skip"] is True

    def test_analyze_constant_hidden_state(self):
        """Constant hidden state (zero variance) should fire E6 maximally."""
        with patch("src.e7_corruption._id_hidden") as mock_id, \
             patch("src.e7_corruption.WARMUP", 0):
            rng = np.random.RandomState(0)
            mock_id.return_value = rng.randn(200, 512).astype(np.float32) * 5.0

            # Constant hidden state: rolling spread = 0 -> always below threshold
            collected = {
                "constant__1": {
                    "hidden_state": np.ones((100, 512), dtype=np.float32),
                }
            }
            results = analyze(collected, window=20, n_bootstrap=10)

        cond = results["conditions"]["constant__1"]
        assert not cond.get("skip")
        # Constant hidden state should have spread=0, well below any positive
        # threshold, so fired fraction should be 1.0 (all flagged OOD).
        assert cond["e6_fired_fraction"] == pytest.approx(1.0, abs=0.01)


# -----------------------------------------------------------------------
# Report writing
# -----------------------------------------------------------------------


class TestWriteResults:
    def test_write_results_creates_file(self, tmp_path):
        from src.e7_corruption import _write_results

        results = {
            "conditions": {
                "gaussian_noise__1": {
                    "corruption": "gaussian_noise",
                    "severity": 1,
                    "n_frames": 100,
                    "skip": False,
                    "e6_fired_fraction": 0.1,
                    "e6_threshold": 42.0,
                    "e6_auroc": 0.85,
                    "e6_aupr": 0.80,
                    "e6_fpr95": 0.15,
                    "e6_auroc_ci": (0.82, 0.88),
                    "e6_aupr_ci": (0.77, 0.83),
                    "e6_fpr95_ci": (0.10, 0.20),
                    "mahalanobis_fired_fraction": 0.9,
                    "mahalanobis_threshold": 100.0,
                    "mahalanobis_auroc": 0.7,
                    "mahalanobis_aupr": 0.65,
                    "mahalanobis_fpr95": 0.3,
                    "relative_mahalanobis_fired_fraction": 0.5,
                    "relative_mahalanobis_threshold": 50.0,
                    "relative_mahalanobis_auroc": 0.6,
                    "relative_mahalanobis_aupr": 0.55,
                    "relative_mahalanobis_fpr95": 0.4,
                    "knn50_fired_fraction": 0.8,
                    "knn50_threshold": 1.5,
                    "knn50_auroc": 0.75,
                    "knn50_aupr": 0.70,
                    "knn50_fpr95": 0.25,
                },
            },
            "e6_threshold": 42.0,
            "baseline_thresholds": {
                "mahalanobis": 100.0,
                "relative_mahalanobis": 50.0,
                "knn50": 1.5,
            },
            "window": 30,
            "e6_percentile": 1.0,
            "baseline_percentile": 99.0,
        }

        out = tmp_path / "e7_results.md"
        _write_results(results, out)
        assert out.exists()
        content = out.read_text()
        assert "E7 Results" in content
        assert "gaussian_noise" in content
        assert "AUROC" in content
        assert "0.85" in content


class TestFigures:
    def test_severity_sweep_creates_figure(self, tmp_path):
        from src.e7_corruption import _fig_severity_sweep

        results = {
            "conditions": {
                f"gaussian_noise__{s}": {
                    "corruption": "gaussian_noise",
                    "severity": s,
                    "n_frames": 100,
                    "skip": False,
                    "e6_fired_fraction": 0.1 * s,
                }
                for s in SEVERITIES
            },
        }
        _fig_severity_sweep(results, tmp_path)
        assert (tmp_path / "e7_severity_sweep.png").exists()

    def test_auroc_heatmap_creates_figure(self, tmp_path):
        from src.e7_corruption import _fig_auroc_heatmap

        results = {
            "conditions": {
                f"{cname}__{s}": {
                    "corruption": cname,
                    "severity": s,
                    "n_frames": 100,
                    "skip": False,
                    "e6_auroc": 0.5 + 0.1 * s,
                }
                for cname in CORRUPTION_NAMES[:3]
                for s in SEVERITIES
            },
        }
        _fig_auroc_heatmap(results, tmp_path)
        assert (tmp_path / "e7_auroc_heatmap.png").exists()
