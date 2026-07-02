"""Tests for Grad-CAM helpers."""

import numpy as np
import pytest

from predictor.gradcam import (
    encode_png,
    make_green_box_overlay,
    make_heatmap_overlay,
)


class TestMakeHeatmapOverlay:
    """Tests for the heatmap overlay function."""

    def test_output_same_size_as_input(self, sample_rgb_image):
        heatmap = np.random.rand(480, 640).astype(np.float32)
        result = make_heatmap_overlay(sample_rgb_image[:480, :640], heatmap)
        assert result.shape == sample_rgb_image[:480, :640].shape

    def test_output_is_uint8(self, sample_rgb_image):
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = make_heatmap_overlay(sample_rgb_image, heatmap)
        assert result.dtype == np.uint8

    def test_alpha_controls_blend(self, sample_rgb_image):
        heatmap = np.ones((224, 224), dtype=np.float32)
        overlay_0 = make_heatmap_overlay(sample_rgb_image, heatmap, alpha=0.0)
        overlay_1 = make_heatmap_overlay(sample_rgb_image, heatmap, alpha=1.0)
        # alpha=0 keeps the original image mostly; alpha=1 is pure heatmap
        assert not np.array_equal(overlay_0, overlay_1)


class TestMakeGreenBoxOverlay:
    """Tests for the green bounding box overlay."""

    def test_output_same_size_as_input(self, sample_rgb_image):
        heatmap = np.random.rand(224, 224).astype(np.float32)
        result = make_green_box_overlay(sample_rgb_image, heatmap)
        assert result.shape == sample_rgb_image.shape

    def test_draws_on_copy_not_original(self, sample_rgb_image):
        original = sample_rgb_image.copy()
        heatmap = np.random.rand(224, 224).astype(np.float32)
        _ = make_green_box_overlay(sample_rgb_image, heatmap)
        np.testing.assert_array_equal(sample_rgb_image, original)

    def test_handles_uniform_heatmap(self, sample_rgb_image):
        """When heatmap is uniform (all same value), should not crash."""
        heatmap = np.full((224, 224), 0.5, dtype=np.float32)
        result = make_green_box_overlay(sample_rgb_image, heatmap)
        assert result.shape == sample_rgb_image.shape

    def test_handles_single_hotspot(self, sample_rgb_image):
        """Heatmap with a single peak should produce a valid output."""
        heatmap = np.zeros((224, 224), dtype=np.float32)
        heatmap[100, 100] = 1.0
        result = make_green_box_overlay(sample_rgb_image, heatmap)
        assert result.shape == sample_rgb_image.shape


class TestEncodePng:
    """Tests for PNG encoding."""

    def test_returns_valid_png_bytes(self, sample_rgb_image):
        png_bytes = encode_png(sample_rgb_image)
        assert isinstance(png_bytes, bytes)
        # PNG magic bytes
        assert png_bytes[:4] == b"\x89PNG"

    def test_roundtrip_decodeable(self, sample_rgb_image):
        import cv2

        png_bytes = encode_png(sample_rgb_image)
        decoded = cv2.imdecode(
            np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_COLOR
        )
        assert decoded is not None
        # OpenCV decodes as BGR, but dimensions should match
        assert decoded.shape[:2] == sample_rgb_image.shape[:2]
