"""Tests for the model loader and predict helper."""

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from predictor.model import CLASS_NAMES, IMG_SIZE, preprocess_image, predict


class TestPreprocessImage:
    """Tests for preprocess_image()."""

    def test_converts_bytes_to_uint8_array(self, sample_image_bytes):
        result = preprocess_image(sample_image_bytes)
        assert result.dtype == np.uint8
        assert result.ndim == 3
        assert result.shape[2] == 3  # RGB channels

    def test_resizes_to_model_input_size(self, large_rgb_image):
        """Image of any size should be resized to IMG_SIZE."""
        result = preprocess_image(large_rgb_image.tobytes())
        assert result.shape[0] == IMG_SIZE[1]  # height
        assert result.shape[1] == IMG_SIZE[0]  # width

    def test_handles_file_like_object(self, sample_image_bytes):
        fobj = io.BytesIO(sample_image_bytes)
        result = preprocess_image(fobj)
        assert result.dtype == np.uint8


class TestPredict:
    """Tests for predict() function with mocked model."""

    @patch("predictor.model.get_model")
    def test_returns_diseased_class(self, mock_get_model, sample_rgb_image):
        """When prob >= 0.5, should predict diseased."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.9]])
        mock_get_model.return_value = (model, MagicMock(), {})

        result = predict(sample_rgb_image)

        assert result["class_index"] == 1
        assert result["prediction"] == CLASS_NAMES[1]
        assert result["confidence"] == pytest.approx(0.9)
        assert result["prob_diseased"] == pytest.approx(0.9)
        assert result["prob_healthy"] == pytest.approx(0.1)

    @patch("predictor.model.get_model")
    def test_returns_healthy_class(self, mock_get_model, sample_rgb_image):
        """When prob < 0.5, should predict healthy."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.2]])
        mock_get_model.return_value = (model, MagicMock(), {})

        result = predict(sample_rgb_image)

        assert result["class_index"] == 0
        assert result["prediction"] == CLASS_NAMES[0]
        assert result["confidence"] == pytest.approx(0.8)

    @patch("predictor.model.get_model")
    def test_returns_all_expected_keys(self, mock_get_model, sample_rgb_image):
        model = MagicMock()
        model.predict.return_value = np.array([[0.7]])
        mock_get_model.return_value = (model, MagicMock(), {})

        result = predict(sample_rgb_image)

        expected_keys = {
            "class_index",
            "prediction",
            "confidence",
            "prob_healthy",
            "prob_diseased",
        }
        assert set(result.keys()) == expected_keys

    @patch("predictor.model.get_model")
    def test_boundary_probability(self, mock_get_model, sample_rgb_image):
        """Exactly 0.5 should map to class 0 (healthy) since >= 0.5 is diseased."""
        model = MagicMock()
        model.predict.return_value = np.array([[0.5]])
        mock_get_model.return_value = (model, MagicMock(), {})

        result = predict(sample_rgb_image)
        assert result["class_index"] == 1  # >= 0.5 -> diseased
