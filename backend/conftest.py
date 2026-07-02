"""Shared fixtures for all backend tests."""

import io
import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Make sure the backend package is importable.
sys.path.insert(0, os.path.dirname(__file__))


@pytest.fixture
def mock_model():
    """Return a mock Keras model that behaves like the real one.

    The mock predict() always returns 0.9 (positive class = diseased).
    """
    model = MagicMock()

    # Mock output shape (batch=1, 1 output neuron)
    model.output_shape = (None, 1)
    model.input_shape = (None, 224, 224, 3)

    # predict returns a fake probability
    model.predict.return_value = np.array([[0.9]])

    # Mock get_layer for sub-models
    def fake_get_layer(name):
        layer = MagicMock()
        if name == "resnet50":
            layer.input = MagicMock()
            layer.output = MagicMock(shape=(None, 7, 7, 2048))
        elif name == "binary_classifier":
            layer.kernel = MagicMock()
            layer.bias = MagicMock()
        elif name == "global_average_pooling":
            layer.return_value = MagicMock()
        elif name == "dropout":
            layer.return_value = MagicMock()
        return layer

    model.get_layer.side_effect = fake_get_layer
    return model


@pytest.fixture
def sample_rgb_image():
    """Return a 224x224x3 uint8 RGB numpy array (fake image)."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)


@pytest.fixture
def sample_image_bytes(sample_rgb_image):
    """Return PNG bytes of the sample image."""
    from PIL import Image

    img = Image.fromarray(sample_rgb_image)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def large_rgb_image():
    """Return a 640x480x3 uint8 RGB numpy array (various size test)."""
    rng = np.random.default_rng(123)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)
