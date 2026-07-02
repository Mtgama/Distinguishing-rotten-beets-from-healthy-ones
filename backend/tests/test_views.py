"""Tests for the Django views (predict endpoint)."""

import io
from unittest.mock import patch

import pytest
from django.test import RequestFactory

from predictor.views import predict_view


@pytest.fixture
def factory():
    return RequestFactory()


class TestPredictView:
    """Tests for POST /api/predict/"""

    @patch("predictor.views.predict")
    @patch("predictor.views.compute_gradcam")
    def test_predict_returns_200_on_valid_image(
        self, mock_gradcam, mock_predict, factory, sample_image_bytes
    ):
        mock_predict.return_value = {
            "class_index": 1,
            "prediction": "افت زده",
            "confidence": 0.9,
            "prob_healthy": 0.1,
            "prob_diseased": 0.9,
        }
        # Mock Grad-CAM returns a fake heatmap (224x224 float)
        import numpy as np

        mock_gradcam.return_value = (np.zeros((224, 224), dtype=np.float32), 0.9)

        request = factory.post(
            "/api/predict/",
            data={"image": io.BytesIO(sample_image_bytes)},
            format="multipart",
        )

        response = predict_view(request)
        assert response.status_code == 200

    def test_predict_returns_400_on_no_image(self, factory):
        request = factory.post("/api/predict/")
        response = predict_view(request)
        assert response.status_code == 400
        assert b"No image" in response.content

    def test_predict_returns_405_on_get(self, factory):
        request = factory.get("/api/predict/")
        response = predict_view(request)
        # @require_http_methods returns 405 Method Not Allowed
        assert response.status_code == 405
