"""TensorFlow model loader for the beet classifier."""

import os
import io

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow import keras


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# The model file is at project root (next to make_model.py).
MODEL_PATH = os.path.join(PROJECT_ROOT, "..", "best_resnet50_beet_model-99.keras")
MODEL_PATH = os.path.abspath(MODEL_PATH)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["سالم", "آفت‌زده"]
BASE_MODEL_NAME = "resnet50"

_preprocess = keras.applications.resnet50.preprocess_input

_model = None
_grad_model = None
_head = None


def _build_grad_submodel(model):
    """Build the Grad-CAM feature extractor and classifier head references."""
    resnet = model.get_layer(BASE_MODEL_NAME)
    grad_model = keras.Model(resnet.input, resnet.output)

    classifier = model.get_layer("binary_classifier")
    head = {
        "gap": model.get_layer("global_average_pooling"),
        "dropout": model.get_layer("dropout"),
        "W": classifier.kernel,
        "b": classifier.bias,
    }
    return grad_model, head


def get_model():
    """Load the model lazily and cache it globally."""
    global _model, _grad_model, _head
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        _model = keras.models.load_model(MODEL_PATH, compile=False)
        _grad_model, _head = _build_grad_submodel(_model)
    return _model, _grad_model, _head


def preprocess_image(file_obj):
    """Read an uploaded file and convert it to a uint8 RGB numpy array."""
    if hasattr(file_obj, "read"):
        file_obj.seek(0)
        raw = file_obj.read()
    else:
        raw = file_obj

    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.uint8).numpy()
    return image


def predict(image_uint8):
    """Run inference and return class index, label, and confidence."""
    model, _, _ = get_model()

    x = np.expand_dims(image_uint8.astype("float32"), axis=0)
    x = _preprocess(x)
    prob = float(model.predict(x, verbose=0)[0][0])

    pred_class = int(prob >= 0.5)
    confidence = prob if pred_class == 1 else (1 - prob)

    return {
        "class_index": pred_class,
        "prediction": CLASS_NAMES[pred_class],
        "confidence": float(confidence),
        "prob_healthy": float(1 - prob),
        "prob_diseased": float(prob),
    }
