"""Grad-CAM helpers for the beet classifier."""

import io
import os

import cv2
import numpy as np
import tensorflow as tf

from .model import BASE_MODEL_NAME, IMG_SIZE, get_model


def _preprocess_input(image):
    from tensorflow import keras
    return keras.applications.resnet50.preprocess_input(image)


def compute_gradcam(image_uint8):
    """Compute a Grad-CAM heatmap for the given RGB uint8 image.

    Returns the heatmap (resized to original image size) and the positive-class
    probability.
    """
    grad_model, head = get_model()[1:]

    img_resized = cv2.resize(image_uint8, IMG_SIZE)
    x = np.expand_dims(img_resized.astype("float32"), axis=0)
    x = _preprocess_input(x)

    with tf.GradientTape() as tape:
        feat = grad_model(x, training=False)
        pooled = head["gap"](feat)
        dropped = head["dropout"](pooled, training=False)
        logit = tf.matmul(dropped, head["W"]) + head["b"]
        score = logit[0, 0]

    grads = tape.gradient(score, feat)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * feat[0], axis=-1).numpy()

    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    heatmap = cv2.resize(cam, (image_uint8.shape[1], image_uint8.shape[0]))
    return heatmap, _prob(image_uint8)


def _prob(image_uint8):
    from .model import get_model as _get_model
    model = _get_model()[0]
    x = np.expand_dims(cv2.resize(image_uint8, IMG_SIZE).astype("float32"), axis=0)
    x = _preprocess_input(x)
    return float(model.predict(x, verbose=0)[0][0])


def make_heatmap_overlay(image_uint8, heatmap, alpha=0.4):
    """Blend the jet-colored heatmap over the original image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(heatmap_color, alpha, image_uint8, 1 - alpha, 0)
    return overlay


def make_green_box_overlay(image_uint8, heatmap, threshold_frac=0.5):
    """Draw a green bounding box around the hottest region."""
    GREEN = (0, 255, 0)
    out = image_uint8.copy()
    h, w = out.shape[:2]

    binary = (heatmap >= (heatmap.max() * threshold_frac)).astype("uint8") * 255
    if binary.sum() == 0:
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x1, y1 = max(idx[1] - 10, 0), max(idx[0] - 10, 0)
        x2, y2 = min(idx[1] + 10, w), min(idx[0] + 10, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 3)
        return out

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out

    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)
    pad = int(0.04 * max(w, h))
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + bw + pad, w), min(y + bh + pad, h)

    cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, thickness=max(3, h // 120))
    cv2.putText(
        out,
        "focus region",
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        GREEN,
        2,
        cv2.LINE_AA,
    )
    return out


def encode_png(image_rgb):
    """Encode an RGB image to PNG bytes."""
    _, buffer = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    return io.BytesIO(buffer).getvalue()
