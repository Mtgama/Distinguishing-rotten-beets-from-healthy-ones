# =========================
# Grad-CAM + Gradio for the beet ResNet50 model
# =========================
#
# What this does:
#   - Loads the trained ResNet50 transfer-learning model (from make_model.py).
#   - For any uploaded image, predicts the class and probability.
#   - Computes a Grad-CAM heatmap (the regions the model looked at).
#   - Produces two outputs:
#       1. Heatmap overlay -> the image blended with a colored heatmap.
#       2. Green box      -> a green bounding box around the hottest region,
#                             drawn on the *original* (unmodified) image.
#
# Run:
#   pyenv activate data-science
#   python grad_cam_app.py
#
# =========================

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TF logs

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras

import gradio as gr


# =========================
# Config
# =========================

# Resolve relative to this file so it works regardless of the working
# directory (e.g. when imported by the Django backend from another folder).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_resnet50_beet_model.keras")

IMG_SIZE = (224, 224)

# The model was trained with label_mode="binary".
# image_dataset_from_directory assigns label 0 to the FIRST folder
# alphabetically, label 1 to the second. Folder order here is:
#   healthy   -> label 0  ->  "سالم"
#   infected  -> label 1  ->  "افت زده"
CLASS_NAMES = ["سالم", "افت زده"]

# Sub-model name inside the saved model (the ResNet50 backbone).
BASE_MODEL_NAME = "resnet50"

# ResNet50's own preprocessing (the saved model applies this as a layer too).
preprocess_input = keras.applications.resnet50.preprocess_input

GREEN = (0, 255, 0)  # RGB green, used for the focus box


# =========================
# Model loading
# =========================

def load_model():
    """Load the saved model and build a Grad-CAM sub-model.

    The saved model is: Input -> data_augmentation -> preprocess_input ->
    resnet50 (7x7x2048) -> GAP -> Dropout -> Dense(sigmoid).

    For Grad-CAM we need the final 7x7x2048 feature map WITH a gradient path
    to the prediction. In Keras 3, reaching into the nested `resnet50`'s
    internals from the *full* model's inputs breaks the graph, so instead we
    build a sub-model from `resnet.input` straight to `resnet.output`. That
    output IS the 7x7x2048 feature map (post conv5_block3 BN+ReLU) — exactly
    the last conv-stage activation Grad-CAM targets.

    We then apply the trained classifier head (GAP + weights) by hand and
    differentiate the RAW LOGIT (not the sigmoid probability), so the gradient
    does not vanish for confident predictions.

    Returns:
        grad_model: Model(resnet.input -> resnet.output), the feature extractor.
        head: dict with the trained GAP layer + classifier weights/bias.
    """
    print(f"Loading model from: {MODEL_PATH} ...")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded. Input:", model.input_shape, "| Output:", model.output_shape)

    resnet = model.get_layer(BASE_MODEL_NAME)

    # Single-output sub-model: input -> 7x7x2048 feature map.
    grad_model = keras.Model(resnet.input, resnet.output)

    classifier = model.get_layer("binary_classifier")
    head = {
        "gap": model.get_layer("global_average_pooling"),
        "dropout": model.get_layer("dropout"),
        "W": classifier.kernel,  # (2048, 1)
        "b": classifier.bias,    # (1,)
    }
    return grad_model, head


# =========================
# Grad-CAM core
# =========================

def compute_gradcam(image_uint8, grad_model, head):
    """Return a Grad-CAM heatmap resized to the original image size.

    Args:
        image_uint8: HxWx3 RGB uint8 image (any size).
        grad_model: sub-model returning the 7x7x2048 feature map.
        head: dict with the trained GAP layer + classifier weights.

    Returns:
        heatmap: HxW float in [0, 1], same H,W as the input image.
        prob: float, the model's positive-class probability.
    """
    # Resize to model input and add batch dim.
    img_resized = cv2.resize(image_uint8, IMG_SIZE)
    x = np.expand_dims(img_resized.astype("float32"), axis=0)

    # Apply ResNet50 preprocessing (the saved model had this baked in too).
    x = preprocess_input(x)

    # Forward + gradient in one pass.
    # We differentiate the RAW LOGIT (not the sigmoid probability) so the
    # gradient does not vanish for confident predictions. We also pick the
    # score for the PREDICTED class: for a binary sigmoid head there is a
    # single logit z, so "class 1 (positive)" uses +z and "class 0 (negative)"
    # uses -z. This shows why the model picked its decision either way.
    with tf.GradientTape() as tape:
        feat = grad_model(x, training=False)          # (1, 7, 7, 2048)
        pooled = head["gap"](feat)                    # (1, 2048)
        pooled = head["dropout"](pooled, training=False)
        logit = tf.linalg.matvec(pooled, tf.squeeze(head["W"])) + head["b"]  # (1,)
        # Decide predicted class from a quick forward pass (detached).
        prob = float(tf.math.sigmoid(logit[0]).numpy())
        score = logit[0] if prob >= 0.5 else -logit[0]

    grads = tape.gradient(score, feat)                # (1, 7, 7, 2048)

    # Global-average the gradients -> per-channel weights.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # (2048,)

    feat0 = feat[0]                                   # (7, 7, 2048)
    heatmap = feat0 @ pooled_grads[..., tf.newaxis]   # (7, 7, 1)
    heatmap = tf.squeeze(heatmap)                     # (7, 7)

    # ReLU + normalize to [0, 1].
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    # Resize heatmap to original image size.
    heatmap = cv2.resize(heatmap, (image_uint8.shape[1], image_uint8.shape[0]))

    return heatmap, prob


# =========================
# Overlays
# =========================

def make_heatmap_overlay(image_uint8, heatmap, alpha=0.4):
    """Blend the jet-colored heatmap over the image."""
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(heatmap_color, alpha, image_uint8, 1 - alpha, 0)
    return overlay


def make_green_box_overlay(image_uint8, heatmap, threshold_frac=0.5):
    """Draw a green bounding box around the hottest region on the ORIGINAL image.

    We threshold the heatmap, take the largest connected contour, and box it.
    Returns a copy of the original image with a green rectangle + label.
    """
    out = image_uint8.copy()
    h, w = out.shape[:2]

    # Threshold at threshold_frac of the max activation.
    binary = (heatmap >= (heatmap.max() * threshold_frac)).astype("uint8") * 255

    if binary.sum() == 0:
        # Fallback: box the single hottest pixel.
        idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        x1, y1 = max(idx[1] - 10, 0), max(idx[0] - 10, 0)
        x2, y2 = min(idx[1] + 10, w), min(idx[0] + 10, h)
        cv2.rectangle(out, (x1, y1), (x2, y2), GREEN, 3)
        return out

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return out

    # Largest contour -> tightest bounding box.
    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    # Expand the box slightly for visual padding (clamped to image bounds).
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


# =========================
# Predict (used by Gradio)
# =========================

def predict(image):
    """Gradio callback. Returns heatmap image, green-box image, and a label dict."""
    if image is None:
        raise gr.Error("لطفاً ابتدا یک تصویر بارگذاری کنید.")

    # Gradio gives RGB uint8 (HxWx3). Work on a uint8 copy.
    image_uint8 = np.array(image, dtype=np.uint8)
    if image_uint8.ndim != 3 or image_uint8.shape[2] != 3:
        raise gr.Error("تصویر باید RGB باشد.")

    heatmap, prob = compute_gradcam(image_uint8, GRAD_MODEL, HEAD)
    pred_class = int(prob >= 0.5)
    label_name = CLASS_NAMES[pred_class]

    overlay_img = make_heatmap_overlay(image_uint8, heatmap)
    box_img = make_green_box_overlay(image_uint8, heatmap)

    confidence = prob if pred_class == 1 else (1 - prob)
    confidence = float(confidence)

    label_dict = {
        CLASS_NAMES[0]: float(1 - prob),
        CLASS_NAMES[1]: float(prob),
    }

    summary = (
        f"پیش‌بینی: **{label_name}**  |  اطمینان: **{confidence:.1%}**"
    )

    return overlay_img, box_img, label_dict, summary


# =========================
# Build / launch UI
# =========================

def build_ui():
    with gr.Blocks(title="Beet Classifier - Grad-CAM") as demo:
        gr.Markdown("# دسته‌بندی چغندر با Grad-CAM")
        gr.Markdown(
            "یک تصویر بارگذاری کنید تا پیش‌بینی مدل و ناحیه‌ی تمرکز آن را ببینید. "
            "**نقشه‌ی حرارتی** بخش‌های مهم تصویر را برجسته می‌کند و "
            "**باکس سبز** مرز همان ناحیه را روی تصویر اصلی نشان می‌دهد."
        )

        with gr.Row():
            with gr.Column(scale=1):
                img_in = gr.Image(type="numpy", label="تصویر ورودی")
                btn = gr.Button("پیش‌بینی", variant="primary")
                summary = gr.Markdown(label="نتیجه")
                label_out = gr.Label(num_top_classes=2, label="احتمال کلاس‌ها")

            with gr.Column(scale=1):
                gr.Markdown("### نقشه‌ی حرارتی Grad-CAM")
                heatmap_out = gr.Image(type="numpy", label="نقشه‌ی حرارتی")
                gr.Markdown("### باکس سبز روی تصویر اصلی")
                box_out = gr.Image(type="numpy", label="ناحیه‌ی تمرکز")

        btn.click(
            predict,
            inputs=img_in,
            outputs=[heatmap_out, box_out, label_out, summary],
        )

    return demo


# =========================
# Main
# =========================

if __name__ == "__main__":
    GRAD_MODEL, HEAD = load_model()
    demo = build_ui()
    demo.launch()
