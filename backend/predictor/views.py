"""Django views for the beet classifier."""

import base64

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from .gradcam import compute_gradcam, encode_png, make_green_box_overlay, make_heatmap_overlay
from .model import CLASS_NAMES, preprocess_image, predict


def index(request):
    """Serve the built single-file React frontend."""
    return render(request, "index.html")


@csrf_exempt
@require_http_methods(["POST"])
def predict_view(request):
    """Accept an uploaded image and return prediction + Grad-CAM visuals."""
    image_file = request.FILES.get("image")
    if not image_file:
        return JsonResponse({"error": "No image provided."}, status=400)

    try:
        image_uint8 = preprocess_image(image_file)
    except Exception as exc:
        return JsonResponse({"error": f"Invalid image: {exc}"}, status=400)

    try:
        pred_info = predict(image_uint8)
        heatmap, _ = compute_gradcam(image_uint8)

        overlay = make_heatmap_overlay(image_uint8, heatmap)
        box = make_green_box_overlay(image_uint8, heatmap)

        overlay_b64 = base64.b64encode(encode_png(overlay)).decode("ascii")
        box_b64 = base64.b64encode(encode_png(box)).decode("ascii")

        return JsonResponse(
            {
                "prediction": "diseased" if pred_info["class_index"] == 1 else "healthy",
                "confidence": pred_info["confidence"],
                "prob_healthy": pred_info["prob_healthy"],
                "prob_diseased": pred_info["prob_diseased"],
                "label_fa": pred_info["prediction"],
                "heatmap_url": f"data:image/png;base64,{overlay_b64}",
                "visualization_url": f"data:image/png;base64,{box_b64}",
            }
        )
    except Exception as exc:
        return JsonResponse({"error": f"Prediction failed: {exc}"}, status=500)
