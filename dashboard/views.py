import base64
import logging

from django.conf import settings
from django.contrib import messages
from django.shortcuts import redirect, render

from dashboard.models import CameraCalibration
from navigation.direction import classify_direction

logger = logging.getLogger("dashboard.views")


def index(request):
    """Homepage (Section 4, Step 1)."""
    return render(request, "dashboard/index.html")


def camera(request):
    """Live camera + AI vision page (Section 4, Step 2 onward)."""
    context = {
        "ai_fps_options": settings.ALLOWED_AI_FPS,
        "default_ai_fps": settings.AI_PROCESSING_FPS,
        "camera_width": settings.CAMERA_WIDTH,
        "camera_height": settings.CAMERA_HEIGHT,
        "performance_mode": settings.PERFORMANCE_MODE,
    }
    return render(request, "dashboard/camera.html", context)


def calibrate(request):
    """Calibration page (Section 16, 43): capture (known_distance, depth_value)
    pairs and store them so distance_estimator can convert future MiDaS
    depth values into approximate meters."""

    if request.method == "POST":
        try:
            reference_distance = float(request.POST.get("reference_distance"))
            depth_value = float(request.POST.get("depth_value"))
            if reference_distance <= 0 or depth_value <= 0:
                raise ValueError("Values must be positive.")
            CameraCalibration.objects.create(
                reference_distance=reference_distance, depth_value=depth_value
            )
            from vision import calibration as calib
            calib.clear_k_cache()
            messages.success(
                request,
                f"Calibration point saved: {reference_distance} m -> depth {depth_value:.3f}",
            )
        except (TypeError, ValueError):
            messages.error(request, "Please provide valid numeric values.")
        return redirect("dashboard:calibrate")

    points = CameraCalibration.objects.all().order_by("reference_distance")
    context = {
        "points": points,
        "suggested_distances": [0.5, 1.0, 1.5, 2.0, 3.0],
    }
    return render(request, "dashboard/calibration.html", context)


def test_image(request):
    """Debug/offline test page (Section 42): upload a static image and run
    the full pipeline (detection -> depth -> distance -> direction) without
    needing a live webcam. Nothing is written to disk."""

    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        import cv2
        import numpy as np

        from vision import calibration as calib
        from vision.depth_estimator import DepthEstimator
        from vision.detector import ObjectDetector
        from vision.distance_estimator import estimate_object_distance
        from vision.object_registry import display_name

        uploaded = request.FILES["image"]
        file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            messages.error(request, "Could not read that image file.")
            return render(request, "dashboard/test.html", context)

        detector = ObjectDetector.get_instance()
        depth_estimator = DepthEstimator.get_instance()

        detections, inference_ms = detector.detect(frame)
        depth_map, depth_ms = depth_estimator.predict(frame)
        calibration_k = calib.get_current_k()

        frame_height, frame_width = frame.shape[:2]
        results = []
        annotated = frame.copy()

        for det in detections:
            direction = classify_direction(det.center[0], frame_width)
            distance_info = estimate_object_distance(
                depth_map, det.bbox, class_name=det.class_name, calibration_k=calibration_k
            )

            x1, y1, x2, y2 = [int(v) for v in det.bbox]
            color = (0, 200, 0) if det.confidence >= settings.CONFIDENCE_THRESHOLD else (0, 165, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            label = f"{display_name(det.class_name)} {det.confidence:.0%} {distance_info['label']}"
            cv2.putText(annotated, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            results.append(
                {
                    "class_name": display_name(det.class_name),
                    "raw_class": det.raw_class_name,
                    "confidence": round(det.confidence * 100, 1),
                    "direction": direction,
                    "distance_label": distance_info["label"],
                    "category": distance_info["category"],
                    "low_confidence": det.confidence < settings.CONFIDENCE_THRESHOLD,
                }
            )

        success, buffer = cv2.imencode(".jpg", annotated)
        annotated_b64 = base64.b64encode(buffer).decode("utf-8") if success else None

        context.update(
            {
                "results": results,
                "annotated_image_b64": annotated_b64,
                "inference_ms": round(inference_ms, 1),
                "depth_ms": round(depth_ms, 1),
                "model_status": detector.status(),
                "depth_status": depth_estimator.status(),
                "frame_size": f"{frame_width}x{frame_height}",
            }
        )

    return render(request, "dashboard/test.html", context)


def help_about(request):
    """About/Help page, including the mandatory safety/limitations
    disclosure (Section 55)."""
    return render(request, "dashboard/help.html")
