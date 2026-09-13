"""
api/views.py

Stateless-ish REST endpoints (Section 33). Per-user tracking state (the
currently locked target) is kept in the Django session so start-tracking /
stop-tracking / detect behave consistently across requests without needing
a database table for live state.

Note: the primary real-time experience (Section 34) uses the WebSocket
consumer in dashboard/consumers.py, which is far more efficient for
continuous frame streaming. These REST endpoints exist for: (a) simple
integrations/testing with curl/Postman, and (b) the voice-command /
tracking control-plane actions, which are infrequent and fit REST well.
"""

import base64
import logging

from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import JSONParser, MultiPartParser
from rest_framework.response import Response

from dashboard.models import CameraCalibration, DetectionLog
from navigation.direction import classify_direction
from vision.object_registry import display_name, normalize_object_name

from .serializers import CalibrationSerializer, StartTrackingSerializer, VoiceCommandSerializer

logger = logging.getLogger("api.views")

SESSION_TARGET_KEY = "vision_navigator_target"


def _decode_frame_from_request(request):
    """Accepts either a multipart file under 'image' or a JSON body with
    'image_base64'. Returns a BGR numpy array or None."""
    import cv2
    import numpy as np

    if "image" in request.FILES:
        file_bytes = np.frombuffer(request.FILES["image"].read(), dtype=np.uint8)
        return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    image_b64 = None
    if isinstance(request.data, dict):
        image_b64 = request.data.get("image_base64")

    if image_b64:
        header_split = image_b64.split(",", 1)
        raw_b64 = header_split[1] if len(header_split) == 2 else header_split[0]
        try:
            img_bytes = base64.b64decode(raw_b64)
        except Exception:
            return None
        np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    return None


@api_view(["POST"])
@parser_classes([MultiPartParser, JSONParser])
def detect(request):
    """POST /api/detect/

    Runs the detection + depth + distance + direction pipeline on a single
    submitted frame (Section 33 response format)."""
    from vision import calibration as calib
    from vision.depth_estimator import DepthEstimator
    from vision.detector import ObjectDetector
    from vision.distance_estimator import estimate_object_distance

    frame = _decode_frame_from_request(request)
    if frame is None:
        return Response(
            {"error": "No valid image provided. Send multipart 'image' or JSON 'image_base64'."},
            status=status.HTTP_400_BAD_REQUEST,
        )

    detector = ObjectDetector.get_instance()
    depth_estimator = DepthEstimator.get_instance()

    if not detector.is_available:
        return Response(
            {"error": "AI detection model is unavailable. Please place the model in "
                      "models/custom/best.pt or ensure ultralytics is installed."},
            status=status.HTTP_503_SERVICE_UNAVAILABLE,
        )

    detections, inference_ms = detector.detect(frame)
    depth_map, depth_ms = depth_estimator.predict(frame) if detections else (None, 0.0)
    calibration_k = calib.get_current_k()

    frame_height, frame_width = frame.shape[:2]
    objects_payload = []
    for det in detections:
        direction = classify_direction(det.center[0], frame_width)
        distance_info = estimate_object_distance(
            depth_map, det.bbox, class_name=det.class_name, calibration_k=calibration_k
        )
        payload = det.as_dict()
        payload["direction"] = direction
        payload["distance"] = distance_info["distance_m"]
        payload["distance_label"] = distance_info["label"]
        payload["category"] = distance_info["category"]
        objects_payload.append(payload)

        # Lightweight logging (metadata only -- never the frame itself).
        try:
            DetectionLog.objects.create(
                object_name=det.class_name,
                confidence=det.confidence,
                distance=distance_info["distance_m"],
                direction=direction,
            )
        except Exception:
            logger.warning("Could not write DetectionLog entry", exc_info=True)

    return Response(
        {
            "objects": objects_payload,
            "fps": settings.AI_PROCESSING_FPS,
            "device": "cpu",
            "inference_ms": round(inference_ms, 1),
            "depth_ms": round(depth_ms, 1),
            "model": detector.status(),
        }
    )


@api_view(["POST"])
def voice_command(request):
    """POST /api/voice-command/ {"text": "Where is my mobile?"}"""
    from voice import responses
    from voice.command_parser import parse_command

    serializer = VoiceCommandSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    text = serializer.validated_data["text"]

    parsed = parse_command(text)
    intent = parsed["intent"]
    obj = parsed["object"]

    if intent in ("find_object", "start_tracking") and obj:
        request.session[SESSION_TARGET_KEY] = obj
        speak_text = f"Looking for your {display_name(obj)}."
    elif intent == "stop_tracking":
        request.session.pop(SESSION_TARGET_KEY, None)
        speak_text = "Stopped tracking."
    elif intent == "list_objects":
        from vision.object_registry import SUPPORTED_OBJECTS
        speak_text = "I can look for: " + ", ".join(display_name(o) for o in SUPPORTED_OBJECTS) + "."
    else:
        speak_text = "Sorry, I didn't understand that command."

    return Response({"intent": intent, "object": obj, "speak_text": speak_text})


@api_view(["POST"])
def start_tracking(request):
    """POST /api/start-tracking/ {"object": "mobile_phone"}"""
    serializer = StartTrackingSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    canonical = normalize_object_name(serializer.validated_data["object"])
    request.session[SESSION_TARGET_KEY] = canonical
    return Response({"tracking": canonical})


@api_view(["POST"])
def stop_tracking(request):
    """POST /api/stop-tracking/"""
    request.session.pop(SESSION_TARGET_KEY, None)
    return Response({"tracking": None})


@api_view(["GET"])
def api_status(request):
    """GET /api/status/ -- developer/performance panel data (Section 29)."""
    from vision.depth_estimator import DepthEstimator
    from vision.detector import ObjectDetector

    detector = ObjectDetector.get_instance()
    depth_estimator = DepthEstimator.get_instance()

    return Response(
        {
            "model": detector.status(),
            "depth": depth_estimator.status(),
            "ai_fps": settings.AI_PROCESSING_FPS,
            "performance_mode": settings.PERFORMANCE_MODE,
            "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
            "device": "cpu",
            "current_target": request.session.get(SESSION_TARGET_KEY),
        }
    )


@api_view(["POST"])
def calibration(request):
    """POST /api/calibration/ {"reference_distance": 1.0, "depth_value": 5.8}"""
    serializer = CalibrationSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)
    instance = serializer.save()
    return Response(CalibrationSerializer(instance).data, status=status.HTTP_201_CREATED)
