"""
dashboard/consumers.py

The real-time pipeline (Section 34):

    Browser --(sampled frame, base64 JPEG)--> WebSocket --> Django
        -> YOLO detection
        -> Depth estimation (only every Nth AI frame -- Section 51/52)
        -> Distance estimation
        -> Tracking (stable IDs)
        -> Navigation (direction / movement analysis / voice cooldown)
        -> JSON --> Browser

Each browser tab gets its own consumer instance, and each consumer owns
its own ObjectTracker + TargetLock so multiple simultaneous users never
share tracking state. The YOLO and depth *models* themselves remain
process-wide singletons (Section 53) to avoid loading them more than once.

Only small JPEG frames sampled at ~5-8 FPS are ever sent over the socket
(never raw 30 FPS video -- Section 34), and frames are discarded
immediately after processing (Section 6: no frame queues/history kept).
"""

import base64
import json
import logging
import time

from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

logger = logging.getLogger("dashboard.consumers")


class VisionConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

        # Deferred imports: keep Django's app startup fast and avoid
        # importing heavy CV/ML libraries unless a client actually connects.
        from navigation.navigator import Navigator
        from navigation.tracking import TargetLock
        from vision.depth_estimator import DepthEstimator
        from vision.detector import ObjectDetector
        from vision.tracker import ObjectTracker

        self.detector = ObjectDetector.get_instance()
        self.depth_estimator = DepthEstimator.get_instance()
        self.object_tracker = ObjectTracker()
        self.target_lock = TargetLock()
        self.navigator = None  # created once we know the frame width

        self.ai_fps = settings.AI_PROCESSING_FPS
        self.depth_every_n_ai_frames = settings.PERFORMANCE_PROFILES[settings.PERFORMANCE_MODE][
            "depth_every_n_ai_frames"
        ]
        self.ai_frame_counter = 0
        self.last_inference_ms = 0.0

        await self.send(text_data=json.dumps({
            "type": "status",
            "model": self.detector.status(),
            "depth": self.depth_estimator.status(),
            "ai_fps": self.ai_fps,
            "device": "cpu",
        }))

    async def disconnect(self, close_code):
        # No persistent per-connection resources (no open file handles,
        # no large buffers) -- nothing extra to release here beyond the
        # lightweight Python objects created in connect().
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if not text_data:
            return

        try:
            message = json.loads(text_data)
        except json.JSONDecodeError:
            await self._send_error("Invalid JSON message.")
            return

        msg_type = message.get("type")

        if msg_type == "frame":
            await self._handle_frame(message)
        elif msg_type == "set_target":
            await self._handle_set_target(message)
        elif msg_type == "clear_target":
            self.target_lock.clear_target()
            await self.send(text_data=json.dumps({"type": "target_cleared"}))
        elif msg_type == "voice_command":
            await self._handle_voice_command(message)
        elif msg_type == "set_ai_fps":
            await self._handle_set_ai_fps(message)
        else:
            await self._send_error(f"Unknown message type: {msg_type}")

    # ------------------------------------------------------------------
    async def _handle_frame(self, message):
        import asyncio
        import cv2
        import numpy as np

        from vision import calibration as calib
        from vision.distance_estimator import estimate_object_distance
        from navigation.navigator import Navigator
        from navigation.direction import classify_direction

        b64_image = message.get("image")
        if not b64_image:
            await self._send_error("Frame message missing 'image'.")
            return

        try:
            header_split = b64_image.split(",", 1)
            raw_b64 = header_split[1] if len(header_split) == 2 else header_split[0]
            img_bytes = base64.b64decode(raw_b64)
            np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception:
            logger.exception("Failed to decode incoming frame")
            await self._send_error("Could not decode frame image.")
            return

        if frame is None:
            await self._send_error("Could not decode frame image.")
            return

        frame_height, frame_width = frame.shape[:2]
        if self.navigator is None or self.navigator.frame_width != frame_width:
            self.navigator = Navigator(frame_width=frame_width)

        # --- 1. Object detection (every AI frame, offloaded to worker thread) ---
        start = time.perf_counter()
        detections, yolo_ms = await asyncio.to_thread(self.detector.detect, frame)

        # --- 2. Depth estimation (only every Nth AI frame -- Section 52) ---
        self.ai_frame_counter += 1
        run_depth = (self.ai_frame_counter % self.depth_every_n_ai_frames) == 0
        depth_map = None
        depth_ms = 0.0
        if run_depth and detections:
            depth_map, depth_ms = await asyncio.to_thread(self.depth_estimator.predict, frame)

        # --- 3. Tracking (stable IDs across frames) ---
        detections = self.object_tracker.update(detections)

        # --- 4. Distance estimation per detection ---
        calibration_k = calib.get_current_k()
        distances_by_index = {}
        objects_payload = []
        for det in detections:
            distance_info = estimate_object_distance(
                depth_map, det.bbox, class_name=det.class_name, calibration_k=calibration_k
            )
            distances_by_index[id(det)] = distance_info
            direction = classify_direction(det.center[0], frame_width)

            payload = det.as_dict()
            payload["direction"] = direction
            payload["distance"] = distance_info["distance_m"]
            payload["distance_label"] = distance_info["label"]
            payload["category"] = distance_info["category"]
            payload["low_confidence"] = det.confidence < settings.CONFIDENCE_THRESHOLD
            objects_payload.append(payload)

        total_ms = (time.perf_counter() - start) * 1000.0
        self.last_inference_ms = total_ms

        # --- 5. Auto performance backoff (Section 31) ---
        self._maybe_adjust_performance(total_ms)

        # --- 6. Navigation update for the currently locked target ---
        navigation_result = self.navigator.process(detections, distances_by_index, self.target_lock)

        await self.send(text_data=json.dumps({
            "type": "detections",
            "objects": objects_payload,
            "fps": self.ai_fps,
            "inference_ms": round(total_ms, 1),
            "yolo_ms": round(yolo_ms, 1),
            "depth_ms": round(depth_ms, 1),
            "depth_ran_this_frame": run_depth,
            "device": "cpu",
            "navigation": navigation_result,
        }))

    # ------------------------------------------------------------------
    async def _handle_set_target(self, message):
        from vision.object_registry import normalize_object_name

        raw_object = message.get("object")
        if not raw_object:
            await self._send_error("set_target requires an 'object' field.")
            return

        canonical = normalize_object_name(raw_object)
        self.target_lock.set_target(canonical)

        await self.send(text_data=json.dumps({
            "type": "target_set",
            "object": canonical,
        }))

    # ------------------------------------------------------------------
    async def _handle_voice_command(self, message):
        from voice.command_parser import parse_command
        from voice import responses
        from vision.object_registry import display_name, SUPPORTED_OBJECTS

        text = message.get("text", "")
        parsed = parse_command(text)
        intent = parsed["intent"]
        obj = parsed["object"]

        if intent in ("find_object", "start_tracking"):
            if not obj:
                await self.send(text_data=json.dumps({
                    "type": "voice_response",
                    "intent": intent,
                    "speak_text": "Sorry, I didn't catch which object you mean.",
                }))
                return
            self.target_lock.set_target(obj)
            await self.send(text_data=json.dumps({
                "type": "voice_response",
                "intent": intent,
                "object": obj,
                "speak_text": f"Looking for your {display_name(obj)}.",
            }))

        elif intent == "stop_tracking":
            self.target_lock.clear_target()
            await self.send(text_data=json.dumps({
                "type": "voice_response",
                "intent": intent,
                "speak_text": "Stopped tracking.",
            }))

        elif intent == "list_objects":
            names = ", ".join(display_name(o) for o in SUPPORTED_OBJECTS)
            await self.send(text_data=json.dumps({
                "type": "voice_response",
                "intent": intent,
                "speak_text": f"I can look for: {names}.",
            }))

        else:
            await self.send(text_data=json.dumps({
                "type": "voice_response",
                "intent": "unknown",
                "speak_text": "Sorry, I didn't understand that command.",
            }))

    # ------------------------------------------------------------------
    async def _handle_set_ai_fps(self, message):
        requested = message.get("fps")
        if requested in settings.ALLOWED_AI_FPS:
            self.ai_fps = requested
            await self.send(text_data=json.dumps({"type": "ai_fps_set", "fps": self.ai_fps}))
        else:
            await self._send_error(f"fps must be one of {settings.ALLOWED_AI_FPS}")

    # ------------------------------------------------------------------
    def _maybe_adjust_performance(self, inference_ms):
        """Section 31: automatically reduce AI FPS / depth frequency if
        inference is consistently too slow for the target machine."""
        if inference_ms > settings.INFERENCE_SLOW_THRESHOLD_MS:
            if self.ai_fps > min(settings.ALLOWED_AI_FPS):
                lower_options = [f for f in settings.ALLOWED_AI_FPS if f < self.ai_fps]
                if lower_options:
                    self.ai_fps = max(lower_options)
                    logger.info("Auto performance backoff: AI FPS reduced to %s", self.ai_fps)
            if self.depth_every_n_ai_frames < 4:
                self.depth_every_n_ai_frames += 1
                logger.info(
                    "Auto performance backoff: depth now runs every %s AI frames",
                    self.depth_every_n_ai_frames,
                )

    # ------------------------------------------------------------------
    async def _send_error(self, message):
        await self.send(text_data=json.dumps({"type": "error", "message": message}))
