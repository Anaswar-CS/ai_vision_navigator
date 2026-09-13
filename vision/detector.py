"""
vision/detector.py

CPU-optimized object detector built on Ultralytics YOLO.

Design goals (Section 8, 9, 53):
    * Load the model exactly once (module-level singleton).
    * Prefer a custom-trained model (models/custom/best.pt) for objects the
      standard COCO-pretrained model cannot recognize (spectacles, pencil,
      medicine box, instrumentation box, pen).
    * Fall back to the lightweight pretrained YOLO nano model when no
      custom model is present, and clearly report which model is active.
    * Never require a GPU. Always run with device="cpu" unless the caller
      explicitly overrides FORCE_CPU in settings.
"""

import logging
import os
import threading
import time

from django.conf import settings

from vision.object_registry import COCO_TO_CANONICAL, normalize_object_name

logger = logging.getLogger("vision.detector")


class DetectionResult:
    """Plain container for a single detected object."""

    __slots__ = (
        "class_name", "raw_class_name", "confidence", "bbox",
        "center", "track_id",
    )

    def __init__(self, class_name, raw_class_name, confidence, bbox, center, track_id=None):
        self.class_name = class_name
        self.raw_class_name = raw_class_name
        self.confidence = confidence
        self.bbox = bbox            # (x1, y1, x2, y2) in pixels
        self.center = center        # (cx, cy) in pixels
        self.track_id = track_id

    def as_dict(self):
        x1, y1, x2, y2 = self.bbox
        return {
            "class": self.class_name,
            "raw_class": self.raw_class_name,
            "confidence": round(float(self.confidence), 4),
            "bbox": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            "center": [round(float(self.center[0]), 1), round(float(self.center[1]), 1)],
            "track_id": self.track_id,
        }


class ObjectDetector:
    """Loads a YOLO model exactly once and exposes `.detect(frame)`.

    Usage:
        detector = ObjectDetector.get_instance()
        results = detector.detect(frame_bgr)
    """

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.model_source = "none"     # "dual", "custom", or "pretrained"
        self.model_name = None
        self.is_custom = False
        self.error = None
        self.imgsz = settings.PERFORMANCE_PROFILES.get(
            settings.PERFORMANCE_MODE, settings.PERFORMANCE_PROFILES["balanced"]
        )["yolo_imgsz"]
        self._load_model()

    # ------------------------------------------------------------------
    @classmethod
    def get_instance(cls):
        """Thread-safe singleton accessor. Loads the model only once."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def _load_model(self):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            self.error = (
                "The 'ultralytics' package is not installed. "
                "Run: pip install -r requirements.txt"
            )
            logger.error(self.error)
            return

        custom_path = settings.CUSTOM_MODEL_PATH
        try:
            if os.path.isfile(custom_path):
                # ── DUAL-MODEL MODE ──────────────────────────────────────────
                # Load both the fine-tuned custom model AND the pretrained COCO
                # model. detect() will run both and merge results so that:
                #   • Custom classes (pen, pencil, spectacles, medicine_box …)
                #     are detected by the custom model.
                #   • General COCO classes (laptop, mobile_phone, book, bottle …)
                #     are detected by the pretrained model.
                # Overlap resolution: if a pretrained detection shares >50% IoU
                # with a custom detection, the custom detection wins (it is more
                # specific) and the pretrained detection is dropped.
                logger.info("Loading CUSTOM YOLO model from %s", custom_path)
                self.custom_model = YOLO(custom_path)
                self.model_source = "dual"
                self.model_name = os.path.basename(custom_path)
                self.is_custom = True

                logger.info(
                    "Also loading pretrained model '%s' for COCO-class fallback",
                    settings.PRETRAINED_MODEL_NAME,
                )
                self.pretrained_model = YOLO(settings.PRETRAINED_MODEL_NAME)

                # Keep self.model pointing at custom for backward-compat status checks.
                self.model = self.custom_model

            else:
                # ── PRETRAINED-ONLY MODE (no custom model present) ───────────
                logger.info(
                    "No custom model found at %s -- falling back to pretrained '%s'",
                    custom_path, settings.PRETRAINED_MODEL_NAME,
                )
                self.model = YOLO(settings.PRETRAINED_MODEL_NAME)
                self.custom_model = None
                self.pretrained_model = self.model
                self.model_source = "pretrained"
                self.model_name = settings.PRETRAINED_MODEL_NAME
                self.is_custom = False

            # Force CPU inference unless explicitly disabled.
            self.device = "cpu" if settings.FORCE_CPU else None

            # Warm-up pass(es) to eliminate cold-start latency.
            import numpy as np
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            for mdl, label in self._active_models():
                try:
                    mdl.predict(
                        source=dummy_frame,
                        device=self.device,
                        imgsz=self.imgsz,
                        verbose=False,
                        batch=1,
                    )
                    logger.info("YOLO warm-up completed for %s model.", label)
                except Exception as warmup_exc:
                    logger.warning("YOLO warm-up issue (%s): %s", label, warmup_exc)

        except Exception as exc:  # pragma: no cover - defensive
            self.error = f"Failed to load YOLO model: {exc}"
            logger.exception(self.error)
            self.model = None

    # ------------------------------------------------------------------
    def _active_models(self):
        """Yield (model, label) pairs for all loaded models."""
        if self.model_source == "dual":
            yield self.custom_model, "custom"
            yield self.pretrained_model, "pretrained"
        elif self.model_source == "pretrained":
            yield self.pretrained_model, "pretrained"

    # ------------------------------------------------------------------
    @property
    def is_available(self):
        return self.model is not None

    # ------------------------------------------------------------------
    def status(self):
        return {
            "available": self.is_available,
            "model_source": self.model_source,
            "model_name": self.model_name,
            "is_custom": self.is_custom,
            "error": self.error,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _iou(box_a, box_b):
        """Compute Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
        ix1 = max(box_a[0], box_b[0])
        iy1 = max(box_a[1], box_b[1])
        ix2 = min(box_a[2], box_b[2])
        iy2 = min(box_a[3], box_b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter == 0.0:
            return 0.0
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # ------------------------------------------------------------------
    def _run_single_model_raw(self, model, frame, base_conf, is_custom_model):
        """Run one model and return a list of (DetectionResult, passes_report_threshold) tuples."""
        try:
            results = model.predict(
                source=frame,
                conf=base_conf,
                device=getattr(self, "device", "cpu"),
                imgsz=self.imgsz,
                verbose=False,
                batch=1,
            )
        except Exception:
            logger.exception("YOLO inference failed")
            return []

        candidates = []
        if results:
            result = results[0]
            names = result.names or {}
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    cls_idx = int(box.cls[0].item())
                    raw_name = names.get(cls_idx, str(cls_idx))
                    
                    # Scope filter: Pretrained model MUST only detect our known 12 classes
                    if not is_custom_model and raw_name.lower() not in COCO_TO_CANONICAL:
                        continue

                    confidence = float(box.conf[0].item())
                    canonical = self._map_class_name(raw_name, is_custom_model)
                    
                    # Check report threshold
                    class_threshold = getattr(settings, "CLASS_CONFIDENCE_THRESHOLDS", {}).get(
                        canonical, getattr(settings, "CONFIDENCE_THRESHOLD", 0.50)
                    )
                    passes_report = (confidence >= class_threshold)

                    x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    det = DetectionResult(
                        class_name=canonical,
                        raw_class_name=raw_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                    )
                    candidates.append((det, passes_report))
        del results
        return candidates

    # ------------------------------------------------------------------
    def detect(self, frame, confidence_threshold=None):
        """Run detection on a single BGR frame (numpy array).

        In dual-model mode, both the custom and pretrained models are run.
        Two-tier merge logic with VETO_THRESHOLD (Section 8/9):
            1. Custom model runs down to VETO_THRESHOLD (0.25).
               All custom detections >= VETO_THRESHOLD form a 'veto shield'.
               Custom detections >= class report threshold are kept in output.
            2. Pretrained model runs down to VETO_THRESHOLD.
            3. Pretrained detections overlapping any custom veto box (>50% IoU)
               are DROPPED (custom model vetoes competing pretrained guesses,
               preventing medicine_box from ever misidentifying as mobile_phone/book).
            4. Unvetoed pretrained detections >= class report threshold are added.

        Returns a tuple: (list[DetectionResult], inference_ms)
        """
        if not self.is_available:
            return [], 0.0

        veto_conf = getattr(settings, "VETO_THRESHOLD", 0.25)
        start = time.perf_counter()

        if self.model_source == "dual":
            # 1. Custom model raw predictions
            custom_candidates = self._run_single_model_raw(
                self.custom_model, frame, base_conf=veto_conf, is_custom_model=True
            )
            custom_veto_boxes = [
                det.bbox for det, _ in custom_candidates if det.confidence >= veto_conf
            ]
            merged = [det for det, passes in custom_candidates if passes]

            # 2. Pretrained model raw predictions
            pretrained_candidates = self._run_single_model_raw(
                self.pretrained_model, frame, base_conf=veto_conf, is_custom_model=False
            )

            # 3. Overlap resolution with veto protection
            IOU_THRESHOLD = 0.50
            for p_det, p_passes in pretrained_candidates:
                vetoed = any(
                    self._iou(p_det.bbox, cv_box) > IOU_THRESHOLD
                    for cv_box in custom_veto_boxes
                )
                if not vetoed and p_passes:
                    merged.append(p_det)

            detections = merged

        else:
            # Pretrained-only mode
            candidates = self._run_single_model_raw(
                self.pretrained_model, frame, base_conf=veto_conf, is_custom_model=False
            )
            detections = [det for det, passes in candidates if passes]

        inference_ms = (time.perf_counter() - start) * 1000.0
        return detections, inference_ms

    # ------------------------------------------------------------------
    def _map_class_name(self, raw_name, is_custom_model):
        """Map a raw model class name to our canonical vocabulary."""
        if is_custom_model:
            # Custom model is trained directly on our canonical class names.
            return normalize_object_name(raw_name)
        # Pretrained COCO model -> map through COCO_TO_CANONICAL, else normalize.
        mapped = COCO_TO_CANONICAL.get(raw_name.lower())
        return mapped if mapped else normalize_object_name(raw_name)
