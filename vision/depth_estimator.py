"""
vision/depth_estimator.py

Lightweight monocular depth estimation, loaded once and reused (Section 14).

Primary backend: MiDaS "small" (via torch.hub, intel-isl/MiDaS), which is
small enough (~21 MB) to run on a CPU-only Intel i5 laptop at a few frames
per second when only run every 2nd-3rd AI frame (Section 51/52).

If the MiDaS weights cannot be downloaded/loaded (e.g. no internet access
on first run, or torch/timm missing), the estimator marks itself
unavailable. The rest of the pipeline (detection + known-object-size
distance + navigation) continues to function without it (Section 38).
"""

import logging
import threading
import time

from django.conf import settings

logger = logging.getLogger("vision.depth")


class DepthEstimator:
    """Singleton wrapper around a MiDaS-small depth model."""

    _instance = None
    _lock = threading.Lock()

    def __init__(self):
        self.model = None
        self.transform = None
        self.device = "cpu"
        self.available = False
        self.error = None
        self._load()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    def _load(self):
        try:
            import torch
        except ImportError:
            self.error = "torch is not installed; depth estimation disabled."
            logger.warning(self.error)
            return

        try:
            # torch.hub caches the model+weights locally after first download.
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
            self.model.to(self.device)
            self.model.eval()

            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
            self.transform = midas_transforms.small_transform
            self.available = True
            self.error = None
        except Exception as exc:
            # No internet, no cached weights, or incompatible torch version.
            self.error = f"Depth model unavailable ({exc}). Falling back to known-size distance estimation."
            logger.warning(self.error)
            self.model = None
            self.available = False

    # ------------------------------------------------------------------
    def predict(self, frame_bgr):
        """Run monocular depth estimation on a BGR frame.

        Returns (depth_map: np.ndarray | None, inference_ms: float)
        depth_map values are *relative* inverse-depth (MiDaS convention:
        higher value = closer). They are NOT metric distances -- metric
        conversion happens in distance_estimator.py using calibration data.
        """
        if not self.available:
            return None, 0.0

        import numpy as np
        import torch

        start = time.perf_counter()
        try:
            img_rgb = frame_bgr[:, :, ::-1]  # BGR -> RGB, no extra copy needed downstream
            input_batch = self.transform(img_rgb).to(self.device)

            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img_rgb.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth_map = prediction.cpu().numpy().astype(np.float32)
        except Exception:
            logger.exception("Depth inference failed")
            return None, (time.perf_counter() - start) * 1000.0

        inference_ms = (time.perf_counter() - start) * 1000.0
        return depth_map, inference_ms

    def status(self):
        return {"available": self.available, "error": self.error, "backend": "MiDaS_small"}
