"""
vision/tracker.py

A lightweight, dependency-free IoU-based multi-object tracker.

Design note (Section 22): Ultralytics also ships a ByteTrack integration
(`model.track(persist=True, tracker="bytetrack.yaml")`), but that couples
tracking directly to the YOLO call and re-runs its own Kalman-filter/
association logic every frame -- extra CPU work on an already CPU-bound
i5 machine. Because we already run YOLO via `.predict()` (Section 9) and
only need to keep a stable identity for a handful of objects at 5-8 FPS,
a simple greedy IoU tracker gives us "ByteTrack-like" stable IDs at a
fraction of the CPU cost, which matches the low-RAM/low-CPU goals in
Section 2 and 54. The tracker is intentionally swappable: replace
`ObjectTracker.update()` with a ByteTrack call later without touching any
other module (Section 56).
"""

import itertools
import time


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class _Track:
    __slots__ = ("track_id", "class_name", "bbox", "confidence", "last_seen")

    def __init__(self, track_id, class_name, bbox, confidence):
        self.track_id = track_id
        self.class_name = class_name
        self.bbox = bbox
        self.confidence = confidence
        self.last_seen = time.monotonic()


class ObjectTracker:
    """Per-session tracker instance (NOT a global singleton -- each browser
    session/WebSocket connection should own its own ObjectTracker so that
    tracks from different users/tabs never collide)."""

    def __init__(self, iou_match_threshold=0.3, max_age_seconds=4.0):
        self.iou_match_threshold = iou_match_threshold
        self.max_age_seconds = max_age_seconds
        self._tracks = {}  # track_id -> _Track
        self._id_counter = itertools.count(1)

    def update(self, detections):
        """Match new detections to existing tracks by class + IoU.

        `detections` is a list of DetectionResult (see vision/detector.py).
        Mutates each detection's `.track_id` in place and returns the same
        list for convenience.
        """
        now = time.monotonic()

        # Drop stale tracks first.
        stale_ids = [
            tid for tid, trk in self._tracks.items()
            if now - trk.last_seen > self.max_age_seconds
        ]
        for tid in stale_ids:
            del self._tracks[tid]

        unmatched_track_ids = set(self._tracks.keys())

        for det in detections:
            best_id = None
            best_iou = 0.0
            for tid in unmatched_track_ids:
                trk = self._tracks[tid]
                if trk.class_name != det.class_name:
                    continue
                score = _iou(trk.bbox, det.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = tid

            if best_id is not None and best_iou >= self.iou_match_threshold:
                trk = self._tracks[best_id]
                trk.bbox = det.bbox
                trk.confidence = det.confidence
                trk.last_seen = now
                det.track_id = best_id
                unmatched_track_ids.discard(best_id)
            else:
                new_id = next(self._id_counter)
                self._tracks[new_id] = _Track(new_id, det.class_name, det.bbox, det.confidence)
                det.track_id = new_id

        return detections

    def get_track(self, track_id):
        return self._tracks.get(track_id)

    def is_alive(self, track_id):
        return track_id in self._tracks

    def seconds_since_seen(self, track_id):
        trk = self._tracks.get(track_id)
        if trk is None:
            return None
        return time.monotonic() - trk.last_seen
