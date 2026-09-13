"""
navigation/tracking.py

Holds the "locked target" state for a single user session (Section 22, 27).
One instance should be created per WebSocket connection / browser tab.
"""

import time
from collections import deque

from django.conf import settings


class TargetLock:
    """Tracks the currently-locked object across frames, including a grace
    period for temporary occlusion/loss (Section 27) and a short history
    used to require several consistent observations before announcing a
    movement trend (Section 46)."""

    def __init__(self, history_size=10):
        self.target_class = None          # canonical object name, e.g. "mobile_phone"
        self.track_id = None
        self.history = deque(maxlen=history_size)   # list of dicts: direction, distance, ts
        self.smoothed_distance = None
        self.last_seen_ts = None
        self.lost_announced = False
        self.last_voice_ts = 0.0
        self.last_spoken_state = None      # to detect "significant change" (Section 26)

    # ------------------------------------------------------------------
    def set_target(self, canonical_class):
        self.target_class = canonical_class
        self.track_id = None
        self.history.clear()
        self.smoothed_distance = None
        self.last_seen_ts = None
        self.lost_announced = False
        self.last_spoken_state = None

    def clear_target(self):
        self.set_target(None)

    @property
    def has_target(self):
        return self.target_class is not None

    # ------------------------------------------------------------------
    def record_observation(self, track_id, direction, distance_m, category, confidence):
        now = time.monotonic()
        self.track_id = track_id
        self.last_seen_ts = now
        self.lost_announced = False
        self.history.append(
            {
                "direction": direction,
                "distance_m": distance_m,
                "category": category,
                "confidence": confidence,
                "ts": now,
            }
        )

    def seconds_since_last_seen(self):
        if self.last_seen_ts is None:
            return None
        return time.monotonic() - self.last_seen_ts

    def is_within_grace_period(self):
        elapsed = self.seconds_since_last_seen()
        if elapsed is None:
            return True
        return elapsed <= settings.TARGET_LOST_GRACE_SECONDS

    def is_permanently_lost(self):
        elapsed = self.seconds_since_last_seen()
        if elapsed is None:
            return False
        return elapsed > settings.TARGET_LOST_GRACE_SECONDS

    # ------------------------------------------------------------------
    def recent_distances(self, n=None):
        n = n or settings.NAVIGATION_CONSISTENCY_FRAMES
        items = list(self.history)[-n:]
        return [item["distance_m"] for item in items if item["distance_m"] is not None]

    def recent_directions(self, n=None):
        n = n or settings.NAVIGATION_CONSISTENCY_FRAMES
        items = list(self.history)[-n:]
        return [item["direction"] for item in items]

    def can_speak_now(self):
        return (time.monotonic() - self.last_voice_ts) >= settings.VOICE_COOLDOWN

    def mark_spoken(self, state_key):
        self.last_voice_ts = time.monotonic()
        self.last_spoken_state = state_key
