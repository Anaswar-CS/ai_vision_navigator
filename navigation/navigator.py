"""
navigation/navigator.py

The main navigation engine (Section 24-27, 46). Given the current frame's
detections and a per-session TargetLock, this module:

    1. Selects the best matching detection for the locked target class
       (nearest / highest confidence when multiple exist -- Section 23).
    2. Computes direction + smoothed distance.
    3. Compares against recent history to decide on a movement status
       ("getting closer", "moving away", "correct direction", "moved past").
    4. Requires several consistent observations before declaring a trend
       (Section 46) and applies a voice cooldown so it doesn't repeat
       instructions every frame (Section 26).
    5. Handles target-lost / reacquired transitions with a grace period
       (Section 27).
"""

import logging

from django.conf import settings

from navigation.direction import classify_direction, smooth_distance
from voice import responses

logger = logging.getLogger("navigation.navigator")


def _select_best_detection(detections, target_class):
    """Pick the nearest/highest-confidence detection of the target class
    among possibly-multiple matches (Section 23)."""
    if target_class == "computer":
        candidates = [d for d in detections if d.class_name in ("computer", "computer_tower", "computer_monitor")]
    elif target_class == "bag":
        candidates = [d for d in detections if d.class_name in ("bag", "backpack", "handbag", "suitcase")]
    else:
        candidates = [d for d in detections if d.class_name == target_class]
    if not candidates:
        return None, candidates

    def sort_key(det):
        # Prefer a known distance (smaller = nearer) then higher confidence.
        return (det.confidence,)

    best = max(candidates, key=sort_key)
    return best, candidates


class Navigator:
    def __init__(self, frame_width):
        self.frame_width = frame_width

    def process(self, detections, distances_by_index, target_lock):
        """
        detections: list[DetectionResult] for the current frame (already
            has `.track_id` set by ObjectTracker).
        distances_by_index: dict mapping id(detection) -> distance_estimator
            result dict (so we don't recompute distance twice).
        target_lock: navigation.tracking.TargetLock for this session.

        Returns a JSON-serializable dict describing the current navigation
        state, including an optional `speak_text` field.
        """
        if not target_lock.has_target:
            return {"target_active": False}

        target_class = target_lock.target_class
        best, candidates = _select_best_detection(detections, target_class)

        if best is None:
            return self._handle_not_found(target_lock)

        distance_info = distances_by_index.get(id(best), {})
        direction = classify_direction(best.center[0], self.frame_width)
        raw_distance = distance_info.get("distance_m")

        smoothed = smooth_distance(raw_distance, target_lock.smoothed_distance)
        target_lock.smoothed_distance = smoothed
        category = distance_info.get("category")

        # The target was marked "lost" (beyond grace period) on a previous
        # frame and has now reappeared -- treat this as a reacquisition.
        reacquired = target_lock.lost_announced

        target_lock.record_observation(
            track_id=best.track_id,
            direction=direction,
            distance_m=smoothed,
            category=category,
            confidence=best.confidence,
        )

        result = {
            "target_active": True,
            "target_class": target_class,
            "found": True,
            "direction": direction,
            "distance_m": round(smoothed, 1) if smoothed is not None else None,
            "distance_label": distance_info.get("label", "distance unavailable"),
            "category": category,
            "confidence": round(best.confidence, 3),
            "track_id": best.track_id,
            "multiple_detected": len(candidates) > 1,
            "detected_count": len(candidates),
            "speak_text": None,
        }

        if best.confidence < settings.CONFIDENCE_THRESHOLD:
            result["low_confidence"] = True

        speak_text = self._decide_speech(target_lock, direction, smoothed, category,
                                          reacquired, len(candidates))
        result["speak_text"] = speak_text
        return result

    # ------------------------------------------------------------------
    def _handle_not_found(self, target_lock):
        target_class = target_lock.target_class

        if target_lock.last_seen_ts is None:
            # Never seen this session -- simple "not found yet" state.
            speak_text = None
            if target_lock.can_speak_now() and target_lock.last_spoken_state != "never_found":
                speak_text = responses.not_found_message(target_class)
                target_lock.mark_spoken("never_found")
            return {
                "target_active": True,
                "target_class": target_class,
                "found": False,
                "speak_text": speak_text,
            }

        if target_lock.is_within_grace_period():
            # Grace period: don't announce loss yet, keep last known info.
            return {
                "target_active": True,
                "target_class": target_class,
                "found": False,
                "grace_period": True,
                "speak_text": None,
            }

        # Permanently lost (beyond grace period).
        speak_text = None
        if not target_lock.lost_announced and target_lock.can_speak_now():
            speak_text = responses.target_lost_message(target_class)
            target_lock.lost_announced = True
            target_lock.mark_spoken("lost")

        return {
            "target_active": True,
            "target_class": target_class,
            "found": False,
            "lost": True,
            "speak_text": speak_text,
        }

    # ------------------------------------------------------------------
    def _decide_speech(self, target_lock, direction, smoothed_distance, category,
                        reacquired, candidate_count):
        """Implements Section 26 (cooldown + only speak on significant
        change) and Section 46 (require consistent frames before declaring
        a movement trend)."""

        if not target_lock.can_speak_now():
            return None

        # Target reacquired after being lost.
        if reacquired:
            target_lock.lost_announced = False
            distance_label = f"approximately {round(smoothed_distance, 1)} meters" \
                if smoothed_distance else "distance unavailable"
            text = responses.target_reacquired_message(target_lock.target_class, direction, distance_label)
            target_lock.mark_spoken("reacquired")
            return text

        history = list(target_lock.history)
        if len(history) < 2:
            # First-ever sighting this session: announce location once.
            distance_label = f"approximately {round(smoothed_distance, 1)} meters" \
                if smoothed_distance else "distance unavailable"
            text = responses.found_object_message(target_lock.target_class, direction, distance_label)
            if candidate_count > 1:
                text += " " + responses.multiple_objects_message(target_lock.target_class, candidate_count)
            target_lock.mark_spoken("first_sighting")
            return text

        prev = history[-2]
        state_key = f"{direction}:{category}"

        # Direction changed significantly.
        if prev["direction"] != direction:
            if prev["direction"] == "center" and direction != "center":
                text = responses.moved_past_message(direction)
            else:
                text = responses.turn_message(direction)
            target_lock.mark_spoken(state_key)
            return text

        # Distance/category trend -- require N consistent frames (Section 46).
        recent = target_lock.recent_distances(settings.NAVIGATION_CONSISTENCY_FRAMES)
        if len(recent) >= settings.NAVIGATION_CONSISTENCY_FRAMES:
            if all(recent[i] > recent[i + 1] for i in range(len(recent) - 1)):
                if target_lock.last_spoken_state != f"closer:{direction}":
                    target_lock.mark_spoken(f"closer:{direction}")
                    if direction == "center":
                        return responses.correct_direction_message()
                    return responses.moving_toward_message(target_lock.target_class)
            elif all(recent[i] < recent[i + 1] for i in range(len(recent) - 1)):
                if target_lock.last_spoken_state != f"farther:{direction}":
                    target_lock.mark_spoken(f"farther:{direction}")
                    return responses.moving_away_message(target_lock.target_class)

        # Category changed (e.g. MEDIUM -> NEAR) even if direction is stable.
        if target_lock.last_spoken_state != state_key and prev.get("category") != category:
            target_lock.mark_spoken(state_key)
            distance_label = f"approximately {round(smoothed_distance, 1)} meters" \
                if smoothed_distance else "distance unavailable"
            return responses.found_object_message(target_lock.target_class, direction, distance_label)

        return None
