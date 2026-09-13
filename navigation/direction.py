"""
navigation/direction.py

Left / Center / Right zone classification (Section 12) and distance
smoothing (Section 47).
"""

from django.conf import settings


def classify_direction(center_x, frame_width, left_threshold=None, right_threshold=None):
    """Classify a detection's horizontal center into left/center/right."""
    if frame_width <= 0:
        return "center"

    left_threshold = left_threshold if left_threshold is not None else settings.LEFT_THRESHOLD
    right_threshold = right_threshold if right_threshold is not None else settings.RIGHT_THRESHOLD

    normalized_x = center_x / frame_width

    if normalized_x < left_threshold:
        return "left"
    elif normalized_x > right_threshold:
        return "right"
    return "center"


def smooth_distance(current_distance, previous_smoothed, alpha=None):
    """Exponential smoothing to reduce noisy frame-to-frame distance jitter
    (Section 47).

        smoothed = alpha * current + (1 - alpha) * previous
    """
    if current_distance is None:
        return previous_smoothed

    alpha = alpha if alpha is not None else settings.DISTANCE_SMOOTHING_ALPHA

    if previous_smoothed is None:
        return current_distance

    return alpha * current_distance + (1 - alpha) * previous_smoothed
