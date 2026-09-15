"""
vision/person_distance.py

Third-party-camera scenario: estimate the straight-line distance between
a PERSON and a TARGET OBJECT when both are visible in the same frame.

This is fundamentally different from the assistive self-use case:
  - Self-use:  camera IS the user's eyes. "Where is my phone?" = distance
               from the camera (= from the user) to the object.
  - Third-party: camera watches a room. A person is visible in the frame.
                 "How far is that person from the laptop?" = distance
                 between the person and the laptop, not from the camera
                 to either.

Method (Law of Cosines):
    Given:
      d_person  = camera→person distance (from MiDaS depth estimation)
      d_object  = camera→object distance (from known-size or MiDaS)
      theta     = horizontal angular separation between person and object
                  in the camera's field of view

    person_to_object = sqrt(d_person² + d_object² - 2·d_person·d_object·cos(θ))

Simplification note:
    Only HORIZONTAL angular separation is used (X-axis only). Vertical
    offset (Y-axis) is ignored. This is a deliberate simplification —
    full 2D angular separation is a possible future improvement but adds
    complexity for likely small accuracy gain in typical room layouts where
    person and object are roughly at the same height in the frame.

Compounding uncertainty:
    This estimate stacks three independent sources of error:
      1. d_person error (MiDaS depth, no per-session calibration for person)
      2. d_object error (known-size ±15-25% uncalibrated, or MiDaS depth)
      3. theta error (depends on webcam actual FOV vs assumed 65°)
    Final error is typically LARGER than individual errors. See the
    verification report for measured compounding factors.
"""

import logging
import math

from django.conf import settings

logger = logging.getLogger("vision.person_distance")

# Maximum plausible person-to-object distance in a normal indoor room.
# Beyond this the Law-of-Cosines result is treated as unreliable.
MAX_PLAUSIBLE_P2O_DISTANCE_M = 20.0

# Minimum plausible result (anything less than this in a room is likely a
# depth or FOV error rather than a real measurement).
MIN_PLAUSIBLE_P2O_DISTANCE_M = 0.05


def compute_angular_separation_deg(center_x_a: float, center_x_b: float, frame_width: int) -> float:
    """
    Compute horizontal angular separation (degrees) between two detected
    objects given their bounding-box center X coordinates and the frame width.

    Formula:
        angular_offset_from_center = ((center_x / frame_width) - 0.5) * FOV_DEG
        separation = abs(offset_a - offset_b)

    Note: uses settings.CAMERA_HORIZONTAL_FOV_DEG (default 65°, clearly
    flagged in settings as an assumption not a measured value).

    Simplification: horizontal separation only. Vertical/Y-axis offset is
    ignored. Full 2D angular separation is a possible future improvement.

    Args:
        center_x_a: horizontal center pixel of object A (e.g. person)
        center_x_b: horizontal center pixel of object B (e.g. laptop)
        frame_width: total frame width in pixels

    Returns:
        angular separation in degrees (always non-negative)
    """
    if frame_width <= 0:
        return 0.0

    fov_deg = getattr(settings, "CAMERA_HORIZONTAL_FOV_DEG", 65.0)
    offset_a = ((center_x_a / frame_width) - 0.5) * fov_deg
    offset_b = ((center_x_b / frame_width) - 0.5) * fov_deg
    return abs(offset_a - offset_b)


def estimate_person_to_object_distance(
    d_person_m: float | None,
    d_object_m: float | None,
    person_center_x: float,
    object_center_x: float,
    frame_width: int,
) -> dict:
    """
    Estimate the straight-line distance between a detected person and a
    target object, both visible in the same frame, using the Law of Cosines.

    Args:
        d_person_m:     camera-to-person distance in metres (from depth estimation)
        d_object_m:     camera-to-object distance in metres (from distance_estimator)
        person_center_x: horizontal centre-pixel of the person bounding box
        object_center_x: horizontal centre-pixel of the object bounding box
        frame_width:     total frame width in pixels

    Returns a dict:
        {
            "distance_m": float | None,    # rounded to 1dp; None if unavailable
            "label": str,                  # human-readable
            "angle_deg": float | None,     # angular separation used
            "d_person_m": float | None,    # cam→person distance used
            "d_object_m": float | None,    # cam→object distance used
            "source": "law_of_cosines" | "unavailable",
            "unavailable_reason": str | None,
        }
    """
    # ── Guard: both distances must be available ──────────────────────────────
    if d_person_m is None or d_object_m is None:
        reason = (
            "person distance unavailable" if d_person_m is None
            else "object distance unavailable"
        )
        return _unavailable(reason, d_person_m, d_object_m, None)

    if d_person_m <= 0 or d_object_m <= 0:
        return _unavailable("non-positive input distance", d_person_m, d_object_m, None)

    # ── Compute angular separation ────────────────────────────────────────────
    angle_deg = compute_angular_separation_deg(person_center_x, object_center_x, frame_width)
    angle_rad = math.radians(angle_deg)

    # ── Law of Cosines: c² = a² + b² - 2ab·cos(C) ────────────────────────────
    cos_angle = math.cos(angle_rad)
    discriminant = (
        d_person_m ** 2
        + d_object_m ** 2
        - 2.0 * d_person_m * d_object_m * cos_angle
    )

    # ── Plausibility guard ────────────────────────────────────────────────────
    if discriminant < 0 or math.isnan(discriminant) or math.isinf(discriminant):
        # Should not happen with valid positive distances and angle ∈ [0°,180°],
        # but guard defensively against floating-point edge cases.
        return _unavailable(
            f"non-physical result (discriminant={discriminant:.4f})",
            d_person_m, d_object_m, angle_deg,
        )

    p2o = math.sqrt(discriminant)

    if p2o > MAX_PLAUSIBLE_P2O_DISTANCE_M or p2o < MIN_PLAUSIBLE_P2O_DISTANCE_M:
        return _unavailable(
            f"implausible result ({p2o:.2f}m — outside [{MIN_PLAUSIBLE_P2O_DISTANCE_M}m, "
            f"{MAX_PLAUSIBLE_P2O_DISTANCE_M}m] plausible range)",
            d_person_m, d_object_m, angle_deg,
        )

    rounded = round(p2o, 1)
    return {
        "distance_m": rounded,
        "label": f"approximately {rounded} meters",
        "angle_deg": round(angle_deg, 1),
        "d_person_m": round(d_person_m, 1),
        "d_object_m": round(d_object_m, 1),
        "source": "law_of_cosines",
        "unavailable_reason": None,
    }


def _unavailable(reason: str, d_person_m, d_object_m, angle_deg) -> dict:
    logger.debug("person-to-object distance unavailable: %s", reason)
    return {
        "distance_m": None,
        "label": "person-to-object distance unavailable",
        "angle_deg": angle_deg,
        "d_person_m": d_person_m,
        "d_object_m": d_object_m,
        "source": "unavailable",
        "unavailable_reason": reason,
    }


def determine_movement_trend(history) -> str:
    """
    Given a sequence of recent person-to-object distances (e.g. list or deque of floats),
    determine if the person is moving toward, moving away, or stable.

    Returns:
        "moving_toward" | "moving_away" | "stable"
    """
    if not history or len(history) < 2:
        return "stable"

    recent = list(history)
    first_half = recent[: max(1, len(recent) // 2)]
    first_half_avg = sum(first_half) / len(first_half)
    latest = recent[-1]
    diff = latest - first_half_avg

    if diff < -0.15:
        return "moving_toward"
    elif diff > 0.15:
        return "moving_away"
    return "stable"

