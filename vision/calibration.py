"""
vision/calibration.py

A normal RGB webcam cannot measure physical distance directly. MiDaS
produces *relative inverse depth*, not meters. To turn that into an
approximate physical distance we fit a simple inverse relationship using
one or more (known_distance_m, depth_value) calibration points captured on
the /calibrate/ page (Section 16):

    depth_value ≈ k / distance_m        (simple pinhole-like inverse model)
    =>  distance_m ≈ k / depth_value

With a single calibration point we solve directly for k.
With multiple points we fit k via least squares for robustness against
noisy captures.

If no calibration data exists yet, a conservative default k is used so the
app still produces *approximate* (clearly labeled as rough) distances out
of the box.
"""

import logging

logger = logging.getLogger("vision.calibration")

# Calibrated default constants mapping relative inverse depth from MiDaS-small
# to approximate real-world meters: distance_m ≈ k / depth_value.
K_SMALL_OBJECT = 135.0
K_MEDIUM_OBJECT = 420.0
K_LARGE_OBJECT = 550.0
DEFAULT_K = 135.0

CATEGORY_K = {
    "small": K_SMALL_OBJECT,
    "medium": K_MEDIUM_OBJECT,
    "large": K_LARGE_OBJECT,
}

CLASS_TO_CATEGORY = {
    "pen": "small",
    "pencil": "small",
    "medicine_box": "small",
    "spectacles": "small",
    "notebook": "medium",
    "book": "medium",
    "bag": "medium",
    "table": "large",
    "computer": "medium",
    "computer_tower": "medium",
    "computer_monitor": "medium",
}


def fit_k_from_points(points):
    """Fit k in distance = k / depth_value from a list of
    (reference_distance_m, depth_value) tuples using least squares
    on the linearized form: depth_value = k * (1/distance_m).

    Returns the fitted k, or DEFAULT_K if fewer than 1 usable point exists.
    """
    usable = [(d, v) for (d, v) in points if d and v and d > 0 and v > 0]
    if not usable:
        return DEFAULT_K

    # depth_value = k * x, where x = 1/distance_m  -> simple least squares through origin
    numerator = 0.0
    denominator = 0.0
    for distance_m, depth_value in usable:
        x = 1.0 / distance_m
        numerator += x * depth_value
        denominator += x * x

    if denominator == 0:
        return DEFAULT_K

    k = numerator / denominator
    if k <= 0:
        return DEFAULT_K
    return k


def get_k_for_class(class_name=None, base_k=None):
    """Return category-specific k for a given class.
    If base_k is supplied (or fitted from DB), scales it proportionally for larger categories.
    """
    base = base_k if base_k is not None else get_current_k()
    if not class_name:
        return base

    from vision.object_registry import normalize_object_name
    canonical = normalize_object_name(class_name)
    category = CLASS_TO_CATEGORY.get(canonical, "small")
    target_k = CATEGORY_K.get(category, DEFAULT_K)

    if base == DEFAULT_K or abs(base - 132.0) < 5.0 or abs(base - 135.0) < 5.0:
        return target_k
    # Scale proportionally if user has custom baseline calibration points
    ratio = target_k / K_SMALL_OBJECT
    return base * ratio



def depth_value_to_distance_m(depth_value, k=None, class_name=None):
    """Convert a single (robust-statistic) MiDaS depth value into an
    approximate distance in meters."""
    if depth_value is None or depth_value <= 0:
        return None
    if k is None:
        k = get_k_for_class(class_name)
    distance_m = k / depth_value
    # Clamp to a sane physical range for an indoor desk-distance use case.
    return max(0.1, min(distance_m, 15.0))


_CACHED_K = None

def clear_k_cache():
    global _CACHED_K
    _CACHED_K = None

def get_current_k():
    """Load all stored calibration points from the DB and fit k.

    Caches the value in memory so high-frequency per-frame checks (6-8 FPS)
    never hit the database repeatedly.
    """
    global _CACHED_K
    if _CACHED_K is not None:
        return _CACHED_K

    try:
        from dashboard.models import CameraCalibration
        points = list(
            CameraCalibration.objects.values_list("reference_distance", "depth_value")
        )
        _CACHED_K = fit_k_from_points(points)
        return _CACHED_K
    except Exception:
        # DB not ready or no table exists yet -- return fallback without spamming logs
        return DEFAULT_K

