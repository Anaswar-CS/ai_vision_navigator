"""
vision/distance_estimator.py

Turns a bounding box (+ optional depth map) into an *approximate* distance
in meters, plus a human-friendly distance category (Section 13, 15, 18).

Combines up to three signals, in order of preference:
    1. Depth map (MiDaS) + calibration -> most accurate when available.
    2. Known physical object width + focal length geometry (Section 17).
    3. If neither is available, returns None and the caller must clearly
       communicate that distance is temporarily unavailable (Section 38).

Never reports a suspiciously precise number (e.g. "2.134728 m") -- always
rounds to one decimal place and adds a coarse category label.
"""

import logging

import numpy as np
from django.conf import settings

from vision import calibration as calib
from vision.object_registry import normalize_object_name

logger = logging.getLogger("vision.distance")


# Classes with rigid/consistent dimensions where pinhole geometry is the primary estimator
KNOWN_SIZE_CLASSES = {
    "laptop",
    "mobile_phone",
    "water_bottle",
    "computer",
    "computer_tower",
    "computer_monitor",
}


def _robust_depth_from_bbox(depth_map, bbox):
    """Take a robust statistic (median) of depth values inside the bounding
    box, ignoring the extreme upper/lower percentiles to reduce noise from
    background pixels leaking into the box edges (Section 15)."""
    if depth_map is None:
        return None

    h, w = depth_map.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None

    region = depth_map[y1:y2, x1:x2]
    if region.size == 0:
        return None

    # Trim to the central 60% of values (15th-85th percentile) to suppress
    # background/edge noise while remaining cheap to compute.
    flat = region.flatten()
    low, high = np.percentile(flat, [15, 85])
    trimmed = flat[(flat >= low) & (flat <= high)]
    if trimmed.size == 0:
        trimmed = flat

    return float(np.median(trimmed))


def _known_size_distance_m(class_name, bbox, focal_length_px):
    """distance = (real_width_m * focal_length_px) / pixel_width (Section 17)."""
    canonical = normalize_object_name(class_name) if class_name else None
    if not canonical:
        return None
    real_width_m = settings.KNOWN_OBJECT_WIDTHS_M.get(canonical)
    if not real_width_m:
        return None

    x1, y1, x2, y2 = bbox
    pixel_width = max(1.0, float(x2 - x1))
    distance_m = (real_width_m * focal_length_px) / pixel_width
    return distance_m


def distance_category(distance_m):
    if distance_m is None:
        return None
    for label, lo, hi in settings.DISTANCE_CATEGORIES:
        if lo <= distance_m < hi:
            return label
    return None


def estimate_object_distance(depth_map, bounding_box, class_name=None,
                              focal_length_px=None, calibration_k=None):
    """Main entry point with hybrid class-aware routing (Section 15, 17).

    Returns a dict:
        {
            "distance_m": float | None,       # rounded to 1 decimal
            "label": str,                      # e.g. "approximately 2 meters"
            "category": str | None,            # e.g. "MEDIUM"
            "source": "depth" | "known_size" | "unavailable",
        }
    """
    focal_length_px = focal_length_px or settings.DEFAULT_FOCAL_LENGTH_PX
    canonical = normalize_object_name(class_name) if class_name else None
    distance_m = None
    source = "unavailable"

    if canonical in KNOWN_SIZE_CLASSES:
        # 1. Primary for rigid objects: Pinhole geometry
        geom_dist = _known_size_distance_m(canonical, bounding_box, focal_length_px)
        if geom_dist is not None and 0.1 <= geom_dist <= 20.0:
            distance_m = geom_dist
            source = "known_size"
        else:
            # Fallback to category-aware MiDaS depth if geometric distance is implausible/unavailable
            depth_value = _robust_depth_from_bbox(depth_map, bounding_box)
            if depth_value is not None:
                k = calib.get_k_for_class(canonical, base_k=calibration_k)
                d = calib.depth_value_to_distance_m(depth_value, k=k)
                if d is not None and 0.1 <= d <= 20.0:
                    distance_m = d
                    source = "depth"
    else:
        # 2. Primary for variable-size / room-scale objects: Category-aware MiDaS depth
        depth_value = _robust_depth_from_bbox(depth_map, bounding_box)
        if depth_value is not None:
            k = calib.get_k_for_class(canonical, base_k=calibration_k)
            d = calib.depth_value_to_distance_m(depth_value, k=k)
            if d is not None and 0.1 <= d <= 20.0:
                distance_m = d
                source = "depth"

        # Fallback to known-size geometry if depth map unavailable or depth value is implausible
        if distance_m is None and canonical:
            geom_dist = _known_size_distance_m(canonical, bounding_box, focal_length_px)
            if geom_dist is not None and 0.1 <= geom_dist <= 20.0:
                distance_m = geom_dist
                source = "known_size"

    if distance_m is None:
        return {
            "distance_m": None,
            "raw_distance_m": None,
            "label": "distance temporarily unavailable",
            "category": None,
            "source": "unavailable",
        }

    rounded = round(distance_m, 1)
    category = distance_category(rounded)
    return {
        "distance_m": rounded,
        "raw_distance_m": float(distance_m),
        "label": f"approximately {rounded} meters",
        "category": category,
        "source": source,
    }

