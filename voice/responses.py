"""
voice/responses.py

Generates the natural-language strings spoken via the browser's
speechSynthesis API and displayed in the UI (Section 21, 45).
Deliberately simple template-based generation -- no external LLM call, so
it works fully offline and with near-zero latency/CPU cost.
"""

from vision.object_registry import display_name

_DIRECTION_PHRASES = {
    "left": "on your left",
    "right": "on your right",
    "center": "directly in front of you",
}


def found_object_message(canonical_class, direction, distance_label):
    name = display_name(canonical_class)
    where = _DIRECTION_PHRASES.get(direction, "nearby")
    return f"I found your {name} {where}, {distance_label}."


def not_found_message(canonical_class):
    name = display_name(canonical_class)
    return f"I can't see your {name} right now. Try moving the camera around."


def low_confidence_message(canonical_class):
    name = display_name(canonical_class)
    return f"I think I see your {name}, but I'm not fully sure. Object detected with low confidence."


def moving_toward_message(canonical_class):
    name = display_name(canonical_class)
    return f"You are moving toward the {name}."


def moving_away_message(canonical_class):
    name = display_name(canonical_class)
    return f"You are moving away from the {name}."


def correct_direction_message():
    return "Correct direction. Continue forward."


def turn_message(new_direction):
    if new_direction == "left":
        return "The object has moved to your left. Turn slightly left."
    if new_direction == "right":
        return "The object has moved to your right. Turn slightly right."
    return "The object is now in front of you."


def moved_past_message(new_direction):
    side = "left" if new_direction == "left" else "right"
    return f"You moved past the object. Turn slightly {side}."


def target_lost_message(canonical_class):
    name = display_name(canonical_class)
    return f"I can no longer see the {name}. Please move the camera around."


def target_reacquired_message(canonical_class, direction, distance_label):
    name = display_name(canonical_class)
    where = _DIRECTION_PHRASES.get(direction, "nearby")
    return f"Found it again. Your {name} is {where}, {distance_label}."


def custom_model_required_message(canonical_class):
    name = display_name(canonical_class)
    return (
        f"The {name} requires a custom-trained model for reliable detection. "
        f"The current pretrained model may not recognize it well."
    )


def multiple_objects_message(canonical_class, count):
    name = display_name(canonical_class)
    plural = "s" if count != 1 else ""
    return f"{count} {name}{plural} detected. Tracking the nearest one."
