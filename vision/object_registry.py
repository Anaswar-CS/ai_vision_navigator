"""
Central registry for supported object classes and their aliases.

Every module (detector, voice command parser, navigator, UI) should route
object names through `normalize_object_name()` so that "phone", "mobile",
and "smartphone" are always treated as the same canonical object:
"mobile_phone".
"""

# Canonical object classes supported by the application (Section 7)
SUPPORTED_OBJECTS = [
    "laptop",
    "computer",
    "computer_tower",
    "computer_monitor",
    "mobile_phone",
    "table",
    "pen",
    "pencil",
    "spectacles",
    "water_bottle",
    "medicine_box",
    "notebook",
    "bag",
    "instrumentation_box",
]

# Friendly display names for UI / voice output
DISPLAY_NAMES = {
    "laptop": "laptop",
    "computer": "computer",
    "computer_tower": "computer tower",
    "computer_monitor": "computer monitor",
    "mobile_phone": "mobile phone",
    "table": "table",
    "pen": "pen",
    "pencil": "pencil",
    "spectacles": "spectacles",
    "water_bottle": "water bottle",
    "medicine_box": "medicine box",
    "notebook": "notebook",
    "bag": "bag",
    "instrumentation_box": "instrumentation box",
}

# Alias map -> canonical name (Section 7)
OBJECT_ALIASES = {
    # mobile phone
    "phone": "mobile_phone",
    "mobile": "mobile_phone",
    "mobile phone": "mobile_phone",
    "smartphone": "mobile_phone",
    "cell phone": "mobile_phone",
    "cellphone": "mobile_phone",

    # notebook / book
    "book": "notebook",
    "textbook": "notebook",
    "notebook": "notebook",
    "diary": "notebook",

    # spectacles
    "glasses": "spectacles",
    "specs": "spectacles",
    "spectacle": "spectacles",
    "eyeglasses": "spectacles",

    # bag
    "school bag": "bag",
    "college bag": "bag",
    "backpack": "bag",
    "handbag": "bag",
    "rucksack": "bag",

    # bottle
    "bottle": "water_bottle",
    "water bottle": "water_bottle",

    # laptop / computer / tower / monitor
    "laptop": "laptop",
    "notebook computer": "laptop",
    "computer": "computer",
    "pc": "computer",
    "desktop": "computer",
    "computer tower": "computer_tower",
    "tower": "computer_tower",
    "cpu tower": "computer_tower",
    "pc tower": "computer_tower",
    "computer monitor": "computer_monitor",
    "monitor": "computer_monitor",
    "screen": "computer_monitor",
    "display": "computer_monitor",

    # writing instruments
    "pen": "pen",
    "pencil": "pencil",

    # boxes
    "medicine box": "medicine_box",
    "medicine": "medicine_box",
    "medkit": "medicine_box",
    "pill box": "medicine_box",
    "instrumentation box": "instrumentation_box",
    "instrument box": "instrumentation_box",
    "toolbox": "instrumentation_box",

    # table
    "table": "table",
    "desk": "table",
}

# Mapping from the classes a *standard pretrained COCO YOLO model* actually
# knows, onto our canonical vocabulary (Section 8/9). COCO does not contain
# "spectacles", "pencil", "medicine box", "instrumentation box" or bag
# subtypes — those require a custom-trained model (models/custom/best.pt).
COCO_TO_CANONICAL = {
    "cell phone": "mobile_phone",
    "cellphone": "mobile_phone",
    "mobile phone": "mobile_phone",
    "laptop": "laptop",
    "tv": "computer_monitor",          # pretrained COCO tv / tvmonitor maps to computer_monitor
    "tvmonitor": "computer_monitor",
    "book": "notebook",
    "backpack": "bag",
    "handbag": "bag",
    "suitcase": "bag",
    "bag": "bag",
    "bottle": "water_bottle",
    "dining table": "table",
    "dining_table": "table",
    "table": "table",
    "desk": "table",
}

# Classes that pretrained COCO YOLO simply cannot recognize.
# The app must clearly report that these require the custom model.
CUSTOM_ONLY_OBJECTS = {"spectacles", "pencil", "medicine_box", "instrumentation_box", "pen", "computer_tower"}


def normalize_object_name(raw_name: str) -> str:
    """Map any raw/alias object name to its canonical form.

    Falls back to a lowercase/underscored version of the input if no
    alias mapping exists, so unknown classes degrade gracefully instead
    of crashing.
    """
    if not raw_name:
        return ""
    key = raw_name.strip().lower()
    if key in OBJECT_ALIASES:
        return OBJECT_ALIASES[key]
    if key in SUPPORTED_OBJECTS:
        return key
    return key.replace(" ", "_")


def display_name(canonical_name: str) -> str:
    return DISPLAY_NAMES.get(canonical_name, canonical_name.replace("_", " "))


def is_custom_only(canonical_name: str) -> bool:
    return canonical_name in CUSTOM_ONLY_OBJECTS
