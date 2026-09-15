"""
voice/command_parser.py

Parses recognized speech text into a structured command, entirely locally
(no external LLM dependency -- Section 20). Works on lowercased text from
the browser's SpeechRecognition API.
"""

import re

from vision.object_registry import OBJECT_ALIASES, SUPPORTED_OBJECTS, normalize_object_name

# Longest-alias-first so "school bag" matches before generic "bag".
_ALIAS_KEYS_BY_LENGTH = sorted(OBJECT_ALIASES.keys(), key=len, reverse=True)

_STOP_PATTERNS = [
    r"\bstop tracking\b",
    r"\bstop\b",
    r"\bcancel tracking\b",
]

_START_PATTERNS = [
    r"\bstart tracking\b",
    r"\btrack (the |my )?",
]

_LIST_PATTERNS = [
    r"\bwhat objects can you see\b",
    r"\bwhat do you see\b",
    r"\blist objects\b",
]

_FIND_PATTERNS = [
    r"\bwhere is\b",
    r"\bwhere('?s| is)\b",
    r"\bwhere are\b",
    r"\bfind\b",
    r"\blocate\b",
]

# New command intent: person-to-object distance (third-party camera scenario).
# These patterns are SEPARATE from _FIND_PATTERNS — they must not interfere
# with the existing "where is my [object]" flow.
# Matched phrases (case-insensitive, after normalization):
#   "how far is the person from the laptop"
#   "distance between person and phone"
#   "how far from the person to the notebook"
#   "person to laptop distance"
#   "how close is the person to the bag"
_PERSON_DISTANCE_PATTERNS = [
    r"\bhow far (is |are )?the person (from|to)\b",
    r"\bdistance between (the )?person and\b",
    r"\bhow far from (the )?person\b",
    r"\bperson (to|from|and) .+ distance\b",
    r"\bhow close is (the )?person (to|from)\b",
    r"\bperson distance\b",
]


def _extract_object(text):
    """Find the first known object alias/name mentioned in the text."""
    for alias in _ALIAS_KEYS_BY_LENGTH:
        if re.search(r"\b" + re.escape(alias) + r"s?\b", text):
            return normalize_object_name(alias)

    for canonical in SUPPORTED_OBJECTS:
        spaced = canonical.replace("_", " ")
        if spaced in text:
            return canonical

    return None


def parse_command(text):
    """Parse a raw recognized-speech string into a structured command.

    Returns a dict:
        {"intent": "find_object" | "start_tracking" | "stop_tracking"
                   | "list_objects" | "unknown",
         "object": <canonical_name> | None,
         "raw_text": <original text>}
    """
    if not text:
        return {"intent": "unknown", "object": None, "raw_text": text}

    normalized_text = text.strip().lower()
    normalized_text = re.sub(r"[^\w\s?']", " ", normalized_text)
    normalized_text = re.sub(r"\s+", " ", normalized_text).strip()

    for pattern in _STOP_PATTERNS:
        if re.search(pattern, normalized_text):
            return {"intent": "stop_tracking", "object": None, "raw_text": text}

    for pattern in _LIST_PATTERNS:
        if re.search(pattern, normalized_text):
            return {"intent": "list_objects", "object": None, "raw_text": text}

    obj = _extract_object(normalized_text)

    # Check person-to-object distance BEFORE generic find_object patterns so
    # "how far is the person from the laptop" is never mis-routed as a
    # regular find_object command.
    for pattern in _PERSON_DISTANCE_PATTERNS:
        if re.search(pattern, normalized_text):
            return {"intent": "person_object_distance", "object": obj, "raw_text": text}

    for pattern in _START_PATTERNS:
        if re.search(pattern, normalized_text):
            return {"intent": "start_tracking", "object": obj, "raw_text": text}

    for pattern in _FIND_PATTERNS:
        if re.search(pattern, normalized_text):
            return {"intent": "find_object", "object": obj, "raw_text": text}

    # If no explicit verb matched but an object was mentioned, assume the
    # user wants to find/track it -- this keeps the parser forgiving of
    # imperfect speech-to-text transcriptions.
    if obj:
        return {"intent": "find_object", "object": obj, "raw_text": text}

    return {"intent": "unknown", "object": None, "raw_text": text}
