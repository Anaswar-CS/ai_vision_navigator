"""
Django settings for AI Vision Navigator.

Tuned for the target client hardware:
    Intel Core i5, 8 GB RAM, SSD, no dedicated GPU (Windows 10/11).

All AI / performance related values are configurable via environment
variables (see .env.example) so the app can be tuned without touching code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present (falls back to safe defaults otherwise)
load_dotenv(BASE_DIR / ".env")


def _bool(env_key, default=False):
    return os.environ.get(env_key, str(default)).strip().lower() in ("1", "true", "yes", "on")


def _float(env_key, default):
    try:
        return float(os.environ.get(env_key, default))
    except (TypeError, ValueError):
        return default


def _int(env_key, default):
    try:
        return int(os.environ.get(env_key, default))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------
# Core Django settings
# --------------------------------------------------------------------------
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-insecure-secret-key")
DEBUG = _bool("DEBUG", True)

ALLOWED_HOSTS = [
    h.strip() for h in os.environ.get("ALLOWED_HOSTS", "127.0.0.1,localhost,testserver,*").split(",") if h.strip()
]

INSTALLED_APPS = [
    # 'daphne' must be listed first so that `manage.py runserver` transparently
    # becomes an ASGI (WebSocket-capable) server during development, per the
    # official Django Channels pattern -- no separate process needed to test
    # the live camera page locally.
    "daphne",
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "rest_framework",
    "channels",
    "dashboard",
    "api",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "config.context_processors.app_settings",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

# --------------------------------------------------------------------------
# Database - SQLite only (no video/image data stored, only small metadata)
# --------------------------------------------------------------------------
DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATICFILES_DIRS = [BASE_DIR / "static"]
STATIC_ROOT = BASE_DIR / "staticfiles"

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# --------------------------------------------------------------------------
# Django REST Framework
# --------------------------------------------------------------------------
REST_FRAMEWORK = {
    "DEFAULT_RENDERER_CLASSES": ["rest_framework.renderers.JSONRenderer"],
    "DEFAULT_PARSER_CLASSES": ["rest_framework.parsers.JSONParser"],
}

# --------------------------------------------------------------------------
# Channels (WebSockets) - in-memory layer only.
# No Redis dependency: keeps RAM/CPU footprint low for a single-machine,
# single-user deployment on an 8 GB laptop.
# --------------------------------------------------------------------------
CHANNEL_LAYERS = {
    "default": {
        "BACKEND": "channels.layers.InMemoryChannelLayer",
    }
}

# --------------------------------------------------------------------------
# AI Vision Navigator — application configuration
# (Section 5, 10, 12, 26, 30 of the spec)
# --------------------------------------------------------------------------

# Frame sampling: how many AI frames per second the pipeline targets.
AI_PROCESSING_FPS = _int("AI_PROCESSING_FPS", 6)
ALLOWED_AI_FPS = [4, 6, 8]

# Default YOLO confidence threshold fallback
CONFIDENCE_THRESHOLD = _float("CONFIDENCE_THRESHOLD", 0.50)

# Lower threshold for custom-model vetoing of competing pretrained detections
VETO_THRESHOLD = _float("VETO_THRESHOLD", 0.25)

# Per-class confidence thresholds tailored to specific model accuracy & object scale
CLASS_CONFIDENCE_THRESHOLDS = {
    "pen": _float("CONFIDENCE_PEN", 0.38),
    "pencil": _float("CONFIDENCE_PENCIL", 0.42),
    "spectacles": _float("CONFIDENCE_SPECTACLES", 0.50),
    "medicine_box": _float("CONFIDENCE_MEDICINE_BOX", 0.42),
    "instrumentation_box": _float("CONFIDENCE_INSTRUMENTATION_BOX", 0.40),
    "computer": _float("CONFIDENCE_COMPUTER", 0.50),
    "computer_tower": _float("CONFIDENCE_COMPUTER_TOWER", 0.45),
    "computer_monitor": _float("CONFIDENCE_COMPUTER_MONITOR", 0.45),
    "laptop": _float("CONFIDENCE_LAPTOP", 0.38),
    "mobile_phone": _float("CONFIDENCE_MOBILE_PHONE", 0.38),
    "water_bottle": _float("CONFIDENCE_WATER_BOTTLE", 0.38),
    "notebook": _float("CONFIDENCE_NOTEBOOK", 0.38),
    "bag": _float("CONFIDENCE_BAG", 0.38),
    "table": _float("CONFIDENCE_TABLE", 0.32),
}

# Camera resolution (informational — the browser enforces this via getUserMedia)
CAMERA_WIDTH = _int("CAMERA_WIDTH", 640)
CAMERA_HEIGHT = _int("CAMERA_HEIGHT", 480)

# Model paths
CUSTOM_MODEL_PATH = str(BASE_DIR / os.environ.get("CUSTOM_MODEL_PATH", "models/custom/best.pt"))
PRETRAINED_MODEL_NAME = os.environ.get("PRETRAINED_MODEL_NAME", "yolov8n.pt")

# Left / center / right thresholds (Section 12)
LEFT_THRESHOLD = _float("LEFT_THRESHOLD", 0.35)
RIGHT_THRESHOLD = _float("RIGHT_THRESHOLD", 0.65)

# Voice feedback cooldown in seconds (Section 26)
VOICE_COOLDOWN = _float("VOICE_COOLDOWN", 1.5)

# Distance smoothing factor (Section 47) — exponential smoothing alpha
DISTANCE_SMOOTHING_ALPHA = _float("DISTANCE_SMOOTHING_ALPHA", 0.4)

# Number of consistent frames required before declaring a movement trend (Section 46)
NAVIGATION_CONSISTENCY_FRAMES = _int("NAVIGATION_CONSISTENCY_FRAMES", 3)

# Target-lost grace period in seconds (Section 27)
TARGET_LOST_GRACE_SECONDS = _float("TARGET_LOST_GRACE_SECONDS", 4.0)

# Performance mode: low | balanced | high (Section 30)
PERFORMANCE_MODE = os.environ.get("PERFORMANCE_MODE", "balanced").strip().lower()

PERFORMANCE_PROFILES = {
    "low": {"ai_fps": 4, "depth_every_n_ai_frames": 3, "yolo_imgsz": 384},
    "balanced": {"ai_fps": 6, "depth_every_n_ai_frames": 2, "yolo_imgsz": 480},
    "high": {"ai_fps": 8, "depth_every_n_ai_frames": 2, "yolo_imgsz": 640},
}
if PERFORMANCE_MODE not in PERFORMANCE_PROFILES:
    PERFORMANCE_MODE = "balanced"

# Auto performance backoff (Section 31)
INFERENCE_SLOW_THRESHOLD_MS = _float("INFERENCE_SLOW_THRESHOLD_MS", 300.0)

# Force CPU (Section 39) — set to False only if you have a working CUDA setup
FORCE_CPU = _bool("FORCE_CPU", True)

# Known physical object widths in meters, used as a fallback distance
# estimation method when depth-model output is unavailable (Section 17).
KNOWN_OBJECT_WIDTHS_M = {
    "mobile_phone": 0.075,
    "laptop": 0.32,
    "water_bottle": 0.07,
    "notebook": 0.21,
    "spectacles": 0.14,
    "pen": 0.014,
    "pencil": 0.008,
    "bag": 0.30,
    "medicine_box": 0.09,
    "instrumentation_box": 0.25,
    "table": 1.0,
    # computer_tower: average of 0.28m mini-tower and 0.40m standard tower: (0.28 + 0.40) / 2 = 0.34m
    "computer_tower": 0.34,
    # computer_monitor: standard 24"-27" monitor width: 0.55m
    "computer_monitor": 0.55,
    "computer": 0.34,
}

# Approximate focal length in pixels for a typical laptop webcam at 640x480.
# Overridden per-machine by the calibration page (Section 16).
DEFAULT_FOCAL_LENGTH_PX = _float("DEFAULT_FOCAL_LENGTH_PX", 600.0)

# Distance categories in meters (Section 18)
DISTANCE_CATEGORIES = [
    ("VERY CLOSE", 0.0, 0.75),
    ("NEAR", 0.75, 1.5),
    ("MEDIUM", 1.5, 3.0),
    ("FAR", 3.0, float("inf")),
]

# --------------------------------------------------------------------------
# Person-to-object distance estimation (third-party camera scenario)
# --------------------------------------------------------------------------

# Horizontal field of view of the camera in degrees, used to compute the
# angular separation between a detected person and a target object when
# both are visible in the same frame (Law-of-Cosines triangulation).
#
# [ASSUMPTION — NOT A MEASURED VALUE]: Typical consumer laptop webcams
# range from 60° to 78° horizontal FOV. 65° is a reasonable central
# estimate. For improved accuracy, measure your specific webcam's FOV
# (e.g. using a known-width ruler at a known distance) and override here
# or via the CAMERA_HORIZONTAL_FOV_DEG environment variable.
CAMERA_HORIZONTAL_FOV_DEG = _float("CAMERA_HORIZONTAL_FOV_DEG", 65.0)
