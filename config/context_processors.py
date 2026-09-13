from django.conf import settings


def app_settings(request):
    """Expose a curated subset of settings to all templates."""
    return {
        "AI_PROCESSING_FPS": settings.AI_PROCESSING_FPS,
        "CAMERA_WIDTH": settings.CAMERA_WIDTH,
        "CAMERA_HEIGHT": settings.CAMERA_HEIGHT,
        "PERFORMANCE_MODE": settings.PERFORMANCE_MODE,
    }
