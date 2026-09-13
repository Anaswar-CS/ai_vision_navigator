from django.apps import AppConfig


class DashboardConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "dashboard"

    def ready(self):
        # Pre-warm AI models in a background thread at Django startup
        # so first camera load has zero initialization lag.
        import threading
        def _prewarm():
            try:
                from vision.detector import ObjectDetector
                from vision.depth_estimator import DepthEstimator
                ObjectDetector.get_instance()
                DepthEstimator.get_instance()
            except Exception:
                pass

        threading.Thread(target=_prewarm, daemon=True).start()
