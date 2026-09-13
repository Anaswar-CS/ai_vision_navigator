from django.urls import path

from . import views

app_name = "api"

urlpatterns = [
    path("detect/", views.detect, name="detect"),
    path("voice-command/", views.voice_command, name="voice_command"),
    path("start-tracking/", views.start_tracking, name="start_tracking"),
    path("stop-tracking/", views.stop_tracking, name="stop_tracking"),
    path("status/", views.api_status, name="status"),
    path("calibration/", views.calibration, name="calibration"),
]
