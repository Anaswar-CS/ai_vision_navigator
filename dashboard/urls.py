from django.urls import path

from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("camera/", views.camera, name="camera"),
    path("calibrate/", views.calibrate, name="calibrate"),
    path("test/", views.test_image, name="test_image"),
    path("help/", views.help_about, name="help_about"),
]
