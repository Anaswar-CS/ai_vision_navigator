from django.contrib import admin

from .models import CameraCalibration, DetectionLog


@admin.register(CameraCalibration)
class CameraCalibrationAdmin(admin.ModelAdmin):
    list_display = ("reference_distance", "depth_value", "created_at")
    ordering = ("reference_distance",)


@admin.register(DetectionLog)
class DetectionLogAdmin(admin.ModelAdmin):
    list_display = ("object_name", "confidence", "distance", "direction", "timestamp")
    list_filter = ("object_name",)
    ordering = ("-timestamp",)
