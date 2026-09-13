from django.db import models


class CameraCalibration(models.Model):
    """A single calibration point captured on the /calibrate/ page
    (Section 16, 36). Only small numeric metadata is stored -- never an
    image or video frame."""

    reference_distance = models.FloatField(help_text="Known real-world distance in meters")
    depth_value = models.FloatField(help_text="Raw MiDaS depth value observed at that distance")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["reference_distance"]

    def __str__(self):
        return f"{self.reference_distance} m -> depth {self.depth_value:.3f}"


class DetectionLog(models.Model):
    """Optional lightweight log of detections for debugging/analytics
    (Section 36). No images or video are ever stored here."""

    object_name = models.CharField(max_length=64)
    confidence = models.FloatField()
    distance = models.FloatField(null=True, blank=True)
    direction = models.CharField(max_length=16, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-timestamp"]

    def __str__(self):
        return f"{self.object_name} ({self.confidence:.2f}) @ {self.timestamp}"
