from rest_framework import serializers

from dashboard.models import CameraCalibration


class VoiceCommandSerializer(serializers.Serializer):
    text = serializers.CharField(allow_blank=False, max_length=500)


class StartTrackingSerializer(serializers.Serializer):
    object = serializers.CharField(allow_blank=False, max_length=100)


class CalibrationSerializer(serializers.ModelSerializer):
    class Meta:
        model = CameraCalibration
        fields = ["id", "reference_distance", "depth_value", "created_at"]
        read_only_fields = ["id", "created_at"]

    def validate_reference_distance(self, value):
        if value <= 0:
            raise serializers.ValidationError("reference_distance must be positive.")
        return value

    def validate_depth_value(self, value):
        if value <= 0:
            raise serializers.ValidationError("depth_value must be positive.")
        return value


class DetectRequestSerializer(serializers.Serializer):
    """Accepts either a base64-encoded JPEG/PNG string or a multipart file
    upload under the 'image' key (handled separately in the view since DRF
    JSONParser won't carry file uploads)."""

    image_base64 = serializers.CharField(required=False, allow_blank=True)
