from rest_framework import serializers

class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField(required=True)
    job_description = serializers.CharField(required=False, allow_blank=True)