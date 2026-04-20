from django.db import models

class DocumentResult(models.Model):
    file = models.FileField(upload_to="documents/")
    document_type = models.CharField(max_length=100)
    confidence = models.FloatField()
    entities = models.JSONField()
    amount = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
