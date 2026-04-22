from django.urls import path
from .views import DocumentUploadView, extractor

urlpatterns = [
    path("upload/", DocumentUploadView.as_view(), name="document-upload"),
    path("upload/extract/", extractor, name="document-extractor"),
]