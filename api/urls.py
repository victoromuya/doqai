from django.urls import path

from .views import DocumentUploadView, extractor, AskDocumentView

urlpatterns = [
    path("upload/", DocumentUploadView.as_view(), name="document-upload"),
    path("upload/extract/", extractor, name="document-extractor"),
    path("query/", AskDocumentView.as_view(), name="rag-query"),
]