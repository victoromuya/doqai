from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from .serializers import DocumentUploadSerializer
from django.conf import settings
import os
from .tasks import process_document
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi



class DocumentUploadView(APIView):
    parser_classes = [MultiPartParser]

    @swagger_auto_schema(
        operation_description="Upload a document file (PDF, image, text) for AI processing and analysis",
        consumes=['multipart/form-data'],
        request_body=DocumentUploadSerializer,
        responses={
            202: openapi.Response("Document processing started", openapi.Schema(type=openapi.TYPE_OBJECT, properties={
                'message': openapi.Schema(type=openapi.TYPE_STRING),
                'document_type': openapi.Schema(type=openapi.TYPE_STRING),
                'text': openapi.Schema(type=openapi.TYPE_STRING),
            })),
            400: "Validation error"
        }
    )
    def post(self, request):
        try:
            print(request.FILES)
            serializer = DocumentUploadSerializer(data=request.data)
            if serializer.is_valid():
                file_obj = serializer.validated_data['file']

                # Ensure uploads folder exists
                upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
                os.makedirs(upload_dir, exist_ok=True)

                # Avoid overwriting files with the same name
                base_name, extension = os.path.splitext(file_obj.name)
                file_path = os.path.join(upload_dir, file_obj.name)
                counter = 1
                while os.path.exists(file_path):
                    file_path = os.path.join(upload_dir, f"{base_name}_{counter}{extension}")
                    counter += 1

                with open(file_path, 'wb+') as f:
                    for chunk in file_obj.chunks():
                        f.write(chunk)

                # Trigger async processing
                task = process_document(file_path)
            

                return Response({
                    "message": "Document processed successfully",
                    "document_type": task.get("document_type"),
                    "confidence": task.get("confidence"),
                    "entities": task.get("entities"),
                    "amount": task.get("amount"),
                    "text": task.get("text"),
                }, status=status.HTTP_202_ACCEPTED)

        finally:
        # Step 2: Delete the file regardless of success or error
            if os.path.exists(file_path):
                os.remove(file_path)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)