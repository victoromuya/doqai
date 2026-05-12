import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser
from .serializers import DocumentUploadSerializer, DocumentQuerySerializer
from django.conf import settings
import os
from .tasks import process_document, rewrite_cv_section, only_extract
from drf_yasg.utils import swagger_auto_schema
from drf_yasg import openapi
from rest_framework.decorators import api_view
from rag.rag_service import answer_query

import time

logger = logging.getLogger(__name__)


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
        # Initialize file_path as None to prevent crashes in the 'finally' block
        file_path = None
        
        try:
            serializer = DocumentUploadSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            file_obj = serializer.validated_data['file']
            job_description = serializer.validated_data.get("job_description", "")

            # 1. Save file temporarily
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Use a hex prefix to ensure the filename is unique and safe
            safe_filename = f"{os.urandom(4).hex()}_{file_obj.name}"
            file_path = os.path.join(upload_dir, safe_filename)

            with open(file_path, 'wb+') as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)

            # 2. Extract and Classify
            task_result = process_document(file_path)

            # 3. Curate Error Messages for the User
            if isinstance(task_result, dict) and "error" in task_result:
                error_msg = str(task_result.get("message", ""))
                
                # Specifically curate the 3-page limit error
                if "maximum page limit of 3" in error_msg.lower():
                    return Response({
                        "error": "Document Too Long",
                        "message": "To ensure fast and accurate processing, please upload a document with 3 pages or fewer.",
                        "code": "PAGE_LIMIT_EXCEEDED"
                    }, status=status.HTTP_400_BAD_REQUEST)
                
                # Handle other general extraction/OCR errors
                return Response({
                    "error": "Processing Failed",
                    "message": "We couldn't read this document. Please ensure the file is not password protected and try again.",
                    "details": error_msg
                }, status=status.HTTP_400_BAD_REQUEST)

            # 4. Trigger CV rewriting ONLY if it's a resume
            rewrite_cv = None
            doc_type = task_result.get("document_type", "").lower()
            
            if "resume" in doc_type or "cv" in doc_type:
                if job_description:
                    # Use the stronger model for rewriting
                    rewrite_cv = rewrite_cv_section(task_result.get("text"), job_description)
                else:
                    rewrite_cv = "Job description missing. Please provide one to tailor your CV."

            else:
                # FIX: Explicitly tell the user why no rewrite was performed
                rewrite_cv = "CV rewriting is only available for documents classified as Resumes. This document was identified as a " + doc_type + "."

            # 5. Return Successful Response
            return Response({
                "message": "Document processed successfully",
                "document_type": task_result.get("document_type"),
                "confidence": task_result.get("confidence"),
                "entities": task_result.get("entities"),
                "amount": task_result.get("amount"),
                "text": task_result.get("text"),
                "rewritten_cv": rewrite_cv,
            }, status=status.HTTP_200_OK)

        except Exception as e:
            print(f"Internal System Error: {e}")
            return Response({
                "error": "System Error", 
                "message": "An unexpected error occurred on our server. Please try again later."
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Cleanup: Ensure file is deleted from Render storage even if the process failed
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    print(f"Cleanup failed for {file_path}: {cleanup_error}")


@swagger_auto_schema(
        operation_description="Upload a document file (PDF, image, text) for text extraction",
        method='post', 
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
@api_view(['POST'])
def extractor(request):
    if request.method == 'POST':
        file_path = None
        try:
            serializer = DocumentUploadSerializer(data=request.data)
            if not serializer.is_valid():
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            file_obj = serializer.validated_data['file']

            # 1. Save file temporarily
            upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
            os.makedirs(upload_dir, exist_ok=True)
            
            # Use a hex prefix to ensure the filename is unique and safe
            safe_filename = f"{os.urandom(4).hex()}_{file_obj.name}"
            file_path = os.path.join(upload_dir, safe_filename)

            with open(file_path, 'wb+') as f:
                for chunk in file_obj.chunks():
                    f.write(chunk)
        
            
            extract_result = only_extract(file_path)

            return Response({
                    "message": "Document processed successfully",
                    "text": extract_result.get("text"),
                
                }, status=status.HTTP_200_OK)

        except Exception as e:
                print(f"Internal System Error: {e}")
                return Response({
                    "error": "System Error", 
                    "message": "An unexpected error occurred on our server. Please try again later."
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        finally:
            # Cleanup: Ensure file is deleted from Render storage even if the process failed
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as cleanup_error:
                    print(f"Cleanup failed for {file_path}: {cleanup_error}")

    return Response({"error": "Method not allowed"}, status=status.HTTP_405_METHOD_NOT_ALLOWED)



class AskDocumentView(APIView):

    @swagger_auto_schema(
        tags=["RAG Query"],
        operation_description="""
        Query uploaded documents using Retrieval-Augmented Generation (RAG).

        The system performs:
        - Semantic embedding generation
        - Vector similarity retrieval
        - Context grounding
        - AI response generation

        Only information retrieved from indexed documents
        is used to generate the response.
        """,
        request_body=DocumentQuerySerializer,
        responses={
            200: openapi.Response(
                description="Successful RAG response",
                schema=openapi.Schema(
                    type=openapi.TYPE_OBJECT,
                    properties={
                        "answer": openapi.Schema(
                            type=openapi.TYPE_STRING,
                            description="AI-generated grounded answer"
                        ),
                        "sources": openapi.Schema(
                            type=openapi.TYPE_ARRAY,
                            items=openapi.Items(
                                type=openapi.TYPE_STRING
                            ),
                            description="Retrieved document chunks used as context"
                        )
                    }
                ),
                examples={
                    "application/json": {
                        "answer": "The invoice specifies that payment is due within 30 days of issue.",
                        "sources": [
                            "Payment terms: Net 30 days from invoice date.",
                            "Late payments may incur additional charges."
                        ]
                    }
                }
            ),

            400: openapi.Response(
                description="Validation Error",
                examples={
                    "application/json": {
                        "query": [
                            "This field is required."
                        ]
                    }
                }
            ),

            500: openapi.Response(
                description="Internal Server Error",
                examples={
                    "application/json": {
                        "error": "System Error",
                        "message": "An unexpected error occurred."
                    }
                }
            )
        }
    )
    def post(self, request):

        serializer = DocumentQuerySerializer(data=request.data)

        if not serializer.is_valid():
            return Response(
                serializer.errors,
                status=status.HTTP_400_BAD_REQUEST
            )

        query = serializer.validated_data["query"]


        try:
            start_time = time.time()
            result = answer_query(query)

            # Safety check: ensure result is valid
            if not result:
                return Response(
                    {
                        "error": "EmptyResponse",
                        "message": "No result returned from RAG pipeline."
                    },
                    status=status.HTTP_502_BAD_GATEWAY
                )

            response_time = round(time.time() - start_time, 3)

            return Response(
                {
                    "data": result,
                    "meta": {
                        "response_time_seconds": response_time
                    },
                  
                },
                status=status.HTTP_200_OK
            )

        except ValueError as ve:
            logger.warning(f"Validation error: {ve}")
            return Response(
                {
                    "error": "ValidationError",
                    "message": str(ve)
                },
                status=status.HTTP_400_BAD_REQUEST
            )

        except Exception as e:
            logger.exception(f"RAG Query Error: {e}")
            return Response(
                {
                    "error": "InternalServerError",
                    "message": "Failed to process query."
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )