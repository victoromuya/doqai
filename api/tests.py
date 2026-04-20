from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile
from django.urls import reverse
from rest_framework.test import APITestCase
from rest_framework import status
import os
from unittest.mock import patch, MagicMock


class DocumentUploadViewTest(APITestCase):
    def setUp(self):
        self.url = reverse('document-upload')

    @patch('api.views.process_document')
    def test_successful_document_upload(self, mock_process):
        """Test successful document upload with valid file"""
        # Mock the processing function
        mock_process.return_value = {
            "document_type": "invoice",
            "confidence": 0.95,
            "entities": {"dates": [], "money": [], "organizations": [], "persons": []},
            "text": "Sample invoice text",
            "amount": "$100.00",
        }

        # Create a simple text file for upload
        file_content = b"This is a test document content."
        uploaded_file = SimpleUploadedFile(
            "test.txt",
            file_content,
            content_type="text/plain"
        )

        data = {'file': uploaded_file}
        response = self.client.post(self.url, data, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertIn('message', response.data)
        self.assertIn('document_type', response.data)
        self.assertIn('text', response.data)
        self.assertEqual(response.data['document_type'], 'invoice')
        self.assertEqual(response.data['text'], 'Sample invoice text')

        # Verify process_document was called
        mock_process.assert_called_once()

    def test_upload_without_file(self):
        """Test upload request without file should fail validation"""
        response = self.client.post(self.url, {}, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn('file', response.data)

    @patch('api.views.process_document')
    @patch('api.tasks.extract_text')
    def test_upload_unsupported_file_type(self, mock_extract, mock_process):
        """Test upload with unsupported file type"""
        # Mock extract_text to return some text for unsupported type
        mock_extract.return_value = "Unsupported file content"
        mock_process.return_value = {
            "document_type": "unknown",
            "confidence": 0.0,
            "entities": {"dates": [], "money": [], "organizations": [], "persons": []},
            "text": "Unsupported file content",
            "amount": None,
        }

        # Create a file with unsupported extension
        uploaded_file = SimpleUploadedFile(
            "test.xyz",
            b"test content",
            content_type="application/octet-stream"
        )

        data = {'file': uploaded_file}
        response = self.client.post(self.url, data, format='multipart')

        # Should return 202 as processing completes (even for unsupported types)
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)

    @patch('api.views.process_document')
    @patch('api.tasks.extract_text')
    def test_processing_with_pdf_file(self, mock_extract, mock_process):
        """Test processing a PDF file"""
        mock_extract.return_value = "This is PDF content"
        mock_process.return_value = {
            "document_type": "contract",
            "confidence": 0.88,
            "entities": {"dates": ["2024-01-01"], "money": [], "organizations": [], "persons": []},
            "text": "This is a contract document.",
            "amount": None,
        }

        # Create a simple file for upload
        uploaded_file = SimpleUploadedFile(
            "test.pdf",
            b"fake pdf content",
            content_type="application/pdf"
        )

        data = {'file': uploaded_file}
        response = self.client.post(self.url, data, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertEqual(response.data['document_type'], 'contract')

    @patch('api.views.process_document')
    @patch('api.tasks.extract_text')
    def test_processing_with_image_file(self, mock_extract, mock_process):
        """Test processing an image file"""
        mock_extract.return_value = "Receipt for $25.99"
        mock_process.return_value = {
            "document_type": "receipt",
            "confidence": 0.92,
            "entities": {"dates": [], "money": ["$25.99"], "organizations": [], "persons": []},
            "text": "Receipt for $25.99",
            "amount": "$25.99",
        }

        # Create a simple file for upload
        uploaded_file = SimpleUploadedFile(
            "test.png",
            b"fake image content",
            content_type="image/png"
        )

        data = {'file': uploaded_file}
        response = self.client.post(self.url, data, format='multipart')

        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED)
        self.assertEqual(response.data['document_type'], 'receipt')


class DocumentProcessingTest(TestCase):
    """Test the document processing pipeline"""

    @patch('api.tasks.classifier')
    @patch('api.tasks.extract_text')
    @patch('api.tasks.clean_text')
    @patch('api.tasks.tokenize')
    @patch('api.tasks.extract_entities')
    @patch('api.tasks.extract_amount')
    def test_process_document_pipeline(self, mock_extract_amount, mock_extract_entities,
                                     mock_tokenize, mock_clean, mock_extract, mock_classifier):
        """Test that process_document calls all pipeline steps"""
        from api.tasks import process_document

        # Setup mocks
        mock_extract.return_value = "raw text"
        mock_clean.return_value = "cleaned text"
        mock_tokenize.return_value = "tokenized text"
        mock_classifier.predict_with_confidence.return_value = ("invoice", 0.95)
        mock_extract_entities.return_value = ({"dates": [], "money": []}, "processed text")
        mock_extract_amount.return_value = "$100.00"

        # Call the function
        result = process_document("fake_path.pdf")

        # Verify all steps were called
        mock_extract.assert_called_once_with("fake_path.pdf")
        mock_clean.assert_called_once_with("raw text")
        mock_tokenize.assert_called_once_with("cleaned text")
        mock_classifier.predict_with_confidence.assert_called_once_with("tokenized text")
        mock_extract_entities.assert_called_once_with("raw text")
        mock_extract_amount.assert_called_once_with("raw text")

        # Verify result structure
        expected_result = {
            "document_type": "invoice",
            "confidence": 0.95,
            "entities": {"dates": [], "money": []},
            "text": "processed text",
            "amount": "$100.00",
        }
        self.assertEqual(result, expected_result)
