# from celery import shared_task
import os
from services.file_loader import extract_text
from services.extractor import extract_entities, extract_amount
from services.inference import inference

from api.models import DocumentResult


def process_document(file_path):
    # Step 1: Extract text
    raw_text = extract_text(file_path)

    # Step 2: Classify with the deep learning model
    doc_type, confidence, _ = inference.predict(raw_text)

    # Step 3: Extract structured data
    entities, text = extract_entities(raw_text)
    amount = extract_amount(raw_text)

    # Optional: persist result history
    try:
        DocumentResult.objects.create(
            file=file_path,
            document_type=doc_type,
            confidence=confidence,
            entities=entities,
            amount=amount,
        )
    except Exception:
        pass

    return {
        "document_type": doc_type,
        "confidence": float(confidence),
        "entities": entities,
        "text": text,
        "amount": amount,
    }
