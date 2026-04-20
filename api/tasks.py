# from celery import shared_task
from pathlib import Path

from services.classifier import DocumentClassifier
from services.file_loader import extract_text
from services.extractor import extract_entities, extract_amount
from services.inference import inference
from services.preprocess import clean_text, tokenize

from api.models import DocumentResult

BASE_DIR = Path(__file__).resolve().parent.parent
SKLEARN_MODEL_PATH = BASE_DIR / "services" / "model" / "model.pkl"


def classify_document(raw_text):
    try:
        return inference.predict(raw_text)
    except (RuntimeError, FileNotFoundError, ImportError):
        classifier = DocumentClassifier()
        classifier.load(SKLEARN_MODEL_PATH)
        processed_text = tokenize(clean_text(raw_text))
        document_type, confidence = classifier.predict_with_confidence(processed_text)
        return document_type, float(confidence), None


def process_document(file_path):
    # Step 1: Extract text
    raw_text = extract_text(file_path)

    # Step 2: Classify, preferring the deep learning model when available.
    doc_type, confidence, _ = classify_document(raw_text)

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
