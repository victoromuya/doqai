# from celery import shared_task
from pathlib import Path

from django.conf import settings

from services.file_loader import extract_text
from services.extractor import extract_entities, extract_amount

from services.preprocess import clean_text, tokenize
from api.models import DocumentResult
from huggingface_hub import InferenceClient, login


HF_TOKEN = settings.DOC_HF_TOKEN
# login(token=HF_TOKEN, add_to_git_credential=False)

# Initialize the client (only once at the top of your file)
client = InferenceClient(token=HF_TOKEN)

def classify_document(raw_text):
    # Ensure text isn't empty and stay within token limits
    text_to_classify = raw_text[:1000] if raw_text else "No text found"
    
   
  
        # 2. API Fallback using the official Client
    try:
        # The client handles the URL formatting for you
        result = client.zero_shot_classification(
            text=text_to_classify,
            candidate_labels=["professional resume or CV", "business invoice or receipt",
                               "formal letter or email or memo",
                              "handwritten note", "academic research paper", "legal contract", "general correspondence"],
            model="facebook/bart-large-mnli",
            hypothesis_template="This document is a {}.",

        )
        
        # Result is a list of dicts: [{'label': 'invoice', 'score': 0.9}, ...]
        # We want the top prediction
        top_prediction = result[0]
        return top_prediction['label'], top_prediction['score']

    except Exception as e:
        print(f"Hugging Face Client Error: {e}")
        return "unknown", 0.0
  

def process_document(file_path):
    # Step 1: Extract text
    raw_text = extract_text(file_path)
    if not raw_text:
        return {"error": "No text could be extracted"}

    # Step 2: Classify using the separate function
    doc_type, confidence = classify_document(raw_text)

    if confidence < 0.5:  # Threshold for uncertain classifications
        doc_type = "Uncertain (require review)"

    # Step 3: Extract structured data
    entities, cleaned_text = extract_entities(raw_text)
    amount = extract_amount(raw_text)

    # Step 4: Persist result
    try:
        DocumentResult.objects.create(
            file=file_path,
            document_type=doc_type,
            confidence=confidence,
            entities=entities,
            amount=amount,
        )
    except Exception as db_e:
        print(f"Database save failed: {db_e}")

    return {
        "document_type": doc_type,
        "confidence": confidence,
        "entities": entities,
        "text": cleaned_text,
        "amount": amount,
    }
