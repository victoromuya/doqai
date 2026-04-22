# from celery import shared_task
from pathlib import Path
import re
from django.conf import settings
from services.file_loader import extract_text
from api.models import DocumentResult
from huggingface_hub import InferenceClient, login


HF_TOKEN = settings.DOC_HF_TOKEN
# login(token=HF_TOKEN, add_to_git_credential=False)

# Initialize the client (only once at the top of your file)
client = InferenceClient(token=HF_TOKEN)


def extract_entities(text):
    """
    Uses Hugging Face API to extract Persons and Organizations.
    Uses Regex for Dates and Money (to save API calls/latency).
    """
    # 1. Quick Regex for Dates and Money (Save API tokens)
    data = {
        "dates": re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}\b', text),
        "money": re.findall(r'[\$₦€]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text),
        "organizations": [],
        "persons": []
    }

    # 2. Hugging Face API for Persons and Organizations
    try:
        # NER models have a 512 token limit (~700-800 characters)
        # We only need the first part of the doc to find the main entities
        api_result = client.token_classification(
            text[:800], 
            model="dslim/bert-base-NER"
        )

        for entity in api_result:
            # B-PER/I-PER = Person, B-ORG/I-ORG = Organization
            word = entity['word'].replace('##', '') # Clean up BERT wordpieces
            label = entity['entity_group'] 
            
            if label == 'PER' and word not in data["persons"]:
                data["persons"].append(word)
            elif label == 'ORG' and word not in data["organizations"]:
                data["organizations"].append(word)

    except Exception as e:
        print(f"HF Entity Extraction failed: {e}")

    return data, text


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

    if confidence < 0.3:  # Threshold for uncertain classifications
        doc_type = "Uncertain (require review)"

    # Step 3: Extract structured data
    entities, cleaned_text = extract_entities(raw_text)
    amount = entities["money"][0] if entities["money"] else None

  

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
