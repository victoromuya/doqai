# from celery import shared_task
from pathlib import Path
import re
from django.conf import settings
from .file_loader import extract_text
from huggingface_hub import InferenceClient, login

from rag.chunking import chunk_text
from rag.embedding_service import generate_passage_embedding
from rag.vectordb import collection
import uuid


HF_TOKEN = settings.DOC_HF_TOKEN
# login(token=HF_TOKEN, add_to_git_credential=False)

# Initialize the client (only once at the top of your file)
client = InferenceClient(token=HF_TOKEN, timeout=120, headers={"x-wait-for-model": "true"} )


import re

def extract_entities(text):
    """
    Extracts expanded entities using Regex and Hugging Face API.
    Added: Locations, Miscellaneous, Emails, and Phone Numbers.
    """
    data = {
        "dates": re.findall(r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4})|(?:\d{4}[/-]\d{1,2}[/-]\d{1,2})|(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4})\b', text),
        "money": re.findall(r'[\$₦€]\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text),
        # New Regex categories (Zero RAM usage)
        "emails": re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text),
        "phones": re.findall(r'\+?\d{1,4}?[-.\s]?\(?\d{1,3}?\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}', text),
        "organizations": [],
        "persons": [],
        "locations": [],
        "misc": [] # Includes events, nationalities, products, etc.
    }

    if text and len(text.strip()) > 0:
        try:
            api_result = client.token_classification(
                text[:800], 
                model="dslim/bert-base-NER",
                aggregation_strategy="simple" 
            )

            for entity in api_result:
                label = entity.get('entity_group')
                word = entity.get('word', '').strip()
                
                if not word or len(word) < 2:
                    continue

                if label == 'PER':
                    if word not in data["persons"]: data["persons"].append(word)
                elif label == 'ORG':
                    if word not in data["organizations"]: data["organizations"].append(word)
                # New Label Mappings
                elif label == 'LOC':
                    if word not in data["locations"]: data["locations"].append(word)
                elif label == 'MISC':
                    if word not in data["misc"]: data["misc"].append(word)

        except Exception as e:
            print(f"HF Entity Extraction Error: {e}")

    return data, text




def classify_document(raw_text):
   
    text_to_classify = raw_text[:700].strip() if raw_text else ""
    
    if not text_to_classify:
        return "empty_document", 0.0

    try:
        # Using the official client for Zero-Shot Classification
        result = client.zero_shot_classification(
            text=text_to_classify,
            candidate_labels=[
                "professional resume or cv", 
                "business invoice or receipt",
                "formal letter or memo",
                "questionnaire or form", 
                "legal contract", 
              
            ],
            model="facebook/bart-large-mnli",
            hypothesis_template="This document is a {}.",
            # CRUCIAL: Tells HF to wait if the model is currently 'sleeping'
            
        )
        
        # The client returns a list of dicts sorted by score: 
        # [{'label': '...', 'score': 0.9}, ...]
        if isinstance(result, list) and len(result) > 0:
            top_prediction = result[0]
            return top_prediction['label'], top_prediction['score']
        
        return "unknown", 0.0

    except Exception as e:
        # Log the specific error for debugging on Render
        print(f"Hugging Face Classification Error: {e}")
        return "unknown", 0.0

  
def process_document(file_path):
    # Step 1: Extract text
    result = extract_text(file_path)

    
    # Check if extract_text returned an error dictionary (e.g., 3-page limit)
    if isinstance(result, dict) and "error" in result:
        return result  # Returns {"error": "...", "message": "..."} directly to the user

    raw_text = result
    if not raw_text or not raw_text.strip():
        return {"error": "Extraction failure", "message": "No text could be extracted from this file."}
    
    # Step 2: Classify using the separate function
    doc_type, confidence = classify_document(raw_text)

    if confidence < 0.3:  # Threshold for uncertain classifications
        doc_type = "Uncertain (require review)"

    # Step 3: Extract structured data
    entities, cleaned_text = extract_entities(raw_text)
    
    # Safely extract amount from the dictionary returned by extract_entities
    amount = entities.get("money")[0] if entities.get("money") else None

    return {
        "document_type": doc_type,
        "confidence": float(confidence),
        "entities": entities,
        "text": cleaned_text,
        "amount": amount,
    }




def rewrite_cv_section(cv_text, job_description):
        
    user_prompt = f"CV Content: {cv_text}\n\nJob Description: {job_description}"
    SYSTEM_PROMPT = "You are a helpful assistant that rewrites CV sections to better \
        match the job description. \
            Focus on highlighting relevant skills and experience. Use correct keywords from the job description. \
              Keep the original meaning but improve the alignment with the job requirements."
    
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            model="meta-llama/Meta-Llama-3-8B-Instruct",
            max_tokens=1500,
            temperature=0.7
        )
        
        return response.choices[0].message.content
    except Exception as e:
        print(f"Rewriter Error: {e}")
        return cv_text  # Fallback to original text



def only_extract(file_path):
    # Step 1: Extract text
    result = extract_text(file_path)
    
    # Check if extract_text returned an error dictionary (e.g., 3-page limit)
    if isinstance(result, dict) and "error" in result:
        return result  # Returns {"error": "...", "message": "..."} directly to the user

    raw_text = result
    if not raw_text or not raw_text.strip():
        return {"error": "Extraction failure", "message": "No text could be extracted from this file."}
    
    document_id = str(uuid.uuid4())
    
    # Clear the existing ChromaDB collection to reset for new upload
    all_ids = collection.get()['ids']
    if all_ids:
        collection.delete(ids=all_ids)
    
    index_document(document_id, raw_text) # Index and Add the document in ChromaDB for RAG retrieval
    
    # Step 3: Extract structured data
    entities, cleaned_text = extract_entities(raw_text)

    return {
       
        "entities": entities,
        "text": cleaned_text,
       
    }


def index_document(document_id, raw_text):
    chunks = chunk_text(raw_text)

    for i, chunk in enumerate(chunks):
        embedding = generate_passage_embedding(chunk)

        collection.add(
            ids=[f"{document_id}_{i}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[{
                "document_id": document_id
            }]
        )