import os
import requests

# Get your free API key from https://ocr.space/ocrapi
OCR_API_KEY = os.getenv("OCR_API_KEY")

def extract_via_cloud_ocr(file_path):
    payload = {
        'apikey': OCR_API_KEY,
        'language': 'eng',
        'isOverlayRequired': False,
        'OCREngine': 2,
        'isTable': True,
    }
    
    try:
        with open(file_path, 'rb') as f:
            response = requests.post(
                'https://api.ocr.space/parse/image',
                files={'file': f},
                data=payload,
                timeout=90
            )
        
        result = response.json()
        
        # 1. Check for basic API success
        if result.get('OCRExitCode') == 1:
            pages = result.get('ParsedResults', [])
            
            # 2. Check page count
            if len(pages) > 3:
                return {
                    "error": "Document too long",
                    "message": f"This document has {len(pages)} pages. Please upload a document with 3 pages or fewer."
                }
            
            # 3. Return joined text if within limit
            extracted_text = " ".join([page['ParsedText'] for page in pages])
            return extracted_text
            
        else:
            error_msg = result.get('ErrorMessage', ['Unknown OCR error'])[0]
            print(f"OCR Error: {error_msg}")
            return {"error": "OCR failure", "message": error_msg}
            
    except Exception as e:
        print(f"Cloud OCR Request failed: {e}")
        return {"error": "Connection error", "message": "The OCR service timed out or failed."}

def extract_from_txt(file_path):
    """Handle .txt files locally (Low RAM)."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback for different encodings
        with open(file_path, "r", encoding="latin-1") as f:
            return f.read()

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".pdf"]:
        # Offload heavy PDF/Image processing to the cloud
        return extract_via_cloud_ocr(file_path)

    elif ext == ".txt":
        return extract_from_txt(file_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")
