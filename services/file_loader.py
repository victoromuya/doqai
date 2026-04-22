import os
import requests

# Get your free API key from https://ocr.space/ocrapi
OCR_API_KEY = os.getenv("OCR_API_KEY")



def extract_via_cloud_ocr(file_path):
    payload = {
        'apikey': OCR_API_KEY,
        'language': 'eng',
        'isOverlayRequired': False,
        'OCREngine': 2,    # Engine 2 is superior for alphanumeric data like invoices
        'isTable': True,   # This preserves the table/line structure
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
        
        if result.get('OCRExitCode') == 1:
            # OCR.space returns a list of results (one per page for PDFs)
            return " ".join([page['ParsedText'] for page in result['ParsedResults']])
        else:
            print(f"OCR Error: {result.get('ErrorMessage')}")
            return ""
            
    except Exception as e:
        print(f"Cloud OCR Request failed: {e}")
        return ""

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
