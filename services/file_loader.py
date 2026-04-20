import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
import PyPDF2


def extract_from_image(file_path):
    image = Image.open(file_path)
    return pytesseract.image_to_string(image)


def extract_from_pdf(file_path):
    text = ""

    # Try reading as text PDF first
    try:
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() or ""
    except:
        pass

    # If empty → scanned PDF → use OCR
    if len(text.strip()) == 0:
        images = convert_from_path(file_path)
        for img in images:
            text += pytesseract.image_to_string(img)

    return text


def extract_from_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg"]:
        return extract_from_image(file_path)

    elif ext == ".pdf":
        return extract_from_pdf(file_path)

    elif ext == ".txt":
        return extract_from_txt(file_path)

    else:
        raise ValueError("Unsupported file type")
