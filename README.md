# DocAI API

A Django REST API for document intelligence: upload documents (PDF/image/text), extract text with OCR/NLP, classify document type, and extract entities/amounts.

## 🔧 Features

- Document upload endpoint (`POST /api/v1/upload/`)
- Text extraction from `.txt`, `.pdf`, `.jpg`, `.jpeg`, `.png`
- OCR support with `pytesseract` + `pdf2image` for scanned PDFs and images
- NLP processing with spaCy (`en_core_web_sm`)
- Document classification with TF-IDF + LogisticRegression (`services/classifier.py`)
- Entity extraction (dates, money, organizations, persons)
- Amount extraction (currency and numeric patterns)
- Data model `DocumentResult` persisted in Django DB

## 📦 Tech stack

- Python 3.11+ (recommended)
- Django 5.2
- Django REST Framework
- spaCy
- PyPDF2
- Pillow
- pytesseract
- pdf2image
- scikit-learn
- joblib

## 🚀 Setup

1. Clone repository

```bash
git clone <repo-url>
cd Docai_api
```

2. Create virtual environment and install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

3. Install spaCy model

```bash
python -m spacy download en_core_web_sm
```

4. Configure media folder (optional but required for uploads)

In `Docai_api/settings.py` add:

```py
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'
```

5. Apply migrations

```bash
python manage.py migrate
```

6. (Optional) Create a superuser

```bash
python manage.py createsuperuser
```

7. Start server

```bash
python manage.py runserver
```

## 🗂️ Model training (optional)

The model can be retrained on a custom dataset organized by class in `data/<label>/*.pdf|.txt|.jpg`.

```bash
python services/train.py
```

Generated model file: `services/models/model.pkl` (or `model/model.pkl` depending on config).

## 🧪 API Usage

- URL: `http://127.0.0.1:8000/api/v1/upload/`
- Method: `POST`
- Body: multipart/form-data
- Field: `file` (upload file)

### Example curl

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/upload/" \
  -F "file=@/path/to/document.pdf"
```

### Success response (202)

```json
{
  "message": "Document is being processed",
  "document_type": "invoice",
  "text": "...",
  "confidence": 0.92,
  "entities": {
    "dates": [...],
    "money": [...],
    "organizations": [...],
    "persons": [...]
  },
  "amount": "$123.45"
}
```

## 🛠️ Project structure

- `api/`: Django app with endpoint logic
- `services/`: ML/NLP pipelines
  - `file_loader.py`: OCR/PDF/text extraction
  - `preprocess.py`: text cleaning and tokenization
  - `extractor.py`: entity and amount extraction
  - `classifier.py`: model training/predict
  - `train.py`: training script
- `Docai_api/`: Django project config
- `db.sqlite3`: local database

## ⚠️ Notes

- `MEDIA_ROOT` must exist and be writable by app
- `pytesseract` may require system binary (`tesseract`) installed in PATH
- `pdf2image` requires `poppler` installed (e.g., `choco install poppler` on Windows)
- `services/tasks.py` currently runs sync; replace with async task queue (Celery/RQ) if needed

## 📄 Testing

No dedicated tests yet; add tests in `api/tests.py`.

```bash
python manage.py test
```

## 🗂️ Improvements

- Add endpoint for listing `DocumentResult` records
- Persist returned fields in DB in `api/views.py` after processing
- Add Swagger/OpenAPI docs (drf_yasg already in `INSTALLED_APPS`)
- Add CI and linting
