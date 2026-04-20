from file_loader import extract_text
from preprocess import clean_text, tokenize
from classifier import DocumentClassifier
from extractor import extract_entities, extract_amount

# Load trained model
classifier = DocumentClassifier()
classifier.load("model/model.pkl")


def process_document(file_path):
    # Step 1: Extract text
    raw_text = extract_text(file_path)

    # Step 2: Preprocess
    cleaned = clean_text(raw_text)
    processed = tokenize(cleaned)

    # Step 3: Classify
    # doc_type = classifier.predict(processed)
    doc_type, confidence = classifier.predict_with_confidence(processed)

    # Step 4: Extract data
    entities, text = extract_entities(raw_text)
    amount = extract_amount(raw_text)

    return {
        "document_type": doc_type,
        "confidence": float(confidence),
        "entities": entities,
        "text": text,
        "amount": amount,
    }


# Example usage
if __name__ == "__main__":
    file_path = "data/handwritten/502262504_502262505.jpg"

    result = process_document(file_path)

    print("\n Result:")
    print("Type:", result["document_type"])
    print("Confidence:", result["confidence"])
    print("Entities:", result["entities"])
    print("Amount:", result["amount"])
    print("Text:", result["text"][:200], "...")
