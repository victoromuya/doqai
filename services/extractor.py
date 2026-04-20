import re
import spacy

nlp = spacy.load("en_core_web_sm")


def extract_entities(text):
    doc = nlp(text)

    data = {"dates": [], "money": [], "organizations": [], "persons": []}

    for ent in doc.ents:
        if ent.label_ == "DATE":
            data["dates"].append(ent.text)
        elif ent.label_ == "MONEY":
            data["money"].append(ent.text)
        elif ent.label_ == "ORG":
            data["organizations"].append(ent.text)
        elif ent.label_ == "PERSON":
            data["persons"].append(ent.text)

    return data, text


def extract_amount(text):
    # supports $, ₦, €, commas, decimals
    match = re.search(r"[\$₦€]?\s?\d{1,3}(?:,\d{3})*(?:\.\d{2})?", text)
    return match.group() if match else None
