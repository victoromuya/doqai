import json
import re
from typing import List

import spacy
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.text import tokenizer_from_json

DEFAULT_MAX_WORDS = 15000
DEFAULT_MAX_SEQUENCE_LENGTH = 256

try:
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
except OSError:
    nlp = spacy.blank("en")


def clean_text(text: str) -> str:
    if text is None:
        return ""

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text.lower()


def tokenize(text: str) -> str:
    if not text:
        return ""

    doc = nlp(text)
    tokens = [
        token.lemma_.strip()
        for token in doc
        if not token.is_stop and not token.is_punct and token.lemma_.strip()
    ]
    return " ".join(tokens)


def normalize_texts(texts: List[str]) -> List[str]:
    return [tokenize(clean_text(text)) for text in texts]


def build_tokenizer(texts: List[str], num_words: int = DEFAULT_MAX_WORDS) -> Tokenizer:
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    return tokenizer


def texts_to_padded_sequences(
    texts: List[str], tokenizer: Tokenizer, max_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
):
    sequences = tokenizer.texts_to_sequences(texts)
    return pad_sequences(sequences, maxlen=max_length, padding="post", truncating="post")


def save_tokenizer(tokenizer: Tokenizer, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(tokenizer.to_json())


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "r", encoding="utf-8") as handle:
        data = handle.read()
    return tokenizer_from_json(data)
