import pickle
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

from services.preprocess import (
    clean_text,
    load_tokenizer,
    normalize_texts,
    texts_to_padded_sequences,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "document_classifier.keras"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.json"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
MAX_SEQUENCE_LENGTH = 256


class DocumentInference:
    def __init__(self,
                 model_path: Path = MODEL_FILE,
                 tokenizer_path: Path = TOKENIZER_FILE,
                 label_encoder_path: Path = LABEL_ENCODER_FILE,
                 max_length: int = MAX_SEQUENCE_LENGTH):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.max_length = max_length
        
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self._loaded = False

    def _load_artifacts(self):
        """Lazy load model artifacts on first use."""
        if self._loaded:
            return

        if load_model is None:
            raise RuntimeError(
                "TensorFlow is not installed, so the deep-learning classifier is unavailable."
            )
        
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Please train the model first using services.train.train()"
            )
        
        self.model = load_model(self.model_path)
        self.tokenizer = load_tokenizer(self.tokenizer_path)
        
        with open(self.label_encoder_path, "rb") as handle:
            self.label_encoder = pickle.load(handle)
        
        self._loaded = True

    def preprocess(self, raw_text: str):
        self._load_artifacts()
        cleaned = clean_text(raw_text)
        normalized = normalize_texts([cleaned])
        return texts_to_padded_sequences(normalized, self.tokenizer, max_length=self.max_length)

    def predict(self, raw_text: str):
        self._load_artifacts()
        sequence = self.preprocess(raw_text)
        probabilities = self.model.predict(sequence, verbose=0)[0]
        prediction_index = int(np.argmax(probabilities))
        label = self.label_encoder.inverse_transform([prediction_index])[0]
        confidence = float(probabilities[prediction_index])
        return label, confidence, probabilities.tolist()


inference = DocumentInference()
