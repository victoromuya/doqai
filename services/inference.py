import pickle
from typing import Optional
import numpy as np
from pathlib import Path
# Import the lightweight runtime instead of full TensorFlow
from tensorflow.lite.python.interpreter import Interpreter

from services.preprocess import (
    clean_text,
    load_tokenizer,
    normalize_texts,
    texts_to_padded_sequences,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
# Update path to the .tflite file
MODEL_FILE = MODEL_DIR / "document_classifier.tflite"
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
        
        self.interpreter = None
        self.tokenizer = None
        self.label_encoder = None
        self._loaded = False

    def _load_artifacts(self):
        """Lazy load TFLite interpreter and artifacts."""
        if self._loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"TFLite model not found at {self.model_path}")
        
        # Initialize TFLite Interpreter
        self.interpreter = Interpreter(model_path=str(self.model_path))
        self.interpreter.allocate_tensors()
        
        # Get input/output details for inference
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.tokenizer = load_tokenizer(self.tokenizer_path)
        
        with open(self.label_encoder_path, "rb") as handle:
            self.label_encoder = pickle.load(handle)
        
        self._loaded = True

    def preprocess(self, raw_text: str):
        self._load_artifacts()
        cleaned = clean_text(raw_text)
        normalized = normalize_texts([cleaned])
        # TFLite expects float32 or int32; ensure the sequence matches your model's input type
        sequence = texts_to_padded_sequences(normalized, self.tokenizer, max_length=self.max_length)
        return sequence.astype(np.float32) 

    def predict(self, raw_text: str):
        self._load_artifacts()
        sequence = self.preprocess(raw_text)
        
        # TFLite Inference Step
        self.interpreter.set_tensor(self.input_details[0]['index'], sequence)
        self.interpreter.invoke()
        probabilities = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        
        prediction_index = int(np.argmax(probabilities))
        label = self.label_encoder.inverse_transform([prediction_index])[0]
        confidence = float(probabilities[prediction_index])
        
        return label, confidence, probabilities.tolist()

inference = DocumentInference()