import os
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Embedding, GlobalAveragePooling1D, LSTM

from file_loader import extract_text
from preprocess import (
    build_tokenizer,
    clean_text,
    normalize_texts,
    save_tokenizer,
    texts_to_padded_sequences,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "document_classifier.keras"
TOKENIZER_FILE = MODEL_DIR / "tokenizer.json"
LABEL_ENCODER_FILE = MODEL_DIR / "label_encoder.pkl"
MAX_WORDS = 20000
MAX_SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 256
BATCH_SIZE = 16
EPOCHS = 50
VALIDATION_SPLIT = 0.15
LEARNING_RATE = 0.001
L2_REGULARIZATION = 0.001


def load_dataset(data_dir: str, limit_per_class: Optional[int] = None):
    texts = []
    labels = []

    for label in sorted(os.listdir(data_dir)):
        folder = os.path.join(data_dir, label)
        if not os.path.isdir(folder):
            continue

        files = sorted(os.listdir(folder))
        if limit_per_class is not None:
            files = files[:limit_per_class]

        for file_name in files:
            path = os.path.join(folder, file_name)
            print(f"Processing {path}...")
            try:
                raw_text = extract_text(path)
                texts.append(raw_text)
                labels.append(label)
                print(f"Loaded {file_name} as {label}")
            except Exception as exc:
                print(f"Skipping {file_name}: {exc}")

    return texts, labels


def build_text_classification_model(
    vocab_size: int,
    num_classes: int,
    embedding_dim: int = EMBEDDING_DIM,
    input_length: int = MAX_SEQUENCE_LENGTH,
) -> tf.keras.Model:
    model = Sequential(
        [
            Embedding(vocab_size, embedding_dim, input_length=input_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            GlobalAveragePooling1D(),
            Dropout(0.4),
            Dense(128, activation="relu"),
            Dropout(0.3),
            Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_artifacts(tokenizer, label_encoder):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    save_tokenizer(tokenizer, TOKENIZER_FILE)
    with open(LABEL_ENCODER_FILE, "wb") as f:
        pickle.dump(label_encoder, f)


def evaluate_predictions(y_true, y_pred, labels):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nEvaluation results")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
    }


def prepare_training_data(texts, labels, max_words=MAX_WORDS, max_sequence_length=MAX_SEQUENCE_LENGTH):
    cleaned_texts = normalize_texts(texts)
    tokenizer = build_tokenizer(cleaned_texts, num_words=max_words)
    X = texts_to_padded_sequences(cleaned_texts, tokenizer, max_length=max_sequence_length)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(labels)
    y = tf.keras.utils.to_categorical(y_encoded, num_classes=len(label_encoder.classes_))

    return X, y, tokenizer, label_encoder


def train(data_dir: str = "data", limit_per_class: Optional[int] = 50):
    texts, labels = load_dataset(data_dir, limit_per_class=limit_per_class)
    if not texts:
        raise ValueError("No documents were loaded from the dataset directory.")

    print(f"Loaded {len(texts)} documents from {data_dir}")

    X, y, tokenizer, label_encoder = prepare_training_data(texts, labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=np.argmax(y, axis=1), random_state=42
    )

    model = build_text_classification_model(
        vocab_size=MAX_WORDS,
        num_classes=y.shape[1],
        input_length=X.shape[1],
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    model.fit(
        X_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )

    results = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest loss: {results[0]:.4f}, Test accuracy: {results[1]:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    y_true = np.argmax(y_test, axis=1)
    evaluate_predictions(y_true, y_pred, label_encoder.classes_)

    model.save(MODEL_FILE)
    save_artifacts(tokenizer, label_encoder)

    print(f"Model and artifacts saved to {MODEL_DIR}")
    return model, tokenizer, label_encoder


if __name__ == "__main__":
    train(DATA_DIR, limit_per_class=50)
