from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
import numpy as np


def evaluate_model(model, text, labels):
    # Get predictions
    predictions = model.predict(text)

    # Convert logits to predicted class indices
    # y_pred = np.argmax(predictions.logits, axis=1)
    #new code
    y_pred = np.argmax(predictions, axis=1)

    # Extract true labels
    # y_true = []
    # for batch in tf_dataset:
    #     y_true.extend(batch[1].numpy())

    # y_true = np.array(y_true)

    #new code
    y_true = labels.numpy() if hasattr(labels, "numpy") else labels

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred))