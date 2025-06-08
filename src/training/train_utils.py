import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(p):
    """Función para pasar a Trainer.compute_metrics."""
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}