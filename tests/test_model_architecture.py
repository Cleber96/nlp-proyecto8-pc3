import torch
import pytest
from src.models.model import TransformerClassifier
from src.models.helpers import count_parameters

def test_transformer_classifier_forward():
    model_name = "distilbert-base-uncased"
    num_labels = 2
    model = TransformerClassifier(model_name, num_labels)
    # Entradas dummy
    input_ids = torch.randint(0, 1000, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    # Forward sin etiquetas
    logits = model(input_ids, attention_mask)
    assert logits.shape == (2, num_labels)

def test_count_parameters():
    model_name = "distilbert-base-uncased"
    num_labels = 2
    model = TransformerClassifier(model_name, num_labels)
    total = count_parameters(model)
    # Debe ser mayor que cero
    assert total > 0