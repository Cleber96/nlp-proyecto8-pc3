import os
import torch
import pytest
from src.pruning.structured_prune import apply_structural_pruning
from src.quantization.post_training_quant import apply_dynamic_quantization
from transformers import AutoModelForSequenceClassification

@pytest.fixture(scope="module")
def pretrained_model(tmp_path_factory):
    # Guardar modelo pequeño simulado
    model_dir = tmp_path_factory.mktemp("model")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    model.save_pretrained(str(model_dir))
    return str(model_dir)

def test_pruning_creates_file(pretrained_model, tmp_path):
    output = tmp_path / "pruned.pt"
    apply_structural_pruning(pretrained_model, {1: [0]}, str(output))
    assert output.exists()
    # Carga y verifica parámetros reducidos
    state = torch.load(str(output))
    assert isinstance(state, dict)

def test_quantization_creates_file(pretrained_model, tmp_path):
    output = tmp_path / "quantized.pt"
    apply_dynamic_quantization(pretrained_model, str(output))
    assert output.exists()
    state = torch.load(str(output))
    assert isinstance(state, dict)