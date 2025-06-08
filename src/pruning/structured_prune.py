import torch
from transformers import AutoModelForSequenceClassification

def apply_structural_pruning(model_dir: str, prune_heads: dict, output_path: str):
    """Carga un modelo, aplica pruning por heads y guarda nuevos pesos."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.distilbert.prune_heads(prune_heads)
    torch.save(model.state_dict(), output_path)
    return model
