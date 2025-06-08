import torch
from transformers import AutoModelForSequenceClassification

def apply_dynamic_quantization(model_dir: str, output_path: str):
    """Carga un modelo, aplica cuantización dinámica y guarda nuevos pesos."""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    quantized = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
    torch.save(quantized.state_dict(), output_path)
    return quantized