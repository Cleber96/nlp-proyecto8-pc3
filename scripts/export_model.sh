#!/usr/bin/env bash
# export_model.sh
# --------------------------------
# Exporta el modelo cuantizado a ONNX y TorchScript.

set -e

echo "== Exportando modelo a ONNX y TorchScript =="

# Crear directorio de salida y logs
mkdir -p outputs/models/quantized outputs/logs

# Export ONNX y TorchScript
python - << 'PYCODE'
import torch
from transformers import AutoTokenizer, pipeline
import yaml
from src.quantization.post_training_quant import apply_dynamic_quantization

# Cargar configuración de exportación
with open("config/base_config.yaml") as f:
    cfg = yaml.safe_load(f)

tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["pretrained_model_name"])
quantized_model = apply_dynamic_quantization("outputs/models/pruned", "outputs/models/quantized/model_quantized.pt")

# Export ONNX
onnx_path = cfg["export"]["onnx_output"]
nlp = pipeline("text-classification", model=quantized_model, tokenizer=tokenizer)
nlp.model.eval()
dummy_input = ("This is a test.", "Another sentence.")
input_ids = torch.tensor([tokenizer.encode(dummy_input[0], dummy_input[1])])
torch.onnx.export(nlp.model, (input_ids,), onnx_path, opset_version=cfg["export"]["onnx_opset"])
print(f"ONNX guardado en {onnx_path}")

# Export TorchScript
ts_path = cfg["export"]["torchscript_output"]
scripted = torch.jit.script(quantized_model)
torch.jit.save(scripted, ts_path)
print(f"TorchScript guardado en {ts_path}")
PYCODE 2>&1 | tee outputs/logs/export_model.log

echo "== Exportación completada =="
