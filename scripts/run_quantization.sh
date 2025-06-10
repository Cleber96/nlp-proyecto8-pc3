#!/usr/bin/env bash
# run_quantization.sh
# --------------------------------
# Este script aplica cuantización dinámica post-entrenamiento al modelo podado.

set -e

echo "== Iniciando cuantización dinámica =="

# Crear directorio de salida y logs
mkdir -p outputs/models/quantized outputs/logs

# Ejecutar cuantización
python - << 'PYCODE'
from src.quantization.post_training_quant import apply_dynamic_quantization
import yaml

# Cargar configuración de cuantización
with open("config/base_config.yaml") as f:
    cfg = yaml.safe_load(f)
apply_dynamic_quantization("outputs/models/pruned", "outputs/models/quantized/model_quantized.pt")
print("Cuantización completada.")
PYCODE 2>&1 | tee outputs/logs/run_quantization.log

echo "== Cuantización finalizada: pesos guardados en outputs/models/quantized =="
