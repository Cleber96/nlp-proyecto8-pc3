#!/usr/bin/env bash
# run_pruning.sh
# --------------------------------
# Este script aplica pruning estructural al modelo fine-tuneado.

set -e

echo "== Iniciando pruning de cabezas de atención =="

# Crear directorio de salida y logs
mkdir -p outputs/models/pruned outputs/logs

# Ejecutar pruning
python - << 'PYCODE'
from src.pruning.structured_prune import apply_structural_pruning
import yaml

# Cargar configuración de pruning
with open("config/base_config.yaml") as f:
    cfg = yaml.safe_load(f)
prune_heads = {int(k): v for k, v in cfg["pruning"]["prune_heads"].items()}
apply_structural_pruning("outputs/models/finetuned", prune_heads, "outputs/models/pruned/model_pruned.pt")
print("Pruning completado.")
PYCODE 2>&1 | tee outputs/logs/run_pruning.log

echo "== Pruning finalizado: pesos guardados en outputs/models/pruned =="
