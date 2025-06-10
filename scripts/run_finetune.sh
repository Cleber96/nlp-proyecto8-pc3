#!/usr/bin/env bash
# run_finetune.sh
# --------------------------------
# Este script ejecuta el fine-tuning del modelo Transformer.

set -e  # Salir si ocurre algún error

echo "== Iniciando fine-tuning de DistilBERT en MRPC =="

# Crear directorio de salida y logs
mkdir -p outputs/models/finetuned outputs/logs

# Ejecutar el script de fine-tuning y registrar salida
python src/training/finetune.py 2>&1 | tee outputs/logs/run_finetune.log

echo "== Fine-tuning completado: pesos guardados en outputs/models/finetuned =="
