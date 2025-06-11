# PC3: Proy8. Fine-tuning y pruning en Transformer
## Descripción y Objetivos

En este proyecto profundizamos en técnicas de compresión y optimización de modelos Transformer aplicadas a la tarea **MRPC** de GLUE:

- **Fine-tuning** de DistilBERT para clasificación de pares de oraciones.
- **Pruning estructural**: eliminación selectiva de cabezas de atención para acelerar inferencia.
- **Cuantización post-entrenamiento**: reducción de precisión (`float32 → int8`) para reducir tamaño y mejorar latencia.
- **Exportación** a ONNX y TorchScript para despliegue en entornos restringidos.


## Estructura del Repositorio

```plaintext
practica3-CC0C2/
├── README.md
├── requirements.txt
├── .gitignore
├── config/
│   ├── base_config.yaml
│   └── finetune_config.yaml
├── data/
│   ├── raw/                 # CSV originales MRPC
│   └── processed/           # Datos tokenizados guardados
├── notebooks/               # Exploración y demos interactivas
├── src/                     # Código modular en PyTorch
├── scripts/                 # Wrappers bash y scripts Python
├── outputs/                 # logs, modelos y figuras generadas
├── reports/                 # tablas comparativas y presentación
└── tests/                   # Suites de pruebas automatizadas
```

## Instalación de Dependencias

1. Clona el repositorio y entra en la carpeta:
   ```bash
   git clone <https://github.com/Cleber96/nlp-proyecto8-pc3>
   cd practica3-CC0C2
   ```
2. Crea y activa un entorno virtual:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # macOS/Linux
   venv\Scripts\activate       # Windows
   ```
3. Instala las librerías:
   ```bash
   pip install -r requirements.txt
   ```

## Preparación de Datos

1. **Raw**: descarga y guarda GLUE MRPC en CSV en `data/raw/`  
   ```bash
   python scripts/prepare_data.py --step raw \
     --raw_dir data/raw
   ```
2. **Processed**: tokeniza y guarda en formato HuggingFace en `data/processed/`  
   ```bash
   python scripts/prepare_data.py --step processed \
     --raw_dir data/raw \
     --proc_dir data/processed \
     --model_name distilbert-base-uncased \
     --max_length 128
   ```

## Ejecución de Pipelines

1. **Fine-tuning**  
   ```bash
   bash scripts/run_finetune.sh
   ```
   - Checkpoints en `outputs/models/finetuned/`  
   - Logs en `outputs/logs/run_finetune.log`

2. **Pruning estructural**  
   ```bash
   bash scripts/run_pruning.sh
   ```
   - Modelo podado en `outputs/models/pruned/model_pruned.pt`  
   - Logs en `outputs/logs/run_pruning.log`

3. **Cuantización dinámica**  
   ```bash
   bash scripts/run_quantization.sh
   ```
   - Pesos cuantizados en `outputs/models/quantized/model_quantized.pt`  
   - Logs en `outputs/logs/run_quantization.log`

4. **Exportación ONNX / TorchScript**  
   ```bash
   bash scripts/export_model.sh
   ```
   - Archivos `model_quantized.onnx` y TorchScript en `outputs/models/quantized/`  
   - Logs en `outputs/logs/export_model.log`

---

## Resultados

- Modelos entrenados y comprimidos: `outputs/models/finetuned/`, `pruned/`, `quantized/`  
- Figuras (curvas de accuracy, histogramas): `outputs/figures/`  
- Comparativa tamaño vs. precisión: `reports/tables/size_accuracy.csv`  
- Presentación: `reports/presentation.pptx`

## Ejecutar Tests

Comprueba la integridad de todo el pipeline:
```bash
pytest tests/ --maxfail=1 --disable-warnings -q
```
