### **Práctica calificada 3 CC0C2**

- **Fecha de Entrega:** 19 de junio
- **Modalidad:** Grupal (máximo 2 integrantes)
- **Referencia general:** ["The Transformer Family Version 2.0"](https://lilianweng.github.io/posts/2023-01-27-the-transformer-family-v2/) de Lilian Weng

#### **Instrucciones generales**

El objetivo es profundizar en arquitecturas de Transformers más allá del modelo base, explorando optimizaciones y aplicaciones novedosas. Todos los proyectos deben cumplir con los siguientes requisitos:

* **Implementación:** El código debe ser desarrollado en **PyTorch**, superando las **500 líneas de código** (excluyendo comentarios y boilerplate). El código debe estar bien comentado y estructurado.
* **Repositorio:** El proyecto completo debe ser entregado a través de un repositorio de **Git**  que incluya un archivo `README.md` detallado con:
    * Descripción del proyecto y objetivos.
    * Instrucciones para instalar dependencias (`requirements.txt`).
    * Guía para ejecutar el código y reproducir los resultados.
    * Verificación de trabajo en conjunto y entrega individual de repositorios
* **Presentación:** Se realizará una presentación de **20 minutos** por grupo, dividida en:
    * **Contexto teórico:** ¿Qué problema se aborda y cuál es la idea fundamental?
    * **Detalles de implementación:** Arquitectura, decisiones clave y desafíos.
    * **Resultados y análisis:** Gráficos, tablas y conclusiones.
    * **Lecciones aprendidas:** ¿Qué funcionó, qué no y qué se haría diferente?

> Es requisito la entrega del proyecto para poder realizar la presentación. La evaluación es de tipo expositiva.

#### **8. Fine-tuning y pruning en Transformer**

* **Contexto teórico:**
    Los modelos de Transformer grandes (como BERT) son muy potentes pero también muy pesados para desplegar en dispositivos con recursos limitados. El **pruning (poda)** y la **cuantización** son técnicas para comprimir estos modelos.
    * **Pruning estructural:** En lugar de eliminar pesos individuales (poda no estructurada), se eliminan componentes enteros del modelo, como cabeceras de atención completas o neuronas en las capas FFN. Esto preserva la estructura densa de las matrices y acelera la inferencia en hardware estándar.
    * **Cuantización:** Consiste en reducir la precisión numérica de los pesos del modelo, por ejemplo, pasando de `float32` (32 bits) a `int8` (8 bits). Esto reduce el tamaño del modelo 4 veces y puede acelerar los cálculos.

* **Objetivos específicos:**
    -  Tomar un modelo Transformer pre-entrenado y pequeño, como **TinyBERT** o DistilBERT.
    -  Realizar un fine-tuning en una tarea de clasificación (e.g., GLUE).
    -  Aplicar un algoritmo de **pruning estructural** para eliminar un porcentaje de las cabeceras de atención menos importantes.
    -  Aplicar **cuantización post-entrenamiento** al modelo podado.

* **Entregables clave:**
    * **Comparativa de tamaños y accuracy:** Una tabla que compare cuatro versiones del modelo:
        -  Modelo original.
        -  Modelo con fine-tuning.
        -  Modelo podado.
        -  Modelo podado y cuantizado.
        La tabla debe mostrar el **tamaño en disco (MB)** y la **precisión** en el conjunto de prueba para cada uno.
    * **Script de exportación:** Un script que guarde el modelo final (cuantizado) en un formato optimizado para inferencia, como **ONNX** o **TorchScript**.