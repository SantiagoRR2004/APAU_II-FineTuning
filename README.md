# Proyecto de Reconocimiento de Entidades Nombradas (NER)

Este proyecto implementa un sistema de Reconocimiento de Entidades Nombradas (NER) utilizando modelos de aprendizaje automático y procesamiento de lenguaje natural (NLP). El objetivo principal es entrenar y evaluar modelos para identificar y clasificar entidades en texto.

## Documentación Detallada

Para una documentación completa del proyecto, incluyendo la guía de ejecución, descripción de cada script y módulo, y detalles sobre la estructura de datos y modelos, por favor dirígete a la carpeta `docs`:

-   **[Documentación Principal](./docs/README.md)**

## Vista Rápida de la Estructura del Proyecto

El proyecto está organizado de la siguiente manera:

```
.
├── convert.py
├── convert2.py
├── crf.es.model
├── encodeType.py
├── evaluate_model.py
├── IOB.py
├── labelPropagation.py
├── main.py
├── predict_crf.py
├── requirements.txt
├── run_ner.py
├── run_train.sh
├── train.txt             # Archivo de log de ejemplo de un proceso de entrenamiento
├── vocab-token_encode.py
├── docs/                 # Contiene toda la documentación detallada
├── data/                 # Datos de entrenamiento, validación, etc.
├── labelOptions/         # Archivos generados por la propagación de etiquetas
└── models/               # Modelos entrenados y artefactos relacionados
```

### Archivos y Carpetas Clave

-   **Scripts Python (`.py`)**: Contienen la lógica para la conversión de datos, codificación, entrenamiento, evaluación, predicción y propagación de etiquetas.
-   **`run_train.sh`**: Script de shell para facilitar el proceso de entrenamiento.
-   **`requirements.txt`**: Lista las dependencias de Python necesarias.
-   **`crf.es.model`**: Un modelo CRF (Conditional Random Fields) preentrenado o generado.
-   **`train.txt`**: Un ejemplo de archivo de log que puede ser generado durante la ejecución de los scripts de entrenamiento. Su contenido puede variar.
-   **`docs/`**: Directorio que contiene toda la documentación detallada del proyecto.
-   **`data/`**: Almacena los conjuntos de datos.
-   **`labelOptions/`**: Utilizada para almacenar opciones de etiquetado generadas, por ejemplo, por el script `labelPropagation.py`.
-   **`models/`**: Guarda los modelos entrenados.

## Requisitos

Para ejecutar este proyecto, asegúrate de tener instaladas las dependencias listadas en `requirements.txt`. Puedes instalarlas ejecutando:

```bash
pip install -r requirements.txt
```

## Uso Básico

Consulta la **[Guía de Ejecución](./docs/guia_ejecucion.md)** dentro de la carpeta `docs` para instrucciones detalladas sobre cómo entrenar modelos, realizar predicciones y evaluar resultados.

De forma general:

### Entrenamiento

```bash
./run_train.sh
```

### Evaluación y Predicción

Revisa los scripts como `evaluate_model.py` y `predict_crf.py`. El uso detallado se encuentra en la documentación.

## Contribución

Si deseas contribuir a este proyecto, por favor abre un issue o envía un pull request.

## Licencia

Este proyecto está bajo la Licencia MIT (o la licencia que aplique). Consulta el archivo `LICENSE` (si existe) para más detalles.
