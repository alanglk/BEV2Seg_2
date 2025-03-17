Estructura principal de la memoria del TFM:


## 1. Introducción

- **Contexto y motivación**: Importancia de la segmentación semántica en BEV (Bird’s Eye View) en el ámbito de percepción vehicular.

- **Planteamiento del problema**: Explicación del desafío de identificar superficies planares en BEV.

- **Pregunta de investigación**: Presentar la pregunta principal y su relevancia.

- **Objetivos**:
    Evaluar la segmentación semántica en BEV con dos enfoques distintos.
    Implementar un pipeline para la generación de máscaras BEV locales.

- **Metodología general**: Breve resumen de la aproximación empleada (experimentos con modelos + desarrollo del pipeline).

- **Estructura del documento**: Breve explicación de los capítulos.

## 2. Estado del arte

- **Segmentación semántica**:
    CNN, ViT, modelos fundacionales...

- **Segmentación semántica en BEV**:
    Técnicas actuales y modelos utilizados en percepción vehicular.
    Diferencias entre entrenar en BEV directamente vs. reproyectar imágenes segmentadas.

- **Conclusión del estado del arte**: Identificación de la brecha en la literatura que tu trabajo aborda.

## 3. Metodología

- **Diseño del experimento de segmentación**:
    - Modelo utilizado (Segformer), arquitectura y pipelines segmentacion.
    - Dataset de entrenamiento -> generación de BEVDataset y NuImagesFormatted a partir de NuImages
     - Transformación para la generación de datos. Explicar cómo se pasa a BEV (IPM)
    - Data augmentations
    - Proceso de entrenamiento y validación en ambos casos.
    - Metodología de evaluación / comparación

- **Estructura del pipeline de fusión BEV**:
    Descripción de cada etapa:
        Captura de imágenes y depthmaps.
        Generación de detecciones 3D.
        Fusión de información.
        Creación de la máscara BEV local.
    Diagrama de flujo del sistema.

- **Métricas de evaluación**:
    Qué métricas se usan para comparar la calidad de segmentación (IoU, precisión, recall, etc.).

## 5. Evaluación y Resultados

- **Análisis comparativo de la segmentación BEV**:
    Desempeño de ambos enfoques con gráficos y tablas.
    Ventajas y desventajas de cada método.

- **Validación del pipeline BEV**:
    Pruebas en distintos escenarios.
    Errores detectados y mejoras potenciales.

- **Discusión**:
    Interpretación de los resultados en contexto.
    Comparación con estudios previos.

## 6. Conclusiones y Trabajo Futuro

- **Resumen de hallazgos clave**:
    ¿Cuál enfoque de segmentación BEV funciona mejor y por qué?
    ¿Cómo contribuye el pipeline al problema de percepción vehicular?
- **Limitaciones del estudio: Datos, modelos, restricciones computacionales**.
- **Futuras líneas de investigación**:
    Mejoras en la segmentación BEV (más datos, modelos más avanzados).
    Integración con sensores LiDAR o radares.
    Optimización del pipeline en tiempo real.

