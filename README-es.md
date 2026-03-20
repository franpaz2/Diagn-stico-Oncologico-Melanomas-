# Diagnóstico Oncológico - Multimodal Deep Learning & ABCDE CNN

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

> **Diagnóstico Oncológico** es un sistema multimodal de Deep Learning diseñado para la detección precoz de melanomas. Supera las arquitecturas tradicionales fusionando visión artificial heurística (Regla ABCDE clínica) con una Red Neuronal Convolucional (CNN) entrenada bajo estrictas penalizaciones matemáticas para priorizar la sensibilidad sobre la precisión global.

---

## El Reto Clínico y Matemático
El diagnóstico dermatológico asistido por ordenador se enfrenta a dos problemas críticos. Primero, el **desbalanceo extremo de datos**: en los diagnósticos reales (y en datasets médicos como HAM10000 o ISIC), las lesiones benignas superan masivamente a las malignas. Segundo, el **coste del error**: los modelos estándar de Machine Learning buscan maximizar el "Accuracy" global, lo que en oncología genera una tasa inaceptable de Falsos Negativos (casos de cáncer omitidos).

## La Solución
Una arquitectura neuronal híbrida construida desde cero. En lugar de depender ciegamente de Transfer Learning genérico, el modelo ingesta paralelamente patrones topológicos crudos (imágenes) y variables biomatemáticas extraídas ad-hoc, desplazando la frontera de decisión para asegurar que ninguna anomalía maligna pase desapercibida.

---

## Arquitectura Técnica y Feature Engineering

### Fase 1: Extracción Biomatemática (OpenCV)
Las imágenes (redimensionadas a 128x128x3) pasan por un pipeline de visión artificial clásica que emula el protocolo médico ABCDE, generando un vector numérico tabular:
- **A (Asimetría):** Proyección focal y cálculo de diferencias absolutas espaciales (`cv2.absdiff`).
- **B (Bordes):** Operadores de gradientes laplacianos y filtros espectrales de Canny.
- **C (Color):** Conversión a espacio cromático HSV y análisis de heterogeneidad visual mediante clústering no supervisado (K-Means).
- **D/E (Diámetro y Elevación):** Detección de contornos perimetrales mediante Otsu Thresholding.

### Fase 2: Fusión Multimodal (Keras Functional API)
- **Rama A (Feature Extraction CNN):** 3 bloques jerárquicos de filtros `Conv2D` con `BatchNormalization` y `MaxPooling2D`, finalizando en un compresor vectorizado `GlobalAveragePooling2D`.
- **Rama B (Input Tabular FNN):** Ingesta paralela del vector de características ABCDE extraído previamente.
- **Capa de Fusión:** Concatenación de ambos *embeddings*, procesados por una capa densa con regularización por `Dropout` (30%) y una salida de divergencia no lineal sigmoide.

### Fase 3: Optimización y Mitigación de Sesgos
- **Class Weights:** Se descartó la generación de píxeles sintéticos (SMOTE). El desbalanceo (11,700 imágenes en total) se atacó directamente desde el optimizador, aplicando penalizaciones asimétricas (`class_weight='balanced'`) en el motor de *Backpropagation*.
- **Data Augmentation Estocástico:** Rotaciones de ±15°, traslaciones de ±10%, zoom y reflexión aplicadas en vuelo mediante `ImageDataGenerator` para evitar el sobreajuste.

---

## Resultados y Trade-Off Clínico

El modelo fue calibrado con un umbral de probabilidad personalizado (~0.47) para maximizar la detección de la clase minoritaria (Maligno), asumiendo empíricamente un aumento de falsos positivos en favor de la seguridad del paciente.

* **Sensibilidad (Recall): 75.14%** (El modelo detecta de forma autónoma a 3 de cada 4 pacientes con melanoma maligno).
* **Precisión en Positivos: 43.26%** (Trade-off intencionado: es preferible recomendar biopsias preventivas a omitir un diagnóstico letal).
* **ROC-AUC: 0.8431** (Demuestra una robusta capacidad intrínseca para discriminar clases a pesar de la varianza topográfica).
* **Exactitud Media Global (Accuracy): ~76%**.

## Stack Tecnológico
* **Deep Learning:** TensorFlow, Keras (Functional API, Callbacks: EarlyStopping, ReduceLROnPlateau).
* **Computer Vision:** OpenCV (`cv2`), PIL (Pillow).
* **Data Processing & ML:** Scikit-Learn (Escalado Z-Score, Métricas ROC), Pandas, NumPy.
* **Persistencia:** `.h5` (Pesos de red) y `joblib` (Parámetros estadísticos).

---

## Cómo ejecutar en local

Para replicar el entorno, probar inferencias o evaluar el preprocesamiento ABCDE:

1. Clona el repositorio:
```bash
git clone https://github.com/tu-usuario/oncology-melanoma-cnn.git
cd oncology-melanoma-cnn
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

3. Ejecuta el pipeline principal (entrenamiento o inferencia según el script que hayas preparado):
```bash
python main_pipeline.py
```
