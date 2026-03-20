# Oncology Diagnosis - Multimodal Deep Learning & ABCDE CNN

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

> **Oncology Diagnosis** is a multimodal Deep Learning system designed for the early detection of melanomas. It outperforms traditional architectures by fusing heuristic computer vision (Clinical ABCDE rule) with a Convolutional Neural Network (CNN) trained under strict mathematical penalties to prioritize sensitivity over global accuracy.

---

## The Clinical and Mathematical Challenge
Computer-aided dermatological diagnosis faces two critical problems. First, **extreme data imbalance**: in real-world diagnostics (and in medical datasets like HAM10000 or ISIC), benign lesions massively outnumber malignant ones. Second, the **cost of error**: standard Machine Learning models seek to maximize global "Accuracy," which in oncology generates an unacceptable rate of False Negatives (missed cancer cases).

## The Solution
A hybrid neural architecture built from scratch. Instead of blindly relying on generic Transfer Learning, the model parallelly ingests raw topological patterns (images) and ad-hoc extracted biomathematical variables, shifting the decision boundary to ensure no malignant anomaly goes unnoticed.

---

## Technical Architecture & Feature Engineering

### Phase 1: Biomathematical Extraction (OpenCV)
Images (resized to 128x128x3) pass through a classic computer vision pipeline that emulates the medical ABCDE protocol, generating a tabular numerical vector:
- **A (Asymmetry):** Focal projection and spatial absolute difference calculation (`cv2.absdiff`).
- **B (Borders):** Laplacian gradient operators and Canny spectral filters.
- **C (Color):** Conversion to HSV color space and visual heterogeneity analysis via unsupervised clustering (K-Means).
- **D/E (Diameter & Elevation):** Perimeter contour detection using Otsu Thresholding.

### Phase 2: Multimodal Fusion (Keras Functional API)
- **Branch A (Feature Extraction CNN):** 3 hierarchical blocks of `Conv2D` filters with `BatchNormalization` and `MaxPooling2D`, ending in a vectorized `GlobalAveragePooling2D` compressor.
- **Branch B (Tabular Input FNN):** Parallel ingestion of the previously extracted ABCDE feature vector.
- **Fusion Layer:** Concatenation of both embeddings, processed by a dense layer with `Dropout` regularization (30%) and a non-linear sigmoid divergence output.

### Phase 3: Optimization & Bias Mitigation
- **Class Weights:** Synthetic pixel generation (SMOTE) was discarded. The imbalance (11,700 total images) was addressed directly from the optimizer by applying asymmetric penalties (`class_weight='balanced'`) in the backpropagation engine.
- **Stochastic Data Augmentation:** ±15° rotations, ±10% translations, zoom, and reflection applied on-the-fly via `ImageDataGenerator` to prevent overfitting.

---

## Results & Clinical Trade-Off

The model was calibrated with a custom probability threshold (~0.47) to maximize the detection of the minority class (Malignant), empirically assuming an increase in false positives in favor of patient safety.

* **Sensitivity (Recall): 75.14%** (The model autonomously detects 3 out of every 4 patients with malignant melanoma).
* **Precision (Positive Predictive Value): 43.26%** (Intentional trade-off: recommending preventive biopsies is vastly preferable to missing a lethal diagnosis).
* **ROC-AUC: 0.8431** (Demonstrates a robust intrinsic capacity to discriminate classes despite topographic variance).
* **Global Average Accuracy: ~76%**.

## Tech Stack
* **Deep Learning:** TensorFlow, Keras (Functional API, Callbacks: EarlyStopping, ReduceLROnPlateau).
* **Computer Vision:** OpenCV (`cv2`), PIL (Pillow).
* **Data Processing & ML:** Scikit-Learn (Z-Score Scaling, ROC Metrics), Pandas, NumPy.
* **Persistence:** `.h5` (Network weights) and `joblib` (Statistical parameters).

---

## How to run locally

To replicate the environment, test inferences, or evaluate the ABCDE preprocessing:

1. Clone the repository:
```bash
git clone [https://github.com/tu-usuario/oncology-melanoma-cnn.git](https://github.com/tu-usuario/oncology-melanoma-cnn.git)
cd oncology-melanoma-cnn
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the main pipeline (training or inference depending on your setup):
```bash
python main_pipeline.py
```
