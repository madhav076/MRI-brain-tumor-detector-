# Model Card - Brain MRI Tumor Classifier

This Model Card documents the diagnostic capabilities, training configurations, and ethical details of the Brain MRI Tumor Classifier.

---

## 1. Intended Use

- **Intended Patient Profile**: Adult patients undergoing clinical T1-weighted, T2-weighted, or FLAIR brain MRI slice scans.
- **Intended Environment**: Clinical researchers, MLOps testing environments, and educational demonstrations.
- **Out-of-Scope Usage**: This model is **not clinically certified** and must **not** be used for autonomous medical decision support or diagnostic evaluations in healthcare environments.

---

## 2. Model Specifications

- **Model Name**: Brain MRI Tumor Classifier
- **Architecture**: EfficientNetB0 Transfer Learning (ImageNet weights)
- **Modality**: Grayscale 2D MRI scans (rescaled, converted to 3-channel RGB)
- **Input Dimensions**: `224 x 224 x 3`
- **Output Target**: Softmax probability over 4 classes:
  1. `glioma`
  2. `meningioma`
  3. `pituitary`
  4. `notumor` (normal)

---

## 3. Training & Validation Pipelines

- **Optimizer**: Adam (learning rate = `0.0001` for Phase 1; `0.00001` for Phase 2).
- **Loss Scheme**: Categorical Crossentropy.
- **Batch Size**: 32.
- **Mixed Precision**: Supported (`mixed_float16` policies automatically applied on GPU).
- **Two-Phase Transfer Learning**:
  - **Phase 1**: Backbone frozen, custom head trained for 10 epochs.
  - **Phase 2**: Unfreezes last 25 layers of EfficientNet, fine-tuning for 15 epochs (Batch Normalization layers remain frozen).
- **Imbalance Mitigation**: Dynamic class weighting calculated via `compute_class_weight` from scikit-learn.

---

## 4. Evaluation & Diagnostic Diagnostics

- **Tracking Metrics**: Accuracy, Balanced Accuracy, Precision, Recall, F1-score, and Matthews Correlation Coefficient (MCC).
- **Calibration Checks**: Reliability diagrams tracking Expected Calibration Error (ECE) and Maximum Calibration Error (MCE) across 10 confidence bins.
- **Explainability**: Grad-CAM activation heatmaps showing localized feature highlights relative to predicted labels.

---

## 5. Limitations & Ethical Warnings

- **Data Limitations**: The model's diagnostic accuracy is sensitive to scan orientation (axial vs. sagittal slices) and image contrast settings.
- **Ethical Risks**: Diagnostic overconfidence is mitigated by computing ECE/MCE. Users must always cross-reference predictions with certified radiologists.
