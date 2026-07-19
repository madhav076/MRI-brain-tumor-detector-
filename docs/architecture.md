# System Architecture - Brain MRI Tumor Classification

This document outlines the software engineering and data processing layout of the Brain MRI Classifier.

---

## 1. Architectural Pipeline Diagram

```mermaid
graph TD
    A[Raw MRI Image JPG/PNG] --> B[tf.data Loader & Reader]
    B --> C[Preprocessing: Resize to 224x224, Convert to RGB, Normalize [0,1]]
    C --> D[Keras Augmentation Pipeline: Random Rotation, Flips, Translations, Shear]
    D --> E[EfficientNetB0 Backbone: Feature Extractor]
    E --> F[Classification Head: AvgPool -> BatchNorm -> Dropout -> Dense]
    F --> G[Predictions Softmax Probability Vector]
    G --> H[Evaluation Metrics: Accuracy, F1, MCC, ECE]
    G --> I[Grad-CAM: Guided Gradients Heatmap Overlay]
    G --> J[Streamlit Dashboard Interface]
```

---

## 2. Process Flow Descriptions

1. **Dataset Scanning & Loader**: The `MRIDatasetLoader` scans directories, parses sizes, detects classes, checks file header validities, and registers statistics logs.
2. **Preprocessing Pipeline**: Rescales, pads/resizes, and casts single images and batches to unified float32 tensors with range `[0, 1]`.
3. **Data Augmentation**: A Keras layer containing translation, rotation, flips, contrast, and shear matrices. Automatically skipped during inference.
4. **CNN Backbone**: EfficientNetB0 pre-trained model mapping input maps to dense feature representations.
5. **Softmax Outputs**: Classification layer predicting tumor diagnoses.
6. **Diagnostics & Explanations**: Computes metrics, ROC curves, ECE calibration charts, and overlays Grad-CAM hotspots.
7. **Streamlit App**: Unified dashboard hosting single predictions, batch classifications, settings, and PDF report exporters.
