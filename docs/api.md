# Developer API Documentation - Brain MRI Tumor Classification

This document provides developer reference documentation for the key modules and classes of the project.

---

## 1. Configurations (`src/config.py`)
Parses settings from `configs/config.yaml` and exposes them as Python constants.

- `IMAGE_SIZE`: Target dimensions `(height, width)` as `(224, 224)`.
- `BATCH_SIZE`: Training batch size (default: `32`).
- `LEARNING_RATE`: Phase 1 learning rate (default: `0.0001`).
- `FINE_TUNE_EPOCHS`: Number of epochs for Phase 2 fine-tuning.
- `MIXED_PRECISION`: True to enable TensorFlow mixed precision.

---

## 2. Dataset Loader (`src/data/dataset_loader.py`)
Class `MRIDatasetLoader` manages scanning data folders.

### Methods
- `__init__(self, dataset_path: str)`: Initializes path and files statistics dicts.
- `validate_structure(self) -> bool`: Verifies existence of `train`, `validation`, and `test` splits.
- `scan_dataset(self) -> Dict[str, pd.DataFrame]`: Walks directories, logs unreadable/empty files, and returns metadata.
- `get_dataset_statistics(self) -> Dict[str, Any]`: Calculates mean intensity, standard deviation, and class counts.
- `print_summary(self)`: Prints summary report.

---

## 3. Preprocessing (`src/data/preprocessing.py`)
TensorFlow-based functions for tensor standardization.

### Functions
- `validate_image(image) -> bool`: Checks rank, empty sizes, and NaN/Inf values (only on floating-point tensors).
- `convert_to_rgb(image) -> tf.Tensor`: Replicates single-channel grayscale arrays into 3-channel RGB.
- `resize_image(image, target_size) -> tf.Tensor`: Resizes tensors using bilinear interpolation.
- `normalize_image(image, method) -> tf.Tensor`: Normalizes pixel intensities (supports Min-Max, Z-score).

---

## 4. Augmentation Pipeline (`src/data/augmentation.py`)
Integrated Keras layer `MRIAugmentationPipeline` executing GPU augmentations during training.

### Classes
- `RandomShear(shear_factor)`: Custom layer applying projective shearing matrices.
- `MRIAugmentationPipeline(image_size, ...)`: Sequential layer wrapping rotation, flip, zoom, translations, contrast, and shear.

---

## 5. Model Builder (`src/models/efficientnet_model.py`)
Handles building the model.

### Functions
- `build_model(input_shape, num_classes, learning_rate) -> tf.keras.Model`: Instantiates the pre-trained EfficientNetB0 backbone, appends head layers, and compiles the model.

---

## 6. Training pipeline (`src/training/train.py`)
Coordinates two-phase training.

### Functions
- `configure_mixed_precision()`: Sets `mixed_float16` policies.
- `create_tf_dataset(df, classes, batch_size, is_training) -> tf.data.Dataset`: Builds cached and prefetched `tf.data` datasets.
- `calculate_class_weights(df, classes) -> Dict[int, float]`: Computes scikit-learn class balances.
- `fine_tune_backbone(model) -> tf.keras.Model`: Unfreezes final convolutional layers.

---

## 7. Explainability Framework (`src/evaluation/explainability/`)
Renders predictions interpretations.

### Classes
- `BaseExplainer(ABC)`: Abstract explainer interface.
- `GradCAMExplainer(BaseExplainer)`: Calculates class score gradients relative to final conv feature maps.
