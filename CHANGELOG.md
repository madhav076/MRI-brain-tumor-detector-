# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] - 2026-07-17

### Added
- **Module 1**: Created structured workspace, requirements, `.gitignore`, and configuration manager (`config.py`).
- **Module 1**: Implemented `dataset_loader.py` scanning datasets and printing statistics tables.
- **Module 1**: Implemented TF-based `preprocessing.py` and Keras `augmentation.py` with custom shear layers and medical justifications.
- **Module 1**: Designed comprehensive `EDA.ipynb` Jupyter Notebook.
- **Module 2**: Created compiled EfficientNetB0 classification model (`efficientnet_model.py`) with GPU-accelerated online augmentations.
- **Module 2**: Created high-performance `tf.data` training loop (`train.py`) including two-phase transfer learning, mixed precision, and class weight adjustments.
- **Module 2**: Automated saving model summaries, training configurations JSON, history csv/pkl files, and plotting training graphs.
- **Module 3**: Created metrics evaluator (`evaluate.py`) calculating F1, accuracy, balanced accuracy, MCC, kappa, and ECE/MCE calibration.
- **Module 3**: Created explainability framework (`base_explainer.py` & `gradcam.py`) performing dynamic final conv layer detection and heatmaps overlaying.
- **Module 3**: Generated failure dashboards (HTML/CSV), ROC and PR curves, and reliability diagrams.
- **Module 4**: Created Streamlit dashboard (`streamlit_app.py`) with sidebar navigations, single/batch file uploaders, demo selectors, prediction cards, interactive Grad-CAM sliders, history tables, and PDF report exporters.
- **Module 5**: Package setup: CI/CD actions, Docker configurations, pre-commit styling hooks, CLI launchers (`scripts/`), code coverage, model cards, and architecture diagrams.
