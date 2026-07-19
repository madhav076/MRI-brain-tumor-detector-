# Quick Start Guide - Brain MRI Classification

This guide gets you up and running with training, evaluating, and running the Streamlit app.

---

## 1. Quick Development Commands

Use the `Makefile` shortcuts to run commands instantly:

- **Initialize Codebase**:
  ```bash
  make install
  ```
- **Reformat Code (Black & isort)**:
  ```bash
  make format
  ```
- **Lint Codebase (Flake8 & mypy)**:
  ```bash
  make lint
  ```
- **Run Unit Tests (Pytest with Coverage)**:
  ```bash
  make test
  ```

---

## 2. Pipeline Execution Commands

### Step 1: Place Scans
Download the Brain MRI dataset and place the folders under the target paths:
```text
dataset/train/
dataset/validation/
dataset/test/
```
*(Ensure class subdirectories `glioma`, `meningioma`, `pituitary`, `notumor` exist inside each split).*

### Step 2: Run Training
Runs Phase 1 (classification head) and Phase 2 (backbone fine-tuning):
```bash
make train
```

### Step 3: Run Evaluation & Diagnostics
Generates performance metrics, confusion matrices, ROC charts, reliability curves, and HTML reports:
```bash
make evaluate
```

### Step 4: Run Streamlit Web Application
Launches the interactive medical dashboard:
```bash
make run
```
Open your browser and navigate to: `http://localhost:8501`.
