# Installation Guide - Brain MRI Classification Pipeline

This guide outlines setup instructions across Windows, Linux, and Docker environments.

---

## 1. Local Environment Setup

### Prerequisites
- **Python**: Version 3.10.x.
- **Git**: Installed.

### Step 1: Clone and Navigate
```bash
git clone <repository_url>
cd "MRI brain tumor detection"
```

### Step 2: Virtual Environment Setup
- **Windows**:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
- **Linux/macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

### Step 3: Install Package Dependencies
Using the Makefile:
```bash
make install
```
Or manually:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

## 2. Docker Deployment Setup

### Option 1: Standalone Container
Build the docker image:
```bash
docker build -t mri_classifier:latest .
```
Launch the container:
```bash
docker run -p 8501:8501 mri_classifier:latest
```

### Option 2: Docker Compose (Recommended)
Launch the service in detached mode:
```bash
docker compose up -d
```
Access the application dashboard at: `http://localhost:8501`.
To shut down the container service:
```bash
docker compose down
```
