Write-Host "====================================" -ForegroundColor Cyan
Write-Host "Brain MRI Tumor Classification" -ForegroundColor Cyan
Write-Host "Production Version 1.0" -ForegroundColor Cyan
Write-Host "Starting Application..." -ForegroundColor Cyan
Write-Host "====================================" -ForegroundColor Cyan

# [1/10] Project Directory
Write-Host "[1/10] Project Directory..." -ForegroundColor Green
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir
Write-Host "Working Directory: $scriptDir"

# [2/10] Python
Write-Host "[2/10] Python..." -ForegroundColor Green
python --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAILED"
    Write-Host "Reason: python command failed." -ForegroundColor Red
    Write-Host "Suggested Fix: Install Python 3.10 and check 'Add Python to PATH'." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    Exit 1
}

# [3/10] pip
Write-Host "[3/10] pip..." -ForegroundColor Green
python -m pip --version
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAILED"
    Write-Host "Reason: pip is missing or broken." -ForegroundColor Red
    Write-Host "Suggested Fix: Reinstall Python or run 'python -m ensurepip'." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    Exit 1
}

# [4/10] Virtual Environment
Write-Host "[4/10] Virtual Environment..." -ForegroundColor Green
$venvPath = $null
if (Test-Path ".venv/Scripts/Activate.ps1") {
    $venvPath = ".venv"
} elseif (Test-Path "venv/Scripts/Activate.ps1") {
    $venvPath = "venv"
} elseif (Test-Path "env/Scripts/Activate.ps1") {
    $venvPath = "env"
}

if ($null -eq $venvPath) {
    Write-Host "Virtual environment folder not found. Creating (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED"
        Write-Host "Reason: Failed to create virtual environment." -ForegroundColor Red
        Write-Host "Suggested Fix: Check directory permissions." -ForegroundColor Yellow
        Read-Host "Press Enter to exit..."
        Exit 1
    }
    $venvPath = ".venv"
}

Write-Host "Activating virtual environment: $venvPath"
. "$venvPath/Scripts/Activate.ps1"
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAILED"
    Write-Host "Reason: Failed to activate virtual environment." -ForegroundColor Red
    Write-Host "Suggested Fix: Verify that Scripts/Activate.ps1 exists inside $venvPath." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    Exit 1
}

# [5/10] requirements.txt
Write-Host "[5/10] requirements.txt..." -ForegroundColor Green
if (-not (Test-Path "requirements.txt")) {
    Write-Error "FAILED"
    Write-Host "Reason: requirements.txt is missing from the directory." -ForegroundColor Red
    Write-Host "Suggested Fix: Put the requirements.txt file in the root folder." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    Exit 1
}
Write-Host "Verified requirements.txt exists."

# [6/10] Streamlit
Write-Host "[6/10] Streamlit..." -ForegroundColor Green
try {
    python -c "import streamlit" 2>$null
    $streamlitInstalled = ($LASTEXITCODE -eq 0)
} catch {
    $streamlitInstalled = $false
}

if (-not $streamlitInstalled) {
    Write-Host "Streamlit is missing. Installing requirements from requirements.txt..." -ForegroundColor Yellow
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED"
        Write-Host "Reason: Failed to install package dependencies." -ForegroundColor Red
        Write-Host "Suggested Fix: Check internet connection or pip proxy configuration." -ForegroundColor Yellow
        Read-Host "Press Enter to exit..."
        Exit 1
    }
}
python -m streamlit --version

# [7/10] TensorFlow
Write-Host "[7/10] TensorFlow..." -ForegroundColor Green
python -c "import tensorflow, numpy, cv2, sklearn, PIL"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[WARNING] Dependencies validation check failed. Attempting reinstall..." -ForegroundColor Yellow
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Error "FAILED"
        Write-Host "Reason: Dependencies installation failed." -ForegroundColor Red
        Write-Host "Suggested Fix: Install package dependencies manually." -ForegroundColor Yellow
        Read-Host "Press Enter to exit..."
        Exit 1
    }
}
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"

# [8/10] app/streamlit_app.py
Write-Host "[8/10] app/streamlit_app.py..." -ForegroundColor Green
$streamlitEntry = python -c "from pathlib import Path; print([p for p in ['app/streamlit_app.py', 'streamlit_app.py', 'main.py', 'app.py'] if Path(p).exists()][0])"
if ([string]::IsNullOrEmpty($streamlitEntry)) {
    Write-Error "FAILED"
    Write-Host "Reason: Streamlit entrypoint script is missing." -ForegroundColor Red
    Write-Host "Suggested Fix: Restore folder structures and place streamlit_app.py inside app/ directory." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    Exit 1
}
Write-Host "Detected Streamlit Entrypoint: $streamlitEntry"

# Verify imports and script initialization
Write-Host "Verifying imports and script initialization..." -ForegroundColor Green
python -c "import sys; sys.path.append('.'); import app.streamlit_app"
if ($LASTEXITCODE -ne 0) {
    Write-Error "FAILED"
    Write-Host "Reason: Pre-check import test failed." -ForegroundColor Red
    Write-Host "Suggested Fix: Check traceback above for package compatibility issues." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    Exit 1
}
Write-Host "Pre-check imports check passed."

# [9/10] saved_models/best_model.keras
Write-Host "[9/10] saved_models/best_model.keras..." -ForegroundColor Green
$modelFilePath = python -c "from src import config; print(config.MODEL_PATH)"
if (-not (Test-Path $modelFilePath)) {
    Write-Error "FAILED"
    Write-Host "Reason: Model file is missing." -ForegroundColor Red
    Write-Host "Suggested Fix: Please train the model first by running scripts/run_training.py." -ForegroundColor Yellow
    Read-Host "Press Enter to exit..."
    Exit 1
}
Write-Host "Detected Saved Model: $modelFilePath"

# [10/10] Starting Application
Write-Host "[10/10] Starting Application..." -ForegroundColor Green
Start-Process cmd -ArgumentList "/c timeout /t 3 && start http://localhost:8501"

python -m streamlit run $streamlitEntry

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Streamlit application crashed or exited with error code $LASTEXITCODE." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    Exit $LASTEXITCODE
}
