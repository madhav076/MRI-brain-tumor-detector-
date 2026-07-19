@echo off
setlocal enabledelayedexpansion

echo =====================================
echo Brain MRI Tumor Classification
echo Debug Mode Launcher
echo =====================================

rem [1/10] Project Directory
echo [1/10] Project Directory...
cd /d "%~dp0"
if %errorlevel% neq 0 (
    echo FAILED
    echo Reason: Cannot navigate to project directory: "%~dp0"
    echo Suggested Fix: Verify directory exists and has correct permissions.
    pause
    exit /b 1
)
echo Working Directory: %CD%

rem [2/10] Python
echo [2/10] Python...
set "PYTHON_CMD="
py -3.8 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_CMD=py -3.8"
) else (
    python --version >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON_CMD=python"
    )
)
if not defined PYTHON_CMD (
    echo FAILED
    echo Reason: No usable Python command was found.
    echo Suggested Fix: Install Python 3.8, 3.9, or 3.10 and check "Add Python to PATH".
    pause
    exit /b 1
)
%PYTHON_CMD% --version

rem [3/10] pip
echo [3/10] pip...
%PYTHON_CMD% -m pip --version
if %errorlevel% neq 0 (
    echo FAILED
    echo Reason: pip is missing or broken.
    echo Suggested Fix: Reinstall Python or run "%PYTHON_CMD% -m ensurepip".
    pause
    exit /b 1
)

rem [4/10] Virtual Environment
echo [4/10] Virtual Environment...
set "VENV_PATH="
if exist ".venv\Scripts\activate.bat" set "VENV_PATH=.venv"
if not defined VENV_PATH if exist "venv\Scripts\activate.bat" set "VENV_PATH=venv"
if not defined VENV_PATH if exist "env\Scripts\activate.bat" set "VENV_PATH=env"

if not defined VENV_PATH (
    echo Virtual environment folder not found (.venv, venv, env).
    echo Automatically creating virtual environment (.venv)...
    %PYTHON_CMD% -m venv .venv
    if !errorlevel! neq 0 (
        echo FAILED
        echo Reason: Failed to create virtual environment.
        echo Suggested Fix: Check directory permissions.
        pause
        exit /b 1
    )
    set "VENV_PATH=.venv"
)

echo Activating virtual environment: !VENV_PATH!
call !VENV_PATH!\Scripts\activate.bat
if !errorlevel! neq 0 (
    echo FAILED
    echo Reason: Failed to activate virtual environment.
    echo Suggested Fix: Verify Scripts\activate.bat exists.
    pause
    exit /b 1
)

python -c "import sys; ok = sys.version_info.major == 3 and sys.version_info.minor in [8, 9, 10]; sys.exit(0 if ok else 1)"
if !errorlevel! neq 0 (
    echo Existing virtual environment uses an unsupported Python for native Windows TensorFlow.
    echo Recreating .venv with the selected compatible Python...
    deactivate >nul 2>&1
    if exist ".venv" rmdir /s /q ".venv"
    %PYTHON_CMD% -m venv .venv
    if !errorlevel! neq 0 (
        echo FAILED
        echo Reason: Failed to recreate compatible .venv.
        pause
        exit /b 1
    )
    set "VENV_PATH=.venv"
    call .venv\Scripts\activate.bat
)

rem [5/10] requirements.txt
echo [5/10] requirements.txt...
if not exist "requirements.txt" (
    echo FAILED
    echo Reason: requirements.txt is missing from the directory.
    echo Suggested Fix: Put requirements.txt in the root folder.
    pause
    exit /b 1
)
echo Verified requirements.txt exists.

echo Ensuring required folders exist...
python -c "from pathlib import Path; [Path(p).mkdir(parents=True, exist_ok=True) for p in ['logs','outputs','outputs/evaluation','outputs/evaluation/reports','saved_models','app/demo_images','dataset']]"

rem [6/10] Streamlit
echo [6/10] Streamlit...
python -c "import streamlit"
if !errorlevel! neq 0 (
    echo Dependencies are missing. Installing requirements from requirements.txt...
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo FAILED
        echo Reason: Failed to install dependencies.
        echo Suggested Fix: Check internet connection.
        pause
        exit /b 1
    )
)
python -m streamlit --version

rem [7/10] TensorFlow
echo [7/10] TensorFlow...
python -c "import tensorflow, numpy, cv2, sklearn, PIL"
if !errorlevel! neq 0 (
    echo [WARNING] Dependencies validation check failed. Attempting reinstall...
    python -m pip install -r requirements.txt
    if !errorlevel! neq 0 (
        echo FAILED
        echo Reason: Dependencies installation failed.
        echo Suggested Fix: Install package dependencies manually.
        pause
        exit /b 1
    )
)
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import tensorflow as tf; raise SystemExit(0 if tf.__version__.startswith('2.10.') else 1)"
if !errorlevel! neq 0 (
    echo FAILED
    echo Reason: Installed TensorFlow is not the native Windows-compatible 2.10.x runtime required by this project.
    pause
    exit /b 1
)

rem [8/10] app/streamlit_app.py
echo [8/10] app/streamlit_app.py...
set "STREAMLIT_ENTRY="
for /f "delims=" %%i in ('python -c "from pathlib import Path; print([p for p in ['app/streamlit_app.py', 'streamlit_app.py', 'main.py', 'app.py'] if Path(p).exists()][0])"') do set "STREAMLIT_ENTRY=%%i"

if not defined STREAMLIT_ENTRY (
    echo FAILED
    echo Reason: Streamlit entrypoint script is missing.
    echo Suggested Fix: Restore files structure and place streamlit_app.py inside app\ directory.
    pause
    exit /b 1
)
echo Detected Streamlit Entrypoint: !STREAMLIT_ENTRY!

rem Detect import errors before starting (MED-14 FIX)
rem Do NOT import app.streamlit_app directly — it triggers st.set_page_config() at import time,
rem which causes a Streamlit error outside the Streamlit server context.
rem Instead, test only the critical src dependencies.
echo Verifying core src imports and script initialization...
python -c "import sys; sys.path.append('.'); from src import config; from src.data.augmentation import MRIAugmentationPipeline, RandomShear; from src.utils import set_seed; print('Core imports: OK')"
if !errorlevel! neq 0 (
    echo FAILED
    echo Reason: Core src package import test failed.
    echo Suggested Fix: Check traceback above for missing packages or path issues.
    pause
    exit /b 1
)
echo Pre-check imports check passed.

rem [9/10] saved_models/best_model.keras
echo [9/10] saved_models/best_model.keras...
set "MODEL_FILE_PATH="
for /f "delims=" %%i in ('python -c "from src import config; print(config.MODEL_PATH)"') do set "MODEL_FILE_PATH=%%i"

if not exist "!MODEL_FILE_PATH!" (
    echo WARNING: Trained model is missing at "!MODEL_FILE_PATH!".
    echo The dashboard will still launch and show training instructions until a model is created.
) else (
    echo Detected Saved Model: !MODEL_FILE_PATH!
)

rem [10/10] Starting Application
echo [10/10] Starting Application...
rem Launch default web browser in the background after a short delay
start "" cmd /c "timeout /t 3 && start http://localhost:8501"

rem Launch Streamlit using python module wrapper
python -m streamlit run !STREAMLIT_ENTRY!

if !errorlevel! neq 0 (
    echo ERROR: Streamlit dashboard application crashed or exited with error code !errorlevel!.
    pause
    exit /b !errorlevel!
)

endlocal
