@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "BASE_PYTHON="

for %%P in (
    "C:\miniconda3\envs\rtm_env_50\python.exe"
    "%USERPROFILE%\miniconda3\envs\rtm_env_50\python.exe"
    "C:\miniconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
) do (
    if not defined BASE_PYTHON if exist "%%~P" set "BASE_PYTHON=%%~P"
)
if not defined BASE_PYTHON (
    where python >nul 2>&1
    if not errorlevel 1 set "BASE_PYTHON=python"
)

if not defined BASE_PYTHON (
    echo [ERROR] Python 3.10+ not found.
    pause
    exit /b 1
)

echo ==================================================
echo   ASAN Socket Server (COCO-23) - Pure ONNX Setup
echo   Base Python: %BASE_PYTHON%
echo ==================================================

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/3] Creating virtual environment: .venv ...
    "%BASE_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
) else (
    echo [1/3] Reusing existing .venv environment.
)

echo [2/3] Installing ONNX Runtime + CUDA 12 packages...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [3/3] Verifying ONNX engine and FFmpeg...
"%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.path.insert(0,'.'); from core.skeleton_engine import SkeletonEngine; eng = SkeletonEngine(); print('Active providers:', eng.active_providers)"
if errorlevel 1 goto :fail

where ffmpeg >nul 2>&1
if errorlevel 1 (
    if not exist "configs\bin\ffmpeg.exe" (
        echo [WARN] ffmpeg not found. For GoPro Webcam support, run: winget install Gyan.FFmpeg
    ) else (
        echo [OK] Bundled configs\bin\ffmpeg.exe found.
    )
) else (
    echo [OK] System ffmpeg found.
)

echo.
echo ==================================================
echo   Setup complete! Run run_server.bat to start.
echo ==================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed.
pause
exit /b 1
