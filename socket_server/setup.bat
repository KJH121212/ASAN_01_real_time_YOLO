@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=%~dp0.venv"
set "BASE_PYTHON="

for %%D in (
    "C:\miniconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\anaconda3"
    "C:\anaconda3"
    "C:\ProgramData\miniconda3"
    "C:\ProgramData\anaconda3"
) do (
    if not defined BASE_PYTHON if exist "%%~D\envs\rtm_env_50\python.exe" set "BASE_PYTHON=%%~D\envs\rtm_env_50\python.exe"
)
if not defined BASE_PYTHON (
    for %%D in (
        "C:\miniconda3"
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\anaconda3"
        "C:\anaconda3"
        "C:\ProgramData\miniconda3"
        "C:\ProgramData\anaconda3"
    ) do (
        if not defined BASE_PYTHON if exist "%%~D\python.exe" set "BASE_PYTHON=%%~D\python.exe"
    )
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
    echo [1/4] Creating virtual environment: .venv ...
    "%BASE_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
) else (
    echo [1/4] Reusing existing .venv environment.
)

echo [2/4] Installing ONNX Runtime + CUDA 12 packages...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [3/4] Verifying ONNX engine...
"%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.path.insert(0,'.'); from core.skeleton_engine import SkeletonEngine; eng = SkeletonEngine(); print('Active providers:', eng.active_providers)"
if errorlevel 1 goto :fail

echo [4/4] Checking FFmpeg and scrcpy (GoPro / Android USB camera)...
set "FFMPEG_FOUND="
where ffmpeg >nul 2>&1 && set "FFMPEG_FOUND=1"
if not defined FFMPEG_FOUND if exist "configs\bin\ffmpeg.exe" set "FFMPEG_FOUND=1"
if not defined FFMPEG_FOUND if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe" set "FFMPEG_FOUND=1"
if not defined FFMPEG_FOUND (
    for /f "delims=" %%F in ('dir /b /s "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg*\ffmpeg.exe" 2^>nul') do set "FFMPEG_FOUND=1"
)
if not defined FFMPEG_FOUND (
    echo [INFO] ffmpeg not found. Installing Gyan.FFmpeg via winget...
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo [WARN] winget install Gyan.FFmpeg failed. Install manually or place ffmpeg.exe in configs\bin\.
    ) else (
        echo [OK] Gyan.FFmpeg installed.
    )
) else (
    echo [OK] ffmpeg found.
)

set "SCRCPY_FOUND="
where scrcpy >nul 2>&1 && set "SCRCPY_FOUND=1"
if not defined SCRCPY_FOUND if exist "configs\bin\scrcpy.exe" set "SCRCPY_FOUND=1"
if not defined SCRCPY_FOUND if exist "..\tmp\scrcpy\scrcpy.exe" set "SCRCPY_FOUND=1"
if not defined SCRCPY_FOUND if exist "%LOCALAPPDATA%\Microsoft\WinGet\Links\scrcpy.exe" set "SCRCPY_FOUND=1"
if not defined SCRCPY_FOUND (
    for /f "delims=" %%F in ('dir /b /s "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Genymobile.scrcpy*\scrcpy.exe" 2^>nul') do set "SCRCPY_FOUND=1"
)
if not defined SCRCPY_FOUND (
    echo [INFO] scrcpy not found. Installing Genymobile.scrcpy via winget...
    winget install -e --id Genymobile.scrcpy --accept-source-agreements --accept-package-agreements
    if errorlevel 1 (
        echo [WARN] winget install Genymobile.scrcpy failed. For Android USB camera, run: winget install Genymobile.scrcpy
    ) else (
        echo [OK] Genymobile.scrcpy installed.
    )
) else (
    echo [OK] scrcpy found.
)

"%VENV_DIR%\Scripts\python.exe" -c "import sys; sys.path.insert(0,'.'); from network.camera import detect_android_usb_device; dev = detect_android_usb_device(); print('[OK] Android USB:', dev if dev else 'not connected')"

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
