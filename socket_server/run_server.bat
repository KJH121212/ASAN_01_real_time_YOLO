@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="
for %%P in (
    "%~dp0.venv\Scripts\python.exe"
    "%~dp0.venv_onnx\Scripts\python.exe"
    "C:\miniconda3\envs\rtm_env_50\python.exe"
    "%USERPROFILE%\miniconda3\envs\rtm_env_50\python.exe"
    "%USERPROFILE%\anaconda3\envs\rtm_env_50\python.exe"
    "C:\anaconda3\envs\rtm_env_50\python.exe"
    "C:\ProgramData\miniconda3\envs\rtm_env_50\python.exe"
    "C:\ProgramData\anaconda3\envs\rtm_env_50\python.exe"
) do (
    if not defined PYTHON_EXE if exist "%%~P" set "PYTHON_EXE=%%~P"
)
if not defined PYTHON_EXE (
    where python >nul 2>&1
    if not errorlevel 1 set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [ERROR] Python environment not found. Run setup.bat first.
    pause
    exit /b 1
)

echo [INFO] Starting ASAN Socket Server (COCO-23 ONNX)
echo [INFO] Python: %PYTHON_EXE%
"%PYTHON_EXE%" server_entry.py
pause
