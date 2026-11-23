@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"
set "BASE_PYTHON="

for %%D in ("C:\miniconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3") do (
    if not defined BASE_PYTHON if exist "%%~D\python.exe" set "BASE_PYTHON=%%~D\python.exe"
)
if not defined BASE_PYTHON (
    where python >nul 2>&1
    if not errorlevel 1 set "BASE_PYTHON=python"
)

if not defined BASE_PYTHON (
    echo [ERROR] Python 3.10+ not found. Please install Python or Miniconda.
    pause
    exit /b 1
)

echo ==================================================
echo   ASAN Streamlit Client Setup (.venv)
echo   Base Python: %BASE_PYTHON%
echo ==================================================

if not exist "%VENV_DIR%\Scripts\python.exe" (
    echo [1/2] Creating virtual environment (%VENV_DIR%)...
    "%BASE_PYTHON%" -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
) else (
    echo [1/2] Reusing existing virtual environment (%VENV_DIR%).
)

echo [2/2] Installing requirements.txt...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
"%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo ==================================================
echo   Setup complete! Run run_client.bat to start.
echo ==================================================
pause
exit /b 0

:fail
echo.
echo [ERROR] Setup failed.
pause
exit /b 1
