@echo off
setlocal
cd /d "%~dp0"

set "PYTHON_EXE="

if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"

if not defined PYTHON_EXE (
    for %%D in ("C:\miniconda3" "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "C:\ProgramData\miniconda3" "C:\ProgramData\anaconda3") do (
        if not defined PYTHON_EXE if exist "%%~D\envs\rtm_env_50\python.exe" set "PYTHON_EXE=%%~D\envs\rtm_env_50\python.exe"
    )
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

echo [INFO] Starting ASAN Streamlit Client (COCO-23)
echo [INFO] Python: %PYTHON_EXE%
"%PYTHON_EXE%" -m streamlit run app.py
pause
