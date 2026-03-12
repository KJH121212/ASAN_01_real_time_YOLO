@echo off
chcp 65001 >nul

:: Set environment path
set "ENV_DIR=%~dp0env"
set "ACTIVATE_PATH=%ENV_DIR%\Scripts\activate.bat"

:: If env exists but Scripts\activate.bat is missing, it's a "broken" environment
IF EXIST "%ENV_DIR%" (
    IF NOT EXIST "%ACTIVATE_PATH%" (
        echo [WARNING] Broken environment detected. Deleting...
        rmdir /s /q "%ENV_DIR%"
    )
)

:: Create virtual environment if it doesn't exist
IF NOT EXIST "%ENV_DIR%" (
    echo =======================================================
    echo [INFO] Creating new virtual environment...
    echo =======================================================
    :: Using -m venv
    python -m venv "%ENV_DIR%"
)

:: Re-verify and Activate
IF EXIST "%ACTIVATE_PATH%" (
    echo [INFO] Activating virtual environment...
    call "%ACTIVATE_PATH%"
) ELSE (
    echo [ERROR] Failed to create virtual environment. 
    echo [TIP] Try running: conda deactivate
    echo [TIP] Then run this script again.
    pause
    exit /b
)

:: Upgrade pip and install requirements
echo [INFO] Syncing libraries with requirements.txt...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

:: Run app
echo [INFO] Starting Streamlit app...
streamlit run app_v1_0.py

pause