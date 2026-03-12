@echo off
chcp 65001 >nul

:: Set env path inside the project folder
set "ENV_DIR=%~dp0env"

:: If the env folder does not exist, create a new one and install
IF NOT EXIST "%ENV_DIR%" (
    echo =======================================================
    echo [INFO] No virtual environment found. Starting installation using Python's native venv.
    echo =======================================================
    
    :: Create virtual environment with Python's built-in venv
    python -m venv "%ENV_DIR%"
    
    :: Activate virtual environment
    call "%ENV_DIR%\Scripts\activate.bat"
    
    :: Install packages
    echo [INFO] Installing packages from requirements.txt...
    pip install --upgrade pip
    pip install -r requirements.txt
    
    echo [INFO] Setup complete!
) ELSE (
    echo =======================================================
    echo [INFO] Activating existing virtual environment.
    echo =======================================================
    
    :: Activate virtual environment
    call "%ENV_DIR%\Scripts\activate.bat"
)

:: Run app
echo [INFO] Starting Streamlit app...
streamlit run app_v1_0.py

pause