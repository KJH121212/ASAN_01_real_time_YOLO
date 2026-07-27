@echo off
chcp 65001 >nul
title Setup RTM Environment

set ENV_NAME=rtm_env

echo =======================================================
echo [1/3] Checking Conda availability...
echo =======================================================
call conda --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Conda is not recognized in CMD/PowerShell.
    echo Please run this script inside Anaconda Prompt.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo [2/3] Creating Python 3.10 environment (%ENV_NAME%)...
echo =======================================================
call conda create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo [3/3] Installing dependencies from requirements.txt...
echo =======================================================
if not exist "requirements.txt" (
    echo [ERROR] requirements.txt file not found in current directory!
    pause
    exit /b 1
)

call conda run -n %ENV_NAME% pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo VERIFYING ENVIRONMENT...
echo =======================================================
call conda run -n %ENV_NAME% python -c "import torch, numpy, mmcv, mmpose; print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__, '| MMCV:', mmcv.__version__, '| MMPose:', mmpose.__version__)"
if errorlevel 1 (
    echo [WARNING] Verification script failed. Please check installation logs above.
    pause
    exit /b 1
)

echo.
echo =======================================================
echo [SUCCESS] Environment setup complete!
echo To activate environment: conda activate %ENV_NAME%
echo =======================================================
pause
exit /b 0