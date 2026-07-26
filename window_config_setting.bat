@echo off
chcp 65001 >nul

set ENV_NAME=rtm_env
set CONDA_ENV_DIR=C:\Users\jihu6\anaconda3\envs\%ENV_NAME%
set PYTHON_CMD=%CONDA_ENV_DIR%\python.exe

echo =======================================================
echo [1/5] Removing existing environment (%ENV_NAME%)...
echo =======================================================
if exist "%CONDA_ENV_DIR%" (
    call conda env remove -n %ENV_NAME% -y
)

echo.
echo =======================================================
echo [2/5] Creating Python 3.10 environment...
echo =======================================================
call conda create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [3/5] Installing PyTorch 2.1.2 (CUDA 12.1)...
echo =======================================================
"%PYTHON_CMD%" -m pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [4/5] Installing MMCV 2.1.0 and OpenMMLab Core...
echo =======================================================
"%PYTHON_CMD%" -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html
if errorlevel 1 goto ERROR_EXIT

"%PYTHON_CMD%" -m pip install mmengine "mmdet>=3.1.0" "mmpose>=1.3.0"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [5/5] Locking NumPy 1.x and OpenCV 4.x...
echo =======================================================
"%PYTHON_CMD%" -m pip install "numpy==1.26.4" "setuptools<70.0.0" "opencv-python<4.10.0" pycocotools ultralytics streamlit
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo VERIFYING ENVIRONMENT...
echo =======================================================
"%PYTHON_CMD%" -c "import torch, numpy, mmcv; print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__, '| MMCV:', mmcv.__version__)"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [SUCCESS] Environment setup complete!
echo =======================================================
pause
exit /b 0

:ERROR_EXIT
echo.
echo =======================================================
echo [ERROR] Setup failed! Check the output above.
echo =======================================================
pause
exit /b 1