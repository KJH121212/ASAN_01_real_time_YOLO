@echo off
chcp 65001 >nul

set ENV_NAME=rtm_env

echo =======================================================
echo [1/4] Conda environment setup (Python 3.10)
echo =======================================================
call conda deactivate >nul 2>&1
call conda env remove -n %ENV_NAME% -y >nul 2>&1
call conda create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [2/4] Installing PyTorch 2.1.2 (CUDA 12.1) + NumPy 1.26.4
echo =======================================================
call conda run -n %ENV_NAME% pip install "numpy==1.26.4" "setuptools<70.0.0" -U openmim
if errorlevel 1 goto ERROR_EXIT

call conda run -n %ENV_NAME% pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [3/4] Installing OpenMMLab (MIM, MMCV, MMPose)
echo =======================================================
call conda run -n %ENV_NAME% mim install mmengine
call conda run -n %ENV_NAME% pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html "numpy<2.0.0"
call conda run -n %ENV_NAME% mim install "mmdet==3.2.0"
call conda run -n %ENV_NAME% mim install "mmpose==1.3.2"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [4/4] Installing Remaining Dependencies (Locking NumPy < 2.0.0)
echo =======================================================
call conda run -n %ENV_NAME% pip install "opencv-python<4.10.0" ultralytics==8.3.204 streamlit==1.55.0 pycocotools xtcocotools chumpy==0.70 pandas matplotlib scipy shapely PyYAML requests pillow tqdm click rich "numpy<2.0.0"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo VERIFYING ENVIRONMENT...
echo =======================================================
call conda run -n %ENV_NAME% python -c "import torch, numpy, mmcv, mmpose; print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__, '| MMCV:', mmcv.__version__, '| MMPose:', mmpose.__version__)"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [SUCCESS] Environment setup completed successfully!
echo =======================================================
pause
exit /b 0

:ERROR_EXIT
echo.
echo =======================================================
echo [ERROR] Setup failed! Check logs above.
echo =======================================================
pause
exit /b 1