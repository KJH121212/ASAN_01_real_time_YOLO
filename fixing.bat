@echo off
chcp 65001 >nul

set ENV_NAME=rtm_env_50

echo =======================================================
echo [1/4] Conda environment setup (Python 3.10)
echo =======================================================
call conda deactivate >nul 2>&1
call conda env remove -n %ENV_NAME% -y >nul 2>&1
call conda create -n %ENV_NAME% python=3.10 -y
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [2/4] Installing PyTorch (CUDA 12.8 for RTX 50 Series sm_120)
echo =======================================================
call conda run -n %ENV_NAME% python -m pip install "numpy==1.26.4" "setuptools<70.0.0"
if errorlevel 1 goto ERROR_EXIT

:: RTX 5060(Blackwell) sm_120 아키텍처 연산 지원을 위해 cu128 PyTorch 설치
call conda run -n %ENV_NAME% python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [3/4] Installing OpenMMLab Packages
echo =======================================================
call conda run -n %ENV_NAME% python -m pip install "mmengine>=0.8.0"
if errorlevel 1 goto ERROR_EXIT

:: OpenMMLab MMCV 설치
call conda run -n %ENV_NAME% python -m pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
if errorlevel 1 goto ERROR_EXIT

call conda run -n %ENV_NAME% python -m pip install "mmdet==3.2.0"
if errorlevel 1 goto ERROR_EXIT

call conda run -n %ENV_NAME% python -m pip install "chumpy==0.70" --no-build-isolation
if errorlevel 1 goto ERROR_EXIT

call conda run -n %ENV_NAME% python -m pip install "mmpose==1.3.2"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [4/4] Installing Remaining Dependencies
echo =======================================================
call conda run -n %ENV_NAME% python -m pip install "opencv-python<4.10.0" ultralytics==8.3.204 streamlit==1.55.0 pycocotools xtcocotools pandas matplotlib scipy shapely PyYAML requests pillow tqdm click rich "numpy<2.0.0" websockets
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo VERIFYING ENVIRONMENT AND GPU TENSOR COMPUTATION...
echo =======================================================
:: GPU 텐서 연산(sm_120 지원 여부)과 MMCV 로딩 동시 검증
call conda run -n %ENV_NAME% python -c "import torch, mmcv, mmpose; x = torch.randn(2, 3).cuda(); print('PyTorch:', torch.__version__, '| GPU:', torch.cuda.get_device_name(0)); print('GPU Tensor Calc Test:', (x+x).sum().item()); print('MMCV:', mmcv.__version__, '| MMPose:', mmpose.__version__)"
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