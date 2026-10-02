@echo off
chcp 65001 >nul

:: [수정 1] RTX 50 전용 가상환경 이름 지정
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
echo [2/4] Installing PyTorch (CUDA 12.8 for RTX 50 Series) + NumPy 1.26.4
echo =======================================================
call conda run -n %ENV_NAME% python -m pip install "numpy==1.26.4" "setuptools<70.0.0" openmim
if errorlevel 1 goto ERROR_EXIT

:: RTX 5060(sm_120) 지원 PyTorch cu128 설치
call conda run -n %ENV_NAME% python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [3/4] Installing OpenMMLab (Bypassing Device Guard mim.exe)
echo =======================================================
:: [수정 2] mim.exe 차단을 피하기 위해 python -m pip 방식으로 통일
call conda run -n %ENV_NAME% python -m pip install mmengine
if errorlevel 1 goto ERROR_EXIT

:: [수정 3] C++ DLL 오류 방지 및 mmdet 충돌 방지를 위해 mmcv-lite 지정
call conda run -n %ENV_NAME% python -m pip install "mmcv-lite>=2.0.0rc4,<2.2.0"
if errorlevel 1 goto ERROR_EXIT

call conda run -n %ENV_NAME% python -m pip install "mmdet==3.2.0"
if errorlevel 1 goto ERROR_EXIT

call conda run -n %ENV_NAME% python -m pip install "chumpy==0.70" --no-build-isolation
if errorlevel 1 goto ERROR_EXIT

call conda run -n %ENV_NAME% python -m pip install "mmpose==1.3.2"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo [4/4] Installing Remaining Dependencies (Locking NumPy < 2.0.0)
echo =======================================================
call conda run -n %ENV_NAME% python -m pip install "opencv-python<4.10.0" ultralytics==8.3.204 streamlit==1.55.0 pycocotools xtcocotools pandas matplotlib scipy shapely PyYAML requests pillow tqdm click rich "numpy<2.0.0"
call conda run -n %ENV_NAME% python -m pip install "websockets"
if errorlevel 1 goto ERROR_EXIT

echo.
echo =======================================================
echo VERIFYING ENVIRONMENT...
echo =======================================================
:: RTX 5060 GPU 인식 상태 확인 출력 추가
call conda run -n %ENV_NAME% python -c "import torch, numpy, mmcv, mmpose; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| MMCV:', mmcv.__version__, '| MMPose:', mmpose.__version__)"
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