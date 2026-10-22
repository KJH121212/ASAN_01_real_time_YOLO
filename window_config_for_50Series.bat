REM 실행 명령 자체의 화면 출력을 끄고 진행 메시지만 보이게 합니다.
@echo off
REM 콘솔 코드 페이지를 UTF-8로 설정합니다. 명령의 상태 출력은 화면에 표시하지 않습니다.
chcp 65001 >nul

:: [수정 1] RTX 50 전용 가상환경 이름 지정
REM 이 설치 스크립트가 사용할 Conda 환경 이름을 지정합니다.
set ENV_NAME=rtm_env

REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo [1/4] Conda environment setup (Python 3.10)
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 활성화한 Conda 환경에서 빠져나옵니다. 표준 출력과 오류 출력은 숨깁니다.
call conda deactivate >nul 2>&1
REM 같은 이름의 기존 Conda 환경을 삭제합니다. 이 명령은 스크립트를 실제 실행했을 때 동작합니다.
call conda env remove -n %ENV_NAME% -y >nul 2>&1
REM 지정 이름과 Python 버전으로 새 Conda 환경을 만듭니다.
call conda create -n %ENV_NAME% python=3.10 -y
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo.
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo [2/4] Installing PyTorch (CUDA 12.8 for RTX 50 Series) + NumPy 1.26.4
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install "numpy==1.26.4" "setuptools<70.0.0" openmim
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

:: RTX 5060(sm_120) 지원 PyTorch cu128 설치
REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo.
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo [3/4] Installing OpenMMLab (Bypassing Device Guard mim.exe)
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
:: [수정 2] mim.exe 차단을 피하기 위해 python -m pip 방식으로 통일
REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install mmengine
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

:: [수정 3] C++ DLL 오류 방지 및 mmdet 충돌 방지를 위해 mmcv-lite 지정
REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install "mmcv-lite>=2.0.0rc4,<2.2.0"
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install "mmdet==3.2.0"
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install "chumpy==0.70" --no-build-isolation
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install "mmpose==1.3.2"
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo.
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo [4/4] Installing Remaining Dependencies (Locking NumPy < 2.0.0)
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install "opencv-python<4.10.0" ultralytics==8.3.204 streamlit==1.55.0 pycocotools xtcocotools pandas matplotlib scipy shapely PyYAML requests pillow tqdm click rich "numpy<2.0.0"
REM 지정 Conda 환경에 아래 패키지와 버전·다운로드 위치 조건을 적용하여 설치합니다.
call conda run -n %ENV_NAME% python -m pip install "websockets"
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo.
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo VERIFYING ENVIRONMENT...
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
:: RTX 5060 GPU 인식 상태 확인 출력 추가
REM 설치한 환경에서 라이브러리를 가져와 버전과 실행 장치 정보를 확인합니다.
call conda run -n %ENV_NAME% python -c "import torch, numpy, mmcv, mmpose; print('PyTorch:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| MMCV:', mmcv.__version__, '| MMPose:', mmpose.__version__)"
REM 직전 명령이 실패했으면 오류 종료 레이블로 이동합니다.
if errorlevel 1 goto ERROR_EXIT

REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo. 
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo [SUCCESS] Environment setup completed successfully!
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 사용자가 키를 누를 때까지 기다려 결과를 읽을 시간을 줍니다.
pause
REM 배치 실행을 종료합니다. 0은 성공, 1은 실패 종료 코드입니다.
exit /b 0

REM 앞에서 오류가 발생했을 때 이동할 레이블을 정의합니다.
:ERROR_EXIT
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo.
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo [ERROR] Setup failed! Check logs above.
REM 현재 설치 단계 또는 성공·실패 안내를 콘솔에 출력합니다.
echo =======================================================
REM 사용자가 키를 누를 때까지 기다려 결과를 읽을 시간을 줍니다.
pause
REM 배치 실행을 종료합니다. 0은 성공, 1은 실패 종료 코드입니다.
exit /b 1