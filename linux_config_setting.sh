#!/bin/bash
set -e  # 에러 발생 시 즉시 중단

ENV_NAME="rtm_env"

echo "======================================================="
echo "[1/4] Conda environment setup (Python 3.10)"
echo "======================================================="
# conda 명령어 활성화 (Shell 연동)
eval "$(conda shell.bash hook)"

conda deactivate > /dev/null 2>&1 || true
conda env remove -n $ENV_NAME -y > /dev/null 2>&1 || true
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo "======================================================="
echo "[2/4] Installing PyTorch 2.1.2 (CUDA 12.1) + NumPy 1.26.4"
echo "======================================================="
conda run -n $ENV_NAME pip install "numpy==1.26.4" "setuptools<70.0.0" -U openmim
conda run -n $ENV_NAME pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "======================================================="
echo "[3/4] Installing OpenMMLab (MIM, MMCV, MMPose)"
echo "======================================================="
conda run -n $ENV_NAME mim install mmengine
conda run -n $ENV_NAME pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html "numpy<2.0.0"
conda run -n $ENV_NAME mim install "mmdet==3.2.0"
conda run -n $ENV_NAME mim install "mmpose==1.3.2"

echo ""
echo "======================================================="
echo "[4/4] Installing Remaining Dependencies (Locking NumPy < 2.0.0)"
echo "======================================================="
conda run -n $ENV_NAME pip install "opencv-python<4.10.0" ultralytics==8.3.204 streamlit==1.55.0 pycocotools xtcocotools chumpy==0.70 pandas matplotlib scipy shapely PyYAML requests pillow tqdm click rich "numpy<2.0.0"

echo ""
echo "======================================================="
echo "VERIFYING ENVIRONMENT..."
echo "======================================================="
conda run -n $ENV_NAME python -c "import torch, numpy, mmcv, mmpose; print('PyTorch:', torch.__version__, '| NumPy:', numpy.__version__, '| MMCV:', mmcv.__version__, '| MMPose:', mmpose.__version__)"

echo ""
echo "======================================================="
echo "[SUCCESS] Linux environment setup completed successfully!"
echo "======================================================="