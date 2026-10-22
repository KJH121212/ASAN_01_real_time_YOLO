# 현재 스크립트가 있는 프로젝트 폴더에서 빌드합니다.
Set-Location -LiteralPath $PSScriptRoot
# 오류가 나도 배치가 성공한 것처럼 계속하지 않습니다.
$ErrorActionPreference = 'Stop'
# AI 패키지가 설치된 현재 Python 환경에 PyInstaller가 있어야 합니다.
python -c "import PyInstaller"
# 패키지를 자동 설치하거나 기존 환경을 수정하지 않고 필요한 설치 명령을 안내합니다.
if ($LASTEXITCODE -ne 0) { throw '현재 환경에서 python -m pip install pyinstaller 를 실행한 뒤 다시 빌드하세요.' }
# 실제 배포에 필요한 모델 파일이 모두 있는지 빌드 전에 확인합니다.
foreach ($modelFile in @('yolov8n.pt', 'RTMPose.pth', 'RTMPose_config.py')) {
    # 프로젝트의 체크포인트 폴더에서 해당 파일을 확인합니다.
    if (-not (Test-Path -LiteralPath (Join-Path 'configs/checkpoints' $modelFile) -PathType Leaf)) {
        # 모델이 빠진 실행 파일을 생성하지 않고 누락 경로를 안내합니다.
        throw "모델 파일이 없습니다: configs/checkpoints/$modelFile"
    }
}
# 동적으로 등록되는 OpenMMLab 모듈과 설정 자료를 함께 수집합니다.
# 폴더 배포 방식은 실행 시 대형 CUDA 파일을 매번 임시 폴더에 압축 해제하지 않습니다.
python -m PyInstaller --noconfirm --onedir --console --name ASANSocketServer --paths . --add-data 'configs;configs' --collect-all mmpose --collect-all mmengine --collect-all mmdet --collect-all mmcv --collect-all ultralytics server_entry.py
# 빌드 오류를 확인하여 성공 경로를 잘못 표시하지 않습니다.
if ($LASTEXITCODE -ne 0) { throw 'exe 빌드에 실패했습니다. 위 PyInstaller 오류를 확인하세요.' }
# exe와 의존 파일이 들어 있는 폴더 전체를 배포해야 합니다.
Write-Host '빌드 완료: dist/ASANSocketServer/ASANSocketServer.exe (폴더 전체를 함께 배포하세요.)'
