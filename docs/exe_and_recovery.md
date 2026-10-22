# exe 실행과 Streamlit 연결 복구

## exe로 서버 실행

현재 모델이 정상 동작하는 Windows Conda 환경에서 빌드합니다.

```powershell
conda activate rtm_env
python -m pip install pyinstaller
.\build_socket_server.ps1
```

결과는 `dist/ASANSocketServer/ASANSocketServer.exe`입니다. **ASANSocketServer 폴더 전체**를 배포하고 그 안의 exe를 실행하세요. exe 파일 하나만 복사하면 의존 라이브러리와 모델이 빠집니다. 실행하면 `ws://127.0.0.1:8080` 소켓을 열고 AI 모델을 준비합니다. 측정 시작 명령을 받기 전까지 카메라를 열거나 계속 추론하지 않습니다. 모델 준비 전에는 기존과 같이 준비 상태 메시지가 옵니다.

기본 로그는 exe 옆 `data`에 기록합니다. 해당 위치는 쓰기 가능한 폴더여야 합니다. `ASAN_LOG_ROOT` 환경 변수 또는 시작 JSON의 `log_root`로 변경할 수 있습니다. 모델과 운동 설정은 배포 폴더의 `_internal/configs`에 포함됩니다. 실행 중인 원본 프로젝트의 JSON을 수정해도 이미 빌드한 배포본에는 반영되지 않습니다.

PyTorch·CUDA 의존 파일의 크기 때문에 폴더 배포 방식을 사용합니다. 단일 exe 방식은 실행할 때 임시 폴더에 파일을 풀므로 시작이 더 느릴 수 있습니다. [PyInstaller 공식 설명](https://pyinstaller.org/en/stable/operating-mode.html)

빌드 스크립트에는 동적으로 가져오는 OpenMMLab 모듈·설정 자료의 수집 옵션을 넣었습니다. 실제 exe 빌드 및 배포 PC의 GPU·DLL 호환성은 별도 확인이 필요합니다. Python 단위 테스트 통과가 배포본의 실제 추론 성공을 보장하지는 않습니다.

## Streamlit 리셋 후 SERVER_BUSY

서버 프로세스가 켜져 있는 것과 측정 연결이 남아 있는 것은 별개입니다. 서버는 한 연결만 카메라·측정 자원을 소유하게 합니다. 화면이 리셋되어도 이전 통신 스레드가 살아 있으면 새 화면의 시작 요청에 `SERVER_BUSY`가 올 수 있습니다.

현재 Streamlit 클라이언트는 화면 생존 신호가 약 10초 끊기면 중지 명령을 보내도록 되어 있습니다. 하지만 즉시 다시 시작하거나 이전 작업이 정리되지 않으면 수동 복구가 필요할 수 있습니다.

1. Streamlit 화면을 다시 엽니다.
2. 사이드바에서 서버 주소를 확인합니다.
3. **이전 서버 연결 정리**를 누릅니다.
4. 정리 완료 메시지를 받은 뒤 세션을 시작합니다.

버튼은 이전 측정 연결을 닫고, 기존 핸들러가 로그를 저장하고 카메라를 해제할 때까지 기다립니다. 소켓 서버와 모델은 유지합니다. 다른 창에서 정상 측정 중이어도 그 측정이 종료되므로 복구가 필요한 경우에만 사용하세요. 이전 세션은 정상 완료가 아닌 `STOPPED`로 마무리됩니다. 완료 응답 없이 오류가 나면 정리가 끝난 것으로 간주하지 마세요.

Unity 등에서도 **새 WebSocket 연결**로 다음을 보낼 수 있습니다.

```json
{"type":"CMD_RESET_SESSION","request_id":"recover-1"}
```

완료 응답의 주요 필드입니다. 초기 `SERVER_STATUS`는 건너뛰고 이 응답을 기다립니다.

```json
{"type":"SERVER_SESSION_RESET","reset":true,"request_id":"recover-1","message":"이전 세션 연결 정리를 완료했습니다."}
```

`reset:false`는 정리할 측정 연결이 없었다는 뜻입니다. `SESSION_RESET_TIMEOUT`은 정리가 아직 끝나지 않았다는 뜻이며 잠시 뒤 재시도합니다. 측정 소유 연결에서 복구 명령을 보내면 `RESET_FROM_MONITOR_REQUIRED`입니다. 같은 측정 연결에서는 기존 `CMD_STOP_SESSION`을 사용하세요.

## 포트가 이미 사용 중인 경우

exe를 두 번 실행했을 때의 8080 포트 충돌은 `SERVER_BUSY`와 다릅니다. 기존 서버가 정상 실행 중이면 새 서버를 띄울 필요 없이 기존 주소에 연결하세요. `SERVER_BUSY`를 해결하려고 exe를 중복 실행하지 마세요.
