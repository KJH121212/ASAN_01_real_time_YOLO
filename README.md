# ASAN 실시간 모션 측정 소켓 서버 & Streamlit 클라이언트 (v3.1 · COCO-23 ONNX)

YOLOv8n과 **RTMPose-M (COCO-23: 몸통 17개 + 발 6개 = 23개 관절)** ONNX 모델을 사용하여 실시간 운동 반복 횟수와 품질 점수를 측정하는 프로젝트입니다.  
소켓 서버(`socket_server/`)와 Streamlit 클라이언트(`streamlit_client/`)는 **완전히 분리된 독립 폴더**로 관리되며, 두 폴더는 어떠한 코드나 설정 파일도 공유하지 않습니다.  
소켓 서버는 **순수 ONNX Runtime**으로 동작하므로 **PyTorch · MMCV · MMPose · Ultralytics 설치가 전혀 필요 없습니다.**

---

## 1. 폴더 구조

```text
ASAN_01_real_time_YOLO/
├── run_all.bat                     # 서버 + 클라이언트 동시 실행
├── socket_server/                  # [독립 1] 순수 ONNX WebSocket 모션 측정 서버
│   ├── server_entry.py             # 소켓 서버 실행 진입점
│   ├── setup.bat                   # 경량 .venv 생성 및 ONNX Runtime + CUDA 12 설치
│   ├── run_server.bat              # 소켓 서버 실행
│   ├── requirements.txt            # numpy, opencv, websockets, onnxruntime-gpu + CUDA 12 DLL
│   ├── configs/
│   │   ├── exercise_kpt_config_coco.json   # 23관절 기준 운동별 측정 설정
│   │   ├── bin/ffmpeg.exe                  # GoPro DirectShow 파이프 캡처용 FFmpeg
│   │   └── checkpoints/
│   │       ├── yolov8n.onnx                # 사람 검출 ONNX 모델 (11.6 MB)
│   │       └── rtmpose_coco23.onnx         # COCO-23 포즈 추정 ONNX 모델 (52.7 MB)
│   ├── core/
│   │   ├── skeleton_engine.py      # 순수 ONNX Runtime 23관절 추론 엔진 (단일/배치)
│   │   ├── motion_engine.py        # 각도·좌표·길이 계산 및 반복·품질 판정 FSM
│   │   ├── session_controller.py   # 캘리브레이션(3회)·본측정 세션 제어
│   │   └── data_manager.py         # CSV · 23관절 NPZ · MP4 · 임계값 JSON 저장
│   ├── network/
│   │   ├── socket_server.py        # WebSocket 서버 (LIGHTWEIGHT / VISUAL / OFFLINE_VIDEO)
│   │   └── camera.py               # 웹캠 및 GoPro(DirectShow FFmpeg / USB UDP) 캡처
│   └── utils/
│       ├── filters.py              # One Euro 관절 좌표 필터
│       ├── normalization.py        # 골반 원점·몸통 길이=1.0 등방성 정규화
│       └── pose_overlay.py         # 실시간 미리보기(VISUAL)용 23관절 오버레이
│
├── streamlit_client/               # [독립 2] Streamlit 제어 클라이언트 및 오프라인 로그 분석
│   ├── app.py                      # Streamlit 메인 앱 (측정 제어 + 영상→NPZ 추출 + 로그 분석)
│   ├── setup.bat                   # 경량 .venv 생성 및 패키지 설치
│   ├── run_client.bat              # 클라이언트 실행
│   ├── configs/
│   │   ├── exercise_kpt_config_coco.json   # 클라이언트·분석 전용 운동 설정
│   │   └── bin/ffmpeg.exe                  # 오버레이 MP4 H.264 인코딩용 FFmpeg
│   ├── client/
│   │   ├── live_cards.py           # 실시간 좌우 측정 카드 UI
│   │   └── video_window.py         # 실시간 미리보기 전용 OpenCV 창
│   └── analysis/
│       ├── ui.py                   # 로그 분석 탭 (시뮬레이션·실전 테스트·스켈레톤·그래프·오버레이)
│       ├── log_reader.py           # 23관절(및 구버전 17관절 호환) NPZ/CSV 로더
│       ├── metrics.py              # 측정 문맥·기준선·상태 구간 계산
│       ├── simulator.py            # 오프라인 시뮬레이션 전용 독립 엔진
│       ├── skeleton_player.py      # 23관절(발 관절 포함) HTML5 Canvas 플레이어
│       └── video_overlay.py        # 23관절 오버레이 MP4 합성 및 다운로드 UI
│
├── data/                           # 환자별 측정 로그(CSV/NPZ/MP4/JSON) 및 테스트 영상 저장 폴더
└── RTMPose_rehab/                  # COCO-23 RTMPose-M 학습 프로젝트 폴더
```

---

## 2. COCO-23 관절 번호 및 운동별 타겟 관절

### 23개 관절 인덱스 (`0~22`)
- **0~16 (Body 17)**: `0: nose`, `1~2: eye`, `3~4: ear`, `5~6: shoulder`, `7~8: elbow`, `9~10: wrist`, `11~12: hip`, `13~14: knee`, `15~16: ankle`
- **17~22 (Foot 6)**:
  - 왼쪽 발: `17: left_big_toe` (왼 엄지발가락), `18: left_small_toe` (왼 새끼발가락), `19: left_heel` (왼 발뒤꿈치)
  - 오른쪽 발: `20: right_big_toe` (오른 엄지발가락), `21: right_small_toe` (오른 새끼발가락), `22: right_heel` (오른 발뒤꿈치)

### 등록된 6종 운동 설정
| 운동 ID | 한글명 | 측정 방식 | 측정 관절 (왼 / 오른) |
| :--- | :--- | :--- | :--- |
| `biceps_curl` | 이두 굴곡 | `COORDINATE` (정규화 Y) | 손목 (`9` / `10`) |
| `shoulder_press` | 숄더 프레스 | `ANGLE` (3점 각도) | 어깨-팔꿈치-손목 (`5,7,9` / `6,8,10`) |
| `clamshell` | 클램쉘 | `ANGLE` (3점 각도) | 반대골반-동측골반-무릎 (`12,11,13` / `11,12,14`) |
| `slr` | 하지 직거상 | `COORDINATE` (정규화 Y) | **엄지발가락 (`17` / `20`)** |
| `hip_knee_flexion` | 고관절 및 슬관절 굴곡 | `ANGLE` (3점 각도) | 어깨-골반-무릎 (`5,11,13` / `6,12,14`) |
| `knee_extension` | 슬관절 신전 | `COORDINATE` (CALIB) → `RAW_COORDINATE` (MAIN) | **엄지발가락 (`17` / `20`)** |

---

## 3. 설치 및 실행 방법

두 폴더 모두 **Python `.venv` 기반 초경량 독립 환경**을 사용합니다 (Conda · PyTorch 불필요).

### 사전 준비 (다른 PC 배포 시)
1. **ONNX 모델 2개**(`yolov8n.onnx`, `rtmpose_coco23.onnx`)를 `socket_server/configs/checkpoints/`에 배치
2. **GoPro 사용 시**: GoPro Webcam 앱 설치 및 FFmpeg 설치 (`winget install Gyan.FFmpeg` 또는 `socket_server/configs/bin/ffmpeg.exe` 복사)

### ① 소켓 서버 (`socket_server/`)
- **최초 1회 설치**: `socket_server\setup.bat` 실행 (`socket_server\.venv` 생성 및 ONNX Runtime + CUDA 12 설치)
- **실행**: `socket_server\run_server.bat` (기본 주소: **`ws://127.0.0.1:8080`**)

### ② Streamlit 클라이언트 (`streamlit_client/`)
- **최초 1회 설치**: `streamlit_client\setup.bat` 실행 (`streamlit_client\.venv` 생성)
- **실행**: `streamlit_client\run_client.bat`

### ③ 동시 실행
- 루트의 `run_all.bat` 더블클릭으로 서버와 클라이언트를 한 번에 실행합니다.
