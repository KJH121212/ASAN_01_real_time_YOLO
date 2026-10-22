# 소켓 서버 사용 및 JSON 입출력 명세

로그는 PatientData 공통 버전·환자·등록 차수·플레이 ID 구조를 사용합니다. 카메라 전용 확장과 Unity 연동 예시는 [PatientData 카메라 로그](patient_data_camera.md)를 참고하세요.

exe 빌드와 화면 재시작 뒤 `SERVER_BUSY`가 남을 때의 `CMD_RESET_SESSION` 사용법은 [exe 실행·연결 복구 안내](exe_and_recovery.md)를 참고하세요.

기본 연동은 [소켓 서버 간단 사용법](socket_server_quickstart.md)부터 보세요. 아래는 선택 필드와 진단 이벤트까지 포함한 상세 명세입니다.

이 문서는 현재 `network/socket_server.py`, `network/session_commands.py`, `core/session_controller.py` 구현을 기준으로 작성했습니다.
현재는 모드 2(LIGHTWEIGHT)와 모드 3(영상 파일 분석)만 지원하고, 영상은 항상 저장합니다.
저장 영상의 바·스켈레톤·카운트 표시는 [일괄 오버레이 안내](video_overlay.md)를 참고하세요.
Unity나 별도 Python 클라이언트에서 서버를 연결할 때 사용하는 안내서입니다.
예제의 환자 번호·경로·측정값·식별자는 설명용입니다. 응답 예제에서 생략한 필드는 실제 응답에 추가될 수 있습니다.

## 목차

1. [서버 역할과 실행](#1-서버-역할과-실행)
2. [연결과 세션 흐름](#2-연결과-세션-흐름)
3. [클라이언트가 보내는 입력](#3-클라이언트가-보내는-입력)
4. [서버가 보내는 출력](#4-서버가-보내는-출력)
5. [측정값·카운트·점수 해석](#5-측정값카운트점수-해석)
6. [캘리브레이션 진행 방법](#6-캘리브레이션-진행-방법)
7. [오프라인 영상 분석](#7-오프라인-영상-분석)
8. [로그와 파일](#8-로그와-파일)
9. [Python 연결 예제](#9-python-연결-예제)
10. [오류 및 구현상 주의점](#10-오류-및-구현상-주의점)

## 1. 서버 역할과 실행

### 서버가 담당하는 작업

- YOLO 사람 검출 모델과 RTMPose 관절 추출 모델 로딩 및 유지
- 서버 PC의 카메라 열기·닫기 또는 서버 PC의 영상 파일 읽기
- 관절 추출, One Euro Filter, 신체 기준 정규화
- 좌우 동작 측정, 반복 카운트, 품질 판정
- 개인 가동 범위 캘리브레이션과 결과 저장
- JSON 이벤트 및 측정값 전송, CSV·NPZ·선택적 MP4 저장

MAIN/TEST의 게임 목표 점수와 종료 시점은 **클라이언트가 결정**합니다.
서버에 `target_score`를 보낸 것만으로 자동 종료하지 않습니다.
CALIBRATION의 완료 여부는 **서버가 결정**합니다.

### 실행

모델 의존성이 설치된 Python 환경에서 프로젝트 루트를 작업 폴더로 실행합니다.

```powershell
conda activate rtm_env
python network/socket_server.py
```

기본 주소:

```text
ws://127.0.0.1:8080
```

Streamlit 테스트 화면은 별도 터미널에서 실행합니다.

```powershell
streamlit run app.py
```

기본 진입점은 `ExerciseSocketServer()`를 사용하며 host/port용 CLI 인자를 제공하지 않습니다.
바꾸려면 Python에서 `ExerciseSocketServer(host="...", port=8080)`를 구성해야 합니다.
현재 기본 host는 `127.0.0.1`이므로 같은 PC의 클라이언트 연결을 전제로 합니다.

### 모델과 카메라 수명

소켓 리스닝을 먼저 시작한 뒤 모델을 백그라운드에서 한 번 로딩합니다.
포트가 열렸다는 사실과 모델 준비 완료는 다릅니다. `SERVER_STATUS.ready`를 확인하세요.
측정 종료 후에도 서버 프로세스와 모델은 유지합니다. 카메라는 실시간 세션 시작 시 열고 종료 시 닫습니다.

필요한 로컬 모델 파일은 다음과 같습니다.

```text
configs/checkpoints/yolov8n.pt
configs/checkpoints/RTMPose_config.py
configs/checkpoints/RTMPose.pth
```

## 2. 연결과 세션 흐름

### 통신 형식

- 일반 TCP 소켓 메시지가 아닌 **WebSocket 텍스트 메시지**를 사용합니다.
- WebSocket 메시지 하나에 JSON 객체 하나를 넣습니다. 별도 줄바꿈 구분자는 필요 없습니다.
- 명령 종류는 `type`으로 구분합니다. 명령 이름과 mode는 아래의 대소문자를 지켜 주세요.
- 클라이언트는 명령을 보내는 동시에 서버 이벤트를 계속 수신해야 합니다.
- 현재 클라이언트가 JPEG나 카메라 프레임을 업로드하는 명령은 없습니다.

### 실시간 MAIN/TEST

```text
WebSocket 연결
  → SERVER_STATUS (LOADING / READY)
  → CMD_SET_SESSION
  → SESSION_PENDING                     모델 준비 전 요청한 경우
  → SESSION_PREPARING                    카메라 연결 중
  → SESSION_LOG_STARTED
  → SESSION_PREPARING                    기본 5초 카운트다운
  → SESSION_MEASUREMENT_STARTED          준비 시간이 0이면 생략
  → POSE_UPDATE 반복
  → REP_COMPLETED                       좌우별 새 반복이 인정될 때
  → 클라이언트에서 점수 누적·종료 결정
  → CMD_STOP_SESSION
  → SESSION_FINISHED                     로그 저장 후 응답
```

`REP_COMPLETED`는 해당 프레임의 `POSE_UPDATE`보다 먼저 전송됩니다.
다른 명령 응답과 상태 메시지도 같은 연결에서 섞여 들어오므로 `type`으로 분기하세요.

### 실시간 CALIBRATION

시작은 같은 방식입니다. 초기 자세를 잡고 양측 각각 일관된 3회 동작을 수행하면
서버가 개인 기준값을 저장하고 `CALIBRATION_FINISHED`를 보냅니다.
완료 후 카메라와 측정 세션은 해제됩니다. WebSocket 연결 자체는 별도로 닫을 수 있습니다.

### 연결 점유

상태 조회용 연결은 여러 개 만들 수 있지만, 실제 측정은 한 연결만 소유할 수 있습니다.
시작·가동 범위 수정·종료는 원칙적으로 **세션을 시작한 같은 연결**에서 수행합니다.
다른 연결이 측정 중이면 `SERVER_BUSY`가 발생합니다.
현재 오프라인 분석 완료 후에는 연결 점유가 남을 수 있으므로, 같은 연결로 다음 영상을 보내거나
해당 연결을 닫은 뒤 다른 클라이언트에서 시작하세요. 시작 실패 후에도 연결을 닫아 점유를 정리하는 편이 확실합니다.

## 3. 클라이언트가 보내는 입력

### 3.1 CMD_GET_SERVER_STATUS — 모델 상태 조회

```json
{"type": "CMD_GET_SERVER_STATUS"}
```

응답은 `SERVER_STATUS`입니다. 세션 없이 조회할 수 있습니다.

### 3.2 CMD_SET_SESSION — 세션 시작

#### 실시간 캘리브레이션 예제

```json
{
  "type": "CMD_SET_SESSION",
  "patient_id": "0001",
  "patient_name": "테스트 환자",
  "exercise_name": "shoulder_press",
  "mode": "CALIBRATION",
  "cal": true,
  "session": 1,
  "round": 0,
  "set": 0,
  "camera_index": 0,
  "output_mode": "LIGHTWEIGHT",
  "preparation_seconds": 5,
  "record_video": true,
  "patient_metadata": {"height_cm": 170, "weight_kg": 65}
}
```

#### 실시간 본 운동 예제

```json
{
  "type": "CMD_SET_SESSION",
  "patient_id": "0001",
  "exercise_name": "shoulder_press",
  "mode": "MAIN",
  "session": 1,
  "round": 1,
  "set": 1,
  "camera_index": 0,
  "output_mode": "LIGHTWEIGHT",
  "preparation_seconds": 5,
  "record_video": true
}
```

#### 필드 명세

`type`을 제외한 다수 필드에 기본값이 있으나, 실제 연동에서는 환자·운동·모드를 명시하는 것을 권장합니다.

| 필드 | 형식 | 기본값 / 처리 | 설명 |
|---|---|---|---|
| type | string | 필수 | `CMD_SET_SESSION` |
| patient_id | string | `player_id`, 없으면 `0001` | 환자 번호. 숫자 문자열은 네 자리로 보정 |
| player_id | string | 호환 별칭 | `patient_id`가 있으면 그 값이 우선 |
| patient_name | string | `Unknown` | 환자 이름 |
| exercise_name | string | `biceps_curl` | 아래 운동 이름 사용 |
| mode | string | `CALIBRATION` | `CALIBRATION`, `MAIN`, `TEST` |
| cal | boolean | 생략 가능 | 있으면 `mode == CALIBRATION`과 반드시 일치 |
| session | integer | 1 | 기존 숫자 Session 메타데이터. 공통 플레이 ID와 별개 |
| data_version | string | `v1.0.0(0)` | Unity BuildInfo의 최종 버전 폴더명 |
| registration | integer/string | 1 | Unity 등록 차수. 1 또는 `01`, 1~99 |
| play_session_id | string | 단독 테스트용 고유 ID | Unity 전체 플레이 ID |
| exercise_record_id | string | 단독 테스트용 고유 ID | 라운드·운동 슬롯의 기록 ID. 여러 세트에서 유지 |
| slot | integer | 1 | 처방 목록의 운동 순서. 0 이상 |
| round | integer | 1 | 0 이상 |
| set | integer | 1 | 0 이상 |
| camera_index | integer | 0 | **서버 PC**에 등록된 카메라 장치 번호 |
| output_mode | string | `LIGHTWEIGHT` | LIGHTWEIGHT만 지원. VISUAL은 UNSUPPORTED_OUTPUT_MODE 오류 |
| preparation_seconds | number | 5.0 | 실시간 측정 전 준비 시간. 0으로 비활성화. 유한한 0 이상 값 |
| video_path | string | 없음 | 있으면 카메라 대신 오프라인 영상 분석 |
| input_source | string | 호환 별칭 | `video_path`가 비어 있으면 사용. 현재는 영상 파일 경로로 해석 |
| batch_size | integer | 16 | 오프라인 추론 배치 크기. 양수 |
| record_video | boolean | 항상 true | 호환 필드. false를 보내도 서버가 true로 처리하여 MP4 저장 |
| log_root | string | 환경 변수 또는 프로젝트 `PatientData` | 서버 PC 기준 PatientData 최상위 폴더 |
| patient_metadata | object | 없음 | `height_cm`, `weight_kg`. 제공 시 유한한 양수 |
| target_score | number | 엔진 기본 참조값 10 | 선택적 호환·메타데이터 필드. **서버 자동 종료 조건이 아님** |
| target_reps | number | 호환 필드 | `target_score`가 없으면 엔진 생성 시 참조. MAIN/TEST 자동 종료 조건이 아님 |

환자·운동 경로 구성 요소에는 영문자·숫자·`_`·`-`만 사용합니다.
Streamlit은 목표 점수와 배점을 클라이언트 안에서 관리하며 시작 요청에서 제외합니다.
다른 클라이언트도 게임 목표를 로컬에 보관하고 종료 시 `client_result`로 전달할 수 있습니다.

#### 사용 가능한 운동

설정 파일: `configs/exercise_kpt_config_coco.json`

| exercise_name | 운동 | 현재 측정 방식 |
|---|---|---|
| biceps_curl | 이두 굴곡 | COORDINATE: 손목의 신체 기준 y좌표 |
| shoulder_press | 숄더프레스 | ANGLE: 어깨–팔꿈치–손목의 팔꿈치 각도 |
| clamshell | 클램쉘 | ANGLE: 반대쪽 골반–측정 쪽 골반–무릎 각도 |
| slr | 하지 직거상 | COORDINATE: 발목의 신체 기준 y좌표 |
| hip_knee_flexion | 고관절·슬관절 굴곡 | ANGLE: 어깨–골반–무릎의 고관절 각도 |
| knee_extension | 슬관절 신전 | LENGTH: 정규화된 무릎–발목의 2D 거리 |

`test_video`의 `leg_raise`, `leg_extension` 폴더명은 서버 운동 이름이 아닙니다.
영상 테스트 클라이언트에서 각각 `hip_knee_flexion`, `knee_extension`으로 매핑합니다.
MAIN/TEST는 개인 캘리브레이션값이 있으면 기본 ROM보다 우선 적용합니다.
측정 방식이 바뀐 개인값은 재캘리브레이션을 요구할 수 있습니다.

### 3.3 CMD_GET_COUNT_THRESHOLD — 현재 가동 범위 조회

```json
{
  "type": "CMD_GET_COUNT_THRESHOLD",
  "exercise_name": "shoulder_press",
  "request_id": "rom-query-001"
}
```

응답: `COUNT_THRESHOLD_STATUS`. 현재 운동 이름이 일치해야 합니다.
모델 준비를 기다리는 세션 시작 요청 뒤에 보낸 조회는 대기열에 들어갈 수 있습니다.
현재 대기 명령 한도는 16개이므로 로딩 중 명령을 반복 전송하지 마세요.

### 3.4 CMD_UPDATE_COUNT_THRESHOLD — 측정 중 가동 범위 수정

```json
{
  "type": "CMD_UPDATE_COUNT_THRESHOLD",
  "exercise_name": "shoulder_press",
  "request_id": "rom-edit-001",
  "thresholds": {
    "left": {"start_val": 90, "target_val": 140},
    "right": {"start_val": 85, "target_val": 140}
  }
}
```

한쪽 또는 한 값만 보낼 수도 있습니다.

```json
{
  "type": "CMD_UPDATE_COUNT_THRESHOLD",
  "exercise_name": "shoulder_press",
  "request_id": "rom-edit-002",
  "thresholds": {"left": {"target_val": 130}}
}
```

- MAIN/TEST에서만 수정 가능합니다. CALIBRATION에서는 오류입니다.
- 생략한 쪽과 값은 유지합니다. 검증은 양측을 함께 수행하며 일부만 적용하지 않습니다.
- 값은 boolean이 아닌 유한한 숫자여야 합니다.
- 증가 방향은 목표값이 시작값보다 커야 하고, 감소 방향은 반대여야 합니다. 각 측에 적용된 방향을 기준으로 검증합니다.
- ANGLE은 0~180도 범위를 사용합니다. LENGTH/RATIO는 음수 범위를 사용할 수 없습니다.
- 실제 변경이 있으면 `threshold_version`을 올립니다. 같은 값은 변경으로 취급하지 않습니다.
- 기존 횟수와 누적 점수는 보존합니다. 수정된 측은 이전 정점을 버리고 시작 영역으로 복귀해야 다시 셉니다.
- 적용 범위는 **현재 세션**입니다. 개인 캘리브레이션 프로필을 영구 수정하는 명령이 아닙니다.
- 오프라인 영상 처리 중에는 이 명령을 사용할 수 없습니다.

### 3.5 CMD_STOP_SESSION — 측정 종료 및 로그 확정

일반 중지:

```json
{
  "type": "CMD_STOP_SESSION",
  "request_id": "stop-001",
  "session_id": "서버가-발급한-session_id",
  "reason": "USER_STOP"
}
```

클라이언트에서 목표를 달성한 경우:

```json
{
  "type": "CMD_STOP_SESSION",
  "request_id": "finish-001",
  "session_id": "서버가-발급한-session_id",
  "reason": "TARGET_REACHED",
  "client_result": {
    "target_score": 10,
    "scores": {"left": 10.5, "right": 7.25},
    "repetitions": {"left": 14, "right": 11},
    "goal_reached": true
  }
}
```

| 필드 | 설명 |
|---|---|
| request_id | 선택적 요청 식별자. 종료 응답에 되돌려줌 |
| session_id | 생략 가능하지만 전달 권장. 실제 활성 세션과 다르면 거부 |
| reason | `TARGET_REACHED`이면 COMPLETED, 다른 값이나 생략이면 STOPPED 처리 |
| client_result | 선택적 게임 결과 메타데이터. 서버가 게임 규칙을 재판정하지 않음 |

`client_result.target_score`와 `scores.left/right`는 유한한 0 이상 숫자일 때 저장합니다.
형식이 맞지 않으면 이 메타데이터만 무시하고 종료 처리는 계속합니다.
로그의 서버 계산 점수와 클라이언트가 결정한 게임 점수는 구분해서 사용하세요.

종료 요청 후 **SESSION_FINISHED를 받은 뒤** 저장 완료로 처리하세요.
연결만 끊어도 서버가 정리를 시도하지만, 저장 확인 응답을 받을 수 없습니다.
실시간 종료 명령은 같은 연결에서 재시도 시 직전 종료 응답을 반환하는 처리가 있습니다.
모델 로딩 중 대기 세션을 취소하면 `summary: null`인 종료 응답이 올 수 있습니다.
CALIBRATION을 중지할 때 `TARGET_REACHED`는 사용하지 마세요. 정상 완료는 서버가 결정합니다.

### 3.6 CMD_STOP_CALIBRATION — 기존 호환 명령

```json
{"type": "CMD_STOP_CALIBRATION"}
```

현재 코드는 활성 CALIBRATION의 `calib_step`을 FINISHED로 설정하는 기존 분기를 남겨 두고 있습니다.
다음 캘리브레이션 갱신에서 상태가 다시 계산되므로 정상적인 중지·저장을 보장하는 API가 아닙니다.
측정을 취소하려면 **CMD_STOP_SESSION**을 사용하세요. 완료 판정을 강제로 우회하는 용도로 사용하지 마세요.

## 4. 서버가 보내는 출력

### 4.1 SERVER_STATUS — 모델 준비 상태

```json
{
  "type": "SERVER_STATUS",
  "model_state": "LOADING",
  "phase": "LOADING_POSE",
  "ready": false,
  "message": "관절 추출 모델을 불러오는 중입니다.",
  "model_error": null,
  "elapsed_seconds": 4.2,
  "phase_seconds": {"IMPORTING": 2.8, "LOADING_YOLO": 0.3, "LOADING_POSE": 1.1}
}
```

`model_state`: NOT_STARTED / LOADING / READY / FAILED.
`phase`: NOT_STARTED / IMPORTING / DEVICE_SETUP / LOADING_YOLO / LOADING_POSE / READY / FAILED.
`elapsed_seconds`와 `phase_seconds`는 모델 준비에 사용한 실제 경과 초입니다.
FAILED이면 `model_error`와 서버 터미널의 traceback을 확인하세요.

### 4.2 SESSION_PENDING / SESSION_PREPARING / SESSION_MEASUREMENT_STARTED

```json
{"type": "SESSION_PENDING", "message": "모델 준비가 끝나면 요청한 세션을 시작합니다."}
```

```json
{"type": "SESSION_PREPARING", "message": "카메라 연결 중입니다. 휴대폰 웹캠 앱의 연결 상태를 확인해 주세요."}
```

```json
{"type": "SESSION_PREPARING", "remaining_seconds": 3, "message": "기본 자세를 취해 주세요. 3초 후 측정을 시작합니다."}
```

```json
{"type": "SESSION_MEASUREMENT_STARTED", "message": "준비 시간이 끝나 측정을 시작합니다."}
```

`remaining_seconds`는 카메라 연결 중 메시지에는 없습니다. 필드가 항상 존재한다고 가정하지 마세요.
준비 중에는 추론·카운트·프레임 로그·녹화를 수행하지 않습니다.
준비 시간을 0으로 설정하면 `SESSION_MEASUREMENT_STARTED`가 오지 않을 수 있으므로,
이 이벤트만을 첫 측정 수신 조건으로 삼지 마세요.

### 4.3 SESSION_LOG_STARTED — 세션 ID와 실제 저장 경로

```json
{
  "type": "SESSION_LOG_STARTED",
  "session_id": "f9a764e76bc4451d82c22d9ad0f493f1",
  "started_at": "2026-10-20T10:00:00+09:00",
  "patient_id": "0001",
  "motion_name": "shoulder_press",
  "mode": "MAIN",
  "session": 1,
  "round": 1,
  "set": 1,
  "directory": "C:\\project\\PatientData\\v1.0.0(0)\\0001\\01\\Sensors\\2026-10-20\\PLAY_ID\\Camera",
  "frame_csv": "C:\\project\\PatientData\\v1.0.0(0)\\0001\\01\\Sensors\\2026-10-20\\PLAY_ID\\Camera\\R01_S01_Set01_Play_Camera_f9a764e76bc4451d82c22d9ad0f493f1.csv",
  "skeleton_npz": "C:\\project\\PatientData\\v1.0.0(0)\\0001\\01\\Sensors\\2026-10-20\\PLAY_ID\\Camera\\R01_S01_Set01_Play_Camera_f9a764e76bc4451d82c22d9ad0f493f1.npz",
  "video_mp4": "C:\\project\\PatientData\\v1.0.0(0)\\0001\\01\\Sensors\\2026-10-20\\PLAY_ID\\Camera\\R01_S01_Set01_Play_Camera_f9a764e76bc4451d82c22d9ad0f493f1.mp4",
  "record_video": true,
  "video_fps": 30.0,
  "target_score": null,
  "scoring_mode": "POINTS",
  "data_version": "v1.0.0(0)",
  "registration": "01",
  "play_session_id": "PLAY_ID",
  "exercise_record_id": "EXERCISE_ID",
  "sensor_file_id": "f9a764e76bc4451d82c22d9ad0f493f1",
  "slot": 1
}
```

실제 응답에는 `log_root`, `reps_csv`, `manifest_json`, `gamedata_csv`도 포함됩니다.
`session`은 사용자가 보낸 세션 번호이고, `session_id`는 서버가 발급한 고유 문자열입니다.
재시도·이벤트 중복 방지에는 `session_id`를 사용하세요.
이 응답은 파일 생성 시작 알림이며 최종 저장 완료 알림은 아닙니다.
여기의 `scoring_mode: POINTS`는 로그 메타데이터입니다. 게임 종료 책임은 POSE_UPDATE의 `completion_owner`를 확인하세요.

### 4.4 POSE_UPDATE — 실시간 측정 프레임

LIGHTWEIGHT의 MAIN 예제:

```json
{
  "type": "POSE_UPDATE",
  "mode": "MAIN",
  "exercise_name": "shoulder_press",
  "metric_type": "ANGLE",
  "motion_direction": "INCREASING",
  "threshold_version": 0,
  "thresholds": {
    "left": {"start_val": 90, "target_val": 150},
    "right": {"start_val": 90, "target_val": 150}
  },
  "left": {
    "val": 132.0,
    "state": "RETURNING",
    "rep_count": 2,
    "progress_ratio": 0.7,
    "quality": "PERFECT",
    "is_count_updated": true,
    "score": 3,
    "score_delta": 1.0,
    "total_score": 1.5,
    "target_score": 10.0
  },
  "right": {
    "val": 118.0,
    "state": "PUSHING",
    "rep_count": 1,
    "progress_ratio": 0.47,
    "quality": "GOOD",
    "is_count_updated": false,
    "score": 0,
    "score_delta": 0.0,
    "total_score": 0.5,
    "target_score": 10.0
  },
  "fps": 18.2,
  "is_occluded": false,
  "calib_step": "FULL_BODY_CHECK",
  "calib_rep_count": 0,
  "calib_reps_left": 0,
  "calib_reps_right": 0,
  "target_calib_reps": 3,
  "target_score": 10.0,
  "scoring_mode": "CLIENT",
  "completion_owner": "CLIENT"
}
```

`calib_*`는 MAIN/TEST에서도 전송될 수 있지만 해당 모드의 진행률로 해석하지 않습니다.
`fps`는 서버 처리 루프의 관측 속도이며 카메라 원본 FPS나 고정 전송 주기를 보장하지 않습니다.
MAIN/TEST의 상태는 보통 READY → PUSHING → RETURNING → READY입니다.
측정값이 없으면 `val`이 null이거나 직전 캐시가 유지될 수 있습니다. 유효 여부는 `is_occluded`와 함께 확인하세요.
`quality`는 마지막 판정을 유지할 수 있으므로 문자열이 PERFECT인 모든 프레임에 점수를 더하면 안 됩니다.

| 좌우 필드 | 의미 |
|---|---|
| val | 해당 운동의 현재 측정값 |
| rep_count | 서버가 인정한 실제 반복 횟수 |
| progress_ratio | 기준 시작값에서 이동한 비율. 0~1로 제한된 ROM 게이지이며 점수가 아님 |
| state | 반복 판정기 상태 |
| quality | 마지막 품질 문자열. READY/CALIBRATING 등도 가능 |
| is_count_updated | 현재 프레임에서 새 반복이 인정됐는지 |
| score | 새 MAIN/TEST 반복의 품질 코드. 0=None, 1=Bad, 2=Good, 3=Perfect |
| score_delta | 서버 기본 배점으로 이번 반복에 추가된 점수 |
| total_score | 서버 기본 배점 누적 참고값. 클라이언트 게임 규칙이 다르면 결과도 다를 수 있음 |
| target_score | 엔진 참조값. 현재 서버의 자동 종료 조건이 아님 |

#### 제거된 실시간 이미지 필드

VISUAL은 더 이상 지원하지 않습니다. POSE_UPDATE에 keypoints, frame_b64, yolo_detected를 넣지 않습니다.
스켈레톤과 영상 확인은 저장된 NPZ와 일괄 오버레이 기능을 사용하세요. 아래 COCO 번호는 NPZ 관절 배열 해석에 사용합니다.

COCO 관절 번호:

| 관절 | 왼쪽 | 오른쪽 |
|---|---:|---:|
| 눈 | 1 | 2 |
| 귀 | 3 | 4 |
| 어깨 | 5 | 6 |
| 팔꿈치 | 7 | 8 |
| 손목 | 9 | 10 |
| 골반 | 11 | 12 |
| 무릎 | 13 | 14 |
| 발목 | 15 | 16 |

0번은 코입니다. 좌우는 화면의 좌우가 아니라 대상자의 관절 좌우입니다.

#### CALIBRATION의 추가 진단

CALIBRATION의 `left/right`는 일반 SideFSM과 다른 캘리브레이션 판정기의 결과입니다.
`score`, `score_delta`, `total_score`는 0이고 `quality`는 CALIBRATING입니다.
각 측에 다음 `calibration` 객체가 추가됩니다.

```json
{
  "state": "RETURNING",
  "baseline": 90.0,
  "baseline_samples": 6,
  "min_baseline_samples": 3,
  "rest_seconds": 0.5,
  "min_movement": 8.0,
  "return_tolerance": 7.5,
  "completed_cycles": 2,
  "consistent": false,
  "direction": "INCREASING"
}
```

| 상태 | 의미 / 대응 |
|---|---|
| BASELINE | 초기 자세 측정 중. 시작 자세에서 잠시 정지 |
| READY | 다음 동작 대기 |
| RISING | 최소 이동량 이상 움직인 구간의 유지 확인 |
| RETURNING | 반전·시작 영역 복귀·복귀 유지 확인 |
| MISSING | 해당 측의 측정값 누락 |
| NEED_CONSISTENT_CYCLES | 최근 3회 크기가 일정하지 않음. 추가 반복 필요 |
| COMPLETE | 해당 측은 일관된 3회 확보 |

상위 `calib_step`은 FULL_BODY_CHECK, COUNTDOWN, COLLECTING, PAUSED,
CHECK_CONSISTENCY, COOLDOWN, FINISHED 중 하나입니다.
`calib_rep_count`는 양측 중 작은 완료 횟수입니다. 일관성 확인을 위해 한쪽 횟수가 3보다 커질 수도 있습니다.

### 4.5 REP_COMPLETED — 새 반복 이벤트

MAIN/TEST에서 새 반복이 인정될 때 좌우별로 전송합니다.

```json
{
  "type": "REP_COMPLETED",
  "session_id": "f9a764e76bc4451d82c22d9ad0f493f1",
  "exercise_name": "shoulder_press",
  "mode": "MAIN",
  "side": "left",
  "rep_id": 2,
  "quality": "PERFECT",
  "threshold_version": 0,
  "timestamp": 4.125,
  "val": 132.0
}
```

게임 점수는 이 이벤트로 누적하는 것을 권장합니다.
중복 방지 키는 **(session_id, side, rep_id)**입니다.
`timestamp`는 실시간에서는 준비 종료 이후 세션 경과 초, 오프라인에서는 영상상의 초입니다.
`val`은 판정이 발생한 시점의 값으로, 최대 정점값 자체라고 보장하지 않습니다.
이 이벤트에는 점수 배점·both 결과가 없습니다. 클라이언트가 품질과 자신의 규칙으로 계산합니다.
CALIBRATION에서는 이 이벤트 대신 POSE_UPDATE의 횟수와 CALIBRATION_FINISHED를 사용하세요.

### 4.6 COUNT_THRESHOLD_STATUS / COUNT_THRESHOLD_UPDATED

```json
{
  "type": "COUNT_THRESHOLD_UPDATED",
  "request_id": "rom-edit-001",
  "exercise_name": "shoulder_press",
  "metric_type": "ANGLE",
  "motion_direction": "INCREASING",
  "threshold_version": 1,
  "thresholds": {
    "left": {"start_val": 90, "target_val": 140},
    "right": {"start_val": 85, "target_val": 140}
  },
  "changed_sides": ["left", "right"],
  "states": {"left": "RETURNING", "right": "RETURNING"},
  "scope": "SESSION",
  "target_score": 10.0,
  "scoring_mode": "CLIENT",
  "completion_owner": "CLIENT"
}
```

조회는 `COUNT_THRESHOLD_STATUS`이며 `changed_sides`는 빈 배열입니다.
수정 요청이 기존 값과 같아도 UPDATED 응답이 올 수 있으므로 실제 변경 여부는 배열과 버전으로 확인하세요.
AUTO 방향 운동의 상위 `motion_direction`과 각 측 개인값의 방향은 다를 수 있습니다.

### 4.7 CALIBRATION_FINISHED — 실시간 캘리브레이션 완료

```json
{
  "type": "CALIBRATION_FINISHED",
  "player_id": "0001",
  "thresholds": {
    "left": {"start_val": 90, "target_val": 141, "metric_type": "ANGLE", "motion_direction": "INCREASING"},
    "right": {"start_val": 88, "target_val": 139, "metric_type": "ANGLE", "motion_direction": "INCREASING"}
  },
  "summary": {"status": "COMPLETED", "saved_frames": 180, "saved_reps_count": 6}
}
```

실제 summary는 아래 공통 요약 필드를 포함합니다. 이 이벤트가 온 뒤에 개인값 저장 완료로 처리합니다.
입력은 `patient_id`를 사용해도 이 응답 필드 이름은 현재 `player_id`입니다.

### 4.8 SESSION_FINISHED — 클라이언트 종료 명령 완료

```json
{
  "type": "SESSION_FINISHED",
  "session_id": "f9a764e76bc4451d82c22d9ad0f493f1",
  "request_id": "finish-001",
  "reason": "TARGET_REACHED",
  "status": "COMPLETED",
  "summary": {
    "status": "COMPLETED",
    "saved_frames": 240,
    "saved_reps_count": 25,
    "left_reps": 14,
    "right_reps": 11,
    "left_total_score": 10.5,
    "right_total_score": 7.25,
    "client_result": {
      "target_score": 10,
      "scores": {"left": 10.5, "right": 7.25}
    }
  }
}
```

`summary.left/right_total_score`는 서버 기본 배점으로 집계합니다.
클라이언트의 최종 게임 결과는 제공했다면 `summary.client_result`에서 확인합니다.
목표 도달 지연 시간 동안 추가 반복이 들어올 수 있으므로 종료 직전 결과를 전송하세요.

### 4.9 VIDEO_STATUS / VIDEO_PROGRESS / VIDEO_ANALYSIS_COMPLETED

```json
{"type": "VIDEO_STATUS", "status": "PROCESSING", "message": "비디오를 고속 배치 분석 중입니다. 잠시만 기다려주세요..."}
```

```json
{"type": "VIDEO_PROGRESS", "phase": "EXTRACTING", "processed_frames": 160, "total_frames": 450}
```

진행 이벤트는 추론 단계에서 대략 1초 간격으로 전송하며, 짧은 영상에서는 생략될 수 있습니다.

```json
{
  "type": "VIDEO_ANALYSIS_COMPLETED",
  "session_id": "서버 세션 ID",
  "request_id": null,
  "reason": "END_OF_VIDEO",
  "mode": "CALIBRATION",
  "video_path": "C:\\videos\\shoulder_press.mp4",
  "total_frames_analyzed": 450,
  "source_duration_sec": 15.0,
  "calibration_completed": false,
  "calib_step": "COLLECTING",
  "thresholds": null,
  "summary": {"status": "INCOMPLETE", "saved_frames": 450, "saved_reps_count": 4}
}
```

영상 처리 완료와 캘리브레이션 성공은 다릅니다.
CALIBRATION이면 `calibration_completed == true`를 확인하세요. 미완료는 INCOMPLETE로 저장하고 개인 기준값을 저장하지 않습니다.
`total_frames_analyzed`는 추론을 수행한 전체 프레임 수입니다. 캘리브레이션이 먼저 끝나도 이 값이 크며,
실제 판정·로그 프레임 수는 `summary.saved_frames`를 확인해야 합니다.
`source_duration_sec`는 읽은 프레임 수/FPS이며, 중간 중지 시 전체 원본 길이보다 짧을 수 있습니다.
오프라인 중지를 요청했다면 완료 메시지 type이 `SESSION_FINISHED`로 바뀌고 request_id가 반환됩니다.

### 4.10 ERROR

```json
{
  "type": "ERROR",
  "code": "INVALID_THRESHOLD",
  "command": "CMD_UPDATE_COUNT_THRESHOLD",
  "request_id": "rom-edit-001",
  "message": "left: INCREASING requires target_val > start_val"
}
```

`command`, `request_id`, `fatal`, `code`는 오류 발생 위치에 따라 없을 수 있습니다.
특히 영상 파일을 열지 못한 일부 응답은 type/message만 포함합니다.
필드를 항상 존재한다고 가정하지 말고, type/message를 우선 처리하세요.

| code | 상황 |
|---|---|
| SERVER_BUSY | 다른 연결이 세션 점유 |
| MODEL_LOAD_FAILED | 모델 준비 실패. fatal=true일 수 있음 |
| SESSION_START_FAILED | 실시간 시작 실패. 카메라·설정·개인 프로필 확인 |
| VIDEO_ANALYSIS_FAILED | 오프라인 시작 또는 분석 중 오류 |
| SESSION_PROCESSING_FAILED | 메인 처리 루프 오류 |
| INVALID_BATCH_SIZE | 오프라인 batch_size가 0 이하 |
| UNSUPPORTED_OUTPUT_MODE | 제거된 VISUAL 등 LIGHTWEIGHT 이외의 출력 요청 |
| NO_ACTIVE_SESSION | 활성 세션이 없거나 해당 연결이 소유하지 않음 |
| SESSION_MISMATCH | 종료 명령의 session_id 불일치 |
| EXERCISE_MISMATCH | 가동 범위 명령의 운동 이름 불일치 |
| INVALID_MODE | 해당 모드에서 사용할 수 없는 명령 |
| INVALID_THRESHOLD | 가동 범위 형식·범위·방향 오류 |
| VIDEO_COMMAND_REJECTED | 오프라인 처리 중 허용하지 않는 명령 |

잘못된 가동 범위 요청 후에도 정상 세션은 유지됩니다.
모델 로딩 실패·처리 루프 오류는 세션을 계속 사용할 수 있다는 뜻이 아니므로 오류 상태를 별도로 표시하세요.

## 5. 측정값·카운트·점수 해석

### 측정값의 단위

| metric_type | 의미 |
|---|---|
| ANGLE | 세 관절의 2D 각도. 단위 도 |
| COORDINATE | 골반 중심·몸통 길이 기준 좌표 또는 설정된 두 관절 간 좌표 차이. 무차원 |
| LENGTH | 신체 기준 좌표에서 두 관절 사이 길이. 미터가 아님 |
| RATIO | 두 관절쌍 길이의 비율. 무차원 |

이미지 좌표의 종횡비를 보정한 뒤 골반 중심과 몸통 길이로 정규화합니다.
사람이 똑바로 보이는 세로 영상도 지원합니다. 사람 자체가 옆으로 돌아온 입력을 자동 회전하는 기능은 별개입니다.

### MAIN/TEST의 카운트 기준

- 기준 ROM의 15% 이상 움직이면 PUSHING 진입
- 정점에서 ROM의 40% 이상 이동한 이력이 있어야 유효
- 정점에서 ROM의 5% 이상 되돌아오면 새 반복 인정
- 이후 시작 영역 30% 이내로 복귀해야 다음 반복 준비

품질은 기준 ROM 대비 정점 이동량으로 PERFECT ≥70%, GOOD ≥50%, BAD ≥35%입니다.
카운트 자체는 40% 진입 조건도 있으므로 BAD의 실질 인정 구간은 이 조건과 함께 해석해야 합니다.
이 기준은 캘리브레이션의 동작 수집 조건과 다릅니다.

### 점수는 클라이언트에서 결정

현재 Streamlit 클라이언트 기본 배점은 PERFECT=1, GOOD=0.5, BAD=0.25입니다.
점수는 REP_COMPLETED마다 한 번씩 더합니다. POSE_UPDATE와 REP_COMPLETED 양쪽에서 동시에 더하지 마세요.

현재 `network/client_progress.py`는 **한쪽 누적 점수가 목표를 초과(`>`)**하면 종료를 예약합니다.
예를 들어 목표가 10이면 정확히 10.0에서는 종료하지 않고 10.25 이상이 된 뒤 종료 조건에 들어갑니다.
실시간은 기본 1.5초 후 CMD_STOP_SESSION을 보내며, 오프라인은 지연 없이 보냅니다.
Unity에서 `>=`, 양측 달성, 다른 배점 등을 원하면 클라이언트 규칙으로 구현할 수 있습니다.
서버가 이러한 목표 비교를 대신 수행하지 않습니다.

현재 프로토콜에 별도 `both` 결과는 없습니다. 양측 동시 판정이나 합산 점수는 클라이언트에서 정의하세요.

## 6. 캘리브레이션 진행 방법

1. CALIBRATION 세션을 시작하고 카메라 준비·기본 5초 준비 시간을 기다립니다.
2. 시작 자세를 유지해 초기 기준을 잡습니다. 기본 최소 정지 시간은 0.5초, 최소 샘플은 3개입니다.
3. 좌우 각각 동작한 뒤 시작 영역으로 복귀하고 잠시 멈춥니다.
4. 최근 3회의 동작 크기가 일관되면 1초 마무리 후 완료 이벤트를 받습니다.

짧은 검출 누락은 진행 중 동작의 유지 시간을 취소하지만 완료 횟수는 보존합니다.
기본 2초 이상 누락되면 초기 자세와 완료 횟수를 다시 측정합니다.
복귀 허용 범위는 동작 크기와 정지 노이즈를 고려하며, 복귀 상태에 머무는 것만으로 중복 카운트하지 않습니다.
개인 목표값은 초기 기준에서 최근 3회 이동량 중앙값의 85%를 적용해 생성합니다.
측정 방향과 metric_type도 개인 프로필에 저장합니다.

운동별 조정 항목과 진단 방법은 [calibration_tuning.md](calibration_tuning.md)를 참고하세요.

## 7. 오프라인 영상 분석

```json
{
  "type": "CMD_SET_SESSION",
  "patient_id": "video_test_001",
  "exercise_name": "shoulder_press",
  "mode": "CALIBRATION",
  "video_path": "C:\\videos\\shoulder_press.mp4",
  "batch_size": 8,
  "record_video": true,
  "log_root": "C:\\project\\data\\video_tests"
}
```

- 경로는 **서버 PC 기준**입니다. 클라이언트 PC의 파일을 자동 업로드하지 않습니다.
- video_path/input_source가 있으면 오프라인 경로로 분기합니다. output_mode를 OFFLINE_VIDEO로 지정하는 방식이 아닙니다.
- 실시간용 5초 준비 시간을 적용하지 않습니다.
- 먼저 배치 추론하고, 추출한 좌표를 영상 시간 순서로 판정합니다.
- FPS 메타데이터가 유효하지 않으면 30으로 대체합니다. 임의 PTS를 사용하는 가변 프레임률 재생과 동일하다고 보장하지 않습니다.
- POSE_UPDATE를 프레임마다 보내는 방식이 아닙니다. VIDEO_PROGRESS와 최종 결과를 받습니다.
- MAIN/TEST에서는 판정 단계에서 REP_COMPLETED도 보냅니다.
- 처리 중 명령은 활성 세션의 CMD_STOP_SESSION만 허용합니다. 배치 경계와 판정 루프에서 확인하므로 즉시 중지를 보장하지 않습니다.
- 중지한 시점에 이미 디코딩·추론한 프레임과 저장된 판정 프레임 수는 다를 수 있습니다.

원본 영상과 카운트 자막을 비교하려면 다음 클라이언트를 사용합니다.

```powershell
python tests/run_calibration_videos.py --exercise shoulder_press
```

이 도구는 실제 소켓 서버에 영상을 차례로 보내고 결과 CSV와 ASS 자막을 생성합니다.
기본 로컬 서버가 없으면 자동 시작합니다. 실행 전에 `--help`로 선택 옵션을 확인하세요.

## 8. 로그와 파일

기본 위치는 프로젝트 루트의 `PatientData`이며 exe에서는 exe 옆 폴더입니다.
우선순위는 시작 JSON의 log_root → ASAN_LOG_ROOT 환경 변수 → 기본 위치 순서입니다.

```text
PatientData/v1.0.0(0)/0001/01/
├─ Sensors/2026-10-22/PLAY_ID/Camera/
│  ├─ R01_S02_Set01_Play_Camera_FILE_ID.csv
│  ├─ R01_S02_Set01_Play_Camera_FILE_ID.npz
│  ├─ R01_S02_Set01_Play_Camera_FILE_ID.mp4
│  ├─ R01_S02_Set01_Play_Camera_FILE_ID_reps.csv
│  └─ R01_S02_Set01_Play_Camera_FILE_ID.json
└─ Management/Camera/
   ├─ profile.json
   ├─ patient_metadata.csv
   ├─ Records/FILE_ID.json
   └─ Daily/2026-10-22.csv
```

각 구간의 고유 파일 ID로 반복 실행을 구분하며 기존 파일을 덮어쓰지 않습니다. Unity의 PlayData·Records·등록 예약·IMU 복구 파일은 변경하지 않습니다.
서버는 유효한 입력 프레임을 항상 MP4로 저장합니다. 준비 중 중지해 프레임이 없으면 실제 영상 파일은 생성되지 않을 수 있습니다.
CSV는 프레임마다 기록하고, NPZ·반복 로그·요약은 종료 시 확정합니다.

### 공통 summary 필드

| 필드 | 의미 |
|---|---|
| session_id, started_at | 고유 세션 ID, 한국 시간 기준 시작 시각 |
| patient_id, motion_name, mode | 환자·운동·실행 모드 |
| session, round, set | 사용자가 지정한 번호 |
| directory 및 *_csv / *_npz / *_mp4 / *_json 경로 | 실제 서버 파일 경로 |
| status | COMPLETED / STOPPED / INCOMPLETE / FAILED 등 |
| saved_frames | 실제 프레임 분석 로그 수 |
| saved_reps_count | 좌우 반복 기록 총수 |
| left_reps, right_reps | 좌우 반복 로그 수 |
| left_total_score, right_total_score | 서버 기본 배점의 누적값 |
| duration_sec | 실시간은 준비 시간을 제외한 경과 시간, 오프라인은 마지막 저장 프레임 시각 기준 |
| video_frames, video_fps, video_duration_sec | 녹화 영상 프레임 수·FPS·재생 길이 |
| video_time_origin_sec | 첫 녹화 프레임의 세션 경과 시간 |
| client_result | 종료 명령에서 제공한 클라이언트 결과. 선택적 |

MP4는 화면용 축소·스켈레톤 오버레이 전 입력입니다. 실시간에서는 캡처 시각을 기준으로 프레임을 유지해 재생 속도를 보정합니다.
NPZ의 keypoints는 필터·신체 정규화 전 이미지 비율 좌표이며 image_aspect_ratio도 저장합니다.
칼로리·최대 속도는 현재 계산하지 않아 gamedata.csv에 빈 값으로 남습니다.
MAIN/TEST 반복별 duration_sec도 현재 0인 항목이 있으므로 실제 측정 시간으로 사용하지 마세요.
자세한 열 구성은 [log_storage.md](log_storage.md)를 참고하세요.

## 9. Python 연결 예제

다음은 MAIN 세션을 시작하고 반복 이벤트를 점수로 누적한 뒤 종료하는 최소 예제입니다.
서버를 먼저 실행한 상태에서 사용합니다. 예제 자체를 실행하면 실제 카메라가 열립니다.

```python
import asyncio
import json
import time
import uuid
import websockets


async def main():
    points = {"PERFECT": 1.0, "GOOD": 0.5, "BAD": 0.25}
    target = 10.0
    scores = {"left": 0.0, "right": 0.0}
    seen = set()
    started = stopping = False
    session_id = None
    goal_at = None

    async with websockets.connect("ws://127.0.0.1:8080") as ws:
        while True:
            # 수신 이벤트가 없어도 종료 타이머는 진행한다.
            if goal_at is not None and not stopping and time.monotonic() - goal_at >= 1.5:
                await ws.send(json.dumps({
                    "type": "CMD_STOP_SESSION",
                    "request_id": uuid.uuid4().hex,
                    "session_id": session_id,
                    "reason": "TARGET_REACHED",
                    "client_result": {"target_score": target, "scores": scores},
                }))
                stopping = True
            try:
                packet = json.loads(await asyncio.wait_for(ws.recv(), timeout=0.1))
            except asyncio.TimeoutError:
                continue
            kind = packet.get("type")
            if kind == "SERVER_STATUS":
                if packet.get("model_state") == "FAILED":
                    raise RuntimeError(packet.get("model_error"))
                if packet.get("ready") and not started:
                    await ws.send(json.dumps({
                        "type": "CMD_SET_SESSION", "patient_id": "0001",
                        "exercise_name": "shoulder_press", "mode": "MAIN",
                        "camera_index": 0, "output_mode": "LIGHTWEIGHT",
                        "preparation_seconds": 5, "record_video": True,
                    }))
                    started = True
            elif kind == "SESSION_LOG_STARTED":
                session_id = packet["session_id"]
            elif kind == "REP_COMPLETED":
                key = (packet["session_id"], packet["side"], packet["rep_id"])
                if packet["session_id"] != session_id or key in seen:
                    continue
                seen.add(key)
                side = packet["side"]
                scores[side] += points[packet["quality"]]
                print(side, packet["quality"], scores)
                if goal_at is None and scores[side] > target:
                    goal_at = time.monotonic()
            elif kind == "SESSION_FINISHED":
                print("저장 완료:", packet.get("summary"))
                break
            elif kind == "ERROR":
                raise RuntimeError(packet.get("message", "Server error"))
            elif kind in {"SESSION_PENDING", "SESSION_PREPARING"}:
                print(packet.get("message"))


asyncio.run(main())
```

실제 제품에서는 연결·모델 준비·종료 응답 제한 시간, 사용자 중지, 재접속 UI도 구현하세요.
한 WebSocket에서 recv를 여러 태스크가 동시에 호출하지 말고, 수신 루프 하나에서 type별로 배분하세요.

## 10. 오류 및 구현상 주의점

### 화면이 끊겼는데 카메라가 계속 켜진 경우

WebSocket의 상대는 브라우저가 아니라 Streamlit Python 백그라운드 클라이언트입니다.
브라우저만 닫혀도 이 연결이 자동으로 닫힌다고 가정할 수 없습니다.
현재 Streamlit은 UI 생존 신호가 10초간 없으면 UI_DISCONNECTED 이유로 종료 요청을 보냅니다.
이는 Streamlit 클라이언트 기능이며 서버 공통 JSON heartbeat 명령은 아닙니다.
Unity도 종료·앱 비활성·연결 해제를 자신의 수명주기에 맞춰 처리해야 합니다.

### 카메라를 못 여는 경우

camera_index는 Windows에 등록된 서버 PC의 장치 번호입니다.
휴대폰 웹캠도 Windows 카메라 장치로 연결되어 있어야 합니다. 현재 명령에 IP 카메라 URL 전용 필드는 없습니다.
서버는 DirectShow로 시도하고 실패하면 Media Foundation으로 다시 시도합니다.
두 방식 모두 유효한 BGR 프레임을 못 읽으면 시작 오류를 보냅니다.
연결 후 프레임이 5초 이상 끊기거나 읽기 오류가 나면 처리 오류를 알립니다.

### 프로토콜의 현재 제한

- 일반 메시지 전체를 검증하는 JSON Schema 계층은 없습니다. 잘못된 JSON이나 예상하지 않은 값은 처리 오류·연결 종료로 이어질 수 있습니다.
- 알 수 없는 일반 명령에 대해 항상 오류 응답을 보내지는 않습니다. 이 문서의 명령만 사용하세요.
- 모든 요청에 request_id가 반영되는 것은 아닙니다. 가동 범위 조회/수정과 종료 명령을 중심으로 지원합니다.
- POSE_UPDATE 자체에는 session_id가 없습니다. SESSION_LOG_STARTED로 활성 세션을 보관하고 REP_COMPLETED의 session_id를 검증하세요.
- 시작 요청은 LIGHTWEIGHT만 지원하며, VISUAL 등 다른 값은 UNSUPPORTED_OUTPUT_MODE 오류를 반환합니다.
- 실시간 세션 시작을 같은 연결에서 다시 요청하면 이전 세션을 STOPPED로 저장하고 새 세션을 시작합니다.
- 모델 준비 중 시작 요청을 여러 번 보내면 대기 요청이 최신 값으로 바뀔 수 있습니다.
- 오프라인 분석 중 새 세션 시작이나 ROM 수정은 허용하지 않습니다. 먼저 결과나 종료 응답을 기다리세요.
- 종료·실패 상태에서 반드시 요약 파일 저장까지 성공했다고 가정하지 말고, 종료 응답과 실제 경로를 확인하세요.

## 관련 파일

- `network/socket_server.py`: 접속·카메라·영상 분석·이벤트 전송
- `network/session_commands.py`: 가동 범위 명령 처리
- `core/session_controller.py`: 캘리브레이션과 측정 진행
- `core/motion_engine.py`: 측정값과 반복·품질 판정
- `network/client_progress.py`: Streamlit의 클라이언트 점수·종료 규칙
- `network/streamlit_client.py`: Streamlit 백그라운드 WebSocket 연결
- [calibration_tuning.md](calibration_tuning.md): 캘리브레이션 설정과 진단
- [count_thresholds.md](count_thresholds.md): 가동 범위 수정 예제
- [log_storage.md](log_storage.md): 로그 저장 구조
