# 소켓 서버 간단 사용법

실행 파일 배포와 화면 재시작 후 `SERVER_BUSY` 복구는 [exe 실행·연결 복구 안내](exe_and_recovery.md)를 참고하세요.

서버 실행: `python network/socket_server.py`  
연결 주소: `ws://127.0.0.1:8080`

같은 WebSocket 연결에서 시작 → 이벤트 수신 → 중지를 진행합니다. 모델은 서버에 유지되고, 측정을 시작할 때 카메라가 열립니다. 기본 준비 시간은 5초이며 CSV·NPZ·MP4는 기본적으로 프로젝트 `PatientData`에 저장됩니다. Unity와 공통 ID를 연결하는 입력은 [PatientData 카메라 로그 규격](patient_data_camera.md)을 참고하세요.

## 입력: 시작과 중지

개인 가동 범위 측정을 시작합니다.

```json
{"type":"CMD_SET_SESSION","patient_id":"0001","exercise_name":"knee_extension","mode":"CALIBRATION"}
```

캘리브레이션이 완료되면 같은 환자와 운동으로 게임을 시작합니다.

```json
{"type":"CMD_SET_SESSION","patient_id":"0001","exercise_name":"knee_extension","mode":"MAIN"}
```

게임 종료 또는 사용자 중단 시 다음을 보냅니다.

```json
{"type":"CMD_STOP_SESSION"}
```

`camera_index`는 기본 0입니다. 다른 카메라라면 시작 JSON에 추가합니다. `session`, `round`, `set`은 로그 구분이 필요할 때 추가합니다. `output_mode`, `record_video`, `cal`은 기본 연동에서 보낼 필요가 없습니다. 동영상 파일 테스트는 시작 JSON에 서버 PC의 `video_path`를 추가합니다.

## 출력: 클라이언트에서 처리할 이벤트

서버 메시지는 `type`으로 구분합니다. 아래 목록은 필수 처리 항목이며 실제 메시지에는 추가 필드가 있습니다.

| type | 처리 |
| --- | --- |
| `POSE_UPDATE` | 실시간 상태 표시. `left`, `right`에서 `val`, `rep_count`, `progress_ratio` 등을 선택해서 사용 |
| `REP_COMPLETED` | MAIN/TEST에서 한쪽 동작이 인정될 때 점수 추가 |
| `CALIBRATION_FINISHED` | 개인 기준값 저장 완료. 다음 MAIN 시작 가능 |
| `SESSION_FINISHED` | 세션 종료 확인 |
| `VIDEO_ANALYSIS_COMPLETED` | 파일 분석 종료. 캘리브레이션이라면 `calibration_completed`도 확인 |
| `ERROR` | 오류 안내 |

`REP_COMPLETED`에서 게임에 필요한 부분만 추린 예시입니다.

```json
{"type":"REP_COMPLETED","session_id":"...","side":"left","rep_id":1,"quality":"PERFECT"}
```

`quality`는 `BAD`, `GOOD`, `PERFECT`입니다. 기본 게임 규칙은 각각 0.25, 0.5, 1점이며 클라이언트가 누적하고 종료를 결정합니다. `(session_id, side, rep_id)`로 중복 반영을 방지하세요. `POSE_UPDATE`의 표시값까지 함께 점수에 더하면 중복됩니다. 캘리브레이션 진행 횟수는 `POSE_UPDATE`로 표시하고 완료는 `CALIBRATION_FINISHED`로 확인합니다.

`SERVER_STATUS`, `SESSION_PENDING`, `SESSION_PREPARING`, `SESSION_MEASUREMENT_STARTED`, `SESSION_LOG_STARTED`, `VIDEO_PROGRESS` 등은 준비 상태·진행률·로그 경로를 표시할 때 사용합니다. 모든 메시지는 계속 수신하되 사용하지 않는 `type`과 필드는 건너뛰면 됩니다. 별도 상태 조회는 선택 사항입니다.

## 선택 입력: 진행 중 가동 범위 수정

MAIN/TEST에서 같은 연결로 수정합니다. 좌우 중 필요한 쪽만 보낼 수 있습니다.

```json
{"type":"CMD_UPDATE_COUNT_THRESHOLD","exercise_name":"knee_extension","thresholds":{"left":{"target_val":0.6}}}
```

값은 해당 운동의 단위입니다. 각도는 도, 좌표와 길이는 신체 정규화 좌표 기준입니다. 측정 방향에 따라 목표값을 늘리는 것이 쉬워질 수도, 줄이는 것이 쉬워질 수도 있으므로 시작값과 함께 확인하세요. 응답 및 상세 오류는 [전체 명세](socket_server_guide.md)를 참고하세요.

## 현재 운동별 측정값

| 방식 | 운동 | 측정값 |
| --- | --- | --- |
| ANGLE | shoulder_press, clamshell, hip_knee_flexion | 설정된 세 관절의 각도 |
| COORDINATE | biceps_curl, slr | 손목 또는 발목의 정규화된 y 좌표 |
| LENGTH | knee_extension | 정규화된 무릎–발목의 2D 거리 |

현재 사용 방식은 3개입니다. 엔진의 RATIO 계산 기능은 남아 있지만 현재 운동 설정에서는 사용하지 않습니다. 측정값을 공통 반복 판정기에 넣어 횟수와 품질을 구합니다.

무릎 신전은 기존 RATIO에서 LENGTH로 변경되었습니다. 이전 RATIO 개인 기준값은 호환되지 않으므로 CALIBRATION을 다시 실행하세요. 길이는 픽셀이나 cm가 아닌 몸통 길이 기준 단위이며, 기본 시작값 2.0/목표값 1.0은 개인 측정값으로 대체해서 사용해야 합니다. 2D 투영 거리이므로 카메라 방향에 따라 변화량이 작을 수 있습니다.
