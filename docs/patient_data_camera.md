# PatientData 공통 규격과 카메라 로그

기준 문서는 `PatientData_사용안내.pdf`(2026-10-22)입니다. Unity와 **공통 폴더·ID를 사용하고 카메라 전용 로그를 저장**합니다. Unity의 PlayData 엑셀·원본 JSON, 등록 예약 파일, IMU 복구 자료는 Python 서버에서 변경하지 않습니다.

## 저장 위치

기본 위치는 Python 실행 시 프로젝트 루트의 `PatientData`, exe 실행 시 exe 옆 `PatientData`입니다. 우선순위는 시작 JSON의 `log_root` → 서버 환경 변수 `ASAN_LOG_ROOT` → 기본 위치입니다. `log_root`에는 **PatientData 최상위 폴더**를 넣습니다. Unity와 서버의 실행 폴더가 다르면 Unity가 쓰는 PatientData 경로를 직접 지정해야 합니다. 경로는 서버 PC 기준입니다.

```text
PatientData/
├─ _Registrations/                      # Unity 관리: 서버는 읽거나 수정하지 않음
└─ v1.0.0(0)/
   └─ 0001/
      └─ 01/                           # Unity가 발급한 등록 차수
         ├─ PlayData/                   # Unity 관리: 일별 엑셀
         ├─ Sensors/
         │  └─ 2026-10-22/             # 수집 시작일(KST)
         │     └─ PLAY_ID/
         │        ├─ ..._Left_....csv   # Unity IMU 파일
         │        ├─ ..._Right_....csv
         │        └─ Camera/
         │           ├─ R01_S02_Set01_Play_Camera_FILE_ID.csv
         │           ├─ R01_S02_Set01_Play_Camera_FILE_ID.npz
         │           ├─ R01_S02_Set01_Play_Camera_FILE_ID.mp4
         │           ├─ R01_S02_Set01_Play_Camera_FILE_ID_reps.csv
         │           └─ R01_S02_Set01_Play_Camera_FILE_ID.json
         └─ Management/
            ├─ Records/                # Unity 플레이 JSON: 서버는 변경하지 않음
            ├─ Recovery/               # Unity IMU 복구: 서버는 변경하지 않음
            └─ Camera/
               ├─ profile.json         # 카메라 개인 가동 범위와 이력
               ├─ patient_metadata.csv # 카메라 측에서 받은 환자 정보
               ├─ Records/FILE_ID.json # 카메라 측정 구간별 관리 기록
               └─ Daily/2026-10-22.csv # 카메라 요약: 측정 종료일(KST)
```

`Camera` 하위 폴더와 `Management/Camera`는 이번 연동을 위한 카메라 전용 확장입니다. IMU CSV의 Accel·Gyro 열을 임의로 채우거나 카메라 결과를 IMU 결과처럼 저장하지 않습니다.

## Unity가 보내는 시작 JSON

```json
{
  "type": "CMD_SET_SESSION",
  "patient_id": "0001",
  "data_version": "v1.0.0(0)",
  "registration": "01",
  "play_session_id": "PLAY_ID",
  "exercise_record_id": "EXERCISE_ID",
  "exercise_name": "knee_extension",
  "round": 1,
  "slot": 2,
  "set": 1,
  "mode": "MAIN",
  "log_root": "C:\\Game\\PatientData"
}
```

| 입력 | 의미 |
| --- | --- |
| `patient_id` | 네 자리 환자번호. 숫자 문자열은 네 자리로 보정하며 앞의 0을 유지 |
| `data_version` | BuildInfo가 만든 최종 폴더명. 예: `v2.3.4(27)`. develop/live 환경명은 제외 |
| `registration` | Unity가 예약한 등록 차수. 정수 1 또는 문자열 `01` |
| `play_session_id` | Unity 전체 플레이의 세션 ID. 같은 플레이의 모든 운동·세트에서 유지 |
| `exercise_record_id` | 해당 라운드·운동 슬롯의 기록 ID. 같은 운동 기록에 속한 여러 세트에서 유지 |
| `round`, `slot`, `set` | 라운드·처방 운동 순서·세트. 슬롯은 운동 이름과 다른 개념 |
| `mode` | `CALIBRATION`이면 Calibration 파일, `MAIN`/`TEST`이면 Play 파일 |

버전 기본값은 `v1.0.0(0)`, 등록 차수는 `01`, 슬롯은 1입니다. 공통 ID가 없으면 단독 테스트용 ID를 생성합니다. **Unity와 실제로 연결할 때는 기본값에 의존하지 말고 Unity가 사용하는 값을 전달하세요.** ID에는 영문·숫자·밑줄·하이픈을 사용할 수 있습니다. 비숫자 환자 ID는 기존 테스트 호환용이며 실제 환자는 네 자리 번호를 사용하세요.

이전 숫자 `session`은 호환용 부가 정보로 유지하지만 공통 플레이 ID를 대신하지 않습니다. 서버 응답의 `session_id`도 기존의 **카메라 측정 구간 ID**입니다. 클라이언트의 이벤트 중복 방지와 중지 명령에는 계속 이 값을 사용합니다.

`SESSION_LOG_STARTED`, 종료 summary, CSV, NPZ, JSON에 `play_session_id`, `exercise_record_id`, `sensor_file_id`가 들어갑니다. `REP_COMPLETED`에도 세 ID를 함께 제공합니다. 각 구간의 새 `sensor_file_id`를 파일명에 넣으므로 같은 라운드·슬롯·세트를 다시 측정해도 파일을 덮어쓰지 않습니다.

응답의 절대 경로와 함께 `frame_csv_relative`, `skeleton_npz_relative`, `video_mp4_relative` 등 등록 차수 폴더 기준 상대 경로도 제공합니다. 전체 폴더를 다른 PC로 복사할 때는 상대 경로로 원본을 찾을 수 있습니다.

## 카메라 전용 저장 항목

| 파일 | 내용 |
| --- | --- |
| 프레임 CSV | 한 행당 분석 프레임. 공통 ID, `time`(초), `timestamp_ms`, 프레임 번호, 운동명, 측정 방식·단위, 좌우 값·횟수·품질·상태, 적용 가동 범위·버전, 가림·캘리브레이션 진단 |
| NPZ | 신체 정규화·평활화 전 관절 좌표 `(N,17,3)`, 원본 종횡비, 측정값·시각·프레임 번호·점수 배열과 공통 ID |
| MP4 | 오버레이 전 입력 영상. 실시간 입력 시각으로 30 FPS 출력 시간을 맞추고, 파일 분석은 원본 FPS를 사용 |
| 반복 CSV | 인정한 반복별 좌우·범위·품질·점수·적용 기준. 프레임 행 수는 동작 횟수가 아님 |
| 옆의 JSON | RECORDING/COMPLETED/STOPPED/FAILED 상태와 파일 위치·공통 ID·기록 수·종료 요약 |
| Management/Camera/Records JSON | 카메라 구간의 관리용 요약. Unity Records JSON과 다른 파일 |
| Management/Camera/Daily CSV | 종료일별 카메라 구간 요약. Unity의 플레이 요약·운동 상세를 대체하지 않음 |

각도 단위는 `deg`, 좌표·거리는 몸통 길이를 1로 한 `body_length`, 비율 지표는 `ratio`입니다. IMU 문서의 FreeAccel·적분 각도·P95 기준과 카메라의 좌표·85% 목표값은 서로 다른 지표입니다. 카메라 기준값을 RefP95라고 기록하지 않습니다.

기본 Perfect/Good/Bad 참고 점수는 1/0.5/0.25이며 실제 반복 횟수와 구분합니다. 게임의 목표 반영·넘어가기·설문·처방 달성률은 Unity가 관리합니다. 칼로리·최대 속도는 현재 카메라 서버가 계산하지 않아 빈칸입니다.

## 날짜·기존 자료·복구 범위

원시 파일은 수집 시작일 KST에 남습니다. 카메라 일별 요약은 `시작 시각 + 측정 경과 시간`의 종료일에 기록합니다. 파일 일괄 분석의 경과 시간은 원본 영상 시각이므로 실제 분석에 걸린 벽시계 시간과 다릅니다. Unity 플레이 전체의 종료일은 Unity가 기록합니다.

기존 `data` 로그나 구형 버전 폴더는 자동 이동·이름 변경·병합하지 않습니다. 최초 기본 버전·첫 등록의 새 카메라 프로필이 없을 때만 구형 개인 기준값을 보조로 읽을 수 있습니다. 새 버전·재등록 환자에게 이전 프로필을 자동 적용하지 않습니다. 새 기준값은 현재 카메라 관리 폴더에만 씁니다.

CSV는 프레임마다 저장하고 NPZ·반복 CSV·요약은 종료 시 확정합니다. JSON은 임시 파일을 완성한 뒤 교체합니다. **PDF의 Unity IMU 복구 큐·5초 재시도·엑셀 재생성 기능을 Python에 복제한 것은 아닙니다.** 강제 종료로 아직 저장하지 못한 NPZ나 미완성 MP4까지 복구를 보장하지 않습니다.

백업 시에는 개별 영상만 복사하지 말고 환자의 등록 차수 폴더 전체를 보관하세요. 전체 Unity 데이터 백업에는 `_Registrations`도 포함합니다. 기존 파일과 새 파일 모두 오버레이 기능에서 읽을 수 있습니다.

## 검증

```powershell
python -m unittest tests.test_patient_data_layout tests.test_session_log tests.test_offline_video
```

합성 데이터와 임시 폴더로 공통 ID·등록/버전 분리·자정 경계·CSV/NPZ 내용·Unity 파일 보존·오버레이 운동 식별을 확인합니다. 실제 환자 자료와 영상을 처리하지 않습니다.
