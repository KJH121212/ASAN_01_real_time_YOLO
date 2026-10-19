# 로그 저장 위치

기본 위치는 프로젝트 루트의 `./data`입니다. 실행 작업 디렉터리와 관계없이 프로젝트 기준으로 결정합니다.
현재 프로젝트에서는 `C:\Users\kjh\code\ASAN_01_real_time_YOLO\data`입니다.
Streamlit의 **로그 저장 폴더 (서버 기준)** 입력칸에서 바꿀 수 있습니다.
UI에 표시하는 최종 경로는 서버가 `SESSION_LOG_STARTED`로 응답한 실제 경로입니다.

서버의 기본 경로는 환경 변수 `ASAN_LOG_ROOT`로도 지정할 수 있습니다.
우선순위는 세션 JSON의 `log_root` → 환경 변수 → 기본 위치 순입니다.
경로는 소켓 서버가 실행되는 컴퓨터의 경로입니다.

```text
data/
└─ 0001/
   ├─ patient_metadata.csv
   ├─ 0001.json                         # 개인별 캘리브레이션 프로필
   └─ 261019/                          # 한국 시간 기준 YYMMDD
      ├─ gamedata.csv                  # 세션별 요약
      └─ biceps_curl/
         ├─ session1_cali.csv
         ├─ session1_cali.npz
         ├─ session1_cali.mp4           # 녹화 선택 시
         ├─ session1_round1_set1.csv
         ├─ session1_round1_set1.npz
         ├─ session1_round1_set1.mp4
         ├─ session1_round1_set1_reps.csv
         └─ session1_round1_set1.json   # 실제 경로·시작 시각·완료 상태·기록 수
```

같은 파일명이 있으면 `_take2`, `_take3` 등을 붙여 이전 파일을 보존합니다.
Session/Round/Set은 0 이상의 정수이며, 생략한 Session은 해당 날짜·운동의 다음 번호를 선택합니다.
같은 캘리브레이션과 본 운동을 한 세션으로 묶으려면 동일한 Session 번호를 보내면 됩니다.

## Unity / Streamlit 시작 요청

```json
{
  "type": "CMD_SET_SESSION",
  "patient_id": "0001",
  "patient_name": "환자 이름",
  "exercise_name": "biceps_curl",
  "mode": "TEST",
  "cal": false,
  "session": 1,
  "round": 1,
  "set": 1,
  "target_score": 40,
  "output_mode": "LIGHTWEIGHT",
  "record_video": true,
  "preparation_seconds": 5,
  "log_root": "D:\\ASAN_Logs",
  "patient_metadata": {"height_cm": 170, "weight_kg": 65}
}
```

기존 `player_id`도 지원합니다. `patient_id`는 `"0001"`처럼 문자열로 보내야 앞자리 0이 유지됩니다.
`mode: CALIBRATION`이면 `cal: true`로 보내거나 `cal`을 생략합니다.
`record_video` 기본값은 false이며, true이면 이미지 전송 여부와 관계없이 서버에서 녹화합니다.
Streamlit은 더 이상 자체 `recordings` 폴더에 영상을 저장하지 않습니다.

실시간 웹캠 세션은 카메라 준비 후 기본 5초 동안 시작 자세를 잡을 시간을 제공합니다.
이 동안 추론·필터·카운트·캘리브레이션·프레임 로그·녹화를 수행하지 않습니다.
서버는 `SESSION_PREPARING`의 `remaining_seconds`로 남은 초를 알리고,
종료 시 `SESSION_MEASUREMENT_STARTED`를 보냅니다. Streamlit에는 카운트다운이 표시됩니다.
CSV/NPZ 경과 시간과 세션 길이에는 준비 시간이 포함되지 않습니다.
준비 중 중지하면 프레임 수와 측정 길이가 0인 로그가 남습니다.

준비 시간을 없애려면 Streamlit의 **측정 전 준비 시간 (초)** 또는 시작 JSON의
`preparation_seconds`를 `0`으로 설정합니다. 공통 기본값은
`network/session_commands.py`의 `DEFAULT_PREPARATION_SECONDS` 한 곳에서 변경합니다.
오프라인 영상 분석에는 이 준비 시간을 적용하지 않습니다.
캘리브레이션 자체의 전신 감지 후 3초 확인 단계는 별도로 유지됩니다.

## 저장 내용

- 프레임 CSV: 세션 시작 후 경과 `timestamp_ms`, 분석 프레임 번호 `frame_idx`, `video_frame_idx`,
  운동·측정 방식, 좌우 측정값/실제 반복 횟수/상태/품질, 누적 점수 `left/right_total_score`,
  해당 프레임 획득 점수 `left/right_score_delta`, 검출 누락 여부, 적용 가동 범위·버전.
- NPZ: 원본 화면 비율 좌표 `keypoints`(프레임 × 17 × x/y/score), 측정값, 시간, 프레임 인덱스.
  smoothing/신체 기준 정규화 이전 좌표이며 누락 좌표는 NaN, 신뢰도는 0입니다.
  `total_scores`, `score_deltas`는 각각 (프레임 수, 2) 모양의 좌우 점수 배열입니다.
- MP4: 화면 축소·스켈레톤 오버레이 이전의 입력 프레임. 실시간에서는 웹캠이 읽기를 완료한
  시각(`perf_counter`)을 프레임과 함께 전달합니다. MP4의 저장 FPS는 30이지만 입력을 30 FPS로
  가정하지 않습니다. 느리거나 불규칙한 입력 사이에는 직전 이미지를 유지하고, 같은 출력 시간대에
  여러 입력이 들어오면 추가 프레임을 쓰지 않아 실제 경과 시간과 재생 시간을 맞춥니다.
  종료·연결 해제 시각까지 마지막 이미지도 유지합니다. 시간 오차는 출력 한 프레임(약 33 ms) 이내이며,
  입력 FPS가 낮으면 움직임은 끊겨 보일 수 있습니다. 같은 캡처 프레임은 중복 분석·기록하지 않습니다.
  첫 영상 프레임의 세션 경과 시간은 manifest의 `video_time_origin_sec`, 실제 재생 길이는
  `video_duration_sec`이며, CSV/NPZ의 `video_frame_idx`로 대응시킵니다.
- 오프라인 녹화는 입력 영상 FPS로 전체 입력을 기록합니다. 목표 도달로 판정을 일찍 끝내면
  CSV/NPZ는 그 시점까지이고, MP4에는 뒤쪽 입력 프레임도 포함될 수 있습니다.
- `*_reps.csv`: 반복별 측정 범위·품질·적용 가동 범위·획득 점수 `score_delta`·누적 점수 `total_score`.
- `gamedata.csv`: 세션 ID·시작 시각·종료 상태·기록 프레임 수·좌우 횟수·파일 경로.
  좌우 누적 점수 `left/right_total_score`와 목표 점수 `target_score`도 기록합니다.
  칼로리와 최대 속도는 아직 계산하지 않으므로 빈 값입니다.
- `patient_metadata.csv`: 환자 ID·이름과 제공된 키/몸무게. 제공하지 않은 기존 값은 유지합니다.

CSV는 프레임마다 기록하며 NPZ·반복 기록·요약은 정상 종료 또는 연결 종료 시 확정합니다.
진행 중에는 같은 이름의 JSON에서 `RECORDING`, 완료 시 `COMPLETED`, 중지 시 `STOPPED`,
처리 오류 시 `FAILED` 상태를 확인할 수 있습니다. 프로세스를 강제 종료하면 NPZ가 없거나
MP4가 완성되지 않을 수 있습니다.

기존 파일은 이동하거나 삭제하지 않습니다. 이전 외부 저장 위치의 로그도 자동으로 옮기지 않습니다.
새 위치에 프로필이 없으면 기존 `data/{환자ID}/{환자ID}.json`을 읽어 이전 임계값을 사용합니다.
새 캘리브레이션과 로그는 새 경로에만 씁니다. 실행 중인 서버는 재시작해야 반영됩니다.

## 검증

```powershell
python -m unittest tests.test_session_log tests.test_live_video_timing tests.test_offline_video tests.test_streamlit_app -v
```

테스트는 임시 폴더에 실제 CSV·NPZ·MP4를 생성하고, 세션 번호·충돌 방지·기존 프로필 유지·
누락 프레임·저장 오류·Streamlit의 요청/경로 표시를 확인합니다.
