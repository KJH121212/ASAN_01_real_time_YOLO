# 운동별 가동 범위 수정 및 테스트

로그 폴더·세션 파일명·Unity 저장 설정은 [log_storage.md](log_storage.md)를 참고하세요.

Count threshold는 운동의 기준 가동 범위(ROM)인 `start_val`과 `target_val`입니다.
누적 목표 점수(`target_score`)와는 별개입니다. 최소 진입 40%, 복귀 30%,
방향 전환 5%, 품질 등급 비율은 변경한 가동 범위를 기준으로 계산됩니다.

MAIN/TEST는 Perfect +1점, Good +0.5점, Bad +0.25점을 누적합니다.
좌우 중 한쪽이 목표 점수에 도달하면 기존처럼 1.5초 후 종료합니다.
실제 반복 횟수 `rep_count`는 별도로 유지하며, 캘리브레이션은 양측 각각 3회 기준입니다.
시작 요청의 `target_score`가 우선하고, 생략하면 기존 `target_reps`를 목표 점수로 해석합니다.
예를 들어 목표 40점은 모두 Perfect면 40회, Good이면 80회, Bad이면 160회입니다.

각 좌우 응답의 `total_score`는 누적 점수, `score_delta`는 이번 프레임의 획득 점수입니다.
`score`는 새 반복 판정 시 1=Bad, 2=Good, 3=Perfect이고 그 외 프레임은 0입니다.
`quality`는 마지막 판정 문자열을 유지합니다. 가동 범위를 수정해도 기존 점수는 보존합니다.

## Unity → 서버

기존 `CMD_SET_SESSION`으로 MAIN 또는 TEST 세션을 시작한 **동일한 연결**에서 보냅니다.
현재 세션의 `exercise_name`을 반드시 지정합니다. 왼쪽/오른쪽을 독립적으로 수정할 수 있고,
생략한 쪽이나 값은 유지됩니다. 다음 예시는 이두 굴곡의 좌우 목표 높이를 줄입니다.

```json
{
  "type": "CMD_UPDATE_COUNT_THRESHOLD",
  "request_id": "rom-001",
  "exercise_name": "biceps_curl",
  "thresholds": {
    "left": {"start_val": 0.1, "target_val": 0.5},
    "right": {"start_val": 0.1, "target_val": 0.5}
  }
}
```

단위는 `ANGLE`은 도(0~180), `COORDINATE`는 골반 중심·몸통 길이 기준 상대 높이,
`LENGTH`는 엔진에서 계산한 거리 지표, `RATIO`는 두 거리의 무차원 비율(0 이상)입니다. 각도값을 상대 높이 운동에 보내면 안 됩니다.
INCREASING은 `target_val > start_val`, DECREASING은 반대여야 합니다.
DECREASING 운동은 목표값을 시작값 쪽으로 **올려야** 요구 가동 범위가 줄어듭니다.

예를 들어 이두 굴곡의 `0.1 → 0.9`를 `0.1 → 0.5`로 줄이면 기준 범위는
`0.8 → 0.4`로 줄어듭니다. 카운트에 필요한 최소 이동량은 `0.32 → 0.16`이 됩니다.
목표값에 도달하는 것만으로 카운트되지는 않고, 기존과 같이 방향 전환을 확인합니다.

조회:

```json
{"type": "CMD_GET_COUNT_THRESHOLD", "request_id": "rom-query-1", "exercise_name": "biceps_curl"}
```

## 서버 → Unity

수정 응답은 `COUNT_THRESHOLD_UPDATED`, 조회 응답은 `COUNT_THRESHOLD_STATUS`입니다.
응답에는 `request_id`, `exercise_name`, `metric_type`, `motion_direction`,
`thresholds`(좌우 전체 값), `threshold_version`, `changed_sides`, `states`, `scope`가 들어갑니다.
`POSE_UPDATE`에도 현재 `thresholds`와 `threshold_version`을 포함합니다.

잘못된 요청은 `ERROR`와 `code`로 반환하며 세션은 계속됩니다.
코드는 `NO_ACTIVE_SESSION`, `EXERCISE_MISMATCH`, `INVALID_MODE`, `INVALID_THRESHOLD`입니다.
양쪽을 수정하는 요청에서 한쪽이라도 잘못되면 양쪽 모두 변경하지 않습니다.

## 변경 시 동작

- 유효한 변경은 즉시 적용됩니다. 완료한 반복 횟수·품질·기록은 유지합니다.
- 수정한 쪽의 진행 중 반복과 최고점만 폐기합니다. 반대쪽 진행 상태는 유지합니다.
- 수정한 쪽은 `RETURNING` 상태가 되며 **새 범위의 30% 이내로 복귀한 뒤** 다음 반복을 셉니다.
- 이미 목표 점수를 채운 쪽은 `WAITING`을 유지합니다.
- 동일한 값을 다시 보내면 진행 중 동작을 초기화하거나 버전을 올리지 않습니다.
- 수정 범위는 **현재 세션**입니다(`scope: SESSION`). 기본 설정 파일과 환자별 저장 프로필은 변경하지 않습니다.
- 오프라인 분석은 처리 중 명령을 수신하지 않으므로 런타임 수정 테스트는 실시간 세션에서 진행합니다.

## 자동 테스트: 카메라와 모델 없이

프로젝트 루트에서 실행합니다. NumPy와 websockets가 설치된 환경이 필요합니다.

```powershell
conda activate rtm_env
python -m unittest discover -s tests -v
```

가동 범위 축소 전후 카운트, 증가/감소 운동, 수정 중 오카운트 방지, 좌우 독립성,
기존 횟수 유지, 요청 검증 및 실제 로컬 WebSocket/JSON 왕복을 검사합니다.
WebSocket 테스트는 모델과 카메라 대신 공용 명령 처리기를 사용하는 임시 서버를 엽니다.

## Streamlit으로 수동 테스트

터미널 두 개에서 같은 Python 환경을 사용합니다.

```powershell
conda activate rtm_env
python network/socket_server.py
```

```powershell
conda activate rtm_env
streamlit run app.py
```

1. 화면에서 환자·운동·카메라 번호를 선택합니다. 가동 범위 수정은 실시간 `TEST` 또는 `MAIN`에서 지원합니다.
2. **세션 시작**을 누릅니다. 모드 1은 영상·스켈레톤, 모드 2는 Unity처럼 수치만 표시합니다.
3. **서버 적용값** 표에서 현재 시작값·목표값·버전을 확인합니다.
4. **수정 대상**을 양쪽/왼쪽/오른쪽으로 선택하고 시작값·목표값을 입력한 다음 **가동 범위 적용**을 누릅니다.
5. 적용 완료와 버전 변경을 확인하고 시작 자세로 돌아온 뒤 같은 작은 움직임을 비교합니다.
6. **세션 중지**를 누르면 연결을 닫습니다. 서버는 기존 종료 경로에서 기록을 저장하고 카메라를 해제합니다.

입력 중에는 요청을 보내지 않으며 **가동 범위 적용** 버튼으로 확정합니다.
입력값과 서버에서 실제 적용된 값은 따로 표시합니다. **현재 적용값 불러오기**로 입력칸을 되돌릴 수 있습니다.
화면과 폼을 갱신해도 세션 시작 명령을 다시 보내지 않고 같은 WebSocket으로 수정 요청을 전송합니다.
연결 처리는 백그라운드에서, 화면 갱신은 Streamlit fragment에서 수행합니다(Streamlit 1.37 이상 필요).
Streamlit은 카메라를 직접 열거나 모델을 로드하지 않습니다.

추가 자동 테스트는 모의 소켓 서버와 Streamlit AppTest로 화면 적용·거절·좌우 선택·연결 유지·재시작을 검사합니다.
실제 GPU·웹캠 검증은 위 절차로 별도 수행합니다.

## 콘솔 클라이언트로 수동 테스트

첫 번째 터미널:

```powershell
conda activate rtm_env
python network/socket_server.py
```

두 번째 터미널:

```powershell
conda activate rtm_env
python network/threshold_test_client.py --exercise-name biceps_curl
```

클라이언트가 `threshold_test` 환자의 MAIN 세션을 시작하고 웹캠 측정을 요청합니다.
다음 명령을 한 줄씩 입력합니다.

```text
show
both 0.1 0.9
both 0.1 0.5
left 0.1 0.4
show
quit
```

같은 작은 움직임을 축소 전후로 비교합니다. 수정 응답에서 버전과 실제 적용 범위를 확인하고,
수정 후에는 팔을 시작 자세로 내린 다음 다시 움직입니다. 콘솔에는 좌우 횟수·상태·품질이 표시됩니다.
`quit`은 연결을 닫아서 서버의 세션 저장 및 카메라 해제를 실행합니다.
CLI는 `--camera-index`, `--player-id`, `--target-reps`, `--uri`도 지원합니다.

## 실제 모델의 오프라인 배치 추론 확인

`SkeletonEngine.extract_keypoints_batch()`는 YOLO 검출과 RTMPose 추론을 배치로 실행합니다.
사람을 검출하지 못한 프레임은 `None`으로 남겨 시간과 프레임 인덱스를 유지합니다.
오프라인 경로는 마지막에 배치 크기보다 적게 남은 프레임도 처리합니다.

로컬 가중치와 실제 영상으로 3프레임을 확인하려면 다음 명령을 실행합니다.

```powershell
python tests/smoke_batch_runtime.py
```

다른 영상은 `--video 경로`, CPU 검증은 `--device cpu`를 지정합니다.
첫 3프레임에 사람이 보여야 하며 성공 시 `BATCH_SMOKE_OK`를 출력합니다.
이 검증은 모델의 배치 추론 확인용이고 전체 영상의 운동 판정·로그 저장 검증과는 별개입니다.
실행 중인 소켓 서버에는 변경 사항이 자동 반영되지 않으므로 코드를 수정한 후 서버를 재시작합니다.

## 서버 시작과 모델 준비 상태

소켓은 먼저 열고 Torch·MMPose import 및 모델 로딩은 백그라운드에서 한 번 실행합니다.
카메라는 세션이 실제로 시작될 때만 엽니다. 모델은 세션 종료 후에도 메모리에 유지합니다.

서버는 연결 시 및 준비 단계가 바뀔 때 `SERVER_STATUS`를 보냅니다.
`model_state`는 `LOADING` / `READY` / `FAILED`, `ready`는 추론을 시작할 수 있는지 나타냅니다.
`phase`, `message`, `elapsed_seconds`, `phase_seconds`로 단계와 소요 시간을 확인할 수 있습니다.
다음 요청으로 현재 상태를 조회할 수도 있습니다.

```json
{"type": "CMD_GET_SERVER_STATUS"}
```

모델 준비 전에 `CMD_SET_SESSION`이 들어오면 `SESSION_PENDING`을 반환하고,
준비 완료 후 요청한 세션을 한 번 시작합니다. 초기 가동 범위 조회도 세션 시작 뒤 처리합니다.
대기 중 연결을 닫으면 그 세션은 시작하지 않습니다. 준비 실패 시 `MODEL_LOAD_FAILED` 오류를 반환합니다.
Streamlit에는 준비 단계와 경과 시간을 표시합니다.

카메라를 열지 않고 실제 모델의 시작 시간을 측정하려면 다음 명령을 실행합니다.

```powershell
python tests/benchmark_server_startup.py
```

개발 PC의 비교 측정에서 기존에는 소켓을 열기 전까지 약 9.0초(import 7.4초, 모델 생성 1.6초)가 걸렸습니다.
변경 후 연결 가능 시점은 약 0.2초였고, 모델 준비는 실행 후 약 9.1초에 완료됐습니다.
이 변경은 연결과 준비 상태 확인을 앞당깁니다. 최초 모델 준비 비용은 남아 있으며 이후 세션에서는 재사용합니다.
