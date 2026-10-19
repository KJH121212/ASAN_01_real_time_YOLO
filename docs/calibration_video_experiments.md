# 실영상 캘리브레이션 실험

`tests/evaluate_calibration_videos.py`는 실제 YOLO/RTMPose 가중치를 한 번 로드하고,
`test_video`의 영상을 현재 소켓 서버의 `process_offline_video` 경로로 분석합니다.
테스트용 가짜 좌표를 사용하는 단위 테스트와 별개의 수동 실험 도구입니다.

## 실행

프로젝트의 모델 의존성이 설치된 Python 환경에서 실행합니다.

```powershell
python tests/evaluate_calibration_videos.py --inventory-only --output data/calibration_experiments/inventory
python tests/evaluate_calibration_videos.py --output data/calibration_experiments/baseline --batch-size 8
python tests/analyze_calibration_experiment.py data/calibration_experiments/baseline
```

특정 운동만 먼저 시험할 수도 있습니다. `--exercise`는 여러 번 지정할 수 있습니다.
`--limit-per-exercise`는 파일명 순서의 첫 N개를 선택합니다.

```powershell
python tests/evaluate_calibration_videos.py --exercise shoulder_press --limit-per-exercise 3 --output data/calibration_experiments/shoulder_sample
```

기존 결과를 덮어쓰지 않도록 `results.csv`가 이미 있는 출력 폴더는 거부합니다.
실험마다 새 출력 폴더를 지정하세요.

## 입력 매핑과 조건

| 영상 폴더 | 코드 운동 이름 |
|---|---|
| biceps_curl | biceps_curl |
| shoulder_press | shoulder_press |
| clamshell | clamshell |
| slr | slr |
| leg_raise | hip_knee_flexion |
| leg_extension | knee_extension |

`leg_raise`에는 sitting/hooklying 등 자세 변형이 있어 동일한 측정식이 적합한지 별도로 확인해야 합니다.

- 프레임 샘플링·영상 축소 없이 원본 입력을 사용합니다.
- 시간은 현재 오프라인 서버와 같은 소스 FPS 기준입니다.
- One Euro Filter, 정규화, 기본 ROM, 양측 각각 3회, 1초 마무리 조건을 그대로 사용합니다.
- 실시간 전용 5초 준비 시간은 적용하지 않습니다.
- 실험용 환자 ID와 프로필은 출력 폴더의 `sessions` 안에 격리합니다.
- 원본 영상은 수정하지 않고 MP4를 다시 저장하지 않습니다.

## 결과 파일

| 파일/폴더 | 내용 |
|---|---|
| inventory.csv | 파일 목록, 운동 매핑, FPS·프레임 수·해상도·길이 |
| results.csv | 영상별 완료 여부, 좌우 횟수, 검출 누락률, 측정값 범위, 추정 기준값, 실행 시간 |
| report.md | 처리된 영상의 기본 집계. 실행 중에도 갱신 |
| diagnostics_summary.csv | 분석 도구가 추가하는 복귀 상태·수집 전 카운트 진단 |
| analysis_ko.md | 분석 도구가 생성하는 한국어 보고서 |
| plots/ | 대표 완료/미완료 영상의 측정값·반복 횟수 그래프 |
| raw_keypoints/ | 전체 입력 영상의 추론 좌표 캐시. 배열은 프레임 × 17 × x/y/score |
| sessions/ | 현재 서버가 저장하는 CSV·NPZ·프로필·세션 요약 |
| diagnostics/ | 영상별 실행 로그와 오류 |
| exercise_config_snapshot.json | 실험 시작 시 운동 설정 |
| environment.json | 장치, 라이브러리 버전, 실행 조건, 관련 코드 해시 |

## 완료 판정의 의미

`FINISHED`는 마지막 프레임 로그가 실제 캘리브레이션 `FINISHED` 상태인 경우입니다.
현재 서버는 영상 끝까지 완료하지 못해도 `COMPLETED` 요약과 개인 기준값을 저장하므로,
실험에서는 이를 그대로 성공으로 집계하지 않습니다. `EOF_NOT_FINISHED`는 영상 종료 시 미완료,
`ERROR`는 실행 오류입니다.

양측 3회라도 1초 마무리 시간이 부족하면 `EOF_NOT_FINISHED`입니다.
또한 현재 코드가 3초 COUNTDOWN 중에도 FSM을 갱신하는 점을 분석 도구에서 별도로 표시합니다.

이는 캘리브레이션 완료 실험입니다. 영상별 실제 횟수·정답 대상자·시작/종료 구간 라벨이 없으므로,
완료율을 동작 인식 정확도로 해석하면 안 됩니다. 원인 후보 분류도 추가 영상 판독을 위한 단서입니다.
