# 캘리브레이션 진행과 설정

캘리브레이션은 일반 카운트와 별도로 초기 자세를 측정하고,
각 측의 상승/하강 및 복귀가 완료된 동작을 수집합니다.
좌우 각각 최근 3회의 움직임이 일관되면 1초 마무리 대기 후 개인 기준값을 저장합니다.
완료 조건을 충족하지 않은 상태에서는 개인 기준값을 저장하지 않습니다.

## 변경된 동작

- 초기 자세는 정지 시간과 최소 3개 샘플로 판정합니다. 0.5초 구간에 15프레임을 요구하지 않습니다.
  낮은 FPS에서는 최소 샘플이 모일 때까지 더 기다립니다. 움직이는 중에는 기준 자세를 확정하지 않습니다.
- 2초 미만의 일시 누락은 진행 중인 한 동작의 유지 시간을 취소하고, 기준 자세와 이미 완료한 횟수는 보존합니다.
  누락이 2초 이상 지속되면 초기 자세와 횟수를 다시 측정합니다.
- 복귀 허용 범위는 `max(정지 허용 범위, 최소 움직임 × 0.4, 해당 동작의 최대 이동량 × 0.25)`이며,
  최대 이동량의 절반을 넘지 않습니다. 반전과 복귀 유지 시간도 함께 확인합니다.
- 복귀한 위치에서 다시 최소 이동량만큼 움직여야 다음 반복을 시작합니다. 복귀 위치에 멈춘 것만으로 재카운트하지 않습니다.
- 최근 3회의 크기 차이가 너무 크면 추가 반복을 요청합니다. 일관성 조건을 강제로 생략하지 않습니다.
- 실시간 판정에는 캡처 시각의 단조 증가 시계를 사용하고, 오프라인 판정에는 영상 시간을 사용합니다.

## 운동별 설정

`configs/exercise_kpt_config_coco.json`에서 각 운동의 `calibration_config`에 필요한 값만 지정합니다.
변경된 파일은 새 세션을 시작할 때 적용됩니다.

```json
"calibration_config": {
  "rest_seconds": 0.5,
  "min_baseline_samples": 3,
  "max_sample_gap": 1.0,
  "occlusion_reset_seconds": 2.0,
  "return_fraction": 0.25,
  "consistency_ratio": 2.5
}
```

| 항목 | 기본값 | 의미 |
|---|---:|---|
| rest_seconds | 0.5 | 초기 자세의 최소 정지 시간(초) |
| min_baseline_samples | 3 | 초기 자세의 최소 관측 샘플 수. 3 이상 |
| max_sample_gap | 1.0 | 유지 판정을 연속으로 볼 최대 관측 간격(초) |
| occlusion_reset_seconds | 2.0 | 누락 후 전체 초기화까지의 시간(초) |
| return_fraction | 0.25 | 최대 이동량에 비례하는 복귀 허용 비율. 0 초과, 0.5 미만 |
| consistency_ratio | 2.5 | 최근 3회 중 최대/최소 이동량의 허용 배수 |
| hold_seconds | 0.15 | 최소 이동량 이상에서 유지할 시간(초) |
| return_seconds | 0.15 | 복귀 범위 안에서 유지할 시간(초) |
| max_cycle_seconds | 10.0 | 한 동작 수집의 제한 시간(초) |

`min_rise`는 ANGLE에서 기본 8도, 그 외 측정 방식에서 0.08입니다.
`baseline_range`는 ANGLE에서 기본 4도, 그 외 방식에서 0.04입니다.
이 두 항목은 운동의 측정 단위에 맞춰 조절해야 합니다.

## 화면과 로그 확인

Streamlit은 초기 자세 대기, 복귀 대기, 일관성 부족, 검출 누락을 구분해 안내합니다.
좌우 응답의 `calibration`에 상태·기준값·최소 이동량·복귀 허용 범위·완료 횟수가 포함됩니다.
프레임 CSV에도 `left/right_calibration_status`, `left/right_calibration_baseline`,
`left/right_calibration_return_tolerance`를 저장합니다.

`BASELINE`이면 초기 자세를 잠시 유지하고, `RETURNING`이면 시작 자세로 복귀해 잠시 멈춥니다.
`NEED_CONSISTENT_CYCLES`이면 비슷한 크기의 동작을 추가로 반복합니다.

## 검증

```powershell
python -m unittest tests.test_calibration_robustness tests.test_measurement_methods -v
```

가짜 신호 테스트는 저속·가변 FPS, 양방향, 짧은 누락과 지속 누락, 정지 상태 중복 카운트,
불규칙한 반복과 완료 후 저장을 검증합니다. 실제 환자 영상의 인식 정확도 검증과는 별개입니다.
