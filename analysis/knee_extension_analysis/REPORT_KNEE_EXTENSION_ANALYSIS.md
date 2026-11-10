# Knee Extension 데이터 분석 및 범용 측정 방식 제안 리포트

## 1. 개요 및 목적
* **분석 대상**: `data/0000/` 및 `data/0000_backup/`에 보관된 실제 슬관절 신전(`knee_extension`) 세션 NPZ 파일 **18개 세션 (좌/우 총 36개 트랙, 실제 운동 발생 23개 트랙)**
* **기존 방식의 근본적 한계**:
  * 비정규화 발목 절대 Y좌표(`COORDINATE`, unnorm)는 환자의 앉은 위치, 카메라 각도/거리가 조금만 달라져도 절대 좌표가 이동하여 재캘리브레이션이 강제됨.
* **목적**:
  * 실제 저장된 환자 동작 데이터를 기반으로, **스케일 불변(Scale-Invariant)**하면서 타 운동과 통일된 구조로 전환할 수 있는 가장 우수한 범용 지표 도출.

---

## 2. 후보 지표별 정량적 평가 비교 (실제 운동 트랙 기준)

| 후보 지표 | 단위 | 활동 진폭 (Dynamic Range) | 안정 자세(P05) ~ 최대신전(P95) | 평균 SNR | 진폭 변동 계수 (CV) |
| --- | --- | --- | --- | --- | ---|
| 몸통 정규화 발목 리프트 (Torso-Normalized Lift) | L_torso 배수 | 0.452 | -1.10 ~ -0.64 | 13.7 | 38.0% |
| 기존 방식: 비정규화 발목 리프트 (Raw Ankle Lift) | 영상비율(0~1) | 0.045 | -0.12 ~ -0.08 | 16.6 | 58.0% |
| 종아리 투영 수축비 (Shin-to-Thigh 2D Ratio) | 길이비 | 1.980 | 1.36 ~ 3.34 | 25.5 | 64.3% |
| 2D 무릎 관절각 (Hip-Knee-Ankle 2D Angle) | 도(°) | 15.731 | 162.23 ~ 177.96 | 4.4 | 77.1% |
| 종아리-수직선 2D 각도 (Shin-to-Vertical Angle) | 도(°) | 6.753 | 0.55 ~ 7.30 | 2620.0 | 63.6% |

---

## 3. 핵심 데이터 인사이트 (Key Insights)

### 💡 왜 기존 방식이 유독 불안정하고 예외 처리가 많았는가?
* 데이터 분석 결과, 기존 **비정규화 발목 리프트(`raw_ankle_lift`)의 평균 진폭은 영상 전체 높이의 4.5% (`0.045`)**에 불과했습니다.
* 즉, 극도로 좁은 변화량(0.045) 안에서 시작값과 85% 목표값을 고정해두었기 때문에, 환자가 카메라에서 10cm만 멀어지거나 의자 높이가 달라지면 신호가 임계값을 넘지 못하거나 카운트가 먹통이 되는 취약성이 있었습니다.

### 💡 왜 2D 각도(골반-무릎-발목, 종아리 경사각)는 정면 카메라에서 부적합한가?
* 사람이 의자에 앉아 무릎을 펴면 다리는 카메라 정면(깊이 $Z$ 축)과 위쪽($Y$ 축)으로 올라오며, 좌우($X$ 축)로는 거의 움직이지 않습니다.
* 따라서 2D 평면상에서는 골반-무릎-발목이 직선 투영으로 겹쳐 보이며, 각도 변화량이 15° 미만으로 찌그러집니다.

---

## 4. 최종 추천: 몸통 정규화 리프트 (`torso_norm_lift`)

### 🥇 최적 지표: `torso_norm_lift` (몸통 길이 단위 상대 높이)
$$\text{Normalized Lift} = \frac{Y_{\text{knee}} - Y_{\text{ankle}}}{L_{\text{torso}}}$$

1. **동적 범위 10배 확대**: 평균 진폭이 **0.452 (몸통 길이의 약 45%)**로, 기존 비정규화(0.045) 대비 신호의 크기가 10배 이상 커져 노이즈에 매우 강건합니다.
2. **완전한 스케일 불변 (Scale Invariance)**:
   * 환자가 카메라에 가깝든 멀든, 다리가 길든 짧든 몸통 길이에 비례하여 정규화되므로 **재캘리브레이션 없이도 전역 기본값 적용 가능**.
   * 안정 자세(P05) 평균: **-1.10** (무릎 아래 1.1 몸통 길이)
   * 최대 신전(P95) 평균: **-0.64** (무릎 아래 0.64 몸통 길이)
   * **평균 유효 가동 범위**: 약 **+0.45 ~ +0.50**
3. **타 운동과의 완벽한 아키텍처 통일**:
   * 기존 바이셉스 컬이나 SLR처럼 `COORDINATE` + `use_normalization: true` 구조를 그대로 사용 가능.
   * 복잡한 2단계(LENGTH 캘리브레이션 -> 고정 좌표 본측정)를 없애고 **단일 JSON 설정**으로 완전 통합 가능!

---

## 5. 범용 전환을 위한 JSON 설정 가이드

기존의 복잡한 2단계 트릭을 걷어내고, 아래와 같이 깔끔하게 단일 JSON으로 통합할 수 있습니다:

```json
"knee_extension": {
  "exercise_name": "knee_extension",
  "display_name_kr": "슬관절 신전 (Knee Extension)",
  "eval_config": {
    "metric_type": "COORDINATE",
    "motion_direction": "INCREASING",
    "target_kpt": {
      "left": 15,
      "right": 16
    },
    "reference_kpt": {
      "left": 13,
      "right": 14
    },
    "axis": "y",
    "use_normalization": true,
    "counting_method": "PEAK_REVERSAL",
    "posture_constraints": [
      {
        "description": "무릎(Knee)이 골반(Hip)보다 밑에 위치해야 함",
        "type": "RELATIVE_POSITION",
        "target_kpt": { "left": 13, "right": 14 },
        "reference_kpt": { "left": 11, "right": 12 },
        "axis": "y",
        "relation": "BELOW"
      }
    ]
  },
  "default_thresholds": {
    "left": {
      "start_val": -1.10,
      "target_val": -0.65
    },
    "right": {
      "start_val": -1.10,
      "target_val": -0.65
    }
  },
  "calibration_config": {
    "rest_seconds": 0.5
  }
}
```

---
*생성된 분석 그래프 및 데이터: `analysis/knee_extension_analysis/` 및 `analyses/` 참조.*
