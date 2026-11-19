"""시작값과 목표값 사이에서 현재 값이 얼마나 진행됐는지(0.0~1.0) 계산하는 모듈."""

import math

# 캘리브레이션에서 움직임 시작 / 복귀로 보는 진행률 기준
CALIBRATION_ACTIVATION_FRACTION = .30
CALIBRATION_REARM_FRACTION = .15


def measurement_progress(value, metric_type, start=None, target=None):
    """현재 측정값(value)을 start~target 구간 기준 0~1 진행률로 변환한다."""
    if any(v is None or not math.isfinite(v) for v in (value, start, target)) or start == target:
        return None
    return max(0., min(1., (value-start)/(target-start)))
