# 한국어 코드 해설: 관절을 신체 좌표로 정규화하고, 개인 가동 범위 캘리브레이션과 본 운동의 프레임 판정을 연결합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# 파일: core/session_controller.py
# 설명: 양측(Left & Right) 모두 3회 완수 기반 캘리브레이션 및 세션 제어기
# ==============================================================================

# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os
# `collections`에서 `deque` 기능을 가져옵니다.
from collections import deque
# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# current 파일 경로에 `Path(__file__).resolve()` 호출 결과를 저장합니다. 대상: `current_file_path`.
current_file_path = Path(__file__).resolve()
# `core_dir` 값에 `current_file_path.parent` 값(`current_file_path.parent`)을 저장합니다. 대상: `core_dir`.
core_dir = current_file_path.parent
# project 최상위 경로에 `core_dir.parent` 값(`core_dir.parent`)을 저장합니다. 대상: `project_root`.
project_root = core_dir.parent
# 조건 `str(project_root) not in sys.path`을 만족하는 경우에만 아래 분기를 실행합니다.
if str(project_root) not in sys.path:
    # `sys.path.insert(0, str(project_root))`을 호출하여 해당 단계의 작업을 수행합니다.
    sys.path.insert(0, str(project_root))

# `utils.normalization`에서 `PoseNormalizer` 기능을 가져옵니다.
from utils.normalization import PoseNormalizer


# `CalibrationCycleDetector` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class CalibrationCycleDetector:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """Shared calibration cycle logic for angle, coordinate, length and ratio signals."""
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, rest_seconds=0.5, min_rise=0.08, noise_multiplier=6.0,
                 # 이 줄의 입력 인수: `hold_seconds` = 움직임이 지속되어야 하는 최소 시간(초); `return_seconds` = 복귀 자세가 지속되어야 하는 시간(초); `max_cycle_seconds` = 한 동작을 기다리는 최대 시간(초). 기본값·형식은 아래 선언을 따릅니다.
                 hold_seconds=0.15, return_seconds=0.15, max_cycle_seconds=10.0,
                 # 이 줄의 입력 인수: `baseline_range` = 기본 자세에서 허용하는 측정값 흔들림 폭; `metric_type` = 측정 방식(각도·좌표·길이·비율); `direction` = 운동 방향. 기본값·형식은 아래 선언을 따릅니다.
                 baseline_range=0.04, metric_type='COORDINATE', direction='INCREASING',
                 # 이 줄의 입력 인수: `min_baseline_samples` = 기본 자세 계산에 필요한 최소 표본 수; `max_sample_gap` = 연속 표본으로 인정할 최대 시간 간격(초); `occlusion_reset_seconds` = 가림이 이어질 때 캘리브레이션 이력을 초기화할 시간(초). 기본값·형식은 아래 선언을 따릅니다.
                 min_baseline_samples=3, max_sample_gap=1.0, occlusion_reset_seconds=2.0,
                 # 이 줄의 입력 인수: `return_fraction` = 최고 변화량에 대한 복귀 허용 비율; `consistency_ratio` = 최근 세 동작의 최대·최소 변화량 허용 배수. 기본값·형식은 아래 선언을 따릅니다.
                 return_fraction=0.25, consistency_ratio=2.5):
        # 측정 방식(각도·좌표·길이·비율)에 측정 방식(각도·좌표·길이·비율)(`metric_type`)을 저장합니다. 대상: `self.metric_type`.
        self.metric_type = metric_type
        # 설정 파일에서 지정한 운동 방향에 운동 방향(`direction`)을 저장합니다. 대상: `self.configured_direction`.
        self.configured_direction = direction
        # 기본 자세를 안정적으로 유지해야 하는 시간(초)에 기본 자세를 안정적으로 유지해야 하는 시간(초)(`rest_seconds`)을 저장합니다. 대상: `self.rest_seconds`.
        self.rest_seconds = rest_seconds
        # 움직임으로 인정할 최소 측정값 변화량에 움직임으로 인정할 최소 측정값 변화량(`min_rise`)을 저장합니다. 대상: `self.min_rise`.
        self.min_rise = min_rise
        # 안정 구간 잡음에 곱하는 판정 여유 배수에 안정 구간 잡음에 곱하는 판정 여유 배수(`noise_multiplier`)을 저장합니다. 대상: `self.noise_multiplier`.
        self.noise_multiplier = noise_multiplier
        # 움직임이 지속되어야 하는 최소 시간(초)에 움직임이 지속되어야 하는 최소 시간(초)(`hold_seconds`)을 저장합니다. 대상: `self.hold_seconds`.
        self.hold_seconds = hold_seconds
        # 복귀 자세가 지속되어야 하는 시간(초)에 복귀 자세가 지속되어야 하는 시간(초)(`return_seconds`)을 저장합니다. 대상: `self.return_seconds`.
        self.return_seconds = return_seconds
        # 한 동작을 기다리는 최대 시간(초)에 한 동작을 기다리는 최대 시간(초)(`max_cycle_seconds`)을 저장합니다. 대상: `self.max_cycle_seconds`.
        self.max_cycle_seconds = max_cycle_seconds
        # 기본 자세에서 허용하는 측정값 흔들림 폭에 기본 자세에서 허용하는 측정값 흔들림 폭(`baseline_range`)을 저장합니다. 대상: `self.baseline_range`.
        self.baseline_range = baseline_range
        # 기본 자세 계산에 필요한 최소 표본 수에 기본 자세 계산에 필요한 최소 표본 수(`min_baseline_samples`)을 저장합니다. 대상: `self.min_baseline_samples`.
        self.min_baseline_samples = min_baseline_samples
        # 연속 표본으로 인정할 최대 시간 간격(초)에 연속 표본으로 인정할 최대 시간 간격(초)(`max_sample_gap`)을 저장합니다. 대상: `self.max_sample_gap`.
        self.max_sample_gap = max_sample_gap
        # 가림이 이어질 때 캘리브레이션 이력을 초기화할 시간(초)에 가림이 이어질 때 캘리브레이션 이력을 초기화할 시간(초)(`occlusion_reset_seconds`)을 저장합니다. 대상: `self.occlusion_reset_seconds`.
        self.occlusion_reset_seconds = occlusion_reset_seconds
        # 최고 변화량에 대한 복귀 허용 비율에 최고 변화량에 대한 복귀 허용 비율(`return_fraction`)을 저장합니다. 대상: `self.return_fraction`.
        self.return_fraction = return_fraction
        # 최근 세 동작의 최대·최소 변화량 허용 배수에 최근 세 동작의 최대·최소 변화량 허용 배수(`consistency_ratio`)을 저장합니다. 대상: `self.consistency_ratio`.
        self.consistency_ratio = consistency_ratio
        # 조건 `isinstance(min_baseline_samples, bool) or not isinstance(min_baseline_samples, int) or min_baseline_samples < 3`을 만족하는 경우에만 아래 분기를 실행합니다.
        if isinstance(min_baseline_samples, bool) or not isinstance(min_baseline_samples, int) or min_baseline_samples < 3:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('min_baseline_samples must be an integer >= 3')` 예외를 발생시킵니다.
            raise ValueError('min_baseline_samples must be an integer >= 3')
        # 조건 `not 0 < return_fraction < 0.5 or consistency_ratio < 1`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not (0 < return_fraction < .5) or consistency_ratio < 1:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('return_fraction must be between 0 and .5; consistency_ratio must be >= 1')` 예외를 발생시킵니다.
            raise ValueError('return_fraction must be between 0 and .5; consistency_ratio must be >= 1')
        # 조건 `direction not in {'INCREASING', 'DECREASING', 'AUTO'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        if direction not in {'INCREASING', 'DECREASING', 'AUTO'}:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Invalid calibration direction')` 예외를 발생시킵니다.
            raise ValueError('Invalid calibration direction')
        # 조건 `not all((math.isfinite(v) and v > 0 for v in (rest_seconds, min_rise, noise_multiplier, hold_seconds, return_seconds, max_cycle_seconds, baseline_range, max_sample_gap, …`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not all(math.isfinite(v) and v > 0 for v in (rest_seconds, min_rise, noise_multiplier,
                   # 앞줄에서 시작한 식·인수·자료 구조를 `hold_seconds, return_seconds, max_cycle_seconds, baseline_range,` 내용으로 이어서 구성합니다.
                   hold_seconds, return_seconds, max_cycle_seconds, baseline_range,
                   # 앞줄에서 시작한 식·인수·자료 구조를 `max_sample_gap, occlusion_reset_seconds, return_fraction, consistency_ratio)):` 내용으로 이어서 구성합니다.
                   max_sample_gap, occlusion_reset_seconds, return_fraction, consistency_ratio)):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Calibration parameters must be finite and positive')` 예외를 발생시킵니다.
            raise ValueError('Calibration parameters must be finite and positive')
        # `self.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.reset()

    # 초기화 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def reset(self):
        # 시각과 측정값의 안정 구간 표본에 양쪽 끝에서 표본을 추가·삭제할 수 있는 큐(`deque()`)을 저장합니다. 대상: `self.samples`.
        self.samples = deque()
        # 운동 방향에 설정 파일에서 지정한 운동 방향(`self.configured_direction`)을 저장합니다. 대상: `self.direction`.
        self.direction = self.configured_direction
        # 아직 확정하지 않은 운동 방향 후보에 None을 저장합니다. 대상: `self.candidate_direction`.
        self.candidate_direction = None
        # 측정값의 초당 변화량에 0.0을 저장합니다. 대상: `self.velocity`.
        self.velocity = 0.0
        # 직전 표본의 측정값에 None을 저장합니다. 대상: `self.previous_value`.
        self.previous_value = None
        # 최고점에서 복귀 방향으로 전환했는지 여부에 False을 저장합니다. 대상: `self.reversing`.
        self.reversing = False
        # 개인의 안정 자세에서 측정한 기준값에 None을 저장합니다. 대상: `self.baseline`.
        self.baseline = None
        # 안정 자세가 확정된 시각(초)에 None을 저장합니다. 대상: `self.baseline_at`.
        self.baseline_at = None
        # 잡음과 최소 변화량을 반영한 동작 시작 기준에 움직임으로 인정할 최소 측정값 변화량(`self.min_rise`)을 저장합니다. 대상: `self.threshold`.
        self.threshold = self.min_rise
        # 처리 단계에 'BASELINE'을 저장합니다. 대상: `self.state`.
        self.state = 'BASELINE'
        # 완료된 동작들의 기준 자세 대비 최대 변화량에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.peaks`.
        self.peaks = []
        # 동작 시작 시각 또는 실행 여부에 None을 저장합니다. 대상: `self.started`.
        self.started = None
        # 복귀 자세 유지가 시작된 시각에 None을 저장합니다. 대상: `self.returned`.
        self.returned = None
        # 현재 동작의 최대 변화량에 0.0을 저장합니다. 대상: `self.peak`.
        self.peak = 0.0
        # 직전 표본 시각(초)에 None을 저장합니다. 대상: `self.last_time`.
        self.last_time = None
        # 유효 관절이 사라지기 시작한 시각에 None을 저장합니다. 대상: `self.missing_since`.
        self.missing_since = None
        # 복귀 뒤 준비 상태에서 관찰한 최저 변화량에 0.0을 저장합니다. 대상: `self.ready_floor`.
        self.ready_floor = 0.0

    # 데코레이터 `property`를 바로 아래 함수·클래스에 적용합니다.
    @property
    # 준비 상태 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def ready(self):
        # 현재 함수 처리를 끝내고 계산식 `self.baseline is not None`의 결과를 호출한 쪽으로 반환합니다.
        return self.baseline is not None

    # 데코레이터 `property`를 바로 아래 함수·클래스에 적용합니다.
    @property
    # 최근 완료 동작 세 개가 존재하고 최대 변화량의 편차가 허용 배수 안인지 확인합니다.
    def consistent(self):
        # 현재 함수 처리를 끝내고 계산식 `len(self.peaks) >= 3 and max(self.peaks[-3:]) <= self.consistency_ratio * min(self.peaks[-3:])`의 결과를 호출한 쪽으로 반환합니다.
        return len(self.peaks) >= 3 and max(self.peaks[-3:]) <= self.consistency_ratio * min(self.peaks[-3:])

    # 데코레이터 `property`를 바로 아래 함수·클래스에 적용합니다.
    @property
    # 안정 구간 흔들림·시작 기준·현재 최고점에서 복귀 허용 폭을 계산하되 최고 변화량의 절반 이하로 제한합니다.
    def return_tolerance(self):
        # `tolerance` 값에 `max(self.baseline_range, self.threshold * 0.4, self.peak * self.return_fraction)` 호출 결과를 저장합니다. 대상: `tolerance`.
        tolerance = max(self.baseline_range, self.threshold * .4, self.peak * self.return_fraction)
        # 현재 함수 처리를 끝내고 `self.peak > 0`이면 `min(tolerance, self.peak * 0.5)`, 아니면 `tolerance`을 호출한 쪽으로 반환합니다.
        return min(tolerance, self.peak * .5) if self.peak > 0 else tolerance

    # 상태 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def status(self):
        # 처리 단계에 `self.missing_since is not None`이면 `'MISSING'`, 아니면 `'COMPLETE' if self.consistent else 'NEED_CONSISTENT_CYCLES' if len(self.peaks) >= 3 and self.state == 'READY' else self.state`을 저장합니다. 대상: `state`.
        state = ('MISSING' if self.missing_since is not None else
                 # 앞줄에서 시작한 식·인수·자료 구조를 `'COMPLETE' if self.consistent else` 내용으로 이어서 구성합니다.
                 'COMPLETE' if self.consistent else
                 # 앞줄에서 시작한 식·인수·자료 구조를 `'NEED_CONSISTENT_CYCLES' if len(self.peaks) >= 3 and self.state == 'READY' else self.state)` 내용으로 이어서 구성합니다.
                 'NEED_CONSISTENT_CYCLES' if len(self.peaks) >= 3 and self.state == 'READY' else self.state)
        # 현재 함수 처리를 끝내고 `dict(state=state, baseline=self.baseline, baseline_samples=len(self.samples), min_baseline_samples=self.min_baseline_samples, rest_seconds=self.rest_seconds, min_movemen…` 호출 결과를 호출한 쪽으로 반환합니다.
        return dict(state=state, baseline=self.baseline, baseline_samples=len(self.samples),
                    # 구성 항목: 기본 자세 계산에 필요한 최소 표본 수(`min_baseline_samples`)에 기본 자세 계산에 필요한 최소 표본 수(`self.min_baseline_samples`); 기본 자세를 안정적으로 유지해야 하는 시간(초)(`rest_seconds`)에 기본 자세를 안정적으로 유지해야 하는 시간(초)(`self.rest_seconds`)을 지정합니다.
                    min_baseline_samples=self.min_baseline_samples, rest_seconds=self.rest_seconds,
                    # 구성 항목: `min_movement` 값(`min_movement`)에 잡음과 최소 변화량을 반영한 동작 시작 기준(`self.threshold`); `return_tolerance` 값(`return_tolerance`)에 `self.return_tolerance` 값(`self.return_tolerance`)을 지정합니다.
                    min_movement=self.threshold, return_tolerance=self.return_tolerance,
                    # 구성 항목: `completed_cycles` 값(`completed_cycles`)에 `len(self.peaks)` 호출 결과; `consistent` 값(`consistent`)에 `self.consistent` 값(`self.consistent`)을 지정합니다.
                    completed_cycles=len(self.peaks), consistent=self.consistent,
                    # 구성 항목: 운동 방향(`direction`)에 운동 방향(`self.direction`)을 지정합니다.
                    direction=self.direction)

    # 측정값 하나를 받아 안정 자세→상승 유지→복귀 유지를 진행합니다. 결측·시간 단절·미완료 동작은 횟수로 인정하지 않습니다.
    def update(self, value, now, allow_count=True):
        # 이번 처리에서 완료 횟수가 바뀌었는지 여부에 False을 저장합니다. 대상: `updated`.
        updated = False
        # 조건 `value is None or not math.isfinite(value)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if value is None or not math.isfinite(value):
            # 유효 관절이 사라지기 시작한 시각에 `self.missing_since is not None`이면 `self.missing_since`, 아니면 `now`을 저장합니다. 대상: `missing_since`.
            missing_since = self.missing_since if self.missing_since is not None else now
            # 조건 `now - missing_since >= self.occlusion_reset_seconds`을 만족하는 경우에만 아래 분기를 실행합니다.
            if now - missing_since >= self.occlusion_reset_seconds:
                # `self.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
                self.reset()
            # 유효 관절이 사라지기 시작한 시각에 유효 관절이 사라지기 시작한 시각(`missing_since`)을 저장합니다. 대상: `self.missing_since`.
            self.missing_since = missing_since
            # 결측 구간은 상승·복귀 유지 시간에 포함하지 않으며 안정 자세 구간을 이어 붙이지 않습니다.
            # `self.samples.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
            self.samples.clear()
            # 처리 단계에 `self.ready`이면 `'READY'`, 아니면 `'BASELINE'`을 저장합니다. 대상: `self.state`.
            self.state = 'READY' if self.ready else 'BASELINE'
            # 동작 시작 시각 또는 실행 여부에 None을 저장합니다. 대상: `self.started, self.returned`.
            self.started = self.returned = None
            # 직전 표본의 측정값에 None을 저장합니다. 대상: `self.previous_value`.
            self.previous_value = None
            # 직전 표본 시각(초)에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self.last_time`.
            self.last_time = now
            # 현재 함수 처리를 끝내고 이번 처리에서 완료 횟수가 바뀌었는지 여부(`updated`)을 호출한 쪽으로 반환합니다.
            return updated
        # 조건 `self.missing_since is not None and now - self.missing_since >= self.occlusion_reset_seconds`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.missing_since is not None and now - self.missing_since >= self.occlusion_reset_seconds:
            # `self.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.reset()
        # 유효 관절이 사라지기 시작한 시각에 None을 저장합니다. 대상: `self.missing_since`.
        self.missing_since = None
        # 조건 `self.last_time is not None and (now <= self.last_time or now - self.last_time > self.occlusion_reset_seconds)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > self.occlusion_reset_seconds):
            # `self.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.reset()
        # 조건 `self.last_time is not None and now - self.last_time > self.max_sample_gap`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.last_time is not None and now - self.last_time > self.max_sample_gap:
            # `self.samples.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
            self.samples.clear()
            # 처리 단계에 `self.ready`이면 `'READY'`, 아니면 `'BASELINE'`을 저장합니다. 대상: `self.state`.
            self.state = 'READY' if self.ready else 'BASELINE'
            # 동작 시작 시각 또는 실행 여부에 None을 저장합니다. 대상: `self.started, self.returned`.
            self.started = self.returned = None
        # 직전 시각과 현재 시각 사이의 간격(초)에 `self.last_time is not None`이면 `now - self.last_time`, 아니면 `0.0`을 저장합니다. 대상: `dt`.
        dt = now - self.last_time if self.last_time is not None else 0.
        # 측정값의 초당 변화량에 `self.previous_value is not None and 0 < dt <= self.max_sample_gap`이면 `(value - self.previous_value) / dt`, 아니면 `0.0`을 저장합니다. 대상: `self.velocity`.
        self.velocity = ((value - self.previous_value) / dt
                         # 앞줄에서 시작한 식·인수·자료 구조를 `if self.previous_value is not None and 0 < dt <= self.max_sample_gap else 0.)` 내용으로 이어서 구성합니다.
                         if self.previous_value is not None and 0 < dt <= self.max_sample_gap else 0.)
        # 직전 표본의 측정값에 측정값(`value`)을 저장합니다. 대상: `self.previous_value`.
        self.previous_value = value
        # 직전 표본 시각(초)에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self.last_time`.
        self.last_time = now
        # 조건 `not self.ready`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.ready:
            # `self.samples.append((now, value))`을 호출하여 항목 하나를 끝에 추가합니다.
            self.samples.append((now, value))
            # 낮은 FPS에서도 최소 표본 수를 유지하면서 안정 구간 길이는 초 단위로 판단합니다.
            # 조건 `len(self.samples) > self.min_baseline_samples and now - self.samples[1][0] >= self.rest_seconds`이 유지되는 동안 아래 처리를 반복합니다.
            while len(self.samples) > self.min_baseline_samples and now - self.samples[1][0] >= self.rest_seconds:
                # `self.samples.popleft()`을 호출하여 가장 오래된 항목을 제거합니다.
                self.samples.popleft()
            # 조건 `len(self.samples) >= self.min_baseline_samples and now - self.samples[0][0] >= self.rest_seconds`을 만족하는 경우에만 아래 분기를 실행합니다.
            if len(self.samples) >= self.min_baseline_samples and now - self.samples[0][0] >= self.rest_seconds:
                # 측정값에 반복·조건식 `[v for (_, v) in self.samples]`으로 구성한 값을 저장합니다. 대상: `values`.
                values = [v for _, v in self.samples]
                # 중앙값 또는 중심 좌표에 `float(np.median(values))` 호출 결과를 저장합니다. 대상: `center`.
                center = float(np.median(values))
                # 중앙절대편차(MAD)에 1.4826을 곱해 안정 구간 잡음을 추정합니다. 일부 튀는 표본에 평균·표준편차보다 덜 끌려갑니다.
                noise = 1.4826 * float(np.median(np.abs(np.asarray(values) - center)))
                # 조건 `np.percentile(values, 95) - np.percentile(values, 5) <= self.baseline_range`을 만족하는 경우에만 아래 분기를 실행합니다.
                if np.percentile(values, 95) - np.percentile(values, 5) <= self.baseline_range:
                    # 개인의 안정 자세에서 측정한 기준값에 중앙값 또는 중심 좌표(`center`)을 저장합니다. 대상: `self.baseline`.
                    self.baseline = center
                    # 안정 자세가 확정된 시각(초)에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self.baseline_at`.
                    self.baseline_at = now
                    # 최소 변화량과 잡음의 지정 배수 중 큰 값을 동작 시작 기준으로 채택합니다. 환자의 실제 움직임과 검출 흔들림을 구분하기 위한 기준입니다.
                    self.threshold = max(self.min_rise, self.noise_multiplier * noise)
                    # 처리 단계에 'READY'을 저장합니다. 대상: `self.state`.
                    self.state = 'READY'
            # 현재 함수 처리를 끝내고 이번 처리에서 완료 횟수가 바뀌었는지 여부(`updated`)을 호출한 쪽으로 반환합니다.
            return updated
        # 조건 `not allow_count`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not allow_count:
            # 현재 함수 처리를 끝내고 이번 처리에서 완료 횟수가 바뀌었는지 여부(`updated`)을 호출한 쪽으로 반환합니다.
            return updated
        # 기본 자세 대비 부호를 포함한 측정값 차이에 계산식 `value - self.baseline`의 결과를 저장합니다. 대상: `displacement`.
        displacement = value - self.baseline
        # 조건 `self.direction == 'AUTO' and self.state == 'READY' and (abs(displacement) >= self.threshold)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.direction == 'AUTO' and self.state == 'READY' and abs(displacement) >= self.threshold:
            # 아직 확정하지 않은 운동 방향 후보에 `displacement > 0`이면 `'INCREASING'`, 아니면 `'DECREASING'`을 저장합니다. 대상: `self.candidate_direction`.
            self.candidate_direction = 'INCREASING' if displacement > 0 else 'DECREASING'
        # 운동 방향에 `self.direction == 'AUTO'`이면 `self.candidate_direction`, 아니면 `self.direction`을 저장합니다. 대상: `direction`.
        direction = self.candidate_direction if self.direction == 'AUTO' else self.direction
        # 증가·감소를 같은 상승 신호로 바꾸는 부호에 `direction == 'DECREASING'`이면 `-1.0`, 아니면 `1.0`을 저장합니다. 대상: `sign`.
        sign = -1. if direction == 'DECREASING' else 1.
        # 운동 방향을 보정한 기준 자세 대비 변화량에 계산식 `sign * displacement`의 결과를 저장합니다. 대상: `rise`.
        rise = sign * displacement
        # 조건 `self.state == 'READY'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.state == 'READY':
            # 복귀 후의 최저 변화량을 추적합니다. 완전히 같은 시작값으로 돌아오지 않아도 새 상승이 있어야 다음 동작이 시작되도록 합니다.
            self.ready_floor = min(self.ready_floor, rise)
            # 개인 기준 자세와 직전 복귀 바닥값 모두에서 충분히 상승했는지 확인합니다. 복귀 허용 영역에서 머무르는 것을 새 동작으로 세지 않습니다.
            if rise >= max(self.threshold, self.ready_floor + self.threshold):
                # 최고점에서 복귀 방향으로 전환했는지 여부에 False을 저장합니다. 대상: `self.reversing`.
                self.reversing = False
                # 처리 단계에 'RISING'을 저장합니다. 대상: `self.state`.
                self.state = 'RISING'
                # 동작 시작 시각 또는 실행 여부에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self.started`.
                self.started = now
                # 현재 동작의 최대 변화량에 운동 방향을 보정한 기준 자세 대비 변화량(`rise`)을 저장합니다. 대상: `self.peak`.
                self.peak = rise
        # 조건 `self.state == 'RISING'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif self.state == 'RISING':
            # 조건 `rise < self.threshold`을 만족하는 경우에만 아래 분기를 실행합니다.
            if rise < self.threshold:
                # 처리 단계에 'READY'을 저장합니다. 대상: `self.state`.
                self.state = 'READY'
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # 현재 동작의 최대 변화량에 `max(self.peak, rise)` 호출 결과를 저장합니다. 대상: `self.peak`.
                self.peak = max(self.peak, rise)
                # 조건 `now - self.started >= self.hold_seconds`을 만족하는 경우에만 아래 분기를 실행합니다.
                if now - self.started >= self.hold_seconds:
                    # 처리 단계에 'RETURNING'을 저장합니다. 대상: `self.state`.
                    self.state = 'RETURNING'
        # 조건 `self.state == 'RETURNING'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif self.state == 'RETURNING':
            # 현재 동작의 최대 변화량에 `max(self.peak, rise)` 호출 결과를 저장합니다. 대상: `self.peak`.
            self.peak = max(self.peak, rise)
            # 시작 기준의 30% 또는 최고 변화량의 10% 이상 감소했을 때 실제 복귀 방향 전환으로 봅니다. 최고점 부근의 작은 떨림은 제외합니다.
            if self.peak - rise >= max(self.threshold * .3, self.peak * .1):
                # 최고점에서 복귀 방향으로 전환했는지 여부에 True을 저장합니다. 대상: `self.reversing`.
                self.reversing = True
            # 조건 `self.reversing and rise <= self.return_tolerance`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.reversing and rise <= self.return_tolerance:
                # 조건 `self.returned is None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self.returned is None:
                    # 복귀 자세 유지가 시작된 시각에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self.returned`.
                    self.returned = now
                # 조건 `now - self.returned >= self.return_seconds`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif now - self.returned >= self.return_seconds:
                    # 최소 상승 유지와 복귀 유지 시간을 모두 통과한 한 동작의 최고 변화량만 완료 이력에 추가합니다.
                    self.peaks.append(self.peak)
                    # 조건 `self.direction == 'AUTO'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if self.direction == 'AUTO':
                        # 운동 방향에 아직 확정하지 않은 운동 방향 후보(`self.candidate_direction`)을 저장합니다. 대상: `self.direction`.
                        self.direction = self.candidate_direction
                    # 처리 단계에 'READY'을 저장합니다. 대상: `self.state`.
                    self.state = 'READY'
                    # 복귀 뒤 준비 상태에서 관찰한 최저 변화량에 운동 방향을 보정한 기준 자세 대비 변화량(`rise`)을 저장합니다. 대상: `self.ready_floor`.
                    self.ready_floor = rise
                    # 복귀 자세 유지가 시작된 시각에 None을 저장합니다. 대상: `self.returned`.
                    self.returned = None
                    # 이번 처리에서 완료 횟수가 바뀌었는지 여부에 True을 저장합니다. 대상: `updated`.
                    updated = True
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # 복귀 자세 유지가 시작된 시각에 None을 저장합니다. 대상: `self.returned`.
                self.returned = None
        # 조건 `self.state != 'READY' and self.started is not None and (now - self.started > self.max_cycle_seconds)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.state != 'READY' and self.started is not None and now - self.started > self.max_cycle_seconds:
            # 처리 단계에 'READY'을 저장합니다. 대상: `self.state`.
            self.state = 'READY'
            # 복귀 자세 유지가 시작된 시각에 None을 저장합니다. 대상: `self.returned`.
            self.returned = None
        # 현재 함수 처리를 끝내고 이번 처리에서 완료 횟수가 바뀌었는지 여부(`updated`)을 호출한 쪽으로 반환합니다.
        return updated

    # 최근 세 동작이 일관될 때 시작값·85% 목표값·측정 방식·방향을 개인 기준값으로 반환합니다.
    def thresholds(self):
        # 조건 `not self.consistent`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.consistent:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Calibration needs three consistent movement/return cycles')` 예외를 발생시킵니다.
            raise ValueError('Calibration needs three consistent movement/return cycles')
        # 현재 함수 처리를 끝내고 키 'start_val', 'target_val', 'metric_type', 'motion_direction'로 구성한 사전을 호출한 쪽으로 반환합니다.
        return {'start_val': round(self.baseline, 4),
                # 구성 항목: 개인 가동 범위의 목표값(`'target_val'`)에 `round(self.baseline + (-1.0 if self.direction == 'DECREASING' else 1.0) * 0.85 * float(np.median(self.peaks[-3:])), 4)` 호출 결과를 지정합니다.
                'target_val': round(self.baseline + (-1. if self.direction == 'DECREASING' else 1.) * .85 * float(np.median(self.peaks[-3:])), 4),
                # 구성 항목: 측정 방식(각도·좌표·길이·비율)(`'metric_type'`)에 측정 방식(각도·좌표·길이·비율)(`self.metric_type`); 측정값 증가·감소 방향(`'motion_direction'`)에 운동 방향(`self.direction`)을 지정합니다.
                'metric_type': self.metric_type, 'motion_direction': self.direction}


# `SessionController` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class SessionController:

    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, data_manager=None, motion_engine=None):
        # 환자 정보 및 세션 로그 관리자에 환자 정보 및 세션 로그 관리자(`data_manager`)을 저장합니다. 대상: `self.data_manager`.
        self.data_manager = data_manager
        # 관절 측정값 계산 및 반복 판정 엔진에 관절 측정값 계산 및 반복 판정 엔진(`motion_engine`)을 저장합니다. 대상: `self.motion_engine`.
        self.motion_engine = motion_engine
        # 좌우 각각 필요한 캘리브레이션 횟수에 3을 저장합니다. 대상: `self.target_calib_reps`.
        self.target_calib_reps = 3  # 양측 각각 채워야 하는 목표 횟수

        # 실행 모드에 `motion_engine`이면 `motion_engine.mode`, 아니면 `'CALIBRATION'`을 저장합니다. 대상: `self.mode`.
        self.mode = motion_engine.mode if motion_engine else "CALIBRATION"
        # 캘리브레이션 전체 진행 단계에 'FULL_BODY_CHECK'을 저장합니다. 대상: `self.calib_step`.
        self.calib_step = "FULL_BODY_CHECK"
        # 단계 대기 시간이 시작된 시각에 0.0을 저장합니다. 대상: `self.timer_start`.
        self.timer_start = 0.0

        # 골반 중심·몸통 길이 기준 좌표 변환기에 `PoseNormalizer(conf_threshold=0.35, invert_y=True)` 호출 결과를 저장합니다. 대상: `self.normalizer`.
        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)
        # 좌우별 캘리브레이션 동작 판정기에 빈 사전을 저장합니다. 대상: `self.calibration_detectors`.
        self.calibration_detectors = {}
        # 조건 `motion_engine`을 만족하는 경우에만 아래 분기를 실행합니다.
        if motion_engine:
            # 기본값에 `motion_engine.metric_type == 'ANGLE'`이면 `{'min_rise': 8.0, 'baseline_range': 4.0}`, 아니면 `{}`을 저장합니다. 대상: `defaults`.
            defaults = ({'min_rise': 8.0, 'baseline_range': 4.0}
                        # 앞줄에서 시작한 식·인수·자료 구조를 `if motion_engine.metric_type == 'ANGLE' else {})` 내용으로 이어서 구성합니다.
                        if motion_engine.metric_type == 'ANGLE' else {})
            # `params` 값에 빈 사전을 저장합니다. 대상: `params`.
            params = {**defaults, **motion_engine.config.get('calibration_config', {})}
            # 좌우별 캘리브레이션 동작 판정기에 반복·조건식 `{side: CalibrationCycleDetector(**params, metric_type=motion_engine.metric_type, direction=motion_engine.motion_direction) for side in ('left', 'right')}`으로 구성한 값을 저장합니다. 대상: `self.calibration_detectors`.
            self.calibration_detectors = {side: CalibrationCycleDetector(
                # 구성 항목: 키워드 인수 펼침(`키워드 인수 펼침`)에 `params` 값(`params`); 측정 방식(각도·좌표·길이·비율)(`metric_type`)에 측정 방식(각도·좌표·길이·비율)(`motion_engine.metric_type`)을 지정합니다.
                **params, metric_type=motion_engine.metric_type,
                # 구성 항목: 운동 방향(`direction`)에 측정값 증가·감소 방향(`motion_engine.motion_direction`)을 지정합니다.
                direction=motion_engine.motion_direction) for side in ('left', 'right')}


        # calib vals 왼쪽에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.calib_vals_left`.
        self.calib_vals_left = []
        # calib vals 오른쪽에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.calib_vals_right`.
        self.calib_vals_right = []
        # calib 반복 횟수 왼쪽에 0을 저장합니다. 대상: `self.calib_reps_left`.
        self.calib_reps_left = 0
        # calib 반복 횟수 오른쪽에 0을 저장합니다. 대상: `self.calib_reps_right`.
        self.calib_reps_right = 0

        # TEST 모드 목표 달성 후 쿨다운 타이머
        # 회귀 검증 시나리오 test_finish_timer에 None을 저장합니다. 대상: `self.test_finish_timer`.
        self.test_finish_timer = None

        # last 왼쪽 데이터에 키 'val', 'rep_count', 'total_score', 'score_delta', 'score', 'progress_ratio', 'state', 'quality'로 구성한 사전을 저장합니다. 대상: `self.last_left_data`.
        self.last_left_data = {
            # 구성 항목: `val` 값(`'val'`)에 0.0을 지정합니다.
            "val": 0.0,
            # 구성 항목: 인정된 누적 반복 횟수(`'rep_count'`)에 0을 지정합니다.
            "rep_count": 0,
            # 구성 항목: 서버 기본 가중치로 계산한 누적 참고 점수(`'total_score'`)에 0.0; 이번 반복에서 추가된 참고 점수(`'score_delta'`)에 0.0; 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 0을 지정합니다.
            "total_score": 0.0, "score_delta": 0.0, "score": 0,
            # 구성 항목: 설정 가동 범위 대비 진행 비율(0~1)(`'progress_ratio'`)에 0.0을 지정합니다.
            "progress_ratio": 0.0,
            # 구성 항목: 처리 단계(`'state'`)에 'READY'을 지정합니다.
            "state": "READY",
            # 구성 항목: 동작 품질 등급(`'quality'`)에 'CALIBRATING'을 지정합니다.
            "quality": "CALIBRATING",
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
        # last 오른쪽 데이터에 키 'val', 'rep_count', 'total_score', 'score_delta', 'score', 'progress_ratio', 'state', 'quality'로 구성한 사전을 저장합니다. 대상: `self.last_right_data`.
        self.last_right_data = {
            # 구성 항목: `val` 값(`'val'`)에 0.0을 지정합니다.
            "val": 0.0,
            # 구성 항목: 인정된 누적 반복 횟수(`'rep_count'`)에 0을 지정합니다.
            "rep_count": 0,
            # 구성 항목: 서버 기본 가중치로 계산한 누적 참고 점수(`'total_score'`)에 0.0; 이번 반복에서 추가된 참고 점수(`'score_delta'`)에 0.0; 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 0을 지정합니다.
            "total_score": 0.0, "score_delta": 0.0, "score": 0,
            # 구성 항목: 설정 가동 범위 대비 진행 비율(0~1)(`'progress_ratio'`)에 0.0을 지정합니다.
            "progress_ratio": 0.0,
            # 구성 항목: 처리 단계(`'state'`)에 'READY'을 지정합니다.
            "state": "READY",
            # 구성 항목: 동작 품질 등급(`'quality'`)에 'CALIBRATING'을 지정합니다.
            "quality": "CALIBRATING",
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[DEBUG][CONTROLLER] Initialized successfully. Target Calib Reps per side: {self.target_calib_reps}")

    # 부분 가동 범위 변경 요청을 검증하고 판정 기준을 갱신합니다. 진행 중이던 동작은 새 범위에서 복귀 후 다시 시작합니다.
    def update_count_thresholds(self, thresholds: dict) -> list:
        # 이번 명령으로 가동 범위가 바뀐 좌우 목록에 `self.motion_engine.update_count_thresholds(thresholds)` 호출 결과를 저장합니다. 대상: `changed_sides`.
        changed_sides = self.motion_engine.update_count_thresholds(thresholds)
        # `changed_sides`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
        for side in changed_sides:
            # 한쪽 동작의 반복 판정 상태 기계에 `side == 'left'`이면 `self.motion_engine.fsm_left`, 아니면 `self.motion_engine.fsm_right`을 저장합니다. 대상: `fsm`.
            fsm = self.motion_engine.fsm_left if side == "left" else self.motion_engine.fsm_right
            # 다음 화면 갱신이나 판정에 재사용할 최신 상태에 `side == 'left'`이면 `self.last_left_data`, 아니면 `self.last_right_data`을 저장합니다. 대상: `cache`.
            cache = self.last_left_data if side == "left" else self.last_right_data
            # 측정값에 `cache.get('val')` 호출 결과를 저장합니다. 대상: `value`.
            value = cache.get("val")
            # 비율에 `value is not None`이면 `abs(value - fsm.start_val) / abs(fsm.target_val - fsm.start_val)`, 아니면 `0.0`을 저장합니다. 대상: `ratio`.
            ratio = abs(value - fsm.start_val) / abs(fsm.target_val - fsm.start_val) if value is not None else 0.0
            # `cache.update(state=fsm.state, rep_count=fsm.rep_count, **fsm.score_status(), progress_ratio=round(min(1.0, max(0.0, ratio)), 2), quality=fsm.last_quality, is_count_updat…`을 호출하여 해당 단계의 작업을 수행합니다.
            cache.update(state=fsm.state, rep_count=fsm.rep_count,
                         # 구성 항목: 키워드 인수 펼침(`키워드 인수 펼침`)에 `fsm.score_status()` 호출 결과를 지정합니다.
                         **fsm.score_status(),
                         # 구성 항목: 설정 가동 범위 대비 진행 비율(0~1)(`progress_ratio`)에 `round(min(1.0, max(0.0, ratio)), 2)` 호출 결과를 지정합니다.
                         progress_ratio=round(min(1.0, max(0.0, ratio)), 2),
                         # 구성 항목: 동작 품질 등급(`quality`)에 마지막으로 확정한 동작 품질(`fsm.last_quality`); 이번 프레임에서 반복 횟수가 증가했는지 여부(`is_count_updated`)에 False을 지정합니다.
                         quality=fsm.last_quality, is_count_updated=False)
        # 현재 함수 처리를 끝내고 이번 명령으로 가동 범위가 바뀐 좌우 목록(`changed_sides`)을 호출한 쪽으로 반환합니다.
        return changed_sides

    # 몸통 기준 관절과 해당 운동에 필요한 관절이 모두 최소 신뢰도를 만족하는지 검사합니다.
    def check_occlusion(self, keypoints: list) -> bool:
        # 조건 `not keypoints or len(keypoints) < 17`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not keypoints or len(keypoints) < 17:
            # 현재 함수 처리를 끝내고 True을 호출한 쪽으로 반환합니다.
            return True
        # 관절 ID로 조회하는 관절 또는 신뢰도 사전에 반복·조건식 `{kp['id']: kp.get('score', 0.0) for kp in keypoints}`으로 구성한 값을 저장합니다. 대상: `kpt_map`.
        kpt_map = {kp["id"]: kp.get("score", 0.0) for kp in keypoints}
        # 현재 운동 판정과 신체 정규화에 필요한 관절 ID 집합에 순서·구성을 지정한 `{5, 6, 11, 12}`을 저장합니다. 대상: `required_ids`.
        required_ids = {5, 6, 11, 12}
        # `cfg` 값에 `self.motion_engine`이면 `self.motion_engine.config.get('eval_config', {})`, 아니면 `{}`을 저장합니다. 대상: `cfg`.
        cfg = self.motion_engine.config.get('eval_config', {}) if self.motion_engine else {}
        # `('left', 'right')`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
        for side in ('left', 'right'):
            # `required_ids.update(cfg.get('primary_kpts', {}).get(side, []))`을 호출하여 해당 단계의 작업을 수행합니다.
            required_ids.update(cfg.get('primary_kpts', {}).get(side, []))
            # `('numerator_kpts', 'denominator_kpts')`에서 항목을 차례로 꺼내 `field`에 넣고 아래 처리를 반복합니다.
            for field in ('numerator_kpts', 'denominator_kpts'):
                # `required_ids.update(cfg.get(field, {}).get(side, []))`을 호출하여 해당 단계의 작업을 수행합니다.
                required_ids.update(cfg.get(field, {}).get(side, []))
            # `('target_kpt', 'reference_kpt')`에서 항목을 차례로 꺼내 `field`에 넣고 아래 처리를 반복합니다.
            for field in ('target_kpt', 'reference_kpt'):
                # 측정값에 `cfg.get(field, {}).get(side)` 호출 결과를 저장합니다. 대상: `value`.
                value = cfg.get(field, {}).get(side)
                # 조건 `value is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if value is not None:
                    # `required_ids.add(value)`을 호출하여 집합에 항목을 등록합니다.
                    required_ids.add(value)
        # 현재 함수 처리를 끝내고 계산식 `not all((kpt_map.get(i, 0.0) >= 0.35 for i in required_ids))`의 결과를 호출한 쪽으로 반환합니다.
        return not all(kpt_map.get(i, 0.0) >= 0.35 for i in required_ids)

    # 양쪽 관절 측정값으로 안정 자세·상승·복귀를 추적하고 최근 세 동작의 일관성과 완료 상태를 갱신합니다.
    def process_calibration_frame(self, keypoints, is_occluded, now=None):
        # 이번 처리에 사용할 현재 시각(초)에 `now is None`이면 `time.time()`, 아니면 `now`을 저장합니다. 대상: `now`.
        now = time.time() if now is None else now
        # 좌우 캘리브레이션 판정기 모음에 좌우별 캘리브레이션 동작 판정기(`self.calibration_detectors`)을 저장합니다. 대상: `detectors`.
        detectors = self.calibration_detectors
        # 신체 기준으로 정규화한 관절 목록에 `keypoints and (not is_occluded)`이면 `self.normalizer.normalize(keypoints)`, 아니면 `None`을 저장합니다. 대상: `normalized`.
        normalized = self.normalizer.normalize(keypoints) if keypoints and not is_occluded else None
        # 필수 관절이 가려졌거나 유효하지 않은지 여부에 계산식 `is_occluded or not normalized`의 결과를 저장합니다. 대상: `is_occluded`.
        is_occluded = is_occluded or not normalized
        # 계산에 사용할 관절 또는 좌표 묶음에 `normalized`이면 `{p['id']: p for p in normalized}`, 아니면 `{}`을 저장합니다. 대상: `points`.
        points = {p['id']: p for p in normalized} if normalized else {}
        # 양쪽 모두 안정 자세를 확보하고 필요한 관절이 보일 때부터 횟수를 인정합니다. 한쪽 준비 중 다른 쪽에서 먼저 잘못 세는 일을 막습니다.
        allow_count = all(d.ready for d in detectors.values()) and not is_occluded
        # `detectors.items()`에서 항목을 차례로 꺼내 `(side, detector)`에 넣고 아래 처리를 반복합니다.
        for side, detector in detectors.items():
            # 측정값에 `self.motion_engine._compute_side_value(side, points)` 호출 결과를 저장합니다. 대상: `value`.
            value = self.motion_engine._compute_side_value(side, points)
            # 이번 처리에서 완료 횟수가 바뀌었는지 여부에 `detector.update(value, now, allow_count=allow_count and (not detector.consistent))` 호출 결과를 저장합니다. 대상: `updated`.
            updated = detector.update(value, now, allow_count=allow_count and not detector.consistent)
            # 횟수에 `len(detector.peaks)` 호출 결과를 저장합니다. 대상: `count`.
            count = len(detector.peaks)
            # 운동 방향에 `detector.direction == 'AUTO'`이면 `detector.candidate_direction`, 아니면 `detector.direction`을 저장합니다. 대상: `direction`.
            direction = detector.candidate_direction if detector.direction == 'AUTO' else detector.direction
            # 증가·감소를 같은 상승 신호로 바꾸는 부호에 `direction == 'DECREASING'`이면 `-1.0`, 아니면 `1.0`을 저장합니다. 대상: `sign`.
            sign = -1. if direction == 'DECREASING' else 1.
            # 다음 화면 갱신이나 판정에 재사용할 최신 상태에 `dict(val=value, rep_count=count, total_score=0.0, score_delta=0.0, score=0, calibration=detector.status(), state=detector.state, quality='CALIBRATING', is_count_updated=…` 호출 결과를 저장합니다. 대상: `cache`.
            cache = dict(val=value, rep_count=count, total_score=0.0, score_delta=0.0, score=0,
                         # 구성 항목: 캘리브레이션(`calibration`)에 `detector.status()` 호출 결과를 지정합니다.
                         calibration=detector.status(),
                         # 구성 항목: 처리 단계(`state`)에 처리 단계(`detector.state`); 동작 품질 등급(`quality`)에 'CALIBRATING'; 이번 프레임에서 반복 횟수가 증가했는지 여부(`is_count_updated`)에 이번 처리에서 완료 횟수가 바뀌었는지 여부(`updated`)을 지정합니다.
                         state=detector.state, quality='CALIBRATING', is_count_updated=updated,
                         # 구성 항목: 설정 가동 범위 대비 진행 비율(0~1)(`progress_ratio`)에 `value is not None and detector.ready`이면 `round(max(0.0, min(1.0, (value - detector.baseline) * sign / detector.threshold)), 2)`, 아니면 `0.0`을 지정합니다.
                         progress_ratio=round(max(0., min(1., (value - detector.baseline) * sign / detector.threshold)), 2)
                         # 앞줄에서 시작한 식·인수·자료 구조를 `if value is not None and detector.ready else 0.)` 내용으로 이어서 구성합니다.
                         if value is not None and detector.ready else 0.)
            # `setattr(self, f'last_{side}_data', cache)`을 호출하여 해당 단계의 작업을 수행합니다.
            setattr(self, f'last_{side}_data', cache)
            # `setattr(self, f'calib_reps_{side}', count)`을 호출하여 해당 단계의 작업을 수행합니다.
            setattr(self, f'calib_reps_{side}', count)
            # 한쪽 동작의 반복 판정 상태 기계에 `getattr(self.motion_engine, f'fsm_{side}')` 호출 결과를 저장합니다. 대상: `fsm`.
            fsm = getattr(self.motion_engine, f'fsm_{side}')
            # 인정된 누적 반복 횟수에 횟수(`count`)을 저장합니다. 대상: `fsm.rep_count`.
            fsm.rep_count = count
            # 처리 단계에 처리 단계(`detector.state`)을 저장합니다. 대상: `fsm.state`.
            fsm.state = detector.state
            # 조건 `detector.ready`을 만족하는 경우에만 아래 분기를 실행합니다.
            if detector.ready:
                # 개인 가동 범위의 시작값에 개인의 안정 자세에서 측정한 기준값(`detector.baseline`)을 저장합니다. 대상: `fsm.start_val`.
                fsm.start_val = detector.baseline
                # 운동할수록 측정값이 감소하는지 여부에 계산식 `direction == 'DECREASING'`의 결과를 저장합니다. 대상: `fsm.is_decreasing`.
                fsm.is_decreasing = direction == 'DECREASING'
                # 개인 가동 범위의 목표값에 계산식 `detector.baseline + (-1.0 if fsm.is_decreasing else 1.0) * detector.threshold`의 결과를 저장합니다. 대상: `fsm.target_val`.
                fsm.target_val = detector.baseline + (-1. if fsm.is_decreasing else 1.) * detector.threshold
            # 조건 `updated`을 만족하는 경우에만 아래 분기를 실행합니다.
            if updated:
                # `endpoint` 값에 계산식 `detector.baseline + (-1.0 if fsm.is_decreasing else 1.0) * detector.peaks[-1]`의 결과를 저장합니다. 대상: `endpoint`.
                endpoint = detector.baseline + (-1. if fsm.is_decreasing else 1.) * detector.peaks[-1]
                # `fsm.completed_reps_history.append(dict(side=side, rep_num=count, duration_sec=now - detector.started, min_val=min(detector.baseline, endpoint), max_val=max(detector.base…`을 호출하여 항목 하나를 끝에 추가합니다.
                fsm.completed_reps_history.append(dict(side=side, rep_num=count, duration_sec=now-detector.started,
                    # 구성 항목: `min_val` 값(`min_val`)에 `min(detector.baseline, endpoint)` 호출 결과; `max_val` 값(`max_val`)에 `max(detector.baseline, endpoint)` 호출 결과를 지정합니다.
                    min_val=min(detector.baseline, endpoint), max_val=max(detector.baseline, endpoint),
                    # 구성 항목: achieved 가동 범위(`achieved_rom`)에 계산식 `detector.peaks[-1]`의 결과; 동작 품질 등급(`quality`)에 'CALIBRATING'; 이번 반복에서 추가된 참고 점수(`score_delta`)에 0.0; 서버 기본 가중치로 계산한 누적 참고 점수(`total_score`)에 0.0을 지정합니다.
                    achieved_rom=detector.peaks[-1], quality='CALIBRATING', score_delta=0., total_score=0.,
                    # 구성 항목: 개인 가동 범위의 시작값(`start_val`)에 개인 가동 범위의 시작값(`fsm.start_val`); 개인 가동 범위의 목표값(`target_val`)에 개인 가동 범위의 목표값(`fsm.target_val`); 현재 가동 범위 설정의 변경 번호(`threshold_version`)에 현재 가동 범위 설정의 변경 번호(`fsm.threshold_version`)을 지정합니다.
                    start_val=fsm.start_val, target_val=fsm.target_val, threshold_version=fsm.threshold_version))
            # 조건 `len(fsm.completed_reps_history) > count`을 만족하는 경우에만 아래 분기를 실행합니다.
            if len(fsm.completed_reps_history) > count:
                # `fsm.completed_reps_history.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
                fsm.completed_reps_history.clear()
        # 조건 `is_occluded`을 만족하는 경우에만 아래 분기를 실행합니다.
        if is_occluded:
            # 캘리브레이션 전체 진행 단계에 `any((d.ready for d in detectors.values()))`이면 `'PAUSED'`, 아니면 `'FULL_BODY_CHECK'`을 저장합니다. 대상: `self.calib_step`.
            self.calib_step = 'PAUSED' if any(d.ready for d in detectors.values()) else 'FULL_BODY_CHECK'
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # 조건 `not all((d.ready for d in detectors.values()))`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not all(d.ready for d in detectors.values()):
                # 캘리브레이션 전체 진행 단계에 'COUNTDOWN'을 저장합니다. 대상: `self.calib_step`.
                self.calib_step = 'COUNTDOWN'
            # 조건 `all((d.consistent for d in detectors.values())) and all((cache['val'] is not None for cache in (self.last_left_data, self.last_right_data)))`을 만족하는 경우에만 아래 분기를 실행합니다.
            elif all(d.consistent for d in detectors.values()) and all(
                    # 앞줄에서 시작한 식·인수·자료 구조를 `cache['val'] is not None for cache in (self.last_left_data, self.last_right_data)):` 내용으로 이어서 구성합니다.
                    cache['val'] is not None for cache in (self.last_left_data, self.last_right_data)):
                # 조건 `self.calib_step not in ('COOLDOWN', 'FINISHED')`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self.calib_step not in ('COOLDOWN', 'FINISHED'):
                    # 캘리브레이션 전체 진행 단계에 'COOLDOWN'을 저장합니다. 대상: `self.calib_step`.
                    self.calib_step = 'COOLDOWN'
                    # 단계 대기 시간이 시작된 시각에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self.timer_start`.
                    self.timer_start = now
                # 양쪽 최근 세 동작의 일관성이 확인된 뒤 1초 대기하여 완료 상태를 확정합니다.
                elif now - self.timer_start >= 1.0:
                    # 캘리브레이션 전체 진행 단계에 'FINISHED'을 저장합니다. 대상: `self.calib_step`.
                    self.calib_step = 'FINISHED'
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # 캘리브레이션 전체 진행 단계에 `any((len(d.peaks) >= 3 and (not d.consistent) for d in detectors.values()))`이면 `'CHECK_CONSISTENCY'`, 아니면 `'COLLECTING'`을 저장합니다. 대상: `self.calib_step`.
                self.calib_step = ('CHECK_CONSISTENCY' if any(len(d.peaks) >= 3 and not d.consistent
                                                            # 앞줄에서 시작한 식·인수·자료 구조를 `for d in detectors.values()) else 'COLLECTING')` 내용으로 이어서 구성합니다.
                                                            for d in detectors.values()) else 'COLLECTING')
        # 현재 함수 처리를 끝내고 `dict(calib_step=self.calib_step, calib_rep_count=min(self.calib_reps_left, self.calib_reps_right), calib_reps_left=self.calib_reps_left, calib_reps_right=self.calib_reps…` 호출 결과를 호출한 쪽으로 반환합니다.
        return dict(calib_step=self.calib_step, calib_rep_count=min(self.calib_reps_left, self.calib_reps_right),
                    # 구성 항목: calib 반복 횟수 왼쪽(`calib_reps_left`)에 calib 반복 횟수 왼쪽(`self.calib_reps_left`); calib 반복 횟수 오른쪽(`calib_reps_right`)에 calib 반복 횟수 오른쪽(`self.calib_reps_right`)을 지정합니다.
                    calib_reps_left=self.calib_reps_left, calib_reps_right=self.calib_reps_right,
                    # 구성 항목: 좌우 각각 필요한 캘리브레이션 횟수(`target_calib_reps`)에 좌우 각각 필요한 캘리브레이션 횟수(`self.target_calib_reps`)을 지정합니다.
                    target_calib_reps=self.target_calib_reps,
                    # 구성 항목: 캘리브레이션(`calibration`)에 반복·조건식 `{side: detector.status() for (side, detector) in detectors.items()}`으로 구성한 값을 지정합니다.
                    calibration={side: detector.status() for side, detector in detectors.items()})

    # 완료된 캘리브레이션의 기준값과 최대 변화량 중앙값의 85%로 개인 가동 범위를 만들어 저장합니다.
    def compute_and_save_thresholds(self, exercise_name: str) -> dict:
        # 조건 `self.calib_step != 'FINISHED'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.calib_step != 'FINISHED':
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Calibration is incomplete; thresholds were not saved')` 예외를 발생시킵니다.
            raise ValueError('Calibration is incomplete; thresholds were not saved')
        # 좌우 가동 범위 시작값·목표값에 반복·조건식 `{side: detector.thresholds() for (side, detector) in self.calibration_detectors.items()}`으로 구성한 값을 저장합니다. 대상: `thresholds`.
        thresholds = {side: detector.thresholds() for side, detector in self.calibration_detectors.items()}
        # 조건 `self.data_manager`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.data_manager:
            # `self.data_manager.save_custom_threshold(exercise_name, thresholds, is_confirmed=True)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.data_manager.save_custom_threshold(exercise_name, thresholds, is_confirmed=True)
        # 현재 함수 처리를 끝내고 좌우 가동 범위 시작값·목표값(`thresholds`)을 호출한 쪽으로 반환합니다.
        return thresholds

    # 유효한 관절을 정규화하여 본 운동 엔진에 전달합니다. 게임 종료 결정은 클라이언트가 하므로 종료 플래그는 False입니다.
    def process_main_frame(self, raw_keypoints: list, is_occluded: bool, now: float = None) -> tuple:
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """본 운동 측정과 판정만 처리; 종료 여부는 클라이언트가 결정"""
        # 조건 `now is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if now is None:
            # 이번 처리에 사용할 현재 시각(초)에 시스템 현재 시각(초)(`time.time()`)을 저장합니다. 대상: `now`.
            now = time.time()

        # 조건 `is_occluded or not raw_keypoints`을 만족하는 경우에만 아래 분기를 실행합니다.
        if is_occluded or not raw_keypoints:
            # 현재 함수 처리를 끝내고 순서·구성을 지정한 `({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},…`을 호출한 쪽으로 반환합니다.
            return ({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},
                    # 구성 항목: 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 0; 이번 반복에서 추가된 참고 점수(`'score_delta'`)에 0.0; 이번 프레임에서 반복 횟수가 증가했는지 여부(`'is_count_updated'`)에 False을 지정합니다.
                    {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, False)

        # 본 운동도 캘리브레이션과 동일하게 신체 정규화를 적용합니다. 두 과정의 단위를 맞춰 저장한 개인 기준값을 그대로 비교합니다.
        norm_keypoints = self.normalizer.normalize(raw_keypoints)
        # 조건 `not norm_keypoints`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not norm_keypoints:
            # 현재 함수 처리를 끝내고 순서·구성을 지정한 `({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},…`을 호출한 쪽으로 반환합니다.
            return ({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},
                    # 구성 항목: 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 0; 이번 반복에서 추가된 참고 점수(`'score_delta'`)에 0.0; 이번 프레임에서 반복 횟수가 증가했는지 여부(`'is_count_updated'`)에 False을 지정합니다.
                    {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, False)

        # 함수 또는 처리 단계의 결과에 `self.motion_engine.process_keypoints(norm_keypoints)` 호출 결과를 저장합니다. 대상: `result`.
        result = self.motion_engine.process_keypoints(norm_keypoints)
        # 조건 `not result`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not result:
            # 현재 함수 처리를 끝내고 순서·구성을 지정한 `({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},…`을 호출한 쪽으로 반환합니다.
            return ({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},
                    # 구성 항목: 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 0; 이번 반복에서 추가된 참고 점수(`'score_delta'`)에 0.0; 이번 프레임에서 반복 횟수가 증가했는지 여부(`'is_count_updated'`)에 False을 지정합니다.
                    {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, False)

        # `(left_data, right_data)` 값에 순서·구성을 지정한 `(result['left'], result['right'])`을 저장합니다. 대상: `(left_data, right_data)`.
        left_data, right_data = result["left"], result["right"]
        # last 왼쪽 데이터에 `left_data.copy()` 호출 결과를 저장합니다. 대상: `self.last_left_data`.
        self.last_left_data = left_data.copy()
        # last 오른쪽 데이터에 `right_data.copy()` 호출 결과를 저장합니다. 대상: `self.last_right_data`.
        self.last_right_data = right_data.copy()

        # MAIN/TEST는 클라이언트 중지 명령으로 끝납니다. 서버 점수 기준은 종료 조건이 아닙니다.
        # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(left_data, right_data, False)`을 호출한 쪽으로 반환합니다.
        return left_data, right_data, False
