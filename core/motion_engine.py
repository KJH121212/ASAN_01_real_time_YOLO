# 한국어 코드 해설: 운동별 각도·좌표·거리 측정값을 계산하고 좌우 독립 상태 기계로 반복 횟수와 동작 품질을 판정합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [파일 정보]
# 파일명: core/motion_engine.py
# 설명: 최고점(Peak) 도달 후 변곡점(Reversal) 확인 시 카운트 트리거 및 복귀 히스테리시스 FSM
#       - 조기 카운트 방지: 65% 통과 즉시 카운트되는 로직 제거
#       - 유효 동작 기준: 가동 범위 40% 이상 도달 확인
#       - 변곡점 판정: 정점(Peak) 대비 가동 범위의 5% 이상 꺾여 내려올 때 카운트 트리거
#       - ANGLE, COORDINATE, LENGTH, RATIO(두 관절 간 거리의 비율) 지원
# ==============================================================================

# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# `numbers`에서 `Real` 기능을 가져옵니다.
from numbers import Real
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# 품질 등급 points에 키 'PERFECT', 'GOOD', 'BAD'로 구성한 사전을 저장합니다. 대상: `QUALITY_POINTS`.
QUALITY_POINTS = {'PERFECT': 1.0, 'GOOD': 0.5, 'BAD': 0.25}
# 품질 등급 codes에 키 'PERFECT', 'GOOD', 'BAD'로 구성한 사전을 저장합니다. 대상: `QUALITY_CODES`.
QUALITY_CODES = {'PERFECT': 3, 'GOOD': 2, 'BAD': 1}


# `SideFSM` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class SideFSM:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, side_name: str, motion_direction: str, thresholds: dict, target_reps: int = 10,
                 # 이 줄의 입력 인수: `target_score` = 게임 목표 점수; `scoring` = 본 운동 점수 판정을 사용하는지 여부. 기본값·형식은 아래 선언을 따릅니다.
                 target_score=None, scoring=True):
        # 좌우 구분에 좌우 이름(`side_name`)을 저장합니다. 대상: `self.side`.
        self.side = side_name
        # 운동할수록 측정값이 감소하는지 여부에 계산식 `motion_direction == 'DECREASING'`의 결과를 저장합니다. 대상: `self.is_decreasing`.
        self.is_decreasing = (motion_direction == "DECREASING")
        # 목표 반복 횟수에 목표 반복 횟수(`target_reps`)을 저장합니다. 대상: `self.target_reps`.
        self.target_reps = target_reps

        # 처리 단계에 'READY'을 저장합니다. 대상: `self.state`.
        self.state = "READY"
        # 인정된 누적 반복 횟수에 0을 저장합니다. 대상: `self.rep_count`.
        self.rep_count = 0
        # 서버 기본 가중치로 계산한 누적 참고 점수에 0.0을 저장합니다. 대상: `self.total_score`.
        self.total_score = 0.0
        # 게임 목표 점수에 `float(target_reps if target_score is None else target_score)` 호출 결과를 저장합니다. 대상: `self.target_score`.
        self.target_score = float(target_reps if target_score is None else target_score)
        # 조건 `not math.isfinite(self.target_score) or self.target_score <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(self.target_score) or self.target_score <= 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('target_score must be finite and positive')` 예외를 발생시킵니다.
            raise ValueError('target_score must be finite and positive')
        # 본 운동 점수 판정을 사용하는지 여부에 본 운동 점수 판정을 사용하는지 여부(`scoring`)을 저장합니다. 대상: `self.scoring`.
        self.scoring = scoring

        # 기본 임계값 마진
        # 개인 가동 범위의 시작값에 `thresholds.get('start_val', 0.15 if not self.is_decreasing else 0.8)` 호출 결과를 저장합니다. 대상: `self.start_val`.
        self.start_val = thresholds.get("start_val", 0.15 if not self.is_decreasing else 0.80)
        # 개인 가동 범위의 목표값에 `thresholds.get('target_val', 0.8 if not self.is_decreasing else 0.15)` 호출 결과를 저장합니다. 대상: `self.target_val`.
        self.target_val = thresholds.get("target_val", 0.80 if not self.is_decreasing else 0.15)

        # 현재 동작 중 기록한 최소 측정값에 999.0을 저장합니다. 대상: `self.min_val_recorded`.
        self.min_val_recorded = 999.0
        # 현재 동작 중 기록한 최대 측정값에 계산식 `-999.0`의 결과를 저장합니다. 대상: `self.max_val_recorded`.
        self.max_val_recorded = -999.0
        # 마지막으로 확정한 동작 품질에 'READY'을 저장합니다. 대상: `self.last_quality`.
        self.last_quality = "READY"
        # 완료된 반복별 분석 기록에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.completed_reps_history`.
        self.completed_reps_history = []
        # 현재 가동 범위 설정의 변경 번호에 0을 저장합니다. 대상: `self.threshold_version`.
        self.threshold_version = 0

    # 이미 검증한 새 가동 범위를 적용하고 이전 동작의 최고점은 버립니다. 완료 횟수·점수는 유지합니다.
    def replace_thresholds(self, start_val: float, target_val: float, version: int):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Apply validated ROM without reusing a peak from the previous range."""
        # 개인 가동 범위의 시작값에 개인 가동 범위의 시작값(`start_val`)을 저장합니다. 대상: `self.start_val`.
        self.start_val = start_val
        # 개인 가동 범위의 목표값에 개인 가동 범위의 목표값(`target_val`)을 저장합니다. 대상: `self.target_val`.
        self.target_val = target_val
        # 현재 가동 범위 설정의 변경 번호에 설정 변경 번호(`version`)을 저장합니다. 대상: `self.threshold_version`.
        self.threshold_version = version
        # 현재 동작 중 기록한 최소 측정값에 999.0을 저장합니다. 대상: `self.min_val_recorded`.
        self.min_val_recorded = 999.0
        # 현재 동작 중 기록한 최대 측정값에 계산식 `-999.0`의 결과를 저장합니다. 대상: `self.max_val_recorded`.
        self.max_val_recorded = -999.0
        # 이미 완료된 횟수·품질은 유지하며 시작 자세 쪽으로 복귀한 뒤 다시 동작을 받습니다.
        # 이전 목표에 도달하지 못했어도 새 가동 범위 기준으로 복귀를 판단합니다.
        # 처리 단계에 `self.target_reached`이면 `'WAITING'`, 아니면 `'RETURNING'`을 저장합니다. 대상: `self.state`.
        self.state = "WAITING" if self.target_reached else "RETURNING"

    # 데코레이터 `property`를 바로 아래 함수·클래스에 적용합니다.
    @property
    # 캘리브레이션 횟수 목표만 확인합니다. MAIN/TEST의 서버 점수만으로 게임을 끝내지 않습니다.
    def target_reached(self):
        # MAIN/TEST는 서버 점수만으로 끝내지 않습니다. 종료는 클라이언트가 결정합니다.
        # MAIN/TEST는 이 서버 목표값으로 중단하지 않습니다. 캘리브레이션 용도로 사용하는 횟수 목표만 서버에서 완료 판정합니다.
        return False if self.scoring else self.rep_count >= self.target_reps

    # 점수 상태 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def score_status(self, awarded=False):
        # 현재 함수 처리를 끝내고 키 'total_score', 'target_score', 'score', 'score_delta'로 구성한 사전을 호출한 쪽으로 반환합니다.
        return {'total_score': self.total_score, 'target_score': self.target_score,
                # 구성 항목: 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 `awarded`이면 `QUALITY_CODES.get(self.last_quality, 0)`, 아니면 `0`을 지정합니다.
                'score': QUALITY_CODES.get(self.last_quality, 0) if awarded else 0,
                # 구성 항목: 이번 반복에서 추가된 참고 점수(`'score_delta'`)에 `awarded`이면 `QUALITY_POINTS.get(self.last_quality, 0.0)`, 아니면 `0.0`을 지정합니다.
                'score_delta': QUALITY_POINTS.get(self.last_quality, 0.0) if awarded else 0.0}

    # 기준 가동 범위 대비 최고 도달 비율로 PERFECT·GOOD·BAD·INVALID 등급을 구합니다.
    def evaluate_quality(self, peak_val: float) -> str:
        # 기준 가동 범위의 크기에 `abs(self.target_val - self.start_val)` 호출 결과를 저장합니다. 대상: `total_range`.
        total_range = abs(self.target_val - self.start_val)
        # 조건 `total_range == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if total_range == 0:
            # 현재 함수 처리를 끝내고 'GOOD'을 호출한 쪽으로 반환합니다.
            return "GOOD"

        # 최고점까지 달성한 실제 변화량에 `abs(peak_val - self.start_val)` 호출 결과를 저장합니다. 대상: `achieved_range`.
        achieved_range = abs(peak_val - self.start_val)
        # 비율에 계산식 `achieved_range / total_range`의 결과를 저장합니다. 대상: `ratio`.
        ratio = achieved_range / total_range

        # 조건 `ratio >= 0.7`을 만족하는 경우에만 아래 분기를 실행합니다.
        if ratio >= 0.70:
            # 현재 함수 처리를 끝내고 'PERFECT'을 호출한 쪽으로 반환합니다.
            return "PERFECT"
        # 조건 `ratio >= 0.5`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif ratio >= 0.50:
            # 현재 함수 처리를 끝내고 'GOOD'을 호출한 쪽으로 반환합니다.
            return "GOOD"
        # 조건 `ratio >= 0.35`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif ratio >= 0.35:
            # 현재 함수 처리를 끝내고 'BAD'을 호출한 쪽으로 반환합니다.
            return "BAD"
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # 현재 함수 처리를 끝내고 'INVALID'을 호출한 쪽으로 반환합니다.
            return "INVALID"

    # 15%에서 동작을 시작하고, 40% 이상 도달한 뒤 5% 되돌아오면 1회로 셉니다. 30% 이하 복귀 후 다음 동작을 받습니다.
    def update(self, current_val: float) -> dict:
        # 조건 `current_val is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if current_val is None:
            # 현재 함수 처리를 끝내고 키 'state', 'rep_count', 'progress_ratio', 'quality', 'is_count_updated'로 구성한 사전을 호출한 쪽으로 반환합니다.
            return {
                # 앞줄에서 시작한 식·인수·자료 구조를 `**self.score_status(),` 내용으로 이어서 구성합니다.
                **self.score_status(),
                # 구성 항목: 처리 단계(`'state'`)에 처리 단계(`self.state`)을 지정합니다.
                "state": self.state,
                # 구성 항목: 인정된 누적 반복 횟수(`'rep_count'`)에 인정된 누적 반복 횟수(`self.rep_count`)을 지정합니다.
                "rep_count": self.rep_count,
                # 구성 항목: 설정 가동 범위 대비 진행 비율(0~1)(`'progress_ratio'`)에 0.0을 지정합니다.
                "progress_ratio": 0.0,
                # 구성 항목: 동작 품질 등급(`'quality'`)에 마지막으로 확정한 동작 품질(`self.last_quality`)을 지정합니다.
                "quality": self.last_quality,
                # 구성 항목: 이번 프레임에서 반복 횟수가 증가했는지 여부(`'is_count_updated'`)에 False을 지정합니다.
                "is_count_updated": False
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            }

        # 1. 실시간 극값 및 실시간 진행률(게이지)은 3회를 채운 후에도 계속 계산
        # 현재 동작 중 기록한 최소 측정값에 `min(self.min_val_recorded, current_val)` 호출 결과를 저장합니다. 대상: `self.min_val_recorded`.
        self.min_val_recorded = min(self.min_val_recorded, current_val)
        # 현재 동작 중 기록한 최대 측정값에 `max(self.max_val_recorded, current_val)` 호출 결과를 저장합니다. 대상: `self.max_val_recorded`.
        self.max_val_recorded = max(self.max_val_recorded, current_val)

        # 시작값과 목표값 사이의 전체 가동 범위에 `abs(self.target_val - self.start_val)` 호출 결과를 저장합니다. 대상: `val_range`.
        val_range = abs(self.target_val - self.start_val)
        # 설정 가동 범위 대비 진행 비율(0~1)에 `val_range > 0`이면 `float(np.clip(abs(current_val - self.start_val) / val_range, 0.0, 1.0))`, 아니면 `0.0`을 저장합니다. 대상: `progress_ratio`.
        progress_ratio = float(np.clip(abs(current_val - self.start_val) / val_range, 0.0, 1.0)) if val_range > 0 else 0.0
        # 설정 가동 범위 대비 진행 비율(0~1)에 `round(progress_ratio, 2)` 호출 결과를 저장합니다. 대상: `progress_ratio`.
        progress_ratio = round(progress_ratio, 2)
        # 이번 프레임에서 반복 횟수가 증가했는지 여부에 False을 저장합니다. 대상: `is_count_updated`.
        is_count_updated = False

        # 2. 목표 횟수를 이미 채운 경우: 카운트 증가 FSM만 스킵하고 실시간 게이지는 계속 반환
        # 조건 `self.target_reached`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.target_reached:
            # 처리 단계에 'WAITING'을 저장합니다. 대상: `self.state`.
            self.state = "WAITING"
            # 현재 함수 처리를 끝내고 키 'state', 'rep_count', 'progress_ratio', 'quality', 'is_count_updated'로 구성한 사전을 호출한 쪽으로 반환합니다.
            return {
                # 앞줄에서 시작한 식·인수·자료 구조를 `**self.score_status(),` 내용으로 이어서 구성합니다.
                **self.score_status(),
                # 구성 항목: 처리 단계(`'state'`)에 'WAITING'을 지정합니다.
                "state": "WAITING",
                # 구성 항목: 인정된 누적 반복 횟수(`'rep_count'`)에 인정된 누적 반복 횟수(`self.rep_count`)을 지정합니다.
                "rep_count": self.rep_count,
                # 구성 항목: 설정 가동 범위 대비 진행 비율(0~1)(`'progress_ratio'`)에 설정 가동 범위 대비 진행 비율(0~1)(`progress_ratio`)을 지정합니다.
                "progress_ratio": progress_ratio,  # <-- 멈추지 않고 실시간 움직임 반영
                # 구성 항목: 동작 품질 등급(`'quality'`)에 마지막으로 확정한 동작 품질(`self.last_quality`)을 지정합니다.
                "quality": self.last_quality,
                # 구성 항목: 이번 프레임에서 반복 횟수가 증가했는지 여부(`'is_count_updated'`)에 False을 지정합니다.
                "is_count_updated": False
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            }

        # ----------------------------------------------------------------------
        # 1. 준비 -> 수축 개시 (15% 움직임 감지 시 진입)
        # ----------------------------------------------------------------------
        # 조건 `self.state == 'READY'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.state == "READY":
            # 설정 가동 범위의 15% 이상 움직이면 READY에서 PUSHING으로 전환합니다. 이 시점에는 아직 횟수를 올리지 않습니다.
            if progress_ratio >= 0.15:
                # 처리 단계에 'PUSHING'을 저장합니다. 대상: `self.state`.
                self.state = "PUSHING"

        # ----------------------------------------------------------------------
        # 2. 수축 상승 -> 최고점(Peak)을 찍고 변곡점(하강) 진입 시 카운트 트리거
        # ----------------------------------------------------------------------
        # 조건 `self.state == 'PUSHING'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif self.state == "PUSHING":
            # 현재 동작의 최고 도달 측정값에 `self.is_decreasing`이면 `self.min_val_recorded`, 아니면 `self.max_val_recorded`을 저장합니다. 대상: `peak_val`.
            peak_val = self.min_val_recorded if self.is_decreasing else self.max_val_recorded
            # 현재 값과 시작값 사이의 거리에 `abs(current_val - self.start_val)` 호출 결과를 저장합니다. 대상: `current_range`.
            current_range = abs(current_val - self.start_val)
            # 최고 도달 값과 시작값 사이의 거리에 `abs(peak_val - self.start_val)` 호출 결과를 저장합니다. 대상: `peak_range`.
            peak_range = abs(peak_val - self.start_val)

            # 유효 동작 최소 진입 임계치 (전체 범위의 40% 이상 도달한 적이 있어야 함)
            # 최고 도달 범위가 전체 기준 범위의 40% 이상이어야 반복 후보가 됩니다. 이 기준과 품질 등급 기준은 별개입니다.
            min_activation_threshold = 0.40
            # 최소 가동 범위를 한 번 이상 달성했는지 여부에 `val_range > 0`이면 `peak_range / val_range >= min_activation_threshold`, 아니면 `False`을 저장합니다. 대상: `has_reached_threshold`.
            has_reached_threshold = (peak_range / val_range >= min_activation_threshold) if val_range > 0 else False

            # 변곡점 판정: 정점 대비 5% 이상 확실히 되돌아 내려왔는지 확인 (노이즈 방지)
            # 전체 기준 범위의 5%를 방향 전환 확인 폭으로 사용합니다. 최고점 도달 직후 작은 잡음으로 반복을 확정하지 않게 합니다.
            reversal_delta = 0.05 * val_range
            # 최고점에서 충분히 되돌아왔는지 여부에 계산식 `has_reached_threshold and peak_range - current_range >= reversal_delta`의 결과를 저장합니다. 대상: `turned_down`.
            turned_down = has_reached_threshold and ((peak_range - current_range) >= reversal_delta)

            # 조기 카운트 방지: 올라가는 도중엔 카운트하지 않고 변곡점(하강)을 통과할 때만 카운트
            # 충분한 가동 범위에 도달한 뒤 복귀 방향 전환까지 확인된 프레임에서만 한 번 카운트합니다.
            if turned_down:
                # 동작 품질 등급에 `self.evaluate_quality(peak_val)` 호출 결과를 저장합니다. 대상: `quality`.
                quality = self.evaluate_quality(peak_val)
                # 조건 `quality == 'INVALID'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if quality == "INVALID":
                    # 동작 품질 등급에 'BAD'을 저장합니다. 대상: `quality`.
                    quality = "BAD"

                # 인정된 누적 반복 횟수에 `1`을 더해 누적합니다. 대상: `self.rep_count`.
                self.rep_count += 1
                # 이번 반복에서 추가된 참고 점수에 계산식 `QUALITY_POINTS[quality]`의 결과를 저장합니다. 대상: `score_delta`.
                score_delta = QUALITY_POINTS[quality]
                # 서버 기본 가중치로 계산한 누적 참고 점수에 `score_delta`을 더해 누적합니다. 대상: `self.total_score`.
                self.total_score += score_delta
                # 이번 프레임에서 반복 횟수가 증가했는지 여부에 True을 저장합니다. 대상: `is_count_updated`.
                is_count_updated = True
                # 마지막으로 확정한 동작 품질에 동작 품질 등급(`quality`)을 저장합니다. 대상: `self.last_quality`.
                self.last_quality = quality

                # `self.completed_reps_history.append({'side': self.side, 'rep_num': self.rep_count, 'duration_sec': 0.0, 'min_val': round(self.min_val_recorded, 2), 'max_val': round(self.…`을 호출하여 항목 하나를 끝에 추가합니다.
                self.completed_reps_history.append({
                    # 구성 항목: 좌우 구분(`'side'`)에 좌우 구분(`self.side`)을 지정합니다.
                    "side": self.side,
                    # 구성 항목: 반복 num(`'rep_num'`)에 인정된 누적 반복 횟수(`self.rep_count`)을 지정합니다.
                    "rep_num": self.rep_count,
                    # 구성 항목: 길이 sec(`'duration_sec'`)에 0.0을 지정합니다.
                    "duration_sec": 0.0,
                    # 구성 항목: `min_val` 값(`'min_val'`)에 `round(self.min_val_recorded, 2)` 호출 결과를 지정합니다.
                    "min_val": round(self.min_val_recorded, 2),
                    # 구성 항목: `max_val` 값(`'max_val'`)에 `round(self.max_val_recorded, 2)` 호출 결과를 지정합니다.
                    "max_val": round(self.max_val_recorded, 2),
                    # 구성 항목: achieved 가동 범위(`'achieved_rom'`)에 `round(abs(self.max_val_recorded - self.min_val_recorded), 2)` 호출 결과를 지정합니다.
                    "achieved_rom": round(abs(self.max_val_recorded - self.min_val_recorded), 2),
                    # 구성 항목: 동작 품질 등급(`'quality'`)에 동작 품질 등급(`quality`)을 지정합니다.
                    "quality": quality,
                    # 구성 항목: 이번 반복에서 추가된 참고 점수(`'score_delta'`)에 이번 반복에서 추가된 참고 점수(`score_delta`)을 지정합니다.
                    "score_delta": score_delta,
                    # 구성 항목: 서버 기본 가중치로 계산한 누적 참고 점수(`'total_score'`)에 서버 기본 가중치로 계산한 누적 참고 점수(`self.total_score`)을 지정합니다.
                    "total_score": self.total_score,
                    # 구성 항목: 개인 가동 범위의 시작값(`'start_val'`)에 개인 가동 범위의 시작값(`self.start_val`)을 지정합니다.
                    "start_val": self.start_val,
                    # 구성 항목: 개인 가동 범위의 목표값(`'target_val'`)에 개인 가동 범위의 목표값(`self.target_val`)을 지정합니다.
                    "target_val": self.target_val,
                    # 구성 항목: 현재 가동 범위 설정의 변경 번호(`'threshold_version'`)에 현재 가동 범위 설정의 변경 번호(`self.threshold_version`)을 지정합니다.
                    "threshold_version": self.threshold_version,
                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                })

                # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                print(f"\n[PEAK REVERSAL COUNT!] {self.side.upper()} Reps: {self.rep_count} | "
                      # 앞줄에서 시작한 식·인수·자료 구조를 `f"Score: {self.total_score:g}/{self.target_score:g} | Quality: {quality} (Peak: {peak_val:.2f})")` 내용으로 이어서 구성합니다.
                      f"Score: {self.total_score:g}/{self.target_score:g} | Quality: {quality} (Peak: {peak_val:.2f})")

                # 처리 단계에 `self.target_reached`이면 `'WAITING'`, 아니면 `'RETURNING'`을 저장합니다. 대상: `self.state`.
                self.state = "WAITING" if self.target_reached else "RETURNING"

        # ----------------------------------------------------------------------
        # 3. 복귀 확인 및 락 해제 (30% 이하로 돌아와야 다음 횟수 준비)
        # ----------------------------------------------------------------------
        # 조건 `self.state == 'RETURNING'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif self.state == "RETURNING":
            # 시작 자세 쪽 30% 범위 안으로 복귀해야 다음 반복을 시작할 수 있습니다. 최고점 부근에서 반복 카운트되는 것을 막습니다.
            if progress_ratio <= 0.30:
                # 처리 단계에 'READY'을 저장합니다. 대상: `self.state`.
                self.state = "READY"
                # 현재 동작 중 기록한 최소 측정값에 999.0을 저장합니다. 대상: `self.min_val_recorded`.
                self.min_val_recorded = 999.0
                # 현재 동작 중 기록한 최대 측정값에 계산식 `-999.0`의 결과를 저장합니다. 대상: `self.max_val_recorded`.
                self.max_val_recorded = -999.0

        # 현재 함수 처리를 끝내고 키 'state', 'rep_count', 'progress_ratio', 'quality', 'is_count_updated'로 구성한 사전을 호출한 쪽으로 반환합니다.
        return {
            # 구성 항목: `awarded` 값(`awarded`)에 이번 프레임에서 반복 횟수가 증가했는지 여부(`is_count_updated`)을 지정합니다.
            **self.score_status(awarded=is_count_updated),
            # 구성 항목: 처리 단계(`'state'`)에 처리 단계(`self.state`)을 지정합니다.
            "state": self.state,
            # 구성 항목: 인정된 누적 반복 횟수(`'rep_count'`)에 인정된 누적 반복 횟수(`self.rep_count`)을 지정합니다.
            "rep_count": self.rep_count,
            # 구성 항목: 설정 가동 범위 대비 진행 비율(0~1)(`'progress_ratio'`)에 설정 가동 범위 대비 진행 비율(0~1)(`progress_ratio`)을 지정합니다.
            "progress_ratio": progress_ratio,
            # 구성 항목: 동작 품질 등급(`'quality'`)에 마지막으로 확정한 동작 품질(`self.last_quality`)을 지정합니다.
            "quality": self.last_quality,
            # 구성 항목: 이번 프레임에서 반복 횟수가 증가했는지 여부(`'is_count_updated'`)에 이번 프레임에서 반복 횟수가 증가했는지 여부(`is_count_updated`)을 지정합니다.
            "is_count_updated": is_count_updated
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }


# `MotionEngine` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class MotionEngine:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, config: dict, custom_thresholds: dict = None, mode: str = "MAIN", target_reps: int = 10,
                 # 이 줄의 입력 인수: `target_score` = 게임 목표 점수. 기본값·형식은 아래 선언을 따릅니다.
                 target_score=None):
        # 현재 작업 설정에 현재 작업 설정(`config`)을 저장합니다. 대상: `self.config`.
        self.config = config
        # 운동 설정의 식별 이름에 `config.get('exercise_name', 'unknown')` 호출 결과를 저장합니다. 대상: `self.exercise_name`.
        self.exercise_name = config.get("exercise_name", "unknown")
        # 현재 가동 범위 설정의 변경 번호에 0을 저장합니다. 대상: `self.threshold_version`.
        self.threshold_version = 0
        # 운동별 측정 방식 설정에 `config.get('eval_config', {})` 호출 결과를 저장합니다. 대상: `eval_cfg`.
        eval_cfg = config.get("eval_config", {})

        # `aliases` 값에 키 'RELATIVE_Y', 'ANKLE_HEIGHT', 'DISTANCE'로 구성한 사전을 저장합니다. 대상: `aliases`.
        aliases = {'RELATIVE_Y': 'COORDINATE', 'ANKLE_HEIGHT': 'COORDINATE', 'DISTANCE': 'LENGTH'}
        # 측정 방식(각도·좌표·길이·비율)에 `aliases.get(eval_cfg.get('metric_type'), eval_cfg.get('metric_type', 'ANGLE'))` 호출 결과를 저장합니다. 대상: `self.metric_type`.
        self.metric_type = aliases.get(eval_cfg.get('metric_type'), eval_cfg.get('metric_type', 'ANGLE'))
        # 조건 `self.metric_type not in {'ANGLE', 'COORDINATE', 'LENGTH', 'RATIO'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.metric_type not in {'ANGLE', 'COORDINATE', 'LENGTH', 'RATIO'}:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('metric_type must be ANGLE, COORDINATE, LENGTH or RATIO')` 예외를 발생시킵니다.
            raise ValueError('metric_type must be ANGLE, COORDINATE, LENGTH or RATIO')
        # 조건 `self.metric_type == 'RATIO'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.metric_type == 'RATIO':
            # `('numerator_kpts', 'denominator_kpts')`에서 항목을 차례로 꺼내 `field`에 넣고 아래 처리를 반복합니다.
            for field in ('numerator_kpts', 'denominator_kpts'):
                # `('left', 'right')`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
                for side in ('left', 'right'):
                    # `pair` 값에 `eval_cfg.get(field, {}).get(side)` 호출 결과를 저장합니다. 대상: `pair`.
                    pair = eval_cfg.get(field, {}).get(side)
                    # 조건 `not isinstance(pair, list) or len(pair) != 2 or any((type(i) is not int or not 0 <= i < 17 for i in pair)) or (pair[0] == pair[1])`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if (not isinstance(pair, list) or len(pair) != 2
                            # 앞줄에서 시작한 식·인수·자료 구조를 `or any(type(i) is not int or not 0 <= i < 17 for i in pair)` 내용으로 이어서 구성합니다.
                            or any(type(i) is not int or not 0 <= i < 17 for i in pair)
                            # 앞줄에서 시작한 식·인수·자료 구조를 `or pair[0] == pair[1]):` 내용으로 이어서 구성합니다.
                            or pair[0] == pair[1]):
                        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'RATIO {field}.{side} requires two distinct COCO joint IDs')` 예외를 발생시킵니다.
                        raise ValueError(f'RATIO {field}.{side} requires two distinct COCO joint IDs')
        # 측정값 증가·감소 방향에 `eval_cfg.get('motion_direction', 'INCREASING')` 호출 결과를 저장합니다. 대상: `self.motion_direction`.
        self.motion_direction = eval_cfg.get("motion_direction", "INCREASING")
        # 실행 모드에 실행 모드(`mode`)을 저장합니다. 대상: `self.mode`.
        self.mode = mode

        # `requires_metric_tag` 값에 계산식 `bool(eval_cfg.get('reference_kpt')) or self.metric_type in {'LENGTH', 'RATIO'}`의 결과를 저장합니다. 대상: `requires_metric_tag`.
        requires_metric_tag = bool(eval_cfg.get('reference_kpt')) or self.metric_type in {'LENGTH', 'RATIO'}
        # 조건 `custom_thresholds and any(((requires_metric_tag or custom_thresholds.get(side, {}).get('metric_type') is not None) and aliases.get(custom_thresholds.get(side, {}).get('m…`을 만족하는 경우에만 아래 분기를 실행합니다.
        if custom_thresholds and any(
                # 앞줄에서 시작한 식·인수·자료 구조를 `(requires_metric_tag or custom_thresholds.get(side, {}).get('metric_type') is not None) and` 내용으로 이어서 구성합니다.
                (requires_metric_tag or custom_thresholds.get(side, {}).get('metric_type') is not None) and
                # 앞줄에서 시작한 식·인수·자료 구조를 `aliases.get(custom_thresholds.get(side, {}).get('metric_type'),` 내용으로 이어서 구성합니다.
                aliases.get(custom_thresholds.get(side, {}).get('metric_type'),
                            # 앞줄에서 시작한 식·인수·자료 구조를 `custom_thresholds.get(side, {}).get('metric_type')) != self.metric_type` 내용으로 이어서 구성합니다.
                            custom_thresholds.get(side, {}).get('metric_type')) != self.metric_type
                # 앞줄에서 시작한 식·인수·자료 구조를 `for side in ('left', 'right')):` 내용으로 이어서 구성합니다.
                for side in ('left', 'right')):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Measurement metric changed; run CALIBRATION again before MAIN/TEST')` 예외를 발생시킵니다.
            raise ValueError('Measurement metric changed; run CALIBRATION again before MAIN/TEST')

        # 운동 설정의 기본 가동 범위에 `config.get('default_thresholds', {})` 호출 결과를 저장합니다. 대상: `default_th`.
        default_th = config.get("default_thresholds", {})
        # th 왼쪽에 `custom_thresholds`이면 `custom_thresholds.get('left')`, 아니면 `default_th.get('left', {})`을 저장합니다. 대상: `th_left`.
        th_left = custom_thresholds.get("left") if custom_thresholds else default_th.get("left", {})
        # th 오른쪽에 `custom_thresholds`이면 `custom_thresholds.get('right')`, 아니면 `default_th.get('right', {})`을 저장합니다. 대상: `th_right`.
        th_right = custom_thresholds.get("right") if custom_thresholds else default_th.get("right", {})

        # 왼쪽 반복 판정 상태 기계에 `SideFSM('left', th_left.get('motion_direction', self.motion_direction), th_left, target_reps, target_score, mode != 'CALIBRATION')` 호출 결과를 저장합니다. 대상: `self.fsm_left`.
        self.fsm_left = SideFSM("left", th_left.get("motion_direction", self.motion_direction), th_left, target_reps, target_score, mode != 'CALIBRATION')
        # 오른쪽 반복 판정 상태 기계에 `SideFSM('right', th_right.get('motion_direction', self.motion_direction), th_right, target_reps, target_score, mode != 'CALIBRATION')` 호출 결과를 저장합니다. 대상: `self.fsm_right`.
        self.fsm_right = SideFSM("right", th_right.get("motion_direction", self.motion_direction), th_right, target_reps, target_score, mode != 'CALIBRATION')

    # 조회 횟수 가동 범위 기준 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def get_count_thresholds(self) -> dict:
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Return a copy of each side's reference ROM, in the metric's units."""
        # 현재 함수 처리를 끝내고 반복·조건식 `{side: {'start_val': fsm.start_val, 'target_val': fsm.target_val} for (side, fsm) in (('left', self.fsm_left), ('right', self.fsm_right))}`으로 구성한 값을 호출한 쪽으로 반환합니다.
        return {
            # 구성 항목: 개인 가동 범위의 시작값(`'start_val'`)에 개인 가동 범위의 시작값(`fsm.start_val`); 개인 가동 범위의 목표값(`'target_val'`)에 개인 가동 범위의 목표값(`fsm.target_val`)을 지정합니다.
            side: {"start_val": fsm.start_val, "target_val": fsm.target_val}
            # 앞줄에서 시작한 식·인수·자료 구조를 `for side, fsm in (("left", self.fsm_left), ("right", self.fsm_right))` 내용으로 이어서 구성합니다.
            for side, fsm in (("left", self.fsm_left), ("right", self.fsm_right))
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }

    # 횟수 가동 범위 기준 상태 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def count_threshold_status(self) -> dict:
        # 현재 함수 처리를 끝내고 키 'exercise_name', 'metric_type', 'motion_direction', 'threshold_version', 'thresholds', 'target_score', 'scori…로 구성한 사전을 호출한 쪽으로 반환합니다.
        return {
            # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 운동 설정의 식별 이름(`self.exercise_name`)을 지정합니다.
            "exercise_name": self.exercise_name,
            # 구성 항목: 측정 방식(각도·좌표·길이·비율)(`'metric_type'`)에 측정 방식(각도·좌표·길이·비율)(`self.metric_type`)을 지정합니다.
            "metric_type": self.metric_type,
            # 구성 항목: 측정값 증가·감소 방향(`'motion_direction'`)에 측정값 증가·감소 방향(`self.motion_direction`)을 지정합니다.
            "motion_direction": self.motion_direction,
            # 구성 항목: 현재 가동 범위 설정의 변경 번호(`'threshold_version'`)에 현재 가동 범위 설정의 변경 번호(`self.threshold_version`)을 지정합니다.
            "threshold_version": self.threshold_version,
            # 구성 항목: 좌우 가동 범위 시작값·목표값(`'thresholds'`)에 `self.get_count_thresholds()` 호출 결과를 지정합니다.
            "thresholds": self.get_count_thresholds(),
            # 구성 항목: 게임 목표 점수(`'target_score'`)에 게임 목표 점수(`self.fsm_left.target_score`)을 지정합니다.
            "target_score": self.fsm_left.target_score,
            # 구성 항목: 점수 판정 mode(`'scoring_mode'`)에 `self.mode == 'CALIBRATION'`이면 `'REPETITIONS'`, 아니면 `'CLIENT'`을 지정합니다.
            "scoring_mode": "REPETITIONS" if self.mode == 'CALIBRATION' else "CLIENT",
            # 구성 항목: completion 점유 연결(`'completion_owner'`)에 `self.mode == 'CALIBRATION'`이면 `'SERVER'`, 아니면 `'CLIENT'`을 지정합니다.
            "completion_owner": "SERVER" if self.mode == 'CALIBRATION' else "CLIENT",
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }

    # 부분 가동 범위 변경 요청을 검증하고 판정 기준을 갱신합니다. 진행 중이던 동작은 새 범위에서 복귀 후 다시 시작합니다.
    def update_count_thresholds(self, thresholds: dict) -> list:
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Validate both sides atomically, then update only changed sides.

        A partial patch (e.g. just left.target_val) preserves omitted values.
        This changes reference ROM, not the 40% activation/quality ratios.
        """
        # 조건 `not isinstance(thresholds, dict) or not thresholds`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not isinstance(thresholds, dict) or not thresholds:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('thresholds must be a nonempty object with left/right')` 예외를 발생시킵니다.
            raise ValueError("thresholds must be a nonempty object with left/right")
        # 조건 `set(thresholds) - {'left', 'right'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        if set(thresholds) - {"left", "right"}:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('thresholds supports only left and right')` 예외를 발생시킵니다.
            raise ValueError("thresholds supports only left and right")

        # `current` 값에 `self.get_count_thresholds()` 호출 결과를 저장합니다. 대상: `current`.
        current = self.get_count_thresholds()
        # `validated` 값에 빈 사전을 저장합니다. 대상: `validated`.
        validated = {}
        # `thresholds.items()`에서 항목을 차례로 꺼내 `(side, patch)`에 넣고 아래 처리를 반복합니다.
        for side, patch in thresholds.items():
            # 조건 `not isinstance(patch, dict) or not patch`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not isinstance(patch, dict) or not patch:
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side} must contain start_val and/or target_val')` 예외를 발생시킵니다.
                raise ValueError(f"{side} must contain start_val and/or target_val")
            # 조건 `set(patch) - {'start_val', 'target_val'}`을 만족하는 경우에만 아래 분기를 실행합니다.
            if set(patch) - {"start_val", "target_val"}:
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side} supports only start_val and target_val')` 예외를 발생시킵니다.
                raise ValueError(f"{side} supports only start_val and target_val")
            # 측정값에 빈 사전을 저장합니다. 대상: `values`.
            values = {**current[side], **patch}
            # `values.items()`에서 항목을 차례로 꺼내 `(key, value)`에 넣고 아래 처리를 반복합니다.
            for key, value in values.items():
                # 조건 `isinstance(value, bool) or not isinstance(value, Real)`을 만족하는 경우에만 아래 분기를 실행합니다.
                if isinstance(value, bool) or not isinstance(value, Real):
                    # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}.{key} must be a finite number')` 예외를 발생시킵니다.
                    raise ValueError(f"{side}.{key} must be a finite number")
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `converted` 값에 `float(value)` 호출 결과를 저장합니다. 대상: `converted`.
                    converted = float(value)
                # `(ValueError, OverflowError)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except (ValueError, OverflowError):
                    # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}.{key} must be a finite number')` 예외를 발생시킵니다.
                    raise ValueError(f"{side}.{key} must be a finite number") from None
                # 조건 `not math.isfinite(converted)`을 만족하는 경우에만 아래 분기를 실행합니다.
                if not math.isfinite(converted):
                    # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}.{key} must be a finite number')` 예외를 발생시킵니다.
                    raise ValueError(f"{side}.{key} must be a finite number")
                # `values[key]` 값에 `converted` 값(`converted`)을 저장합니다. 대상: `values[key]`.
                values[key] = converted
            # `(start, target)` 값에 순서·구성을 지정한 `(values['start_val'], values['target_val'])`을 저장합니다. 대상: `(start, target)`.
            start, target = values["start_val"], values["target_val"]
            # 조건 `not math.isfinite(target - start)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not math.isfinite(target - start):
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}: ROM must be finite')` 예외를 발생시킵니다.
                raise ValueError(f"{side}: ROM must be finite")
            # 한쪽 동작의 반복 판정 상태 기계에 `side == 'left'`이면 `self.fsm_left`, 아니면 `self.fsm_right`을 저장합니다. 대상: `fsm`.
            fsm = self.fsm_left if side == "left" else self.fsm_right
            # 조건 `fsm.is_decreasing`을 만족하는 경우에만 아래 분기를 실행합니다.
            if fsm.is_decreasing:
                # 조건 `target >= start`을 만족하는 경우에만 아래 분기를 실행합니다.
                if target >= start:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}: DECREASING requires target_val < start_val')` 예외를 발생시킵니다.
                    raise ValueError(f"{side}: DECREASING requires target_val < start_val")
            # 조건 `target <= start`을 만족하는 경우에만 아래 분기를 실행합니다.
            elif target <= start:
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}: INCREASING requires target_val > start_val')` 예외를 발생시킵니다.
                raise ValueError(f"{side}: INCREASING requires target_val > start_val")
            # 조건 `self.metric_type == 'ANGLE' and (not all((0 <= v <= 180 for v in values.values())))`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.metric_type == "ANGLE" and not all(0 <= v <= 180 for v in values.values()):
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}: ANGLE values must be between 0 and 180 degrees')` 예외를 발생시킵니다.
                raise ValueError(f"{side}: ANGLE values must be between 0 and 180 degrees")
            # 조건 `self.metric_type in {'LENGTH', 'RATIO'} and any((v < 0 for v in values.values()))`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.metric_type in {"LENGTH", "RATIO"} and any(v < 0 for v in values.values()):
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{side}: {self.metric_type} values must be nonnegative')` 예외를 발생시킵니다.
                raise ValueError(f"{side}: {self.metric_type} values must be nonnegative")
            # 조건 `values != current[side]`을 만족하는 경우에만 아래 분기를 실행합니다.
            if values != current[side]:
                # `validated[side]` 값에 측정값(`values`)을 저장합니다. 대상: `validated[side]`.
                validated[side] = values

        # 조건 `validated`을 만족하는 경우에만 아래 분기를 실행합니다.
        if validated:
            # 현재 가동 범위 설정의 변경 번호에 `1`을 더해 누적합니다. 대상: `self.threshold_version`.
            self.threshold_version += 1
            # `validated.items()`에서 항목을 차례로 꺼내 `(side, values)`에 넣고 아래 처리를 반복합니다.
            for side, values in validated.items():
                # 한쪽 동작의 반복 판정 상태 기계에 `side == 'left'`이면 `self.fsm_left`, 아니면 `self.fsm_right`을 저장합니다. 대상: `fsm`.
                fsm = self.fsm_left if side == "left" else self.fsm_right
                # `fsm.replace_thresholds(**values, version=self.threshold_version)`을 호출하여 해당 단계의 작업을 수행합니다.
                fsm.replace_thresholds(**values, version=self.threshold_version)
        # 현재 함수 처리를 끝내고 `list(validated)` 호출 결과를 호출한 쪽으로 반환합니다.
        return list(validated)

    # calculate 각도 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _calculate_angle(self, p1: dict, p2: dict, p3: dict) -> float:
        # `v1` 값에 `np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])` 호출 결과를 저장합니다. 대상: `v1`.
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
        # `v2` 값에 `np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])` 호출 결과를 저장합니다. 대상: `v2`.
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

        # `(norm_v1, norm_v2)` 값에 순서·구성을 지정한 `(np.linalg.norm(v1), np.linalg.norm(v2))`을 저장합니다. 대상: `(norm_v1, norm_v2)`.
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        # 조건 `norm_v1 == 0 or norm_v2 == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if norm_v1 == 0 or norm_v2 == 0:
            # 현재 함수 처리를 끝내고 0.0을 호출한 쪽으로 반환합니다.
            return 0.0

        # `cosine` 값에 지정한 최솟값·최댓값 안으로 제한한 값(`np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)`)을 저장합니다. 대상: `cosine`.
        cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        # 현재 함수 처리를 끝내고 `round(math.degrees(np.arccos(cosine)), 2)` 호출 결과를 호출한 쪽으로 반환합니다.
        return round(math.degrees(np.arccos(cosine)), 2)

    # `_valid_points` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _valid_points(self, ids, kpt_map, confidence=.35):
        # 계산에 사용할 관절 또는 좌표 묶음에 반복·조건식 `[kpt_map.get(i) for i in ids]`으로 구성한 값을 저장합니다. 대상: `points`.
        points = [kpt_map.get(i) for i in ids]
        # 조건 `not points or not all((p and p.get('score', 0.0) >= confidence and all((math.isfinite(p[axis]) for axis in ('x', 'y'))) for p in points))`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not points or not all(p and p.get('score', 0.) >= confidence
                # 앞줄에서 시작한 식·인수·자료 구조를 `and all(math.isfinite(p[axis]) for axis in ('x', 'y')) for p in points):` 내용으로 이어서 구성합니다.
                and all(math.isfinite(p[axis]) for axis in ('x', 'y')) for p in points):
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 현재 함수 처리를 끝내고 계산에 사용할 관절 또는 좌표 묶음(`points`)을 호출한 쪽으로 반환합니다.
        return points

    # 설정된 세 관절로 가운데 관절의 각도를 계산합니다. 필요한 관절 신뢰도가 낮으면 None입니다.
    def _compute_angle(self, side, kpt_map):
        # `ids` 값에 `self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])` 호출 결과를 저장합니다. 대상: `ids`.
        ids = self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])
        # 계산에 사용할 관절 또는 좌표 묶음에 `self._valid_points(ids, kpt_map, 0.4)` 호출 결과를 저장합니다. 대상: `points`.
        points = self._valid_points(ids, kpt_map, .4)
        # 조건 `points and len(points) == 3`을 만족하는 경우에만 아래 분기를 실행합니다.
        if points and len(points) == 3:
            # 현재 함수 처리를 끝내고 `self._calculate_angle(*points)` 호출 결과를 호출한 쪽으로 반환합니다.
            return self._calculate_angle(*points)
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return None

    # 설정 축의 관절 위치를 구합니다. 기준 관절이 지정되어 있으면 두 관절 사이의 좌표 차이를 사용합니다.
    def _compute_coordinate(self, side, kpt_map):
        # `cfg` 값에 `self.config.get('eval_config', {})` 호출 결과를 저장합니다. 대상: `cfg`.
        cfg = self.config.get('eval_config', {})
        # `axis` 값에 `cfg.get('axis', 'y')` 호출 결과를 저장합니다. 대상: `axis`.
        axis = cfg.get('axis', 'y')
        # 조건 `axis not in ('x', 'y')`을 만족하는 경우에만 아래 분기를 실행합니다.
        if axis not in ('x', 'y'):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('COORDINATE axis must be x or y')` 예외를 발생시킵니다.
            raise ValueError('COORDINATE axis must be x or y')
        # 목표에 `cfg.get('target_kpt', {}).get(side)` 호출 결과를 저장합니다. 대상: `target`.
        target = cfg.get('target_kpt', {}).get(side)
        # `reference` 값에 `cfg.get('reference_kpt', {}).get(side)` 호출 결과를 저장합니다. 대상: `reference`.
        reference = cfg.get('reference_kpt', {}).get(side)
        # 기존 두 관절 입력은 첫 점을 기준 관절, 둘째 점을 대상 관절로 해석합니다.
        # 조건 `target is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if target is None:
            # `pair` 값에 `cfg.get('primary_kpts', {}).get(side, [])` 호출 결과를 저장합니다. 대상: `pair`.
            pair = cfg.get('primary_kpts', {}).get(side, [])
            # 조건 `len(pair) != 2`을 만족하는 경우에만 아래 분기를 실행합니다.
            if len(pair) != 2:
                # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                return None
            # `(reference, target)` 값에 `pair` 값(`pair`)을 저장합니다. 대상: `(reference, target)`.
            reference, target = pair
        # 계산에 사용할 관절 또는 좌표 묶음에 `self._valid_points([target] + ([reference] if reference is not None else []), kpt_map)` 호출 결과를 저장합니다. 대상: `points`.
        points = self._valid_points([target] + ([reference] if reference is not None else []), kpt_map)
        # 조건 `not points`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not points:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 현재 함수 처리를 끝내고 `round(float(points[0][axis] - (points[1][axis] if reference is not None else 0.0)), 4)` 호출 결과를 호출한 쪽으로 반환합니다.
        return round(float(points[0][axis] - (points[1][axis] if reference is not None else 0.)), 4)

    # 신체 정규화 좌표에서 두 관절의 유클리드 거리를 구합니다. 픽셀·cm 단위가 아닙니다.
    def _compute_length(self, side, kpt_map):
        # `ids` 값에 `self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])` 호출 결과를 저장합니다. 대상: `ids`.
        ids = self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])
        # 계산에 사용할 관절 또는 좌표 묶음에 `self._valid_points(ids, kpt_map)` 호출 결과를 저장합니다. 대상: `points`.
        points = self._valid_points(ids, kpt_map)
        # 조건 `points and len(points) == 2`을 만족하는 경우에만 아래 분기를 실행합니다.
        if points and len(points) == 2:
            # 입력 좌표는 이미 PoseNormalizer로 신체 정규화를 마친 상태입니다.
            # 현재 함수 처리를 끝내고 `round(math.hypot(points[1]['x'] - points[0]['x'], points[1]['y'] - points[0]['y']), 4)` 호출 결과를 호출한 쪽으로 반환합니다.
            return round(math.hypot(points[1]['x']-points[0]['x'], points[1]['y']-points[0]['y']), 4)
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return None

    # 두 관절 거리의 비율을 구합니다. 분모가 거의 0이거나 필요한 관절이 없으면 None입니다.
    def _compute_ratio(self, side, kpt_map):
        # `cfg` 값에 계산식 `self.config['eval_config']`의 결과를 저장합니다. 대상: `cfg`.
        cfg = self.config['eval_config']
        # `numerator` 값에 `self._valid_points(cfg['numerator_kpts'][side], kpt_map)` 호출 결과를 저장합니다. 대상: `numerator`.
        numerator = self._valid_points(cfg['numerator_kpts'][side], kpt_map)
        # `denominator` 값에 `self._valid_points(cfg['denominator_kpts'][side], kpt_map)` 호출 결과를 저장합니다. 대상: `denominator`.
        denominator = self._valid_points(cfg['denominator_kpts'][side], kpt_map)
        # 조건 `not numerator or not denominator`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not numerator or not denominator:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # `distance` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def distance(points):
            # 현재 함수 처리를 끝내고 x·y 차이로 구한 유클리드 거리(`math.hypot(points[1]['x'] - points[0]['x'], points[1]['y'] - points[0]['y'])`)을 호출한 쪽으로 반환합니다.
            return math.hypot(points[1]['x'] - points[0]['x'],
                              # 앞줄에서 시작한 식·인수·자료 구조를 `points[1]['y'] - points[0]['y'])` 내용으로 이어서 구성합니다.
                              points[1]['y'] - points[0]['y'])
        # `bottom` 값에 `distance(denominator)` 호출 결과를 저장합니다. 대상: `bottom`.
        bottom = distance(denominator)
        # 조건 `not math.isfinite(bottom) or bottom <= 1e-06`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(bottom) or bottom <= 1e-6:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 반올림 전 거리로 나눕니다. 분자·분모의 공통 몸통 정규화 배율은 서로 상쇄됩니다.
        # 비율에 계산식 `distance(numerator) / bottom`의 결과를 저장합니다. 대상: `ratio`.
        ratio = distance(numerator) / bottom
        # 현재 함수 처리를 끝내고 `math.isfinite(ratio)`이면 `round(ratio, 4)`, 아니면 `None`을 호출한 쪽으로 반환합니다.
        return round(ratio, 4) if math.isfinite(ratio) else None

    # 계산 side 측정값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _compute_side_value(self, side: str, kpt_map: dict):
        # 현재 함수 처리를 끝내고 `{'ANGLE': self._compute_angle, 'COORDINATE': self._compute_coordinate, 'LENGTH': self._compute_length, 'RATIO': self._compute_ratio}[self.metric_type](side, kpt_map)` 호출 결과를 호출한 쪽으로 반환합니다.
        return {'ANGLE': self._compute_angle, 'COORDINATE': self._compute_coordinate,
                # 구성 항목: 길이(`'LENGTH'`)에 계산 길이(`self._compute_length`); 비율(`'RATIO'`)에 계산 비율(`self._compute_ratio`)을 지정합니다.
                'LENGTH': self._compute_length, 'RATIO': self._compute_ratio}[self.metric_type](side, kpt_map)

    # 관절 ID별 조회 표를 만든 뒤 좌우 측정값과 각 상태 기계의 반복·품질 결과를 함께 반환합니다.
    def process_keypoints(self, keypoints: list) -> dict:
        # 조건 `not keypoints`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not keypoints:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None

        # 관절 ID로 조회하는 관절 또는 신뢰도 사전에 반복·조건식 `{kp['id']: kp for kp in keypoints}`으로 구성한 값을 저장합니다. 대상: `kpt_map`.
        kpt_map = {kp["id"]: kp for kp in keypoints}

        # val 왼쪽에 `self._compute_side_value('left', kpt_map)` 호출 결과를 저장합니다. 대상: `val_left`.
        val_left = self._compute_side_value("left", kpt_map)
        # val 오른쪽에 `self._compute_side_value('right', kpt_map)` 호출 결과를 저장합니다. 대상: `val_right`.
        val_right = self._compute_side_value("right", kpt_map)

        # res 왼쪽에 `self.fsm_left.update(val_left)` 호출 결과를 저장합니다. 대상: `res_left`.
        res_left = self.fsm_left.update(val_left)
        # res 오른쪽에 `self.fsm_right.update(val_right)` 호출 결과를 저장합니다. 대상: `res_right`.
        res_right = self.fsm_right.update(val_right)

        # 현재 함수 처리를 끝내고 키 'mode', 'left', 'right'로 구성한 사전을 호출한 쪽으로 반환합니다.
        return {
            # 구성 항목: 실행 모드(`'mode'`)에 실행 모드(`self.mode`)을 지정합니다.
            "mode": self.mode,
            # 구성 항목: 왼쪽(`'left'`)에 키 'val'로 구성한 사전; `val` 값(`'val'`)에 val 왼쪽(`val_left`)을 지정합니다.
            "left": {**res_left, "val": val_left},
            # 구성 항목: 오른쪽(`'right'`)에 키 'val'로 구성한 사전; `val` 값(`'val'`)에 val 오른쪽(`val_right`)을 지정합니다.
            "right": {**res_right, "val": val_right}
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
