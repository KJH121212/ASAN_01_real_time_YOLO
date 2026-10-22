# 한국어 코드 해설: 입력 시각 간격을 반영하는 One Euro Filter로 관절의 x·y 흔들림을 줄이고, 결측 관절의 오래된 필터 이력을 버립니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [모듈 정보]
# 파일: utils/filters.py
# 설명: 입력 시각을 반영한 One Euro 평활화와 기존 필터 호환 기능입니다.
# ==============================================================================

# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time



# `OneEuroFilter` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class OneEuroFilter:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """Adaptive low-pass filter for one scalar; timestamps are in seconds.

    Algorithm: https://github.com/casiez/OneEuroFilter
    Lower min_cutoff reduces resting jitter; higher beta reduces motion lag.
    """

    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0, max_gap=0.5):
        # `params` 값에 순서·구성을 지정한 `(min_cutoff, beta, d_cutoff, max_gap)`을 저장합니다. 대상: `params`.
        params = (min_cutoff, beta, d_cutoff, max_gap)
        # 조건 `not all((math.isfinite(p) for p in params))`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not all(math.isfinite(p) for p in params):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Filter parameters must be finite')` 예외를 발생시킵니다.
            raise ValueError("Filter parameters must be finite")
        # 조건 `min_cutoff <= 0 or beta < 0 or d_cutoff <= 0 or (max_gap <= 0)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if min_cutoff <= 0 or beta < 0 or d_cutoff <= 0 or max_gap <= 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Cutoffs/max_gap must be positive and beta nonnegative')` 예외를 발생시킵니다.
            raise ValueError("Cutoffs/max_gap must be positive and beta nonnegative")
        # 정지 상태에서 사용할 최소 차단 주파수(Hz)에 정지 상태에서 사용할 최소 차단 주파수(Hz)(`min_cutoff`)을 저장합니다. 대상: `self.min_cutoff`.
        self.min_cutoff = min_cutoff
        # 움직임 속도에 따른 차단 주파수 증가 계수에 움직임 속도에 따른 차단 주파수 증가 계수(`beta`)을 저장합니다. 대상: `self.beta`.
        self.beta = beta
        # 속도 추정 신호의 차단 주파수(Hz)에 속도 추정 신호의 차단 주파수(Hz)(`d_cutoff`)을 저장합니다. 대상: `self.d_cutoff`.
        self.d_cutoff = d_cutoff
        # 필터 이력을 이어 사용할 최대 입력 간격(초)에 필터 이력을 이어 사용할 최대 입력 간격(초)(`max_gap`)을 저장합니다. 대상: `self.max_gap`.
        self.max_gap = max_gap
        # `self.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.reset()

    # 초기화 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def reset(self):
        # 입력 표본 또는 세션 기준 시각(초)에 None을 저장합니다. 대상: `self.timestamp`.
        self.timestamp = None
        # 직전 필터 입력 원본값에 None을 저장합니다. 대상: `self.raw_value`.
        self.raw_value = None
        # 저주파 통과 필터를 적용한 좌표값에 None을 저장합니다. 대상: `self.filtered_value`.
        self.filtered_value = None
        # 평활화한 좌표 변화 속도에 0.0을 저장합니다. 대상: `self.filtered_derivative`.
        self.filtered_derivative = 0.0

    # 데코레이터 `staticmethod`를 바로 아래 함수·클래스에 적용합니다.
    @staticmethod
    # 차단 주파수와 실제 표본 간격으로 저주파 통과 필터의 새 값 반영 비율을 계산합니다.
    def _alpha(cutoff, dt):
        # 차단 주파수(Hz)를 각주파수로 바꾸고 실제 프레임 간격(초)을 곱합니다. 고정 30 FPS를 가정하지 않습니다.
        r = 2.0 * math.pi * cutoff * dt
        # 새 표본의 반영 계수 alpha=r/(1+r)를 반환합니다. 차단 주파수나 시간 간격이 클수록 새 값을 더 빠르게 따라갑니다.
        return r / (1.0 + r)

    # 새 표본을 필터에 반영합니다. 실제 시간 간격을 사용하고 입력 단절 시 오래된 표본을 이어 쓰지 않습니다.
    def update(self, value, timestamp):
        # 조건 `not math.isfinite(value) or not math.isfinite(timestamp)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(value) or not math.isfinite(timestamp):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Value and timestamp must be finite')` 예외를 발생시킵니다.
            raise ValueError("Value and timestamp must be finite")
        # 직전 시각과 현재 시각 사이의 간격(초)에 `self.timestamp is not None`이면 `timestamp - self.timestamp`, 아니면 `None`을 저장합니다. 대상: `dt`.
        dt = timestamp - self.timestamp if self.timestamp is not None else None
        # 조건 `dt is None or dt <= 0 or dt > self.max_gap`을 만족하는 경우에만 아래 분기를 실행합니다.
        if dt is None or dt <= 0 or dt > self.max_gap:
            # 새 입력이나 시간 단절 뒤에는 오래된 자세를 이어서 보간하지 않습니다.
            # 저주파 통과 필터를 적용한 좌표값에 측정값(`value`)을 저장합니다. 대상: `self.filtered_value`.
            self.filtered_value = value
            # 평활화한 좌표 변화 속도에 0.0을 저장합니다. 대상: `self.filtered_derivative`.
            self.filtered_derivative = 0.0
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # 직전 평활화 결과가 아닌 직전 원본 표본과의 차이를 실제 경과 시간으로 나누어 속도를 추정합니다.
            derivative = (value - self.raw_value) / dt
            # 속도 평활화에 사용하는 반영 비율에 `self._alpha(self.d_cutoff, dt)` 호출 결과를 저장합니다. 대상: `a_d`.
            a_d = self._alpha(self.d_cutoff, dt)
            # 속도 자체의 잡음을 먼저 저주파 통과 필터로 줄입니다. 순간 검출 잡음 때문에 차단 주파수가 불필요하게 커지는 것을 줄입니다.
            self.filtered_derivative += a_d * (derivative - self.filtered_derivative)
            # 정지 시에는 낮은 차단 주파수로 흔들림을 줄이고, 빠르게 움직일수록 차단 주파수를 높여 좌표 반응 지연을 줄입니다.
            cutoff = self.min_cutoff + self.beta * abs(self.filtered_derivative)
            # `a` 값에 `self._alpha(cutoff, dt)` 호출 결과를 저장합니다. 대상: `a`.
            a = self._alpha(cutoff, dt)
            # 이전 평활화 좌표에 현재 원본과의 차이 중 alpha만큼을 반영하는 1차 저주파 통과 필터입니다.
            self.filtered_value += a * (value - self.filtered_value)
        # 입력 표본 또는 세션 기준 시각(초)에 입력 표본 또는 세션 기준 시각(초)(`timestamp`)을 저장합니다. 대상: `self.timestamp`.
        self.timestamp = timestamp
        # 직전 필터 입력 원본값에 측정값(`value`)을 저장합니다. 대상: `self.raw_value`.
        self.raw_value = value
        # 현재 함수 처리를 끝내고 저주파 통과 필터를 적용한 좌표값(`self.filtered_value`)을 호출한 쪽으로 반환합니다.
        return self.filtered_value


# `OneEuroKeypointFilter` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class OneEuroKeypointFilter:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """Filter each joint's x/y independently in SkeletonEngine's [0, 1] space.

    Joint identity comes from COCO id, not list position. Scores are preserved;
    missing/low-confidence joints discard history rather than inventing a pose.
    """

    # 영상 정규화 좌표에서 움직임을 빠르게 따라가면서 작은 흔들림을 억제합니다.
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, min_cutoff=3.0, beta=0.5, d_cutoff=1.0,
                 # 이 줄의 입력 인수: `conf_threshold` = 관절 검출 결과를 유효하게 사용할 최소 신뢰도; `max_gap` = 필터 이력을 이어 사용할 최대 입력 간격(초). 기본값·형식은 아래 선언을 따릅니다.
                 conf_threshold=0.35, max_gap=0.5):
        # `self.params` 값에 `dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff, max_gap=max_gap)` 호출 결과를 저장합니다. 대상: `self.params`.
        self.params = dict(min_cutoff=min_cutoff, beta=beta,
                           # 구성 항목: 속도 추정 신호의 차단 주파수(Hz)(`d_cutoff`)에 속도 추정 신호의 차단 주파수(Hz)(`d_cutoff`); 필터 이력을 이어 사용할 최대 입력 간격(초)(`max_gap`)에 필터 이력을 이어 사용할 최대 입력 간격(초)(`max_gap`)을 지정합니다.
                           d_cutoff=d_cutoff, max_gap=max_gap)
        # `OneEuroFilter(**self.params)`을 호출하여 해당 단계의 작업을 수행합니다.
        OneEuroFilter(**self.params)  # Validate even before the first frame.
        # 조건 `not math.isfinite(conf_threshold) or not 0 <= conf_threshold <= 1`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(conf_threshold) or not 0 <= conf_threshold <= 1:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('conf_threshold must be between 0 and 1')` 예외를 발생시킵니다.
            raise ValueError("conf_threshold must be between 0 and 1")
        # 관절 검출 결과를 유효하게 사용할 최소 신뢰도에 관절 검출 결과를 유효하게 사용할 최소 신뢰도(`conf_threshold`)을 저장합니다. 대상: `self.conf_threshold`.
        self.conf_threshold = conf_threshold
        # `self.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.reset()

    # 초기화 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def reset(self):
        # 관절 ID와 축별 필터 상태에 빈 사전을 저장합니다. 대상: `self.filters`.
        self.filters = {}

    # 새 표본을 필터에 반영합니다. 실제 시간 간격을 사용하고 입력 단절 시 오래된 표본을 이어 쓰지 않습니다.
    def update(self, current_kpts, timestamp=None):
        # 조건 `not current_kpts`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not current_kpts:
            # `self.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.reset()
            # 현재 함수 처리를 끝내고 현재 프레임의 관절 목록(`current_kpts`)을 호출한 쪽으로 반환합니다.
            return current_kpts
        # 조건 `timestamp is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if timestamp is None:
            # 입력 표본 또는 세션 기준 시각(초)에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `timestamp`.
            timestamp = time.perf_counter()
        # 조건 `not math.isfinite(timestamp)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(timestamp):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Timestamp must be finite')` 예외를 발생시킵니다.
            raise ValueError("Timestamp must be finite")

        # 함수 또는 처리 단계의 결과에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `result`.
        result = []
        # 이번 프레임에서 유효한 관절 ID 집합에 `set()` 호출 결과를 저장합니다. 대상: `valid_ids`.
        valid_ids = set()
        # `current_kpts`에서 항목을 차례로 꺼내 `kp`에 넣고 아래 처리를 반복합니다.
        for kp in current_kpts:
            # 하나의 관절 좌표 및 신뢰도에 `dict(kp)` 호출 결과를 저장합니다. 대상: `point`.
            point = dict(kp)
            # COCO 관절 식별 번호에 계산식 `kp['id']`의 결과를 저장합니다. 대상: `joint_id`.
            joint_id = kp["id"]
            # 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)에 `kp.get('score', 0.0)` 호출 결과를 저장합니다. 대상: `score`.
            score = kp.get("score", 0.0)
            # `finite` 값에 `all((math.isfinite(kp[axis]) for axis in ('x', 'y')))` 호출 결과를 저장합니다. 대상: `finite`.
            finite = all(math.isfinite(kp[axis]) for axis in ("x", "y"))
            # 조건 `not finite or not math.isfinite(score)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not finite or not math.isfinite(score):
                # `point.update(x=0.0, y=0.0, score=0.0)`을 호출하여 해당 단계의 작업을 수행합니다.
                point.update(x=0.0, y=0.0, score=0.0)
            # 조건 `score >= self.conf_threshold`을 만족하는 경우에만 아래 분기를 실행합니다.
            elif score >= self.conf_threshold:
                # `valid_ids.add(joint_id)`을 호출하여 집합에 항목을 등록합니다.
                valid_ids.add(joint_id)
                # 조건 `joint_id not in self.filters`을 만족하는 경우에만 아래 분기를 실행합니다.
                if joint_id not in self.filters:
                    # `self.filters[joint_id]` 값에 순서·구성을 지정한 `(OneEuroFilter(**self.params), OneEuroFilter(**self.params))`을 저장합니다. 대상: `self.filters[joint_id]`.
                    self.filters[joint_id] = (
                        # 구성 항목: 키워드 인수 펼침(`키워드 인수 펼침`)에 `self.params` 값(`self.params`); 키워드 인수 펼침(`키워드 인수 펼침`)에 `self.params` 값(`self.params`)을 지정합니다.
                        OneEuroFilter(**self.params), OneEuroFilter(**self.params)
                    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                    )
                # `(fx, fy)` 값에 계산식 `self.filters[joint_id]`의 결과를 저장합니다. 대상: `(fx, fy)`.
                fx, fy = self.filters[joint_id]
                # `point['x']` 값에 `fx.update(kp['x'], timestamp)` 호출 결과를 저장합니다. 대상: `point['x']`.
                point["x"] = fx.update(kp["x"], timestamp)
                # `point['y']` 값에 `fy.update(kp['y'], timestamp)` 호출 결과를 저장합니다. 대상: `point['y']`.
                point["y"] = fy.update(kp["y"], timestamp)
            # `result.append(point)`을 호출하여 항목 하나를 끝에 추가합니다.
            result.append(point)

        # 관절 ID와 축별 필터 상태에 반복·조건식 `{i: f for (i, f) in self.filters.items() if i in valid_ids}`으로 구성한 값을 저장합니다. 대상: `self.filters`.
        self.filters = {i: f for i, f in self.filters.items() if i in valid_ids}
        # 현재 함수 처리를 끝내고 함수 또는 처리 단계의 결과(`result`)을 호출한 쪽으로 반환합니다.
        return result
