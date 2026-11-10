# 설명: 입력 시각을 반영한 One Euro 평활화와 기존 필터 호환 기능입니다.

import math
import time

class OneEuroFilter:
    """Adaptive low-pass filter for one scalar; timestamps are in seconds.

    Algorithm: https://github.com/casiez/OneEuroFilter
    Lower min_cutoff reduces resting jitter; higher beta reduces motion lag.
    """

    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0, max_gap=0.5):
        params = (min_cutoff, beta, d_cutoff, max_gap)
        if not all(math.isfinite(p) for p in params):
            raise ValueError("Filter parameters must be finite")
        if min_cutoff <= 0 or beta < 0 or d_cutoff <= 0 or max_gap <= 0:
            raise ValueError("Cutoffs/max_gap must be positive and beta nonnegative")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.timestamp = None
        self.raw_value = None
        self.filtered_value = None
        self.filtered_derivative = 0.0

    @staticmethod
    # 차단 주파수와 실제 표본 간격으로 저주파 통과 필터의 새 값 반영 비율을 계산합니다.
    def _alpha(cutoff, dt):
        # 차단 주파수(Hz)를 각주파수로 바꾸고 실제 프레임 간격(초)을 곱합니다. 고정 30 FPS를 가정하지 않습니다.
        r = 2.0 * math.pi * cutoff * dt
        # 새 표본의 반영 계수 alpha=r/(1+r)를 반환합니다. 차단 주파수나 시간 간격이 클수록 새 값을 더 빠르게 따라갑니다.
        return r / (1.0 + r)

    # 새 표본을 필터에 반영합니다. 실제 시간 간격을 사용하고 입력 단절 시 오래된 표본을 이어 쓰지 않습니다.
    def update(self, value, timestamp):
        if not math.isfinite(value) or not math.isfinite(timestamp):
            raise ValueError("Value and timestamp must be finite")
        dt = timestamp - self.timestamp if self.timestamp is not None else None
        if dt is None or dt <= 0 or dt > self.max_gap:
            # 새 입력이나 시간 단절 뒤에는 오래된 자세를 이어서 보간하지 않습니다.
            self.filtered_value = value
            self.filtered_derivative = 0.0
        else:
            # 직전 평활화 결과가 아닌 직전 원본 표본과의 차이를 실제 경과 시간으로 나누어 속도를 추정합니다.
            derivative = (value - self.raw_value) / dt
            a_d = self._alpha(self.d_cutoff, dt)
            # 속도 자체의 잡음을 먼저 저주파 통과 필터로 줄입니다. 순간 검출 잡음 때문에 차단 주파수가 불필요하게 커지는 것을 줄입니다.
            self.filtered_derivative += a_d * (derivative - self.filtered_derivative)
            # 정지 시에는 낮은 차단 주파수로 흔들림을 줄이고, 빠르게 움직일수록 차단 주파수를 높여 좌표 반응 지연을 줄입니다.
            cutoff = self.min_cutoff + self.beta * abs(self.filtered_derivative)
            a = self._alpha(cutoff, dt)
            # 이전 평활화 좌표에 현재 원본과의 차이 중 alpha만큼을 반영하는 1차 저주파 통과 필터입니다.
            self.filtered_value += a * (value - self.filtered_value)
        self.timestamp = timestamp
        self.raw_value = value
        return self.filtered_value

class OneEuroKeypointFilter:
    """Filter each joint's x/y independently in SkeletonEngine's [0, 1] space.

    Joint identity comes from COCO id, not list position. Scores are preserved;
    missing/low-confidence joints discard history rather than inventing a pose.
    """

    # 영상 정규화 좌표에서 움직임을 빠르게 따라가면서 작은 흔들림을 억제합니다.
    def __init__(self, min_cutoff=15.0, beta=2.0, d_cutoff=1.0,
                 conf_threshold=0.35, max_gap=0.5):
        self.params = dict(min_cutoff=min_cutoff, beta=beta,
                           d_cutoff=d_cutoff, max_gap=max_gap)
        OneEuroFilter(**self.params)  # Validate even before the first frame.
        if not math.isfinite(conf_threshold) or not 0 <= conf_threshold <= 1:
            raise ValueError("conf_threshold must be between 0 and 1")
        self.conf_threshold = conf_threshold
        self.reset()

    def reset(self):
        self.filters = {}

    # 새 표본을 필터에 반영합니다. 실제 시간 간격을 사용하고 입력 단절 시 오래된 표본을 이어 쓰지 않습니다.
    def update(self, current_kpts, timestamp=None):
        if not current_kpts:
            self.reset()
            return current_kpts
        if timestamp is None:
            timestamp = time.perf_counter()
        if not math.isfinite(timestamp):
            raise ValueError("Timestamp must be finite")

        result = []
        valid_ids = set()
        for kp in current_kpts:
            point = dict(kp)
            joint_id = kp["id"]
            score = kp.get("score", 0.0)
            finite = all(math.isfinite(kp[axis]) for axis in ("x", "y"))
            if not finite or not math.isfinite(score):
                point.update(x=0.0, y=0.0, score=0.0)
            elif score >= self.conf_threshold:
                valid_ids.add(joint_id)
                if joint_id not in self.filters:
                    self.filters[joint_id] = (
                        OneEuroFilter(**self.params), OneEuroFilter(**self.params)
                    )
                fx, fy = self.filters[joint_id]
                point["x"] = fx.update(kp["x"], timestamp)
                point["y"] = fy.update(kp["y"], timestamp)
            result.append(point)

        self.filters = {i: f for i, f in self.filters.items() if i in valid_ids}
        return result
