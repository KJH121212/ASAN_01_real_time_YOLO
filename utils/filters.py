# ==============================================================================
# [Module Information]
# File: utils/filters.py
# Description: Timestamp-aware One Euro smoothing and the legacy EMA filter.
# ==============================================================================

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
    def _alpha(cutoff, dt):
        r = 2.0 * math.pi * cutoff * dt
        return r / (1.0 + r)

    def update(self, value, timestamp):
        if not math.isfinite(value) or not math.isfinite(timestamp):
            raise ValueError("Value and timestamp must be finite")
        dt = timestamp - self.timestamp if self.timestamp is not None else None
        if dt is None or dt <= 0 or dt > self.max_gap:
            # A new stream or a discontinuity must not interpolate stale poses.
            self.filtered_value = value
            self.filtered_derivative = 0.0
        else:
            derivative = (value - self.raw_value) / dt
            a_d = self._alpha(self.d_cutoff, dt)
            self.filtered_derivative += a_d * (derivative - self.filtered_derivative)
            cutoff = self.min_cutoff + self.beta * abs(self.filtered_derivative)
            a = self._alpha(cutoff, dt)
            self.filtered_value += a * (value - self.filtered_value)
        self.timestamp = timestamp
        self.raw_value = value
        return self.filtered_value


class OneEuroKeypointFilter:
    """Filter each joint's x/y independently in SkeletonEngine's [0, 1] space.

    Joint identity comes from COCO id, not list position. Scores are preserved;
    missing/low-confidence joints discard history rather than inventing a pose.
    """

    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0,
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
