"""Count on reaching a calibrated coordinate, then require a return to rearm."""

import math
import time

from core.motion_engine import SideFSM
from utils.measurement_progress import measurement_progress


class TargetPositionSide(SideFSM):
    RETURN_FRACTION = .15
    MAX_GAP_SECONDS = .5

    def __init__(self, side, direction, thresholds, target_reps, target_score):
        super().__init__(side, direction, thresholds, target_reps, target_score, scoring=True)
        if (not all(math.isfinite(v) for v in (self.start_val, self.target_val))
                or (self.target_val-self.start_val)*(-1 if self.is_decreasing else 1) <= 0):
            raise ValueError('Target coordinate must match the movement direction')
        self.armed = False
        self.last_time = self.started = None
        self.state = 'RETURNING'

    def replace_thresholds(self, start_val, target_val, version):
        super().replace_thresholds(start_val, target_val, version)
        self.armed = False
        self.started = None

    def update(self, value, now=None):
        now = time.monotonic() if now is None else now
        if not math.isfinite(now) or self.last_time is not None and now <= self.last_time:
            raise ValueError('Timestamps must increase')
        if self.last_time is not None and now-self.last_time > self.MAX_GAP_SECONDS:
            self.armed = False
        self.last_time = now
        progress = measurement_progress(value, 'COORDINATE', self.start_val, self.target_val)
        updated = False
        if progress is None:
            self.armed = False
            self.state = 'MISSING'
        elif progress <= self.RETURN_FRACTION:
            if not self.armed:
                self.started = now
                self.min_val_recorded = self.max_val_recorded = value
            self.armed = True
            self.state = 'READY'
        elif self.armed:
            self.min_val_recorded = min(self.min_val_recorded, value)
            self.max_val_recorded = max(self.max_val_recorded, value)
            self.state = 'PUSHING'
            if progress >= 1.:
                self.rep_count += 1
                self.total_score += 1.
                self.last_quality = 'PERFECT'
                self.armed = False
                self.state = 'RETURNING'
                updated = True
                self.completed_reps_history.append(dict(
                    side=self.side, rep_num=self.rep_count, duration_sec=now-self.started,
                    min_val=self.min_val_recorded, max_val=self.max_val_recorded,
                    achieved_rom=self.max_val_recorded-self.min_val_recorded,
                    quality=self.last_quality, score_delta=1., total_score=self.total_score,
                    start_val=self.start_val, target_val=self.target_val,
                    threshold_version=self.threshold_version))
        else:
            self.state = 'RETURNING'
        return dict(val=value, rep_count=self.rep_count, state=self.state, quality=self.last_quality,
                    progress_ratio=progress or 0., is_count_updated=updated,
                    counting_method='TARGET_POSITION', **self.score_status(awarded=updated))
