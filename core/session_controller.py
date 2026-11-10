# 설명: 양측(Left & Right) 모두 3회 완수 기반 캘리브레이션 및 세션 제어기

import os
from collections import deque
import math
from pathlib import Path
import sys
import time
import numpy as np

current_file_path = Path(__file__).resolve()
core_dir = current_file_path.parent
project_root = core_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.normalization import PoseNormalizer
from utils.measurement_progress import (
    measurement_progress, CALIBRATION_ACTIVATION_FRACTION, CALIBRATION_REARM_FRACTION,
)

class CalibrationCycleDetector:
    ACTIVATION_FRACTION = CALIBRATION_ACTIVATION_FRACTION
    REARM_FRACTION = CALIBRATION_REARM_FRACTION
    """Shared calibration cycle logic for angle, coordinate and length signals."""
    def __init__(self, rest_seconds=0.5, min_rise=0.08, noise_multiplier=6.0,
                 hold_seconds=0.15, return_seconds=0.15, max_cycle_seconds=10.0,
                 baseline_range=0.04, metric_type='COORDINATE', direction='INCREASING',
                 min_baseline_samples=3, max_sample_gap=1.0, occlusion_reset_seconds=2.0,
                 return_fraction=0.25, consistency_ratio=2.5, start_val=None, target_val=None,
                 activation_fraction=CALIBRATION_ACTIVATION_FRACTION, rearm_fraction=CALIBRATION_REARM_FRACTION,
                 finish_on_return=False, activation_hold_seconds=0.0, restart_on_missing=False):
        if not math.isfinite(activation_hold_seconds) or activation_hold_seconds < 0:
            raise ValueError('activation_hold_seconds must be finite and nonnegative')
        self.activation_hold_seconds = activation_hold_seconds
        self.restart_on_missing = restart_on_missing
        if not 0 < rearm_fraction < activation_fraction <= 1:
            raise ValueError('Calibration requires 0 < rearm_fraction < activation_fraction <= 1')
        self.ACTIVATION_FRACTION, self.REARM_FRACTION = activation_fraction, rearm_fraction
        self.finish_on_return = finish_on_return
        self.fixed_range = None
        if start_val is not None or target_val is not None:
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start_val, target_val)):
                raise ValueError('Calibration start_val and target_val must be finite numbers')
            if direction not in ('INCREASING', 'DECREASING') or (target_val-start_val)*(1 if direction == 'INCREASING' else -1) <= 0:
                raise ValueError('Calibration thresholds must match motion_direction')
            self.fixed_range = (start_val, target_val)
        self.metric_type = metric_type
        self.configured_direction = direction
        self.rest_seconds = rest_seconds
        self.min_rise = min_rise
        self.noise_multiplier = noise_multiplier
        self.hold_seconds = hold_seconds
        self.return_seconds = return_seconds
        self.max_cycle_seconds = max_cycle_seconds
        self.baseline_range = baseline_range
        self.min_baseline_samples = min_baseline_samples
        self.max_sample_gap = max_sample_gap
        self.occlusion_reset_seconds = occlusion_reset_seconds
        self.return_fraction = return_fraction
        self.consistency_ratio = consistency_ratio
        if isinstance(min_baseline_samples, bool) or not isinstance(min_baseline_samples, int) or min_baseline_samples < 3:
            raise ValueError('min_baseline_samples must be an integer >= 3')
        if not (0 < return_fraction < .5) or consistency_ratio < 1:
            raise ValueError('return_fraction must be between 0 and .5; consistency_ratio must be >= 1')
        if direction not in {'INCREASING', 'DECREASING', 'AUTO'}:
            raise ValueError('Invalid calibration direction')
        if not all(math.isfinite(v) and v > 0 for v in (rest_seconds, min_rise, noise_multiplier,
                   hold_seconds, return_seconds, max_cycle_seconds, baseline_range,
                   max_sample_gap, occlusion_reset_seconds, return_fraction, consistency_ratio)):
            raise ValueError('Calibration parameters must be finite and positive')
        self.reset()

    def reset(self, keep_cycles=False):
        completed = self.peaks if keep_cycles else []
        self.samples = deque()
        self.direction = self.configured_direction
        self.candidate_direction = None
        self.velocity = 0.0
        self.previous_value = None
        self.reversing = False
        self.baseline = None
        self.baseline_at = None
        self.threshold = abs(self.fixed_range[1]-self.fixed_range[0]) if self.fixed_range is not None else self.min_rise
        self.state = 'BASELINE'
        self.peaks = completed
        self.started = None
        self.returned = None
        self.peak = 0.0
        self.last_time = None
        self.missing_since = None
        self.ready_floor = 0.0
        self.armed = False
        self.activation_since = None

    @property
    def ready(self):
        return self.baseline is not None

    @property
    # 최근 완료 동작 세 개가 존재하고 최대 변화량의 편차가 허용 배수 안인지 확인합니다.
    def consistent(self):
        return len(self.peaks) >= 3 and (not self.finish_on_return or self.started is None and self.state == 'READY')

    @property
    # 안정 구간 흔들림·시작 기준·현재 최고점에서 복귀 허용 폭을 계산하되 최고 변화량의 절반 이하로 제한합니다.
    def return_tolerance(self):
        return self.threshold * self.REARM_FRACTION

    def status(self):
        state = ('MISSING' if self.missing_since is not None else
                 'COMPLETE' if self.consistent else
                 'NEED_CONSISTENT_CYCLES' if len(self.peaks) >= 3 and self.state == 'READY' else self.state)
        return dict(state=state, baseline=self.baseline, baseline_samples=len(self.samples),
                    threshold_source='CONFIG' if self.fixed_range is not None else 'ADAPTIVE',
                    min_baseline_samples=self.min_baseline_samples, rest_seconds=self.rest_seconds,
                    min_movement=self.threshold * self.ACTIVATION_FRACTION, return_tolerance=self.return_tolerance,
                    activation_fraction=self.ACTIVATION_FRACTION,
                    activation_value=(self.baseline + (-1 if self.direction == 'DECREASING' else 1)
                                      * self.threshold * self.ACTIVATION_FRACTION) if self.ready else None,
                    count_on='TARGET_CROSSING',
                    completed_cycles=len(self.peaks), consistent=self.consistent,
                    direction=self.direction)

    # 측정값 하나를 받아 안정 자세→상승 유지→복귀 유지를 진행합니다. 결측·시간 단절·미완료 동작은 횟수로 인정하지 않습니다.
    def update(self, value, now, allow_count=True):
        updated = False
        if value is None or not math.isfinite(value):
            missing_since = self.missing_since if self.missing_since is not None else now
            if now - missing_since >= self.occlusion_reset_seconds:
                self.reset(keep_cycles=not self.restart_on_missing)
            self.missing_since = missing_since
            # 결측 구간은 상승·복귀 유지 시간에 포함하지 않으며 안정 자세 구간을 이어 붙이지 않습니다.
            self.samples.clear()
            self.armed = False
            self.activation_since = None
            self.state = 'RETURNING' if self.ready else 'BASELINE'
            self.started = self.returned = None
            self.previous_value = None
            self.last_time = now
            return updated
        if self.missing_since is not None and now - self.missing_since >= self.occlusion_reset_seconds:
            self.reset(keep_cycles=not self.restart_on_missing)
        self.missing_since = None
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > self.occlusion_reset_seconds):
            self.reset(keep_cycles=not self.restart_on_missing)
        if self.last_time is not None and now - self.last_time > self.max_sample_gap:
            self.activation_since = None
            self.samples.clear()
            self.armed = False
            self.state = 'RETURNING' if self.ready else 'BASELINE'
            self.started = self.returned = None
        dt = now - self.last_time if self.last_time is not None else 0.
        self.velocity = ((value - self.previous_value) / dt
                         if self.previous_value is not None and 0 < dt <= self.max_sample_gap else 0.)
        self.previous_value = value
        self.last_time = now
        if not self.ready:
            self.samples.append((now, value))
            # 낮은 FPS에서도 최소 표본 수를 유지하면서 안정 구간 길이는 초 단위로 판단합니다.
            # 조건 `len(self.samples) > self.min_baseline_samples and now - self.samples[1][0] >= self.rest_seconds`이 유지되는 동안 아래 처리를 반복합니다.
            while len(self.samples) > self.min_baseline_samples and now - self.samples[1][0] >= self.rest_seconds:
                self.samples.popleft()
            if len(self.samples) >= self.min_baseline_samples and now - self.samples[0][0] >= self.rest_seconds:
                values = [v for _, v in self.samples]
                center = float(np.median(values))
                # 중앙절대편차(MAD)에 1.4826을 곱해 안정 구간 잡음을 추정합니다. 일부 튀는 표본에 평균·표준편차보다 덜 끌려갑니다.
                noise = 1.4826 * float(np.median(np.abs(np.asarray(values) - center)))
                if np.percentile(values, 95) - np.percentile(values, 5) <= self.baseline_range:
                    self.baseline = center
                    self.baseline_at = now
                    # 최소 변화량과 잡음의 지정 배수 중 큰 값을 동작 시작 기준으로 채택합니다. 환자의 실제 움직임과 검출 흔들림을 구분하기 위한 기준입니다.
                    self.threshold = max(self.min_rise, self.noise_multiplier * noise)
                    if self.fixed_range is not None:
                        self.baseline = self.fixed_range[0]
                        self.threshold = abs(self.fixed_range[1]-self.fixed_range[0])
                    self.state = 'READY'
                    sign = -1 if self.direction == 'DECREASING' else 1
                    self.armed = sign * (value-self.baseline) <= self.threshold * self.REARM_FRACTION
                    if not self.armed:
                        self.state = 'RETURNING'
            return updated
        if not allow_count:
            self.activation_since = None
            return updated
        displacement = value - self.baseline
        if self.direction == 'AUTO' and self.state == 'READY' and abs(displacement) >= self.threshold * self.ACTIVATION_FRACTION:
            self.candidate_direction = 'INCREASING' if displacement > 0 else 'DECREASING'
        direction = self.candidate_direction if self.direction == 'AUTO' else self.direction
        sign = -1. if direction == 'DECREASING' else 1.
        rise = sign * displacement
        if self.state == 'READY' and self.armed:
            if rise >= self.threshold * self.ACTIVATION_FRACTION:
                if self.activation_since is None:
                    self.activation_since = now
                if now - self.activation_since + 1e-9 < self.activation_hold_seconds:
                    return updated
                self.started = now
                self.peak = rise
                self.peaks.append(rise)
                if self.direction == 'AUTO':
                    self.direction = self.candidate_direction
                self.state = 'RETURNING'
                self.armed = False
                self.returned = None
                updated = True
                self.activation_since = None
            else:
                self.activation_since = None
        else:
            # Count on ascent, but keep the actual peak for the learned ROM.
            if self.started is not None and self.peaks:
                self.peak = max(self.peak, rise)
                self.peaks[-1] = self.peak
            if rise <= self.threshold * self.REARM_FRACTION:
                if self.returned is None:
                    self.returned = now
                elif now-self.returned >= self.return_seconds:
                    self.state = 'READY'
                    self.armed = True
                    self.returned = None
                    self.started = None
            else:
                self.returned = None
        return updated

    # 최근 세 동작이 일관될 때 시작값·85% 목표값·측정 방식·방향을 개인 기준값으로 반환합니다.
    def thresholds(self):
        if not self.consistent:
            raise ValueError('Calibration needs three successful movements')
        return {'start_val': round(self.baseline, 4),
                'target_val': round(self.baseline + (-1. if self.direction == 'DECREASING' else 1.) * .85 * float(np.median(self.peaks[-3:])), 4),
                'metric_type': self.metric_type, 'motion_direction': self.direction}

class SessionController:

    def __init__(self, data_manager=None, motion_engine=None):
        self.data_manager = data_manager
        self.motion_engine = motion_engine
        self.target_calib_reps = 3  # 양측 각각 채워야 하는 목표 횟수

        self.mode = motion_engine.mode if motion_engine else "CALIBRATION"
        self.calib_step = "FULL_BODY_CHECK"
        self.timer_start = 0.0

        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)
        self.calibration_detectors = {}
        # Optional MAIN coordinate range learned during another calibration metric.
        self.calibration_coordinates = {
            side: dict(samples=deque(), baseline=None, peaks=[])
            for side in ('left', 'right')
        }
        if motion_engine and motion_engine.mode == 'CALIBRATION':
            defaults = ({'min_rise': 8.0, 'baseline_range': 4.0}
                        if motion_engine.metric_type == 'ANGLE' else {})
            params = {**defaults, **motion_engine.config.get('calibration_config', {})}
            use_config = params.pop('use_config_thresholds', True)
            params.pop('display_range', None)
            self.calibration_detectors = {side: CalibrationCycleDetector(
                **{**params, **(motion_engine.config['default_thresholds'][side] if use_config else {})}, metric_type=motion_engine.metric_type,
                direction=motion_engine.motion_direction) for side in ('left', 'right')}

        self.calib_vals_left = []
        self.calib_vals_right = []
        self.calib_reps_left = 0
        self.calib_reps_right = 0

        self.last_left_data = {
            "val": 0.0,
            "rep_count": 0,
            "total_score": 0.0, "score_delta": 0.0, "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "CALIBRATING",
        }
        self.last_right_data = {
            "val": 0.0,
            "rep_count": 0,
            "total_score": 0.0, "score_delta": 0.0, "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "CALIBRATING",
        }
        print(f"[DEBUG][CONTROLLER] Initialized successfully. Target Calib Reps per side: {self.target_calib_reps}")

    # 부분 가동 범위 변경 요청을 검증하고 판정 기준을 갱신합니다. 진행 중이던 동작은 새 범위에서 복귀 후 다시 시작합니다.
    def update_count_thresholds(self, thresholds: dict) -> list:
        changed_sides = self.motion_engine.update_count_thresholds(thresholds)
        for side in changed_sides:
            fsm = self.motion_engine.fsm_left if side == "left" else self.motion_engine.fsm_right
            cache = self.last_left_data if side == "left" else self.last_right_data
            value = cache.get("val")
            ratio = measurement_progress(value, self.motion_engine.metric_type, fsm.start_val, fsm.target_val) or 0.
            cache.update(state=fsm.state, rep_count=fsm.rep_count,
                         **fsm.score_status(),
                         progress_ratio=round(min(1.0, max(0.0, ratio)), 2),
                         quality=fsm.last_quality, is_count_updated=False)
        return changed_sides

    # 몸통 기준 관절과 해당 운동에 필요한 관절이 모두 최소 신뢰도를 만족하는지 검사합니다.
    def check_occlusion(self, keypoints):
        points = {p['id']: p for p in keypoints or []}
        def visible(j):
            return j in points and points[j].get('score', 0) >= .35 and self._in_frame(points[j])
        cfg = self.motion_engine.config['eval_config']
        required = []
        for side in ('left', 'right'):
            ids = set(cfg.get('primary_kpts', {}).get(side, []))
            for field in ('target_kpt', 'reference_kpt'):
                if cfg.get(field, {}).get(side) is not None:
                    ids.add(cfg[field][side])
            if self.motion_engine.use_normalization:
                ids.update((5, 6, 11, 12))
            required.append(ids)
        if self.motion_engine.coordinate_unit == 'pixel' or self.motion_engine.counting_method == 'TARGET_POSITION':
            return not any(ids and all(visible(j) for j in ids) for ids in required)
        return not points or not all(visible(j) for ids in required for j in ids)

    # 양쪽 관절 측정값으로 안정 자세·상승·복귀를 추적하고 최근 세 동작의 일관성과 완료 상태를 갱신합니다.
    def _collect_main_coordinate(self, side, keypoints, detector, now):
        cfg = self.motion_engine.config.get('main_eval_config', {})
        if not cfg or cfg.get('metric_type') != 'COORDINATE':
            return
        if cfg.get('use_normalization', True) or cfg.get('reference_kpt'):
            raise ValueError('Calibration output requires an unnormalized target coordinate')
        points = {p['id']: p for p in keypoints or []
                  if self._in_frame(p) and p.get('score', 0.) >= .35}
        point = points.get(cfg['target_kpt'][side])
        value = point[cfg.get('axis', 'y')] if point else None
        record = self.calibration_coordinates[side]
        if not detector.ready:
            record['baseline'] = None
            record['peaks'] = record['peaks'][:len(detector.peaks)]
            if value is None:
                record['samples'].clear()
                return
            record['samples'].append((now, value))
            while record['samples'] and now-record['samples'][0][0] > detector.rest_seconds:
                record['samples'].popleft()
        elif record['baseline'] is None and value is not None:
            samples = [v for _, v in record['samples']] + [value]
            record['baseline'] = float(np.median(samples))
        if detector.started is not None and value is not None:
            if len(record['peaks']) < len(detector.peaks):
                record['peaks'].append(value)
            select = min if cfg.get('motion_direction') == 'DECREASING' else max
            record['peaks'][-1] = select(record['peaks'][-1], value)

    def process_calibration_frame(self, keypoints, is_occluded, now=None):
        now = time.time() if now is None else now
        detectors = self.calibration_detectors
        normalized = self.measurement_points(keypoints) if keypoints and not is_occluded else None
        is_occluded = is_occluded or not normalized
        points = {p['id']: p for p in normalized} if normalized else {}
        # 양쪽 모두 안정 자세를 확보하고 필요한 관절이 보일 때부터 횟수를 인정합니다. 한쪽 준비 중 다른 쪽에서 먼저 잘못 세는 일을 막습니다.
        allow_count = all(d.ready for d in detectors.values()) and not is_occluded
        for side, detector in detectors.items():
            value = self.motion_engine._compute_side_value(side, points)
            updated = detector.update(value, now, allow_count=allow_count and not detector.consistent
                                      and self.motion_engine.count_posture_valid(side, points))
            self._collect_main_coordinate(side, keypoints if not is_occluded else None, detector, now)
            count = len(detector.peaks)
            direction = detector.candidate_direction if detector.direction == 'AUTO' else detector.direction
            sign = -1. if direction == 'DECREASING' else 1.
            cache = dict(val=value, rep_count=count, total_score=0.0, score_delta=0.0, score=0,
                         calibration=detector.status(),
                         state=detector.state, quality='CALIBRATING', is_count_updated=updated,
                         progress_ratio=self.motion_engine.display_progress(side, value) or 0.)
            setattr(self, f'last_{side}_data', cache)
            setattr(self, f'calib_reps_{side}', count)
            fsm = getattr(self.motion_engine, f'fsm_{side}')
            fsm.rep_count = count
            fsm.state = detector.state
            if detector.ready:
                if not self.motion_engine.config.get('calibration_config', {}).get('use_config_thresholds', True):
                    span = self.motion_engine.config['calibration_config'].get('display_range', detector.threshold)
                    if span == 'baseline':
                        span = detector.baseline
                    self.motion_engine.calibration_display_thresholds.setdefault(side, dict(
                        start_val=detector.baseline, target_val=detector.baseline+sign*span))
                fsm.start_val = detector.baseline
                fsm.is_decreasing = direction == 'DECREASING'
                fsm.target_val = detector.baseline + (-1. if fsm.is_decreasing else 1.) * detector.threshold
            if updated:
                endpoint = detector.baseline + (-1. if fsm.is_decreasing else 1.) * detector.peaks[-1]
                fsm.completed_reps_history.append(dict(side=side, rep_num=count, duration_sec=now-detector.started,
                    min_val=min(detector.baseline, endpoint), max_val=max(detector.baseline, endpoint),
                    achieved_rom=detector.peaks[-1], quality='CALIBRATING', score_delta=0., total_score=0.,
                    start_val=fsm.start_val, target_val=fsm.target_val, threshold_version=fsm.threshold_version))
            elif detector.started is not None and fsm.completed_reps_history:
                row = fsm.completed_reps_history[-1]
                endpoint = detector.baseline + sign * detector.peak
                row.update(min_val=min(detector.baseline, endpoint),
                           max_val=max(detector.baseline, endpoint),
                           achieved_rom=detector.peak, duration_sec=now-detector.started)
            if len(fsm.completed_reps_history) > count:
                fsm.completed_reps_history.clear()
        if is_occluded:
            self.calib_step = 'PAUSED' if any(d.ready for d in detectors.values()) else 'FULL_BODY_CHECK'
        else:
            if not all(d.ready for d in detectors.values()):
                self.calib_step = 'COUNTDOWN'
            elif all(d.consistent for d in detectors.values()) and all(
                    cache['val'] is not None for cache in (self.last_left_data, self.last_right_data)):
                if self.calib_step not in ('COOLDOWN', 'FINISHED'):
                    self.calib_step = 'COOLDOWN'
                    self.timer_start = now
                # 양쪽 최근 세 동작의 일관성이 확인된 뒤 1초 대기하여 완료 상태를 확정합니다.
                elif now - self.timer_start >= 1.0:
                    self.calib_step = 'FINISHED'
            else:
                self.calib_step = ('CHECK_CONSISTENCY' if any(len(d.peaks) >= 3 and not d.consistent
                                                            for d in detectors.values()) else 'COLLECTING')
        return dict(calib_step=self.calib_step, calib_rep_count=min(self.calib_reps_left, self.calib_reps_right),
                    calib_reps_left=self.calib_reps_left, calib_reps_right=self.calib_reps_right,
                    target_calib_reps=self.target_calib_reps,
                    calibration={side: detector.status() for side, detector in detectors.items()})

    # 완료된 캘리브레이션의 기준값과 최대 변화량 중앙값의 85%로 개인 가동 범위를 만들어 저장합니다.
    def compute_and_save_thresholds(self, exercise_name):
        if self.calib_step != 'FINISHED':
            raise ValueError('Calibration is incomplete; thresholds were not saved')
        engine = self.motion_engine
        cfg = engine.config['eval_config']
        thresholds = {side: detector.thresholds() for side, detector in self.calibration_detectors.items()}
        main_cfg = engine.config.get('main_eval_config', {})
        if main_cfg:
            fraction = main_cfg.get('calibration_target_fraction', .85)
            if not math.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError('calibration_target_fraction must be in (0, 1]')
            converted = {}
            for side, record in self.calibration_coordinates.items():
                if record['baseline'] is None or len(record['peaks']) < 3:
                    raise ValueError('Calibration needs three coordinate ranges')
                start = record['baseline']
                peak = float(np.median(record['peaks'][-3:]))
                target = start + fraction * (peak-start)
                sign = -1 if main_cfg['motion_direction'] == 'DECREASING' else 1
                if not math.isfinite(target) or sign*(target-start) <= 0:
                    raise ValueError('Coordinate range does not match the MAIN direction; recalibrate')
                converted[side] = dict(
                    start_val=round(start, 6), target_val=round(target, 6),
                    metric_type='COORDINATE', motion_direction=main_cfg['motion_direction'],
                    use_normalization=False, target_kpt=main_cfg['target_kpt'][side],
                    axis=main_cfg.get('axis', 'y'), counting_method=main_cfg.get('counting_method', 'PEAK_REVERSAL'),
                    calibration_peak_val=peak, calibration_target_fraction=fraction,
                    calibration_length_thresholds=thresholds[side])
            thresholds = converted
        for side, values in thresholds.items():
            values['use_normalization'] = engine.use_normalization
            if engine.coordinate_unit == 'pixel':
                if engine.frame_size is None:
                    raise ValueError('Pixel calibration requires image dimensions')
                values.update(coordinate_unit='pixel', axis=cfg.get('axis', 'y'),
                    target_kpt=cfg['target_kpt'][side], reference_kpt=cfg.get('reference_kpt', {}).get(side),
                    frame_size=list(engine.frame_size), counting_method=engine.counting_method)
        if self.data_manager:
            self.data_manager.session_log.info['measurement'] = engine.count_threshold_status()
            self.data_manager.save_custom_threshold(exercise_name, thresholds, is_confirmed=True)
        return thresholds

    # 유효한 관절을 정규화하여 본 운동 엔진에 전달합니다. 게임 종료 결정은 클라이언트가 하므로 종료 플래그는 False입니다.
    @staticmethod
    def _in_frame(point):
        return all(math.isfinite(point.get(axis, math.nan)) and 0 <= point[axis] <= 1 for axis in ('x', 'y'))

    def measurement_points(self, keypoints):
        if not keypoints:
            return None
        keypoints = [p if self._in_frame(p) else {**p, 'score': 0.} for p in keypoints]
        if self.motion_engine.use_normalization:
            return self.normalizer.normalize(keypoints)
        if self.motion_engine.coordinate_unit == 'pixel':
            width, height = keypoints[0].get('image_width'), keypoints[0].get('image_height')
            self.motion_engine.set_frame_size(width, height)
            return [{**p, 'x': p['x']*width, 'y': p['y']*height} for p in keypoints]
        aspect = float(keypoints[0].get('image_aspect_ratio', 1.))
        if not math.isfinite(aspect) or aspect <= 0:
            raise ValueError('image_aspect_ratio must be positive')
        return [{**p, 'x': p['x']*aspect, 'image_aspect_ratio': 1.} for p in keypoints]

    def _missing_frame(self):
        if self.motion_engine.coordinate_unit == 'pixel':
            for fsm in (self.motion_engine.fsm_left, self.motion_engine.fsm_right):
                fsm.state, fsm.min_val_recorded, fsm.max_val_recorded = 'RETURNING', 999., -999.
        missing = [dict(data, val=None, state='MISSING', progress_ratio=0.,
                        score=0, score_delta=0., is_count_updated=False)
                   for data in (self.last_left_data, self.last_right_data)]
        return missing[0], missing[1], False

    def process_main_frame(self, raw_keypoints: list, is_occluded: bool, now: float = None) -> tuple:
        """본 운동 측정과 판정만 처리; 종료 여부는 클라이언트가 결정"""
        if now is None:
            now = time.time()

        if self.motion_engine.counting_method == 'TARGET_POSITION':
            result = self.motion_engine.process_keypoints(self.measurement_points(raw_keypoints), now)
            self.last_left_data, self.last_right_data = result['left'], result['right']
            return self.last_left_data, self.last_right_data, False
        if is_occluded or not raw_keypoints:
            return self._missing_frame()

        # 본 운동도 캘리브레이션과 동일하게 신체 정규화를 적용합니다. 두 과정의 단위를 맞춰 저장한 개인 기준값을 그대로 비교합니다.
        norm_keypoints = self.measurement_points(raw_keypoints)
        if not norm_keypoints:
            return self._missing_frame()

        result = self.motion_engine.process_keypoints(norm_keypoints)
        if not result:
            return self._missing_frame()

        left_data, right_data = result["left"], result["right"]
        self.last_left_data = left_data.copy()
        self.last_right_data = right_data.copy()

        # MAIN는 클라이언트 중지 명령으로 끝납니다. 서버 점수 기준은 종료 조건이 아닙니다.
        return left_data, right_data, False
