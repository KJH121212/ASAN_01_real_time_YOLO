# ==============================================================================
# File: core/session_controller.py
# Description: 양측(Left & Right) 모두 3회 완수 기반 캘리브레이션 및 세션 제어기
# ==============================================================================

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


class CalibrationCycleDetector:
    """Shared calibration cycle logic for angle, coordinate, length and ratio signals."""
    def __init__(self, rest_seconds=0.5, min_rise=0.08, noise_multiplier=6.0,
                 hold_seconds=0.15, return_seconds=0.15, max_cycle_seconds=10.0,
                 baseline_range=0.04, metric_type='COORDINATE', direction='INCREASING'):
        self.metric_type = metric_type
        self.configured_direction = direction
        self.rest_seconds = rest_seconds
        self.min_rise = min_rise
        self.noise_multiplier = noise_multiplier
        self.hold_seconds = hold_seconds
        self.return_seconds = return_seconds
        self.max_cycle_seconds = max_cycle_seconds
        self.baseline_range = baseline_range
        if direction not in {'INCREASING', 'DECREASING', 'AUTO'}:
            raise ValueError('Invalid calibration direction')
        if not all(math.isfinite(v) and v > 0 for v in (rest_seconds, min_rise, noise_multiplier,
                   hold_seconds, return_seconds, max_cycle_seconds, baseline_range)):
            raise ValueError('Calibration parameters must be finite and positive')
        self.reset()

    def reset(self):
        self.samples = deque()
        self.direction = self.configured_direction
        self.candidate_direction = None
        self.velocity = 0.0
        self.previous_value = None
        self.reversing = False
        self.baseline = None
        self.baseline_at = None
        self.threshold = self.min_rise
        self.state = 'BASELINE'
        self.peaks = []
        self.started = None
        self.returned = None
        self.peak = 0.0
        self.last_time = None

    @property
    def ready(self):
        return self.baseline is not None

    @property
    def consistent(self):
        return len(self.peaks) >= 3 and max(self.peaks[-3:]) <= 2.5 * min(self.peaks[-3:])

    def update(self, value, now, allow_count=True):
        updated = False
        if value is None or not math.isfinite(value):
            # Missing data cannot complete a held ascent/return or bridge a rest window.
            self.samples.clear()
            self.state = 'READY' if self.ready else 'BASELINE'
            self.started = self.returned = None
            self.previous_value = None
            self.last_time = now
            return updated
        if self.last_time is not None and (now <= self.last_time or now - self.last_time > .5):
            self.samples.clear()
            self.state = 'READY' if self.ready else 'BASELINE'
            self.started = self.returned = None
        dt = now - self.last_time if self.last_time is not None else 0.
        self.velocity = ((value - self.previous_value) / dt
                         if self.previous_value is not None and 0 < dt <= .5 else 0.)
        self.previous_value = value
        self.last_time = now
        if not self.ready:
            self.samples.append((now, value))
            while len(self.samples) > 1 and now - self.samples[1][0] >= self.rest_seconds:
                self.samples.popleft()
            if len(self.samples) >= 15 and now - self.samples[0][0] >= self.rest_seconds:
                values = [v for _, v in self.samples]
                center = float(np.median(values))
                noise = 1.4826 * float(np.median(np.abs(np.asarray(values) - center)))
                if np.percentile(values, 95) - np.percentile(values, 5) <= self.baseline_range:
                    self.baseline = center
                    self.baseline_at = now
                    self.threshold = max(self.min_rise, self.noise_multiplier * noise)
                    self.state = 'READY'
            return updated
        if not allow_count:
            return updated
        displacement = value - self.baseline
        if self.direction == 'AUTO' and self.state == 'READY' and abs(displacement) >= self.threshold:
            self.candidate_direction = 'INCREASING' if displacement > 0 else 'DECREASING'
        direction = self.candidate_direction if self.direction == 'AUTO' else self.direction
        sign = -1. if direction == 'DECREASING' else 1.
        rise = sign * displacement
        speed = sign * self.velocity
        if self.state == 'RETURNING' and speed < -self.threshold * .1:
            self.reversing = True
        if self.state == 'READY':
            if rise >= self.threshold:
                self.reversing = False
                self.state = 'RISING'
                self.started = now
                self.peak = rise
        elif self.state == 'RISING':
            if rise < self.threshold:
                self.state = 'READY'
            else:
                self.peak = max(self.peak, rise)
                if now - self.started >= self.hold_seconds:
                    self.state = 'RETURNING'
        elif self.state == 'RETURNING':
            self.peak = max(self.peak, rise)
            if self.reversing and rise <= self.threshold * .4:
                if self.returned is None:
                    self.returned = now
                elif now - self.returned >= self.return_seconds:
                    self.peaks.append(self.peak)
                    if self.direction == 'AUTO':
                        self.direction = self.candidate_direction
                    self.state = 'READY'
                    self.returned = None
                    updated = True
            else:
                self.returned = None
        if self.state != 'READY' and self.started is not None and now - self.started > self.max_cycle_seconds:
            self.state = 'READY'
            self.returned = None
        return updated

    def thresholds(self):
        if not self.consistent:
            raise ValueError('Calibration needs three consistent movement/return cycles')
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
        if motion_engine:
            defaults = ({'min_rise': 8.0, 'baseline_range': 4.0}
                        if motion_engine.metric_type == 'ANGLE' else {})
            params = {**defaults, **motion_engine.config.get('calibration_config', {})}
            self.calibration_detectors = {side: CalibrationCycleDetector(
                **params, metric_type=motion_engine.metric_type,
                direction=motion_engine.motion_direction) for side in ('left', 'right')}


        self.calib_vals_left = []
        self.calib_vals_right = []
        self.calib_reps_left = 0
        self.calib_reps_right = 0

        # TEST 모드 목표 달성 후 쿨다운 타이머
        self.test_finish_timer = None

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
<<<<<<< HEAD

    def update_count_thresholds(self, thresholds: dict) -> list:
        changed_sides = self.motion_engine.update_count_thresholds(thresholds)
        for side in changed_sides:
            fsm = self.motion_engine.fsm_left if side == "left" else self.motion_engine.fsm_right
            cache = self.last_left_data if side == "left" else self.last_right_data
            value = cache.get("val")
            ratio = abs(value - fsm.start_val) / abs(fsm.target_val - fsm.start_val) if value is not None else 0.0
            cache.update(state=fsm.state, rep_count=fsm.rep_count,
                         **fsm.score_status(),
                         progress_ratio=round(min(1.0, max(0.0, ratio)), 2),
                         quality=fsm.last_quality, is_count_updated=False)
        return changed_sides
=======
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0

    def check_occlusion(self, keypoints: list) -> bool:
        if not keypoints or len(keypoints) < 17:
            return True
        kpt_map = {kp["id"]: kp.get("score", 0.0) for kp in keypoints}
        required_ids = {5, 6, 11, 12}
        cfg = self.motion_engine.config.get('eval_config', {}) if self.motion_engine else {}
        for side in ('left', 'right'):
            required_ids.update(cfg.get('primary_kpts', {}).get(side, []))
            for field in ('numerator_kpts', 'denominator_kpts'):
                required_ids.update(cfg.get(field, {}).get(side, []))
            for field in ('target_kpt', 'reference_kpt'):
                value = cfg.get(field, {}).get(side)
                if value is not None:
                    required_ids.add(value)
        return not all(kpt_map.get(i, 0.0) >= 0.35 for i in required_ids)

<<<<<<< HEAD
    def process_calibration_frame(self, keypoints, is_occluded, now=None):
        now = time.time() if now is None else now
        detectors = self.calibration_detectors
        if is_occluded:
            for detector in detectors.values():
                detector.reset()
            self.calib_step = 'FULL_BODY_CHECK'
        normalized = self.normalizer.normalize(keypoints) if keypoints and not is_occluded else None
        points = {p['id']: p for p in normalized} if normalized else {}
        allow_count = all(d.ready for d in detectors.values()) and not is_occluded
        for side, detector in detectors.items():
            value = self.motion_engine._compute_side_value(side, points)
            updated = detector.update(value, now, allow_count=allow_count and not detector.consistent)
            count = len(detector.peaks)
            direction = detector.candidate_direction if detector.direction == 'AUTO' else detector.direction
            sign = -1. if direction == 'DECREASING' else 1.
            cache = dict(val=value, rep_count=count, total_score=0.0, score_delta=0.0, score=0,
                         state=detector.state, quality='CALIBRATING', is_count_updated=updated,
                         progress_ratio=round(max(0., min(1., (value - detector.baseline) * sign / detector.threshold)), 2)
                         if value is not None and detector.ready else 0.)
            setattr(self, f'last_{side}_data', cache)
            setattr(self, f'calib_reps_{side}', count)
            fsm = getattr(self.motion_engine, f'fsm_{side}')
            fsm.rep_count = count
            fsm.state = detector.state
            if detector.ready:
                fsm.start_val = detector.baseline
                fsm.is_decreasing = direction == 'DECREASING'
                fsm.target_val = detector.baseline + (-1. if fsm.is_decreasing else 1.) * detector.threshold
            if updated:
                endpoint = detector.baseline + (-1. if fsm.is_decreasing else 1.) * detector.peaks[-1]
                fsm.completed_reps_history.append(dict(side=side, rep_num=count, duration_sec=now-detector.started,
                    min_val=min(detector.baseline, endpoint), max_val=max(detector.baseline, endpoint),
                    achieved_rom=detector.peaks[-1], quality='CALIBRATING', score_delta=0., total_score=0.,
                    start_val=fsm.start_val, target_val=fsm.target_val, threshold_version=fsm.threshold_version))
            if is_occluded:
                fsm.completed_reps_history.clear()
        if not is_occluded:
            if not all(d.ready for d in detectors.values()):
                self.calib_step = 'COUNTDOWN'
            elif all(d.consistent for d in detectors.values()) and all(
                    cache['val'] is not None for cache in (self.last_left_data, self.last_right_data)):
                if self.calib_step not in ('COOLDOWN', 'FINISHED'):
                    self.calib_step = 'COOLDOWN'
                    self.timer_start = now
                elif now - self.timer_start >= 1.0:
                    self.calib_step = 'FINISHED'
=======
    def process_calibration_frame(self, raw_keypoints: list, is_occluded: bool, now: float = None) -> dict:
        if now is None:
            now = time.time()

        if is_occluded:
            if self.calib_step not in ["FULL_BODY_CHECK", "FINISHED"]:
                print("[WARN][CALIB] Occlusion detected. Resetting to FULL_BODY_CHECK.")
            self.calib_step = "FULL_BODY_CHECK"
            self.timer_start = 0.0
            self.calib_vals_left.clear()
            self.calib_vals_right.clear()
            self.calib_reps_left = 0
            self.calib_reps_right = 0
            if self.motion_engine:
                self.motion_engine.fsm_left.rep_count = 0
                self.motion_engine.fsm_right.rep_count = 0
                self.motion_engine.fsm_left.state = "READY"
                self.motion_engine.fsm_right.state = "READY"
            return {
                "calib_step": self.calib_step,
                "calib_rep_count": 0,
                "calib_reps_left": 0,
                "calib_reps_right": 0,
                "target_calib_reps": self.target_calib_reps
            }

        # 1. 전신 감지 대기
        if self.calib_step == "FULL_BODY_CHECK":
            self.calib_step = "COUNTDOWN"
            self.timer_start = now
            print("[DEBUG][CALIB] Full body detected. Starting 3-second countdown.")

        # 2. 3초 카운트다운
        elif self.calib_step == "COUNTDOWN":
            elapsed = now - self.timer_start
            if elapsed >= 3.0:
                self.calib_step = "COLLECTING"
                self.timer_start = now
                self.calib_reps_left = 0
                self.calib_reps_right = 0
                print(f"[DEBUG][CALIB] Countdown finished. Collecting {self.target_calib_reps} active repetitions on BOTH sides.")

        # 3. 데이터 수집 및 양측 완료 판정
        if self.motion_engine and raw_keypoints:
            norm_keypoints = self.normalizer.normalize(raw_keypoints)
            if norm_keypoints:
                kpt_map = {kp["id"]: kp for kp in norm_keypoints}
                vl = self.motion_engine._compute_side_value("left", kpt_map)
                vr = self.motion_engine._compute_side_value("right", kpt_map)

                res_l = self.motion_engine.fsm_left.update(vl)
                res_r = self.motion_engine.fsm_right.update(vr)

                # 실시간 패킷 캐시 갱신
                self.last_left_data = {**res_l, "val": vl}
                self.last_right_data = {**res_r, "val": vr}

                # 수집 중일 때
                if self.calib_step == "COLLECTING":
                    if vl is not None:
                        self.calib_vals_left.append(vl)
                    if vr is not None:
                        self.calib_vals_right.append(vr)

                    self.calib_reps_left = res_l["rep_count"]
                    self.calib_reps_right = res_r["rep_count"]

                    # [핵심 수정] 좌/우측 둘 다 target_calib_reps(3회) 이상 완수해야만 COOLDOWN 진입
                    both_sides_completed = (
                        self.calib_reps_left >= self.target_calib_reps and
                        self.calib_reps_right >= self.target_calib_reps
                    )

                    if both_sides_completed:
                        self.calib_step = "COOLDOWN"
                        self.timer_start = now
                        print(f"[CALIB COMPLETE] 양측 모두 {self.target_calib_reps}회 완수 (L:{self.calib_reps_left}, R:{self.calib_reps_right})! 1초간 마무리 프레임 수집...")

                # 양측 모두 3회 달성 후 1.0초간 추가 버퍼링 유지
                elif self.calib_step == "COOLDOWN":
                    if vl is not None:
                        self.calib_vals_left.append(vl)
                    if vr is not None:
                        self.calib_vals_right.append(vr)

                    if now - self.timer_start >= 1.0:
                        self.calib_step = "FINISHED"
                        print("[CALIB FINISHED] 1초 쿨다운 종료. 양측 캘리브레이션을 확정합니다.")

        # 통합 횟수 표시는 둘 중 작은 값(최소 달성 횟수)을 기준으로 표기 (둘 다 3이 되어야 3/3 표시)
        min_completed_reps = min(self.calib_reps_left, self.calib_reps_right)

        return {
            "calib_step": self.calib_step,
            "calib_rep_count": min_completed_reps,
            "calib_reps_left": self.calib_reps_left,
            "calib_reps_right": self.calib_reps_right,
            "target_calib_reps": self.target_calib_reps
        }

    def compute_and_save_thresholds(self, exercise_name: str) -> dict:
        def _calc_side(values: list, side: str) -> dict:
            def_th = self.motion_engine.config.get("default_thresholds", {}).get(side, {})
            fb_start = def_th.get("start_val", 0.15)
            fb_target = def_th.get("target_val", 0.80)

            if not values or len(values) < 15:
                return {"start_val": fb_start, "target_val": fb_target}

            p_min = float(np.percentile(values, 5))
            p_max = float(np.percentile(values, 95))

            if self.motion_engine and self.motion_engine.motion_direction == "DECREASING":
                return {
                    "start_val": round(p_max, 2),
                    "target_val": round(p_max - (p_max - p_min) * 0.85, 2)
                }
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
            else:
                self.calib_step = 'COLLECTING'
        return dict(calib_step=self.calib_step, calib_rep_count=min(self.calib_reps_left, self.calib_reps_right),
                    calib_reps_left=self.calib_reps_left, calib_reps_right=self.calib_reps_right,
                    target_calib_reps=self.target_calib_reps)

    def compute_and_save_thresholds(self, exercise_name: str) -> dict:
        if self.calib_step != 'FINISHED':
            raise ValueError('Calibration is incomplete; thresholds were not saved')
        thresholds = {side: detector.thresholds() for side, detector in self.calibration_detectors.items()}
        if self.data_manager:
            self.data_manager.save_custom_threshold(exercise_name, thresholds, is_confirmed=True)
        return thresholds

<<<<<<< HEAD
    def process_main_frame(self, raw_keypoints: list, is_occluded: bool, now: float = None) -> tuple:
        """본 운동 측정과 판정만 처리; 종료 여부는 클라이언트가 결정"""
=======
        return custom_th

    def process_main_frame(self, raw_keypoints: list, is_occluded: bool, now: float = None) -> tuple:
        """본 운동 세션 프레임 처리 (목표 달성 후 1.5초 쿨다운 유지)"""
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
        if now is None:
            now = time.time()

        if is_occluded or not raw_keypoints:
<<<<<<< HEAD
            return ({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},
                    {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, False)

        norm_keypoints = self.normalizer.normalize(raw_keypoints)
        if not norm_keypoints:
            return ({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},
                    {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, False)

        result = self.motion_engine.process_keypoints(norm_keypoints)
        if not result:
            return ({**self.last_left_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False},
                    {**self.last_right_data, 'score': 0, 'score_delta': 0.0, 'is_count_updated': False}, False)
=======
            return self.last_left_data.copy(), self.last_right_data.copy(), False

        norm_keypoints = self.normalizer.normalize(raw_keypoints)
        if not norm_keypoints:
            return self.last_left_data.copy(), self.last_right_data.copy(), False

        result = self.motion_engine.process_keypoints(norm_keypoints)
        if not result:
            return self.last_left_data.copy(), self.last_right_data.copy(), False
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0

        left_data, right_data = result["left"], result["right"]
        self.last_left_data = left_data.copy()
        self.last_right_data = right_data.copy()

<<<<<<< HEAD
        # A client stop command, not a score threshold, ends MAIN/TEST.
        return left_data, right_data, False
=======
        target_reps = self.motion_engine.fsm_left.target_reps if self.motion_engine else 10
        reps_reached = (left_data["rep_count"] >= target_reps or right_data["rep_count"] >= target_reps)

        is_finished = False

        if reps_reached:
            if self.test_finish_timer is None:
                self.test_finish_timer = now
                print(f"[DEBUG][TEST] 목표 횟수({target_reps}회) 달성! 1.5초 쿨다운 유지 후 종료합니다.")

            if now - self.test_finish_timer >= 1.5:
                is_finished = True
                print("[DEBUG][TEST] 1.5초 쿨다운 종료. 세션 데이터를 저장합니다.")

        return left_data, right_data, is_finished
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
