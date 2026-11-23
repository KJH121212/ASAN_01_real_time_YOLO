"""Streamlit 오프라인 로그 분석(카운트 시뮬레이션 · 실전 캘리브레이션/본측정 테스트) 전용 독립 엔진.

소켓 서버 폴더(socket_server/)와 어떠한 코드도 공유하지 않도록
오프라인 분석에 필요한 필터·측정·FSM·캘리브레이션·임계값 파일 입출력을 자체 내장한다.
"""

from collections import deque
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

import numpy as np

from analysis.log_reader import NUM_KEYPOINTS, PoseNormalizer
from client.live_cards import QUALITY_CODES, QUALITY_POINTS, QUALITY_THRESHOLDS

KST = timezone(timedelta(hours=9))
CALIBRATION_ACTIVATION_FRACTION = 0.30
CALIBRATION_REARM_FRACTION = 0.15


def measurement_progress(value, metric_type, start=None, target=None):
    if any(v is None or not math.isfinite(v) for v in (value, start, target)) or start == target:
        return None
    return max(0.0, min(1.0, (value - start) / (target - start)))


def config_for_mode(config, mode):
    result = deepcopy(config)
    if mode == "MAIN" and result.get("main_eval_config"):
        result["eval_config"] = {**result.get("eval_config", {}), **result["main_eval_config"]}
    return result


class OneEuroFilter:
    def __init__(self, min_cutoff=1.0, beta=0.1, d_cutoff=1.0, max_gap=0.5):
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
        dt = timestamp - self.timestamp if self.timestamp is not None else None
        if dt is None or dt <= 0 or dt > self.max_gap:
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
    def __init__(self, min_cutoff=15.0, beta=2.0, d_cutoff=1.0, conf_threshold=0.35, max_gap=0.5):
        self.params = dict(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff, max_gap=max_gap)
        self.conf_threshold = conf_threshold
        self.filters = {}

    def update(self, current_kpts, timestamp=0.0):
        if not current_kpts:
            self.filters = {}
            return current_kpts
        result = []
        valid_ids = set()
        for kp in current_kpts:
            point = dict(kp)
            joint_id = kp["id"]
            score = kp.get("score", 0.0)
            if not all(math.isfinite(kp[axis]) for axis in ("x", "y")) or not math.isfinite(score):
                point.update(x=0.0, y=0.0, score=0.0)
            elif score >= self.conf_threshold:
                valid_ids.add(joint_id)
                if joint_id not in self.filters:
                    self.filters[joint_id] = (OneEuroFilter(**self.params), OneEuroFilter(**self.params))
                fx, fy = self.filters[joint_id]
                point["x"] = fx.update(kp["x"], timestamp)
                point["y"] = fy.update(kp["y"], timestamp)
            result.append(point)
        self.filters = {i: f for i, f in self.filters.items() if i in valid_ids}
        return result


class SideFSM:
    """오프라인 시뮬레이션용 변곡점(Peak Reversal) 카운트 상태 기계."""

    def __init__(
        self,
        side_name: str,
        motion_direction: str,
        thresholds: dict,
        target_reps: int = 1000,
        target_score=None,
        scoring=True,
        min_activation_ratio=0.40,
        return_ratio=0.30,
        reversal_ratio=0.05,
    ):
        self.side = side_name
        self.min_activation_ratio = min_activation_ratio
        self.return_ratio = return_ratio
        self.reversal_ratio = float(reversal_ratio)
        self.is_decreasing = motion_direction == "DECREASING"
        self.target_reps = target_reps
        self.state = "READY"
        self.rep_count = 0
        self.total_score = 0.0
        self.target_score = float(target_reps if target_score is None else target_score)
        self.scoring = scoring
        self.start_val = thresholds.get("start_val", 0.15 if not self.is_decreasing else 0.80)
        self.target_val = thresholds.get("target_val", 0.80 if not self.is_decreasing else 0.15)
        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        self.last_quality = "READY"
        self.completed_reps_history = []
        self.threshold_version = 0

    def score_status(self, awarded=False):
        return {
            "total_score": self.total_score,
            "target_score": self.target_score,
            "score": QUALITY_CODES.get(self.last_quality, 0) if awarded else 0,
            "score_delta": QUALITY_POINTS.get(self.last_quality, 0.0) if awarded else 0.0,
        }

    def evaluate_quality(self, peak_val: float) -> str:
        total_range = abs(self.target_val - self.start_val)
        if total_range == 0:
            return "GOOD"
        ratio = abs(peak_val - self.start_val) / total_range
        if ratio + 1e-9 >= QUALITY_THRESHOLDS["PERFECT"]:
            return "PERFECT"
        if ratio + 1e-9 >= QUALITY_THRESHOLDS["GOOD"]:
            return "GOOD"
        if ratio + 1e-9 >= QUALITY_THRESHOLDS["BAD"]:
            return "BAD"
        return "INVALID"

    def update(self, current_val: float, allow_count: bool = True) -> dict:
        if current_val is None:
            return {
                **self.score_status(),
                "state": self.state,
                "rep_count": self.rep_count,
                "progress_ratio": 0.0,
                "quality": self.last_quality,
                "is_count_updated": False,
            }
        self.min_val_recorded = min(self.min_val_recorded, current_val)
        self.max_val_recorded = max(self.max_val_recorded, current_val)
        val_range = abs(self.target_val - self.start_val)
        progress_ratio = round(
            measurement_progress(current_val, "COORDINATE", self.start_val, self.target_val) or 0.0,
            2,
        )
        is_count_updated = False

        if self.state == "READY":
            if progress_ratio >= 0.15:
                self.state = "PUSHING"
        elif self.state == "PUSHING":
            peak_val = self.min_val_recorded if self.is_decreasing else self.max_val_recorded
            current_range = abs(current_val - self.start_val)
            peak_range = abs(peak_val - self.start_val)
            has_reached = (peak_range / val_range + 1e-9 >= self.min_activation_ratio) if val_range > 0 else False
            turned_down = has_reached and ((peak_range - current_range) >= self.reversal_ratio * val_range)
            if turned_down and not allow_count:
                self.state = "RETURNING"
            elif turned_down and allow_count:
                quality = self.evaluate_quality(peak_val)
                if quality == "INVALID":
                    quality = "BAD"
                self.rep_count += 1
                score_delta = QUALITY_POINTS[quality]
                self.total_score += score_delta
                is_count_updated = True
                self.last_quality = quality
                self.completed_reps_history.append({
                    "side": self.side,
                    "rep_num": self.rep_count,
                    "duration_sec": 0.0,
                    "min_val": round(self.min_val_recorded, 2),
                    "max_val": round(self.max_val_recorded, 2),
                    "achieved_rom": round(abs(self.max_val_recorded - self.min_val_recorded), 2),
                    "quality": quality,
                    "score_delta": score_delta,
                    "total_score": self.total_score,
                    "start_val": self.start_val,
                    "target_val": self.target_val,
                    "threshold_version": self.threshold_version,
                })
                self.state = "RETURNING"
        elif self.state == "RETURNING":
            if progress_ratio <= self.return_ratio:
                self.state = "READY"
                self.min_val_recorded = 999.0
                self.max_val_recorded = -999.0

        return {
            **self.score_status(awarded=is_count_updated),
            "state": self.state,
            "rep_count": self.rep_count,
            "progress_ratio": progress_ratio,
            "quality": self.last_quality,
            "is_count_updated": is_count_updated,
        }


class CalibrationCycleDetector:
    """오프라인 캘리브레이션 주기 감지기."""

    def __init__(
        self,
        rest_seconds=0.5,
        min_rise=0.08,
        noise_multiplier=6.0,
        return_seconds=0.15,
        baseline_range=0.04,
        metric_type="COORDINATE",
        direction="INCREASING",
        min_baseline_samples=3,
        max_sample_gap=1.0,
        occlusion_reset_seconds=2.0,
        start_val=None,
        target_val=None,
        activation_fraction=CALIBRATION_ACTIVATION_FRACTION,
        rearm_fraction=CALIBRATION_REARM_FRACTION,
        **kwargs,
    ):
        self.ACTIVATION_FRACTION = activation_fraction
        self.REARM_FRACTION = rearm_fraction
        self.fixed_range = (
            (float(start_val), float(target_val))
            if (start_val is not None and target_val is not None)
            else None
        )
        self.metric_type = metric_type
        self.configured_direction = direction
        self.rest_seconds = rest_seconds
        self.min_rise = min_rise
        self.noise_multiplier = noise_multiplier
        self.return_seconds = return_seconds
        self.baseline_range = baseline_range
        self.min_baseline_samples = min_baseline_samples
        self.max_sample_gap = max_sample_gap
        self.occlusion_reset_seconds = occlusion_reset_seconds
        self.reset()

    def reset(self, keep_cycles=False):
        completed = getattr(self, "peaks", []) if keep_cycles else []
        self.samples = deque()
        self.direction = self.configured_direction
        self.candidate_direction = None
        self.baseline = None
        self.threshold = (
            abs(self.fixed_range[1] - self.fixed_range[0])
            if self.fixed_range is not None
            else self.min_rise
        )
        self.state = "BASELINE"
        self.peaks = completed
        self.started = None
        self.returned = None
        self.peak = 0.0
        self.last_time = None
        self.missing_since = None
        self.armed = False

    @property
    def ready(self):
        return self.baseline is not None

    @property
    def consistent(self):
        return len(self.peaks) >= 3

    def status(self):
        return dict(
            state="COMPLETE" if self.consistent else self.state,
            baseline=self.baseline,
            completed_cycles=len(self.peaks),
            consistent=self.consistent,
            direction=self.direction,
        )

    def update(self, value, now, allow_count=True):
        updated = False
        if value is None or not math.isfinite(value):
            self.samples.clear()
            self.armed = False
            self.state = "RETURNING" if self.ready else "BASELINE"
            self.started = self.returned = None
            self.last_time = now
            return updated

        self.last_time = now
        if not self.ready:
            self.samples.append((now, value))
            while len(self.samples) > self.min_baseline_samples and now - self.samples[1][0] >= self.rest_seconds:
                self.samples.popleft()
            if len(self.samples) >= self.min_baseline_samples and now - self.samples[0][0] >= self.rest_seconds:
                values = [v for _, v in self.samples]
                center = float(np.median(values))
                noise = 1.4826 * float(np.median(np.abs(np.asarray(values) - center)))
                if np.percentile(values, 95) - np.percentile(values, 5) <= self.baseline_range:
                    self.baseline = self.fixed_range[0] if self.fixed_range is not None else center
                    self.threshold = (
                        abs(self.fixed_range[1] - self.fixed_range[0])
                        if self.fixed_range is not None
                        else max(self.min_rise, self.noise_multiplier * noise)
                    )
                    self.state = "READY"
                    sign = -1 if self.direction == "DECREASING" else 1
                    self.armed = sign * (value - self.baseline) <= self.threshold * self.REARM_FRACTION
                    if not self.armed:
                        self.state = "RETURNING"
            return updated

        if not allow_count:
            return updated

        sign = -1.0 if self.direction == "DECREASING" else 1.0
        rise = sign * (value - self.baseline)
        if self.state == "READY" and self.armed:
            if rise >= self.threshold * self.ACTIVATION_FRACTION:
                self.started = now
                self.peak = rise
                self.peaks.append(rise)
                self.state = "RETURNING"
                self.armed = False
                self.returned = None
                updated = True
        else:
            if self.started is not None and self.peaks:
                self.peak = max(self.peak, rise)
                self.peaks[-1] = self.peak
            if rise <= self.threshold * self.REARM_FRACTION:
                if self.returned is None:
                    self.returned = now
                elif now - self.returned >= self.return_seconds:
                    self.state = "READY"
                    self.armed = True
                    self.returned = None
                    self.started = None
            else:
                self.returned = None
        return updated

    def thresholds(self):
        return {
            "start_val": round(self.baseline, 4),
            "target_val": round(
                self.baseline
                + (-1.0 if self.direction == "DECREASING" else 1.0)
                * 0.85
                * float(np.median(self.peaks[-3:])),
                4,
            ),
            "metric_type": self.metric_type,
            "motion_direction": self.direction,
        }


class MotionEngine:
    """오프라인 분석용 측정값 계산 및 FSM 엔진."""

    def __init__(self, config, custom_thresholds=None, mode="MAIN", target_reps=1000):
        self.config = config = config_for_mode(config, mode)
        self.exercise_name = config.get("exercise_name", "unknown")
        self.mode = mode
        cfg = config.get("eval_config", {})
        self.metric_type = cfg.get("metric_type", "COORDINATE")
        self.use_normalization = cfg.get("use_normalization", self.metric_type != "RAW_COORDINATE")
        self.coordinate_unit = cfg.get("coordinate_unit", "image")
        self.motion_direction = cfg.get(
            "motion_direction",
            "DECREASING" if self.metric_type == "RAW_COORDINATE" else "INCREASING",
        )
        self.counting_method = cfg.get("counting_method", "PEAK_REVERSAL")
        self.posture_constraints = cfg.get("posture_constraints") or []
        defaults = config.get("default_thresholds", {})
        left = (custom_thresholds or {}).get("left", defaults.get("left", {}))
        right = (custom_thresholds or {}).get("right", defaults.get("right", {}))
        self.fsm_left = SideFSM("left", left.get("motion_direction", self.motion_direction), left, target_reps)
        self.fsm_right = SideFSM("right", right.get("motion_direction", self.motion_direction), right, target_reps)

    @staticmethod
    def _valid_points(ids, kpt_map, confidence=0.35):
        points = [kpt_map.get(i) for i in ids]
        if not points or not all(
            p and p.get("score", 0.0) >= confidence and all(math.isfinite(p[a]) for a in ("x", "y"))
            for p in points
        ):
            return None
        return points

    def _compute_side_value(self, side: str, kpt_map: dict):
        cfg = self.config.get("eval_config", {})
        if self.metric_type == "ANGLE":
            pts = self._valid_points(cfg.get("primary_kpts", {}).get(side, []), kpt_map, 0.4)
            if not pts or len(pts) != 3:
                return None
            v1 = np.array([pts[0]["x"] - pts[1]["x"], pts[0]["y"] - pts[1]["y"]])
            v2 = np.array([pts[2]["x"] - pts[1]["x"], pts[2]["y"] - pts[1]["y"]])
            n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
            return round(math.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))), 2) if n1 * n2 > 0 else 0.0
        if self.metric_type == "LENGTH":
            pts = self._valid_points(cfg.get("primary_kpts", {}).get(side, []), kpt_map)
            return round(math.hypot(pts[1]["x"] - pts[0]["x"], pts[1]["y"] - pts[0]["y"]), 4) if pts and len(pts) == 2 else None
        axis = cfg.get("axis", "y")
        target = cfg.get("target_kpt", {}).get(side)
        if target is None:
            pk = cfg.get("primary_kpts", {}).get(side, [])
            target = pk[0] if len(pk) == 1 else None
        pts = self._valid_points([target] if target is not None else [], kpt_map)
        return round(float(pts[0][axis]), 4) if pts else None

    def count_posture_valid(self, side: str, kpt_map: dict) -> bool:
        for c in self.posture_constraints:
            t_spec = c.get("target_kpt")
            r_spec = c.get("reference_kpt")
            t_id = t_spec.get(side) if isinstance(t_spec, dict) else t_spec
            r_id = r_spec.get(side) if isinstance(r_spec, dict) else r_spec
            pts = self._valid_points((r_id, t_id), kpt_map, float(c.get("min_confidence", 0.4)))
            if not pts:
                return False
            ref_pt, target_pt = pts
            rel = str(c.get("relation", "ABOVE")).upper()
            margin = float(c.get("margin", 0.0))
            if rel == "ABOVE":
                ok = (target_pt["y"] > ref_pt["y"] + margin) if self.use_normalization else (target_pt["y"] < ref_pt["y"] - margin)
                if not ok:
                    return False
            elif rel == "BELOW":
                ok = (target_pt["y"] < ref_pt["y"] - margin) if self.use_normalization else (target_pt["y"] > ref_pt["y"] + margin)
                if not ok:
                    return False
        return True

    def display_progress(self, side, value):
        defaults = self.config.get("default_thresholds", {}).get(side, {"start_val": 0.0, "target_val": 1.0})
        return measurement_progress(value, "COORDINATE", defaults["start_val"], defaults["target_val"])

    def process_keypoints(self, keypoints: list) -> dict:
        if not keypoints:
            return None
        kpt_map = {kp["id"]: kp for kp in keypoints}
        vl = self._compute_side_value("left", kpt_map)
        vr = self._compute_side_value("right", kpt_map)
        rl = self.fsm_left.update(vl, allow_count=self.count_posture_valid("left", kpt_map))
        rr = self.fsm_right.update(vr, allow_count=self.count_posture_valid("right", kpt_map))
        return {
            "mode": self.mode,
            "left": {**rl, "val": vl, "counting_method": self.counting_method},
            "right": {**rr, "val": vr, "counting_method": self.counting_method},
        }


class SessionController:
    """오프라인 NPZ 캘리브레이션·본측정 재현용 세션 제어기."""

    def __init__(self, motion_engine=None):
        self.motion_engine = motion_engine
        self.target_calib_reps = 3
        self.mode = motion_engine.mode if motion_engine else "CALIBRATION"
        self.calib_step = "FULL_BODY_CHECK"
        self.timer_start = 0.0
        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)
        self.calibration_detectors = {}
        self.calibration_coordinates = {
            side: dict(samples=deque(), baseline=None, peaks=[]) for side in ("left", "right")
        }
        if motion_engine and motion_engine.mode == "CALIBRATION":
            defaults = {"min_rise": 8.0, "baseline_range": 4.0} if motion_engine.metric_type == "ANGLE" else {}
            params = {**defaults, **motion_engine.config.get("calibration_config", {})}
            self.calibration_detectors = {
                side: CalibrationCycleDetector(
                    **{**params, **motion_engine.config["default_thresholds"][side]},
                    metric_type=motion_engine.metric_type,
                    direction=motion_engine.motion_direction,
                )
                for side in ("left", "right")
            }
        self.calib_reps_left = 0
        self.calib_reps_right = 0
        self.last_left_data = {"val": None, "rep_count": 0}
        self.last_right_data = {"val": None, "rep_count": 0}

    @staticmethod
    def _in_frame(point):
        return all(math.isfinite(point.get(a, math.nan)) and 0 <= point[a] <= 1 for a in ("x", "y"))

    def check_occlusion(self, keypoints):
        points = {p["id"]: p for p in keypoints or []}

        def visible(j):
            return j in points and points[j].get("score", 0) >= 0.35 and self._in_frame(points[j])

        cfg = self.motion_engine.config["eval_config"]
        required = []
        for side in ("left", "right"):
            ids = set(cfg.get("primary_kpts", {}).get(side, []))
            if cfg.get("target_kpt", {}).get(side) is not None:
                ids.add(cfg["target_kpt"][side])
            if self.motion_engine.use_normalization:
                ids.update((5, 6, 11, 12))
            required.append(ids)
        return not points or not all(visible(j) for ids in required for j in ids)

    def measurement_points(self, keypoints):
        if not keypoints:
            return None
        keypoints = [p if self._in_frame(p) else {**p, "score": 0.0} for p in keypoints]
        if self.motion_engine.use_normalization:
            return self.normalizer.normalize(keypoints)
        aspect = float(keypoints[0].get("image_aspect_ratio", 1.0))
        return [{**p, "x": p["x"] * aspect, "image_aspect_ratio": 1.0} for p in keypoints]

    def _collect_main_coordinate(self, side, keypoints, detector, now):
        cfg = self.motion_engine.config.get("main_eval_config", {})
        if not cfg or cfg.get("metric_type") not in ("RAW_COORDINATE", "COORDINATE"):
            return
        tgt_id = cfg.get("target_kpt", {}).get(side)
        points = {p["id"]: p for p in keypoints or [] if self._in_frame(p) and p.get("score", 0.0) >= 0.35}
        point = points.get(tgt_id)
        value = point[cfg.get("axis", "y")] if point else None
        record = self.calibration_coordinates[side]
        if not detector.ready:
            record["baseline"] = None
            record["peaks"] = record["peaks"][: len(detector.peaks)]
            if value is not None:
                record["samples"].append((now, value))
                while record["samples"] and now - record["samples"][0][0] > detector.rest_seconds:
                    record["samples"].popleft()
        elif record["baseline"] is None and value is not None:
            record["baseline"] = float(np.median([v for _, v in record["samples"]] + [value]))
        if detector.started is not None and value is not None:
            if len(record["peaks"]) < len(detector.peaks):
                record["peaks"].append(value)
            select = min if cfg.get("motion_direction", "DECREASING") == "DECREASING" else max
            record["peaks"][-1] = select(record["peaks"][-1], value)

    def process_calibration_frame(self, keypoints, is_occluded, now=0.0):
        detectors = self.calibration_detectors
        normalized = self.measurement_points(keypoints) if keypoints and not is_occluded else None
        is_occluded = is_occluded or not normalized
        points = {p["id"]: p for p in normalized} if normalized else {}
        allow_count = all(d.ready for d in detectors.values()) and not is_occluded
        for side, detector in detectors.items():
            value = self.motion_engine._compute_side_value(side, points)
            updated = detector.update(
                value,
                now,
                allow_count=allow_count
                and not detector.consistent
                and self.motion_engine.count_posture_valid(side, points),
            )
            self._collect_main_coordinate(side, keypoints if not is_occluded else None, detector, now)
            count = len(detector.peaks)
            setattr(self, f"last_{side}_data", dict(val=value, rep_count=count, is_count_updated=updated))
            setattr(self, f"calib_reps_{side}", count)
        if all(d.consistent for d in detectors.values()):
            self.calib_step = "FINISHED"
        elif all(d.ready for d in detectors.values()):
            self.calib_step = "COLLECTING"
        else:
            self.calib_step = "COUNTDOWN"
        return dict(
            calib_step=self.calib_step,
            calib_reps_left=self.calib_reps_left,
            calib_reps_right=self.calib_reps_right,
        )

    def compute_thresholds(self):
        thresholds = {side: detector.thresholds() for side, detector in self.calibration_detectors.items()}
        main_cfg = self.motion_engine.config.get("main_eval_config", {})
        if main_cfg:
            fraction = main_cfg.get("calibration_target_fraction", 0.85)
            converted = {}
            for side, record in self.calibration_coordinates.items():
                if record["baseline"] is None or len(record["peaks"]) < 3:
                    raise ValueError("Calibration needs three coordinate ranges")
                start = record["baseline"]
                peak = float(np.median(record["peaks"][-3:]))
                target = start + fraction * (peak - start)
                tgt_id = main_cfg.get("target_kpt", {}).get(side)
                converted[side] = dict(
                    start_val=round(start, 6),
                    target_val=round(target, 6),
                    metric_type=main_cfg.get("metric_type", "RAW_COORDINATE"),
                    motion_direction=main_cfg.get("motion_direction", "DECREASING"),
                    use_normalization=False,
                    target_kpt=tgt_id,
                    axis=main_cfg.get("axis", "y"),
                    counting_method=main_cfg.get("counting_method", "PEAK_REVERSAL"),
                    calibration_peak_val=peak,
                    calibration_target_fraction=fraction,
                    calibration_normalized_thresholds=thresholds[side],
                )
            thresholds = converted
        return thresholds

    def process_main_frame(self, raw_keypoints: list, is_occluded: bool, now: float = 0.0) -> tuple:
        if is_occluded or not raw_keypoints:
            return {"val": None, "is_count_updated": False}, {"val": None, "is_count_updated": False}, False
        norm_keypoints = self.measurement_points(raw_keypoints)
        if not norm_keypoints:
            return {"val": None, "is_count_updated": False}, {"val": None, "is_count_updated": False}, False
        result = self.motion_engine.process_keypoints(norm_keypoints)
        if not result:
            return {"val": None, "is_count_updated": False}, {"val": None, "is_count_updated": False}, False
        self.last_left_data = result["left"].copy()
        self.last_right_data = result["right"].copy()
        return self.last_left_data, self.last_right_data, False


def find_saved_threshold(data_root: Path, patient_id: str, exercise_name: str, round_number: int = 1):
    """data/{환자번호}/*/{운동명}/roundXX_thresholds_v*.json에서 최신 확정 임계값을 찾는다."""
    patient_dir = data_root / str(patient_id).zfill(4)
    if not patient_dir.is_dir():
        return None
    prefix = f"round{int(round_number):02d}_thresholds_v"
    candidates = sorted(patient_dir.rglob(f"{exercise_name}/{prefix}*.json"), reverse=True)
    for path in candidates:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("is_confirmed") and isinstance(record.get("thresholds"), dict):
                return record["thresholds"]
        except Exception:
            continue
    return None


def save_threshold_file(
    data_root: Path,
    patient_id: str,
    exercise_name: str,
    round_number: int,
    thresholds: dict,
    date_str: str = None,
) -> Path:
    """오프라인 캘리브레이션에서 확정된 임계값을 정규 JSON 파일로 저장한다."""
    patient_id = str(patient_id).zfill(4)
    date_str = date_str or datetime.now(KST).strftime("%Y-%m-%d")
    target_dir = data_root / patient_id / date_str / exercise_name
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"round{int(round_number):02d}_thresholds_v"
    existing = [
        int(p.stem[len(prefix):])
        for p in target_dir.glob(f"{prefix}*.json")
        if p.stem[len(prefix):].isdigit()
    ]
    version = max(existing, default=0) + 1
    out_path = target_dir / f"{prefix}{version:02d}.json"
    record = dict(
        schema="round-calibration/1",
        patient_id=patient_id,
        data_version="v1.0.0(0)",
        registration="01",
        date=date_str,
        exercise_name=exercise_name,
        round=int(round_number),
        version=version,
        created_at=datetime.now(KST).isoformat(),
        calibration_session_id="offline_analysis",
        is_confirmed=True,
        thresholds=thresholds,
    )
    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
