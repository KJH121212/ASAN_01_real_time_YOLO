"""양측 3회 캘리브레이션 주기 감지기 및 실시간·파일 세션 제어기."""

from collections import deque
import math
import time
import numpy as np

from utils.normalization import PoseNormalizer
from utils.pose_overlay import (
    CALIBRATION_ACTIVATION_FRACTION,
    CALIBRATION_REARM_FRACTION,
    measurement_progress,
)


class CalibrationCycleDetector:
    """안정 자세(Baseline) 확인 후 3회 동작 상승·복귀 주기를 추적해 개인 임계값(85%)을 산출한다."""

    def __init__(
        self,
        rest_seconds=0.5,
        min_rise=0.08,
        noise_multiplier=6.0,
        hold_seconds=0.15,
        return_seconds=0.15,
        max_cycle_seconds=10.0,
        baseline_range=0.04,
        metric_type="COORDINATE",
        direction="INCREASING",
        min_baseline_samples=3,
        max_sample_gap=1.0,
        occlusion_reset_seconds=2.0,
        return_fraction=0.25,
        consistency_ratio=2.5,
        start_val=None,
        target_val=None,
        activation_fraction=CALIBRATION_ACTIVATION_FRACTION,
        rearm_fraction=CALIBRATION_REARM_FRACTION,
        finish_on_return=False,
        activation_hold_seconds=0.0,
        restart_on_missing=False,
    ):
        self.activation_hold_seconds = activation_hold_seconds
        self.restart_on_missing = restart_on_missing
        self.ACTIVATION_FRACTION = activation_fraction
        self.REARM_FRACTION = rearm_fraction
        self.finish_on_return = finish_on_return
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
        self.hold_seconds = hold_seconds
        self.return_seconds = return_seconds
        self.max_cycle_seconds = max_cycle_seconds
        self.baseline_range = baseline_range
        self.min_baseline_samples = min_baseline_samples
        self.max_sample_gap = max_sample_gap
        self.occlusion_reset_seconds = occlusion_reset_seconds
        self.return_fraction = return_fraction
        self.consistency_ratio = consistency_ratio
        self.reset()

    def reset(self, keep_cycles=False):
        completed = self.peaks if keep_cycles else []
        self.samples = deque()
        self.direction = self.configured_direction
        self.candidate_direction = None
        self.velocity = 0.0
        self.previous_value = None
        self.baseline = None
        self.baseline_at = None
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
        self.activation_since = None

    @property
    def ready(self):
        return self.baseline is not None

    @property
    def consistent(self):
        return len(self.peaks) >= 3 and (
            not self.finish_on_return or (self.started is None and self.state == "READY")
        )

    @property
    def return_tolerance(self):
        return self.threshold * self.REARM_FRACTION

    def status(self):
        state = (
            "MISSING"
            if self.missing_since is not None
            else "COMPLETE"
            if self.consistent
            else "NEED_CONSISTENT_CYCLES"
            if len(self.peaks) >= 3 and self.state == "READY"
            else self.state
        )
        return dict(
            state=state,
            baseline=self.baseline,
            baseline_samples=len(self.samples),
            threshold_source="CONFIG" if self.fixed_range is not None else "ADAPTIVE",
            min_baseline_samples=self.min_baseline_samples,
            rest_seconds=self.rest_seconds,
            min_movement=self.threshold * self.ACTIVATION_FRACTION,
            return_tolerance=self.return_tolerance,
            activation_fraction=self.ACTIVATION_FRACTION,
            activation_value=(
                self.baseline
                + (-1 if self.direction == "DECREASING" else 1)
                * self.threshold
                * self.ACTIVATION_FRACTION
            )
            if self.ready
            else None,
            count_on="TARGET_CROSSING",
            completed_cycles=len(self.peaks),
            consistent=self.consistent,
            direction=self.direction,
        )

    def update(self, value, now, allow_count=True):
        updated = False
        if value is None or not math.isfinite(value):
            missing_since = self.missing_since if self.missing_since is not None else now
            if now - missing_since >= self.occlusion_reset_seconds:
                self.reset(keep_cycles=not self.restart_on_missing)
            self.missing_since = missing_since
            self.samples.clear()
            self.armed = False
            self.activation_since = None
            self.state = "RETURNING" if self.ready else "BASELINE"
            self.started = self.returned = None
            self.previous_value = None
            self.last_time = now
            return updated

        if self.missing_since is not None and now - self.missing_since >= self.occlusion_reset_seconds:
            self.reset(keep_cycles=not self.restart_on_missing)
        self.missing_since = None

        if self.last_time is not None and (
            now <= self.last_time or now - self.last_time > self.occlusion_reset_seconds
        ):
            self.reset(keep_cycles=not self.restart_on_missing)
        if self.last_time is not None and now - self.last_time > self.max_sample_gap:
            self.activation_since = None
            self.samples.clear()
            self.armed = False
            self.state = "RETURNING" if self.ready else "BASELINE"
            self.started = self.returned = None

        dt = now - self.last_time if self.last_time is not None else 0.0
        self.velocity = (
            (value - self.previous_value) / dt
            if self.previous_value is not None and 0 < dt <= self.max_sample_gap
            else 0.0
        )
        self.previous_value = value
        self.last_time = now

        if not self.ready:
            self.samples.append((now, value))
            while (
                len(self.samples) > self.min_baseline_samples
                and now - self.samples[1][0] >= self.rest_seconds
            ):
                self.samples.popleft()
            if (
                len(self.samples) >= self.min_baseline_samples
                and now - self.samples[0][0] >= self.rest_seconds
            ):
                values = [v for _, v in self.samples]
                center = float(np.median(values))
                noise = 1.4826 * float(np.median(np.abs(np.asarray(values) - center)))
                if np.percentile(values, 95) - np.percentile(values, 5) <= self.baseline_range:
                    self.baseline = center
                    self.baseline_at = now
                    self.threshold = max(self.min_rise, self.noise_multiplier * noise)
                    if self.fixed_range is not None:
                        self.baseline = self.fixed_range[0]
                        self.threshold = abs(self.fixed_range[1] - self.fixed_range[0])
                    self.state = "READY"
                    sign = -1 if self.direction == "DECREASING" else 1
                    self.armed = sign * (value - self.baseline) <= self.threshold * self.REARM_FRACTION
                    if not self.armed:
                        self.state = "RETURNING"
            return updated

        if not allow_count:
            self.activation_since = None
            return updated

        displacement = value - self.baseline
        if (
            self.direction == "AUTO"
            and self.state == "READY"
            and abs(displacement) >= self.threshold * self.ACTIVATION_FRACTION
        ):
            self.candidate_direction = "INCREASING" if displacement > 0 else "DECREASING"
        direction = self.candidate_direction if self.direction == "AUTO" else self.direction
        sign = -1.0 if direction == "DECREASING" else 1.0
        rise = sign * displacement

        if self.state == "READY" and self.armed:
            if rise >= self.threshold * self.ACTIVATION_FRACTION:
                if self.activation_since is None:
                    self.activation_since = now
                if now - self.activation_since + 1e-9 < self.activation_hold_seconds:
                    return updated
                self.started = now
                self.peak = rise
                self.peaks.append(rise)
                if self.direction == "AUTO":
                    self.direction = self.candidate_direction
                self.state = "RETURNING"
                self.armed = False
                self.returned = None
                updated = True
                self.activation_since = None
            else:
                self.activation_since = None
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
        if not self.consistent:
            raise ValueError("Calibration needs three successful movements")
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


class SessionController:
    """캘리브레이션 및 본 측정(MAIN) 세션을 제어하고 관절 가림(occlusion)을 검사한다."""

    def __init__(self, data_manager=None, motion_engine=None):
        self.data_manager = data_manager
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
            defaults = (
                {"min_rise": 8.0, "baseline_range": 4.0}
                if motion_engine.metric_type == "ANGLE"
                else {}
            )
            params = {**defaults, **motion_engine.config.get("calibration_config", {})}
            use_config = params.pop("use_config_thresholds", True)
            params.pop("display_range", None)
            self.calibration_detectors = {
                side: CalibrationCycleDetector(
                    **{
                        **params,
                        **(motion_engine.config["default_thresholds"][side] if use_config else {}),
                    },
                    metric_type=motion_engine.metric_type,
                    direction=motion_engine.motion_direction,
                )
                for side in ("left", "right")
            }

        self.calib_reps_left = 0
        self.calib_reps_right = 0
        self.last_left_data = {
            "val": 0.0,
            "rep_count": 0,
            "total_score": 0.0,
            "score_delta": 0.0,
            "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "CALIBRATING",
        }
        self.last_right_data = dict(self.last_left_data)

    def update_count_thresholds(self, thresholds: dict) -> list:
        changed_sides = self.motion_engine.update_count_thresholds(thresholds)
        for side in changed_sides:
            fsm = self.motion_engine.fsm_left if side == "left" else self.motion_engine.fsm_right
            cache = self.last_left_data if side == "left" else self.last_right_data
            value = cache.get("val")
            ratio = (
                measurement_progress(value, self.motion_engine.metric_type, fsm.start_val, fsm.target_val)
                or 0.0
            )
            cache.update(
                state=fsm.state,
                rep_count=fsm.rep_count,
                **fsm.score_status(),
                progress_ratio=round(min(1.0, max(0.0, ratio)), 2),
                quality=fsm.last_quality,
                is_count_updated=False,
            )
        return changed_sides

    def check_occlusion(self, keypoints):
        """운동 평가에 필수인 관절(몸통 기준 + 운동 대상 관절)이 화면에 보이는지 확인한다."""
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
        if self.motion_engine.coordinate_unit == "pixel" or self.motion_engine.counting_method == "TARGET_POSITION":
            return not any(ids and all(visible(j) for j in ids) for ids in required)
        return not points or not all(visible(j) for ids in required for j in ids)

    def _collect_main_coordinate(self, side, keypoints, detector, now):
        """본 측정이 비정규화 좌표(RAW_COORDINATE)일 때 캘리브레이션 중 원본 좌표 범위를 함께 수집한다."""
        cfg = self.motion_engine.config.get("main_eval_config", {})
        if not cfg or cfg.get("metric_type") not in ("RAW_COORDINATE", "COORDINATE"):
            return
        tgt_id = cfg.get("target_kpt", {}).get(side)
        if tgt_id is None:
            pk = cfg.get("primary_kpts", {}).get(side, [])
            tgt_id = pk[0] if isinstance(pk, list) and len(pk) == 1 else None
        points = {p["id"]: p for p in keypoints or [] if self._in_frame(p) and p.get("score", 0.0) >= 0.35}
        point = points.get(tgt_id)
        value = point[cfg.get("axis", "y")] if point else None
        record = self.calibration_coordinates[side]
        if not detector.ready:
            record["baseline"] = None
            record["peaks"] = record["peaks"][: len(detector.peaks)]
            if value is None:
                record["samples"].clear()
                return
            record["samples"].append((now, value))
            while record["samples"] and now - record["samples"][0][0] > detector.rest_seconds:
                record["samples"].popleft()
        elif record["baseline"] is None and value is not None:
            samples = [v for _, v in record["samples"]] + [value]
            record["baseline"] = float(np.median(samples))
        if detector.started is not None and value is not None:
            if len(record["peaks"]) < len(detector.peaks):
                record["peaks"].append(value)
            select = min if cfg.get("motion_direction", "DECREASING") == "DECREASING" else max
            record["peaks"][-1] = select(record["peaks"][-1], value)

    def process_calibration_frame(self, keypoints, is_occluded, now=None):
        now = time.time() if now is None else now
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
            direction = detector.candidate_direction if detector.direction == "AUTO" else detector.direction
            sign = -1.0 if direction == "DECREASING" else 1.0
            cache = dict(
                val=value,
                rep_count=count,
                total_score=0.0,
                score_delta=0.0,
                score=0,
                calibration=detector.status(),
                state=detector.state,
                quality="CALIBRATING",
                is_count_updated=updated,
                progress_ratio=self.motion_engine.display_progress(side, value) or 0.0,
            )
            setattr(self, f"last_{side}_data", cache)
            setattr(self, f"calib_reps_{side}", count)
            fsm = getattr(self.motion_engine, f"fsm_{side}")
            fsm.rep_count = count
            fsm.state = detector.state
            if detector.ready:
                fsm.start_val = detector.baseline
                fsm.is_decreasing = direction == "DECREASING"
                fsm.target_val = detector.baseline + (-1.0 if fsm.is_decreasing else 1.0) * detector.threshold
            if updated:
                endpoint = detector.baseline + (-1.0 if fsm.is_decreasing else 1.0) * detector.peaks[-1]
                fsm.completed_reps_history.append(
                    dict(
                        side=side,
                        rep_num=count,
                        duration_sec=now - detector.started,
                        min_val=min(detector.baseline, endpoint),
                        max_val=max(detector.baseline, endpoint),
                        achieved_rom=detector.peaks[-1],
                        quality="CALIBRATING",
                        score_delta=0.0,
                        total_score=0.0,
                        start_val=fsm.start_val,
                        target_val=fsm.target_val,
                        threshold_version=fsm.threshold_version,
                    )
                )
            elif detector.started is not None and fsm.completed_reps_history:
                row = fsm.completed_reps_history[-1]
                endpoint = detector.baseline + sign * detector.peak
                row.update(
                    min_val=min(detector.baseline, endpoint),
                    max_val=max(detector.baseline, endpoint),
                    achieved_rom=detector.peak,
                    duration_sec=now - detector.started,
                )

        if is_occluded:
            self.calib_step = "PAUSED" if any(d.ready for d in detectors.values()) else "FULL_BODY_CHECK"
        else:
            if not all(d.ready for d in detectors.values()):
                self.calib_step = "COUNTDOWN"
            elif all(d.consistent for d in detectors.values()) and all(
                cache["val"] is not None for cache in (self.last_left_data, self.last_right_data)
            ):
                if self.calib_step not in ("COOLDOWN", "FINISHED"):
                    self.calib_step = "COOLDOWN"
                    self.timer_start = now
                elif now - self.timer_start >= 1.0:
                    self.calib_step = "FINISHED"
            else:
                self.calib_step = (
                    "CHECK_CONSISTENCY"
                    if any(len(d.peaks) >= 3 and not d.consistent for d in detectors.values())
                    else "COLLECTING"
                )
        return dict(
            calib_step=self.calib_step,
            calib_rep_count=min(self.calib_reps_left, self.calib_reps_right),
            calib_reps_left=self.calib_reps_left,
            calib_reps_right=self.calib_reps_right,
            target_calib_reps=self.target_calib_reps,
            calibration={side: detector.status() for side, detector in detectors.items()},
        )

    def compute_and_save_thresholds(self, exercise_name):
        if self.calib_step != "FINISHED":
            raise ValueError("Calibration is incomplete; thresholds were not saved")
        engine = self.motion_engine
        thresholds = {side: detector.thresholds() for side, detector in self.calibration_detectors.items()}
        main_cfg = engine.config.get("main_eval_config", {})
        if main_cfg:
            fraction = main_cfg.get("calibration_target_fraction", 0.85)
            converted = {}
            for side, record in self.calibration_coordinates.items():
                if record["baseline"] is None or len(record["peaks"]) < 3:
                    raise ValueError("Calibration needs three coordinate ranges")
                start = record["baseline"]
                peak = float(np.median(record["peaks"][-3:]))
                target = start + fraction * (peak - start)
                main_dir = main_cfg.get("motion_direction", "DECREASING")
                sign = -1 if main_dir == "DECREASING" else 1
                if not math.isfinite(target) or sign * (target - start) <= 0:
                    raise ValueError("Coordinate range does not match the MAIN direction; recalibrate")
                tgt_id = main_cfg.get("target_kpt", {}).get(side)
                if tgt_id is None:
                    pk = main_cfg.get("primary_kpts", {}).get(side, [])
                    tgt_id = pk[0] if isinstance(pk, list) and len(pk) == 1 else None
                converted[side] = dict(
                    start_val=round(start, 6),
                    target_val=round(target, 6),
                    metric_type=main_cfg.get("metric_type", "RAW_COORDINATE"),
                    motion_direction=main_dir,
                    use_normalization=False,
                    target_kpt=tgt_id,
                    axis=main_cfg.get("axis", "y"),
                    counting_method=main_cfg.get("counting_method", "PEAK_REVERSAL"),
                    calibration_peak_val=peak,
                    calibration_target_fraction=fraction,
                    calibration_normalized_thresholds=thresholds[side],
                )
            thresholds = converted

        for side, values in thresholds.items():
            values.setdefault("use_normalization", engine.use_normalization)
        if self.data_manager:
            self.data_manager.session_log.info["measurement"] = engine.count_threshold_status()
            self.data_manager.save_custom_threshold(exercise_name, thresholds, is_confirmed=True)
        return thresholds

    @staticmethod
    def _in_frame(point):
        return all(math.isfinite(point.get(axis, math.nan)) and 0 <= point[axis] <= 1 for axis in ("x", "y"))

    def measurement_points(self, keypoints):
        if not keypoints:
            return None
        keypoints = [p if self._in_frame(p) else {**p, "score": 0.0} for p in keypoints]
        if self.motion_engine.use_normalization:
            return self.normalizer.normalize(keypoints)
        if self.motion_engine.coordinate_unit == "pixel":
            width, height = keypoints[0].get("image_width", 1280), keypoints[0].get("image_height", 720)
            self.motion_engine.set_frame_size(width, height)
            return [{**p, "x": p["x"] * width, "y": p["y"] * height} for p in keypoints]
        aspect = float(keypoints[0].get("image_aspect_ratio", 1.0))
        return [{**p, "x": p["x"] * aspect, "image_aspect_ratio": 1.0} for p in keypoints]

    def _missing_frame(self):
        missing = [
            dict(
                data,
                val=None,
                state="MISSING",
                progress_ratio=0.0,
                score=0,
                score_delta=0.0,
                is_count_updated=False,
            )
            for data in (self.last_left_data, self.last_right_data)
        ]
        return missing[0], missing[1], False

    def process_main_frame(self, raw_keypoints: list, is_occluded: bool, now: float = None) -> tuple:
        now = time.time() if now is None else now
        if self.motion_engine.counting_method == "TARGET_POSITION":
            result = self.motion_engine.process_keypoints(self.measurement_points(raw_keypoints), now)
            self.last_left_data, self.last_right_data = result["left"], result["right"]
            return self.last_left_data, self.last_right_data, False
        if is_occluded or not raw_keypoints:
            return self._missing_frame()
        norm_keypoints = self.measurement_points(raw_keypoints)
        if not norm_keypoints:
            return self._missing_frame()
        result = self.motion_engine.process_keypoints(norm_keypoints)
        if not result:
            return self._missing_frame()
        left_data, right_data = result["left"], result["right"]
        self.last_left_data = left_data.copy()
        self.last_right_data = right_data.copy()
        return left_data, right_data, False
