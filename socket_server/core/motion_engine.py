"""23개 관절(COCO-23) 기반 운동 측정값 계산, 변곡점(Peak Reversal)·목표 도달 카운트 및 품질 판정 엔진."""

import math
import time
from copy import deepcopy
from numbers import Real
import numpy as np

from utils.pose_overlay import (
    CALIBRATION_ACTIVATION_FRACTION,
    CALIBRATION_REARM_FRACTION,
    QUALITY_CODES,
    QUALITY_POINTS,
    QUALITY_THRESHOLDS,
    measurement_progress,
)

NUM_KEYPOINTS = 23


class SideFSM:
    """한쪽(left/right) 반복 횟수와 품질 점수를 판정하는 상태 기계."""

    def __init__(
        self,
        side_name: str,
        motion_direction: str,
        thresholds: dict,
        target_reps: int = 10,
        target_score=None,
        scoring=True,
        min_activation_ratio=0.40,
        return_ratio=0.30,
        reversal_ratio=0.05,
    ):
        self.side = side_name
        if not math.isfinite(min_activation_ratio) or not 0 < min_activation_ratio <= 1:
            raise ValueError("min_activation_ratio must be between 0 and 1")
        self.min_activation_ratio = min_activation_ratio
        if not math.isfinite(return_ratio) or not 0 <= return_ratio < min_activation_ratio:
            raise ValueError("return_ratio must be below min_activation_ratio")
        self.return_ratio = return_ratio
        if not math.isfinite(reversal_ratio) or not 0 <= reversal_ratio <= 1:
            raise ValueError("reversal_ratio must be between 0 and 1")
        self.reversal_ratio = float(reversal_ratio)
        self.is_decreasing = motion_direction == "DECREASING"
        self.target_reps = target_reps

        self.state = "READY"
        self.rep_count = 0
        self.total_score = 0.0
        self.target_score = float(target_reps if target_score is None else target_score)
        if not math.isfinite(self.target_score) or self.target_score <= 0:
            raise ValueError("target_score must be finite and positive")
        self.scoring = scoring

        self.start_val = thresholds.get("start_val", 0.15 if not self.is_decreasing else 0.80)
        self.target_val = thresholds.get("target_val", 0.80 if not self.is_decreasing else 0.15)

        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        self.last_quality = "READY"
        self.completed_reps_history = []
        self.threshold_version = 0

    def replace_thresholds(self, start_val: float, target_val: float, version: int):
        """새 가동 범위를 적용하고 진행 중이던 극값을 초기화한다."""
        self.start_val = start_val
        self.target_val = target_val
        self.threshold_version = version
        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        self.state = "WAITING" if self.target_reached else "RETURNING"

    @property
    def target_reached(self):
        return False if self.scoring else self.rep_count >= self.target_reps

    def score_status(self, awarded=False):
        return {
            "total_score": self.total_score,
            "target_score": self.target_score,
            "score": QUALITY_CODES.get(self.last_quality, 0) if awarded else 0,
            "score_delta": QUALITY_POINTS.get(self.last_quality, 0.0) if awarded else 0.0,
        }

    def evaluate_quality(self, peak_val: float) -> str:
        """가동 범위 대비 최고 도달 비율로 PERFECT·GOOD·BAD·INVALID 등급을 판정한다."""
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
        """측정값을 받아 상태 전이(READY -> PUSHING -> RETURNING) 및 카운트를 수행한다."""
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

        if self.target_reached:
            self.state = "WAITING"
            return {
                **self.score_status(),
                "state": "WAITING",
                "rep_count": self.rep_count,
                "progress_ratio": progress_ratio,
                "quality": self.last_quality,
                "is_count_updated": False,
            }

        if self.state == "READY":
            if progress_ratio >= 0.15:
                self.state = "PUSHING"

        elif self.state == "PUSHING":
            peak_val = self.min_val_recorded if self.is_decreasing else self.max_val_recorded
            current_range = abs(current_val - self.start_val)
            peak_range = abs(peak_val - self.start_val)
            has_reached = (peak_range / val_range + 1e-9 >= self.min_activation_ratio) if val_range > 0 else False
            reversal_delta = self.reversal_ratio * val_range
            turned_down = has_reached and ((peak_range - current_range) >= reversal_delta)

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
                self.state = "WAITING" if self.target_reached else "RETURNING"

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


class TargetPositionSide(SideFSM):
    """목표 좌표 도달 시 즉시 1회 인정하고 복귀 후 재준비하는 카운터."""

    RETURN_FRACTION = 0.15
    MAX_GAP_SECONDS = 0.5

    def __init__(self, side, direction, thresholds, target_reps, target_score):
        super().__init__(side, direction, thresholds, target_reps, target_score, scoring=True)
        if (
            not all(math.isfinite(v) for v in (self.start_val, self.target_val))
            or (self.target_val - self.start_val) * (-1 if self.is_decreasing else 1) <= 0
        ):
            raise ValueError("Target coordinate must match the movement direction")
        self.armed = False
        self.last_time = self.started = None
        self.state = "RETURNING"

    def replace_thresholds(self, start_val, target_val, version):
        super().replace_thresholds(start_val, target_val, version)
        self.armed = False
        self.started = None

    def update(self, value, now=None):
        now = time.monotonic() if now is None else now
        if not math.isfinite(now) or (self.last_time is not None and now <= self.last_time):
            raise ValueError("Timestamps must increase")
        if self.last_time is not None and now - self.last_time > self.MAX_GAP_SECONDS:
            self.armed = False
        self.last_time = now
        progress = measurement_progress(value, "COORDINATE", self.start_val, self.target_val)
        updated = False
        if progress is None:
            self.armed = False
            self.state = "MISSING"
        elif progress <= self.RETURN_FRACTION:
            if not self.armed:
                self.started = now
                self.min_val_recorded = self.max_val_recorded = value
            self.armed = True
            self.state = "READY"
        elif self.armed:
            self.min_val_recorded = min(self.min_val_recorded, value)
            self.max_val_recorded = max(self.max_val_recorded, value)
            self.state = "PUSHING"
            if progress >= 1.0:
                self.rep_count += 1
                self.total_score += 1.0
                self.last_quality = "PERFECT"
                self.armed = False
                self.state = "RETURNING"
                updated = True
                self.completed_reps_history.append(
                    dict(
                        side=self.side,
                        rep_num=self.rep_count,
                        duration_sec=now - self.started,
                        min_val=self.min_val_recorded,
                        max_val=self.max_val_recorded,
                        achieved_rom=self.max_val_recorded - self.min_val_recorded,
                        quality=self.last_quality,
                        score_delta=1.0,
                        total_score=self.total_score,
                        start_val=self.start_val,
                        target_val=self.target_val,
                        threshold_version=self.threshold_version,
                    )
                )
        else:
            self.state = "RETURNING"
        return dict(
            val=value,
            rep_count=self.rep_count,
            state=self.state,
            quality=self.last_quality,
            progress_ratio=progress or 0.0,
            is_count_updated=updated,
            counting_method="TARGET_POSITION",
            **self.score_status(awarded=updated),
        )


def config_for_mode(config, mode):
    """MAIN 모드일 때 main_eval_config가 있으면 eval_config에 병합해 반환한다."""
    result = deepcopy(config)
    if mode == "MAIN" and result.get("main_eval_config"):
        result["eval_config"] = {**result.get("eval_config", {}), **result["main_eval_config"]}
    return result


class MotionEngine:
    """운동 설정(ANGLE / COORDINATE / RAW_COORDINATE / LENGTH)에 따라 좌우 관절 측정 및 카운트를 수행한다."""

    def __init__(self, config, custom_thresholds=None, mode="MAIN", target_reps=10, target_score=None):
        if mode not in {"CALIBRATION", "MAIN"}:
            raise ValueError("mode must be CALIBRATION or MAIN")
        self.config = config = config_for_mode(config, mode)
        self.exercise_name = config.get("exercise_name", "unknown")
        self.threshold_version = 0
        self.mode = mode
        cfg = config.get("eval_config", {})
        aliases = {
            "RELATIVE_Y": "COORDINATE",
            "ANKLE_HEIGHT": "COORDINATE",
            "DISTANCE": "LENGTH",
            "UNNORM_COORDINATE": "RAW_COORDINATE",
        }
        self.metric_type = aliases.get(cfg.get("metric_type"), cfg.get("metric_type", "ANGLE"))
        if self.metric_type not in {"ANGLE", "COORDINATE", "RAW_COORDINATE", "LENGTH"}:
            raise ValueError("metric_type must be ANGLE, COORDINATE, RAW_COORDINATE or LENGTH")

        default_norm = False if self.metric_type == "RAW_COORDINATE" else True
        self.use_normalization = cfg.get("use_normalization", default_norm)
        if type(self.use_normalization) is not bool:
            raise ValueError("use_normalization must be true or false")
        if self.metric_type == "RAW_COORDINATE" and self.use_normalization:
            raise ValueError("RAW_COORDINATE requires use_normalization=false")

        self.coordinate_unit = cfg.get("coordinate_unit", "image")
        if self.coordinate_unit not in {"image", "pixel"}:
            raise ValueError("coordinate_unit must be image or pixel")
        self.motion_direction = cfg.get(
            "motion_direction",
            "DECREASING" if self.metric_type == "RAW_COORDINATE" else "INCREASING",
        )
        self.counting_method = cfg.get("counting_method", "PEAK_REVERSAL")
        if self.counting_method not in {"PEAK_REVERSAL", "TARGET_POSITION"}:
            raise ValueError("Unsupported counting_method")

        self.posture_constraints = cfg.get("posture_constraints") or config.get("posture_constraints") or []
        self.frame_size = None
        self.calibration_display_thresholds = {}
        self.expected_frame_size = None

        for side in ("left", "right"):
            if self.metric_type in {"COORDINATE", "RAW_COORDINATE"}:
                if cfg.get("axis", "y") not in ("x", "y"):
                    raise ValueError("COORDINATE axis must be x or y")
                joint = cfg.get("target_kpt", {}).get(side)
                if joint is None:
                    pk = cfg.get("primary_kpts", {}).get(side, [])
                    joint = pk[0] if isinstance(pk, list) and len(pk) == 1 else None
                if joint is not None and (type(joint) is not int or not 0 <= joint < NUM_KEYPOINTS):
                    raise ValueError("target_kpt/primary_kpts must contain COCO-23 joint IDs (0~22)")
            elif self.metric_type == "LENGTH":
                pair = cfg.get("primary_kpts", {}).get(side, [])
                if not isinstance(pair, list) or len(pair) != 2 or any(type(j) is not int or not 0 <= j < NUM_KEYPOINTS for j in pair):
                    raise ValueError("LENGTH requires 2 COCO-23 joint IDs in primary_kpts")
            elif self.metric_type == "ANGLE":
                triplet = cfg.get("primary_kpts", {}).get(side, [])
                if not isinstance(triplet, list) or len(triplet) != 3 or any(type(j) is not int or not 0 <= j < NUM_KEYPOINTS for j in triplet):
                    raise ValueError("ANGLE requires 3 COCO-23 joint IDs in primary_kpts")

        defaults = config.get("default_thresholds", {})
        left = custom_thresholds["left"] if custom_thresholds else defaults.get("left", {})
        right = custom_thresholds["right"] if custom_thresholds else defaults.get("right", {})
        self.fsm_left = SideFSM(
            "left",
            left.get("motion_direction", self.motion_direction),
            left,
            target_reps,
            target_score,
            mode != "CALIBRATION",
            cfg.get("min_activation_ratio", 0.40),
            cfg.get("return_ratio", 0.30),
        )
        self.fsm_right = SideFSM(
            "right",
            right.get("motion_direction", self.motion_direction),
            right,
            target_reps,
            target_score,
            mode != "CALIBRATION",
            cfg.get("min_activation_ratio", 0.40),
            cfg.get("return_ratio", 0.30),
        )
        if self.counting_method == "TARGET_POSITION":
            if self.metric_type != "COORDINATE" or mode != "MAIN" or not custom_thresholds:
                raise ValueError("TARGET_POSITION requires MAIN coordinate measurement and calibrated thresholds")
            self.fsm_left = TargetPositionSide("left", self.motion_direction, left, target_reps, target_score)
            self.fsm_right = TargetPositionSide("right", self.motion_direction, right, target_reps, target_score)

    @property
    def metric_unit(self):
        return (
            "px"
            if self.coordinate_unit == "pixel"
            else "deg"
            if self.metric_type == "ANGLE"
            else "body_length"
            if self.use_normalization
            else "image_height"
        )

    def set_frame_size(self, width, height):
        self.frame_size = [int(width), int(height)]

    def get_display_thresholds(self):
        if self.mode == "CALIBRATION":
            defaults = self.config.get("default_thresholds", {})
            return {
                side: dict(
                    self.calibration_display_thresholds.get(
                        side, defaults.get(side, self.get_count_thresholds()[side])
                    )
                )
                for side in ("left", "right")
            }
        return self.get_count_thresholds()

    def display_progress(self, side, value):
        bounds = self.get_display_thresholds()[side]
        return measurement_progress(value, "COORDINATE", bounds["start_val"], bounds["target_val"])

    def get_count_thresholds(self) -> dict:
        return {
            side: {"start_val": fsm.start_val, "target_val": fsm.target_val}
            for side, fsm in (("left", self.fsm_left), ("right", self.fsm_right))
        }

    def count_threshold_status(self):
        cal = self.config.get("calibration_config", {})
        adaptive = not cal.get("use_config_thresholds", True)
        return dict(
            exercise_name=self.exercise_name,
            metric_type=self.metric_type,
            metric_unit=self.metric_unit,
            display_thresholds=self.get_display_thresholds(),
            measurement={
                **self.config["eval_config"],
                "use_normalization": self.use_normalization,
                "metric_unit": self.metric_unit,
                "frame_size": self.frame_size,
                "display_thresholds": self.get_display_thresholds(),
                "calibration_threshold_source": "ADAPTIVE" if adaptive else "CONFIG",
                "calibration_min_movement": cal.get("min_rise") if adaptive else None,
                "calibration_activation_fraction": None if adaptive else CALIBRATION_ACTIVATION_FRACTION,
                "calibration_rearm_fraction": cal.get("rearm_fraction", CALIBRATION_REARM_FRACTION),
                "calibration_count_on": "TARGET_CROSSING",
                "count_on": "TARGET_CROSSING" if self.mode == "CALIBRATION" else self.counting_method,
            },
            motion_direction=self.motion_direction,
            threshold_version=self.threshold_version,
            thresholds=self.get_count_thresholds(),
            target_score=self.fsm_left.target_score,
            scoring_mode="REPETITIONS" if self.mode == "CALIBRATION" else "CLIENT",
            completion_owner="SERVER" if self.mode == "CALIBRATION" else "CLIENT",
            target_reps=self.fsm_left.target_reps,
        )

    def update_count_thresholds(self, thresholds: dict) -> list:
        """실행 중 좌우 카운트 시작/목표 기준값을 검증하고 업데이트한다."""
        if not isinstance(thresholds, dict) or not thresholds:
            raise ValueError("thresholds must be a nonempty object with left/right")
        if set(thresholds) - {"left", "right"}:
            raise ValueError("thresholds supports only left and right")

        current = self.get_count_thresholds()
        validated = {}
        for side, patch in thresholds.items():
            if not isinstance(patch, dict) or not patch:
                raise ValueError(f"{side} must contain start_val and/or target_val")
            if set(patch) - {"start_val", "target_val"}:
                raise ValueError(f"{side} supports only start_val and target_val")
            values = {**current[side], **patch}
            for key, value in values.items():
                if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                    raise ValueError(f"{side}.{key} must be a finite number")
                values[key] = float(value)
            start, target = values["start_val"], values["target_val"]
            fsm = self.fsm_left if side == "left" else self.fsm_right
            if fsm.is_decreasing and target >= start:
                raise ValueError(f"{side}: DECREASING requires target_val < start_val")
            if not fsm.is_decreasing and target <= start:
                raise ValueError(f"{side}: INCREASING requires target_val > start_val")
            if values != current[side]:
                validated[side] = values

        if validated:
            self.threshold_version += 1
            for side, values in validated.items():
                fsm = self.fsm_left if side == "left" else self.fsm_right
                fsm.replace_thresholds(**values, version=self.threshold_version)
        return list(validated)

    @staticmethod
    def _calculate_angle(p1: dict, p2: dict, p3: dict) -> float:
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return 0.0
        cosine = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        return round(math.degrees(np.arccos(cosine)), 2)

    @staticmethod
    def _valid_points(ids, kpt_map, confidence=0.35):
        points = [kpt_map.get(i) for i in ids]
        if not points or not all(
            p and p.get("score", 0.0) >= confidence and all(math.isfinite(p[axis]) for axis in ("x", "y"))
            for p in points
        ):
            return None
        return points

    def _compute_angle(self, side, kpt_map):
        ids = self.config.get("eval_config", {}).get("primary_kpts", {}).get(side, [])
        points = self._valid_points(ids, kpt_map, 0.4)
        return self._calculate_angle(*points) if points and len(points) == 3 else None

    def _compute_coordinate(self, side, kpt_map):
        cfg = self.config.get("eval_config", {})
        axis = cfg.get("axis", "y")
        target = cfg.get("target_kpt", {}).get(side)
        if target is None:
            pts = cfg.get("primary_kpts", {}).get(side, [])
            target = pts[0] if len(pts) == 1 else None
        points = self._valid_points([target] if target is not None else [], kpt_map)
        return round(float(points[0][axis]), 4) if points else None

    def _compute_length(self, side, kpt_map):
        ids = self.config.get("eval_config", {}).get("primary_kpts", {}).get(side, [])
        points = self._valid_points(ids, kpt_map)
        if points and len(points) == 2:
            return round(math.hypot(points[1]["x"] - points[0]["x"], points[1]["y"] - points[0]["y"]), 4)
        return None

    @staticmethod
    def _resolve_kpt_id(kpt_spec, side: str):
        return kpt_spec.get(side) if isinstance(kpt_spec, dict) else kpt_spec

    def count_posture_valid(self, side: str, kpt_map: dict) -> bool:
        """posture_constraints 자세 제약 조건을 만족하는지 확인한다."""
        for c in self.posture_constraints:
            c_type = str(c.get("type", "RELATIVE_POSITION")).upper()
            min_conf = float(c.get("min_confidence", 0.4))
            if c_type == "RELATIVE_POSITION":
                target_id = self._resolve_kpt_id(c.get("target_kpt") or c.get("kpt_a"), side)
                ref_id = self._resolve_kpt_id(c.get("reference_kpt") or c.get("kpt_b"), side)
                if target_id is None or ref_id is None:
                    continue
                points = self._valid_points((ref_id, target_id), kpt_map, min_conf)
                if not points:
                    return False
                ref_pt, target_pt = points
                relation = str(c.get("relation", "ABOVE")).upper()
                margin = float(c.get("margin", 0.0))
                if relation == "ABOVE":
                    ok = (target_pt["y"] > ref_pt["y"] + margin) if self.use_normalization else (target_pt["y"] < ref_pt["y"] - margin)
                    if not ok:
                        return False
                elif relation == "BELOW":
                    ok = (target_pt["y"] < ref_pt["y"] - margin) if self.use_normalization else (target_pt["y"] > ref_pt["y"] + margin)
                    if not ok:
                        return False
        return True

    def _compute_side_value(self, side: str, kpt_map: dict):
        return {
            "ANGLE": self._compute_angle,
            "COORDINATE": self._compute_coordinate,
            "RAW_COORDINATE": self._compute_coordinate,
            "LENGTH": self._compute_length,
        }[self.metric_type](side, kpt_map)

    def process_keypoints(self, keypoints: list, now=None) -> dict:
        if self.counting_method == "TARGET_POSITION":
            points = {p["id"]: p for p in keypoints or []}
            return {
                "mode": self.mode,
                **{
                    side: fsm.update(self._compute_side_value(side, points), now)
                    for side, fsm in (("left", self.fsm_left), ("right", self.fsm_right))
                },
            }
        if not keypoints:
            return None
        kpt_map = {kp["id"]: kp for kp in keypoints}
        val_left = self._compute_side_value("left", kpt_map)
        val_right = self._compute_side_value("right", kpt_map)
        res_left = self.fsm_left.update(val_left, allow_count=self.count_posture_valid("left", kpt_map))
        res_right = self.fsm_right.update(val_right, allow_count=self.count_posture_valid("right", kpt_map))
        return {
            "mode": self.mode,
            "left": {**res_left, "val": val_left, "counting_method": self.counting_method},
            "right": {**res_right, "val": val_right, "counting_method": self.counting_method},
        }
