# ==============================================================================
# [파일 정보]
# 파일명: core/motion_engine.py
# 설명: 최고점(Peak) 도달 후 변곡점(Reversal) 확인 시 카운트 트리거 및 복귀 히스테리시스 FSM
#       - 조기 카운트 방지: 65% 통과 즉시 카운트되는 로직 제거
#       - 유효 동작 기준: 가동 범위 40% 이상 도달 확인
#       - 변곡점 판정: 정점(Peak) 대비 가동 범위의 5% 이상 꺾여 내려올 때 카운트 트리거
<<<<<<< HEAD
#       - ANGLE, COORDINATE, LENGTH, RATIO(두 관절 간 거리의 비율) 지원
=======
#       - ANGLE, RELATIVE_Y, DISTANCE(정규화된 관절 간 거리) 지원
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
# ==============================================================================

import math
from numbers import Real
import numpy as np

<<<<<<< HEAD
QUALITY_POINTS = {'PERFECT': 1.0, 'GOOD': 0.5, 'BAD': 0.25}
QUALITY_CODES = {'PERFECT': 3, 'GOOD': 2, 'BAD': 1}

=======
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0

class SideFSM:
    def __init__(self, side_name: str, motion_direction: str, thresholds: dict, target_reps: int = 10,
                 target_score=None, scoring=True):
        self.side = side_name
        self.is_decreasing = (motion_direction == "DECREASING")
        self.target_reps = target_reps

        self.state = "READY"
        self.rep_count = 0
        self.total_score = 0.0
        self.target_score = float(target_reps if target_score is None else target_score)
        if not math.isfinite(self.target_score) or self.target_score <= 0:
            raise ValueError('target_score must be finite and positive')
        self.scoring = scoring

        # 기본 임계값 마진
        self.start_val = thresholds.get("start_val", 0.15 if not self.is_decreasing else 0.80)
        self.target_val = thresholds.get("target_val", 0.80 if not self.is_decreasing else 0.15)

        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        self.last_quality = "READY"
        self.completed_reps_history = []
        self.threshold_version = 0

    def replace_thresholds(self, start_val: float, target_val: float, version: int):
        """Apply validated ROM without reusing a peak from the previous range."""
        self.start_val = start_val
        self.target_val = target_val
        self.threshold_version = version
        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        # Preserve completed reps/quality. Re-arm only after returning to start
        # under the NEW range, including when the old target was unreachable.
        self.state = "WAITING" if self.target_reached else "RETURNING"

    @property
    def target_reached(self):
        # MAIN/TEST never stop on a server-side score. Client owns completion.
        return False if self.scoring else self.rep_count >= self.target_reps

    def score_status(self, awarded=False):
        return {'total_score': self.total_score, 'target_score': self.target_score,
                'score': QUALITY_CODES.get(self.last_quality, 0) if awarded else 0,
                'score_delta': QUALITY_POINTS.get(self.last_quality, 0.0) if awarded else 0.0}

    def evaluate_quality(self, peak_val: float) -> str:
        total_range = abs(self.target_val - self.start_val)
        if total_range == 0:
            return "GOOD"

        achieved_range = abs(peak_val - self.start_val)
        ratio = achieved_range / total_range

        if ratio >= 0.70:
            return "PERFECT"
        elif ratio >= 0.50:
            return "GOOD"
        elif ratio >= 0.35:
            return "BAD"
        else:
            return "INVALID"

    def update(self, current_val: float) -> dict:
        if current_val is None:
            return {
                **self.score_status(),
                "state": self.state,
                "rep_count": self.rep_count,
                "progress_ratio": 0.0,
                "quality": self.last_quality,
                "is_count_updated": False
            }

        # 1. 실시간 극값 및 실시간 진행률(게이지)은 3회를 채운 후에도 계속 계산
        self.min_val_recorded = min(self.min_val_recorded, current_val)
        self.max_val_recorded = max(self.max_val_recorded, current_val)

        val_range = abs(self.target_val - self.start_val)
        progress_ratio = float(np.clip(abs(current_val - self.start_val) / val_range, 0.0, 1.0)) if val_range > 0 else 0.0
        progress_ratio = round(progress_ratio, 2)
        is_count_updated = False

        # 2. 목표 횟수를 이미 채운 경우: 카운트 증가 FSM만 스킵하고 실시간 게이지는 계속 반환
<<<<<<< HEAD
        if self.target_reached:
            self.state = "WAITING"
            return {
                **self.score_status(),
=======
        if self.rep_count >= self.target_reps:
            self.state = "WAITING"
            return {
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
                "state": "WAITING",
                "rep_count": self.rep_count,
                "progress_ratio": progress_ratio,  # <-- 멈추지 않고 실시간 움직임 반영
                "quality": self.last_quality,
                "is_count_updated": False
            }

        # ----------------------------------------------------------------------
        # 1. 준비 -> 수축 개시 (15% 움직임 감지 시 진입)
        # ----------------------------------------------------------------------
        if self.state == "READY":
            if progress_ratio >= 0.15:
                self.state = "PUSHING"

        # ----------------------------------------------------------------------
        # 2. 수축 상승 -> 최고점(Peak)을 찍고 변곡점(하강) 진입 시 카운트 트리거
        # ----------------------------------------------------------------------
        elif self.state == "PUSHING":
            peak_val = self.min_val_recorded if self.is_decreasing else self.max_val_recorded
            current_range = abs(current_val - self.start_val)
            peak_range = abs(peak_val - self.start_val)

            # 유효 동작 최소 진입 임계치 (전체 범위의 40% 이상 도달한 적이 있어야 함)
            min_activation_threshold = 0.40
            has_reached_threshold = (peak_range / val_range >= min_activation_threshold) if val_range > 0 else False

            # 변곡점 판정: 정점 대비 5% 이상 확실히 되돌아 내려왔는지 확인 (노이즈 방지)
            reversal_delta = 0.05 * val_range
            turned_down = has_reached_threshold and ((peak_range - current_range) >= reversal_delta)

            # 조기 카운트 방지: 올라가는 도중엔 카운트하지 않고 변곡점(하강)을 통과할 때만 카운트
            if turned_down:
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

<<<<<<< HEAD
                print(f"\n[PEAK REVERSAL COUNT!] {self.side.upper()} Reps: {self.rep_count} | "
                      f"Score: {self.total_score:g}/{self.target_score:g} | Quality: {quality} (Peak: {peak_val:.2f})")

                self.state = "WAITING" if self.target_reached else "RETURNING"
=======
                print(f"\n[PEAK REVERSAL COUNT!] {self.side.upper()} Rep: {self.rep_count}/{self.target_reps} | Quality: {quality} (Peak: {peak_val:.2f})")

                self.state = "WAITING" if self.rep_count >= self.target_reps else "RETURNING"
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0

        # ----------------------------------------------------------------------
        # 3. 복귀 확인 및 락 해제 (30% 이하로 돌아와야 다음 횟수 준비)
        # ----------------------------------------------------------------------
        elif self.state == "RETURNING":
            if progress_ratio <= 0.30:
                self.state = "READY"
                self.min_val_recorded = 999.0
                self.max_val_recorded = -999.0

        return {
            **self.score_status(awarded=is_count_updated),
            "state": self.state,
            "rep_count": self.rep_count,
            "progress_ratio": progress_ratio,
            "quality": self.last_quality,
            "is_count_updated": is_count_updated
        }


class MotionEngine:
    def __init__(self, config: dict, custom_thresholds: dict = None, mode: str = "MAIN", target_reps: int = 10,
                 target_score=None):
        self.config = config
        self.exercise_name = config.get("exercise_name", "unknown")
        self.threshold_version = 0
        eval_cfg = config.get("eval_config", {})

        aliases = {'RELATIVE_Y': 'COORDINATE', 'ANKLE_HEIGHT': 'COORDINATE', 'DISTANCE': 'LENGTH'}
        self.metric_type = aliases.get(eval_cfg.get('metric_type'), eval_cfg.get('metric_type', 'ANGLE'))
        if self.metric_type not in {'ANGLE', 'COORDINATE', 'LENGTH', 'RATIO'}:
            raise ValueError('metric_type must be ANGLE, COORDINATE, LENGTH or RATIO')
        if self.metric_type == 'RATIO':
            for field in ('numerator_kpts', 'denominator_kpts'):
                for side in ('left', 'right'):
                    pair = eval_cfg.get(field, {}).get(side)
                    if (not isinstance(pair, list) or len(pair) != 2
                            or any(type(i) is not int or not 0 <= i < 17 for i in pair)
                            or pair[0] == pair[1]):
                        raise ValueError(f'RATIO {field}.{side} requires two distinct COCO joint IDs')
        self.motion_direction = eval_cfg.get("motion_direction", "INCREASING")
        self.mode = mode

        if (eval_cfg.get('reference_kpt') or self.metric_type == 'RATIO') and custom_thresholds and any(
                aliases.get(custom_thresholds.get(side, {}).get('metric_type'),
                            custom_thresholds.get(side, {}).get('metric_type')) != self.metric_type
                for side in ('left', 'right')):
            raise ValueError('Measurement metric changed; run CALIBRATION again before MAIN/TEST')

        default_th = config.get("default_thresholds", {})
        th_left = custom_thresholds.get("left") if custom_thresholds else default_th.get("left", {})
        th_right = custom_thresholds.get("right") if custom_thresholds else default_th.get("right", {})

        self.fsm_left = SideFSM("left", th_left.get("motion_direction", self.motion_direction), th_left, target_reps, target_score, mode != 'CALIBRATION')
        self.fsm_right = SideFSM("right", th_right.get("motion_direction", self.motion_direction), th_right, target_reps, target_score, mode != 'CALIBRATION')

    def get_count_thresholds(self) -> dict:
        """Return a copy of each side's reference ROM, in the metric's units."""
        return {
            side: {"start_val": fsm.start_val, "target_val": fsm.target_val}
            for side, fsm in (("left", self.fsm_left), ("right", self.fsm_right))
        }

    def count_threshold_status(self) -> dict:
        return {
            "exercise_name": self.exercise_name,
            "metric_type": self.metric_type,
            "motion_direction": self.motion_direction,
            "threshold_version": self.threshold_version,
            "thresholds": self.get_count_thresholds(),
            "target_score": self.fsm_left.target_score,
            "scoring_mode": "REPETITIONS" if self.mode == 'CALIBRATION' else "CLIENT",
            "completion_owner": "SERVER" if self.mode == 'CALIBRATION' else "CLIENT",
        }

    def update_count_thresholds(self, thresholds: dict) -> list:
        """Validate both sides atomically, then update only changed sides.

        A partial patch (e.g. just left.target_val) preserves omitted values.
        This changes reference ROM, not the 40% activation/quality ratios.
        """
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
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(f"{side}.{key} must be a finite number")
                try:
                    converted = float(value)
                except (ValueError, OverflowError):
                    raise ValueError(f"{side}.{key} must be a finite number") from None
                if not math.isfinite(converted):
                    raise ValueError(f"{side}.{key} must be a finite number")
                values[key] = converted
            start, target = values["start_val"], values["target_val"]
            if not math.isfinite(target - start):
                raise ValueError(f"{side}: ROM must be finite")
            fsm = self.fsm_left if side == "left" else self.fsm_right
            if fsm.is_decreasing:
                if target >= start:
                    raise ValueError(f"{side}: DECREASING requires target_val < start_val")
            elif target <= start:
                raise ValueError(f"{side}: INCREASING requires target_val > start_val")
            if self.metric_type == "ANGLE" and not all(0 <= v <= 180 for v in values.values()):
                raise ValueError(f"{side}: ANGLE values must be between 0 and 180 degrees")
            if self.metric_type in {"LENGTH", "RATIO"} and any(v < 0 for v in values.values()):
                raise ValueError(f"{side}: {self.metric_type} values must be nonnegative")
            if values != current[side]:
                validated[side] = values

        if validated:
            self.threshold_version += 1
            for side, values in validated.items():
                fsm = self.fsm_left if side == "left" else self.fsm_right
                fsm.replace_thresholds(**values, version=self.threshold_version)
        return list(validated)

    def _calculate_angle(self, p1: dict, p2: dict, p3: dict) -> float:
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        return round(math.degrees(np.arccos(cosine)), 2)

<<<<<<< HEAD
    def _valid_points(self, ids, kpt_map, confidence=.35):
        points = [kpt_map.get(i) for i in ids]
        if not points or not all(p and p.get('score', 0.) >= confidence
                and all(math.isfinite(p[axis]) for axis in ('x', 'y')) for p in points):
=======
    def _get_torso_scale(self, kpt_map: dict) -> float:
        """카메라 거리에 무관한 신체 비율 기준치 계산 (어깨 중심과 골반 중심 간 거리)"""
        # COCO: Left Shoulder=5, Right Shoulder=6, Left Hip=11, Right Hip=12
        ls, rs = kpt_map.get(5), kpt_map.get(6)
        lh, rh = kpt_map.get(11), kpt_map.get(12)

        if ls and rs and lh and rh:
            shoulder_mid_y = (ls["y"] + rs["y"]) / 2.0
            shoulder_mid_x = (ls["x"] + rs["x"]) / 2.0
            hip_mid_y = (lh["y"] + rh["y"]) / 2.0
            hip_mid_x = (lh["x"] + rh["x"]) / 2.0
            torso_len = math.hypot(shoulder_mid_x - hip_mid_x, shoulder_mid_y - hip_mid_y)
            if torso_len > 10.0:  # 유효한 픽셀 크기인지 확인
                return torso_len

        return 200.0  # 기본 스케일 fallback (정규화용 분모)

    def _compute_side_value(self, side: str, kpt_map: dict) -> float:
        eval_cfg = self.config.get("eval_config", {})

        # 1. Y축 상대 높이 기반
        if self.metric_type == "RELATIVE_Y":
            target_id = eval_cfg.get("target_kpt", {}).get(side)
            target_kp = kpt_map.get(target_id)
            if target_kp and target_kp.get("score", 0.0) >= 0.35:
                return round(float(target_kp["y"]), 3)
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
            return None
        return points

<<<<<<< HEAD
    def _compute_angle(self, side, kpt_map):
        ids = self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])
        points = self._valid_points(ids, kpt_map, .4)
        if points and len(points) == 3:
            return self._calculate_angle(*points)
        return None

    def _compute_coordinate(self, side, kpt_map):
        cfg = self.config.get('eval_config', {})
        axis = cfg.get('axis', 'y')
        if axis not in ('x', 'y'):
            raise ValueError('COORDINATE axis must be x or y')
        target = cfg.get('target_kpt', {}).get(side)
        reference = cfg.get('reference_kpt', {}).get(side)
        # Legacy two-point input is interpreted as reference -> target.
        if target is None:
            pair = cfg.get('primary_kpts', {}).get(side, [])
            if len(pair) != 2:
                return None
            reference, target = pair
        points = self._valid_points([target] + ([reference] if reference is not None else []), kpt_map)
        if not points:
=======
        # 2. 거리 기반 (무릎-발목 간 정규화 거리)
        elif self.metric_type == "DISTANCE":
            p_indices = eval_cfg.get("primary_kpts", {}).get(side, [])
            if len(p_indices) >= 2:
                kp1 = kpt_map.get(p_indices[0])
                kp2 = kpt_map.get(p_indices[1])

                if kp1 and kp2 and kp1.get("score", 0.0) >= 0.35 and kp2.get("score", 0.0) >= 0.35:
                    raw_dist = math.hypot(kp1["x"] - kp2["x"], kp1["y"] - kp2["y"])
                    scale = self._get_torso_scale(kpt_map)
                    normalized_dist = raw_dist / scale
                    return round(float(normalized_dist), 3)
            return None

        # 3. 3관절 사이 각도 기반 (기본값)
        else:
            p_indices = eval_cfg.get("primary_kpts", {}).get(side, [])
            pts = [kpt_map.get(i) for i in p_indices if kpt_map.get(i)]
            if len(pts) == 3 and all(p.get("score", 0.0) >= 0.4 for p in pts):
                return self._calculate_angle(pts[0], pts[1], pts[2])
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
            return None
        return round(float(points[0][axis] - (points[1][axis] if reference is not None else 0.)), 4)

    def _compute_length(self, side, kpt_map):
        ids = self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])
        points = self._valid_points(ids, kpt_map)
        if points and len(points) == 2:
            # Coordinates are already normalized by PoseNormalizer.
            return round(math.hypot(points[1]['x']-points[0]['x'], points[1]['y']-points[0]['y']), 4)
        return None

    def _compute_ratio(self, side, kpt_map):
        cfg = self.config['eval_config']
        numerator = self._valid_points(cfg['numerator_kpts'][side], kpt_map)
        denominator = self._valid_points(cfg['denominator_kpts'][side], kpt_map)
        if not numerator or not denominator:
            return None
        def distance(points):
            return math.hypot(points[1]['x'] - points[0]['x'],
                              points[1]['y'] - points[0]['y'])
        bottom = distance(denominator)
        if not math.isfinite(bottom) or bottom <= 1e-6:
            return None
        # Use unrounded lengths: the common torso normalization scale cancels.
        ratio = distance(numerator) / bottom
        return round(ratio, 4) if math.isfinite(ratio) else None

    def _compute_side_value(self, side: str, kpt_map: dict):
        return {'ANGLE': self._compute_angle, 'COORDINATE': self._compute_coordinate,
                'LENGTH': self._compute_length, 'RATIO': self._compute_ratio}[self.metric_type](side, kpt_map)

    def process_keypoints(self, keypoints: list) -> dict:
        if not keypoints:
            return None

        kpt_map = {kp["id"]: kp for kp in keypoints}

        val_left = self._compute_side_value("left", kpt_map)
        val_right = self._compute_side_value("right", kpt_map)

        res_left = self.fsm_left.update(val_left)
        res_right = self.fsm_right.update(val_right)

        return {
            "mode": self.mode,
            "left": {**res_left, "val": val_left},
            "right": {**res_right, "val": val_right}
        }
