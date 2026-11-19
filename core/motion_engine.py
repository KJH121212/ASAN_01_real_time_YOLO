# 설명: 최고점(Peak) 도달 후 변곡점(Reversal) 확인 시 카운트 트리거 및 복귀 히스테리시스 FSM
#       - 조기 카운트 방지: 65% 통과 즉시 카운트되는 로직 제거
#       - 유효 동작 기준: 가동 범위 40% 이상 도달 확인
#       - 변곡점 판정: 정점(Peak) 대비 가동 범위의 5% 이상 꺾여 내려올 때 카운트 트리거

import math
from copy import deepcopy
from numbers import Real
import numpy as np
from utils.measurement_progress import (
    measurement_progress, CALIBRATION_ACTIVATION_FRACTION, CALIBRATION_REARM_FRACTION,
)

QUALITY_POINTS = {'PERFECT': 1.0, 'GOOD': 0.5, 'BAD': 0.25}
QUALITY_THRESHOLDS = {'PERFECT': .70, 'GOOD': .50, 'BAD': .35}
QUALITY_CODES = {'PERFECT': 3, 'GOOD': 2, 'BAD': 1}

class SideFSM:
    def __init__(self, side_name: str, motion_direction: str, thresholds: dict, target_reps: int = 10,
                 target_score=None, scoring=True, min_activation_ratio=.40, return_ratio=.30, reversal_ratio=.05):
        self.side = side_name
        if not math.isfinite(min_activation_ratio) or not 0 < min_activation_ratio <= 1:
            raise ValueError('min_activation_ratio must be between 0 and 1')
        self.min_activation_ratio = min_activation_ratio
        if not math.isfinite(return_ratio) or not 0 <= return_ratio < min_activation_ratio:
            raise ValueError('return_ratio must be below min_activation_ratio')
        self.return_ratio = return_ratio
        if not math.isfinite(reversal_ratio) or not 0 <= reversal_ratio <= 1:
            raise ValueError('reversal_ratio must be between 0 and 1')
        self.reversal_ratio = float(reversal_ratio)
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

    # 이미 검증한 새 가동 범위를 적용하고 이전 동작의 최고점은 버립니다. 완료 횟수·점수는 유지합니다.
    def replace_thresholds(self, start_val: float, target_val: float, version: int):
        """Apply validated ROM without reusing a peak from the previous range."""
        self.start_val = start_val
        self.target_val = target_val
        self.threshold_version = version
        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        # 이미 완료된 횟수·품질은 유지하며 시작 자세 쪽으로 복귀한 뒤 다시 동작을 받습니다.
        # 이전 목표에 도달하지 못했어도 새 가동 범위 기준으로 복귀를 판단합니다.
        self.state = "WAITING" if self.target_reached else "RETURNING"

    @property
    # 캘리브레이션 횟수 목표만 확인합니다. MAIN의 서버 점수만으로 게임을 끝내지 않습니다.
    def target_reached(self):
        # MAIN는 서버 점수만으로 끝내지 않습니다. 종료는 클라이언트가 결정합니다.
        # MAIN는 이 서버 목표값으로 중단하지 않습니다. 캘리브레이션 용도로 사용하는 횟수 목표만 서버에서 완료 판정합니다.
        return False if self.scoring else self.rep_count >= self.target_reps

    def score_status(self, awarded=False):
        return {'total_score': self.total_score, 'target_score': self.target_score,
                'score': QUALITY_CODES.get(self.last_quality, 0) if awarded else 0,
                'score_delta': QUALITY_POINTS.get(self.last_quality, 0.0) if awarded else 0.0}

    # 기준 가동 범위 대비 최고 도달 비율로 PERFECT·GOOD·BAD·INVALID 등급을 구합니다.
    def evaluate_quality(self, peak_val: float) -> str:
        total_range = abs(self.target_val - self.start_val)
        if total_range == 0:
            return "GOOD"

        achieved_range = abs(peak_val - self.start_val)
        ratio = achieved_range / total_range

        if ratio + 1e-9 >= QUALITY_THRESHOLDS['PERFECT']:
            return "PERFECT"
        elif ratio + 1e-9 >= QUALITY_THRESHOLDS['GOOD']:
            return "GOOD"
        elif ratio + 1e-9 >= QUALITY_THRESHOLDS['BAD']:
            return "BAD"
        else:
            return "INVALID"

    # 15%에서 동작을 시작하고, 40% 이상 도달한 뒤 5% 되돌아오면 1회로 셉니다. 30% 이하 복귀 후 다음 동작을 받습니다.
    def update(self, current_val: float, allow_count: bool = True) -> dict:
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

        # Reverse movement is not progress toward the configured target.
        val_range = abs(self.target_val - self.start_val)
        progress_ratio = measurement_progress(current_val, 'COORDINATE', self.start_val, self.target_val) or 0.
        progress_ratio = round(progress_ratio, 2)
        is_count_updated = False

        # 2. 목표 횟수를 이미 채운 경우: 카운트 증가 FSM만 스킵하고 실시간 게이지는 계속 반환
        if self.target_reached:
            self.state = "WAITING"
            return {
                **self.score_status(),
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
            # 설정 가동 범위의 15% 이상 움직이면 READY에서 PUSHING으로 전환합니다. 이 시점에는 아직 횟수를 올리지 않습니다.
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
            # 최고 도달 범위가 전체 기준 범위의 40% 이상이어야 반복 후보가 됩니다. 이 기준과 품질 등급 기준은 별개입니다.
            min_activation_threshold = self.min_activation_ratio
            has_reached_threshold = (peak_range / val_range + 1e-9 >= min_activation_threshold) if val_range > 0 else False

            # 변곡점 판정: 정점 대비 설정 비율(기본 5%) 이상 확실히 되돌아 내려왔는지 확인 (노이즈 방지)
            # 전체 기준 범위의 5%를 방향 전환 확인 폭으로 사용합니다. 최고점 도달 직후 작은 잡음으로 반복을 확정하지 않게 합니다.
            reversal_delta = self.reversal_ratio * val_range
            turned_down = has_reached_threshold and ((peak_range - current_range) >= reversal_delta)

            # 조기 카운트 방지: 올라가는 도중엔 카운트하지 않고 변곡점(하강)을 통과할 때만 카운트
            # 충분한 가동 범위에 도달한 뒤 복귀 방향 전환까지 확인된 프레임에서만 한 번 카운트합니다.
            if turned_down and not allow_count:
                # Consume an invalid attempt so its peak cannot count later.
                self.state = "RETURNING"
            if turned_down and allow_count:
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

                print(f"\n[PEAK REVERSAL COUNT!] {self.side.upper()} Reps: {self.rep_count} | "
                      f"Score: {self.total_score:g}/{self.target_score:g} | Quality: {quality} (Peak: {peak_val:.2f})")

                self.state = "WAITING" if self.target_reached else "RETURNING"

        # ----------------------------------------------------------------------
        # 3. 복귀 확인 및 락 해제 (30% 이하로 돌아와야 다음 횟수 준비)
        # ----------------------------------------------------------------------
        elif self.state == "RETURNING":
            # 시작 자세 쪽 30% 범위 안으로 복귀해야 다음 반복을 시작할 수 있습니다. 최고점 부근에서 반복 카운트되는 것을 막습니다.
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
            "is_count_updated": is_count_updated
        }

def config_for_mode(config, mode):
    result = deepcopy(config)
    if mode == 'MAIN' and result.get('main_eval_config'):
        result['eval_config'] = {**result.get('eval_config', {}), **result['main_eval_config']}
    return result

class MotionEngine:
    def __init__(self, config, custom_thresholds=None, mode='MAIN', target_reps=10, target_score=None):
        if mode not in {'CALIBRATION', 'MAIN'}:
            raise ValueError('mode must be CALIBRATION or MAIN')
        self.config = config = config_for_mode(config, mode)
        self.exercise_name = config.get('exercise_name', 'unknown')
        self.threshold_version = 0
        self.mode = mode
        cfg = config.get('eval_config', {})
        aliases = {'RELATIVE_Y': 'COORDINATE', 'ANKLE_HEIGHT': 'COORDINATE', 'DISTANCE': 'LENGTH', 'UNNORM_COORDINATE': 'RAW_COORDINATE'}
        self.metric_type = aliases.get(cfg.get('metric_type'), cfg.get('metric_type', 'ANGLE'))
        if self.metric_type not in {'ANGLE', 'COORDINATE', 'RAW_COORDINATE', 'LENGTH'}:
            raise ValueError('metric_type must be ANGLE, COORDINATE, RAW_COORDINATE or LENGTH')
        default_norm = False if self.metric_type == 'RAW_COORDINATE' else True
        self.use_normalization = cfg.get('use_normalization', default_norm)
        if type(self.use_normalization) is not bool:
            raise ValueError('use_normalization must be true or false')
        if self.metric_type == 'RAW_COORDINATE' and self.use_normalization:
            raise ValueError('RAW_COORDINATE requires use_normalization=false')
        self.coordinate_unit = cfg.get('coordinate_unit', 'image')
        if self.coordinate_unit not in {'image', 'pixel'}:
            raise ValueError('coordinate_unit must be image or pixel')
        if self.coordinate_unit == 'pixel' and (self.metric_type not in {'COORDINATE', 'RAW_COORDINATE'} or self.use_normalization):
            raise ValueError('Pixel coordinates require COORDINATE/RAW_COORDINATE and use_normalization=false')
        self.motion_direction = cfg.get('motion_direction', 'DECREASING' if self.metric_type == 'RAW_COORDINATE' else 'INCREASING')
        self.counting_method = cfg.get('counting_method', 'PEAK_REVERSAL')
        if self.counting_method not in {'PEAK_REVERSAL', 'TARGET_POSITION'}:
            raise ValueError('Unsupported counting_method')
        self.posture_constraints = cfg.get('posture_constraints') or config.get('posture_constraints') or []
        self.frame_size = None
        self.calibration_display_thresholds = {}
        self.expected_frame_size = None
        for side in ('left', 'right'):
            saved = custom_thresholds.get(side, {}) if custom_thresholds else None
            if saved is not None:
                if saved.get('use_normalization', True) != self.use_normalization:
                    raise ValueError('Normalization changed; run CALIBRATION again before MAIN')
                tag = saved.get('metric_type')
                if tag is not None:
                    norm_tag = aliases.get(tag, tag)
                    if norm_tag != self.metric_type and not ({norm_tag, self.metric_type} <= {'COORDINATE', 'RAW_COORDINATE'} and not self.use_normalization):
                        raise ValueError('Measurement metric changed; run CALIBRATION again before MAIN')
            if self.metric_type in {'COORDINATE', 'RAW_COORDINATE'}:
                if cfg.get('axis', 'y') not in ('x', 'y'):
                    raise ValueError('COORDINATE axis must be x or y')
                joint = cfg.get('target_kpt', {}).get(side)
                if joint is None:
                    pk = cfg.get('primary_kpts', {}).get(side, [])
                    joint = pk[0] if isinstance(pk, list) and len(pk) == 1 else None
                if joint is not None and (type(joint) is not int or not 0 <= joint < 17):
                    raise ValueError('target_kpt/primary_kpts must contain COCO joint IDs')
                if self.coordinate_unit == 'pixel':
                    if joint is None:
                        raise ValueError('Pixel coordinates require target_kpt')
                    if saved is not None:
                        signature = dict(coordinate_unit='pixel', axis=cfg.get('axis', 'y'),
                            target_kpt=joint)
                        if any(saved.get(k) != v for k, v in signature.items()):
                            raise ValueError('Coordinate definition changed; run CALIBRATION again before MAIN')
                        size = saved.get('frame_size')
                        if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
                            raise ValueError('Pixel calibration requires a recorded frame_size; run CALIBRATION again')
                        if self.expected_frame_size is not None and self.expected_frame_size != size:
                            raise ValueError('Left/right calibration resolutions differ')
                        self.expected_frame_size = size
            elif self.metric_type == 'LENGTH':
                pair = cfg.get('primary_kpts', {}).get(side, [])
                if not isinstance(pair, list) or len(pair) != 2 or any(type(j) is not int or not 0 <= j < 17 for j in pair):
                    raise ValueError('LENGTH requires 2 COCO joint IDs in primary_kpts')
            elif self.metric_type == 'ANGLE':
                triplet = cfg.get('primary_kpts', {}).get(side, [])
                if not isinstance(triplet, list) or len(triplet) != 3 or any(type(j) is not int or not 0 <= j < 17 for j in triplet):
                    raise ValueError('ANGLE requires 3 COCO joint IDs in primary_kpts')
        if self.expected_frame_size is not None:
            self.frame_size = list(self.expected_frame_size)
        defaults = config.get('default_thresholds', {})
        left = custom_thresholds['left'] if custom_thresholds else defaults.get('left', {})
        right = custom_thresholds['right'] if custom_thresholds else defaults.get('right', {})
        self.fsm_left = SideFSM('left', left.get('motion_direction', self.motion_direction), left,
            target_reps, target_score, mode != 'CALIBRATION', cfg.get('min_activation_ratio', .40), cfg.get('return_ratio', .30))
        self.fsm_right = SideFSM('right', right.get('motion_direction', self.motion_direction), right,
            target_reps, target_score, mode != 'CALIBRATION', cfg.get('min_activation_ratio', .40), cfg.get('return_ratio', .30))
        if self.counting_method == 'TARGET_POSITION':
            if self.metric_type != 'COORDINATE' or mode != 'MAIN' or not custom_thresholds:
                raise ValueError('TARGET_POSITION requires MAIN coordinate measurement and calibrated thresholds')
            from core.target_position import TargetPositionSide
            self.fsm_left = TargetPositionSide('left', self.motion_direction, left, target_reps, target_score)
            self.fsm_right = TargetPositionSide('right', self.motion_direction, right, target_reps, target_score)

    @property
    def metric_unit(self):
        return ('px' if self.coordinate_unit == 'pixel' else 'deg' if self.metric_type == 'ANGLE'
                else 'body_length' if self.use_normalization else 'image_height')

    def set_frame_size(self, width, height):
        size = [width, height]
        if any(type(v) is not int or v <= 0 for v in size):
            raise ValueError('Pixel measurement requires positive image_width/image_height')
        if self.expected_frame_size is not None and size != self.expected_frame_size:
            raise ValueError('Camera resolution changed; run CALIBRATION again before MAIN')
        if self.frame_size is not None and self.frame_size != size:
            raise ValueError('Camera resolution changed during measurement')
        self.frame_size = size

    def get_display_thresholds(self):
        if self.mode == 'CALIBRATION':
            defaults = self.config.get('default_thresholds', {})
            return {side: dict(self.calibration_display_thresholds.get(side, defaults.get(side, self.get_count_thresholds()[side])))
                    for side in ('left', 'right')}
        return self.get_count_thresholds()

    def display_progress(self, side, value):
        bounds = self.get_display_thresholds()[side]
        return measurement_progress(value, 'COORDINATE', bounds['start_val'], bounds['target_val'])

    def get_count_thresholds(self) -> dict:
        """Return a copy of each side's reference ROM, in the metric's units."""
        return {
            side: {"start_val": fsm.start_val, "target_val": fsm.target_val}
            for side, fsm in (("left", self.fsm_left), ("right", self.fsm_right))
        }

    def count_threshold_status(self):
        cal = self.config.get('calibration_config', {})
        adaptive = not cal.get('use_config_thresholds', True)
        return dict(exercise_name=self.exercise_name, metric_type=self.metric_type, metric_unit=self.metric_unit,
            display_thresholds=self.get_display_thresholds(),
            measurement={**self.config['eval_config'], 'use_normalization': self.use_normalization,
                'metric_unit': self.metric_unit, 'frame_size': self.frame_size,
                'display_thresholds': self.get_display_thresholds(),
                'calibration_threshold_source': 'ADAPTIVE' if adaptive else 'CONFIG',
                'calibration_min_movement': cal.get('min_rise') if adaptive else None,
                'calibration_activation_fraction': None if adaptive else CALIBRATION_ACTIVATION_FRACTION,
                'calibration_rearm_fraction': cal.get('rearm_fraction', CALIBRATION_REARM_FRACTION),
                'calibration_count_on': 'TARGET_CROSSING',
                'count_on': 'TARGET_CROSSING' if self.mode == 'CALIBRATION' else self.counting_method},
            motion_direction=self.motion_direction, threshold_version=self.threshold_version,
            thresholds=self.get_count_thresholds(), target_score=self.fsm_left.target_score,
            scoring_mode='REPETITIONS' if self.mode == 'CALIBRATION' else 'CLIENT',
            completion_owner='SERVER' if self.mode == 'CALIBRATION' else 'CLIENT', target_reps=self.fsm_left.target_reps)

    # 부분 가동 범위 변경 요청을 검증하고 판정 기준을 갱신합니다. 진행 중이던 동작은 새 범위에서 복귀 후 다시 시작합니다.
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
            if self.metric_type == "LENGTH" and any(v < 0 for v in values.values()):
                raise ValueError(f"{side}: {self.metric_type} values must be nonnegative")
            if values != current[side]:
                validated[side] = values

        if validated:
            # 현재 가동 범위 설정의 변경 번호에 `1`을 더해 누적합니다. 대상: `self.threshold_version`.
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

    def _valid_points(self, ids, kpt_map, confidence=.35):
        points = [kpt_map.get(i) for i in ids]
        if not points or not all(p and p.get('score', 0.) >= confidence
                and all(math.isfinite(p[axis]) for axis in ('x', 'y')) for p in points):
            return None
        return points

    # 정해진 3가지 측정 방식(ANGLE / COORDINATE / LENGTH) 중 설정(eval_config.metric_type)에 지정된 방식을 호출합니다.
    # 1) ANGLE: 설정된 세 관절(primary_kpts)로 가운데 관절의 각도를 계산합니다.
    def _compute_angle(self, side, kpt_map):
        ids = self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])
        points = self._valid_points(ids, kpt_map, .4)
        if points and len(points) == 3:
            return self._calculate_angle(*points)
        return None

    # 2) COORDINATE: 설정된 단일 대상 관절(target_kpt)의 Y좌표(또는 설정 축 axis) 위치를 구합니다.
    def _compute_coordinate(self, side, kpt_map):
        cfg = self.config.get('eval_config', {})
        axis = cfg.get('axis', 'y')
        if axis not in ('x', 'y'):
            raise ValueError('COORDINATE axis must be x or y')
        target = cfg.get('target_kpt', {}).get(side)
        if target is None:
            pts = cfg.get('primary_kpts', {}).get(side, [])
            target = pts[0] if len(pts) == 1 else None
        points = self._valid_points([target] if target is not None else [], kpt_map)
        if not points:
            return None
        return round(float(points[0][axis]), 4)

    # 3) LENGTH: 설정된 두 관절(primary_kpts) 사이의 2D 유클리드 길이를 구합니다.
    def _compute_length(self, side, kpt_map):
        ids = self.config.get('eval_config', {}).get('primary_kpts', {}).get(side, [])
        points = self._valid_points(ids, kpt_map)
        if points and len(points) == 2:
            return round(math.hypot(points[1]['x'] - points[0]['x'], points[1]['y'] - points[0]['y']), 4)
        return None

    @staticmethod
    def _resolve_kpt_id(kpt_spec, side: str):
        if isinstance(kpt_spec, dict):
            return kpt_spec.get(side)
        return kpt_spec

    def count_posture_valid(self, side: str, kpt_map: dict) -> bool:
        """JSON 설정(posture_constraints)에 정의된 추가 자세 제약 조건을 검증합니다."""
        constraints = self.posture_constraints
        if not constraints:
            return True

        for c in constraints:
            c_type = str(c.get('type', 'RELATIVE_POSITION')).upper()
            min_conf = float(c.get('min_confidence', 0.4))

            if c_type == 'RELATIVE_POSITION':
                target_id = self._resolve_kpt_id(c.get('target_kpt') or c.get('kpt_a'), side)
                ref_id = self._resolve_kpt_id(c.get('reference_kpt') or c.get('kpt_b'), side)
                if target_id is None or ref_id is None:
                    continue
                points = self._valid_points((ref_id, target_id), kpt_map, min_conf)
                if not points:
                    return False
                ref_pt, target_pt = points
                axis = c.get('axis', 'y').lower()
                relation = str(c.get('relation', 'ABOVE')).upper()
                margin = float(c.get('margin', 0.0))

                if relation == 'ABOVE':
                    is_above = (target_pt['y'] > ref_pt['y'] + margin if self.use_normalization
                                else target_pt['y'] < ref_pt['y'] - margin)
                    if not is_above:
                        return False
                elif relation == 'BELOW':
                    is_below = (target_pt['y'] < ref_pt['y'] - margin if self.use_normalization
                                else target_pt['y'] > ref_pt['y'] + margin)
                    if not is_below:
                        return False
                elif relation == 'LEFT_OF':
                    if not (target_pt['x'] < ref_pt['x'] - margin):
                        return False
                elif relation == 'RIGHT_OF':
                    if not (target_pt['x'] > ref_pt['x'] + margin):
                        return False
                elif relation in ('>', 'GREATER_THAN'):
                    if not (target_pt[axis] > ref_pt[axis] + margin):
                        return False
                elif relation in ('<', 'LESS_THAN'):
                    if not (target_pt[axis] < ref_pt[axis] - margin):
                        return False

            elif c_type == 'ANGLE':
                kpt_ids = self._resolve_kpt_id(c.get('primary_kpts') or c.get('kpts'), side)
                if not kpt_ids or len(kpt_ids) != 3:
                    continue
                points = self._valid_points(kpt_ids, kpt_map, min_conf)
                if not points or len(points) != 3:
                    return False
                ang = self._calculate_angle(*points)
                if 'min_angle' in c and ang < float(c['min_angle']):
                    return False
                if 'max_angle' in c and ang > float(c['max_angle']):
                    return False

            elif c_type == 'COORDINATE_BOUND':
                target_id = self._resolve_kpt_id(c.get('target_kpt'), side)
                if target_id is None:
                    continue
                points = self._valid_points([target_id], kpt_map, min_conf)
                if not points:
                    return False
                axis = c.get('axis', 'y').lower()
                val = points[0][axis]
                if 'min_val' in c and val < float(c['min_val']):
                    return False
                if 'max_val' in c and val > float(c['max_val']):
                    return False

        return True

    def _compute_side_value(self, side: str, kpt_map: dict):
        return {'ANGLE': self._compute_angle, 'COORDINATE': self._compute_coordinate,
                'RAW_COORDINATE': self._compute_coordinate, 'LENGTH': self._compute_length}[self.metric_type](side, kpt_map)

    # 관절 ID별 조회 표를 만든 뒤 좌우 측정값과 각 상태 기계의 반복·품질 결과를 함께 반환합니다.
    def process_keypoints(self, keypoints: list, now=None) -> dict:
        if self.counting_method == 'TARGET_POSITION':
            points = {p['id']: p for p in keypoints or []}
            return {'mode': self.mode, **{side: fsm.update(self._compute_side_value(side, points), now)
                    for side, fsm in (('left', self.fsm_left), ('right', self.fsm_right))}}
        if not keypoints:
            return None

        kpt_map = {kp["id"]: kp for kp in keypoints}

        val_left = self._compute_side_value("left", kpt_map)
        val_right = self._compute_side_value("right", kpt_map)

        if self.coordinate_unit == "pixel":
            for value, fsm in ((val_left, self.fsm_left), (val_right, self.fsm_right)):
                if value is None:
                    fsm.state, fsm.min_val_recorded, fsm.max_val_recorded = "RETURNING", 999.0, -999.0
        res_left = self.fsm_left.update(val_left, allow_count=self.count_posture_valid('left', kpt_map))
        res_right = self.fsm_right.update(val_right, allow_count=self.count_posture_valid('right', kpt_map))

        return {
            "mode": self.mode,
            "left": {**res_left, "val": val_left, "counting_method": self.counting_method},
            "right": {**res_right, "val": val_right, "counting_method": self.counting_method}
        }
