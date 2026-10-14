# ==============================================================================
# [파일 정보]
# 파일명: core/motion_engine.py
# 설명: 최고점(Peak) 도달 후 변곡점(Reversal) 확인 시 카운트 트리거 및 복귀 히스테리시스 FSM
#       - 조기 카운트 방지: 65% 통과 즉시 카운트되는 로직 제거
#       - 유효 동작 기준: 가동 범위 40% 이상 도달 확인
#       - 변곡점 판정: 정점(Peak) 대비 가동 범위의 5% 이상 꺾여 내려올 때 카운트 트리거
#       - ANGLE, RELATIVE_Y, DISTANCE(정규화된 관절 간 거리) 지원
# ==============================================================================

import math
import numpy as np


class SideFSM:
    def __init__(self, side_name: str, motion_direction: str, thresholds: dict, target_reps: int = 10):
        self.side = side_name
        self.is_decreasing = (motion_direction == "DECREASING")
        self.target_reps = target_reps

        self.state = "READY"
        self.rep_count = 0

        # 기본 임계값 마진
        self.start_val = thresholds.get("start_val", 0.15 if not self.is_decreasing else 0.80)
        self.target_val = thresholds.get("target_val", 0.80 if not self.is_decreasing else 0.15)

        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        self.last_quality = "READY"
        self.completed_reps_history = []

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
        if self.rep_count >= self.target_reps:
            self.state = "WAITING"
            return {
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
                })

                print(f"\n[PEAK REVERSAL COUNT!] {self.side.upper()} Rep: {self.rep_count}/{self.target_reps} | Quality: {quality} (Peak: {peak_val:.2f})")

                self.state = "WAITING" if self.rep_count >= self.target_reps else "RETURNING"

        # ----------------------------------------------------------------------
        # 3. 복귀 확인 및 락 해제 (30% 이하로 돌아와야 다음 횟수 준비)
        # ----------------------------------------------------------------------
        elif self.state == "RETURNING":
            if progress_ratio <= 0.30:
                self.state = "READY"
                self.min_val_recorded = 999.0
                self.max_val_recorded = -999.0

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "progress_ratio": progress_ratio,
            "quality": self.last_quality,
            "is_count_updated": is_count_updated
        }


class MotionEngine:
    def __init__(self, config: dict, custom_thresholds: dict = None, mode: str = "MAIN", target_reps: int = 10):
        self.config = config
        eval_cfg = config.get("eval_config", {})

        self.metric_type = eval_cfg.get("metric_type", "ANGLE")
        self.motion_direction = eval_cfg.get("motion_direction", "INCREASING")
        self.mode = mode

        default_th = config.get("default_thresholds", {})
        th_left = custom_thresholds.get("left") if custom_thresholds else default_th.get("left", {})
        th_right = custom_thresholds.get("right") if custom_thresholds else default_th.get("right", {})

        self.fsm_left = SideFSM("left", self.motion_direction, th_left, target_reps)
        self.fsm_right = SideFSM("right", self.motion_direction, th_right, target_reps)

    def _calculate_angle(self, p1: dict, p2: dict, p3: dict) -> float:
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])

        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0

        cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        return round(math.degrees(np.arccos(cosine)), 2)

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
            return None

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
            return None

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