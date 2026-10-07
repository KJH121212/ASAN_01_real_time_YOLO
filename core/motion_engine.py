# ==============================================================================
# [파일 정보]
# 파일명: core/motion_engine.py
# 설명: 최고점(Peak) 도달 즉시 카운트 트리거 및 복귀 히스테리시스(Lock) FSM 모듈
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
        if self.rep_count >= self.target_reps:
            self.state = "WAITING"
            return {
                "state": "WAITING",
                "rep_count": self.rep_count,
                "progress_ratio": 1.0,
                "quality": "FINISHED",
                "is_count_updated": False
            }

        if current_val is None:
            return {
                "state": self.state,
                "rep_count": self.rep_count,
                "progress_ratio": 0.0,
                "quality": self.last_quality,
                "is_count_updated": False
            }

        # 극값 실시간 누적 추적
        self.min_val_recorded = min(self.min_val_recorded, current_val)
        self.max_val_recorded = max(self.max_val_recorded, current_val)

        val_range = abs(self.target_val - self.start_val)
        progress_ratio = float(np.clip(abs(current_val - self.start_val) / val_range, 0.0, 1.0)) if val_range > 0 else 0.0
        progress_ratio = round(progress_ratio, 2)
        is_count_updated = False

        # 1. 준비 -> 수축 개시 (15%만 넘어도 반응)
        if self.state == "READY":
            if progress_ratio >= 0.15:
                self.state = "PUSHING"

        # 2. 수축 상승 -> 정점 터치 즉시 카운트
        elif self.state == "PUSHING":
            peak_val = self.min_val_recorded if self.is_decreasing else self.max_val_recorded
            current_range = abs(current_val - self.start_val)
            peak_range = abs(peak_val - self.start_val)

            # [개선] 65% 이상 찍으면 정점 판정! 또는 45% 이상에서 살짝(2%)만 꺾여도 즉시 카운트!
            reached_target = (progress_ratio >= 0.65)
            turned_down = (progress_ratio >= 0.45) and (peak_range - current_range >= 0.02 * val_range)

            if reached_target or turned_down:
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
                    "min_angle": round(self.min_val_recorded, 2),
                    "max_angle": round(self.max_val_recorded, 2),
                    "achieved_rom": round(abs(self.max_val_recorded - self.min_val_recorded), 2),
                    "quality": quality,
                })

                print(f"\n[PEAK HIT COUNT!] {self.side.upper()} Rep: {self.rep_count}/{self.target_reps} | Quality: {quality} (Peak: {peak_val:.2f})")

                # 정점 즉시 중복 카운트 방지 락
                self.state = "WAITING" if self.rep_count >= self.target_reps else "RETURNING"

        # 3. 복귀 락 해제 (30% 이하로 적당히 내리기만 해도 다음 회차 준비 완료)
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

    def _compute_side_value(self, side: str, kpt_map: dict) -> float:
        eval_cfg = self.config.get("eval_config", {})

        if self.metric_type == "RELATIVE_Y":
            target_id = eval_cfg.get("target_kpt", {}).get(side)
            target_kp = kpt_map.get(target_id)
            if target_kp and target_kp.get("score", 0.0) >= 0.35:
                return round(float(target_kp["y"]), 3)
            return None

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