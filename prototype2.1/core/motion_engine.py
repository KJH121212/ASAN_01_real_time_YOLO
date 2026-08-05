import math
import numpy as np


class SideFSM:
    """좌/우 단일 관절의 FSM 상태 및 카운트를 독립적으로 관리하는 서브 클래스"""
    def __init__(self, side_name: str, motion_direction: str, thresholds: dict):
        self.side = side_name
        self.is_decreasing = (motion_direction == "DECREASING")
        self.state = "READY"
        self.rep_count = 0
        self.peak_reached = False
        
        self.start_val = thresholds.get("start_val", 160.0 if self.is_decreasing else 90.0)
        self.target_val = thresholds.get("target_val", 50.0 if self.is_decreasing else 160.0)
        
        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        self.calibration_history = []

    def update(self, current_val: float) -> dict:
        if current_val is None:
            return {
                "state": self.state,
                "rep_count": self.rep_count,
                "progress_ratio": 0.0,
                "is_count_updated": False
            }

        self.min_val_recorded = min(self.min_val_recorded, current_val)
        self.max_val_recorded = max(self.max_val_recorded, current_val)

        val_range = abs(self.target_val - self.start_val)
        if val_range == 0:
            progress_ratio = 0.0
        else:
            progress = abs(current_val - self.start_val) / val_range
            progress_ratio = round(float(np.clip(progress, 0.0, 1.0)), 2)

        is_count_updated = False

        if self.state == "READY":
            if progress_ratio > 0.2:
                self.state = "PUSHING"

        elif self.state == "PUSHING":
            if progress_ratio >= 0.85:
                self.peak_reached = True

            if self.peak_reached and progress_ratio <= 0.15:
                self.rep_count += 1
                self.state = "COMPLETED"
                is_count_updated = True

                self.calibration_history.append({
                    "min_val": self.min_val_recorded,
                    "max_val": self.max_val_recorded
                })

                self.min_val_recorded = 999.0
                self.max_val_recorded = -999.0
                self.peak_reached = False

        elif self.state == "COMPLETED":
            self.state = "READY"

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "progress_ratio": progress_ratio,
            "is_count_updated": is_count_updated
        }


class MotionEngine:
    """범용 모듈형 메트릭 기반 좌/우 병렬 동작 분석 엔진"""
    def __init__(self, config: dict, custom_thresholds: dict = None):
        self.config = config
        self.exercise_name = config.get("exercise_name", "unknown")
        
        eval_cfg = config.get("eval_config", {})
        self.metric_type = eval_cfg.get("metric_type", "ANGLE")
        self.motion_direction = eval_cfg.get("motion_direction", "DECREASING")
        self.mode = "CALIBRATION" if custom_thresholds is None else "MAIN"

        default_th = config.get("default_thresholds", {})
        th_left = custom_thresholds.get("left") if custom_thresholds else default_th.get("left", {})
        th_right = custom_thresholds.get("right") if custom_thresholds else default_th.get("right", {})

        self.fsm_left = SideFSM("left", self.motion_direction, th_left)
        self.fsm_right = SideFSM("right", self.motion_direction, th_right)

    def _calculate_angle(self, p1: dict, p2: dict, p3: dict) -> float:
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y']])
        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
        return round(math.degrees(np.arccos(cosine)), 2)

    def _calculate_relative_y(self, target_p: dict, base_p: dict, ref1: dict, ref2: dict) -> float:
        torso_len = math.sqrt((ref1['x'] - ref2['x'])**2 + (ref1['y'] - ref2['y'])**2)
        if torso_len == 0:
            return 0.0
        return round((target_p['y'] - base_p['y']) / torso_len, 3)

    def _compute_side_value(self, side: str, kpt_map: dict) -> float:
        eval_cfg = self.config.get("eval_config", {})
        
        if self.metric_type == "ANGLE":
            p_indices = eval_cfg["primary_kpts"][side]
            pts = [kpt_map.get(i) for i in p_indices if kpt_map.get(i)]
            if len(pts) == 3 and all(p['score'] >= 0.4 for p in pts):
                return self._calculate_angle(pts[0], pts[1], pts[2])

        elif self.metric_type == "RELATIVE_Y":
            base_idx = eval_cfg["base_kpt"][side]
            target_idx = eval_cfg["target_kpt"][side]
            scale_indices = eval_cfg["scale_kpts"][side]
            
            b_pt, t_pt = kpt_map.get(base_idx), kpt_map.get(target_idx)
            r1_pt, r2_pt = kpt_map.get(scale_indices[0]), kpt_map.get(scale_indices[1])

            if all([b_pt, t_pt, r1_pt, r2_pt]) and min(b_pt['score'], t_pt['score'], r1_pt['score'], r2_pt['score']) >= 0.4:
                return self._calculate_relative_y(t_pt, b_pt, r1_pt, r2_pt)

        return None

    def process_keypoints(self, keypoints: list) -> dict:
        if not keypoints:
            return None

        kpt_map = {kp["id"]: kp for kp in keypoints}

        val_left = self._compute_side_value("left", kpt_map)
        val_right = self._compute_side_value("right", kpt_map)

        res_left = self.fsm_left.update(val_left)
        res_right = self.fsm_right.update(val_right)

        new_custom_thresholds = None
        if self.mode == "CALIBRATION":
            if len(self.fsm_left.calibration_history) >= 3 and len(self.fsm_right.calibration_history) >= 3:
                new_custom_thresholds = self._generate_bilateral_thresholds()

        return {
            "mode": self.mode,
            "left": {
                "val": val_left,
                "rep_count": res_left["rep_count"],
                "progress_ratio": res_left["progress_ratio"],
                "state": res_left["state"],
                "is_updated": res_left["is_count_updated"]
            },
            "right": {
                "val": val_right,
                "rep_count": res_right["rep_count"],
                "progress_ratio": res_right["progress_ratio"],
                "state": res_right["state"],
                "is_updated": res_right["is_count_updated"]
            },
            "new_custom_thresholds": new_custom_thresholds
        }

    def _generate_bilateral_thresholds(self) -> dict:
        def _calc_side_th(history: list) -> dict:
            # 🌟 [수정] 하드코딩된 3.0 대신 실제 history 리스트 길이를 사용
            h_len = len(history) if history else 1
            avg_min = sum(h["min_val"] for h in history) / h_len
            avg_max = sum(h["max_val"] for h in history) / h_len
            
            if self.motion_direction == "DECREASING":
                return {"start_val": round(avg_max, 1), "target_val": round(avg_max - (avg_max - avg_min) * 0.85, 1)}
            else:
                return {"start_val": round(avg_min, 1), "target_val": round(avg_min + (avg_max - avg_min) * 0.85, 1)}

        return {
            "left": _calc_side_th(self.fsm_left.calibration_history),
            "right": _calc_side_th(self.fsm_right.calibration_history)
        }