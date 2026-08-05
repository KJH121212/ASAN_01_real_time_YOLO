import math
import time
import numpy as np

class SideFSM:

  def __init__(
      self,
      side_name: str,
      motion_direction: str,
      thresholds: dict,
      is_calibration: bool = False,
      target_reps: int = 3,
  ):
    self.side = side_name
    self.is_decreasing = motion_direction == "DECREASING"
    self.is_calibration = is_calibration
    self.target_reps = target_reps
    self.state = "READY"
    self.rep_count = 0
    self.peak_reached = False

    self.start_val = thresholds.get(
        "start_val", 160.0 if self.is_decreasing else 90.0
    )
    self.target_val = thresholds.get(
        "target_val", 50.0 if self.is_decreasing else 160.0
    )

    self.min_val_recorded = 999.0
    self.max_val_recorded = -999.0

    self.min_x, self.max_x = 999.0, -999.0
    self.min_y, self.max_y = 999.0, -999.0

    self.rep_start_time = None
    self.last_rep_peak = None
    self.last_quality = "CALIBRATING" if is_calibration else "READY"

    self.calibration_history = []
    self.completed_reps_history = []

  def evaluate_quality(self, peak_val: float) -> str:
    if self.is_calibration:
      return "CALIBRATING"

    total_range = abs(self.target_val - self.start_val)
    if total_range == 0:
      return "GOOD"

    achieved_range = abs(peak_val - self.start_val)
    ratio = achieved_range / total_range

    # 설정 요청에 맞춘 달성도 비율 기준 변경
    if ratio >= 0.80:
      return "PERFECT"
    elif ratio >= 0.60:
      return "GOOD"
    elif ratio >= 0.40:
      return "BAD"
    else:
      return "INVALID"

  def update(self, current_val: float, primary_pts: list = None) -> dict:
    if self.rep_count >= self.target_reps:
      self.state = "WAITING"
      return {
          "state": "WAITING",
          "rep_count": self.rep_count,
          "progress_ratio": 1.0,
          "is_count_updated": False,
          "quality": (
              "FINISHED" if not self.is_calibration else "CALIBRATING"
          ),
      }

    if current_val is None:
      return {
          "state": self.state,
          "rep_count": self.rep_count,
          "progress_ratio": 0.0,
          "is_count_updated": False,
          "quality": self.last_quality,
      }

    now_time = time.time()
    self.min_val_recorded = min(self.min_val_recorded, current_val)
    self.max_val_recorded = max(self.max_val_recorded, current_val)

    if primary_pts:
      for pt in primary_pts:
        if pt and "x" in pt and "y" in pt:
          self.min_x = min(self.min_x, pt["x"])
          self.max_x = max(self.max_x, pt["x"])
          self.min_y = min(self.min_y, pt["y"])
          self.max_y = max(self.max_y, pt["y"])

    val_range = abs(self.target_val - self.start_val)
    progress_ratio = (
        round(
            float(
                np.clip(
                    abs(current_val - self.start_val) / val_range, 0.0, 1.0
                )
            ),
            2,
        )
        if val_range != 0
        else 0.0
    )
    is_count_updated = False

    if self.state == "READY":
      if progress_ratio > 0.2:
        self.state = "PUSHING"
        self.rep_start_time = now_time

    elif self.state == "PUSHING":
      # 0.40 이상 도달 시 Peak 도달로 인정 (BAD, GOOD, PERFECT 모두 카운트 대상 포함)
      if progress_ratio >= 0.40:
        self.peak_reached = True

      # 원위치 복귀(0.15 이하) 시 카운팅 판정
      if progress_ratio <= 0.15:
        if self.peak_reached:
          peak_val = (
              self.min_val_recorded
              if self.is_decreasing
              else self.max_val_recorded
          )
          quality = (
              "CALIBRATING"
              if self.is_calibration
              else self.evaluate_quality(peak_val)
          )

          # INVALID(0.4 미만)가 아닌 경우에만 카운팅 수행
          if quality in ["PERFECT", "GOOD", "BAD", "CALIBRATING"]:
            self.rep_count += 1
            is_count_updated = True
            self.last_rep_peak = peak_val
            self.last_quality = quality

            duration_sec = (
                round(now_time - self.rep_start_time, 2)
                if self.rep_start_time
                else 0.0
            )
            achieved_rom = round(
                abs(self.max_val_recorded - self.min_val_recorded), 2
            )

            self.completed_reps_history.append({
                "side": self.side,
                "rep_num": self.rep_count,
                "duration_sec": duration_sec,
                "min_angle": round(self.min_val_recorded, 2),
                "max_angle": round(self.max_val_recorded, 2),
                "achieved_rom": achieved_rom,
                "min_x": (
                    round(self.min_x, 3) if self.min_x != 999.0 else 0.0
                ),
                "max_x": (
                    round(self.max_x, 3) if self.max_x != -999.0 else 0.0
                ),
                "min_y": (
                    round(self.min_y, 3) if self.min_y != 999.0 else 0.0
                ),
                "max_y": (
                    round(self.max_y, 3) if self.max_y != -999.0 else 0.0
                ),
                "quality": quality,
            })

            self.calibration_history.append({
                "min_val": self.min_val_recorded,
                "max_val": self.max_val_recorded,
            })

        # 카운트 완료 후 또는 0.40 미달 후 복귀 시 상태 초기화
        self.min_val_recorded, self.max_val_recorded = 999.0, -999.0
        self.min_x, self.max_x, self.min_y, self.max_y = (
            999.0,
            -999.0,
            999.0,
            -999.0,
        )
        self.peak_reached = False

        if self.rep_count >= self.target_reps:
          self.state = "WAITING"
        else:
          self.state = "READY"

    return {
        "state": self.state,
        "rep_count": self.rep_count,
        "progress_ratio": progress_ratio,
        "is_count_updated": is_count_updated,
        "quality": self.last_quality,
    }


class MotionEngine:

  def __init__(
      self,
      config: dict,
      custom_thresholds: dict = None,
      mode: str = "CALIBRATION",
      target_reps: int = 10,
  ):
    self.config = config
    self.exercise_name = config.get("exercise_name", "unknown")

    eval_cfg = config.get("eval_config", {})
    self.metric_type = eval_cfg.get("metric_type", "ANGLE")
    self.motion_direction = eval_cfg.get("motion_direction", "DECREASING")
    self.mode = mode

    default_th = config.get("default_thresholds", {})

    if self.mode == "CALIBRATION":
      th_left = default_th.get("left", {})
      th_right = default_th.get("right", {})
      reps_limit = 3
    else:
      th_left = (
          custom_thresholds.get("left")
          if custom_thresholds
          else default_th.get("left", {})
      )
      th_right = (
          custom_thresholds.get("right")
          if custom_thresholds
          else default_th.get("right", {})
      )
      reps_limit = target_reps

    is_calib = self.mode == "CALIBRATION"
    self.fsm_left = SideFSM(
        "left",
        self.motion_direction,
        th_left,
        is_calibration=is_calib,
        target_reps=reps_limit,
    )
    self.fsm_right = SideFSM(
        "right",
        self.motion_direction,
        th_right,
        is_calibration=is_calib,
        target_reps=reps_limit,
    )

  def _calculate_angle(self, p1: dict, p2: dict, p3: dict) -> float:
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:
      return 0.0
    cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
    return round(math.degrees(np.arccos(cosine)), 2)

  def _get_primary_pts(self, side: str, kpt_map: dict) -> list:
    eval_cfg = self.config.get("eval_config", {})
    p_indices = eval_cfg.get("primary_kpts", {}).get(side, [])
    return [kpt_map.get(i) for i in p_indices if kpt_map.get(i)]

  def _compute_side_value(self, side: str, kpt_map: dict) -> float:
    pts = self._get_primary_pts(side, kpt_map)
    if len(pts) == 3 and all(p["score"] >= 0.4 for p in pts):
      return self._calculate_angle(pts[0], pts[1], pts[2])
    return None

  def process_keypoints(self, keypoints: list) -> dict:
    if not keypoints:
      return None

    kpt_map = {kp["id"]: kp for kp in keypoints}
    val_left = self._compute_side_value("left", kpt_map)
    val_right = self._compute_side_value("right", kpt_map)

    pts_left = self._get_primary_pts("left", kpt_map)
    pts_right = self._get_primary_pts("right", kpt_map)

    res_left = self.fsm_left.update(val_left, pts_left)
    res_right = self.fsm_right.update(val_right, pts_right)

    new_custom_thresholds = None
    if self.mode == "CALIBRATION":
      if (
          len(self.fsm_left.calibration_history) >= 3
          and len(self.fsm_right.calibration_history) >= 3
      ):
        new_custom_thresholds = self._generate_bilateral_thresholds()

    return {
        "mode": self.mode,
        "left": {
            "val": val_left,
            "rep_count": res_left["rep_count"],
            "progress_ratio": res_left["progress_ratio"],
            "state": res_left["state"],
            "quality": res_left["quality"],
            "is_updated": res_left["is_count_updated"],
        },
        "right": {
            "val": val_right,
            "rep_count": res_right["rep_count"],
            "progress_ratio": res_right["progress_ratio"],
            "state": res_right["state"],
            "quality": res_right["quality"],
            "is_updated": res_right["is_count_updated"],
        },
        "new_custom_thresholds": new_custom_thresholds,
    }

  def _generate_bilateral_thresholds(self) -> dict:

    def _calc_side_th(history: list) -> dict:
      h_len = len(history) if history else 1
      avg_min = sum(h["min_val"] for h in history) / h_len
      avg_max = sum(h["max_val"] for h in history) / h_len

      if self.motion_direction == "DECREASING":
        return {
            "start_val": round(avg_max, 1),
            "target_val": round(avg_max - (avg_max - avg_min) * 0.85, 1),
        }
      else:
        return {
            "start_val": round(avg_min, 1),
            "target_val": round(avg_min + (avg_max - avg_min) * 0.85, 1),
        }

    return {
        "left": _calc_side_th(self.fsm_left.calibration_history),
        "right": _calc_side_th(self.fsm_right.calibration_history),
    }