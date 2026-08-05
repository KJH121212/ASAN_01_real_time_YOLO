import csv
from datetime import datetime
import json
import os
from pathlib import Path
import numpy as np


def load_exercise_config(exercise_name: str) -> dict:
  project_root = Path(__file__).resolve().parent.parent.parent
  config_path = (
      project_root
      / "prototype2.1"
      / "configs"
      / "exercise_kpt_config_coco.json"
  )
  if not config_path.exists():
    config_path = project_root / "configs" / "exercise_kpt_config_coco.json"

  if config_path.exists():
    try:
      with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        exercise_info = data.get("exercises", {}).get(exercise_name)
        if exercise_info:
          return exercise_info
    except Exception as e:
      print(f"[DataManager Warning] 설정 로드 실패: {e}")

  return {
      "exercise_name": exercise_name,
      "eval_config": {
          "metric_type": "ANGLE",
          "motion_direction": "DECREASING",
          "primary_kpts": {"left": [5, 7, 9], "right": [6, 8, 10]},
      },
      "default_thresholds": {
          "left": {"start_val": 160.0, "target_val": 50.0},
          "right": {"start_val": 160.0, "target_val": 50.0},
      },
  }


class DataManager:

  def __init__(
      self, player_id: str = "patient_1", patient_name: str = "미지정"
  ):
    self.player_id = player_id
    self.patient_name = patient_name

    project_root = Path(__file__).resolve().parent.parent.parent
    self.patient_dir = (project_root / "data" / player_id).resolve()
    self.skeleton_dir = (self.patient_dir / "skeleton memory").resolve()

    self.patient_dir.mkdir(parents=True, exist_ok=True)
    self.skeleton_dir.mkdir(parents=True, exist_ok=True)

    self.json_path = self.patient_dir / f"{player_id}.json"
    self.csv_path = self.patient_dir / "rep_details.csv"

  def save_custom_threshold(
      self, exercise_name: str, threshold_data: dict, is_confirmed: bool = True
  ):
    """기존 파일 데이터를 읽어와서 덮어쓰지 않고 히스토리를 누적 저장"""
    data = {}
    if self.json_path.exists():
      try:
        with open(self.json_path, "r", encoding="utf-8") as f:
          data = json.load(f)
      except Exception:
        data = {}

    data["patient_id"] = self.player_id
    data["patient_name"] = self.patient_name

    if "custom_thresholds" not in data or not isinstance(
        data["custom_thresholds"], dict
    ):
      data["custom_thresholds"] = {}
    if "calibration_history" not in data or not isinstance(
        data["calibration_history"], list
    ):
      data["calibration_history"] = []

    # 1. 해당 운동의 최신 임계값 덮어쓰기/추가
    data["custom_thresholds"][exercise_name] = threshold_data

    # 2. 시범 동작 측정이 수행될 때마다 히스토리 리스트에 추가
    history_entry = {
        "exercise_name": exercise_name,
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_confirmed": is_confirmed,
        "thresholds": threshold_data,
    }
    data["calibration_history"].append(history_entry)

    with open(self.json_path, "w", encoding="utf-8") as f:
      json.dump(data, f, ensure_ascii=False, indent=4)

  def get_custom_threshold(self, exercise_name: str) -> dict:
    if self.json_path.exists():
      try:
        with open(self.json_path, "r", encoding="utf-8") as f:
          data = json.load(f)
          # custom_thresholds 딕셔너리에서 해당 운동 종목 정보 추출
          custom_th = data.get("custom_thresholds", {})
          if exercise_name in custom_th:
            return custom_th[exercise_name]
          return data.get("thresholds", {})
      except Exception:
        pass
    return None

  def save_rep_details_csv(self, rep_rows: list):
    if not rep_rows:
      return

    fieldnames = [
        "session_id",
        "timestamp",
        "side",
        "rep_num",
        "duration_sec",
        "min_angle",
        "max_angle",
        "achieved_rom",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "quality",
    ]

    file_exists = self.csv_path.exists()
    with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      if not file_exists:
        writer.writeheader()
      for row in rep_rows:
        writer.writerow(row)

  def save_trajectory_npz(
      self,
      session_id: str,
      exercise_name: str,
      timestamps: list,
      values: list,
      keypoints: list,
  ):
    if not timestamps:
      return

    npz_path = self.skeleton_dir / f"{session_id}_skeleton.npz"
    np.savez_compressed(
        npz_path,
        timestamps=np.array(timestamps, dtype=np.float32),
        values=np.array(values, dtype=np.float32),
        keypoints=np.array(keypoints, dtype=np.float32),
        player_id=self.player_id,
        exercise_name=exercise_name,
    )