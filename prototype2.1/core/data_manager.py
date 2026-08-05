import os
import json
import csv
import numpy as np
from datetime import datetime


def load_exercise_config(exercise_name: str) -> dict:
    """prototype2.1/configs 디렉토리의 JSON 설정을 절대 경로로 로드"""
    current_dir = os.path.dirname(os.path.abspath(__file__))  # core
    proto_dir = os.path.dirname(current_dir)                   # prototype2.1
    config_path = os.path.join(proto_dir, "configs", "exercise_kpt_config_coco.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"운동 설정 파일을 찾을 수 없습니다: {config_path}")
        
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    exercises = data.get("exercises", {})
    if exercise_name not in exercises:
        raise KeyError(f"설정 파일 내 '{exercise_name}' 항목이 존재하지 않습니다.")
        
    return exercises[exercise_name]


class DataManager:
    def __init__(self, player_id: str, patient_name: str = "미지정"):
        self.player_id = player_id
        self.patient_name = patient_name
        
        proto_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.patient_dir = os.path.join(proto_dir, "data", player_id)
        os.makedirs(self.patient_dir, exist_ok=True)
        
        self.user_file_path = os.path.join(self.patient_dir, f"{player_id}.json")
        self.csv_summary_path = os.path.join(self.patient_dir, "test_summary_records.csv")
        self._ensure_user_file()

    def _ensure_user_file(self):
        if not os.path.exists(self.user_file_path):
            initial_data = {
                "player_id": self.player_id,
                "patient_name": self.patient_name,
                "custom_thresholds": {},
                "test_sessions": []
            }
            with open(self.user_file_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, ensure_ascii=False, indent=2)

    def get_custom_threshold(self, exercise_name: str) -> dict:
        with open(self.user_file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        return user_data.get("custom_thresholds", {}).get(exercise_name)

    def save_custom_threshold(self, exercise_name: str, thresholds: dict, is_confirmed: bool = False):
        with open(self.user_file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

        thresholds["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        thresholds["is_confirmed"] = is_confirmed
        
        user_data["custom_thresholds"][exercise_name] = thresholds
        
        with open(self.user_file_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
        print(f"[DataManager] [{self.player_id}] {exercise_name} 개인 맞춤 임계값 저장 완료")

    def save_trajectory_npz(self, session_id: str, exercise_name: str, timestamps: list, values: list, keypoints: list):
        """세션 전체 프레임 데이터를 .npz 압축 파일로 저장"""
        traj_dir = os.path.join(self.patient_dir, "trajectories")
        os.makedirs(traj_dir, exist_ok=True)
        
        filename = f"{session_id}_{exercise_name}.npz"
        file_path = os.path.join(traj_dir, filename)
        
        arr_timestamps = np.array(timestamps, dtype=np.float32)
        arr_values = np.array(values, dtype=np.float32)
        arr_keypoints = np.array(keypoints, dtype=np.float32)
        
        np.savez_compressed(
            file_path,
            timestamps=arr_timestamps,
            values=arr_values,
            keypoints=arr_keypoints
        )
        print(f"[DataManager] [{self.player_id}] NPZ 궤적 저장 완료: {filename}")

    def save_test_session_summary(self, session_summary: dict):
        """본 운동 요약 데이터를 환자 JSON 및 CSV에 저장"""
        with open(self.user_file_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)
        
        user_data.setdefault("test_sessions", []).append(session_summary)
        
        with open(self.user_file_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)

        file_exists = os.path.exists(self.csv_summary_path)
        fieldnames = [
            "session_id", "timestamp", "player_id", "patient_name", "exercise_name",
            "target_reps", "left_completed_reps", "right_completed_reps",
            "total_duration_sec"
        ]
        
        csv_row = {k: session_summary.get(k, "") for k in fieldnames}
        
        with open(self.csv_summary_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(csv_row)
            
        print(f"[DataManager] [{self.player_id}] 세션 요약 동기화 완료 ({self.patient_dir})")