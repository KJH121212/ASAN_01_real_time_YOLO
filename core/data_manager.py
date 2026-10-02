# ==============================================================================
# [Module Information]
# File: core/data_manager.py
# Description: Integrated I/O manager for patient JSON profiles, CSV rep logs,
#              and raw, unprocessed skeleton trajectory NPZ archives.
# ==============================================================================

import csv
import json
import os
from datetime import datetime
from pathlib import Path
import traceback
import numpy as np


def load_exercise_config(exercise_name: str) -> dict:
    """JSON 설정 파일로부터 특정 운동의 평가 규칙 및 기본 임계값을 로드합니다."""
    project_root = Path(__file__).resolve().parent.parent
    config_path = project_root / "configs" / "exercise_kpt_config_coco.json"

    print(f"[DEBUG][CONFIG] Reading exercise configuration file: {config_path}")

    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                exercise_info = data.get("exercises", {}).get(exercise_name)
                if exercise_info:
                    print(f"[DEBUG][CONFIG] Exercise '{exercise_name}' configuration loaded.")
                    return exercise_info
                else:
                    print(f"[WARN][CONFIG] Exercise '{exercise_name}' not defined. Falling back to default.")
        except Exception as e:
            print(f"[ERROR][CONFIG] Failed reading JSON config: {e}")
            traceback.print_exc()

    # 파싱 실패 또는 파일 미존재 시 시스템 다운을 방지하기 위한 폴백 기본 구조체
    fallback = {
        "exercise_name": exercise_name,
        "eval_config": {
            "metric_type": "RELATIVE_Y",
            "motion_direction": "DECREASING",
        },
        "default_thresholds": {
            "left": {"start_val": 0.8, "target_val": -0.3},
            "right": {"start_val": 0.8, "target_val": -0.3},
        },
    }
    print(f"[DEBUG][CONFIG] Returning fallback configuration for: {exercise_name}")
    return fallback


class DataManager:
    """환자 프로필, 임계값 이력, 운동 성과 CSV 및 순수 원본 관절 궤적(NPZ) I/O 클래스"""

    def __init__(self, player_id: str = "patient_1", patient_name: str = "Unknown"):
        self.player_id = player_id
        self.patient_name = patient_name

        project_root = Path(__file__).resolve().parent.parent
        self.patient_dir = (project_root / "data" / player_id).resolve()
        self.skeleton_dir = (self.patient_dir / "skeleton_memory").resolve()

        # 필요한 하위 디렉터리 사전 생성
        self.patient_dir.mkdir(parents=True, exist_ok=True)
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)

        self.json_path = self.patient_dir / f"{player_id}.json"
        self.csv_path = self.patient_dir / "rep_details.csv"

        print(f"[DEBUG][INIT] DataManager initialized for Player: {self.player_id}")
        print(f"[DEBUG][INIT] Patient Directory: {self.patient_dir}")
        print(f"[DEBUG][INIT] Skeleton Storage Directory: {self.skeleton_dir}")

    def save_custom_threshold(self, exercise_name: str, threshold_data: dict, is_confirmed: bool = True):
        """캘리브레이션 결과로 산출된 임계값을 JSON 프로필에 원자적(Atomic)으로 기록합니다."""
        print(f"[DEBUG][IO] Writing custom threshold profile for '{exercise_name}'...")
        data = {}

        if self.json_path.exists():
            try:
                with open(self.json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                print(f"[ERROR][IO] Existing profile JSON corrupted. Resetting data. Cause: {e}")
                data = {}

        data["patient_id"] = self.player_id
        data["patient_name"] = self.patient_name

        if "custom_thresholds" not in data or not isinstance(data["custom_thresholds"], dict):
            data["custom_thresholds"] = {}
        if "calibration_history" not in data or not isinstance(data["calibration_history"], list):
            data["calibration_history"] = []

        data["custom_thresholds"][exercise_name] = threshold_data

        history_entry = {
            "exercise_name": exercise_name,
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_confirmed": is_confirmed,
            "thresholds": threshold_data,
        }
        data["calibration_history"].append(history_entry)

        # 쓰기 중단으로 인한 파일 깨짐 방지를 위한 원자적 교체(.tmp -> .json)
        temp_path = self.patient_dir / f"{self.player_id}.json.tmp"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            temp_path.replace(self.json_path)
            print(f"[DEBUG][IO] Profile update successfully committed: {self.json_path}")
        except Exception as e:
            print(f"[ERROR][IO] Failed saving profile JSON: {e}")
            traceback.print_exc()

    def get_custom_threshold(self, exercise_name: str) -> dict:
        """기존 JSON 파일에서 등록된 맞춤 가동 범위 임계값을 조회합니다."""
        if not self.json_path.exists():
            print(f"[DEBUG][IO] Profile file does not exist: {self.json_path}")
            return None

        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                custom_th = data.get("custom_thresholds", {})
                if exercise_name in custom_th:
                    print(f"[DEBUG][IO] Found custom threshold for '{exercise_name}': {custom_th[exercise_name]}")
                    return custom_th[exercise_name]
                else:
                    print(f"[DEBUG][IO] No custom threshold entry for exercise: {exercise_name}")
        except Exception as e:
            print(f"[ERROR][IO] Failed reading profile JSON: {e}")
            traceback.print_exc()

        return None

    def save_rep_details_csv(self, rep_rows: list):
        """운동 세션 중 완수한 반복 회차(Repetition)의 정량 통계를 CSV에 누적 추가합니다."""
        if not rep_rows:
            print("[DEBUG][IO] Rep rows buffer is empty. Skipping CSV write.")
            return

        fieldnames = [
            "session_id", "timestamp", "side", "rep_num", "duration_sec",
            "min_angle", "max_angle", "achieved_rom", "quality"
        ]

        file_exists = self.csv_path.exists()
        try:
            with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if not file_exists:
                    writer.writeheader()
                    print(f"[DEBUG][IO] Created new CSV header: {self.csv_path}")
                for row in rep_rows:
                    writer.writerow(row)

            print(f"[DEBUG][IO] Appended {len(rep_rows)} rows to CSV: {self.csv_path}")
        except Exception as e:
            print(f"[ERROR][IO] Failed writing to CSV: {e}")
            traceback.print_exc()

    def save_raw_trajectory_npz(
        self,
        session_id: str,
        exercise_name: str,
        timestamps: list,
        values: list,
        raw_keypoints: list,
        prefix: str = "SESSION"
    ):
        """
        후처리가 전혀 적용되지 않은 순수 원본 17개 관절 시계열 좌표 행렬을 NPZ로 압축 저장합니다.
        
        Args:
            session_id: 세션 고유 식별 타임스탬프 문자열
            exercise_name: 대상 운동 종목명
            timestamps: 프레임별 경과 시간 리스트 (N,)
            values: 실시간 계산 수치 리스트 (N, 2)
            raw_keypoints: 비전 추론 모델에서 추출된 순수 원본 관절 리스트 (N, 17, 3)
            prefix: 파일명 접두사 ('SESSION' 또는 'CALIB')
        """
        if not timestamps or not raw_keypoints:
            print("[DEBUG][IO] Trajectory buffer is empty. Aborting raw NPZ export.")
            return

        # 고유 파일명 조립: {prefix}_{session_id}_skeleton.npz
        npz_filename = f"{prefix}_{session_id}_skeleton.npz"
        npz_path = self.skeleton_dir / npz_filename

        try:
            # 고속 직렬화 및 메모리 정렬을 위해 NumPy float32 배열로 변환
            ts_array = np.array(timestamps, dtype=np.float32)
            val_array = np.array(values, dtype=np.float32)
            kpt_array = np.array(raw_keypoints, dtype=np.float32)

            print(f"[DEBUG][IO] Exporting RAW NPZ trajectory: {npz_path}")
            print(f"       Prefix: {prefix}")
            print(f"       Timestamps Shape: {ts_array.shape}")
            print(f"       Values Shape: {val_array.shape}")
            print(f"       Raw Keypoints Shape: {kpt_array.shape} (Expected: (N, 17, 3))")

            # 순수 원본 좌표를 zlib으로 고압축 덤프 (후처리/필터링 전 원형 유지)
            np.savez_compressed(
                npz_path,
                timestamps=ts_array,
                values=val_array,
                keypoints=kpt_array,
                player_id=self.player_id,
                exercise_name=exercise_name,
                session_type=prefix
            )

            file_size_kb = round(os.path.getsize(npz_path) / 1024.0, 2)
            print(f"[DEBUG][IO] Raw NPZ archive committed successfully. Size: {file_size_kb} KB")

        except Exception as e:
            print(f"[ERROR][IO] Failed saving raw NPZ file: {e}")
            traceback.print_exc()

    # 하위 호환성을 위한 기존 시그니처 래핑
    def save_trajectory_npz(self, session_id: str, exercise_name: str, timestamps: list, values: list, keypoints: list):
        self.save_raw_trajectory_npz(
            session_id=session_id,
            exercise_name=exercise_name,
            timestamps=timestamps,
            values=values,
            raw_keypoints=keypoints,
            prefix="SESSION"
        )