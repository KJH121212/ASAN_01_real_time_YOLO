# ==============================================================================
# [Module Information]
# File: core/data_manager.py
# Description: Integrated I/O manager for patient JSON profiles, CSV rep logs,
#              and raw, unprocessed skeleton trajectory NPZ archives.
# ==============================================================================

import csv
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from core.session_log import KST, SessionLog, default_log_root, path_component, write_json_atomic


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
            "metric_type": "COORDINATE",
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
    """Patient profiles and session logs under a configurable external root."""

    def __init__(self, player_id="patient_1", patient_name="Unknown", log_root=None, patient_metadata=None):
        self.player_id = path_component(player_id, "patient_id")
        self.patient_name = patient_name
        self.log_root = Path(log_root).expanduser().resolve() if log_root else default_log_root()
        for directory in (self.log_root, *self.log_root.parents):
            if directory.exists() and not directory.is_dir():
                raise NotADirectoryError(
                    f"로그 저장 폴더는 영상 파일이 아닌 폴더로 지정하세요: {directory}. "
                    "영상 경로는 Video File Path에 입력하세요.")
        self.patient_dir = self.log_root / self.player_id
        self.patient_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.patient_dir / f"{self.player_id}.json"
        self.metadata_path = self.patient_dir / "patient_metadata.csv"
        self.legacy_json_path = Path(__file__).resolve().parent.parent / "data" / self.player_id / f"{self.player_id}.json"
        self.patient_metadata = patient_metadata or {}
        if not isinstance(self.patient_metadata, dict):
            raise ValueError("patient_metadata must be an object")
        self.session_log = None
        self.buf_timestamps, self.buf_values, self.buf_raw_keypoints = [], [], []
        print(f"[LOG] Patient directory: {self.patient_dir}", flush=True)

    def _save_metadata(self):
        fields = ["patient_id", "patient_name", "height_cm", "weight_kg"]
        metadata = {}
        if self.metadata_path.exists():
            with self.metadata_path.open(newline="", encoding="utf-8-sig") as file:
                metadata = next(csv.DictReader(file), {})
        metadata.update(patient_id=self.player_id, patient_name=self.patient_name)
        for key in ("height_cm", "weight_kg"):
            if key in self.patient_metadata and self.patient_metadata[key] is not None:
                value = float(self.patient_metadata[key])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{key} must be positive")
                metadata[key] = value
        temporary = self.metadata_path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(metadata)
            temporary.replace(self.metadata_path)
        finally:
            temporary.unlink(missing_ok=True)

    def begin_session(self, exercise_name, mode, session=None, round_number=1, set_number=1,
                      record_video=False, video_fps=30.0, started_at=None, target_score=None):
        if self.session_log is not None and self.session_log._summary is None:
            raise RuntimeError("Finish the active log before starting another")
        self._save_metadata()
        self.session_log = SessionLog(self.patient_dir, exercise_name, mode, session,
                                      round_number, set_number, record_video, video_fps, started_at, target_score)
        self.buf_timestamps = self.session_log.timestamps
        self.buf_values = self.session_log.values
        self.buf_raw_keypoints = self.session_log.keypoints
        print(f"[LOG] Session files: {self.session_log.base}.*", flush=True)
        return dict(self.session_log.info)

    def save_custom_threshold(self, exercise_name, threshold_data, is_confirmed=True):
        data = {}
        # Preserve the previous profile when writing the first new calibration.
        source = self.json_path if self.json_path.exists() else self.legacy_json_path
        if source.exists():
            data = json.loads(source.read_text(encoding="utf-8"))
        data.update(patient_id=self.player_id, patient_name=self.patient_name)
        data.setdefault("custom_thresholds", {})[exercise_name] = threshold_data
        data.setdefault("calibration_history", []).append({
            "exercise_name": exercise_name,
            "calibration_date": datetime.now(KST).isoformat(),
            "is_confirmed": is_confirmed, "thresholds": threshold_data,
        })
        write_json_atomic(self.json_path, data)

    def get_custom_threshold(self, exercise_name):
        for source in (self.json_path, self.legacy_json_path):
            if not source.exists():
                continue
            try:
                profile = json.loads(source.read_text(encoding="utf-8"))
                threshold = profile.get("custom_thresholds", {}).get(exercise_name)
                if threshold is not None:
                    return threshold
            except (OSError, ValueError, AttributeError) as exc:
                print(f"[WARN][LOG] Could not read profile {source}: {exc}")
        return None

    def finish_session(self, rep_rows, status="COMPLETED", end_elapsed_sec=None):
        if self.session_log is None:
            return None
        return self.session_log.finish(rep_rows, status, end_elapsed_sec=end_elapsed_sec)
