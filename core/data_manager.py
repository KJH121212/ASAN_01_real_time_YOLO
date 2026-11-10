# 설명: 환자 JSON 프로필과 반복별 CSV 로그를 통합 관리합니다.
# 신체 정규화 전 스켈레톤 궤적은 NPZ로 보관합니다.

import csv
import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import numpy as np
from core.session_log import KST, SessionLog, default_log_root, path_component
# 공통 버전 폴더와 등록 차수 검증 규칙입니다.
from core.patient_data import DEFAULT_DATA_VERSION, registration_number, version_folder

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

    def __init__(self, player_id="0001", patient_name="Unknown", log_root=None, patient_metadata=None,
                 data_version=DEFAULT_DATA_VERSION, registration=1):
        self.player_id = path_component(player_id, "patient_id")
        # 숫자 환자번호는 앞의 0을 포함한 네 자리로 보관합니다. 비숫자 ID는 기존 테스트 호환용입니다.
        if self.player_id.isdigit():
            if len(self.player_id) > 4:
                raise ValueError('patient_id must have at most four digits')
            self.player_id = self.player_id.zfill(4)
        # 환자번호가 같아도 등록 차수·버전이 다르면 새 폴더를 사용합니다.
        self.data_version = version_folder(data_version)
        self.registration = registration_number(registration)
        self.patient_name = patient_name
        self.log_root = Path(log_root).expanduser().resolve() if log_root else default_log_root()
        for directory in (self.log_root, *self.log_root.parents):
            if directory.exists() and not directory.is_dir():
                raise NotADirectoryError(
                    f"로그 저장 폴더는 영상 파일이 아닌 폴더로 지정하세요: {directory}. "
                    "영상 경로는 Video File Path에 입력하세요.")
        # 로그는 data/환자번호/날짜/동작 아래에 저장합니다.
        self.patient_dir = self.log_root / self.player_id
        self.patient_dir.mkdir(parents=True, exist_ok=True)
        # 프로필과 환자 정보는 환자 폴더에 보관합니다.
        self.camera_management = self.patient_dir
        self.camera_management.mkdir(parents=True, exist_ok=True)
        self.json_path = self.patient_dir / f'{self.player_id}.json'
        # 카메라 측에 제공된 환자 정보는 Unity 소유 파일과 분리해 관리합니다.
        self.metadata_path = self.camera_management / "patient_metadata.csv"
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
                      record_video=False, video_fps=30.0, started_at=None, target_score=None,
                      play_session_id=None, exercise_record_id=None, slot=1):
        if self.session_log is not None and self.session_log._summary is None:
            raise RuntimeError("Finish the active log before starting another")
        self._save_metadata()
        self.session_log = SessionLog(self.patient_dir, exercise_name, mode, session,
                                      round_number, set_number, record_video, video_fps, started_at, target_score,
                                      play_session_id=play_session_id, exercise_record_id=exercise_record_id,
                                      slot=slot, patient_id=self.player_id, data_version=self.data_version,
                                      registration=self.registration)
        self.buf_timestamps = self.session_log.timestamps
        self.buf_values = self.session_log.values
        self.buf_raw_keypoints = self.session_log.keypoints
        print(f"[LOG] Session files: {self.session_log.base}.*", flush=True)
        return dict(self.session_log.info)

    def _round_threshold_files(self, exercise_name):
        log = self.session_log
        if log is None or log.exercise_name != exercise_name:
            raise ValueError('Begin the matching exercise session before accessing calibration')
        prefix = f'round{log.round:02d}_thresholds_v'
        files = []
        for path in log.directory.glob(prefix + '*.json'):
            version = path.stem[len(prefix):]
            if version.isdigit():
                files.append((int(version), path))
        return sorted(files, reverse=True)

    def _calibration_reference(self, record, path):
        return dict(file=path.name, path=path.relative_to(self.patient_dir).as_posix(),
                    version=record['version'], created_at=record['created_at'],
                    calibration_session_id=record['calibration_session_id'],
                    date=record['date'], round=record['round'], exercise_name=record['exercise_name'],
                    thresholds=deepcopy(record['thresholds']))

    def save_custom_threshold(self, exercise_name, threshold_data, is_confirmed=True):
        """Preserve each successful calibration beside its round's session logs."""
        files = self._round_threshold_files(exercise_name)
        log = self.session_log
        if log.mode != 'CALIBRATION' or log._summary is not None:
            raise ValueError('Thresholds can only be saved from an active calibration session')
        if not isinstance(threshold_data, dict) or not all(isinstance(threshold_data.get(side), dict)
                                                          for side in ('left', 'right')):
            raise ValueError('Thresholds require left and right settings')
        version = max((number for number, _ in files), default=0) + 1
        record = dict(schema='round-calibration/1', patient_id=self.player_id,
                      data_version=self.data_version, registration=self.registration,
                      date=log.started_at.strftime('%Y-%m-%d'), exercise_name=exercise_name, round=log.round,
                      created_at=datetime.now(KST).isoformat(), calibration_session_id=log.session_id,
                      calibration_measurement=deepcopy(log.info.get('measurement', {})),
                      calibration_log=log.basename, is_confirmed=bool(is_confirmed), thresholds=threshold_data)
        while True:
            path = log.directory / f'round{log.round:02d}_thresholds_v{version:02d}.json'
            record['version'] = version
            encoded = json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False)
            try:
                stream = path.open('x', encoding='utf-8')
            except FileExistsError:
                version += 1
                continue
            try:
                with stream:
                    stream.write(encoded)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            break
        log.info['calibration'] = self._calibration_reference(record, path)
        return dict(log.info['calibration'])

    def get_custom_threshold(self, exercise_name):
        """Select the latest confirmed version in this date/exercise/round only."""
        for version, path in self._round_threshold_files(exercise_name):
            record = json.loads(path.read_text(encoding='utf-8'))
            log = self.session_log
            expected = dict(patient_id=self.player_id, data_version=self.data_version,
                            registration=self.registration, exercise_name=exercise_name,
                            date=log.started_at.strftime('%Y-%m-%d'), round=log.round)
            if any(record.get(key) != value for key, value in expected.items()):
                continue
            if not record.get('is_confirmed'):
                continue
            if record.get('version') != version or not isinstance(record.get('thresholds'), dict):
                raise ValueError(f'Invalid calibration file: {path.name}')
            for side in ('left', 'right'):
                if not isinstance(record['thresholds'].get(side), dict):
                    raise ValueError(f'Missing {side} thresholds: {path.name}')
            log.info['calibration'] = self._calibration_reference(record, path)
            return record['thresholds']
        return None

    def finish_session(self, rep_rows, status="COMPLETED", end_elapsed_sec=None):
        if self.session_log is None:
            return None
        return self.session_log.finish(rep_rows, status, end_elapsed_sec=end_elapsed_sec)
