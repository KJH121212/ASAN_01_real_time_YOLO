"""환자 메타데이터 및 23개 관절(COCO-23) 세션 로그(CSV · NPZ · MP4 · 임계값 JSON) 통합 관리 모듈."""

import csv
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import sys
import uuid

import numpy as np

NUM_KEYPOINTS = 23
DEFAULT_DATA_VERSION = "v1.0.0(0)"
KST = timezone(timedelta(hours=9))

FRAME_FIELDS = [
    "time", "play_session_id", "exercise_record_id", "sensor_file_id", "source", "collection_phase", "metric_unit",
    "timestamp_ms", "frame_idx", "video_frame_idx", "motion_name", "metric_type",
    "left_value", "right_value", "left_rep_count", "right_rep_count",
    "left_total_score", "right_total_score", "left_score_delta", "right_score_delta",
    "left_state", "right_state", "left_quality", "right_quality",
    "left_calibration_status", "right_calibration_status",
    "left_calibration_baseline", "right_calibration_baseline",
    "left_calibration_return_tolerance", "right_calibration_return_tolerance",
    "left_count_updated", "right_count_updated", "is_occluded", "calib_step",
    "threshold_version", "left_start_val", "left_target_val", "right_start_val", "right_target_val",
    "use_normalization", "counting_method",
    "left_activation_val", "right_activation_val", "left_reversal_val", "right_reversal_val",
    "patient_id", "session", "round", "set", "target_reps", "target_score",
    "calibration_file", "calibration_version", "calibration_created_at",
]


def default_log_root() -> Path:
    """기본 데이터 저장 루트(프로젝트 최상위 data/ 폴더)를 반환한다."""
    configured = os.environ.get("ASAN_LOG_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return (Path(sys.executable).resolve().parent / "data").resolve()
    return (Path(__file__).resolve().parents[2] / "data").resolve()


def path_component(value, name):
    value = str(value)
    if not re.fullmatch(r"[A-Za-z0-9_-]+", value) or Path(value).is_reserved():
        raise ValueError(f"{name} must contain only letters, numbers, underscores or hyphens")
    return value


def sequence_number(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def version_folder(value):
    value = DEFAULT_DATA_VERSION if value is None else str(value).strip()
    if not re.fullmatch(r"v\d+\.\d+\.\d+\(\d+\)", value):
        raise ValueError("data_version must look like v1.0.0(0)")
    return value


def registration_number(value):
    if value is None:
        return "01"
    if isinstance(value, str) and value.isdigit():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 99:
        raise ValueError("registration must be an integer from 1 to 99")
    return f"{value:02d}"


def shared_identifier(value, name):
    if value is None or str(value).strip() == "":
        return uuid.uuid4().hex
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", value):
        raise ValueError(f"{name} must contain 1..80 letters, numbers, underscores or hyphens")
    return value


def load_exercise_config(exercise_name: str) -> dict:
    """socket_server/configs/exercise_kpt_config_coco.json에서 운동 설정을 불러온다."""
    config_path = Path(__file__).resolve().parent.parent / "configs" / "exercise_kpt_config_coco.json"
    if config_path.exists():
        data = json.loads(config_path.read_text(encoding="utf-8"))
        info = data.get("exercises", {}).get(exercise_name)
        if info:
            return info
    return {
        "exercise_name": exercise_name,
        "eval_config": {"metric_type": "COORDINATE", "motion_direction": "DECREASING"},
        "default_thresholds": {
            "left": {"start_val": 0.8, "target_val": -0.3},
            "right": {"start_val": 0.8, "target_val": -0.3},
        },
    }


class SessionLog:
    """한 세션의 CSV, 23관절 NPZ, MP4 파일을 기록한다."""

    def __init__(
        self,
        patient_dir,
        exercise_name,
        mode,
        session=None,
        round_number=1,
        set_number=1,
        record_video=False,
        video_fps=30.0,
        started_at=None,
        target_score=None,
        *,
        play_session_id=None,
        exercise_record_id=None,
        slot=1,
        patient_id=None,
        data_version=DEFAULT_DATA_VERSION,
        registration="01",
        filename_stem=None,
    ):
        self.patient_dir = Path(patient_dir)
        self.exercise_name = path_component(exercise_name, "exercise_name")
        if mode not in {"CALIBRATION", "MAIN"}:
            raise ValueError("mode must be CALIBRATION or MAIN")
        self.started_at = (started_at or datetime.now(KST)).astimezone(KST)
        self.mode = mode
        self.round = sequence_number(round_number, "round")
        self.set = sequence_number(set_number, "set")
        self.play_session_id = shared_identifier(play_session_id, "play_session_id")
        self.exercise_record_id = shared_identifier(exercise_record_id, "exercise_record_id")
        self.sensor_file_id = uuid.uuid4().hex
        self.slot = sequence_number(slot, "slot")
        self.collection_phase = "Calibration" if mode == "CALIBRATION" else "Play"
        self.patient_id = patient_id or self.patient_dir.name
        self.date_dir = self.patient_dir / self.started_at.strftime("%Y-%m-%d")
        self.directory = self.date_dir / self.exercise_name
        self.directory.mkdir(parents=True, exist_ok=True)
        self.session = sequence_number(1 if session is None else session, "session")

        if filename_stem is None:
            stem = f"{self.started_at:%H-%M-%S}_{mode}_round{self.round:02d}"
            if mode == "MAIN":
                stem += f"_set{self.set:02d}"
        else:
            stem = str(filename_stem)

        take = 1
        while True:
            self.basename = stem if take == 1 else f"{stem}_take{take}"
            self.base = self.directory / self.basename
            if any(self.base.with_suffix(s).exists() for s in (".csv", ".npz", ".mp4")):
                take += 1
                continue
            try:
                self._csv = self.base.with_suffix(".csv").open("x", newline="", encoding="utf-8-sig")
                break
            except FileExistsError:
                take += 1

        self.session_id = self.sensor_file_id
        self._writer = csv.DictWriter(self._csv, fieldnames=FRAME_FIELDS)
        self._writer.writeheader()
        self._csv.flush()

        self.timestamps, self.values, self.keypoints = [], [], []
        self.frame_indices, self.video_indices = [], []
        self.image_aspect_ratios, self.frame_sizes = [], []
        self.total_scores, self.score_deltas = [], []
        self.rep_counts, self.progress_ratios, self.states = [], [], []

        self.record_video = bool(record_video)
        self.video_fps = float(video_fps)
        self._video_writer = None
        self._video_size = None
        self._video_frames = 0
        self._video_origin = None
        self._last_video_elapsed = None
        self._previous_video_frame = None
        self._summary = None

        self.info = {
            "storage_schema": "data.patient.date.exercise/3",
            "num_keypoints": NUM_KEYPOINTS,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat(),
            "patient_id": self.patient_id,
            "motion_name": self.exercise_name,
            "mode": mode,
            "play_session_id": self.play_session_id,
            "exercise_record_id": self.exercise_record_id,
            "sensor_file_id": self.sensor_file_id,
            "source": "Camera",
            "collection_phase": self.collection_phase,
            "data_version": data_version,
            "registration": registration,
            "slot": self.slot,
            "session": self.session,
            "round": self.round,
            "set": self.set,
            "log_root": str(self.patient_dir.parent),
            "directory": str(self.directory),
            "frame_csv": str(self.base.with_suffix(".csv")),
            "skeleton_npz": str(self.base.with_suffix(".npz")),
            "video_mp4": str(self.base.with_suffix(".mp4")) if record_video else None,
            "record_video": record_video,
            "video_fps": self.video_fps,
            "target_score": target_score,
            "scoring_mode": "REPETITIONS" if mode == "CALIBRATION" else "POINTS",
        }
        for field in ("frame_csv", "skeleton_npz", "video_mp4"):
            path = self.info.get(field)
            self.info[f"{field}_relative"] = Path(path).relative_to(self.patient_dir).as_posix() if path else None

    def write_video_frame(self, frame, elapsed_sec):
        if not self.record_video:
            return -1
        import cv2
        elapsed_sec = float(elapsed_sec)
        size = (frame.shape[1], frame.shape[0])
        if self._video_writer is None:
            if size[0] > 1280 or size[1] > 720:
                scale = min(1280.0 / float(size[0]), 720.0 / float(size[1]))
                size = (
                    max(2, int(round(size[0] * scale / 2.0)) * 2),
                    max(2, int(round(size[1] * scale / 2.0)) * 2),
                )
            self._video_size = size
            self.info["video_width"], self.info["video_height"] = size
            self._video_writer = cv2.VideoWriter(
                str(self.base.with_suffix(".mp4")),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.video_fps,
                size,
            )
        if (frame.shape[1], frame.shape[0]) != self._video_size:
            frame = cv2.resize(frame, self._video_size, interpolation=cv2.INTER_LINEAR)
        if self._video_origin is None:
            self._video_origin = elapsed_sec
        target_index = math.floor((elapsed_sec - self._video_origin) * self.video_fps + 1e-6)
        while self._video_frames < target_index:
            self._video_writer.write(self._previous_video_frame)
            self._video_frames += 1
        if self._video_frames == target_index:
            self._video_writer.write(frame)
            self._video_frames += 1
        self._previous_video_frame = frame.copy()
        self._last_video_elapsed = elapsed_sec
        return target_index

    def append_frame(
        self,
        elapsed_sec,
        frame_idx,
        raw_keypoints,
        left,
        right,
        metric_type,
        thresholds,
        threshold_version=0,
        is_occluded=False,
        calib_step="",
        video_frame_idx=-1,
        use_normalization=None,
    ):
        matrix = np.full((NUM_KEYPOINTS, 3), np.nan, dtype=np.float32)
        matrix[:, 2] = 0.0
        for kp in raw_keypoints or []:
            index = kp["id"]
            if 0 <= index < NUM_KEYPOINTS:
                matrix[index] = [kp["x"], kp["y"], kp.get("score", 0.0)]
        values = [np.nan if side.get("val") is None else float(side["val"]) for side in (left, right)]
        self.rep_counts.append([side.get("rep_count", 0) for side in (left, right)])
        self.progress_ratios.append([side.get("progress_ratio", 0.0) for side in (left, right)])
        self.states.append([side.get("state", "") for side in (left, right)])
        if use_normalization is None:
            use_normalization = True

        row = {
            "timestamp_ms": round(elapsed_sec * 1000, 3),
            "frame_idx": frame_idx,
            "calibration_file": self.info.get("calibration", {}).get("file", ""),
            "calibration_version": self.info.get("calibration", {}).get("version", ""),
            "calibration_created_at": self.info.get("calibration", {}).get("created_at", ""),
            "patient_id": self.patient_id,
            "session": self.session,
            "round": self.round,
            "set": self.set,
            "target_reps": self.info.get("target_reps"),
            "target_score": self.info.get("target_score"),
            "time": float(elapsed_sec),
            "play_session_id": self.play_session_id,
            "exercise_record_id": self.exercise_record_id,
            "sensor_file_id": self.sensor_file_id,
            "source": "Camera",
            "collection_phase": self.collection_phase,
            "metric_unit": self.info.get("measurement", {}).get(
                "metric_unit",
                "deg" if metric_type == "ANGLE" else ("body_length" if use_normalization else "image_height"),
            ),
            "use_normalization": use_normalization,
            "counting_method": left.get("counting_method", ""),
            "video_frame_idx": video_frame_idx,
            "motion_name": self.exercise_name,
            "metric_type": metric_type,
            "is_occluded": is_occluded,
            "calib_step": calib_step,
            "threshold_version": threshold_version,
        }
        for side, data, value in zip(("left", "right"), (left, right), values):
            calibration = data.get("calibration", {})
            row.update({
                f"{side}_activation_val": data.get("activation_val", ""),
                f"{side}_reversal_val": data.get("reversal_val", ""),
                f"{side}_value": "" if np.isnan(value) else value,
                f"{side}_calibration_status": calibration.get("state", ""),
                f"{side}_calibration_baseline": calibration.get("baseline", ""),
                f"{side}_calibration_return_tolerance": calibration.get("return_tolerance", ""),
                f"{side}_rep_count": data.get("rep_count", 0),
                f"{side}_state": data.get("state", ""),
                f"{side}_total_score": data.get("total_score", 0.0),
                f"{side}_score_delta": 0.0 if is_occluded else data.get("score_delta", 0.0),
                f"{side}_quality": data.get("quality", ""),
                f"{side}_count_updated": data.get("is_count_updated", False),
                f"{side}_start_val": thresholds[side]["start_val"],
                f"{side}_target_val": thresholds[side]["target_val"],
            })
        self._writer.writerow(row)
        self._csv.flush()

        self.timestamps.append(float(elapsed_sec))
        self.values.append(values)
        self.keypoints.append(matrix)
        frame_meta = (raw_keypoints or [{}])[0]
        self.frame_sizes.append([frame_meta.get("image_width", np.nan), frame_meta.get("image_height", np.nan)])
        self.image_aspect_ratios.append(
            float(raw_keypoints[0].get("image_aspect_ratio", 1.0)) if raw_keypoints else np.nan
        )
        self.frame_indices.append(frame_idx)
        self.video_indices.append(video_frame_idx)
        self.total_scores.append([data.get("total_score", 0.0) for data in (left, right)])
        self.score_deltas.append([0.0 if is_occluded else data.get("score_delta", 0.0) for data in (left, right)])

    def finish(self, rep_rows, status="COMPLETED", end_elapsed_sec=None):
        if self._summary is not None:
            return dict(self._summary)
        self._csv.close()
        if self._video_writer is not None:
            try:
                if end_elapsed_sec is not None and self._video_origin is not None:
                    end_frames = math.ceil((float(end_elapsed_sec) - self._video_origin) * self.video_fps - 1e-6)
                    while self._video_frames < end_frames:
                        self._video_writer.write(self._previous_video_frame)
                        self._video_frames += 1
            finally:
                self._video_writer.release()
                self._video_writer = None

        duration_sec = (
            float(end_elapsed_sec)
            if end_elapsed_sec is not None
            else (self.timestamps[-1] if self.timestamps else 0.0)
        )
        summary = {
            **self.info,
            "status": status,
            "saved_frames": len(self.timestamps),
            "saved_reps_count": len(rep_rows),
            "left_reps": sum(row["side"] == "left" for row in rep_rows),
            "right_reps": sum(row["side"] == "right" for row in rep_rows),
            "left_total_score": sum(row.get("score_delta", 0.0) for row in rep_rows if row["side"] == "left"),
            "right_total_score": sum(row.get("score_delta", 0.0) for row in rep_rows if row["side"] == "right"),
            "duration_sec": duration_sec,
            "video_frames": self._video_frames,
            "video_time_origin_sec": self._video_origin,
            "video_duration_sec": self._video_frames / self.video_fps,
            "ended_at": (self.started_at + timedelta(seconds=duration_sec)).isoformat(),
        }

        npz_path = self.base.with_suffix(".npz")
        temporary = self.base.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as file:
                np.savez_compressed(
                    file,
                    timestamps=np.asarray(self.timestamps, dtype=np.float64),
                    timestamp_ms=np.asarray(self.timestamps, dtype=np.float64) * 1000,
                    frame_idx=np.asarray(self.frame_indices, dtype=np.int64),
                    video_frame_idx=np.asarray(self.video_indices, dtype=np.int64),
                    values=np.asarray(self.values, dtype=np.float32).reshape(-1, 2),
                    total_scores=np.asarray(self.total_scores, dtype=np.float32).reshape(-1, 2),
                    score_deltas=np.asarray(self.score_deltas, dtype=np.float32).reshape(-1, 2),
                    rep_counts=np.asarray(self.rep_counts, dtype=np.int32).reshape(-1, 2),
                    progress_ratios=np.asarray(self.progress_ratios, dtype=np.float32).reshape(-1, 2),
                    states=np.asarray(self.states, dtype=str).reshape(-1, 2),
                    keypoints=np.asarray(self.keypoints, dtype=np.float32).reshape(-1, NUM_KEYPOINTS, 3),
                    image_aspect_ratio=np.asarray(self.image_aspect_ratios, dtype=np.float32),
                    frame_sizes=np.asarray(self.frame_sizes, dtype=np.float32).reshape(-1, 2),
                    player_id=self.patient_id,
                    exercise_name=self.exercise_name,
                    play_session_id=self.play_session_id,
                    exercise_record_id=self.exercise_record_id,
                    sensor_file_id=self.sensor_file_id,
                    data_version=self.info["data_version"],
                    registration=self.info["registration"],
                    slot=self.slot,
                    session_id=self.session_id,
                    session_type=self.mode,
                    summary_json=json.dumps(summary, ensure_ascii=False),
                    reps_json=json.dumps(rep_rows, ensure_ascii=False),
                )
            temporary.replace(npz_path)
        finally:
            temporary.unlink(missing_ok=True)
        self._summary = summary
        return dict(summary)


class DataManager:
    """환자별 폴더와 캘리브레이션 임계값 JSON, 세션 로그를 관리한다."""

    def __init__(
        self,
        player_id="0001",
        patient_name="Unknown",
        log_root=None,
        patient_metadata=None,
        data_version=DEFAULT_DATA_VERSION,
        registration=1,
    ):
        self.player_id = path_component(player_id, "patient_id")
        if self.player_id.isdigit():
            if len(self.player_id) > 4:
                raise ValueError("patient_id must have at most four digits")
            self.player_id = self.player_id.zfill(4)
        self.data_version = version_folder(data_version)
        self.registration = registration_number(registration)
        self.patient_name = patient_name
        self.log_root = Path(log_root).expanduser().resolve() if log_root else default_log_root()
        self.patient_dir = self.log_root / self.player_id
        self.patient_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.patient_dir / "patient_metadata.csv"
        self.patient_metadata = patient_metadata or {}
        self.session_log = None

    def _save_metadata(self):
        fields = ["patient_id", "patient_name", "height_cm", "weight_kg"]
        metadata = {}
        if self.metadata_path.exists():
            with self.metadata_path.open(newline="", encoding="utf-8-sig") as file:
                metadata = next(csv.DictReader(file), {})
        metadata.update(patient_id=self.player_id, patient_name=self.patient_name)
        for key in ("height_cm", "weight_kg"):
            if key in self.patient_metadata and self.patient_metadata[key] is not None:
                metadata[key] = float(self.patient_metadata[key])
        temporary = self.metadata_path.with_suffix(".csv.tmp")
        try:
            with temporary.open("w", newline="", encoding="utf-8-sig") as file:
                writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerow(metadata)
            temporary.replace(self.metadata_path)
        finally:
            temporary.unlink(missing_ok=True)

    def begin_session(
        self,
        exercise_name,
        mode,
        session=None,
        round_number=1,
        set_number=1,
        record_video=False,
        video_fps=30.0,
        started_at=None,
        target_score=None,
        play_session_id=None,
        exercise_record_id=None,
        slot=1,
        filename_stem=None,
    ):
        self._save_metadata()
        self.session_log = SessionLog(
            self.patient_dir,
            exercise_name,
            mode,
            session,
            round_number,
            set_number,
            record_video,
            video_fps,
            started_at,
            target_score,
            play_session_id=play_session_id,
            exercise_record_id=exercise_record_id,
            slot=slot,
            patient_id=self.player_id,
            data_version=self.data_version,
            registration=self.registration,
            filename_stem=filename_stem,
        )
        return dict(self.session_log.info)

    def _round_threshold_files(self, exercise_name):
        log = self.session_log
        prefix = f"round{log.round:02d}_thresholds_v"
        files = []
        for path in log.directory.glob(prefix + "*.json"):
            version = path.stem[len(prefix):]
            if version.isdigit():
                files.append((int(version), path))
        return sorted(files, reverse=True)

    def _calibration_reference(self, record, path):
        return dict(
            file=path.name,
            path=path.relative_to(self.patient_dir).as_posix(),
            version=record["version"],
            created_at=record["created_at"],
            calibration_session_id=record["calibration_session_id"],
            date=record["date"],
            round=record["round"],
            exercise_name=record["exercise_name"],
            thresholds=deepcopy(record["thresholds"]),
        )

    def save_custom_threshold(self, exercise_name, threshold_data, is_confirmed=True):
        files = self._round_threshold_files(exercise_name)
        log = self.session_log
        version = max((number for number, _ in files), default=0) + 1
        record = dict(
            schema="round-calibration/1",
            patient_id=self.player_id,
            data_version=self.data_version,
            registration=self.registration,
            date=log.started_at.strftime("%Y-%m-%d"),
            exercise_name=exercise_name,
            round=log.round,
            created_at=datetime.now(KST).isoformat(),
            calibration_session_id=log.session_id,
            calibration_measurement=deepcopy(log.info.get("measurement", {})),
            calibration_log=log.basename,
            is_confirmed=bool(is_confirmed),
            thresholds=threshold_data,
        )
        while True:
            path = log.directory / f"round{log.round:02d}_thresholds_v{version:02d}.json"
            record["version"] = version
            try:
                with path.open("x", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False, indent=2, allow_nan=False))
                break
            except FileExistsError:
                version += 1
        log.info["calibration"] = self._calibration_reference(record, path)
        return dict(log.info["calibration"])

    def get_custom_threshold(self, exercise_name):
        log = self.session_log
        for version, path in self._round_threshold_files(exercise_name):
            record = json.loads(path.read_text(encoding="utf-8"))
            expected = dict(
                patient_id=self.player_id,
                data_version=self.data_version,
                registration=self.registration,
                exercise_name=exercise_name,
                date=log.started_at.strftime("%Y-%m-%d"),
                round=log.round,
            )
            if any(record.get(k) != v for k, v in expected.items()) or not record.get("is_confirmed"):
                continue
            log.info["calibration"] = self._calibration_reference(record, path)
            return record["thresholds"]
        return None

    def finish_session(self, rep_rows, status="COMPLETED", end_elapsed_sec=None):
        if self.session_log is None:
            return None
        return self.session_log.finish(rep_rows, status, end_elapsed_sec=end_elapsed_sec)
