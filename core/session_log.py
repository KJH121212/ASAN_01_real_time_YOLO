"""data/환자번호/날짜/동작 경로에 MP4·NPZ·CSV만 저장합니다."""

import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
import sys
from pathlib import Path
import re
import uuid

import numpy as np
from core.patient_data import DEFAULT_DATA_VERSION, shared_identifier


KST = timezone(timedelta(hours=9))
FRAME_FIELDS = [
    'time', 'play_session_id', 'exercise_record_id', 'sensor_file_id', 'source', 'collection_phase', 'metric_unit',
    'timestamp_ms', 'frame_idx', 'video_frame_idx', 'motion_name', 'metric_type',
    'left_value', 'right_value', 'left_rep_count', 'right_rep_count',
    'left_total_score', 'right_total_score', 'left_score_delta', 'right_score_delta',
    'left_state', 'right_state', 'left_quality', 'right_quality',
    'left_calibration_status', 'right_calibration_status',
    'left_calibration_baseline', 'right_calibration_baseline',
    'left_calibration_return_tolerance', 'right_calibration_return_tolerance',
    'left_count_updated', 'right_count_updated', 'is_occluded', 'calib_step',
    'threshold_version', 'left_start_val', 'left_target_val', 'right_start_val', 'right_target_val',
    'use_normalization', 'counting_method',
    'left_activation_val', 'right_activation_val', 'left_reversal_val', 'right_reversal_val',
    'patient_id', 'session', 'round', 'set', 'target_reps', 'target_score',
    'calibration_file', 'calibration_version', 'calibration_created_at',
]


def default_log_root():
    """세션 로그가 저장될 기본 루트(data) 폴더 경로를 반환한다."""
    configured = os.environ.get('ASAN_LOG_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    if getattr(sys, "frozen", False):
        return (Path(sys.executable).resolve().parent / 'data').resolve()
    return (Path(__file__).resolve().parent.parent / 'data').resolve()


def path_component(value, name):
    """파일·폴더 이름에 안전한 문자만 들어있는지 검사한다."""
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value) or Path(value).is_reserved():
        raise ValueError(f'{name} must contain only letters, numbers, underscores or hyphens')
    return value


def sequence_number(value, name):
    """회차/라운드/세트 번호가 0 이상의 정수인지 검사한다."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def write_json_atomic(path, data):
    """임시 파일에 먼저 쓰고 교체해 JSON 저장 중 파일 깨짐을 막는다."""
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SessionLog:
    """한 운동 세션의 CSV, NPZ, MP4 파일을 생성하고 프레임·요약 데이터를 기록하는 클래스."""

    def __init__(self, patient_dir, exercise_name, mode, session=None, round_number=1,
                 set_number=1, record_video=False, video_fps=30.0, started_at=None, target_score=None,
                 *, play_session_id=None, exercise_record_id=None, slot=1, patient_id=None,
                 data_version=DEFAULT_DATA_VERSION, registration='01', filename_stem=None):
        self.patient_dir = Path(patient_dir)
        self.exercise_name = path_component(exercise_name, 'exercise_name')
        if mode not in {'CALIBRATION', 'MAIN'}:
            raise ValueError('mode must be CALIBRATION or MAIN')
        if not isinstance(record_video, bool):
            raise ValueError('record_video must be a boolean')
        if not np.isfinite(video_fps) or video_fps <= 0:
            raise ValueError('video_fps must be positive')
        self.started_at = started_at or datetime.now(KST)
        if self.started_at.tzinfo is None:
            self.started_at = self.started_at.replace(tzinfo=KST)
        self.started_at = self.started_at.astimezone(KST)
        self.mode = mode
        self.round = sequence_number(round_number, 'round')
        self.set = sequence_number(set_number, 'set')
        self.play_session_id = shared_identifier(play_session_id, 'play_session_id')
        self.exercise_record_id = shared_identifier(exercise_record_id, 'exercise_record_id')
        self.sensor_file_id = uuid.uuid4().hex
        self.slot = sequence_number(slot, 'slot')
        self.collection_phase = 'Calibration' if mode == 'CALIBRATION' else 'Play'
        self.patient_id = patient_id or self.patient_dir.name
        self.date_dir = self.patient_dir / self.started_at.strftime('%Y-%m-%d')
        self.directory = self.date_dir / self.exercise_name
        self.directory.mkdir(parents=True, exist_ok=True)
        if session is None:
            session = 1
        self.session = sequence_number(session, 'session')
        take = 1
        # 같은 세트 번호라도 라운드를 파일명에서 바로 구분합니다.
        if filename_stem is None:
            stem = f'{self.started_at:%H-%M-%S}_{mode}_round{self.round:02d}'
            if mode == 'MAIN':
                stem += f'_set{self.set:02d}'
        else:
            stem = str(filename_stem)
            if (not stem or stem in {'.', '..'} or Path(stem).name != stem
                    or re.search(r'[<>:"/\\|?*\x00-\x1f]', stem)
                    or stem.endswith((' ', '.')) or Path(stem).is_reserved()):
                raise ValueError('filename_stem must be a valid filename without an extension')
        while True:
            self.basename = stem if take == 1 else f'{stem}_take{take}'
            self.base = self.directory / self.basename
            if any(self.base.with_suffix(suffix).exists() for suffix in ('.csv', '.npz', '.mp4')):
                take += 1
                continue
            try:
                self._csv = self.base.with_suffix('.csv').open('x', newline='', encoding='utf-8-sig')
                break
            except FileExistsError:
                take += 1
        self.session_id = self.sensor_file_id
        self._writer = csv.DictWriter(self._csv, fieldnames=FRAME_FIELDS)
        self._writer.writeheader()
        self._csv.flush()
        self.timestamps, self.values, self.keypoints = [], [], []
        self.frame_indices, self.video_indices = [], []
        self.image_aspect_ratios = []
        self.frame_sizes = []
        self.total_scores, self.score_deltas = [], []
        self.rep_counts, self.progress_ratios, self.states = [], [], []
        self.record_video = record_video
        self.video_fps = float(video_fps)
        self._video_writer = None
        self._video_size = None
        self._video_frames = 0
        self._video_origin = None
        self._last_video_elapsed = None
        self._previous_video_frame = None
        self._summary = None
        self.info = {
            'storage_schema': 'data.patient.date.exercise/2',
            'session_id': self.session_id, 'started_at': self.started_at.isoformat(),
            'patient_id': self.patient_id, 'motion_name': self.exercise_name, 'mode': mode,
            'play_session_id': self.play_session_id, 'exercise_record_id': self.exercise_record_id,
            'sensor_file_id': self.sensor_file_id, 'source': 'Camera', 'collection_phase': self.collection_phase,
            'data_version': data_version, 'registration': registration, 'slot': self.slot,
            'session': self.session, 'round': self.round, 'set': self.set,
            'log_root': str(self.patient_dir.parent), 'directory': str(self.directory),
            'frame_csv': str(self.base.with_suffix('.csv')),
            'skeleton_npz': str(self.base.with_suffix('.npz')),
            'video_mp4': str(self.base.with_suffix('.mp4')) if record_video else None,
            'record_video': record_video, 'video_fps': self.video_fps,
            'target_score': target_score,
            'scoring_mode': 'REPETITIONS' if mode == 'CALIBRATION' else 'POINTS',
        }
        for field in ('frame_csv', 'skeleton_npz', 'video_mp4'):
            path = self.info.get(field)
            self.info[f'{field}_relative'] = Path(path).relative_to(self.patient_dir).as_posix() if path else None

    def write_video_frame(self, frame, elapsed_sec):
        """AI 추론 속도와 무관하게 실제 경과 시간에 맞춰 MP4 프레임 간격을 채워 저장한다."""
        if not self.record_video:
            return -1
        if self._summary is not None:
            raise RuntimeError('Session log is already closed')
        elapsed_sec = float(elapsed_sec)
        if not math.isfinite(elapsed_sec) or elapsed_sec < 0:
            raise ValueError('Video timestamp must be finite and nonnegative')
        if self._last_video_elapsed is not None and elapsed_sec < self._last_video_elapsed:
            raise ValueError('Video timestamps must be nondecreasing')
        import cv2
        size = (frame.shape[1], frame.shape[0])
        if self._video_writer is None:
            if size[0] > 1280 or size[1] > 720:
                scale = min(1280.0 / float(size[0]), 720.0 / float(size[1]))
                size = (max(2, int(round(size[0] * scale / 2.0)) * 2), max(2, int(round(size[1] * scale / 2.0)) * 2))
            self._video_size = size
            self.info['video_width'], self.info['video_height'] = size
            self._video_writer = cv2.VideoWriter(
                str(self.base.with_suffix('.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, size)
            if not self._video_writer.isOpened():
                self._video_writer.release()
                self._video_writer = None
                raise RuntimeError(f'Could not open MP4 recorder: {self.base.with_suffix(".mp4")}')
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

    def append_frame(self, elapsed_sec, frame_idx, raw_keypoints, left, right, metric_type,
                     thresholds, threshold_version=0, is_occluded=False, calib_step='', video_frame_idx=-1,
                     use_normalization=None):
        """프레임 1개 분량의 관절 좌표와 좌우 측정 결과를 CSV와 메모리 버퍼에 추가한다."""
        if self._summary is not None:
            raise RuntimeError('Session log is already closed')
        matrix = np.full((17, 3), np.nan, dtype=np.float32)
        matrix[:, 2] = 0.0
        for kp in raw_keypoints or []:
            index = kp['id']
            if 0 <= index < 17:
                matrix[index] = [kp['x'], kp['y'], kp.get('score', 0.0)]
        values = [np.nan if side.get('val') is None else float(side['val']) for side in (left, right)]
        self.rep_counts.append([side.get('rep_count', 0) for side in (left, right)])
        self.progress_ratios.append([side.get('progress_ratio', 0.) for side in (left, right)])
        self.states.append([side.get('state', '') for side in (left, right)])
        if use_normalization is None:
            use_normalization = True
        row = {'timestamp_ms': round(elapsed_sec * 1000, 3), 'frame_idx': frame_idx,
               'calibration_file': self.info.get('calibration', {}).get('file', ''),
               'calibration_version': self.info.get('calibration', {}).get('version', ''),
               'calibration_created_at': self.info.get('calibration', {}).get('created_at', ''),
               'patient_id': self.patient_id, 'session': self.session, 'round': self.round, 'set': self.set,
               'target_reps': self.info.get('target_reps'), 'target_score': self.info.get('target_score'),
               'time': float(elapsed_sec), 'play_session_id': self.play_session_id,
               'exercise_record_id': self.exercise_record_id, 'sensor_file_id': self.sensor_file_id,
               'source': 'Camera', 'collection_phase': self.collection_phase,
               'metric_unit': self.info.get('measurement', {}).get('metric_unit',
                   'deg' if metric_type == 'ANGLE' else 'body_length' if use_normalization else 'image_height'),
               'use_normalization': use_normalization, 'counting_method': left.get('counting_method', ''),
               'video_frame_idx': video_frame_idx, 'motion_name': self.exercise_name,
               'metric_type': metric_type, 'is_occluded': is_occluded, 'calib_step': calib_step,
               'threshold_version': threshold_version}
        for side, data, value in zip(('left', 'right'), (left, right), values):
            row.update({f'{side}_activation_val': data.get('activation_val', ''),
                        f'{side}_reversal_val': data.get('reversal_val', '')})
            calibration = data.get('calibration', {})
            row.update({f'{side}_value': '' if np.isnan(value) else value,
                        f'{side}_calibration_status': calibration.get('state', ''),
                        f'{side}_calibration_baseline': calibration.get('baseline', ''),
                        f'{side}_calibration_return_tolerance': calibration.get('return_tolerance', ''),
                        f'{side}_rep_count': data.get('rep_count', 0), f'{side}_state': data.get('state', ''),
                        f'{side}_total_score': data.get('total_score', 0.0),
                        f'{side}_score_delta': 0.0 if is_occluded else data.get('score_delta', 0.0),
                        f'{side}_quality': data.get('quality', ''),
                        f'{side}_count_updated': data.get('is_count_updated', False),
                        f'{side}_start_val': thresholds[side]['start_val'],
                        f'{side}_target_val': thresholds[side]['target_val']})
        self._writer.writerow(row)
        self._csv.flush()
        self.timestamps.append(float(elapsed_sec))
        self.values.append(values)
        self.keypoints.append(matrix)
        frame_meta = (raw_keypoints or [{}])[0]
        self.frame_sizes.append([frame_meta.get('image_width', np.nan), frame_meta.get('image_height', np.nan)])
        self.image_aspect_ratios.append(
            float(raw_keypoints[0].get('image_aspect_ratio', 1.0)) if raw_keypoints else np.nan)
        self.frame_indices.append(frame_idx)
        self.video_indices.append(video_frame_idx)
        self.total_scores.append([data.get('total_score', 0.0) for data in (left, right)])
        self.score_deltas.append([0.0 if is_occluded else data.get('score_delta', 0.0) for data in (left, right)])

    def finish(self, rep_rows, status='COMPLETED', end_elapsed_sec=None):
        """CSV·MP4 기록을 마무리하고 요약·반복 결과를 포함한 NPZ 파일을 저장한다."""
        if self._summary is not None:
            return dict(self._summary)
        if end_elapsed_sec is not None:
            end_elapsed_sec = float(end_elapsed_sec)
            if not math.isfinite(end_elapsed_sec) or end_elapsed_sec < 0:
                raise ValueError('Session end timestamp must be finite and nonnegative')
            if self._last_video_elapsed is not None and end_elapsed_sec < self._last_video_elapsed:
                raise ValueError('Session cannot end before the last video timestamp')
        self._csv.close()
        if self._video_writer is not None:
            try:
                if end_elapsed_sec is not None:
                    end_frames = math.ceil((end_elapsed_sec - self._video_origin) * self.video_fps - 1e-6)
                    while self._video_frames < end_frames:
                        self._video_writer.write(self._previous_video_frame)
                        self._video_frames += 1
            finally:
                self._video_writer.release()
                self._video_writer = None
        summary = {**self.info, 'status': status, 'saved_frames': len(self.timestamps),
                   'saved_reps_count': len(rep_rows), 'left_reps': sum(row['side'] == 'left' for row in rep_rows),
                   'right_reps': sum(row['side'] == 'right' for row in rep_rows),
                   'left_total_score': sum(row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'left'),
                   'right_total_score': sum(row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'right'),
                   'duration_sec': (end_elapsed_sec if end_elapsed_sec is not None
                                    else self.timestamps[-1] if self.timestamps else 0.0),
                   'video_frames': self._video_frames, 'video_time_origin_sec': self._video_origin,
                   'video_duration_sec': self._video_frames / self.video_fps}
        ended_at = self.started_at + timedelta(seconds=summary['duration_sec'])
        summary['ended_at'] = ended_at.isoformat()

        npz_path = self.base.with_suffix('.npz')
        temporary = self.base.with_suffix('.npz.tmp')
        try:
            with temporary.open('wb') as file:
                np.savez_compressed(file, timestamps=np.asarray(self.timestamps, dtype=np.float64),
                    timestamp_ms=np.asarray(self.timestamps, dtype=np.float64) * 1000,
                    frame_idx=np.asarray(self.frame_indices, dtype=np.int64),
                    video_frame_idx=np.asarray(self.video_indices, dtype=np.int64),
                    values=np.asarray(self.values, dtype=np.float32).reshape(-1, 2),
                    total_scores=np.asarray(self.total_scores, dtype=np.float32).reshape(-1, 2),
                    score_deltas=np.asarray(self.score_deltas, dtype=np.float32).reshape(-1, 2),
                    rep_counts=np.asarray(self.rep_counts, dtype=np.int32).reshape(-1, 2),
                    progress_ratios=np.asarray(self.progress_ratios, dtype=np.float32).reshape(-1, 2),
                    states=np.asarray(self.states, dtype=str).reshape(-1, 2),
                    keypoints=np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 17, 3),
                    image_aspect_ratio=np.asarray(self.image_aspect_ratios, dtype=np.float32),
                    frame_sizes=np.asarray(self.frame_sizes, dtype=np.float32).reshape(-1, 2),
                    player_id=self.patient_id, exercise_name=self.exercise_name,
                    play_session_id=self.play_session_id, exercise_record_id=self.exercise_record_id,
                    sensor_file_id=self.sensor_file_id, data_version=self.info['data_version'],
                    registration=self.info['registration'], slot=self.slot,
                    session_id=self.session_id, session_type=self.mode,
                    # 별도 파일 없이 요약과 반복 결과를 함께 보관합니다.
                    summary_json=json.dumps(summary, ensure_ascii=False),
                    reps_json=json.dumps(rep_rows, ensure_ascii=False))
            temporary.replace(npz_path)
        finally:
            temporary.unlink(missing_ok=True)
        self._summary = summary
        return dict(summary)
