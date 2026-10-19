"""Patient/date/exercise logs, with a single basename for CSV/NPZ/MP4."""

import csv
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np


KST = timezone(timedelta(hours=9))
FRAME_FIELDS = [
    'timestamp_ms', 'frame_idx', 'video_frame_idx', 'motion_name', 'metric_type',
    'left_value', 'right_value', 'left_rep_count', 'right_rep_count',
    'left_total_score', 'right_total_score', 'left_score_delta', 'right_score_delta',
    'left_state', 'right_state', 'left_quality', 'right_quality',
    'left_count_updated', 'right_count_updated', 'is_occluded', 'calib_step',
    'threshold_version', 'left_start_val', 'left_target_val', 'right_start_val', 'right_target_val',
]
REP_FIELDS = ['session_id', 'side', 'rep_num', 'duration_sec', 'min_val', 'max_val',
              'achieved_rom', 'quality', 'score_delta', 'total_score', 'start_val', 'target_val', 'threshold_version']
GAME_FIELDS = ['session_id', 'started_at', 'patient_id', 'motion_name', 'mode', 'session', 'round', 'set',
               'status', 'saved_frames', 'left_reps', 'right_reps', 'left_total_score', 'right_total_score',
               'target_score', 'duration_sec',
               'calories_kcal', 'max_speed', 'frame_csv', 'skeleton_npz', 'video_mp4']


def default_log_root():
    configured = os.environ.get('ASAN_LOG_ROOT')
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path(__file__).resolve().parent.parent / 'data').resolve()


def path_component(value, name):
    value = str(value)
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value) or Path(value).is_reserved():
        raise ValueError(f'{name} must contain only letters, numbers, underscores or hyphens')
    return value


def sequence_number(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def write_json_atomic(path, data):
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class SessionLog:
    def __init__(self, patient_dir, exercise_name, mode, session=None, round_number=1,
                 set_number=1, record_video=False, video_fps=30.0, started_at=None, target_score=None):
        self.patient_dir = Path(patient_dir)
        self.exercise_name = path_component(exercise_name, 'exercise_name')
        if mode not in {'CALIBRATION', 'MAIN', 'TEST'}:
            raise ValueError('mode must be CALIBRATION, MAIN or TEST')
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
        self.date_dir = self.patient_dir / self.started_at.strftime('%y%m%d')
        self.directory = self.date_dir / self.exercise_name
        self.directory.mkdir(parents=True, exist_ok=True)
        if session is None:
            numbers = [int(match.group(1)) for path in self.directory.glob('session*.csv')
                       if (match := re.match(r'session(\d+)_', path.name))]
            session = max(numbers, default=0) + 1
        self.session = sequence_number(session, 'session')
        stem = (f'session{session}_cali' if mode == 'CALIBRATION'
                else f'session{session}_round{self.round}_set{self.set}')
        take = 1
        while True:
            self.basename = stem if take == 1 else f'{stem}_take{take}'
            self.base = self.directory / self.basename
            if any(self.base.with_suffix(suffix).exists() for suffix in ('.csv', '.npz', '.mp4', '.json')):
                take += 1
                continue
            try:
                self._csv = self.base.with_suffix('.csv').open('x', newline='', encoding='utf-8-sig')
                break
            except FileExistsError:
                take += 1
        self.session_id = uuid.uuid4().hex
        self._writer = csv.DictWriter(self._csv, fieldnames=FRAME_FIELDS)
        self._writer.writeheader()
        self._csv.flush()
        self.timestamps, self.values, self.keypoints = [], [], []
        self.frame_indices, self.video_indices = [], []
        self.total_scores, self.score_deltas = [], []
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
            'session_id': self.session_id, 'started_at': self.started_at.isoformat(),
            'patient_id': self.patient_dir.name, 'motion_name': self.exercise_name, 'mode': mode,
            'session': self.session, 'round': self.round, 'set': self.set,
            'log_root': str(self.patient_dir.parent), 'directory': str(self.directory),
            'frame_csv': str(self.base.with_suffix('.csv')),
            'skeleton_npz': str(self.base.with_suffix('.npz')),
            'video_mp4': str(self.base.with_suffix('.mp4')) if record_video else None,
            'reps_csv': str(self.directory / f'{self.basename}_reps.csv'),
            'manifest_json': str(self.base.with_suffix('.json')),
            'gamedata_csv': str(self.date_dir / 'gamedata.csv'),
            'record_video': record_video, 'video_fps': self.video_fps,
            'target_score': target_score,
            'scoring_mode': 'REPETITIONS' if mode == 'CALIBRATION' else 'POINTS',
        }
        try:
            write_json_atomic(self.base.with_suffix('.json'), {**self.info, 'status': 'RECORDING'})
        except Exception:
            self._csv.close()
            raise

    def write_video_frame(self, frame, elapsed_sec):
        """Resample capture timestamps onto the MP4 clock, independent of inference FPS."""
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
            self._video_size = size
            self._video_writer = cv2.VideoWriter(
                str(self.base.with_suffix('.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, size)
            if not self._video_writer.isOpened():
                self._video_writer.release()
                self._video_writer = None
                raise RuntimeError(f'Could not open MP4 recorder: {self.base.with_suffix(".mp4")}')
        if size != self._video_size:
            frame = cv2.resize(frame, self._video_size)
        if self._video_origin is None:
            self._video_origin = elapsed_sec
        target_index = math.floor((elapsed_sec - self._video_origin) * self.video_fps + 1e-6)
        # Fill gaps at low/irregular input FPS. Multiple inputs within one output
        # slot must not append extra frames and stretch the recording either.
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
                     thresholds, threshold_version=0, is_occluded=False, calib_step='', video_frame_idx=-1):
        if self._summary is not None:
            raise RuntimeError('Session log is already closed')
        matrix = np.full((17, 3), np.nan, dtype=np.float32)
        matrix[:, 2] = 0.0
        for kp in raw_keypoints or []:
            index = kp['id']
            if 0 <= index < 17:
                matrix[index] = [kp['x'], kp['y'], kp.get('score', 0.0)]
        values = [np.nan if side.get('val') is None else float(side['val']) for side in (left, right)]
        row = {'timestamp_ms': round(elapsed_sec * 1000, 3), 'frame_idx': frame_idx,
               'video_frame_idx': video_frame_idx, 'motion_name': self.exercise_name,
               'metric_type': metric_type, 'is_occluded': is_occluded, 'calib_step': calib_step,
               'threshold_version': threshold_version}
        for side, data, value in zip(('left', 'right'), (left, right), values):
            row.update({f'{side}_value': '' if np.isnan(value) else value,
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
        self.frame_indices.append(frame_idx)
        self.video_indices.append(video_frame_idx)
        self.total_scores.append([data.get('total_score', 0.0) for data in (left, right)])
        self.score_deltas.append([0.0 if is_occluded else data.get('score_delta', 0.0) for data in (left, right)])

    def finish(self, rep_rows, status='COMPLETED', end_elapsed_sec=None):
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
                    # Include the last inference/wait interval up to stop/disconnect.
                    while self._video_frames < end_frames:
                        self._video_writer.write(self._previous_video_frame)
                        self._video_frames += 1
            finally:
                self._video_writer.release()
                self._video_writer = None
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
                    keypoints=np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 17, 3),
                    player_id=self.patient_dir.name, exercise_name=self.exercise_name,
                    session_id=self.session_id, session_type=self.mode)
            temporary.replace(npz_path)
        finally:
            temporary.unlink(missing_ok=True)
        with Path(self.info['reps_csv']).open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=REP_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows({**row, 'session_id': self.session_id} for row in rep_rows)
        summary = {**self.info, 'status': status, 'saved_frames': len(self.timestamps),
                   'saved_reps_count': len(rep_rows), 'left_reps': sum(row['side'] == 'left' for row in rep_rows),
                   'right_reps': sum(row['side'] == 'right' for row in rep_rows),
                   'left_total_score': sum(row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'left'),
                   'right_total_score': sum(row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'right'),
                   'duration_sec': (end_elapsed_sec if end_elapsed_sec is not None
                                    else self.timestamps[-1] if self.timestamps else 0.0),
                   'video_frames': self._video_frames, 'video_time_origin_sec': self._video_origin,
                   'video_duration_sec': self._video_frames / self.video_fps}
        # No calorie/speed estimates are fabricated; these fields remain empty.
        daily = Path(self.info['gamedata_csv'])
        existing_rows = []
        if daily.exists():
            with daily.open(newline='', encoding='utf-8-sig') as file:
                existing_rows = [row for row in csv.DictReader(file) if row['session_id'] != self.session_id]
        daily_temporary = daily.with_suffix('.csv.tmp')
        with daily_temporary.open('w', newline='', encoding='utf-8-sig') as file:
            writer = csv.DictWriter(file, fieldnames=GAME_FIELDS, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(existing_rows)
            writer.writerow(summary)
        daily_temporary.replace(daily)
        write_json_atomic(self.base.with_suffix('.json'), summary)
        self._summary = summary
        return dict(summary)
