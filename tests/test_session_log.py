import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.data_manager import DataManager
from core.session_log import KST, default_log_root


THRESHOLDS = {side: {'start_val': 0.1, 'target_val': 0.9} for side in ('left', 'right')}


def rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        return list(csv.DictReader(file))


class SessionLogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.manager = DataManager('0001', 'Test Patient', log_root=self.root,
                                   patient_metadata={'height_cm': 170, 'weight_kg': 65})
        self.start = datetime(2026, 1, 1, 12, 0, tzinfo=KST)

    def tearDown(self):
        if self.manager.session_log:
            self.manager.finish_session([], status='TEST')
        self.temporary.cleanup()

    def begin(self, **kwargs):
        return self.manager.begin_session('biceps_curl', 'MAIN', session=1, started_at=self.start, **kwargs)

    def test_video_file_cannot_be_used_as_log_directory(self):
        video = self.root / 'source.mp4'
        video.write_bytes(b'original video')
        for destination in (video, video / 'logs'):
            with self.assertRaisesRegex(NotADirectoryError, 'Video File Path'):
                DataManager('0001', log_root=destination)
        self.assertEqual(video.read_bytes(), b'original video')

    def append(self, frame=0, missing=False, video_index=-1):
        points = [{'id': i, 'x': 0.25, 'y': 0.75, 'score': 0.9} for i in range(17)]
        self.manager.session_log.append_frame(frame / 30, frame, None if missing else points,
            {'val': None if missing else 0.5, 'rep_count': 2}, {'val': 0.7},
            'RELATIVE_Y', THRESHOLDS, threshold_version=3,
            is_occluded=missing, video_frame_idx=video_index)

    def test_csv_npz_and_metadata_follow_patient_date_exercise_structure(self):
        info = self.begin(round_number=2, set_number=3)
        expected = self.root / '0001' / '260101' / 'biceps_curl' / 'session1_round2_set3.csv'
        self.assertEqual(Path(info['frame_csv']), expected)
        self.append()
        self.append(1, missing=True)
        summary = self.manager.finish_session([])
        self.assertEqual(summary['saved_frames'], 2)
        frames = rows(expected)
        self.assertEqual(frames[0]['left_value'], '0.5')
        self.assertEqual(frames[0]['threshold_version'], '3')
        self.assertEqual(frames[1]['left_value'], '')
        self.assertEqual(frames[1]['is_occluded'], 'True')
        with np.load(info['skeleton_npz'], allow_pickle=False) as data:
            self.assertEqual(data['keypoints'].shape, (2, 17, 3))
            np.testing.assert_array_equal(data['frame_idx'], [0, 1])
            self.assertTrue(np.isnan(data['keypoints'][1, :, :2]).all())
            self.assertEqual(data['timestamps'].dtype, np.float64)
            self.assertAlmostEqual(data['timestamp_ms'][1], 1000 / 30)
        metadata = rows(self.manager.metadata_path)[0]
        self.assertEqual(metadata['patient_id'], '0001')
        self.assertEqual(metadata['height_cm'], '170.0')
        self.assertEqual(rows(summary['gamedata_csv'])[0]['calories_kcal'], '')

    def test_actual_mp4_uses_same_basename_and_frame_indices(self):
        info = self.begin(record_video=True)
        for i in range(3):
            index = self.manager.session_log.write_video_frame(
                np.full((24, 32, 3), i * 40, np.uint8), elapsed_sec=i / 30)
            self.assertEqual(index, i)
            self.append(i, video_index=index)
        self.manager.finish_session([])
        for field in ('frame_csv', 'skeleton_npz', 'video_mp4'):
            self.assertTrue(Path(info[field]).is_file())
            self.assertEqual(Path(info[field]).stem, 'session1_round1_set1')
        cap = cv2.VideoCapture(info['video_mp4'])
        try:
            self.assertTrue(cap.isOpened())
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
        finally:
            cap.release()
        with np.load(info['skeleton_npz']) as data:
            np.testing.assert_array_equal(data['video_frame_idx'], [0, 1, 2])

    def test_repeat_does_not_overwrite_and_finish_is_idempotent(self):
        first = self.begin()
        self.append()
        self.manager.finish_session([])
        original = Path(first['frame_csv']).read_bytes()
        self.manager.finish_session([])
        second = self.begin()
        self.assertTrue(second['frame_csv'].endswith('session1_round1_set1_take2.csv'))
        self.manager.finish_session([], status='STOPPED')
        self.assertEqual(Path(first['frame_csv']).read_bytes(), original)
        self.assertEqual(len(rows(first['gamedata_csv'])), 2)

    def test_live_video_preserves_elapsed_time_between_analysis_frames(self):
        info = self.begin(record_video=True)
        log = self.manager.session_log
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        indices = [log.write_video_frame(frame, elapsed_sec=time) for time in (0.5, 0.6, 0.7)]
        self.assertEqual(indices, [0, 3, 6])
        summary = self.manager.finish_session([])
        self.assertEqual(summary['video_time_origin_sec'], 0.5)
        cap = cv2.VideoCapture(info['video_mp4'])
        try:
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
        finally:
            cap.release()

    def test_recorded_duration_matches_real_time_at_different_input_rates(self):
        # A four-second capture must play for four seconds even at 6 or 60 FPS.
        for input_fps in (6, 13, 30, 60):
            with self.subTest(input_fps=input_fps):
                info = self.begin(record_video=True)
                log = self.manager.session_log
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                for i in range(input_fps * 4):
                    log.write_video_frame(frame, elapsed_sec=0.25 + i / input_fps)
                summary = self.manager.finish_session([], end_elapsed_sec=4.25)
                cap = cv2.VideoCapture(info['video_mp4'])
                try:
                    playback = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    self.assertAlmostEqual(playback, 4.0, delta=1 / 30)
                    self.assertEqual(summary['video_duration_sec'], playback)
                finally:
                    cap.release()

    def test_irregular_capture_holds_previous_image_and_includes_final_wait(self):
        info = self.begin(record_video=True)
        log = self.manager.session_log
        # Includes a slow gap, a burst of frames, and a final wait before stop.
        times = (0.5, 0.7, 0.92, 1.4, 1.41, 2.3, 3.0)
        for i, timestamp in enumerate(times):
            log.write_video_frame(np.full((24, 32, 3), i * 32, np.uint8), timestamp)
        summary = self.manager.finish_session([], status='STOPPED', end_elapsed_sec=5.5)
        self.assertEqual(summary['video_frames'], 150)
        self.assertEqual(summary['duration_sec'], 5.5)
        cap = cv2.VideoCapture(info['video_mp4'])
        frames = []
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(frame)
        finally:
            cap.release()
        self.assertEqual(len(frames), 150)
        # MP4 compression slightly changes pixel values. Check images as well as duration.
        self.assertAlmostEqual(float(frames[5].mean()), 0, delta=6)
        self.assertAlmostEqual(float(frames[20].mean()), 64, delta=6)
        self.assertAlmostEqual(float(frames[35].mean()), 128, delta=6)
        self.assertAlmostEqual(float(frames[-1].mean()), 192, delta=6)

    def test_video_rejects_invalid_or_backward_timestamps(self):
        self.begin(record_video=True)
        log = self.manager.session_log
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        for invalid in (-1, float('nan'), float('inf')):
            with self.subTest(timestamp=invalid), self.assertRaises(ValueError):
                log.write_video_frame(frame, invalid)
        log.write_video_frame(frame, 1.0)
        with self.assertRaises(ValueError):
            log.write_video_frame(frame, 0.9)
        with self.assertRaises(ValueError):
            self.manager.finish_session([], end_elapsed_sec=0.9)

    def test_calibration_uses_cali_name_and_zero_indices_are_allowed(self):
        info = self.manager.begin_session('biceps_curl', 'CALIBRATION', session=0,
                                         round_number=0, set_number=0, started_at=self.start)
        self.assertEqual(Path(info['frame_csv']).name, 'session0_cali.csv')
        self.manager.finish_session([])
        info = self.manager.begin_session('biceps_curl', 'TEST', started_at=self.start)
        self.assertEqual(Path(info['frame_csv']).name, 'session1_round1_set1.csv')

    def test_date_uses_korea_timezone(self):
        start_utc = datetime(2025, 12, 31, 16, 0, tzinfo=timezone.utc)
        info = self.manager.begin_session('biceps_curl', 'MAIN', started_at=start_utc)
        self.assertEqual(Path(info['directory']).parent.name, '260101')

    def test_legacy_profile_is_read_without_moving_or_overwriting_it(self):
        legacy = self.root / 'legacy.json'
        legacy.write_text(json.dumps({'custom_thresholds': {'biceps_curl': THRESHOLDS}}), encoding='utf-8')
        before = legacy.read_bytes()
        self.manager.legacy_json_path = legacy
        self.assertEqual(self.manager.get_custom_threshold('biceps_curl'), THRESHOLDS)
        changed = {side: {'start_val': 0.1, 'target_val': 0.5} for side in ('left', 'right')}
        self.manager.save_custom_threshold('biceps_curl', changed)
        self.assertEqual(self.manager.get_custom_threshold('biceps_curl'), changed)
        self.assertEqual(legacy.read_bytes(), before)
        self.assertTrue(self.manager.json_path.is_relative_to(self.root))

    def test_default_root_is_project_data_and_environment_can_override_it(self):
        with patch.dict(os.environ, {'ASAN_LOG_ROOT': str(self.root / 'custom')}):
            self.assertEqual(default_log_root(), self.root / 'custom')
        with patch.dict(os.environ, {'LOCALAPPDATA': str(self.root)}, clear=True):
            self.assertEqual(default_log_root(), Path(__file__).resolve().parent.parent / 'data')

    def test_invalid_ids_indices_and_metadata_are_rejected(self):
        for invalid in ('../escape', 'a/b', 'a\\b', '.', 'C:', ''):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                DataManager(invalid, log_root=self.root)
        for invalid in (-1, True, 1.5):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.begin(round_number=invalid)
        self.manager.patient_metadata['height_cm'] = float('nan')
        with self.assertRaises(ValueError):
            self.begin()

    def test_storage_failure_is_not_reported_as_success(self):
        path = self.root / 'not_a_directory'
        path.write_text('occupied', encoding='utf-8')
        with self.assertRaises(OSError):
            DataManager('0002', log_root=path)


if __name__ == '__main__':
    unittest.main()
