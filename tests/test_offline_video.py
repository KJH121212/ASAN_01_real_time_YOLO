import json
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
from websockets.exceptions import ConnectionClosedOK

from tests.test_skeleton_batch import load_model_free_module
from core.data_manager import DataManager


def keypoints():
    return [{'id': i, 'x': 0.5, 'y': 0.5, 'score': 0.9} for i in range(17)]


class OfflineVideoTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        skeleton = load_model_free_module('core/skeleton_engine.py')
        cls.module = load_model_free_module('network/socket_server.py', {'core.skeleton_engine': skeleton})

    def setUp(self):
        self.cap = Mock()
        self.cap.isOpened.return_value = True
        self.cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 3
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        self.cap.read.side_effect = [(True, frame)] * 3 + [(False, None)]
        self.module.cv2 = SimpleNamespace(VideoCapture=Mock(return_value=self.cap),
                                           CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)
        self.temporary = tempfile.TemporaryDirectory()
        self.manager = None

        def create_manager(**kwargs):
            kwargs['log_root'] = self.temporary.name
            self.manager = DataManager(**kwargs)
            return self.manager

        self.module.DataManager = Mock(side_effect=create_manager)
        self.controller = None

        def create_controller(data_manager, motion_engine):
            self.controller = SimpleNamespace(
                data_manager=data_manager, motion_engine=motion_engine,
                timer_start=0.0, check_occlusion=lambda kpts: False,
                last_left_data={'val': 0.1}, last_right_data={'val': 0.2},
                process_main_frame=Mock(return_value=({'val': 0.1}, {'val': 0.2}, False)))
            return self.controller

        self.module.SessionController = create_controller
        self.server = self.module.ExerciseSocketServer.__new__(self.module.ExerciseSocketServer)
        self.server.controller = None
        self.server.camera = None
        self.server._active_log = None
        self.server.stop_camera = Mock()
        self.batch_sizes = []

        def extract_batch(frames):
            self.batch_sizes.append(len(frames))
            return [keypoints() for _ in frames]

        self.server.skeleton_engine = SimpleNamespace(extract_keypoints_batch=Mock(side_effect=extract_batch))
        self.server._session_owner = None
        self.server.model_loader = SimpleNamespace(
            start=lambda: None, engine=self.server.skeleton_engine,
            snapshot=lambda: {'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})
        self.server._save_session_files = Mock(return_value={'saved_frames': 3, 'saved_reps_count': 0})
        self.ws = SimpleNamespace(send=AsyncMock(), remote_address=('127.0.0.1', 1234))
        self.packet = dict(type='CMD_SET_SESSION', video_path='example.mp4',
                           exercise_name='biceps_curl', mode='MAIN', target_reps=10, batch_size=2)

    def tearDown(self):
        if self.manager and self.manager.session_log:
            self.manager.finish_session([], status='TEST')
        self.temporary.cleanup()

    async def test_full_and_partial_batches_are_both_analyzed_and_completed(self):
        await self.server.process_offline_video(self.ws, self.packet)
        self.assertEqual(self.batch_sizes, [2, 1])
        self.cap.release.assert_called_once()
        self.assertEqual(self.controller.process_main_frame.call_count, 3)
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])
        result = json.loads(self.ws.send.call_args_list[-1].args[0])
        self.assertEqual(result['type'], 'VIDEO_ANALYSIS_COMPLETED')
        self.assertEqual(result['total_frames_analyzed'], 3)
        self.assertIsNone(self.server.controller)

    async def test_client_stop_interrupts_offline_extraction_and_acknowledges_save(self):
        self.ws.recv = AsyncMock(return_value=json.dumps({
            'type': 'CMD_STOP_SESSION', 'reason': 'USER_STOP', 'request_id': 'stop-video'}))
        await self.server.process_offline_video(self.ws, self.packet)
        result = json.loads(self.ws.send.call_args_list[-1].args[0])
        self.assertEqual(result['type'], 'SESSION_FINISHED')
        self.assertEqual(result['request_id'], 'stop-video')
        self.assertEqual(result['status'], 'STOPPED')
        self.assertEqual(self.batch_sizes, [2])
        self.controller.process_main_frame.assert_not_called()
        self.cap.release.assert_called_once()

    async def test_offline_main_ignores_legacy_controller_finish_flag(self):
        original = self.module.SessionController
        def controller_with_legacy_flag(**kwargs):
            result = original(**kwargs)
            result.process_main_frame.return_value = ({'val': .1}, {'val': .2}, True)
            return result
        self.module.SessionController = controller_with_legacy_flag
        await self.server.process_offline_video(self.ws, self.packet)
        self.assertEqual(self.controller.process_main_frame.call_count, 3)

    async def test_missing_person_does_not_shift_following_frame_timestamp(self):
        self.server.skeleton_engine.extract_keypoints_batch.side_effect = [[keypoints(), None], [keypoints()]]
        await self.server.process_offline_video(self.ws, self.packet)
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])
        self.assertTrue(np.isnan(self.manager.buf_raw_keypoints[1][:, :2]).all())

    async def test_capture_released_even_when_batch_inference_fails(self):
        self.server.skeleton_engine.extract_keypoints_batch.side_effect = RuntimeError('inference failed')
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            await self.server.process_offline_video(self.ws, self.packet)
        self.cap.release.assert_called_once()
        self.server._save_session_files.assert_not_called()

    async def test_invalid_batch_size_is_reported_before_opening_video(self):
        await self.server.process_offline_video(self.ws, {**self.packet, 'batch_size': 0})
        self.module.cv2.VideoCapture.assert_not_called()
        self.assertEqual(json.loads(self.ws.send.call_args.args[0])['code'], 'INVALID_BATCH_SIZE')

    async def test_unusable_fps_falls_back_to_30(self):
        self.cap.get.side_effect = lambda prop: float('nan') if prop == 5 else 3
        await self.server.process_offline_video(self.ws, self.packet)
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])

    async def test_actual_csv_npz_mp4_and_summary_use_requested_session_name(self):
        from pathlib import Path
        import csv
        self.server._save_session_files = self.module.ExerciseSocketServer._save_session_files.__get__(self.server)
        await self.server.process_offline_video(self.ws, {**self.packet,
            'session': 2, 'round': 3, 'set': 4, 'record_video': True})
        result = json.loads(self.ws.send.call_args_list[-1].args[0])
        summary = result['summary']
        self.assertEqual(Path(summary['frame_csv']).stem, 'session2_round3_set4')
        for key in ('frame_csv', 'skeleton_npz', 'video_mp4', 'gamedata_csv'):
            self.assertTrue(Path(summary[key]).is_relative_to(Path(self.temporary.name)))
            self.assertTrue(Path(summary[key]).is_file())
        with Path(summary['frame_csv']).open(newline='', encoding='utf-8-sig') as file:
            self.assertEqual(len(list(csv.DictReader(file))), 3)
        with np.load(summary['skeleton_npz']) as data:
            self.assertEqual(data['keypoints'].shape, (3, 17, 3))
            np.testing.assert_array_equal(data['video_frame_idx'], [0, 1, 2])
        self.assertEqual(summary['video_frames'], 3)

    async def test_dispatch_reports_offline_failure_instead_of_silent_disconnect(self):
        self.server.process_offline_video = AsyncMock(side_effect=RuntimeError('batch failed'))
        self.ws.recv = AsyncMock(side_effect=[json.dumps(self.packet),
                                             ConnectionClosedOK(None, None)])
        with patch.object(self.module.os.path, 'isfile', return_value=True), \
                patch.object(self.module.traceback, 'print_exc'):
            await self.server.handle_client(self.ws)
        self.server.process_offline_video.assert_awaited_once()
        response = json.loads(self.ws.send.call_args.args[0])
        self.assertEqual(response['type'], 'ERROR')
        self.assertEqual(response['code'], 'VIDEO_ANALYSIS_FAILED')
        self.server.stop_camera.assert_called()

    async def test_offline_recording_keeps_source_fps_and_not_batch_wall_time(self):
        import cv2
        self.cap.get.side_effect = lambda prop: 6.0 if prop == 5 else 3
        self.server._save_session_files = self.module.ExerciseSocketServer._save_session_files.__get__(self.server)
        await self.server.process_offline_video(self.ws, {**self.packet, 'record_video': True})
        summary = json.loads(self.ws.send.call_args_list[-1].args[0])['summary']
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 6, 2 / 6])
        cap = cv2.VideoCapture(summary['video_mp4'])
        try:
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 3)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 6)
            self.assertAlmostEqual(summary['video_duration_sec'], 0.5)
        finally:
            cap.release()


if __name__ == '__main__':
    unittest.main()
