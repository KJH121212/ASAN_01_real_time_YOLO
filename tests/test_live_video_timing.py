import asyncio
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import cv2
import numpy as np
from websockets.exceptions import ConnectionClosedOK

from core.motion_engine import MotionEngine
from network.socket_server import ExerciseSocketServer, ThreadedWebcam
from network.session_commands import DEFAULT_PREPARATION_SECONDS, get_preparation_seconds


class CaptureTimestampTests(unittest.TestCase):
    def test_capture_time_is_attached_at_read_and_remains_stable_for_consumers(self):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        cap = Mock()
        cap.read.return_value = (True, frame)
        with patch('network.socket_server.cv2.VideoCapture', return_value=cap), \
                patch('network.socket_server.time.perf_counter', return_value=10.25):
            camera = ThreadedWebcam()
        try:
            first = camera.read_sample()
            second = camera.read_sample()
            self.assertEqual(first[2:], (10.25, 0))
            self.assertEqual(second[2:], first[2:])
            first[1][:] = 255
            self.assertEqual(int(second[1].max()), 0)

            def capture_next():
                camera.started = False  # Run exactly one capture iteration.
                return True, np.full_like(frame, 128)

            camera.started = True
            cap.read.side_effect = capture_next
            with patch('network.socket_server.time.perf_counter', return_value=10.45):
                camera.update()
            latest = camera.read_sample()
            self.assertEqual(latest[2:], (10.45, 1))
            self.assertEqual(int(latest[1].max()), 128)
        finally:
            camera.stop()


class LiveVideoTimingTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_initialization_uses_default_delay_and_zero_disables_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, delay in (('MAIN', None), ('CALIBRATION', None), ('TEST', 0)):
                with self.subTest(mode=mode, delay=delay):
                    server = ExerciseSocketServer(engine_factory=lambda report: object())
                    server.start_camera = Mock()
                    packet = {'mode': mode, 'log_root': directory}
                    if delay is not None:
                        packet['preparation_seconds'] = delay
                    with patch('network.socket_server.time.perf_counter', return_value=100.0):
                        server.init_live_session(packet)
                        try:
                            expected = DEFAULT_PREPARATION_SECONDS if delay is None else delay
                            self.assertEqual(server._session_started, 100.0 + expected)
                            self.assertEqual(server._measurement_ready_at, 100.0 + expected if expected else None)
                            server.start_camera.assert_called_once()
                            self.assertEqual(server.controller.motion_engine.fsm_left.rep_count, 0)
                        finally:
                            server._save_session_files(status='STOPPED')

    async def test_capture_timing_and_disconnect_reach_actual_mp4_and_npz(self):
        await self.run_recording(preparation_seconds=0)

    async def test_preparation_frames_never_reach_inference_logs_or_video(self):
        await self.run_recording(preparation_seconds=5)

    async def run_recording(self, preparation_seconds):
        with tempfile.TemporaryDirectory() as directory:
            server = ExerciseSocketServer(engine_factory=lambda report: object())
            manager = server._prepare_data_manager({
                'patient_id': '0001', 'mode': 'MAIN', 'record_video': True,
                'log_root': directory}, live_timing=True)
            server._session_started = 100.0 + preparation_seconds
            if preparation_seconds:
                server._measurement_ready_at = server._session_started
            config = {'eval_config': {'metric_type': 'RELATIVE_Y', 'motion_direction': 'INCREASING'},
                      'default_thresholds': {side: {'start_val': 0.1, 'target_val': 0.9}
                                             for side in ('left', 'right')}}
            engine = MotionEngine(config, mode='MAIN')
            engine.exercise_name = 'biceps_curl'
            side = {'val': 0.2, 'rep_count': 0}
            server.controller = SimpleNamespace(
                data_manager=manager, motion_engine=engine, calib_step='',
                last_left_data=side, last_right_data=side,
                process_main_frame=Mock(return_value=(side, side, False)))
            server.filter_engine = SimpleNamespace(update=lambda points, timestamp: None)
            server.skeleton_engine = SimpleNamespace(extract_keypoints=Mock(return_value=None))
            inference = server.skeleton_engine.extract_keypoints
            server.model_loader = SimpleNamespace(start=lambda: None, snapshot=lambda: {
                'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            origin = server._session_started
            server.camera = SimpleNamespace(read_sample=Mock(side_effect=[
                (True, frame, origin + 0.5, 1), (True, frame, origin + 0.5, 1),
                (True, frame, origin + 0.7, 2), (True, frame, origin + 1.0, 3)]), stop=Mock())
            camera = server.camera
            server.output_mode = 'LIGHTWEIGHT'
            clock = {'now': 100.0}
            steps = iter(([100.0, 104.0] if preparation_seconds else []) +
                         [origin + 0.5, origin + 0.6, origin + 0.7, origin + 1.0, origin + 2.0])

            async def receive():
                clock['now'] = next(steps)
                if clock['now'] == origin + 2.0:
                    raise ConnectionClosedOK(None, None)
                raise asyncio.TimeoutError

            websocket = SimpleNamespace(
                remote_address=('test', 0), send=AsyncMock(),
                recv=receive)
            server._session_owner = websocket
            try:
                with patch('network.socket_server.time.perf_counter', side_effect=lambda: clock['now']):
                    await server.handle_client(websocket)
                # Repeated polling of the same captured image must not infer/log twice.
                self.assertEqual(inference.call_count, 3)
                self.assertEqual(camera.read_sample.call_count, 4)
                packets = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
                countdowns = [packet['remaining_seconds'] for packet in packets
                              if packet['type'] == 'SESSION_PREPARING']
                self.assertEqual(countdowns, [5, 1] if preparation_seconds else [])
                with np.load(manager.session_log.info['skeleton_npz']) as data:
                    np.testing.assert_allclose(data['timestamps'], [0.5, 0.7, 1.0])
                    np.testing.assert_array_equal(data['video_frame_idx'], [0, 6, 15])
                manifest = json.loads(Path(manager.session_log.info['manifest_json']).read_text(encoding='utf-8'))
                self.assertEqual(manifest['status'], 'STOPPED')
                self.assertAlmostEqual(manifest['video_duration_sec'], 1.5)
                cap = cv2.VideoCapture(manifest['video_mp4'])
                try:
                    self.assertTrue(cap.isOpened())
                    self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 1.5)
                finally:
                    cap.release()
            finally:
                manager.finish_session([], status='TEST')

    async def test_stop_during_preparation_saves_empty_logs_with_zero_duration(self):
        with tempfile.TemporaryDirectory() as directory:
            server = ExerciseSocketServer(engine_factory=lambda report: object())
            manager = server._prepare_data_manager({
                'mode': 'MAIN', 'log_root': directory}, live_timing=True)
            server._session_started = 105.0
            server._measurement_ready_at = 105.0
            engine = SimpleNamespace(fsm_left=SimpleNamespace(completed_reps_history=[]),
                                     fsm_right=SimpleNamespace(completed_reps_history=[]))
            server.controller = SimpleNamespace(data_manager=manager, motion_engine=engine)
            server.camera = SimpleNamespace(read_sample=Mock(), stop=Mock())
            camera = server.camera
            server.model_loader = SimpleNamespace(start=lambda: None, snapshot=lambda: {
                'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})
            websocket = SimpleNamespace(remote_address=('test', 0), send=AsyncMock(),
                recv=AsyncMock(side_effect=[asyncio.TimeoutError, ConnectionClosedOK(None, None)]))
            server._session_owner = websocket
            with patch('network.socket_server.time.perf_counter', return_value=100.0):
                await server.handle_client(websocket)
            camera.read_sample.assert_not_called()
            self.assertEqual(manager.session_log._summary['saved_frames'], 0)
            self.assertEqual(manager.session_log._summary['duration_sec'], 0)


class PreparationSettingTests(unittest.TestCase):
    def test_default_and_zero_disable(self):
        self.assertEqual(get_preparation_seconds({}), DEFAULT_PREPARATION_SECONDS)
        self.assertEqual(get_preparation_seconds({'preparation_seconds': 0}), 0)
        self.assertEqual(get_preparation_seconds({'preparation_seconds': 2.5}), 2.5)

    def test_invalid_delay_is_rejected(self):
        for value in (-1, True, '5', None, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                get_preparation_seconds({'preparation_seconds': value})


if __name__ == '__main__':
    unittest.main()
