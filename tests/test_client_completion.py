import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock
from websockets.exceptions import ConnectionClosedOK

from network.client_progress import ClientProgress
from network.streamlit_client import StreamlitSessionClient
from tests.support import MockSessionServer, wait_for
from tests.test_skeleton_batch import load_model_free_module


def rep(rep_id, quality='PERFECT', side='left', session_id='session'):
    return dict(type='REP_COMPLETED', session_id=session_id, mode='MAIN',
                side=side, rep_id=rep_id, quality=quality, threshold_version=1)


class ClientProgressTests(unittest.TestCase):
    def test_exact_goal_does_not_finish_excess_does_after_delay(self):
        progress = ClientProgress(dict(mode='MAIN', target_score=40))
        for i in range(1, 41):
            self.assertTrue(progress.accept(rep(i), 0.))
        self.assertFalse(progress.finish_due(100.))
        progress.accept(rep(41, 'GOOD'), 100.)
        self.assertFalse(progress.finish_due(101.49))
        self.assertTrue(progress.finish_due(101.5))
        self.assertEqual(progress.scores['left'], 40.5)

    def test_duplicate_and_foreign_events_do_not_award_points(self):
        progress = ClientProgress(dict(mode='TEST'))
        progress.bind_session('session')
        self.assertFalse(progress.accept(rep(1, session_id='old'), 0))
        self.assertTrue(progress.accept(rep(1, 'GOOD'), 0))
        self.assertFalse(progress.accept(rep(1, 'GOOD'), 1))
        self.assertTrue(progress.accept(rep(1, 'BAD', 'right'), 2))
        self.assertEqual(progress.scores, {'left': .5, 'right': .25})

    def test_custom_client_weights_and_calibration_exclusion(self):
        cfg = dict(mode='MAIN', target_score=.5, finish_delay_seconds=0,
                   quality_points={'PERFECT': 2., 'GOOD': 1., 'BAD': 0.})
        progress = ClientProgress(cfg)
        progress.accept(rep(1, 'GOOD'), 1.)
        self.assertTrue(progress.finish_due(1.))
        calibration = ClientProgress({**cfg, 'mode': 'CALIBRATION'})
        self.assertFalse(calibration.accept(rep(1), 1.))
        self.assertFalse(calibration.finish_due(100.))

    def test_offline_goal_does_not_wait_for_wall_clock_cooldown(self):
        progress = ClientProgress(dict(mode='TEST', target_score=.5, video_path='input.mp4'))
        progress.accept(rep(1), 0.)
        self.assertTrue(progress.finish_due(0.))


class ClientCompletionTransportTests(unittest.TestCase):
    def setUp(self):
        self.server = MockSessionServer().start()
        self.client = StreamlitSessionClient(self.server.uri, dict(
            player_id='client_test', exercise_name='biceps_curl', mode='MAIN',
            target_score=1, finish_delay_seconds=0, output_mode='LIGHTWEIGHT'))

    def tearDown(self):
        self.client.stop()
        self.server.stop()

    def test_goal_stops_through_command_once_without_pose_packets(self):
        self.client.start()
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)
        start = next(p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION')
        self.assertNotIn('target_score', start)
        self.assertNotIn('target_reps', start)
        self.server.send(rep(1))
        self.server.send(rep(1))
        wait_for(lambda: self.client.snapshot()['progress']['scores']['left'] == 1)
        self.assertTrue(self.client.running)
        self.server.send(rep(2, 'GOOD'))
        wait_for(lambda: not self.client.running)
        stops = [p for p in self.server.packets if p['type'] == 'CMD_STOP_SESSION']
        self.assertEqual(len(stops), 1)
        self.assertEqual(stops[0]['reason'], 'TARGET_REACHED')
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        self.assertEqual(self.client.snapshot()['progress']['scores']['left'], 1.5)

    def test_manual_stop_waits_for_server_acknowledgement(self):
        self.client.start()
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)
        self.client.stop()
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        stops = [p for p in self.server.packets if p['type'] == 'CMD_STOP_SESSION']
        self.assertEqual(stops[0]['reason'], 'USER_STOP')
        self.assertTrue(any(p['type'] == 'SESSION_FINISHED' for p in self.client.drain_events()))


class ServerClientCompletionTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_model_free_module('network/socket_server.py')

    def setUp(self):
        self.server = self.module.ExerciseSocketServer.__new__(self.module.ExerciseSocketServer)
        self.ws = SimpleNamespace(send=AsyncMock())
        self.server._session_owner = self.ws
        self.server.controller = SimpleNamespace(mode='MAIN', data_manager=SimpleNamespace(
            session_log=SimpleNamespace(session_id='session', info={})), motion_engine=SimpleNamespace(
            exercise_name='test', mode='MAIN', threshold_version=3))
        self.server._save_session_files = Mock(return_value={'saved_reps_count': 4})
        self.server.stop_camera = Mock()

    async def test_stop_saves_once_and_duplicate_receives_same_ack(self):
        packet = dict(type='CMD_STOP_SESSION', session_id='session', reason='TARGET_REACHED', request_id='a')
        await self.server._finish_client_session(self.ws, packet)
        self.server._save_session_files.assert_called_once_with(status='COMPLETED')
        self.server.stop_camera.assert_called_once()
        self.assertIsNone(self.server.controller)
        await self.server._finish_client_session(self.ws, {**packet, 'request_id': 'b'})
        self.server._save_session_files.assert_called_once()
        response = json.loads(self.ws.send.call_args.args[0])
        self.assertEqual(response['request_id'], 'b')
        self.assertEqual(response['type'], 'SESSION_FINISHED')

    async def test_other_connection_or_session_cannot_stop_measurement(self):
        other = SimpleNamespace(send=AsyncMock())
        await self.server._finish_client_session(other, dict(type='CMD_STOP_SESSION'))
        self.server._save_session_files.assert_not_called()
        await self.server._finish_client_session(self.ws, dict(session_id='old'))
        self.server._save_session_files.assert_not_called()
        self.assertEqual(json.loads(self.ws.send.call_args.args[0])['code'], 'SESSION_MISMATCH')

    async def test_rep_event_contains_identity_quality_and_threshold_version(self):
        await self.server._send_rep_events(self.ws,
            dict(is_count_updated=True, rep_count=12, quality='GOOD', val=.7),
            dict(is_count_updated=False), timestamp=1.5)
        packet = json.loads(self.ws.send.call_args.args[0])
        self.assertEqual((packet['session_id'], packet['side'], packet['rep_id']), ('session', 'left', 12))
        self.assertEqual(packet['quality'], 'GOOD')
        self.assertEqual(packet['threshold_version'], 3)
        self.assertNotIn('score_delta', packet)

    async def test_client_game_result_is_saved_as_metadata(self):
        log = self.server.controller.data_manager.session_log
        result = {'scores': {'left': 40.5, 'right': 12.}, 'target_score': 40.}
        await self.server._finish_client_session(self.ws, dict(
            reason='TARGET_REACHED', client_result=result))
        self.assertEqual(log.info['client_result'], result)
        self.assertEqual(log.info['target_score'], 40.)

    async def test_production_command_loop_dispatches_stop_and_releases_owner(self):
        self.ws.remote_address = ('127.0.0.1', 1234)
        self.ws.recv = AsyncMock(side_effect=[json.dumps(dict(
            type='CMD_STOP_SESSION', session_id='session', reason='USER_STOP')),
            ConnectionClosedOK(None, None)])
        self.server.model_loader = SimpleNamespace(start=lambda: None, snapshot=lambda: dict(
            type='SERVER_STATUS', model_state='READY', phase='READY', ready=True))
        await self.server.handle_client(self.ws)
        self.server._save_session_files.assert_called_once_with(status='STOPPED')
        self.assertIsNone(self.server._session_owner)
        responses = [json.loads(call.args[0]) for call in self.ws.send.call_args_list]
        self.assertTrue(any(p['type'] == 'SESSION_FINISHED' for p in responses))
