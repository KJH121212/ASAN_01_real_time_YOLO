import unittest

from network.streamlit_client import StreamlitSessionClient
from tests.support import MockSessionServer, wait_for


class StreamlitClientTests(unittest.TestCase):
    def setUp(self):
        self.server = MockSessionServer().start()
        self.config = dict(player_id='ui_test', patient_name='Test', exercise_name='biceps_curl',
                           mode='MAIN', target_reps=40, output_mode='LIGHTWEIGHT', video_path=None)
        self.client = StreamlitSessionClient(self.server.uri, self.config)

    def tearDown(self):
        self.client.stop()
        self.server.stop()

    def start_client(self):
        self.client.start()
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)

    def test_edits_and_queries_use_one_connection_without_resetting_reps(self):
        self.start_client()
        request = self.client.threshold_command({'left': {'target_val': 0.5}})
        wait_for(lambda: self.client.snapshot()['threshold_status']['threshold_version'] == 1)
        self.client.threshold_command()
        wait_for(lambda: len(self.server.packets) >= 4)
        self.assertEqual(self.server.connections, 1)
        self.assertEqual(self.server.starts, 1)
        self.assertEqual(self.client.snapshot()['pose']['left']['rep_count'], 2)
        self.assertTrue(any(event.get('request_id') == request for event in self.client.drain_events()))

    def test_rejected_edit_does_not_disconnect_or_change_applied_values(self):
        self.start_client()
        self.client.drain_events()
        self.client.threshold_command({'left': {'target_val': 0.1}})
        wait_for(lambda: len(self.server.packets) >= 3)
        events = []

        def received_error():
            events.extend(self.client.drain_events())
            return any(event.get('type') == 'ERROR' for event in events)

        wait_for(received_error)
        self.assertEqual(self.client.snapshot()['threshold_status']['thresholds']['left']['target_val'], 0.9)
        self.assertTrue(self.client.running)
        self.client.threshold_command({'left': {'target_val': 0.5}})
        wait_for(lambda: self.client.snapshot()['threshold_status']['threshold_version'] == 1)

    def test_snapshot_and_configuration_are_copies(self):
        self.config['exercise_name'] = 'other_game'
        self.start_client()
        snapshot = self.client.snapshot()
        snapshot['threshold_status']['thresholds']['left']['target_val'] = 999
        self.assertEqual(self.client.config['exercise_name'], 'biceps_curl')
        self.assertEqual(self.client.snapshot()['threshold_status']['thresholds']['left']['target_val'], 0.9)

    def test_stop_closes_socket_and_prevents_new_commands(self):
        self.start_client()
        self.client.stop()
        self.assertFalse(self.client.running)
        self.assertTrue(self.server.disconnected.wait(2))
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        with self.assertRaises(RuntimeError):
            self.client.threshold_command()

    def test_completion_is_reported_and_worker_exits(self):
        self.start_client()
        self.server.send({'type': 'SESSION_FINISHED', 'summary': {'saved_reps_count': 2}})
        wait_for(lambda: not self.client.running)
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        self.assertTrue(any(event['type'] == 'SESSION_FINISHED' for event in self.client.drain_events()))

    def test_calibration_disallows_edits(self):
        self.client.config['mode'] = 'CALIBRATION'
        self.start_client()
        self.assertFalse(self.client.can_edit_thresholds)
        with self.assertRaises(RuntimeError):
            self.client.threshold_command({'left': {'target_val': 0.5}})

    def test_model_loading_state_keeps_connection_until_session_is_ready(self):
        self.server.model_ready.clear()
        self.client.start()
        wait_for(lambda: self.client.snapshot()['server_status'] is not None)
        snapshot = self.client.snapshot()
        self.assertEqual(snapshot['status'], 'PREPARING')
        self.assertIsNone(snapshot['threshold_status'])
        self.assertFalse(self.client.can_edit_thresholds)
        self.server.model_ready.set()
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)
        self.assertTrue(self.client.can_edit_thresholds)
        self.assertEqual(self.server.starts, 1)
        self.assertEqual(self.server.connections, 1)

    def test_offline_does_not_query_or_edit_during_analysis(self):
        self.client.config['video_path'] = 'example.mp4'
        self.start_client()
        self.assertEqual([p['type'] for p in self.server.packets], ['CMD_SET_SESSION'])
        self.assertFalse(self.client.can_edit_thresholds)

    def test_recording_runs_in_worker_and_closes_with_session(self):
        class Recorder:
            def __init__(self):
                self.started = False
                self.stopped = False
                self.poses = []

            def start(self):
                self.started = True

            def record_pose(self, packet):
                self.poses.append(packet)

            def stop(self):
                self.stopped = True

        recorder = Recorder()
        self.client.recorder = recorder
        self.start_client()
        wait_for(lambda: bool(recorder.poses))
        self.assertTrue(recorder.started)
        self.client.stop()
        self.assertTrue(recorder.stopped)


if __name__ == '__main__':
    unittest.main()
