from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from tests.support import MockSessionServer, wait_for


APP_PATH = str(Path(__file__).resolve().parents[1] / 'app.py')


def widget(elements, label):
    return next(element for element in elements if element.label == label)


class StreamlitAppTests(unittest.TestCase):
    def setUp(self):
        self.server = MockSessionServer().start()
        self.app = AppTest.from_file(APP_PATH, default_timeout=8).run()

    def tearDown(self):
        if 'session_client' in self.app.session_state:
            self.app.session_state['session_client'].stop()
        self.server.stop()

    def assert_no_ui_errors(self):
        self.assertEqual([error.message for error in self.app.exception], [])

    def start_session(self, mode='TEST', visual=False, wait_until_ready=True):
        widget(self.app.text_input, 'Socket Server').set_value(self.server.uri)
        pipeline = ('모드 1: 기존 시각화 (Webcam + Image)' if visual
                    else '모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)')
        widget(self.app.radio, '검증 방식 선택').set_value(pipeline)
        widget(self.app.radio, 'Execution Mode').set_value(mode)
        widget(self.app.button, '세션 시작 (Start Session)').click().run()
        self.assert_no_ui_errors()
        client = self.app.session_state['session_client']
        if not wait_until_ready:
            return client
        wait_for(lambda: client.snapshot()['threshold_status'] is not None)
        self.app.run()
        self.assert_no_ui_errors()
        return client

    def test_camera_selection_is_sent_to_server_and_locked_during_session(self):
        widget(self.app.selectbox, '카메라 선택 (Camera)').set_value(2).run()
        self.start_session()
        packet = next(p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION')
        self.assertEqual(packet['camera_index'], 2)
        self.assertTrue(widget(self.app.selectbox, '카메라 선택 (Camera)').disabled)

    def test_custom_camera_index_is_sent_in_visual_mode(self):
        widget(self.app.selectbox, '카메라 선택 (Camera)').set_value('manual').run()
        widget(self.app.number_input, '카메라 번호 (Camera Index)').set_value(8).run()
        self.start_session(visual=True)
        packet = next(p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION')
        self.assertEqual(packet['camera_index'], 8)
        self.assertTrue(widget(self.app.number_input, '카메라 번호 (Camera Index)').disabled)

    def test_offline_mode_hides_camera_selection(self):
        widget(self.app.radio, '검증 방식 선택').set_value(
            '모드 3: 오프라인 비디오 분석 (Video Batch 검증)').run()
        self.assert_no_ui_errors()
        self.assertNotIn('카메라 선택 (Camera)', [item.label for item in self.app.selectbox])

    def test_initial_page_does_not_open_camera(self):
        with patch('cv2.VideoCapture', side_effect=AssertionError('UI must not own the camera')):
            self.app.run()
        self.assert_no_ui_errors()
        self.assertIn('세션 시작 (Start Session)', [button.label for button in self.app.button])

    def test_fractional_score_target_and_accumulated_points_are_displayed(self):
        widget(self.app.number_input, '목표 점수 (Target Score)').set_value(1.25)
        client = self.start_session()
        packet = next(packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION')
        self.assertNotIn('target_score', packet)
        self.assertEqual(client.config['target_score'], 1.25)
        pose = dict(client.snapshot()['pose'])
        pose['left'] = {**pose['left'], 'total_score': 0.75, 'rep_count': 2}
        self.server.send(pose)
        for rep_id, quality in ((3, 'GOOD'), (4, 'BAD')):
            self.server.send({'type': 'REP_COMPLETED', 'session_id': 'test', 'side': 'left',
                              'rep_id': rep_id, 'quality': quality, 'mode': 'TEST'})
        wait_for(lambda: client.snapshot()['pose']['left'].get('total_score') == 0.75)
        self.app.run()
        self.assertEqual(widget(self.app.metric, '왼쪽 (Left)').value, '0.75 / 1.25점')
        self.assert_no_ui_errors()

    def test_preparation_can_be_disabled_and_countdown_is_shown(self):
        widget(self.app.number_input, '측정 전 준비 시간 (초)').set_value(0.0)
        client = self.start_session()
        packet = next(packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION')
        self.assertEqual(packet['preparation_seconds'], 0.0)
        self.server.send({'type': 'SESSION_PREPARING', 'remaining_seconds': 3,
                          'message': '기본 자세를 취해 주세요. 3초 후 측정을 시작합니다.'})
        wait_for(lambda: client.snapshot()['status'] == 'PREPARING')
        self.app.run()
        self.assertTrue(any('3초 후 측정을 시작' in message.value for message in self.app.info))
        self.server.send({'type': 'SESSION_MEASUREMENT_STARTED'})
        wait_for(lambda: client.snapshot()['status'] == 'RUNNING')
        self.app.run()
        self.assertIsNone(self.app.session_state['session_notice'])
        self.assert_no_ui_errors()

    def test_apply_rom_keeps_connection_counts_and_draft_values(self):
        client = self.start_session()
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.5)
        widget(self.app.number_input, '오른쪽 목표값').set_value(0.5)
        widget(self.app.button, '가동 범위 적용').click().run()
        wait_for(lambda: client.snapshot()['threshold_status']['threshold_version'] == 1)
        self.app.run()
        self.assert_no_ui_errors()
        self.assertEqual(self.server.connections, 1)
        self.assertEqual(self.server.starts, 1)
        self.assertIs(self.app.session_state['session_client'], client)
        self.assertEqual(client.snapshot()['pose']['left']['rep_count'], 2)
        self.assertEqual(widget(self.app.number_input, '왼쪽 목표값').value, 0.5)
        self.assertTrue(any('적용 완료' in message.value for message in self.app.success))
        self.assertIsNone(self.app.session_state['pending_rom_request'])

    def test_editing_form_does_not_send_until_apply_and_can_reload_server_values(self):
        self.start_session()
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.6).run()
        self.assertEqual([p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD'], [])
        widget(self.app.button, '현재 적용값 불러오기').click().run()
        self.assert_no_ui_errors()
        self.assertEqual(widget(self.app.number_input, '왼쪽 목표값').value, 0.9)
        self.assertEqual(self.server.starts, 1)

    def test_left_only_update_and_invalid_input_recovery(self):
        client = self.start_session()
        widget(self.app.radio, '수정 대상').set_value('left').run()
        self.assertTrue(widget(self.app.number_input, '오른쪽 목표값').disabled)
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.1)
        widget(self.app.button, '가동 범위 적용').click().run()
        wait_for(lambda: len([p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD']) == 1)
        wait_for(lambda: client.snapshot()['packet_count'] >= 5)
        self.app.run()
        self.assert_no_ui_errors()
        self.assertTrue(any('requires' in error.value for error in self.app.error))
        self.assertEqual(client.snapshot()['threshold_status']['threshold_version'], 0)
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.5)
        widget(self.app.button, '가동 범위 적용').click().run()
        wait_for(lambda: client.snapshot()['threshold_status']['threshold_version'] == 1)
        self.app.run()
        updates = [p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD']
        self.assertEqual(set(updates[-1]['thresholds']), {'left'})
        self.assertEqual(client.snapshot()['threshold_status']['thresholds']['right']['target_val'], 0.9)
        self.assertEqual(self.server.connections, 1)

    def test_calibration_disables_rom_updates(self):
        self.start_session('CALIBRATION')
        self.assertTrue(widget(self.app.button, '가동 범위 적용').disabled)

    def test_log_settings_are_sent_and_actual_server_path_is_displayed(self):
        with tempfile.TemporaryDirectory() as directory:
            widget(self.app.text_input, '로그 저장 폴더 (서버 기준)').set_value(directory)
            widget(self.app.number_input, 'Session').set_value(2)
            widget(self.app.number_input, 'Round').set_value(3)
            widget(self.app.number_input, 'Set').set_value(4)
            self.app.toggle[0].set_value(True)
            client = self.start_session()
            packet = next(packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION')
            self.assertEqual(packet['log_root'], directory)
            self.assertEqual((packet['session'], packet['round'], packet['set']), (2, 3, 4))
            self.assertTrue(packet['record_video'])
            self.assertFalse(packet['cal'])
            self.assertEqual(packet['preparation_seconds'], 5.0)
            self.assertNotIn('target_score', packet)
            self.assertEqual(client.config['target_score'], 40.0)
            self.assertTrue(client.snapshot()['log_info']['video_mp4'].endswith('session2_round3_set4.mp4'))
            self.assertTrue(any(directory in code.value for code in self.app.code))

    def test_visual_mode_renders_while_rom_controls_remain_available(self):
        client = self.start_session(visual=True)
        self.assertEqual(client.config['output_mode'], 'VISUAL')
        self.assertFalse(widget(self.app.button, '가동 범위 적용').disabled)

    def test_model_loading_is_visible_then_session_starts_without_reconnecting(self):
        self.server.model_ready.clear()
        client = self.start_session(wait_until_ready=False)
        wait_for(lambda: client.snapshot()['server_status'] is not None)
        self.app.run()
        self.assert_no_ui_errors()
        self.assertTrue(any('모델을 불러오는 중' in info.value for info in self.app.info))
        self.assertIsNone(client.snapshot()['threshold_status'])
        self.server.model_ready.set()
        wait_for(lambda: client.snapshot()['threshold_status'] is not None)
        self.app.run()
        self.assert_no_ui_errors()
        self.assertFalse(widget(self.app.button, '가동 범위 적용').disabled)
        self.assertEqual(self.server.connections, 1)

    def test_stop_allows_starting_new_session(self):
        client = self.start_session()
        widget(self.app.button, '세션 중지 (Stop Session)').click().run()
        self.assert_no_ui_errors()
        self.assertFalse(client.running)
        self.assertTrue(self.server.disconnected.wait(2))
        widget(self.app.button, '세션 시작 (Start Session)').click().run()
        self.assert_no_ui_errors()
        next_client = self.app.session_state['session_client']
        wait_for(lambda: next_client.snapshot()['threshold_status'] is not None)
        self.app.run()
        self.assertIsNot(client, next_client)
        self.assertEqual(self.server.starts, 2)


if __name__ == '__main__':
    unittest.main()
