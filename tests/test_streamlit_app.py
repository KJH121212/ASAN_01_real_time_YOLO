# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# `types`에서 `SimpleNamespace` 기능을 가져옵니다.
from types import SimpleNamespace
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# `unittest.mock`에서 `patch`, `Mock` 기능을 가져옵니다.
from unittest.mock import patch, Mock

# `streamlit.testing.v1`에서 `AppTest` 기능을 가져옵니다.
from streamlit.testing.v1 import AppTest

# `tests.support`에서 `MockSessionServer`, `wait_for` 기능을 가져옵니다.
from tests.support import MockSessionServer, wait_for


# app 경로에 `str(Path(__file__).resolve().parents[1] / 'app.py')` 호출 결과를 저장합니다. 대상: `APP_PATH`.
APP_PATH = str(Path(__file__).resolve().parents[1] / 'app.py')


# `widget` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def widget(elements, label):
    # 현재 함수 처리를 끝내고 `next((element for element in elements if element.label == label))` 호출 결과를 호출한 쪽으로 반환합니다.
    return next(element for element in elements if element.label == label)


# `StreamlitAppTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class StreamlitAppTests(unittest.TestCase):
    # 회귀 테스트: `test_overlay_button_starts_background_job_and_displays_progress` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_overlay_button_starts_background_job_and_displays_progress(self):
        # 처리 단계에 `dict(running=True, total=2, completed=0, current='sample.mp4', phase='RENDERING', frame=4, frames=12, results=[], output='data/_overlays/test')` 호출 결과를 저장합니다. 대상: `state`.
        state = dict(running=True, total=2, completed=0, current='sample.mp4', phase='RENDERING',
                     # 구성 항목: OpenCV BGR 영상 프레임(`frame`)에 4; 영상 프레임 묶음(`frames`)에 12; `results` 값(`results`)에 순서·구성을 지정한 `[]`; 생성할 결과 데이터 또는 경로(`output`)에 'data/_overlays/test'을 지정합니다.
                     frame=4, frames=12, results=[], output='data/_overlays/test')
        # `job` 값에 `SimpleNamespace(start=Mock(), snapshot=lambda : state, cancel=threading.Event())` 호출 결과를 저장합니다. 대상: `job`.
        job = SimpleNamespace(start=Mock(), snapshot=lambda: state, cancel=threading.Event())
        # `patch('core.video_overlay.VideoOverlayJob', return_value=job)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch('core.video_overlay.VideoOverlayJob', return_value=job) as factory:
            # `widget(self.app.button, 'data 영상 전체 오버레이 생성').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
            widget(self.app.button, 'data 영상 전체 오버레이 생성').click().run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `job.start.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        job.start.assert_called_once()
        # 검증문 `self.assertTrue(factory.call_args.kwargs['analyze_missing'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(factory.call_args.kwargs['analyze_missing'])
        # 검증문 `self.assertTrue(widget(self.app.button, '세션 시작 (Start Session)').disabled)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(widget(self.app.button, '세션 시작 (Start Session)').disabled)
        # 검증문 `self.assertTrue(self.app.get('progress'))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.app.get('progress'))

    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 서버에 `MockSessionServer().start()` 호출 결과를 저장합니다. 대상: `self.server`.
        self.server = MockSessionServer().start()
        # `self.app` 값에 `AppTest.from_file(APP_PATH, default_timeout=8).run()` 호출 결과를 저장합니다. 대상: `self.app`.
        self.app = AppTest.from_file(APP_PATH, default_timeout=8).run()

    # 각 테스트가 끝나면 패치·임시 자원을 해제하여 다음 테스트에 영향을 남기지 않습니다.
    def tearDown(self):
        # 조건 `'session_client' in self.app.session_state`을 만족하는 경우에만 아래 분기를 실행합니다.
        if 'session_client' in self.app.session_state:
            # `self.app.session_state['session_client'].stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.app.session_state['session_client'].stop()
        # `self.server.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.server.stop()

    # 결과 검증 no ui 오류 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def assert_no_ui_errors(self):
        # 검증문 `self.assertEqual([error.message for error in self.app.exception], [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual([error.message for error in self.app.exception], [])

    # 시작 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start_session(self, mode='TEST', visual=False, wait_until_ready=True):
        # `widget(self.app.text_input, 'Socket Server').set_value(self.server.uri)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.text_input, 'Socket Server').set_value(self.server.uri)
        # `pipeline` 값에 '모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)'을 저장합니다. 대상: `pipeline`.
        pipeline = '모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)'
        # `widget(self.app.radio, '검증 방식 선택').set_value(pipeline)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.radio, '검증 방식 선택').set_value(pipeline)
        # `widget(self.app.radio, 'Execution Mode').set_value(mode)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.radio, 'Execution Mode').set_value(mode)
        # `widget(self.app.button, '세션 시작 (Start Session)').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.button, '세션 시작 (Start Session)').click().run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 클라이언트에 계산식 `self.app.session_state['session_client']`의 결과를 저장합니다. 대상: `client`.
        client = self.app.session_state['session_client']
        # 조건 `not wait_until_ready`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not wait_until_ready:
            # 현재 함수 처리를 끝내고 클라이언트(`client`)을 호출한 쪽으로 반환합니다.
            return client
        # `wait_for(lambda : client.snapshot()['threshold_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['threshold_status'] is not None)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 현재 함수 처리를 끝내고 클라이언트(`client`)을 호출한 쪽으로 반환합니다.
        return client

    # 회귀 테스트: `test_camera_selection_is_sent_to_server_and_locked_during_session` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_camera_selection_is_sent_to_server_and_locked_during_session(self):
        # `widget(self.app.selectbox, '카메라 선택 (Camera)').set_value(2).run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.selectbox, '카메라 선택 (Camera)').set_value(2).run()
        # `self.start_session()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_session()
        # 수신 또는 전송할 JSON 메시지 사전에 `next((p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION'))` 호출 결과를 저장합니다. 대상: `packet`.
        packet = next(p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION')
        # 검증문 `self.assertEqual(packet['camera_index'], 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(packet['camera_index'], 2)
        # 검증문 `self.assertTrue(widget(self.app.selectbox, '카메라 선택 (Camera)').disabled)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(widget(self.app.selectbox, '카메라 선택 (Camera)').disabled)

    # 회귀 테스트: `test_custom_camera_index_is_sent_in_visual_mode` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_custom_camera_index_is_sent_in_visual_mode(self):
        # `widget(self.app.selectbox, '카메라 선택 (Camera)').set_value('manual').run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.selectbox, '카메라 선택 (Camera)').set_value('manual').run()
        # `widget(self.app.number_input, '카메라 번호 (Camera Index)').set_value(8).run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '카메라 번호 (Camera Index)').set_value(8).run()
        # `self.start_session(visual=True)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_session(visual=True)
        # 수신 또는 전송할 JSON 메시지 사전에 `next((p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION'))` 호출 결과를 저장합니다. 대상: `packet`.
        packet = next(p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION')
        # 검증문 `self.assertEqual(packet['camera_index'], 8)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(packet['camera_index'], 8)
        # 검증문 `self.assertTrue(widget(self.app.number_input, '카메라 번호 (Camera Index)').disabled)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(widget(self.app.number_input, '카메라 번호 (Camera Index)').disabled)

    # 회귀 테스트: `test_offline_mode_hides_camera_selection` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_offline_mode_hides_camera_selection(self):
        # `widget(self.app.radio, '검증 방식 선택').set_value('모드 3: 오프라인 비디오 분석 (Video Batch 검증)').run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.radio, '검증 방식 선택').set_value(
            # 앞줄에서 시작한 식·인수·자료 구조를 `'모드 3: 오프라인 비디오 분석 (Video Batch 검증)').run()` 내용으로 이어서 구성합니다.
            '모드 3: 오프라인 비디오 분석 (Video Batch 검증)').run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertNotIn('카메라 선택 (Camera)', [item.label for item in self.app.selectbox])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertNotIn('카메라 선택 (Camera)', [item.label for item in self.app.selectbox])

    # 회귀 테스트: `test_initial_page_does_not_open_camera` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_initial_page_does_not_open_camera(self):
        # `patch('cv2.VideoCapture', side_effect=AssertionError('UI must not own the camera'))`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch('cv2.VideoCapture', side_effect=AssertionError('UI must not own the camera')):
            # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.app.run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertIn('세션 시작 (Start Session)', [button.label for button in self.app.button])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIn('세션 시작 (Start Session)', [button.label for button in self.app.button])

    # 회귀 테스트: `test_fractional_score_target_and_accumulated_points_are_displayed` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_fractional_score_target_and_accumulated_points_are_displayed(self):
        # `widget(self.app.number_input, '목표 점수 (Target Score)').set_value(1.25)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '목표 점수 (Target Score)').set_value(1.25)
        # 클라이언트에 `self.start_session()` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session()
        # 수신 또는 전송할 JSON 메시지 사전에 `next((packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION'))` 호출 결과를 저장합니다. 대상: `packet`.
        packet = next(packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION')
        # 검증문 `self.assertNotIn('target_score', packet)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertNotIn('target_score', packet)
        # 검증문 `self.assertEqual(client.config['target_score'], 1.25)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(client.config['target_score'], 1.25)
        # `pose` 값에 `dict(client.snapshot()['pose'])` 호출 결과를 저장합니다. 대상: `pose`.
        pose = dict(client.snapshot()['pose'])
        # `pose['left']` 값에 키 'total_score', 'rep_count'로 구성한 사전을 저장합니다. 대상: `pose['left']`.
        pose['left'] = {**pose['left'], 'total_score': 0.75, 'rep_count': 2}
        # `self.server.send(pose)`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send(pose)
        # `((3, 'GOOD'), (4, 'BAD'))`에서 항목을 차례로 꺼내 `(rep_id, quality)`에 넣고 아래 처리를 반복합니다.
        for rep_id, quality in ((3, 'GOOD'), (4, 'BAD')):
            # `self.server.send({'type': 'REP_COMPLETED', 'session_id': 'test', 'side': 'left', 'rep_id': rep_id, 'quality': quality, 'mode': 'TEST'})`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
            self.server.send({'type': 'REP_COMPLETED', 'session_id': 'test', 'side': 'left',
                              # 구성 항목: 반복 id(`'rep_id'`)에 반복 id(`rep_id`); 동작 품질 등급(`'quality'`)에 동작 품질 등급(`quality`); 실행 모드(`'mode'`)에 'TEST'을 지정합니다.
                              'rep_id': rep_id, 'quality': quality, 'mode': 'TEST'})
        # `wait_for(lambda : client.snapshot()['pose']['left'].get('total_score') == 0.75)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['pose']['left'].get('total_score') == 0.75)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assertEqual(widget(self.app.metric, '왼쪽 (Left)').value, '0.75 / 1.25점')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(widget(self.app.metric, '왼쪽 (Left)').value, '0.75 / 1.25점')
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()

    # 회귀 테스트: `test_preparation_can_be_disabled_and_countdown_is_shown` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_preparation_can_be_disabled_and_countdown_is_shown(self):
        # `widget(self.app.number_input, '측정 전 준비 시간 (초)').set_value(0.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '측정 전 준비 시간 (초)').set_value(0.0)
        # 클라이언트에 `self.start_session()` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session()
        # 수신 또는 전송할 JSON 메시지 사전에 `next((packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION'))` 호출 결과를 저장합니다. 대상: `packet`.
        packet = next(packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION')
        # 검증문 `self.assertEqual(packet['preparation_seconds'], 0.0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(packet['preparation_seconds'], 0.0)
        # `self.server.send({'type': 'SESSION_PREPARING', 'remaining_seconds': 3, 'message': '기본 자세를 취해 주세요. 3초 후 측정을 시작합니다.'})`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send({'type': 'SESSION_PREPARING', 'remaining_seconds': 3,
                          # 구성 항목: 메시지(`'message'`)에 '기본 자세를 취해 주세요. 3초 후 측정을 시작합니다.'을 지정합니다.
                          'message': '기본 자세를 취해 주세요. 3초 후 측정을 시작합니다.'})
        # `wait_for(lambda : client.snapshot()['status'] == 'PREPARING')`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['status'] == 'PREPARING')
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assertTrue(any(('3초 후 측정을 시작' in message.value for message in self.app.info)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any('3초 후 측정을 시작' in message.value for message in self.app.info))
        # `self.server.send({'type': 'SESSION_MEASUREMENT_STARTED'})`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send({'type': 'SESSION_MEASUREMENT_STARTED'})
        # `wait_for(lambda : client.snapshot()['status'] == 'RUNNING')`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['status'] == 'RUNNING')
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assertIsNone(self.app.session_state['session_notice'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(self.app.session_state['session_notice'])
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()

    # 회귀 테스트: `test_apply_rom_keeps_connection_counts_and_draft_values` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_apply_rom_keeps_connection_counts_and_draft_values(self):
        # 클라이언트에 `self.start_session()` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session()
        # `widget(self.app.number_input, '왼쪽 목표값').set_value(0.5)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.5)
        # `widget(self.app.number_input, '오른쪽 목표값').set_value(0.5)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '오른쪽 목표값').set_value(0.5)
        # `widget(self.app.button, '가동 범위 적용').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.button, '가동 범위 적용').click().run()
        # `wait_for(lambda : client.snapshot()['threshold_status']['threshold_version'] == 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['threshold_status']['threshold_version'] == 1)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertEqual(self.server.connections, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.connections, 1)
        # 검증문 `self.assertEqual(self.server.starts, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.starts, 1)
        # 검증문 `self.assertIs(self.app.session_state['session_client'], client)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIs(self.app.session_state['session_client'], client)
        # 검증문 `self.assertEqual(client.snapshot()['pose']['left']['rep_count'], 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(client.snapshot()['pose']['left']['rep_count'], 2)
        # 검증문 `self.assertEqual(widget(self.app.number_input, '왼쪽 목표값').value, 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(widget(self.app.number_input, '왼쪽 목표값').value, 0.5)
        # 검증문 `self.assertTrue(any(('적용 완료' in message.value for message in self.app.success)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any('적용 완료' in message.value for message in self.app.success))
        # 검증문 `self.assertIsNone(self.app.session_state['pending_rom_request'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(self.app.session_state['pending_rom_request'])

    # 회귀 테스트: `test_editing_form_does_not_send_until_apply_and_can_reload_server_values` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_editing_form_does_not_send_until_apply_and_can_reload_server_values(self):
        # `self.start_session()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_session()
        # `widget(self.app.number_input, '왼쪽 목표값').set_value(0.6).run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.6).run()
        # 검증문 `self.assertEqual([p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD'], [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual([p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD'], [])
        # `widget(self.app.button, '현재 적용값 불러오기').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.button, '현재 적용값 불러오기').click().run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertEqual(widget(self.app.number_input, '왼쪽 목표값').value, 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(widget(self.app.number_input, '왼쪽 목표값').value, 0.9)
        # 검증문 `self.assertEqual(self.server.starts, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.starts, 1)

    # 회귀 테스트: `test_left_only_update_and_invalid_input_recovery` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_left_only_update_and_invalid_input_recovery(self):
        # 클라이언트에 `self.start_session()` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session()
        # `widget(self.app.radio, '수정 대상').set_value('left').run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.radio, '수정 대상').set_value('left').run()
        # 검증문 `self.assertTrue(widget(self.app.number_input, '오른쪽 목표값').disabled)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(widget(self.app.number_input, '오른쪽 목표값').disabled)
        # `widget(self.app.number_input, '왼쪽 목표값').set_value(0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.1)
        # `widget(self.app.button, '가동 범위 적용').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.button, '가동 범위 적용').click().run()
        # `wait_for(lambda : len([p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD']) == 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: len([p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD']) == 1)
        # `wait_for(lambda : client.snapshot()['packet_count'] >= 5)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['packet_count'] >= 5)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertTrue(any(('requires' in error.value for error in self.app.error)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any('requires' in error.value for error in self.app.error))
        # 검증문 `self.assertEqual(client.snapshot()['threshold_status']['threshold_version'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(client.snapshot()['threshold_status']['threshold_version'], 0)
        # `widget(self.app.number_input, '왼쪽 목표값').set_value(0.5)`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.number_input, '왼쪽 목표값').set_value(0.5)
        # `widget(self.app.button, '가동 범위 적용').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.button, '가동 범위 적용').click().run()
        # `wait_for(lambda : client.snapshot()['threshold_status']['threshold_version'] == 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['threshold_status']['threshold_version'] == 1)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # `updates` 값에 반복·조건식 `[p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD']`으로 구성한 값을 저장합니다. 대상: `updates`.
        updates = [p for p in self.server.packets if p['type'] == 'CMD_UPDATE_COUNT_THRESHOLD']
        # 검증문 `self.assertEqual(set(updates[-1]['thresholds']), {'left'})`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(set(updates[-1]['thresholds']), {'left'})
        # 검증문 `self.assertEqual(client.snapshot()['threshold_status']['thresholds']['right']['target_val'], 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(client.snapshot()['threshold_status']['thresholds']['right']['target_val'], 0.9)
        # 검증문 `self.assertEqual(self.server.connections, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.connections, 1)

    # 회귀 테스트: `test_calibration_disables_rom_updates` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_calibration_disables_rom_updates(self):
        # `self.start_session('CALIBRATION')`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_session('CALIBRATION')
        # 검증문 `self.assertTrue(widget(self.app.button, '가동 범위 적용').disabled)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(widget(self.app.button, '가동 범위 적용').disabled)

    # 회귀 테스트: `test_log_settings_are_sent_and_actual_server_path_is_displayed` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_log_settings_are_sent_and_actual_server_path_is_displayed(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # `widget(self.app.number_input, '목표 점수 (Target Score)').set_value(40.0)`을 호출하여 해당 단계의 작업을 수행합니다.
            widget(self.app.number_input, '목표 점수 (Target Score)').set_value(40.0)
            # `widget(self.app.text_input, '로그 저장 폴더 (서버 기준)').set_value(directory)`을 호출하여 해당 단계의 작업을 수행합니다.
            widget(self.app.text_input, '로그 저장 폴더 (서버 기준)').set_value(directory)
            # `widget(self.app.number_input, 'Session').set_value(2)`을 호출하여 해당 단계의 작업을 수행합니다.
            widget(self.app.number_input, 'Session').set_value(2)
            # `widget(self.app.number_input, 'Round').set_value(3)`을 호출하여 해당 단계의 작업을 수행합니다.
            widget(self.app.number_input, 'Round').set_value(3)
            # `widget(self.app.number_input, 'Set').set_value(4)`을 호출하여 해당 단계의 작업을 수행합니다.
            widget(self.app.number_input, 'Set').set_value(4)
            # 클라이언트에 `self.start_session()` 호출 결과를 저장합니다. 대상: `client`.
            client = self.start_session()
            # 수신 또는 전송할 JSON 메시지 사전에 `next((packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION'))` 호출 결과를 저장합니다. 대상: `packet`.
            packet = next(packet for packet in self.server.packets if packet['type'] == 'CMD_SET_SESSION')
            # 검증문 `self.assertEqual(packet['log_root'], directory)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(packet['log_root'], directory)
            # 검증문 `self.assertEqual((packet['session'], packet['round'], packet['set']), (2, 3, 4))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual((packet['session'], packet['round'], packet['set']), (2, 3, 4))
            # 검증문 `self.assertTrue(packet['record_video'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(packet['record_video'])
            # 검증문 `self.assertFalse(packet['cal'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertFalse(packet['cal'])
            # 검증문 `self.assertEqual(packet['preparation_seconds'], 5.0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(packet['preparation_seconds'], 5.0)
            # 검증문 `self.assertNotIn('target_score', packet)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertNotIn('target_score', packet)
            # 검증문 `self.assertEqual(client.config['target_score'], 40.0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(client.config['target_score'], 40.0)
            # 검증문 `self.assertTrue(client.snapshot()['log_info']['video_mp4'].endswith('session2_round3_set4.mp4'))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(client.snapshot()['log_info']['video_mp4'].endswith('session2_round3_set4.mp4'))
            # 검증문 `self.assertTrue(any((directory in code.value for code in self.app.code)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(any(directory in code.value for code in self.app.code))

    # 회귀 테스트: `test_only_lightweight_and_offline_modes_are_available` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_only_lightweight_and_offline_modes_are_available(self):
        # 클라이언트에 `self.start_session(visual=True)` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session(visual=True)
        # 검증문 `self.assertEqual(client.config['output_mode'], 'LIGHTWEIGHT')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(client.config['output_mode'], 'LIGHTWEIGHT')
        # 검증문 `self.assertFalse(any(('모드 1' in option for option in widget(self.app.radio, '검증 방식 선택').options)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(any('모드 1' in option for option in widget(self.app.radio, '검증 방식 선택').options))
        # 검증문 `self.assertEqual(len(self.app.toggle), 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(self.app.toggle), 0)
        # 검증문 `self.assertFalse(widget(self.app.button, '가동 범위 적용').disabled)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(widget(self.app.button, '가동 범위 적용').disabled)

    # 회귀 테스트: `test_obsolete_image_payload_is_ignored_in_lightweight_mode` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_obsolete_image_payload_is_ignored_in_lightweight_mode(self):
        # 클라이언트에 `self.start_session(visual=True)` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session(visual=True)
        # `pose` 값에 `dict(client.snapshot()['pose'])` 호출 결과를 저장합니다. 대상: `pose`.
        pose = dict(client.snapshot()['pose'])
        # `pose['frame_b64']` 값에 'invalid image base64!'을 저장합니다. 대상: `pose['frame_b64']`.
        pose['frame_b64'] = 'invalid image base64!'
        # `self.server.send(pose)`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send(pose)
        # `wait_for(lambda : client.snapshot()['pose'].get('frame_b64') == 'invalid image base64!')`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['pose'].get('frame_b64') == 'invalid image base64!')
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertFalse(any(('카메라 미리보기 처리 오류' in message.value for message in self.app.error)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(any('카메라 미리보기 처리 오류' in message.value for message in self.app.error))

    # 회귀 테스트: `test_model_loading_is_visible_then_session_starts_without_reconnecting` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_model_loading_is_visible_then_session_starts_without_reconnecting(self):
        # `self.server.model_ready.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
        self.server.model_ready.clear()
        # 클라이언트에 `self.start_session(wait_until_ready=False)` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session(wait_until_ready=False)
        # `wait_for(lambda : client.snapshot()['server_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['server_status'] is not None)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertTrue(any(('모델을 불러오는 중' in info.value for info in self.app.info)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any('모델을 불러오는 중' in info.value for info in self.app.info))
        # 검증문 `self.assertIsNone(client.snapshot()['threshold_status'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(client.snapshot()['threshold_status'])
        # `self.server.model_ready.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.server.model_ready.set()
        # `wait_for(lambda : client.snapshot()['threshold_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: client.snapshot()['threshold_status'] is not None)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertFalse(widget(self.app.button, '가동 범위 적용').disabled)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(widget(self.app.button, '가동 범위 적용').disabled)
        # 검증문 `self.assertEqual(self.server.connections, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.connections, 1)

    # 회귀 테스트: `test_stop_allows_starting_new_session` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_stop_allows_starting_new_session(self):
        # 클라이언트에 `self.start_session()` 호출 결과를 저장합니다. 대상: `client`.
        client = self.start_session()
        # `widget(self.app.button, '세션 중지 (Stop Session)').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.button, '세션 중지 (Stop Session)').click().run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # 검증문 `self.assertFalse(client.running)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(client.running)
        # 검증문 `self.assertTrue(self.server.disconnected.wait(2))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.server.disconnected.wait(2))
        # `widget(self.app.button, '세션 시작 (Start Session)').click().run()`을 호출하여 해당 단계의 작업을 수행합니다.
        widget(self.app.button, '세션 시작 (Start Session)').click().run()
        # 검증문 `self.assert_no_ui_errors()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assert_no_ui_errors()
        # next 클라이언트에 계산식 `self.app.session_state['session_client']`의 결과를 저장합니다. 대상: `next_client`.
        next_client = self.app.session_state['session_client']
        # `wait_for(lambda : next_client.snapshot()['threshold_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: next_client.snapshot()['threshold_status'] is not None)
        # `self.app.run()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.app.run()
        # 검증문 `self.assertIsNot(client, next_client)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNot(client, next_client)
        # 검증문 `self.assertEqual(self.server.starts, 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.starts, 2)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
