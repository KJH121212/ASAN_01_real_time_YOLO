# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# `network.streamlit_client`에서 `StreamlitSessionClient` 기능을 가져옵니다.
from network.streamlit_client import StreamlitSessionClient
# `tests.support`에서 `MockSessionServer`, `wait_for` 기능을 가져옵니다.
from tests.support import MockSessionServer, wait_for


# `StreamlitClientTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class StreamlitClientTests(unittest.TestCase):
    # 회귀 테스트: `test_disappeared_ui_requests_stop_and_closes_socket` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_disappeared_ui_requests_stop_and_closes_socket(self):
        # 클라이언트에 `StreamlitSessionClient(self.server.uri, self.config, ui_heartbeat_timeout=0.3)` 호출 결과를 저장합니다. 대상: `self.client`.
        self.client = StreamlitSessionClient(self.server.uri, self.config, ui_heartbeat_timeout=0.3)
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `wait_for(lambda : not self.client.running)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: not self.client.running)
        # 명령에 반복·조건식 `[packet for packet in self.server.packets if packet['type'] == 'CMD_STOP_SESSION']`으로 구성한 값을 저장합니다. 대상: `commands`.
        commands = [packet for packet in self.server.packets if packet['type'] == 'CMD_STOP_SESSION']
        # 검증문 `self.assertEqual(len(commands), 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(commands), 1)
        # 검증문 `self.assertEqual(commands[0]['reason'], 'UI_DISCONNECTED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(commands[0]['reason'], 'UI_DISCONNECTED')
        # 검증문 `self.assertTrue(self.server.disconnected.wait(2))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.server.disconnected.wait(2))

    # 회귀 테스트: `test_active_ui_heartbeat_keeps_socket_alive` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_active_ui_heartbeat_keeps_socket_alive(self):
        # 이 파일에서 사용할 `time` 모듈을 불러옵니다.
        import time
        # 클라이언트에 `StreamlitSessionClient(self.server.uri, self.config, ui_heartbeat_timeout=0.3)` 호출 결과를 저장합니다. 대상: `self.client`.
        self.client = StreamlitSessionClient(self.server.uri, self.config, ui_heartbeat_timeout=0.3)
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `range(8)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(8):
            # `self.client.touch_ui()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.client.touch_ui()
            # `time.sleep(0.075)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다.
            time.sleep(0.075)
        # 검증문 `self.assertTrue(self.client.running)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.client.running)
        # 검증문 `self.assertFalse(any((packet['type'] == 'CMD_STOP_SESSION' for packet in self.server.packets)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(any(packet['type'] == 'CMD_STOP_SESSION' for packet in self.server.packets))

    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 서버에 `MockSessionServer().start()` 호출 결과를 저장합니다. 대상: `self.server`.
        self.server = MockSessionServer().start()
        # 현재 작업 설정에 `dict(player_id='ui_test', patient_name='Test', exercise_name='biceps_curl', mode='MAIN', target_reps=40, output_mode='LIGHTWEIGHT', video_path=None)` 호출 결과를 저장합니다. 대상: `self.config`.
        self.config = dict(player_id='ui_test', patient_name='Test', exercise_name='biceps_curl',
                           # 구성 항목: 실행 모드(`mode`)에 'MAIN'; 목표 반복 횟수(`target_reps`)에 40; 출력 mode(`output_mode`)에 'LIGHTWEIGHT'; 서버 PC에서 읽을 원본 영상 경로(`video_path`)에 None을 지정합니다.
                           mode='MAIN', target_reps=40, output_mode='LIGHTWEIGHT', video_path=None)
        # 클라이언트에 `StreamlitSessionClient(self.server.uri, self.config)` 호출 결과를 저장합니다. 대상: `self.client`.
        self.client = StreamlitSessionClient(self.server.uri, self.config)

    # 각 테스트가 끝나면 패치·임시 자원을 해제하여 다음 테스트에 영향을 남기지 않습니다.
    def tearDown(self):
        # `self.client.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.stop()
        # `self.server.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.server.stop()

    # 시작 클라이언트 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start_client(self):
        # `self.client.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.start()
        # `wait_for(lambda : self.client.snapshot()['threshold_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)

    # 회귀 테스트: `test_edits_and_queries_use_one_connection_without_resetting_reps` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_edits_and_queries_use_one_connection_without_resetting_reps(self):
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `request` 값에 `self.client.threshold_command({'left': {'target_val': 0.5}})` 호출 결과를 저장합니다. 대상: `request`.
        request = self.client.threshold_command({'left': {'target_val': 0.5}})
        # `wait_for(lambda : self.client.snapshot()['threshold_status']['threshold_version'] == 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['threshold_status']['threshold_version'] == 1)
        # `self.client.threshold_command()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.threshold_command()
        # `wait_for(lambda : len(self.server.packets) >= 4)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: len(self.server.packets) >= 4)
        # 검증문 `self.assertEqual(self.server.connections, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.connections, 1)
        # 검증문 `self.assertEqual(self.server.starts, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.starts, 1)
        # 검증문 `self.assertEqual(self.client.snapshot()['pose']['left']['rep_count'], 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['pose']['left']['rep_count'], 2)
        # 검증문 `self.assertTrue(any((event.get('request_id') == request for event in self.client.drain_events())))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any(event.get('request_id') == request for event in self.client.drain_events()))

    # 회귀 테스트: `test_rejected_edit_does_not_disconnect_or_change_applied_values` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_rejected_edit_does_not_disconnect_or_change_applied_values(self):
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `self.client.drain_events()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.drain_events()
        # `self.client.threshold_command({'left': {'target_val': 0.1}})`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.threshold_command({'left': {'target_val': 0.1}})
        # `wait_for(lambda : len(self.server.packets) >= 3)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: len(self.server.packets) >= 3)
        # 이벤트에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `events`.
        events = []

        # received 오류 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def received_error():
            # `events.extend(self.client.drain_events())`을 호출하여 여러 항목을 순서대로 추가합니다.
            events.extend(self.client.drain_events())
            # 현재 함수 처리를 끝내고 `any((event.get('type') == 'ERROR' for event in events))` 호출 결과를 호출한 쪽으로 반환합니다.
            return any(event.get('type') == 'ERROR' for event in events)

        # `wait_for(received_error)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(received_error)
        # 검증문 `self.assertEqual(self.client.snapshot()['threshold_status']['thresholds']['left']['target_val'], 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['threshold_status']['thresholds']['left']['target_val'], 0.9)
        # 검증문 `self.assertTrue(self.client.running)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.client.running)
        # `self.client.threshold_command({'left': {'target_val': 0.5}})`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.threshold_command({'left': {'target_val': 0.5}})
        # `wait_for(lambda : self.client.snapshot()['threshold_status']['threshold_version'] == 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['threshold_status']['threshold_version'] == 1)

    # 회귀 테스트: `test_snapshot_and_configuration_are_copies` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_snapshot_and_configuration_are_copies(self):
        # `self.config['exercise_name']` 값에 'other_game'을 저장합니다. 대상: `self.config['exercise_name']`.
        self.config['exercise_name'] = 'other_game'
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `snapshot` 값에 `self.client.snapshot()` 호출 결과를 저장합니다. 대상: `snapshot`.
        snapshot = self.client.snapshot()
        # `snapshot['threshold_status']['thresholds']['left']['target_val']` 값에 999을 저장합니다. 대상: `snapshot['threshold_status']['thresholds']['left']['target_val']`.
        snapshot['threshold_status']['thresholds']['left']['target_val'] = 999
        # 검증문 `self.assertEqual(self.client.config['exercise_name'], 'biceps_curl')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.config['exercise_name'], 'biceps_curl')
        # 검증문 `self.assertEqual(self.client.snapshot()['threshold_status']['thresholds']['left']['target_val'], 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['threshold_status']['thresholds']['left']['target_val'], 0.9)

    # 회귀 테스트: `test_stop_closes_socket_and_prevents_new_commands` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_stop_closes_socket_and_prevents_new_commands(self):
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `self.client.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.stop()
        # 검증문 `self.assertFalse(self.client.running)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(self.client.running)
        # 검증문 `self.assertTrue(self.server.disconnected.wait(2))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.server.disconnected.wait(2))
        # 검증문 `self.assertEqual(self.client.snapshot()['status'], 'FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        # `self.assertRaises(RuntimeError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(RuntimeError):
            # `self.client.threshold_command()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.client.threshold_command()

    # 회귀 테스트: `test_completion_is_reported_and_worker_exits` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_completion_is_reported_and_worker_exits(self):
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `self.server.send({'type': 'SESSION_FINISHED', 'summary': {'saved_reps_count': 2}})`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send({'type': 'SESSION_FINISHED', 'summary': {'saved_reps_count': 2}})
        # `wait_for(lambda : not self.client.running)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: not self.client.running)
        # 검증문 `self.assertEqual(self.client.snapshot()['status'], 'FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        # 검증문 `self.assertTrue(any((event['type'] == 'SESSION_FINISHED' for event in self.client.drain_events())))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any(event['type'] == 'SESSION_FINISHED' for event in self.client.drain_events()))

    # 회귀 테스트: `test_calibration_disallows_edits` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_calibration_disallows_edits(self):
        # `self.client.config['mode']` 값에 'CALIBRATION'을 저장합니다. 대상: `self.client.config['mode']`.
        self.client.config['mode'] = 'CALIBRATION'
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # 검증문 `self.assertFalse(self.client.can_edit_thresholds)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(self.client.can_edit_thresholds)
        # `self.assertRaises(RuntimeError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(RuntimeError):
            # `self.client.threshold_command({'left': {'target_val': 0.5}})`을 호출하여 해당 단계의 작업을 수행합니다.
            self.client.threshold_command({'left': {'target_val': 0.5}})

    # 회귀 테스트: `test_model_loading_state_keeps_connection_until_session_is_ready` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_model_loading_state_keeps_connection_until_session_is_ready(self):
        # `self.server.model_ready.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
        self.server.model_ready.clear()
        # `self.client.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.start()
        # `wait_for(lambda : self.client.snapshot()['server_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['server_status'] is not None)
        # `snapshot` 값에 `self.client.snapshot()` 호출 결과를 저장합니다. 대상: `snapshot`.
        snapshot = self.client.snapshot()
        # 검증문 `self.assertEqual(snapshot['status'], 'PREPARING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(snapshot['status'], 'PREPARING')
        # 검증문 `self.assertIsNone(snapshot['threshold_status'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(snapshot['threshold_status'])
        # 검증문 `self.assertFalse(self.client.can_edit_thresholds)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(self.client.can_edit_thresholds)
        # `self.server.model_ready.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.server.model_ready.set()
        # `wait_for(lambda : self.client.snapshot()['threshold_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)
        # 검증문 `self.assertTrue(self.client.can_edit_thresholds)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.client.can_edit_thresholds)
        # 검증문 `self.assertEqual(self.server.starts, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.starts, 1)
        # 검증문 `self.assertEqual(self.server.connections, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.server.connections, 1)

    # 회귀 테스트: `test_offline_does_not_query_or_edit_during_analysis` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_offline_does_not_query_or_edit_during_analysis(self):
        # `self.client.config['video_path']` 값에 'example.mp4'을 저장합니다. 대상: `self.client.config['video_path']`.
        self.client.config['video_path'] = 'example.mp4'
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # 검증문 `self.assertEqual([p['type'] for p in self.server.packets], ['CMD_SET_SESSION'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual([p['type'] for p in self.server.packets], ['CMD_SET_SESSION'])
        # 검증문 `self.assertFalse(self.client.can_edit_thresholds)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(self.client.can_edit_thresholds)

    # 회귀 테스트: `test_recording_runs_in_worker_and_closes_with_session` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_recording_runs_in_worker_and_closes_with_session(self):
        # `Recorder` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
        class Recorder:
            # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
            def __init__(self):
                # 동작 시작 시각 또는 실행 여부에 False을 저장합니다. 대상: `self.started`.
                self.started = False
                # `self.stopped` 값에 False을 저장합니다. 대상: `self.stopped`.
                self.stopped = False
                # `self.poses` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.poses`.
                self.poses = []

            # 시작 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
            def start(self):
                # 동작 시작 시각 또는 실행 여부에 True을 저장합니다. 대상: `self.started`.
                self.started = True

            # 기록 pose 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
            def record_pose(self, packet):
                # `self.poses.append(packet)`을 호출하여 항목 하나를 끝에 추가합니다.
                self.poses.append(packet)

            # 중지 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
            def stop(self):
                # `self.stopped` 값에 True을 저장합니다. 대상: `self.stopped`.
                self.stopped = True

        # `recorder` 값에 `Recorder()` 호출 결과를 저장합니다. 대상: `recorder`.
        recorder = Recorder()
        # `self.client.recorder` 값에 `recorder` 값(`recorder`)을 저장합니다. 대상: `self.client.recorder`.
        self.client.recorder = recorder
        # `self.start_client()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_client()
        # `wait_for(lambda : bool(recorder.poses))`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: bool(recorder.poses))
        # 검증문 `self.assertTrue(recorder.started)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(recorder.started)
        # `self.client.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.stop()
        # 검증문 `self.assertTrue(recorder.stopped)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(recorder.stopped)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
