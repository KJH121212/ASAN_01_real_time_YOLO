# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `types`에서 `SimpleNamespace` 기능을 가져옵니다.
from types import SimpleNamespace
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# `unittest.mock`에서 `AsyncMock`, `Mock` 기능을 가져옵니다.
from unittest.mock import AsyncMock, Mock
# `websockets.exceptions`에서 `ConnectionClosedOK` 기능을 가져옵니다.
from websockets.exceptions import ConnectionClosedOK

# `network.client_progress`에서 `ClientProgress` 기능을 가져옵니다.
from network.client_progress import ClientProgress
# `network.streamlit_client`에서 `StreamlitSessionClient` 기능을 가져옵니다.
from network.streamlit_client import StreamlitSessionClient
# `tests.support`에서 `MockSessionServer`, `wait_for` 기능을 가져옵니다.
from tests.support import MockSessionServer, wait_for
# `tests.test_skeleton_batch`에서 `load_model_free_module` 기능을 가져옵니다.
from tests.test_skeleton_batch import load_model_free_module


# 반복 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def rep(rep_id, quality='PERFECT', side='left', session_id='session'):
    # 현재 함수 처리를 끝내고 `dict(type='REP_COMPLETED', session_id=session_id, mode='MAIN', side=side, rep_id=rep_id, quality=quality, threshold_version=1)` 호출 결과를 호출한 쪽으로 반환합니다.
    return dict(type='REP_COMPLETED', session_id=session_id, mode='MAIN',
                # 구성 항목: 좌우 구분(`side`)에 좌우 구분(`side`); 반복 id(`rep_id`)에 반복 id(`rep_id`); 동작 품질 등급(`quality`)에 동작 품질 등급(`quality`); 현재 가동 범위 설정의 변경 번호(`threshold_version`)에 1을 지정합니다.
                side=side, rep_id=rep_id, quality=quality, threshold_version=1)


# `ClientProgressTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ClientProgressTests(unittest.TestCase):
    # 회귀 테스트: `test_exact_goal_does_not_finish_excess_does_after_delay` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_exact_goal_does_not_finish_excess_does_after_delay(self):
        # 진행률에 `ClientProgress(dict(mode='MAIN', target_score=40))` 호출 결과를 저장합니다. 대상: `progress`.
        progress = ClientProgress(dict(mode='MAIN', target_score=40))
        # `range(1, 41)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
        for i in range(1, 41):
            # 검증문 `self.assertTrue(progress.accept(rep(i), 0.0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(progress.accept(rep(i), 0.))
        # 검증문 `self.assertFalse(progress.finish_due(100.0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(progress.finish_due(100.))
        # `progress.accept(rep(41, 'GOOD'), 100.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        progress.accept(rep(41, 'GOOD'), 100.)
        # 검증문 `self.assertFalse(progress.finish_due(101.49))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(progress.finish_due(101.49))
        # 검증문 `self.assertTrue(progress.finish_due(101.5))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(progress.finish_due(101.5))
        # 검증문 `self.assertEqual(progress.scores['left'], 40.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(progress.scores['left'], 40.5)

    # 회귀 테스트: `test_duplicate_and_foreign_events_do_not_award_points` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_duplicate_and_foreign_events_do_not_award_points(self):
        # 진행률에 `ClientProgress(dict(mode='TEST'))` 호출 결과를 저장합니다. 대상: `progress`.
        progress = ClientProgress(dict(mode='TEST'))
        # `progress.bind_session('session')`을 호출하여 해당 단계의 작업을 수행합니다.
        progress.bind_session('session')
        # 검증문 `self.assertFalse(progress.accept(rep(1, session_id='old'), 0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(progress.accept(rep(1, session_id='old'), 0))
        # 검증문 `self.assertTrue(progress.accept(rep(1, 'GOOD'), 0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(progress.accept(rep(1, 'GOOD'), 0))
        # 검증문 `self.assertFalse(progress.accept(rep(1, 'GOOD'), 1))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(progress.accept(rep(1, 'GOOD'), 1))
        # 검증문 `self.assertTrue(progress.accept(rep(1, 'BAD', 'right'), 2))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(progress.accept(rep(1, 'BAD', 'right'), 2))
        # 검증문 `self.assertEqual(progress.scores, {'left': 0.5, 'right': 0.25})`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(progress.scores, {'left': .5, 'right': .25})

    # 회귀 테스트: `test_custom_client_weights_and_calibration_exclusion` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_custom_client_weights_and_calibration_exclusion(self):
        # `cfg` 값에 `dict(mode='MAIN', target_score=0.5, finish_delay_seconds=0, quality_points={'PERFECT': 2.0, 'GOOD': 1.0, 'BAD': 0.0})` 호출 결과를 저장합니다. 대상: `cfg`.
        cfg = dict(mode='MAIN', target_score=.5, finish_delay_seconds=0,
                   # 구성 항목: 품질 등급 points(`quality_points`)에 키 'PERFECT', 'GOOD', 'BAD'로 구성한 사전; `PERFECT` 값(`'PERFECT'`)에 2.0; `GOOD` 값(`'GOOD'`)에 1.0; `BAD` 값(`'BAD'`)에 0.0을 지정합니다.
                   quality_points={'PERFECT': 2., 'GOOD': 1., 'BAD': 0.})
        # 진행률에 `ClientProgress(cfg)` 호출 결과를 저장합니다. 대상: `progress`.
        progress = ClientProgress(cfg)
        # `progress.accept(rep(1, 'GOOD'), 1.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        progress.accept(rep(1, 'GOOD'), 1.)
        # 검증문 `self.assertTrue(progress.finish_due(1.0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(progress.finish_due(1.))
        # 캘리브레이션에 `ClientProgress({**cfg, 'mode': 'CALIBRATION'})` 호출 결과를 저장합니다. 대상: `calibration`.
        calibration = ClientProgress({**cfg, 'mode': 'CALIBRATION'})
        # 검증문 `self.assertFalse(calibration.accept(rep(1), 1.0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(calibration.accept(rep(1), 1.))
        # 검증문 `self.assertFalse(calibration.finish_due(100.0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(calibration.finish_due(100.))

    # 회귀 테스트: `test_offline_goal_does_not_wait_for_wall_clock_cooldown` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_offline_goal_does_not_wait_for_wall_clock_cooldown(self):
        # 진행률에 `ClientProgress(dict(mode='TEST', target_score=0.5, video_path='input.mp4'))` 호출 결과를 저장합니다. 대상: `progress`.
        progress = ClientProgress(dict(mode='TEST', target_score=.5, video_path='input.mp4'))
        # `progress.accept(rep(1), 0.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        progress.accept(rep(1), 0.)
        # 검증문 `self.assertTrue(progress.finish_due(0.0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(progress.finish_due(0.))


# `ClientCompletionTransportTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ClientCompletionTransportTests(unittest.TestCase):
    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 서버에 `MockSessionServer().start()` 호출 결과를 저장합니다. 대상: `self.server`.
        self.server = MockSessionServer().start()
        # 클라이언트에 `StreamlitSessionClient(self.server.uri, dict(player_id='client_test', exercise_name='biceps_curl', mode='MAIN', target_score=1, finish_delay_seconds=0, output_mode='LIGH…` 호출 결과를 저장합니다. 대상: `self.client`.
        self.client = StreamlitSessionClient(self.server.uri, dict(
            # 구성 항목: 환자 식별 번호의 기존 필드명(`player_id`)에 'client_test'; 운동 설정의 식별 이름(`exercise_name`)에 'biceps_curl'; 실행 모드(`mode`)에 'MAIN'을 지정합니다.
            player_id='client_test', exercise_name='biceps_curl', mode='MAIN',
            # 구성 항목: 게임 목표 점수(`target_score`)에 1; 종료 delay 초(`finish_delay_seconds`)에 0; 출력 mode(`output_mode`)에 'LIGHTWEIGHT'을 지정합니다.
            target_score=1, finish_delay_seconds=0, output_mode='LIGHTWEIGHT'))

    # 각 테스트가 끝나면 패치·임시 자원을 해제하여 다음 테스트에 영향을 남기지 않습니다.
    def tearDown(self):
        # `self.client.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.stop()
        # `self.server.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.server.stop()

    # 회귀 테스트: `test_goal_stops_through_command_once_without_pose_packets` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_goal_stops_through_command_once_without_pose_packets(self):
        # `self.client.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.start()
        # `wait_for(lambda : self.client.snapshot()['threshold_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)
        # 시작에 `next((p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION'))` 호출 결과를 저장합니다. 대상: `start`.
        start = next(p for p in self.server.packets if p['type'] == 'CMD_SET_SESSION')
        # 검증문 `self.assertNotIn('target_score', start)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertNotIn('target_score', start)
        # 검증문 `self.assertNotIn('target_reps', start)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertNotIn('target_reps', start)
        # `self.server.send(rep(1))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send(rep(1))
        # `self.server.send(rep(1))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send(rep(1))
        # `wait_for(lambda : self.client.snapshot()['progress']['scores']['left'] == 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['progress']['scores']['left'] == 1)
        # 검증문 `self.assertTrue(self.client.running)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.client.running)
        # `self.server.send(rep(2, 'GOOD'))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.server.send(rep(2, 'GOOD'))
        # `wait_for(lambda : not self.client.running)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: not self.client.running)
        # `stops` 값에 반복·조건식 `[p for p in self.server.packets if p['type'] == 'CMD_STOP_SESSION']`으로 구성한 값을 저장합니다. 대상: `stops`.
        stops = [p for p in self.server.packets if p['type'] == 'CMD_STOP_SESSION']
        # 검증문 `self.assertEqual(len(stops), 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(stops), 1)
        # 검증문 `self.assertEqual(stops[0]['reason'], 'TARGET_REACHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(stops[0]['reason'], 'TARGET_REACHED')
        # 검증문 `self.assertEqual(self.client.snapshot()['status'], 'FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        # 검증문 `self.assertEqual(self.client.snapshot()['progress']['scores']['left'], 1.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['progress']['scores']['left'], 1.5)

    # 회귀 테스트: `test_manual_stop_waits_for_server_acknowledgement` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_manual_stop_waits_for_server_acknowledgement(self):
        # `self.client.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.start()
        # `wait_for(lambda : self.client.snapshot()['threshold_status'] is not None)`을 호출하여 해당 단계의 작업을 수행합니다.
        wait_for(lambda: self.client.snapshot()['threshold_status'] is not None)
        # `self.client.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.client.stop()
        # 검증문 `self.assertEqual(self.client.snapshot()['status'], 'FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.client.snapshot()['status'], 'FINISHED')
        # `stops` 값에 반복·조건식 `[p for p in self.server.packets if p['type'] == 'CMD_STOP_SESSION']`으로 구성한 값을 저장합니다. 대상: `stops`.
        stops = [p for p in self.server.packets if p['type'] == 'CMD_STOP_SESSION']
        # 검증문 `self.assertEqual(stops[0]['reason'], 'USER_STOP')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(stops[0]['reason'], 'USER_STOP')
        # 검증문 `self.assertTrue(any((p['type'] == 'SESSION_FINISHED' for p in self.client.drain_events())))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any(p['type'] == 'SESSION_FINISHED' for p in self.client.drain_events()))


# `ServerClientCompletionTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ServerClientCompletionTests(unittest.IsolatedAsyncioTestCase):
    # 데코레이터 `classmethod`를 바로 아래 함수·클래스에 적용합니다.
    @classmethod
    # 설정 up class 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def setUpClass(cls):
        # `cls.module` 값에 `load_model_free_module('network/socket_server.py')` 호출 결과를 저장합니다. 대상: `cls.module`.
        cls.module = load_model_free_module('network/socket_server.py')

    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 서버에 `self.module.ExerciseSocketServer.__new__(self.module.ExerciseSocketServer)` 호출 결과를 저장합니다. 대상: `self.server`.
        self.server = self.module.ExerciseSocketServer.__new__(self.module.ExerciseSocketServer)
        # 생성자를 생략한 모의 서버에도 복구용 연결 완료 신호 저장소가 필요합니다.
        self.server._connection_done = {}
        # WebSocket 연결에 `SimpleNamespace(send=AsyncMock())` 호출 결과를 저장합니다. 대상: `self.ws`.
        self.ws = SimpleNamespace(send=AsyncMock())
        # 현재 서버의 측정 자원을 점유한 연결에 WebSocket 연결(`self.ws`)을 저장합니다. 대상: `self.server._session_owner`.
        self.server._session_owner = self.ws
        # 캘리브레이션·본 운동 진행 제어기에 `SimpleNamespace(mode='MAIN', data_manager=SimpleNamespace(session_log=SimpleNamespace(session_id='session', info={})), motion_engine=SimpleNamespace(exercise_name='test'…` 호출 결과를 저장합니다. 대상: `self.server.controller`.
        self.server.controller = SimpleNamespace(mode='MAIN', data_manager=SimpleNamespace(
            # 구성 항목: 관절 측정값 계산 및 반복 판정 엔진(`motion_engine`)에 `SimpleNamespace(exercise_name='test', mode='MAIN', threshold_version=3)` 호출 결과; 현재 세션의 CSV·NPZ·MP4 기록 객체(`session_log`)에 `SimpleNamespace(session_id='session', info={})` 호출 결과; 현재 측정 세션 식별자(`session_id`)에 'session'; `info` 값(`info`)에 빈 사전을 지정합니다.
            session_log=SimpleNamespace(session_id='session', info={})), motion_engine=SimpleNamespace(
            # 구성 항목: 운동 설정의 식별 이름(`exercise_name`)에 'test'; 실행 모드(`mode`)에 'MAIN'; 현재 가동 범위 설정의 변경 번호(`threshold_version`)에 3을 지정합니다.
            exercise_name='test', mode='MAIN', threshold_version=3))
        # 저장 측정 세션 파일에 `Mock(return_value={'saved_reps_count': 4})` 호출 결과를 저장합니다. 대상: `self.server._save_session_files`.
        self.server._save_session_files = Mock(return_value={'saved_reps_count': 4})
        # 중지 camera에 `Mock()` 호출 결과를 저장합니다. 대상: `self.server.stop_camera`.
        self.server.stop_camera = Mock()

    # 회귀 테스트: `test_stop_saves_once_and_duplicate_receives_same_ack` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_stop_saves_once_and_duplicate_receives_same_ack(self):
        # 수신 또는 전송할 JSON 메시지 사전에 `dict(type='CMD_STOP_SESSION', session_id='session', reason='TARGET_REACHED', request_id='a')` 호출 결과를 저장합니다. 대상: `packet`.
        packet = dict(type='CMD_STOP_SESSION', session_id='session', reason='TARGET_REACHED', request_id='a')
        # `self.server._finish_client_session(self.ws, packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server._finish_client_session(self.ws, packet)
        # 검증문 `self.server._save_session_files.assert_called_once_with(status='COMPLETED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server._save_session_files.assert_called_once_with(status='COMPLETED')
        # 검증문 `self.server.stop_camera.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server.stop_camera.assert_called_once()
        # 검증문 `self.assertIsNone(self.server.controller)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(self.server.controller)
        # `self.server._finish_client_session(self.ws, {**packet, 'request_id': 'b'})`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server._finish_client_session(self.ws, {**packet, 'request_id': 'b'})
        # 검증문 `self.server._save_session_files.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server._save_session_files.assert_called_once()
        # `response` 값에 JSON 문자열을 해석한 Python 값(`json.loads(self.ws.send.call_args.args[0])`)을 저장합니다. 대상: `response`.
        response = json.loads(self.ws.send.call_args.args[0])
        # 검증문 `self.assertEqual(response['request_id'], 'b')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['request_id'], 'b')
        # 검증문 `self.assertEqual(response['type'], 'SESSION_FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['type'], 'SESSION_FINISHED')

    # 회귀 테스트: `test_other_connection_or_session_cannot_stop_measurement` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_other_connection_or_session_cannot_stop_measurement(self):
        # `other` 값에 `SimpleNamespace(send=AsyncMock())` 호출 결과를 저장합니다. 대상: `other`.
        other = SimpleNamespace(send=AsyncMock())
        # `self.server._finish_client_session(other, dict(type='CMD_STOP_SESSION'))`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server._finish_client_session(other, dict(type='CMD_STOP_SESSION'))
        # 검증문 `self.server._save_session_files.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server._save_session_files.assert_not_called()
        # `self.server._finish_client_session(self.ws, dict(session_id='old'))`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server._finish_client_session(self.ws, dict(session_id='old'))
        # 검증문 `self.server._save_session_files.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server._save_session_files.assert_not_called()
        # 검증문 `self.assertEqual(json.loads(self.ws.send.call_args.args[0])['code'], 'SESSION_MISMATCH')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(json.loads(self.ws.send.call_args.args[0])['code'], 'SESSION_MISMATCH')

    # 회귀 테스트: `test_rep_event_contains_identity_quality_and_threshold_version` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_rep_event_contains_identity_quality_and_threshold_version(self):
        # `self.server._send_rep_events(self.ws, dict(is_count_updated=True, rep_count=12, quality='GOOD', val=0.7), dict(is_count_updated=False), timestamp=1.5)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server._send_rep_events(self.ws,
            # 구성 항목: 이번 프레임에서 반복 횟수가 증가했는지 여부(`is_count_updated`)에 True; 인정된 누적 반복 횟수(`rep_count`)에 12; 동작 품질 등급(`quality`)에 'GOOD'; `val` 값(`val`)에 0.7을 지정합니다.
            dict(is_count_updated=True, rep_count=12, quality='GOOD', val=.7),
            # 구성 항목: 입력 표본 또는 세션 기준 시각(초)(`timestamp`)에 1.5; 이번 프레임에서 반복 횟수가 증가했는지 여부(`is_count_updated`)에 False을 지정합니다.
            dict(is_count_updated=False), timestamp=1.5)
        # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(self.ws.send.call_args.args[0])`)을 저장합니다. 대상: `packet`.
        packet = json.loads(self.ws.send.call_args.args[0])
        # 검증문 `self.assertEqual((packet['session_id'], packet['side'], packet['rep_id']), ('session', 'left', 12))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual((packet['session_id'], packet['side'], packet['rep_id']), ('session', 'left', 12))
        # 검증문 `self.assertEqual(packet['quality'], 'GOOD')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(packet['quality'], 'GOOD')
        # 검증문 `self.assertEqual(packet['threshold_version'], 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(packet['threshold_version'], 3)
        # 검증문 `self.assertNotIn('score_delta', packet)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertNotIn('score_delta', packet)

    # 회귀 테스트: `test_client_game_result_is_saved_as_metadata` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_client_game_result_is_saved_as_metadata(self):
        # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.server.controller.data_manager.session_log`)을 저장합니다. 대상: `log`.
        log = self.server.controller.data_manager.session_log
        # 함수 또는 처리 단계의 결과에 키 'scores', 'target_score'로 구성한 사전을 저장합니다. 대상: `result`.
        result = {'scores': {'left': 40.5, 'right': 12.}, 'target_score': 40.}
        # `self.server._finish_client_session(self.ws, dict(reason='TARGET_REACHED', client_result=result))`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server._finish_client_session(self.ws, dict(
            # 구성 항목: `reason` 값(`reason`)에 'TARGET_REACHED'; 클라이언트 결과(`client_result`)에 함수 또는 처리 단계의 결과(`result`)을 지정합니다.
            reason='TARGET_REACHED', client_result=result))
        # 검증문 `self.assertEqual(log.info['client_result'], result)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(log.info['client_result'], result)
        # 검증문 `self.assertEqual(log.info['target_score'], 40.0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(log.info['target_score'], 40.)

    # 회귀 테스트: `test_production_command_loop_dispatches_stop_and_releases_owner` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_production_command_loop_dispatches_stop_and_releases_owner(self):
        # `self.ws.remote_address` 값에 순서·구성을 지정한 `('127.0.0.1', 1234)`을 저장합니다. 대상: `self.ws.remote_address`.
        self.ws.remote_address = ('127.0.0.1', 1234)
        # `self.ws.recv` 값에 `AsyncMock(side_effect=[json.dumps(dict(type='CMD_STOP_SESSION', session_id='session', reason='USER_STOP')), ConnectionClosedOK(None, None)])` 호출 결과를 저장합니다. 대상: `self.ws.recv`.
        self.ws.recv = AsyncMock(side_effect=[json.dumps(dict(
            # 구성 항목: `type` 값(`type`)에 'CMD_STOP_SESSION'; 현재 측정 세션 식별자(`session_id`)에 'session'; `reason` 값(`reason`)에 'USER_STOP'을 지정합니다.
            type='CMD_STOP_SESSION', session_id='session', reason='USER_STOP')),
            # 앞줄에서 시작한 식·인수·자료 구조를 `ConnectionClosedOK(None, None)])` 내용으로 이어서 구성합니다.
            ConnectionClosedOK(None, None)])
        # 백그라운드 AI 모델 준비 관리자에 `SimpleNamespace(start=lambda : None, snapshot=lambda : dict(type='SERVER_STATUS', model_state='READY', phase='READY', ready=True))` 호출 결과를 저장합니다. 대상: `self.server.model_loader`.
        self.server.model_loader = SimpleNamespace(start=lambda: None, snapshot=lambda: dict(
            # 구성 항목: `type` 값(`type`)에 'SERVER_STATUS'; 모델 단계(`model_state`)에 'READY'; `phase` 값(`phase`)에 'READY'; 준비 상태(`ready`)에 True을 지정합니다.
            type='SERVER_STATUS', model_state='READY', phase='READY', ready=True))
        # `self.server.handle_client(self.ws)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.handle_client(self.ws)
        # 검증문 `self.server._save_session_files.assert_called_once_with(status='STOPPED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server._save_session_files.assert_called_once_with(status='STOPPED')
        # 검증문 `self.assertIsNone(self.server._session_owner)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(self.server._session_owner)
        # `responses` 값에 반복·조건식 `[json.loads(call.args[0]) for call in self.ws.send.call_args_list]`으로 구성한 값을 저장합니다. 대상: `responses`.
        responses = [json.loads(call.args[0]) for call in self.ws.send.call_args_list]
        # 검증문 `self.assertTrue(any((p['type'] == 'SESSION_FINISHED' for p in responses)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(any(p['type'] == 'SESSION_FINISHED' for p in responses))
