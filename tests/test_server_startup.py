# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# `types`에서 `SimpleNamespace` 기능을 가져옵니다.
from types import SimpleNamespace
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# `unittest.mock`에서 `Mock` 기능을 가져옵니다.
from unittest.mock import Mock

# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `SessionController` 기능을 가져옵니다.
from core.session_controller import SessionController
# `network.socket_server`에서 `ExerciseSocketServer` 기능을 가져옵니다.
from network.socket_server import ExerciseSocketServer


# `ServerStartupTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ServerStartupTests(unittest.IsolatedAsyncioTestCase):
    # 이전 화면의 연결을 새 관리 연결에서 정리해도 모델은 다시 로드하지 않아야 합니다.
    async def test_monitor_reset_releases_owner_and_allows_a_new_session(self):
        # 모델 준비가 끝난 가짜 서버를 사용하므로 카메라·AI는 실행하지 않습니다.
        self.gate.set()
        await self.server.model_loader.task
        # 첫 연결에서 측정을 시작해 서버 소유권을 확보합니다.
        async with websockets.connect(self.uri) as old:
            await self.send_start(old)
            await self.until(old, lambda packet: packet['type'] == 'SESSION_LOG_STARTED')
            # 기록 중인 로그가 있으면 복구 과정에서도 저장 경로를 거쳐야 합니다.
            self.server._active_log = object()
            # 새 화면을 대신하는 별도 연결에서 이전 측정을 종료합니다.
            async with websockets.connect(self.uri) as monitor:
                await monitor.send(json.dumps({'type': 'CMD_RESET_SESSION', 'request_id': 'recover'}))
                response = await self.until(monitor, lambda packet: packet['type'] == 'SERVER_SESSION_RESET')
                # 완료 응답은 자원 정리 이후에만 전달되어야 합니다.
                self.assertTrue(response['reset'])
                self.assertEqual(response['request_id'], 'recover')
                self.assertIsNone(self.server._session_owner)
                self.assertIsNone(self.server.controller)
                self.server._save_session_files.assert_called_once_with(status='STOPPED')
                self.server.stop_camera.assert_called()
                # 동일한 새 연결에서 다시 측정을 시작할 수 있어야 합니다.
                await self.send_start(monitor)
                await self.until(monitor, lambda packet: packet['type'] == 'SESSION_LOG_STARTED')
                self.assertIsNotNone(self.server._session_owner)
                self.assertEqual(self.factory_calls, 1)

    # 측정이 없을 때 복구를 눌러도 모델·서버를 재시작하지 않습니다.
    async def test_reset_helper_handles_an_idle_server(self):
        # 화면과 동일한 복구 요청 함수를 실제 로컬 WebSocket에 연결합니다.
        from network.server_admin import reset_server_session
        response = await reset_server_session(self.uri)
        self.assertFalse(response['reset'])
        self.assertIsNone(self.server._session_owner)

    # 현재 측정 소유자는 연결을 스스로 끊는 복구 대신 정상 중지 명령을 써야 합니다.
    async def test_owner_cannot_reset_its_own_response_connection(self):
        self.gate.set()
        await self.server.model_loader.task
        async with websockets.connect(self.uri) as owner:
            await self.send_start(owner)
            await self.until(owner, lambda packet: packet['type'] == 'SESSION_LOG_STARTED')
            await owner.send(json.dumps({'type': 'CMD_RESET_SESSION'}))
            response = await self.until(owner, lambda packet: packet['type'] == 'ERROR')
            self.assertEqual(response['code'], 'RESET_FROM_MONITOR_REQUIRED')
            self.assertIsNotNone(self.server._session_owner)

    # 회귀 테스트: `test_removed_visual_mode_is_rejected_without_opening_camera` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_removed_visual_mode_is_rejected_without_opening_camera(self):
        # `self.gate.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.gate.set()
        # 표현식 `await self.server.model_loader.task`을 평가하여 이 단계의 처리를 수행합니다.
        await self.server.model_loader.task
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'output_mode': 'VISUAL'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'output_mode': 'VISUAL'}))
            # 함수 또는 처리 단계의 결과에 계산식 `await self.until(ws, lambda packet: packet['type'] == 'ERROR')`의 결과를 저장합니다. 대상: `result`.
            result = await self.until(ws, lambda packet: packet['type'] == 'ERROR')
            # 검증문 `self.assertEqual(result['code'], 'UNSUPPORTED_OUTPUT_MODE')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['code'], 'UNSUPPORTED_OUTPUT_MODE')
            # 검증문 `self.server.init_live_session.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.server.init_live_session.assert_not_called()

    # async 설정 up 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def asyncSetUp(self):
        # `self.gate` 값에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `self.gate`.
        self.gate = threading.Event()
        # `self.factory_calls` 값에 0을 저장합니다. 대상: `self.factory_calls`.
        self.factory_calls = 0
        # fail 모델에 False을 저장합니다. 대상: `self.fail_model`.
        self.fail_model = False
        # 생성 또는 검증할 분석 엔진에 `object()` 호출 결과를 저장합니다. 대상: `self.engine`.
        self.engine = object()

        # `factory` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def factory(report):
            # `self.factory_calls` 값에 `1`을 더해 누적합니다. 대상: `self.factory_calls`.
            self.factory_calls += 1
            # `report('LOADING_POSE')`을 호출하여 해당 단계의 작업을 수행합니다.
            report('LOADING_POSE')
            # 조건 `not self.gate.wait(3)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not self.gate.wait(3):
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Test gate timeout')` 예외를 발생시킵니다.
                raise RuntimeError('Test gate timeout')
            # 조건 `self.fail_model`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.fail_model:
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Missing test checkpoint')` 예외를 발생시킵니다.
                raise RuntimeError('Missing test checkpoint')
            # 현재 함수 처리를 끝내고 생성 또는 검증할 분석 엔진(`self.engine`)을 호출한 쪽으로 반환합니다.
            return self.engine

        # 서버에 `ExerciseSocketServer(host='127.0.0.1', port=0, engine_factory=factory)` 호출 결과를 저장합니다. 대상: `self.server`.
        self.server = ExerciseSocketServer(host='127.0.0.1', port=0, engine_factory=factory)
        # init 실시간 측정 세션에 `Mock(side_effect=self.init_fake_session)` 호출 결과를 저장합니다. 대상: `self.server.init_live_session`.
        self.server.init_live_session = Mock(side_effect=self.init_fake_session)
        # 중지 camera에 `Mock()` 호출 결과를 저장합니다. 대상: `self.server.stop_camera`.
        self.server.stop_camera = Mock()
        # 저장 측정 세션 파일에 `Mock()` 호출 결과를 저장합니다. 대상: `self.server._save_session_files`.
        self.server._save_session_files = Mock()
        # `self.task` 값에 별도로 진행하도록 등록한 비동기 작업(`asyncio.create_task(self.server.run())`)을 저장합니다. 대상: `self.task`.
        self.task = asyncio.create_task(self.server.run())
        # `range(200)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(200):
            # 조건 `self.server.bound_port is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.server.bound_port is not None:
                # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                break
            # `asyncio.sleep(0.01)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
            await asyncio.sleep(0.01)
        # 검증문 `self.assertIsNotNone(self.server.bound_port)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNotNone(self.server.bound_port)
        # WebSocket 접속 주소에 계산식 `f'ws://127.0.0.1:{self.server.bound_port}'`의 결과를 저장합니다. 대상: `self.uri`.
        self.uri = f'ws://127.0.0.1:{self.server.bound_port}'

    # `asyncTearDown` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def asyncTearDown(self):
        # `self.gate.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.gate.set()
        # 조건 `self.server.model_loader.task`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.server.model_loader.task:
            # 표현식 `await self.server.model_loader.task`을 평가하여 이 단계의 처리를 수행합니다.
            await self.server.model_loader.task
        # `self.task.cancel()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.task.cancel()
        # `asyncio.gather(self.task, return_exceptions=True)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await asyncio.gather(self.task, return_exceptions=True)

    # init fake 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def init_fake_session(self, packet):
        # 현재 작업 설정에 키 'exercise_name', 'eval_config', 'default_thresholds'로 구성한 사전을 저장합니다. 대상: `config`.
        config = {
            # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 'biceps_curl'을 지정합니다.
            'exercise_name': 'biceps_curl',
            # 구성 항목: eval 설정(`'eval_config'`)에 키 'metric_type', 'motion_direction'로 구성한 사전; 측정 방식(각도·좌표·길이·비율)(`'metric_type'`)에 'RELATIVE_Y'; 측정값 증가·감소 방향(`'motion_direction'`)에 'INCREASING'을 지정합니다.
            'eval_config': {'metric_type': 'RELATIVE_Y', 'motion_direction': 'INCREASING'},
            # 구성 항목: 기본 가동 범위 기준(`'default_thresholds'`)에 반복·조건식 `{side: {'start_val': 0.1, 'target_val': 0.9} for side in ('left', 'right')}`으로 구성한 값; 개인 가동 범위의 시작값(`'start_val'`)에 0.1; 개인 가동 범위의 목표값(`'target_val'`)에 0.9을 지정합니다.
            'default_thresholds': {side: {'start_val': 0.1, 'target_val': 0.9}
                                   # 앞줄에서 시작한 식·인수·자료 구조를 `for side in ('left', 'right')},` 내용으로 이어서 구성합니다.
                                   for side in ('left', 'right')},
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
        # `manager` 값에 `SimpleNamespace(session_log=SimpleNamespace(info={}))` 호출 결과를 저장합니다. 대상: `manager`.
        manager = SimpleNamespace(session_log=SimpleNamespace(info={}))
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(data_manager=manager, motion_engine=MotionEngine(config))` 호출 결과를 저장합니다. 대상: `self.server.controller`.
        self.server.controller = SessionController(data_manager=manager, motion_engine=MotionEngine(config))

    # `until` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def until(self, ws, predicate):
        # `range(20)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(20):
            # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(await asyncio.wait_for(ws.recv(), 2))`)을 저장합니다. 대상: `packet`.
            packet = json.loads(await asyncio.wait_for(ws.recv(), 2))
            # 조건 `predicate(packet)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if predicate(packet):
                # 현재 함수 처리를 끝내고 수신 또는 전송할 JSON 메시지 사전(`packet`)을 호출한 쪽으로 반환합니다.
                return packet
        # `self.fail('Expected packet did not arrive')`을 호출하여 해당 단계의 작업을 수행합니다.
        self.fail('Expected packet did not arrive')

    # 전송 시작 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def send_start(self, ws):
        # `ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'mode': 'MAIN', 'exercise_name': 'biceps_curl', 'target_reps': 40}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'mode': 'MAIN',
                                  # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 'biceps_curl'; 목표 반복 횟수(`'target_reps'`)에 40을 지정합니다.
                                  'exercise_name': 'biceps_curl', 'target_reps': 40}))

    # 회귀 테스트: `test_listening_and_status_queries_work_while_models_are_blocked` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_listening_and_status_queries_work_while_models_are_blocked(self):
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `greeting` 값에 계산식 `await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')`의 결과를 저장합니다. 대상: `greeting`.
            greeting = await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')
            # 검증문 `self.assertFalse(greeting['ready'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertFalse(greeting['ready'])
            # 검증문 `self.assertFalse(self.gate.is_set())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertFalse(self.gate.is_set())
            # `ws.send(json.dumps({'type': 'CMD_GET_SERVER_STATUS'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await ws.send(json.dumps({'type': 'CMD_GET_SERVER_STATUS'}))
            # 상태에 계산식 `await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')`의 결과를 저장합니다. 대상: `status`.
            status = await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')
            # 검증문 `self.assertEqual(status['model_state'], 'LOADING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(status['model_state'], 'LOADING')
            # 검증문 `self.server.init_live_session.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.server.init_live_session.assert_not_called()

    # 회귀 테스트: `test_slow_phone_camera_open_does_not_block_status_connections` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_slow_phone_camera_open_does_not_block_status_connections(self):
        # `self.gate.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.gate.set()
        # 표현식 `await self.server.model_loader.task`을 평가하여 이 단계의 처리를 수행합니다.
        await self.server.model_loader.task
        # `(opening, finish_open)` 값에 순서·구성을 지정한 `(threading.Event(), threading.Event())`을 저장합니다. 대상: `(opening, finish_open)`.
        opening, finish_open = threading.Event(), threading.Event()

        # `slow_open` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def slow_open(packet):
            # `opening.set()`을 호출하여 스레드 신호를 활성화합니다.
            opening.set()
            # 조건 `not finish_open.wait(3)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not finish_open.wait(3):
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Camera test gate timeout')` 예외를 발생시킵니다.
                raise RuntimeError('Camera test gate timeout')
            # `self.init_fake_session(packet)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.init_fake_session(packet)

        # init 실시간 측정 세션에 `Mock(side_effect=slow_open)` 호출 결과를 저장합니다. 대상: `self.server.init_live_session`.
        self.server.init_live_session = Mock(side_effect=slow_open)
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `self.send_start(ws)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.send_start(ws)
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `range(100)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
                for _ in range(100):
                    # 조건 `opening.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if opening.is_set():
                        # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                        break
                    # `asyncio.sleep(0.01)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
                    await asyncio.sleep(0.01)
                # 검증문 `self.assertTrue(opening.is_set())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertTrue(opening.is_set())
                # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                async with websockets.connect(self.uri) as monitor:
                    # 상태에 계산식 `await self.until(monitor, lambda packet: packet['type'] == 'SERVER_STATUS')`의 결과를 저장합니다. 대상: `status`.
                    status = await self.until(monitor, lambda packet: packet['type'] == 'SERVER_STATUS')
                    # 검증문 `self.assertTrue(status['ready'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertTrue(status['ready'])
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # `finish_open.set()`을 호출하여 스레드 신호를 활성화합니다.
                finish_open.set()
            # `self.until(ws, lambda packet: packet['type'] == 'SESSION_LOG_STARTED')`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.until(ws, lambda packet: packet['type'] == 'SESSION_LOG_STARTED')

    # 회귀 테스트: `test_session_and_initial_rom_query_wait_then_start_once` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_session_and_initial_rom_query_wait_then_start_once(self):
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `self.send_start(ws)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.send_start(ws)
            # `ws.send(json.dumps({'type': 'CMD_GET_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl', 'request_id': 'initial-rom'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await ws.send(json.dumps({'type': 'CMD_GET_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl',
                                      # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 'initial-rom'을 지정합니다.
                                      'request_id': 'initial-rom'}))
            # `self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')
            # 검증문 `self.server.init_live_session.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.server.init_live_session.assert_not_called()
            # `self.gate.set()`을 호출하여 스레드 신호를 활성화합니다.
            self.gate.set()
            # 함수 또는 처리 단계의 결과에 계산식 `await self.until(ws, lambda packet: packet['type'] == 'COUNT_THRESHOLD_STATUS')`의 결과를 저장합니다. 대상: `result`.
            result = await self.until(ws, lambda packet: packet['type'] == 'COUNT_THRESHOLD_STATUS')
            # 검증문 `self.assertEqual(result['request_id'], 'initial-rom')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['request_id'], 'initial-rom')
            # 검증문 `self.assertEqual(result['thresholds']['left']['target_val'], 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['thresholds']['left']['target_val'], 0.9)
            # 검증문 `self.server.init_live_session.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.server.init_live_session.assert_called_once()
            # 검증문 `self.assertIs(self.server.skeleton_engine, self.engine)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertIs(self.server.skeleton_engine, self.engine)

    # 회귀 테스트: `test_multiple_connections_and_sessions_load_models_only_once` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_multiple_connections_and_sessions_load_models_only_once(self):
        # `self.gate.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.gate.set()
        # 표현식 `await self.server.model_loader.task`을 평가하여 이 단계의 처리를 수행합니다.
        await self.server.model_loader.task
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `range(2)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
            for _ in range(2):
                # `self.send_start(ws)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await self.send_start(ws)
                # `ws.send(json.dumps({'type': 'CMD_GET_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await ws.send(json.dumps({'type': 'CMD_GET_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl'}))
                # `self.until(ws, lambda packet: packet['type'] == 'COUNT_THRESHOLD_STATUS')`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await self.until(ws, lambda packet: packet['type'] == 'COUNT_THRESHOLD_STATUS')
            # stops 이전 monitor에 call 횟수(`self.server.stop_camera.call_count`)을 저장합니다. 대상: `stops_before_monitor`.
            stops_before_monitor = self.server.stop_camera.call_count
            # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            async with websockets.connect(self.uri) as monitor:
                # `self.until(monitor, lambda packet: packet['type'] == 'SERVER_STATUS')`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await self.until(monitor, lambda packet: packet['type'] == 'SERVER_STATUS')
            # 상태 조회 전용 연결을 닫아도 다른 측정 연결의 카메라는 닫히지 않아야 합니다.
            # `asyncio.sleep(0.03)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
            await asyncio.sleep(0.03)
            # 검증문 `self.assertEqual(self.server.stop_camera.call_count, stops_before_monitor)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(self.server.stop_camera.call_count, stops_before_monitor)
            # 검증문 `self.assertEqual(self.server.init_live_session.call_count, 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(self.server.init_live_session.call_count, 2)
        # 검증문 `self.assertEqual(self.factory_calls, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.factory_calls, 1)

    # 회귀 테스트: `test_disconnecting_while_pending_does_not_open_camera_later` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_disconnecting_while_pending_does_not_open_camera_later(self):
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `self.send_start(ws)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.send_start(ws)
            # `self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')
        # `self.gate.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.gate.set()
        # 표현식 `await self.server.model_loader.task`을 평가하여 이 단계의 처리를 수행합니다.
        await self.server.model_loader.task
        # `asyncio.sleep(0.03)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
        await asyncio.sleep(0.03)
        # 검증문 `self.server.init_live_session.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server.init_live_session.assert_not_called()

    # 회귀 테스트: `test_model_failure_is_reported_without_closing_listener` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_model_failure_is_reported_without_closing_listener(self):
        # fail 모델에 True을 저장합니다. 대상: `self.fail_model`.
        self.fail_model = True
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `self.send_start(ws)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.send_start(ws)
            # `self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')
            # `self.gate.set()`을 호출하여 스레드 신호를 활성화합니다.
            self.gate.set()
            # 함수 또는 처리 단계의 결과에 계산식 `await self.until(ws, lambda packet: packet['type'] == 'ERROR')`의 결과를 저장합니다. 대상: `result`.
            result = await self.until(ws, lambda packet: packet['type'] == 'ERROR')
            # 검증문 `self.assertEqual(result['code'], 'MODEL_LOAD_FAILED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['code'], 'MODEL_LOAD_FAILED')
            # 검증문 `self.assertTrue(result['fatal'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(result['fatal'])
            # `ws.send(json.dumps({'type': 'CMD_GET_SERVER_STATUS'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await ws.send(json.dumps({'type': 'CMD_GET_SERVER_STATUS'}))
            # 상태에 계산식 `await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')`의 결과를 저장합니다. 대상: `status`.
            status = await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')
            # 검증문 `self.assertEqual(status['model_state'], 'FAILED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(status['model_state'], 'FAILED')
            # 검증문 `self.assertIn('Missing test checkpoint', status['model_error'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertIn('Missing test checkpoint', status['model_error'])


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
