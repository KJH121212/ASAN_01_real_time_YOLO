# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Real WebSocket/JSON round trips for the shared command handler; no GPU/camera."""

# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `SessionController` 기능을 가져옵니다.
from core.session_controller import SessionController
# `network.session_commands`에서 `handle_count_threshold_command` 기능을 가져옵니다.
from network.session_commands import handle_count_threshold_command


# `ThresholdSocketTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ThresholdSocketTests(unittest.IsolatedAsyncioTestCase):
    # async 설정 up 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def asyncSetUp(self):
        # 생성 또는 검증할 분석 엔진에 `MotionEngine({'exercise_name': 'biceps_curl', 'eval_config': {'metric_type': 'RELATIVE_Y', 'motion_direction': 'INCREASING'}, 'default_thresholds': {side: {'start_val': …` 호출 결과를 저장합니다. 대상: `self.engine`.
        self.engine = MotionEngine({
            # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 'biceps_curl'을 지정합니다.
            'exercise_name': 'biceps_curl',
            # 구성 항목: eval 설정(`'eval_config'`)에 키 'metric_type', 'motion_direction'로 구성한 사전; 측정 방식(각도·좌표·길이·비율)(`'metric_type'`)에 'RELATIVE_Y'; 측정값 증가·감소 방향(`'motion_direction'`)에 'INCREASING'을 지정합니다.
            'eval_config': {'metric_type': 'RELATIVE_Y', 'motion_direction': 'INCREASING'},
            # 구성 항목: 기본 가동 범위 기준(`'default_thresholds'`)에 반복·조건식 `{side: {'start_val': 0.1, 'target_val': 0.9} for side in ('left', 'right')}`으로 구성한 값; 개인 가동 범위의 시작값(`'start_val'`)에 0.1; 개인 가동 범위의 목표값(`'target_val'`)에 0.9을 지정합니다.
            'default_thresholds': {side: {'start_val': 0.1, 'target_val': 0.9}
                                   # 앞줄에서 시작한 식·인수·자료 구조를 `for side in ('left', 'right')},` 내용으로 이어서 구성합니다.
                                   for side in ('left', 'right')},
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        })
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(motion_engine=self.engine)` 호출 결과를 저장합니다. 대상: `self.controller`.
        self.controller = SessionController(motion_engine=self.engine)

        # 요청 처리 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        async def handle(ws):
            # `ws`에서 항목을 차례로 꺼내 `message`에 넣고 아래 처리를 반복합니다.
            async for message in ws:
                # `response` 값에 `handle_count_threshold_command(self.controller, json.loads(message))` 호출 결과를 저장합니다. 대상: `response`.
                response = handle_count_threshold_command(self.controller, json.loads(message))
                # `ws.send(json.dumps(response))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await ws.send(json.dumps(response))

        # 서버에 계산식 `await websockets.serve(handle, '127.0.0.1', 0)`의 결과를 저장합니다. 대상: `self.server`.
        self.server = await websockets.serve(handle, '127.0.0.1', 0)
        # WebSocket 접속 주소에 계산식 `f'ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}'`의 결과를 저장합니다. 대상: `self.uri`.
        self.uri = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    # `asyncTearDown` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def asyncTearDown(self):
        # `self.server.close()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.server.close()
        # `self.server.wait_closed()`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.wait_closed()

    # `exchange` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def exchange(self, ws, **overrides):
        # 수신 또는 전송할 JSON 메시지 사전에 키 'type', 'exercise_name', 'request_id', 'thresholds'로 구성한 사전을 저장합니다. 대상: `packet`.
        packet = {'type': 'CMD_UPDATE_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl',
                  # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 'socket-test'; 좌우 가동 범위 시작값·목표값(`'thresholds'`)에 키 'left'로 구성한 사전; 왼쪽(`'left'`)에 키 'target_val'로 구성한 사전; 개인 가동 범위의 목표값(`'target_val'`)에 0.5을 지정합니다.
                  'request_id': 'socket-test', 'thresholds': {'left': {'target_val': 0.5}}}
        # `packet.update(overrides)`을 호출하여 해당 단계의 작업을 수행합니다.
        packet.update(overrides)
        # `ws.send(json.dumps(packet))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await ws.send(json.dumps(packet))
        # 현재 함수 처리를 끝내고 JSON 문자열을 해석한 Python 값(`json.loads(await asyncio.wait_for(ws.recv(), timeout=2))`)을 호출한 쪽으로 반환합니다.
        return json.loads(await asyncio.wait_for(ws.recv(), timeout=2))

    # 회귀 테스트: `test_query_update_and_recovery_after_error_on_same_connection` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_query_update_and_recovery_after_error_on_same_connection(self):
        # `websockets.connect(self.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri) as ws:
            # `initial` 값에 계산식 `await self.exchange(ws, type='CMD_GET_COUNT_THRESHOLD')`의 결과를 저장합니다. 대상: `initial`.
            initial = await self.exchange(ws, type='CMD_GET_COUNT_THRESHOLD')
            # 검증문 `self.assertEqual(initial['thresholds']['left']['target_val'], 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(initial['thresholds']['left']['target_val'], 0.9)
            # 검증문 `self.assertEqual(initial['threshold_version'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(initial['threshold_version'], 0)

            # `rejected` 값에 계산식 `await self.exchange(ws, thresholds={'left': {'target_val': 0.1}})`의 결과를 저장합니다. 대상: `rejected`.
            rejected = await self.exchange(ws, thresholds={'left': {'target_val': 0.1}})
            # 검증문 `self.assertEqual(rejected['code'], 'INVALID_THRESHOLD')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(rejected['code'], 'INVALID_THRESHOLD')

            # `changed` 값에 계산식 `await self.exchange(ws)`의 결과를 저장합니다. 대상: `changed`.
            changed = await self.exchange(ws)
            # 검증문 `self.assertEqual(changed['type'], 'COUNT_THRESHOLD_UPDATED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(changed['type'], 'COUNT_THRESHOLD_UPDATED')
            # 검증문 `self.assertEqual(changed['request_id'], 'socket-test')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(changed['request_id'], 'socket-test')
            # 검증문 `self.assertEqual(changed['states']['left'], 'RETURNING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(changed['states']['left'], 'RETURNING')
            # 검증문 `self.assertEqual(changed['thresholds']['right']['target_val'], 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(changed['thresholds']['right']['target_val'], 0.9)

            # `final` 값에 계산식 `await self.exchange(ws, type='CMD_GET_COUNT_THRESHOLD')`의 결과를 저장합니다. 대상: `final`.
            final = await self.exchange(ws, type='CMD_GET_COUNT_THRESHOLD')
            # 검증문 `self.assertEqual(final['thresholds']['left']['target_val'], 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(final['thresholds']['left']['target_val'], 0.5)
            # 검증문 `self.assertEqual(final['threshold_version'], 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(final['threshold_version'], 1)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
