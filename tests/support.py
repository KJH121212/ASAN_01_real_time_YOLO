# 한국어 코드 해설: `support`에서 사용하는 실행·분석 보조 기능을 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Local protocol server for client/UI tests, without cameras or AI models."""

# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time

# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `SessionController` 기능을 가져옵니다.
from core.session_controller import SessionController
# `network.session_commands`에서 `handle_count_threshold_command` 기능을 가져옵니다.
from network.session_commands import handle_count_threshold_command


# `wait_for` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def wait_for(predicate, timeout=3):
    # `deadline` 값에 계산식 `time.monotonic() + timeout`의 결과를 저장합니다. 대상: `deadline`.
    deadline = time.monotonic() + timeout
    # 조건 `time.monotonic() < deadline`이 유지되는 동안 아래 처리를 반복합니다.
    while time.monotonic() < deadline:
        # 조건 `predicate()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if predicate():
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return
        # `time.sleep(0.01)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다.
        time.sleep(0.01)
    # 이 조건에서는 처리를 계속할 수 없으므로 `AssertionError('Timed out waiting for client/server state')` 예외를 발생시킵니다.
    raise AssertionError("Timed out waiting for client/server state")


# `MockSessionServer` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class MockSessionServer:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self):
        # 준비 상태에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `self.ready`.
        self.ready = threading.Event()
        # `self.disconnected` 값에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `self.disconnected`.
        self.disconnected = threading.Event()
        # 모델 준비 상태에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `self.model_ready`.
        self.model_ready = threading.Event()
        # `self.model_ready.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.model_ready.set()
        # `self.connections` 값에 0을 저장합니다. 대상: `self.connections`.
        self.connections = 0
        # `self.starts` 값에 0을 저장합니다. 대상: `self.starts`.
        self.starts = 0
        # 메시지에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.packets`.
        self.packets = []
        # `self.clients` 값에 `set()` 호출 결과를 저장합니다. 대상: `self.clients`.
        self.clients = set()
        # `self.thread` 값에 백그라운드 실행 스레드(`threading.Thread(target=self._worker, daemon=True)`)을 저장합니다. 대상: `self.thread`.
        self.thread = threading.Thread(target=self._worker, daemon=True)

    # 시작 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start(self):
        # `self.thread.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.thread.start()
        # 조건 `not self.ready.wait(3)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.ready.wait(3):
            # 이 조건에서는 처리를 계속할 수 없으므로 `AssertionError('Mock server failed to start')` 예외를 발생시킵니다.
            raise AssertionError("Mock server failed to start")
        # 현재 함수 처리를 끝내고 현재 객체(`self`)을 호출한 쪽으로 반환합니다.
        return self

    # 중지 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def stop(self):
        # `self.model_ready.set()`을 호출하여 스레드 신호를 활성화합니다.
        self.model_ready.set()
        # `self.loop.call_soon_threadsafe(self.shutdown.set)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.loop.call_soon_threadsafe(self.shutdown.set)
        # `self.thread.join(timeout=3)`을 호출하여 백그라운드 스레드의 종료를 기다림합니다.
        self.thread.join(timeout=3)

    # 전송 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def send(self, packet):
        # `broadcast` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        async def broadcast():
            # `tuple(self.clients)`에서 항목을 차례로 꺼내 `client`에 넣고 아래 처리를 반복합니다.
            for client in tuple(self.clients):
                # `client.send(json.dumps(packet))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await client.send(json.dumps(packet))
        # `asyncio.run_coroutine_threadsafe(broadcast(), self.loop).result(timeout=3)`을 호출하여 해당 단계의 작업을 수행합니다.
        asyncio.run_coroutine_threadsafe(broadcast(), self.loop).result(timeout=3)

    # 작업 스레드 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _worker(self):
        # `asyncio.run(self._run())`을 호출하여 해당 단계의 작업을 수행합니다.
        asyncio.run(self._run())

    # 실행 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _run(self):
        # `self.loop` 값에 `asyncio.get_running_loop()` 호출 결과를 저장합니다. 대상: `self.loop`.
        self.loop = asyncio.get_running_loop()
        # `self.shutdown` 값에 `asyncio.Event()` 호출 결과를 저장합니다. 대상: `self.shutdown`.
        self.shutdown = asyncio.Event()
        # `websockets.serve(self._handle, '127.0.0.1', 0)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.serve(self._handle, "127.0.0.1", 0) as server:
            # WebSocket 접속 주소에 계산식 `f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'`의 결과를 저장합니다. 대상: `self.uri`.
            self.uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            # `self.ready.set()`을 호출하여 스레드 신호를 활성화합니다.
            self.ready.set()
            # `self.shutdown.wait()`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.shutdown.wait()

    # 요청 처리 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _handle(self, ws):
        # `self.connections` 값에 `1`을 더해 누적합니다. 대상: `self.connections`.
        self.connections += 1
        # `self.clients.add(ws)`을 호출하여 집합에 항목을 등록합니다.
        self.clients.add(ws)
        # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `controller`.
        controller = None
        # 현재 작업 설정에 빈 사전을 저장합니다. 대상: `config`.
        config = {}
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 조건 `not self.model_ready.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not self.model_ready.is_set():
                # `ws.send(json.dumps({'type': 'SERVER_STATUS', 'model_state': 'LOADING', 'phase': 'LOADING_POSE', 'ready': False, 'elapsed_seconds': 0.0, 'message': '관절 추출 모델을 불러오는 중입니다.'…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await ws.send(json.dumps({'type': 'SERVER_STATUS', 'model_state': 'LOADING',
                    # 구성 항목: `phase` 값(`'phase'`)에 'LOADING_POSE'; 준비 상태(`'ready'`)에 False; elapsed 초(`'elapsed_seconds'`)에 0.0을 지정합니다.
                    'phase': 'LOADING_POSE', 'ready': False, 'elapsed_seconds': 0.0,
                    # 구성 항목: 메시지(`'message'`)에 '관절 추출 모델을 불러오는 중입니다.'을 지정합니다.
                    'message': '관절 추출 모델을 불러오는 중입니다.'}))
            # `ws`에서 항목을 차례로 꺼내 `message`에 넣고 아래 처리를 반복합니다.
            async for message in ws:
                # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(message)`)을 저장합니다. 대상: `packet`.
                packet = json.loads(message)
                # `self.packets.append(packet)`을 호출하여 항목 하나를 끝에 추가합니다.
                self.packets.append(packet)
                # 조건 `packet['type'] == 'CMD_SET_SESSION'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet['type'] == 'CMD_SET_SESSION':
                    # 조건 `not self.model_ready.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if not self.model_ready.is_set():
                        # 조건 `not self.model_ready.is_set()`이 유지되는 동안 아래 처리를 반복합니다.
                        while not self.model_ready.is_set():
                            # `asyncio.sleep(0.01)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
                            await asyncio.sleep(0.01)
                        # `ws.send(json.dumps({'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True, 'elapsed_seconds': 0.1, 'message': '모델 준비가 완료되었습니다.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                        await ws.send(json.dumps({'type': 'SERVER_STATUS', 'model_state': 'READY',
                            # 구성 항목: `phase` 값(`'phase'`)에 'READY'; 준비 상태(`'ready'`)에 True; elapsed 초(`'elapsed_seconds'`)에 0.1을 지정합니다.
                            'phase': 'READY', 'ready': True, 'elapsed_seconds': 0.1,
                            # 구성 항목: 메시지(`'message'`)에 '모델 준비가 완료되었습니다.'을 지정합니다.
                            'message': '모델 준비가 완료되었습니다.'}))
                    # `self.starts` 값에 `1`을 더해 누적합니다. 대상: `self.starts`.
                    self.starts += 1
                    # 현재 작업 설정에 수신 또는 전송할 JSON 메시지 사전(`packet`)을 저장합니다. 대상: `config`.
                    config = packet
                    # 경로에 계산식 `Path(__file__).resolve().parents[1] / 'configs/exercise_kpt_config_coco.json'`의 결과를 저장합니다. 대상: `path`.
                    path = Path(__file__).resolve().parents[1] / 'configs/exercise_kpt_config_coco.json'
                    # `exercise` 값에 계산식 `json.loads(path.read_text(encoding='utf-8'))['exercises'][config['exercise_name']]`의 결과를 저장합니다. 대상: `exercise`.
                    exercise = json.loads(path.read_text(encoding='utf-8'))['exercises'][config['exercise_name']]
                    # 생성 또는 검증할 분석 엔진에 `MotionEngine(exercise, mode=config['mode'], target_reps=config.get('target_reps', 10))` 호출 결과를 저장합니다. 대상: `engine`.
                    engine = MotionEngine(exercise, mode=config['mode'], target_reps=config.get('target_reps', 10))
                    # 인정된 누적 반복 횟수에 2을 저장합니다. 대상: `engine.fsm_left.rep_count, engine.fsm_right.rep_count`.
                    engine.fsm_left.rep_count = engine.fsm_right.rep_count = 2
                    # 캘리브레이션·본 운동 진행 제어기에 `SessionController(motion_engine=engine)` 호출 결과를 저장합니다. 대상: `controller`.
                    controller = SessionController(motion_engine=engine)
                    # 조건 `config.get('log_root')`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if config.get('log_root'):
                        # 폴더에 계산식 `Path(config['log_root']) / config.get('player_id', 'patient_1') / '261019' / config['exercise_name']`의 결과를 저장합니다. 대상: `directory`.
                        directory = (Path(config['log_root']) / config.get('player_id', 'patient_1')
                                     # 앞줄에서 시작한 식·인수·자료 구조를 `/ '261019' / config['exercise_name'])` 내용으로 이어서 구성합니다.
                                     / '261019' / config['exercise_name'])
                        # `stem` 값에 계산식 `f"session{config.get('session', 1)}_round{config.get('round', 1)}_set{config.get('set', 1)}"`의 결과를 저장합니다. 대상: `stem`.
                        stem = f"session{config.get('session', 1)}_round{config.get('round', 1)}_set{config.get('set', 1)}"
                        # `ws.send(json.dumps({'type': 'SESSION_LOG_STARTED', 'directory': str(directory), 'frame_csv': str(directory / f'{stem}.csv'), 'skeleton_npz': str(directory / f'{stem}.npz…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                        await ws.send(json.dumps({'type': 'SESSION_LOG_STARTED', 'directory': str(directory),
                            # 구성 항목: 프레임 csv(`'frame_csv'`)에 `str(directory / f'{stem}.csv')` 호출 결과를 지정합니다.
                            'frame_csv': str(directory / f'{stem}.csv'),
                            # 구성 항목: `skeleton_npz` 값(`'skeleton_npz'`)에 `str(directory / f'{stem}.npz')` 호출 결과를 지정합니다.
                            'skeleton_npz': str(directory / f'{stem}.npz'),
                            # 구성 항목: 영상 mp4(`'video_mp4'`)에 `config.get('record_video')`이면 `str(directory / f'{stem}.mp4')`, 아니면 `None`을 지정합니다.
                            'video_mp4': str(directory / f'{stem}.mp4') if config.get('record_video') else None}))
                # 조건 `packet['type'] == 'CMD_STOP_SESSION'`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif packet['type'] == 'CMD_STOP_SESSION':
                    # `ws.send(json.dumps({'type': 'SESSION_FINISHED', 'request_id': packet.get('request_id'), 'status': 'COMPLETED' if packet.get('reason') == 'TARGET_REACHED' else 'STOPPED',…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await ws.send(json.dumps({'type': 'SESSION_FINISHED', 'request_id': packet.get('request_id'),
                        # 구성 항목: 상태(`'status'`)에 `packet.get('reason') == 'TARGET_REACHED'`이면 `'COMPLETED'`, 아니면 `'STOPPED'`을 지정합니다.
                        'status': 'COMPLETED' if packet.get('reason') == 'TARGET_REACHED' else 'STOPPED',
                        # 구성 항목: `reason` 값(`'reason'`)에 `packet.get('reason')` 호출 결과; 저장·종료 결과 요약(`'summary'`)에 키 'saved_reps_count'로 구성한 사전; saved 반복 횟수 횟수(`'saved_reps_count'`)에 2을 지정합니다.
                        'reason': packet.get('reason'), 'summary': {'saved_reps_count': 2}}))
                    # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `controller`.
                    controller = None
                # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                else:
                    # `response` 값에 `handle_count_threshold_command(controller, packet)` 호출 결과를 저장합니다. 대상: `response`.
                    response = handle_count_threshold_command(controller, packet)
                    # `ws.send(json.dumps(response))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await ws.send(json.dumps(response))
                # 조건 `controller`을 만족하는 경우에만 아래 분기를 실행합니다.
                if controller:
                    # 생성 또는 검증할 분석 엔진에 관절 측정값 계산 및 반복 판정 엔진(`controller.motion_engine`)을 저장합니다. 대상: `engine`.
                    engine = controller.motion_engine
                    # `ws.send(json.dumps({'type': 'POSE_UPDATE', 'mode': config['mode'], 'fps': 30, 'left': {**engine.fsm_left.update(None), 'val': 0.1}, 'right': {**engine.fsm_right.update(N…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await ws.send(json.dumps({
                        # 구성 항목: `type` 값(`'type'`)에 'POSE_UPDATE'; 실행 모드(`'mode'`)에 계산식 `config['mode']`의 결과; 영상의 초당 프레임 수(`'fps'`)에 30을 지정합니다.
                        'type': 'POSE_UPDATE', 'mode': config['mode'], 'fps': 30,
                        # 구성 항목: 왼쪽(`'left'`)에 키 'val'로 구성한 사전; `val` 값(`'val'`)에 0.1을 지정합니다.
                        'left': {**engine.fsm_left.update(None), 'val': 0.1},
                        # 구성 항목: 오른쪽(`'right'`)에 키 'val'로 구성한 사전; `val` 값(`'val'`)에 0.1을 지정합니다.
                        'right': {**engine.fsm_right.update(None), 'val': 0.1},
                        # 구성 항목: 필수 관절이 가려졌거나 유효하지 않은지 여부(`'is_occluded'`)에 False을 지정합니다.
                        'is_occluded': False, **engine.count_threshold_status(),
                    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                    }))
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `self.clients.discard(ws)`을 호출하여 집합에서 항목을 제거합니다.
            self.clients.discard(ws)
            # `self.disconnected.set()`을 호출하여 스레드 신호를 활성화합니다.
            self.disconnected.set()
