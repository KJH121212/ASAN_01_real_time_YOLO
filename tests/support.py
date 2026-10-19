"""Local protocol server for client/UI tests, without cameras or AI models."""

import asyncio
import json
from pathlib import Path
import threading
import time

import websockets

from core.motion_engine import MotionEngine
from core.session_controller import SessionController
from network.session_commands import handle_count_threshold_command


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for client/server state")


class MockSessionServer:
    def __init__(self):
        self.ready = threading.Event()
        self.disconnected = threading.Event()
        self.model_ready = threading.Event()
        self.model_ready.set()
        self.connections = 0
        self.starts = 0
        self.packets = []
        self.clients = set()
        self.thread = threading.Thread(target=self._worker, daemon=True)

    def start(self):
        self.thread.start()
        if not self.ready.wait(3):
            raise AssertionError("Mock server failed to start")
        return self

    def stop(self):
        self.model_ready.set()
        self.loop.call_soon_threadsafe(self.shutdown.set)
        self.thread.join(timeout=3)

    def send(self, packet):
        async def broadcast():
            for client in tuple(self.clients):
                await client.send(json.dumps(packet))
        asyncio.run_coroutine_threadsafe(broadcast(), self.loop).result(timeout=3)

    def _worker(self):
        asyncio.run(self._run())

    async def _run(self):
        self.loop = asyncio.get_running_loop()
        self.shutdown = asyncio.Event()
        async with websockets.serve(self._handle, "127.0.0.1", 0) as server:
            self.uri = f"ws://127.0.0.1:{server.sockets[0].getsockname()[1]}"
            self.ready.set()
            await self.shutdown.wait()

    async def _handle(self, ws):
        self.connections += 1
        self.clients.add(ws)
        controller = None
        config = {}
        try:
            if not self.model_ready.is_set():
                await ws.send(json.dumps({'type': 'SERVER_STATUS', 'model_state': 'LOADING',
                    'phase': 'LOADING_POSE', 'ready': False, 'elapsed_seconds': 0.0,
                    'message': '관절 추출 모델을 불러오는 중입니다.'}))
            async for message in ws:
                packet = json.loads(message)
                self.packets.append(packet)
                if packet['type'] == 'CMD_SET_SESSION':
                    if not self.model_ready.is_set():
                        while not self.model_ready.is_set():
                            await asyncio.sleep(0.01)
                        await ws.send(json.dumps({'type': 'SERVER_STATUS', 'model_state': 'READY',
                            'phase': 'READY', 'ready': True, 'elapsed_seconds': 0.1,
                            'message': '모델 준비가 완료되었습니다.'}))
                    self.starts += 1
                    config = packet
                    path = Path(__file__).resolve().parents[1] / 'configs/exercise_kpt_config_coco.json'
                    exercise = json.loads(path.read_text(encoding='utf-8'))['exercises'][config['exercise_name']]
                    engine = MotionEngine(exercise, mode=config['mode'], target_reps=config.get('target_reps', 10))
                    engine.fsm_left.rep_count = engine.fsm_right.rep_count = 2
                    controller = SessionController(motion_engine=engine)
                    if config.get('log_root'):
                        directory = (Path(config['log_root']) / config.get('player_id', 'patient_1')
                                     / '261019' / config['exercise_name'])
                        stem = f"session{config.get('session', 1)}_round{config.get('round', 1)}_set{config.get('set', 1)}"
                        await ws.send(json.dumps({'type': 'SESSION_LOG_STARTED', 'directory': str(directory),
                            'frame_csv': str(directory / f'{stem}.csv'),
                            'skeleton_npz': str(directory / f'{stem}.npz'),
                            'video_mp4': str(directory / f'{stem}.mp4') if config.get('record_video') else None}))
                elif packet['type'] == 'CMD_STOP_SESSION':
                    await ws.send(json.dumps({'type': 'SESSION_FINISHED', 'request_id': packet.get('request_id'),
                        'status': 'COMPLETED' if packet.get('reason') == 'TARGET_REACHED' else 'STOPPED',
                        'reason': packet.get('reason'), 'summary': {'saved_reps_count': 2}}))
                    controller = None
                else:
                    response = handle_count_threshold_command(controller, packet)
                    await ws.send(json.dumps(response))
                if controller:
                    engine = controller.motion_engine
                    await ws.send(json.dumps({
                        'type': 'POSE_UPDATE', 'mode': config['mode'], 'fps': 30,
                        'left': {**engine.fsm_left.update(None), 'val': 0.1},
                        'right': {**engine.fsm_right.update(None), 'val': 0.1},
                        'is_occluded': False, **engine.count_threshold_status(),
                    }))
        finally:
            self.clients.discard(ws)
            self.disconnected.set()
