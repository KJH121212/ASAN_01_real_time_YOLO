import asyncio
import json
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import websockets

from core.motion_engine import MotionEngine
from core.session_controller import SessionController
from network.socket_server import ExerciseSocketServer


class ServerStartupTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.gate = threading.Event()
        self.factory_calls = 0
        self.fail_model = False
        self.engine = object()

        def factory(report):
            self.factory_calls += 1
            report('LOADING_POSE')
            if not self.gate.wait(3):
                raise RuntimeError('Test gate timeout')
            if self.fail_model:
                raise RuntimeError('Missing test checkpoint')
            return self.engine

        self.server = ExerciseSocketServer(host='127.0.0.1', port=0, engine_factory=factory)
        self.server.init_live_session = Mock(side_effect=self.init_fake_session)
        self.server.stop_camera = Mock()
        self.server._save_session_files = Mock()
        self.task = asyncio.create_task(self.server.run())
        for _ in range(200):
            if self.server.bound_port is not None:
                break
            await asyncio.sleep(0.01)
        self.assertIsNotNone(self.server.bound_port)
        self.uri = f'ws://127.0.0.1:{self.server.bound_port}'

    async def asyncTearDown(self):
        self.gate.set()
        if self.server.model_loader.task:
            await self.server.model_loader.task
        self.task.cancel()
        await asyncio.gather(self.task, return_exceptions=True)

    def init_fake_session(self, packet):
        config = {
            'exercise_name': 'biceps_curl',
            'eval_config': {'metric_type': 'RELATIVE_Y', 'motion_direction': 'INCREASING'},
            'default_thresholds': {side: {'start_val': 0.1, 'target_val': 0.9}
                                   for side in ('left', 'right')},
        }
        manager = SimpleNamespace(session_log=SimpleNamespace(info={}))
        self.server.controller = SessionController(data_manager=manager, motion_engine=MotionEngine(config))

    async def until(self, ws, predicate):
        for _ in range(20):
            packet = json.loads(await asyncio.wait_for(ws.recv(), 2))
            if predicate(packet):
                return packet
        self.fail('Expected packet did not arrive')

    async def send_start(self, ws):
        await ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'mode': 'MAIN',
                                  'exercise_name': 'biceps_curl', 'target_reps': 40}))

    async def test_listening_and_status_queries_work_while_models_are_blocked(self):
        async with websockets.connect(self.uri) as ws:
            greeting = await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')
            self.assertFalse(greeting['ready'])
            self.assertFalse(self.gate.is_set())
            await ws.send(json.dumps({'type': 'CMD_GET_SERVER_STATUS'}))
            status = await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')
            self.assertEqual(status['model_state'], 'LOADING')
            self.server.init_live_session.assert_not_called()

    async def test_session_and_initial_rom_query_wait_then_start_once(self):
        async with websockets.connect(self.uri) as ws:
            await self.send_start(ws)
            await ws.send(json.dumps({'type': 'CMD_GET_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl',
                                      'request_id': 'initial-rom'}))
            await self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')
            self.server.init_live_session.assert_not_called()
            self.gate.set()
            result = await self.until(ws, lambda packet: packet['type'] == 'COUNT_THRESHOLD_STATUS')
            self.assertEqual(result['request_id'], 'initial-rom')
            self.assertEqual(result['thresholds']['left']['target_val'], 0.9)
            self.server.init_live_session.assert_called_once()
            self.assertIs(self.server.skeleton_engine, self.engine)

    async def test_multiple_connections_and_sessions_load_models_only_once(self):
        self.gate.set()
        await self.server.model_loader.task
        async with websockets.connect(self.uri) as ws:
            for _ in range(2):
                await self.send_start(ws)
                await ws.send(json.dumps({'type': 'CMD_GET_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl'}))
                await self.until(ws, lambda packet: packet['type'] == 'COUNT_THRESHOLD_STATUS')
            stops_before_monitor = self.server.stop_camera.call_count
            async with websockets.connect(self.uri) as monitor:
                await self.until(monitor, lambda packet: packet['type'] == 'SERVER_STATUS')
            # Closing a status-only connection must not close the active camera.
            await asyncio.sleep(0.03)
            self.assertEqual(self.server.stop_camera.call_count, stops_before_monitor)
            self.assertEqual(self.server.init_live_session.call_count, 2)
        self.assertEqual(self.factory_calls, 1)

    async def test_disconnecting_while_pending_does_not_open_camera_later(self):
        async with websockets.connect(self.uri) as ws:
            await self.send_start(ws)
            await self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')
        self.gate.set()
        await self.server.model_loader.task
        await asyncio.sleep(0.03)
        self.server.init_live_session.assert_not_called()

    async def test_model_failure_is_reported_without_closing_listener(self):
        self.fail_model = True
        async with websockets.connect(self.uri) as ws:
            await self.send_start(ws)
            await self.until(ws, lambda packet: packet['type'] == 'SESSION_PENDING')
            self.gate.set()
            result = await self.until(ws, lambda packet: packet['type'] == 'ERROR')
            self.assertEqual(result['code'], 'MODEL_LOAD_FAILED')
            self.assertTrue(result['fatal'])
            await ws.send(json.dumps({'type': 'CMD_GET_SERVER_STATUS'}))
            status = await self.until(ws, lambda packet: packet['type'] == 'SERVER_STATUS')
            self.assertEqual(status['model_state'], 'FAILED')
            self.assertIn('Missing test checkpoint', status['model_error'])


if __name__ == '__main__':
    unittest.main()
