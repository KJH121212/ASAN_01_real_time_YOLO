"""Real WebSocket/JSON round trips for the shared command handler; no GPU/camera."""

import asyncio
import json
import unittest

import websockets

from core.motion_engine import MotionEngine
from core.session_controller import SessionController
from network.session_commands import handle_count_threshold_command


class ThresholdSocketTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.engine = MotionEngine({
            'exercise_name': 'biceps_curl',
            'eval_config': {'metric_type': 'RELATIVE_Y', 'motion_direction': 'INCREASING'},
            'default_thresholds': {side: {'start_val': 0.1, 'target_val': 0.9}
                                   for side in ('left', 'right')},
        })
        self.controller = SessionController(motion_engine=self.engine)

        async def handle(ws):
            async for message in ws:
                response = handle_count_threshold_command(self.controller, json.loads(message))
                await ws.send(json.dumps(response))

        self.server = await websockets.serve(handle, '127.0.0.1', 0)
        self.uri = f"ws://127.0.0.1:{self.server.sockets[0].getsockname()[1]}"

    async def asyncTearDown(self):
        self.server.close()
        await self.server.wait_closed()

    async def exchange(self, ws, **overrides):
        packet = {'type': 'CMD_UPDATE_COUNT_THRESHOLD', 'exercise_name': 'biceps_curl',
                  'request_id': 'socket-test', 'thresholds': {'left': {'target_val': 0.5}}}
        packet.update(overrides)
        await ws.send(json.dumps(packet))
        return json.loads(await asyncio.wait_for(ws.recv(), timeout=2))

    async def test_query_update_and_recovery_after_error_on_same_connection(self):
        async with websockets.connect(self.uri) as ws:
            initial = await self.exchange(ws, type='CMD_GET_COUNT_THRESHOLD')
            self.assertEqual(initial['thresholds']['left']['target_val'], 0.9)
            self.assertEqual(initial['threshold_version'], 0)

            rejected = await self.exchange(ws, thresholds={'left': {'target_val': 0.1}})
            self.assertEqual(rejected['code'], 'INVALID_THRESHOLD')

            changed = await self.exchange(ws)
            self.assertEqual(changed['type'], 'COUNT_THRESHOLD_UPDATED')
            self.assertEqual(changed['request_id'], 'socket-test')
            self.assertEqual(changed['states']['left'], 'RETURNING')
            self.assertEqual(changed['thresholds']['right']['target_val'], 0.9)

            final = await self.exchange(ws, type='CMD_GET_COUNT_THRESHOLD')
            self.assertEqual(final['thresholds']['left']['target_val'], 0.5)
            self.assertEqual(final['threshold_version'], 1)


if __name__ == '__main__':
    unittest.main()
