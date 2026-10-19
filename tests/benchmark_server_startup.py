"""Measure actual listening/model-ready times without opening the camera."""

import time
STARTED = time.perf_counter()

import asyncio
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import websockets
from network.socket_server import ExerciseSocketServer


async def main():
    server = ExerciseSocketServer(host='127.0.0.1', port=0)
    task = asyncio.create_task(server.run())
    try:
        while server.bound_port is None:
            await asyncio.sleep(0.001)
        print(f'BENCHMARK_LISTEN_SECONDS={time.perf_counter() - STARTED:.3f}', flush=True)
        async with websockets.connect(f'ws://127.0.0.1:{server.bound_port}') as ws:
            while True:
                packet = json.loads(await ws.recv())
                if packet.get('type') == 'SERVER_STATUS':
                    print(json.dumps(packet, ensure_ascii=False), flush=True)
                    if packet['model_state'] == 'FAILED':
                        raise RuntimeError(packet['model_error'])
                    if packet['ready']:
                        print(f'BENCHMARK_READY_SECONDS={time.perf_counter() - STARTED:.3f}', flush=True)
                        break
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


if __name__ == '__main__':
    asyncio.run(main())
