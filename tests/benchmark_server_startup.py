# 한국어 코드 해설: `benchmark_server_startup`에서 사용하는 실행·분석 보조 기능을 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Measure actual listening/model-ready times without opening the camera."""

# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time
# `STARTED` 값에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `STARTED`.
STARTED = time.perf_counter()

# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys

# `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`을 호출하여 해당 단계의 작업을 수행합니다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets
# `network.socket_server`에서 `ExerciseSocketServer` 기능을 가져옵니다.
from network.socket_server import ExerciseSocketServer


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
async def main():
    # 서버에 `ExerciseSocketServer(host='127.0.0.1', port=0)` 호출 결과를 저장합니다. 대상: `server`.
    server = ExerciseSocketServer(host='127.0.0.1', port=0)
    # `task` 값에 별도로 진행하도록 등록한 비동기 작업(`asyncio.create_task(server.run())`)을 저장합니다. 대상: `task`.
    task = asyncio.create_task(server.run())
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # 조건 `server.bound_port is None`이 유지되는 동안 아래 처리를 반복합니다.
        while server.bound_port is None:
            # `asyncio.sleep(0.001)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
            await asyncio.sleep(0.001)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f'BENCHMARK_LISTEN_SECONDS={time.perf_counter() - STARTED:.3f}', flush=True)
        # `websockets.connect(f'ws://127.0.0.1:{server.bound_port}')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(f'ws://127.0.0.1:{server.bound_port}') as ws:
            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:
                # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(await ws.recv())`)을 저장합니다. 대상: `packet`.
                packet = json.loads(await ws.recv())
                # 조건 `packet.get('type') == 'SERVER_STATUS'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet.get('type') == 'SERVER_STATUS':
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(json.dumps(packet, ensure_ascii=False), flush=True)
                    # 조건 `packet['model_state'] == 'FAILED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet['model_state'] == 'FAILED':
                        # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(packet['model_error'])` 예외를 발생시킵니다.
                        raise RuntimeError(packet['model_error'])
                    # 조건 `packet['ready']`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet['ready']:
                        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                        print(f'BENCHMARK_READY_SECONDS={time.perf_counter() - STARTED:.3f}', flush=True)
                        # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                        break
    # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
    finally:
        # `task.cancel()`을 호출하여 해당 단계의 작업을 수행합니다.
        task.cancel()
        # `asyncio.gather(task, return_exceptions=True)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await asyncio.gather(task, return_exceptions=True)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `asyncio.run(main())`을 호출하여 해당 단계의 작업을 수행합니다.
    asyncio.run(main())
