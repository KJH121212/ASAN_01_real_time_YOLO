"""23개 관절(COCO-23) 모션 측정 소켓 서버 진입점.

실행 방법:
    python socket_server/server_entry.py
    (또는 socket_server 폴더 안에서: python server_entry.py)
"""

import asyncio
import multiprocessing
from pathlib import Path
import sys

SERVER_ROOT = Path(__file__).resolve().parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))


async def main():
    from network.camera import configure_local_and_gopro_no_proxy
    from network.socket_server import ExerciseSocketServer

    configure_local_and_gopro_no_proxy()
    print("==========================================", flush=True)
    print("   ASAN Motion Socket Server (COCO-23)    ", flush=True)
    print("==========================================", flush=True)

    server = ExerciseSocketServer(host="127.0.0.1", port=8080)
    await server.start_server()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] 사용자에 의해 서버가 정상 종료되었습니다.")
        sys.exit(0)
