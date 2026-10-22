"""화면을 재시작한 뒤 남은 서버 세션 연결을 정리하는 도구입니다."""

import asyncio
import json
import websockets

# Streamlit이 없어도 같은 복구 기능을 사용할 수 있는 비동기 함수입니다.
async def reset_server_session(uri):
    async with websockets.connect(uri, open_timeout=3, close_timeout=2) as websocket:
        await websocket.send(json.dumps({"type": "CMD_RESET_SESSION"}))
        async def receive_result():
            async for message in websocket:
                packet = json.loads(message)
                if packet.get("type") == "SERVER_SESSION_RESET":
                    return packet
                if packet.get("type") == "ERROR":
                    raise RuntimeError(packet.get("message", "서버 연결 복구에 실패했습니다."))
            raise RuntimeError("복구 완료 응답 전에 서버 연결이 종료되었습니다.")
        return await asyncio.wait_for(receive_result(), timeout=25)
