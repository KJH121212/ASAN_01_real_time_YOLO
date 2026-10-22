"""Windows exe 및 Python 실행에서 공통으로 사용하는 소켓 서버 진입점입니다."""

# 서버 이벤트 루프를 실행합니다.
import asyncio
# exe에서 자식 프로세스가 진입점을 다시 실행하지 않도록 준비합니다.
import multiprocessing
# 진단 메시지를 표준 오류로 출력합니다.
import sys


# 실행 파일을 열면 별도 버튼 없이 소켓 수신과 모델 준비를 시작합니다.
def main():
    # PyInstaller의 프로세스 생성 호환 처리를 AI 라이브러리 가져오기 전에 수행합니다.
    multiprocessing.freeze_support()
    # 실제 서버 구현을 재사용하여 Python·exe의 측정 동작을 동일하게 유지합니다.
    from network.socket_server import ExerciseSocketServer
    # 기본 주소는 로컬 PC의 ws://127.0.0.1:8080입니다.
    server = ExerciseSocketServer()
    # 포트 충돌과 사용자 중단을 실행 진입점에서 안내합니다.
    try:
        # 소켓을 먼저 열고 AI 가중치를 백그라운드에서 준비합니다.
        asyncio.run(server.run())
    # 콘솔에서 Ctrl+C를 누르면 정상 종료 코드로 끝냅니다.
    except KeyboardInterrupt:
        return 0
    # 이미 서버가 실행 중인 경우 새 프로세스에서 기존 서버를 강제로 끄지 않습니다.
    except OSError as exc:
        # Windows 및 일반 소켓의 주소 사용 중 오류를 구분합니다.
        if getattr(exc, "winerror", None) == 10048 or exc.errno in (98, 10048):
            print("8080 포트가 사용 중입니다. 기존 서버에 연결하세요. SERVER_BUSY는 화면의 이전 서버 연결 정리 버튼으로 복구하세요.", file=sys.stderr)
            return 1
        # 다른 파일·장치 오류는 원래 상세 정보를 유지합니다.
        raise
    # 이벤트 루프가 정상적으로 끝난 경우 성공으로 종료합니다.
    return 0


# 모듈 가져오기만으로 서버가 켜지지 않도록 직접 실행일 때만 진입합니다.
if __name__ == "__main__":
    raise SystemExit(main())
