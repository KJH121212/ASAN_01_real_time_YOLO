# 한국어 코드 해설: 진행 중인 소켓 세션에서 가동 범위 조회·수정 기능을 직접 확인하는 클라이언트입니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Interactive Unity-style client: start webcam tracking, inspect/change ROM."""

# 이 파일에서 사용할 `argparse` 모듈을 불러옵니다.
import argparse
# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# 이 파일에서 사용할 `uuid` 모듈을 불러옵니다.
import uuid

# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets


# `HELP` 값에 'Commands: show | left START TARGET | right START TARGET | both START TARGET | quit'을 저장합니다. 대상: `HELP`.
HELP = "Commands: show | left START TARGET | right START TARGET | both START TARGET | quit"


# 생성 명령 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def make_command(text, exercise_name):
    # `words` 값에 `text.strip().split()` 호출 결과를 저장합니다. 대상: `words`.
    words = text.strip().split()
    # 조건 `words == ['show']`을 만족하는 경우에만 아래 분기를 실행합니다.
    if words == ["show"]:
        # 수신 또는 전송할 JSON 메시지 사전에 키 'type'로 구성한 사전을 저장합니다. 대상: `packet`.
        packet = {"type": "CMD_GET_COUNT_THRESHOLD"}
    # 조건 `len(words) == 3 and words[0] in {'left', 'right', 'both'}`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif len(words) == 3 and words[0] in {"left", "right", "both"}:
        # `(start, target)` 값에 순서·구성을 지정한 `(float(words[1]), float(words[2]))`을 저장합니다. 대상: `(start, target)`.
        start, target = float(words[1]), float(words[2])
        # `sides` 값에 `words[0] == 'both'`이면 `('left', 'right')`, 아니면 `(words[0],)`을 저장합니다. 대상: `sides`.
        sides = ("left", "right") if words[0] == "both" else (words[0],)
        # 수신 또는 전송할 JSON 메시지 사전에 키 'type', 'thresholds'로 구성한 사전을 저장합니다. 대상: `packet`.
        packet = {
            # 구성 항목: `type` 값(`'type'`)에 'CMD_UPDATE_COUNT_THRESHOLD'을 지정합니다.
            "type": "CMD_UPDATE_COUNT_THRESHOLD",
            # 구성 항목: 좌우 가동 범위 시작값·목표값(`'thresholds'`)에 반복·조건식 `{side: {'start_val': start, 'target_val': target} for side in sides}`으로 구성한 값; 개인 가동 범위의 시작값(`'start_val'`)에 시작(`start`); 개인 가동 범위의 목표값(`'target_val'`)에 목표(`target`)을 지정합니다.
            "thresholds": {side: {"start_val": start, "target_val": target} for side in sides},
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
    # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
    else:
        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(HELP)` 예외를 발생시킵니다.
        raise ValueError(HELP)
    # `packet.update(exercise_name=exercise_name, request_id=uuid.uuid4().hex)`을 호출하여 해당 단계의 작업을 수행합니다.
    packet.update(exercise_name=exercise_name, request_id=uuid.uuid4().hex)
    # 현재 함수 처리를 끝내고 수신 또는 전송할 JSON 메시지 사전(`packet`)을 호출한 쪽으로 반환합니다.
    return packet


# 실행 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
async def run(args):
    # 세션 시작·가동 범위 수정·이벤트 수신에 같은 WebSocket 연결을 사용합니다.
    # `websockets.connect(args.uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    async with websockets.connect(args.uri) as ws:
        # `ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'player_id': args.player_id, 'patient_name': 'Threshold Test', 'exercise_name': args.exercise_name, 'mode': 'MAIN', 'targe…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await ws.send(json.dumps({
            # 구성 항목: `type` 값(`'type'`)에 'CMD_SET_SESSION'을 지정합니다.
            "type": "CMD_SET_SESSION",
            # 구성 항목: 환자 식별 번호의 기존 필드명(`'player_id'`)에 환자 식별 번호의 기존 필드명(`args.player_id`)을 지정합니다.
            "player_id": args.player_id,
            # 구성 항목: `patient_name` 값(`'patient_name'`)에 'Threshold Test'을 지정합니다.
            "patient_name": "Threshold Test",
            # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 운동 설정의 식별 이름(`args.exercise_name`)을 지정합니다.
            "exercise_name": args.exercise_name,
            # 구성 항목: 실행 모드(`'mode'`)에 'MAIN'을 지정합니다.
            "mode": "MAIN",
            # 구성 항목: 게임 목표 점수(`'target_score'`)에 게임 목표 점수(`args.target_score`)을 지정합니다.
            "target_score": args.target_score,
            # 구성 항목: 출력 mode(`'output_mode'`)에 'LIGHTWEIGHT'을 지정합니다.
            "output_mode": "LIGHTWEIGHT",
            # 구성 항목: `camera_index` 값(`'camera_index'`)에 `args.camera_index` 값(`args.camera_index`)을 지정합니다.
            "camera_index": args.camera_index,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }))
        # `ws.send(json.dumps(make_command('show', args.exercise_name)))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await ws.send(json.dumps(make_command("show", args.exercise_name)))
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(HELP, flush=True)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print("Values use the active metric (degrees or normalized height/distance).", flush=True)

        # 명령에 `asyncio.Queue()` 호출 결과를 저장합니다. 대상: `commands`.
        commands = asyncio.Queue()
        # `loop` 값에 `asyncio.get_running_loop()` 호출 결과를 저장합니다. 대상: `loop`.
        loop = asyncio.get_running_loop()

        # 읽기 console 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def read_console():
            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `text` 값에 `input('\nthreshold> ')` 호출 결과를 저장합니다. 대상: `text`.
                    text = input("\nthreshold> ")
                # `(EOFError, KeyboardInterrupt)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except (EOFError, KeyboardInterrupt):
                    # `text` 값에 'quit'을 저장합니다. 대상: `text`.
                    text = "quit"
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `loop.call_soon_threadsafe(commands.put_nowait, text)`을 호출하여 해당 단계의 작업을 수행합니다.
                    loop.call_soon_threadsafe(commands.put_nowait, text)
                # `RuntimeError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except RuntimeError:
                    # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                    return
                # 조건 `text.strip().lower() == 'quit'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if text.strip().lower() == "quit":
                    # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                    return

        # 전송 명령 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        async def send_commands():
            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:
                # `text` 값에 `(await commands.get()).strip()` 호출 결과를 저장합니다. 대상: `text`.
                text = (await commands.get()).strip()
                # 조건 `text.lower() == 'quit'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if text.lower() == "quit":
                    # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                    return  # Closing this connection saves the session and closes the camera.
                # 조건 `text.lower() == 'help' or not text`을 만족하는 경우에만 아래 분기를 실행합니다.
                if text.lower() == "help" or not text:
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(HELP, flush=True)
                    # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                    continue
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # 수신 또는 전송할 JSON 메시지 사전에 `make_command(text, args.exercise_name)` 호출 결과를 저장합니다. 대상: `packet`.
                    packet = make_command(text, args.exercise_name)
                    # `ws.send(json.dumps(packet, allow_nan=False))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await ws.send(json.dumps(packet, allow_nan=False))
                # `ValueError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except ValueError as exc:
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(f"Invalid command: {exc}", flush=True)

        # 수신 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        async def receive():
            # `previous` 값에 None을 저장합니다. 대상: `previous`.
            previous = None
            # `ws`에서 항목을 차례로 꺼내 `message`에 넣고 아래 처리를 반복합니다.
            async for message in ws:
                # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(message)`)을 저장합니다. 대상: `packet`.
                packet = json.loads(message)
                # 조건 `packet.get('type') == 'POSE_UPDATE'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet.get("type") == "POSE_UPDATE":
                    # `(left, right)` 값에 순서·구성을 지정한 `(packet.get('left', {}), packet.get('right', {}))`을 저장합니다. 대상: `(left, right)`.
                    left, right = packet.get("left", {}), packet.get("right", {})
                    # 상태에 순서·구성을 지정한 `(left.get('rep_count'), right.get('rep_count'), left.get('state'), right.get('state'), packet.get('is_occluded'), packet.get('threshold_version'))`을 저장합니다. 대상: `status`.
                    status = (left.get("rep_count"), right.get("rep_count"),
                              # 앞줄에서 시작한 식·인수·자료 구조를 `left.get("state"), right.get("state"),` 내용으로 이어서 구성합니다.
                              left.get("state"), right.get("state"),
                              # 앞줄에서 시작한 식·인수·자료 구조를 `packet.get("is_occluded"), packet.get("threshold_version"))` 내용으로 이어서 구성합니다.
                              packet.get("is_occluded"), packet.get("threshold_version"))
                    # 조건 `status != previous`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if status != previous:
                        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                        print(f"\nL={left.get('rep_count')} ({left.get('state')}, {left.get('quality')}) "
                              # 앞줄에서 시작한 식·인수·자료 구조를 `f"R={right.get('rep_count')} ({right.get('state')}, {right.get('quality')}) "` 내용으로 이어서 구성합니다.
                              f"R={right.get('rep_count')} ({right.get('state')}, {right.get('quality')}) "
                              # 앞줄에서 시작한 식·인수·자료 구조를 `f"occluded={packet.get('is_occluded')} "` 내용으로 이어서 구성합니다.
                              f"occluded={packet.get('is_occluded')} "
                              # 구성 항목: `flush` 값(`flush`)에 True을 지정합니다.
                              f"ROM version={packet.get('threshold_version')}", flush=True)
                        # `previous` 값에 상태(`status`)을 저장합니다. 대상: `previous`.
                        previous = status
                # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                else:
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print("\n" + json.dumps(packet, ensure_ascii=False, indent=2), flush=True)
                    # 조건 `packet.get('type') in {'SESSION_FINISHED', 'CALIBRATION_FINISHED'}`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet.get("type") in {"SESSION_FINISHED", "CALIBRATION_FINISHED"}:
                        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                        return

        # `threading.Thread(target=read_console, daemon=True).start()`을 호출하여 해당 단계의 작업을 수행합니다.
        threading.Thread(target=read_console, daemon=True).start()
        # `tasks` 값에 순서·구성을 지정한 `{asyncio.create_task(send_commands()), asyncio.create_task(receive())}`을 저장합니다. 대상: `tasks`.
        tasks = {asyncio.create_task(send_commands()), asyncio.create_task(receive())}
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `(done, _)` 값에 계산식 `await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)`의 결과를 저장합니다. 대상: `(done, _)`.
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # `done`에서 항목을 차례로 꺼내 `task`에 넣고 아래 처리를 반복합니다.
            for task in done:
                # `task.result()`을 호출하여 해당 단계의 작업을 수행합니다.
                task.result()
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `tasks`에서 항목을 차례로 꺼내 `task`에 넣고 아래 처리를 반복합니다.
            for task in tasks:
                # `task.cancel()`을 호출하여 해당 단계의 작업을 수행합니다.
                task.cancel()
            # `asyncio.gather(*tasks, return_exceptions=True)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await asyncio.gather(*tasks, return_exceptions=True)


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def main():
    # `parser` 값에 `argparse.ArgumentParser(description=__doc__)` 호출 결과를 저장합니다. 대상: `parser`.
    parser = argparse.ArgumentParser(description=__doc__)
    # `parser.add_argument('--uri', default='ws://127.0.0.1:8080')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--uri", default="ws://127.0.0.1:8080")
    # `parser.add_argument('--exercise-name', default='biceps_curl')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--exercise-name", default="biceps_curl")
    # `parser.add_argument('--player-id', default='threshold_test')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--player-id", default="threshold_test")
    # `parser.add_argument('--camera-index', type=int, default=0)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--camera-index", type=int, default=0)
    # `parser.add_argument('--target-score', '--target-reps', dest='target_score', type=float, default=1000, help='Target points (Perfect=1, Good=0.5, Bad=0.25)')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--target-score", "--target-reps", dest="target_score", type=float, default=1000,
                        # 구성 항목: `help` 값(`help`)에 'Target points (Perfect=1, Good=0.5, Bad=0.25)'을 지정합니다.
                        help="Target points (Perfect=1, Good=0.5, Bad=0.25)")
    # `args` 값에 `parser.parse_args()` 호출 결과를 저장합니다. 대상: `args`.
    args = parser.parse_args()
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # `asyncio.run(run(args))`을 호출하여 해당 단계의 작업을 수행합니다.
        asyncio.run(run(args))
    # `KeyboardInterrupt` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
    except KeyboardInterrupt:
        # 이 분기에는 추가 동작이 없으며 문법적으로 필요한 빈 블록만 유지합니다.
        pass
    # `(OSError, websockets.exceptions.ConnectionClosed)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
    except (OSError, websockets.exceptions.ConnectionClosed) as exc:
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"Connection ended: {exc}")


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == "__main__":
    # `main()`을 호출하여 해당 단계의 작업을 수행합니다.
    main()
