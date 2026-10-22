# 한국어 코드 해설: `run_calibration_videos`에서 사용하는 실행·분석 보조 기능을 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Send test_video clips sequentially to the real socket server; create ASS overlays.

Run: python tests/run_calibration_videos.py
The local server is started automatically if port 8080 isn't already listening.
This client never loads AI models. Video processing happens on the socket server.
ASS files are saved beside each source video with the same basename, ready to load.
Re-running replaces those generated ASS files. Logs are kept separately under data/.
"""

# 이 파일에서 사용할 `argparse` 모듈을 불러옵니다.
import argparse
# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# `collections`에서 `Counter` 기능을 가져옵니다.
from collections import Counter
# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# `datetime`에서 `datetime` 기능을 가져옵니다.
from datetime import datetime
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `subprocess` 모듈을 불러옵니다.
import subprocess
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time

# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets

# 최상위 경로에 계산식 `Path(__file__).resolve().parents[1]`의 결과를 저장합니다. 대상: `ROOT`.
ROOT = Path(__file__).resolve().parents[1]
# `sys.path.insert(0, str(ROOT))`을 호출하여 해당 단계의 작업을 수행합니다.
sys.path.insert(0, str(ROOT))
# `utils.calibration_subtitles`에서 `write_calibration_ass` 기능을 가져옵니다.
from utils.calibration_subtitles import write_calibration_ass

# `EXERCISES` 값에 키 'biceps_curl', 'shoulder_press', 'clamshell', 'slr', 'leg_extension', 'leg_raise'로 구성한 사전을 저장합니다. 대상: `EXERCISES`.
EXERCISES = {'biceps_curl': 'biceps_curl', 'shoulder_press': 'shoulder_press',
             # 구성 항목: `clamshell` 값(`'clamshell'`)에 'clamshell'; `slr` 값(`'slr'`)에 'slr'을 지정합니다.
             'clamshell': 'clamshell', 'slr': 'slr',
             # 구성 항목: leg 신전(`'leg_extension'`)에 'knee_extension'; `leg_raise` 값(`'leg_raise'`)에 'hip_knee_flexion'을 지정합니다.
             'leg_extension': 'knee_extension', 'leg_raise': 'hip_knee_flexion'}


# 지정 폴더 아래의 원본 영상만 찾고 이미 생성한 오버레이 결과는 다시 처리하지 않습니다.
def find_videos(root, selected=None, limit=None):
    # `groups` 값에 빈 사전을 저장합니다. 대상: `groups`.
    groups = {}
    # `EXERCISES.items()`에서 항목을 차례로 꺼내 `(folder, exercise)`에 넣고 아래 처리를 반복합니다.
    for folder, exercise in EXERCISES.items():
        # 조건 `selected and folder not in selected`을 만족하는 경우에만 아래 분기를 실행합니다.
        if selected and folder not in selected:
            # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
            continue
        # `paths` 값에 `sorted((path for path in (root / folder).rglob('*') if path.is_file() and path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}))` 호출 결과를 저장합니다. 대상: `paths`.
        paths = sorted(path for path in (root / folder).rglob('*')
                       # 앞줄에서 시작한 식·인수·자료 구조를 `if path.is_file() and path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.m4v'})` 내용으로 이어서 구성합니다.
                       if path.is_file() and path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.m4v'})
        # `groups[folder]` 값에 반복·조건식 `[(path, exercise) for path in (paths[:limit] if limit else paths)]`으로 구성한 값을 저장합니다. 대상: `groups[folder]`.
        groups[folder] = [(path, exercise) for path in (paths[:limit] if limit else paths)]
    # 운동 폴더를 번갈아 처리해 특정 폴더만 먼저 끝나지 않고 여러 운동의 결과를 일찍 확인합니다.
    # 현재 함수 처리를 끝내고 반복·조건식 `[items[index] for index in range(max(map(len, groups.values()), default=0)) for items in groups.values() if index < len(items)]`으로 구성한 값을 호출한 쪽으로 반환합니다.
    return [items[index] for index in range(max(map(len, groups.values()), default=0))
            # 앞줄에서 시작한 식·인수·자료 구조를 `for items in groups.values() if index < len(items)]` 내용으로 이어서 구성합니다.
            for items in groups.values() if index < len(items)]


# wait for 결과 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
async def wait_for_result(ws, timeout):
    # `deadline` 값에 계산식 `time.monotonic() + timeout`의 결과를 저장합니다. 대상: `deadline`.
    deadline = time.monotonic() + timeout
    # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
    while True:
        # `remaining` 값에 계산식 `deadline - time.monotonic()`의 결과를 저장합니다. 대상: `remaining`.
        remaining = deadline - time.monotonic()
        # 조건 `remaining <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if remaining <= 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `TimeoutError('Video analysis timeout')` 예외를 발생시킵니다.
            raise TimeoutError('Video analysis timeout')
        # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(await asyncio.wait_for(ws.recv(), remaining))`)을 저장합니다. 대상: `packet`.
        packet = json.loads(await asyncio.wait_for(ws.recv(), remaining))
        # `kind` 값에 `packet.get('type')` 호출 결과를 저장합니다. 대상: `kind`.
        kind = packet.get('type')
        # 조건 `kind == 'VIDEO_PROGRESS'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if kind == 'VIDEO_PROGRESS':
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"  frames {packet['processed_frames']}/{packet['total_frames']}", flush=True)
        # 조건 `kind in {'VIDEO_STATUS', 'SESSION_PENDING'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif kind in {'VIDEO_STATUS', 'SESSION_PENDING'}:
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print('  ' + packet.get('message', kind), flush=True)
        # 조건 `kind == 'ERROR'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif kind == 'ERROR':
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(packet.get('message', 'Server error'))` 예외를 발생시킵니다.
            raise RuntimeError(packet.get('message', 'Server error'))
        # 조건 `kind == 'VIDEO_ANALYSIS_COMPLETED'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif kind == 'VIDEO_ANALYSIS_COMPLETED':
            # 현재 함수 처리를 끝내고 수신 또는 전송할 JSON 메시지 사전(`packet`)을 호출한 쪽으로 반환합니다.
            return packet


# `open_connection` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
async def open_connection(args):
    # 현재 함수 처리를 끝내고 계산식 `await websockets.connect(args.uri, open_timeout=2, ping_interval=None, max_size=16 * 1024 * 1024)`의 결과를 호출한 쪽으로 반환합니다.
    return await websockets.connect(args.uri, open_timeout=2, ping_interval=None, max_size=16 * 1024 * 1024)


# 실행 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
async def run(args, output, videos):
    # `(process, log_file, ws)` 값에 순서·구성을 지정한 `(None, None, None)`을 저장합니다. 대상: `(process, log_file, ws)`.
    process, log_file, ws = None, None, None
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # WebSocket 연결에 계산식 `await open_connection(args)`의 결과를 저장합니다. 대상: `ws`.
            ws = await open_connection(args)
        # `OSError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except OSError:
            # 조건 `args.no_start_server or args.uri != 'ws://127.0.0.1:8080'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if args.no_start_server or args.uri != 'ws://127.0.0.1:8080':
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Start the socket server first or use the default local URI')` 예외를 발생시킵니다.
                raise RuntimeError('Start the socket server first or use the default local URI')
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print('Starting local socket server (models load once)...', flush=True)
            # 로그 파일에 `(output / 'server.log').open('w', encoding='utf-8')` 호출 결과를 저장합니다. 대상: `log_file`.
            log_file = (output / 'server.log').open('w', encoding='utf-8')
            # 처리에 `subprocess.Popen([sys.executable, '-u', str(ROOT / 'network/socket_server.py')], cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONIOEN…` 호출 결과를 저장합니다. 대상: `process`.
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'network/socket_server.py')],
                # 구성 항목: `cwd` 값(`cwd`)에 `str(ROOT)` 호출 결과; `stdout` 값(`stdout`)에 로그 파일(`log_file`); `stderr` 값(`stderr`)에 `subprocess.STDOUT` 값(`subprocess.STDOUT`)을 지정합니다.
                cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT,
                # 구성 항목: `env` 값(`env`)에 키 'PYTHONIOENCODING'로 구성한 사전; `PYTHONIOENCODING` 값(`'PYTHONIOENCODING'`)에 'utf-8'을 지정합니다.
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                # 구성 항목: `creationflags` 값(`creationflags`)에 `getattr(subprocess, 'CREATE_NO_WINDOW', 0)` 호출 결과를 지정합니다.
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            # `deadline` 값에 계산식 `time.monotonic() + args.startup_timeout`의 결과를 저장합니다. 대상: `deadline`.
            deadline = time.monotonic() + args.startup_timeout
            # 조건 `ws is None`이 유지되는 동안 아래 처리를 반복합니다.
            while ws is None:
                # 조건 `process.poll() is not None or time.monotonic() >= deadline`을 만족하는 경우에만 아래 분기를 실행합니다.
                if process.poll() is not None or time.monotonic() >= deadline:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(f"Server startup failed; see {output / 'server.log'}")` 예외를 발생시킵니다.
                    raise RuntimeError(f'Server startup failed; see {output / "server.log"}')
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # WebSocket 연결에 계산식 `await open_connection(args)`의 결과를 저장합니다. 대상: `ws`.
                    ws = await open_connection(args)
                # `OSError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except OSError:
                    # `asyncio.sleep(0.25)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
                    await asyncio.sleep(0.25)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f'Connected: {args.uri}', flush=True)
        # 영상을 보내기 전에 이미 실행 중인 서버의 모델 준비 완료 알림을 기다립니다.
        # `deadline` 값에 계산식 `time.monotonic() + args.startup_timeout`의 결과를 저장합니다. 대상: `deadline`.
        deadline = time.monotonic() + args.startup_timeout
        # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
        while True:
            # `remaining` 값에 계산식 `deadline - time.monotonic()`의 결과를 저장합니다. 대상: `remaining`.
            remaining = deadline - time.monotonic()
            # 조건 `remaining <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
            if remaining <= 0:
                # 이 조건에서는 처리를 계속할 수 없으므로 `TimeoutError('Model preparation timeout')` 예외를 발생시킵니다.
                raise TimeoutError('Model preparation timeout')
            # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(await asyncio.wait_for(ws.recv(), remaining))`)을 저장합니다. 대상: `packet`.
            packet = json.loads(await asyncio.wait_for(ws.recv(), remaining))
            # 조건 `packet.get('type') == 'SERVER_STATUS'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if packet.get('type') == 'SERVER_STATUS':
                # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                print('  model: ' + packet.get('phase', packet.get('model_state', '')), flush=True)
                # 조건 `packet.get('model_state') == 'FAILED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet.get('model_state') == 'FAILED':
                    # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(packet.get('model_error', 'Model loading failed'))` 예외를 발생시킵니다.
                    raise RuntimeError(packet.get('model_error', 'Model loading failed'))
                # 조건 `packet.get('ready')`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet.get('ready'):
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break
        # `results` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `results`.
        results = []
        # `fields` 값에 순서·구성을 지정한 `['video', 'exercise', 'result', 'left_reps', 'right_reps', 'final_step', 'count_events', 'subtitle_path', 'frame_csv', 'runtime_sec', 'error']`을 저장합니다. 대상: `fields`.
        fields = ['video', 'exercise', 'result', 'left_reps', 'right_reps', 'final_step',
                  # 앞줄에서 시작한 식·인수·자료 구조를 `'count_events', 'subtitle_path', 'frame_csv', 'runtime_sec', 'error']` 내용으로 이어서 구성합니다.
                  'count_events', 'subtitle_path', 'frame_csv', 'runtime_sec', 'error']
        # `(output / 'results.csv').open('w', encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with (output / 'results.csv').open('w', encoding='utf-8-sig', newline='') as file:
            # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=fields)` 호출 결과를 저장합니다. 대상: `writer`.
            writer = csv.DictWriter(file, fieldnames=fields)
            # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
            writer.writeheader()
            # `file.flush()`을 호출하여 해당 단계의 작업을 수행합니다.
            file.flush()
            # `enumerate(videos, 1)`에서 항목을 차례로 꺼내 `(index, (path, exercise))`에 넣고 아래 처리를 반복합니다.
            for index, (path, exercise) in enumerate(videos, 1):
                # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                print(f'\n[{index}/{len(videos)}] {path.relative_to(args.video_root)}', flush=True)
                # `row` 값에 `dict(video=str(path), exercise=exercise)` 호출 결과를 저장합니다. 대상: `row`.
                row = dict(video=str(path), exercise=exercise)
                # 동작 시작 시각 또는 실행 여부에 시스템 시계 변경에 영향받지 않는 시각(초)(`time.monotonic()`)을 저장합니다. 대상: `started`.
                started = time.monotonic()
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'mode': 'CALIBRATION', 'video_path': str(path), 'exercise_name': exercise, 'patient_id': f'video_{index:04}', 'log_root': …`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'mode': 'CALIBRATION',
                        # 구성 항목: 서버 PC에서 읽을 원본 영상 경로(`'video_path'`)에 `str(path)` 호출 결과; 운동 설정의 식별 이름(`'exercise_name'`)에 `exercise` 값(`exercise`); 환자 식별 번호(`'patient_id'`)에 계산식 `f'video_{index:04}'`의 결과를 지정합니다.
                        'video_path': str(path), 'exercise_name': exercise, 'patient_id': f'video_{index:04}',
                        # 구성 항목: 세션 로그를 저장하는 최상위 폴더(`'log_root'`)에 `str(output / 'sessions')` 호출 결과; 측정 세션(`'session'`)에 1; `round` 값(`'round'`)에 0; 설정(`'set'`)에 0을 지정합니다.
                        'log_root': str(output / 'sessions'), 'session': 1, 'round': 0, 'set': 0,
                        # 구성 항목: 한 번에 추론 또는 학습할 표본 수(`'batch_size'`)에 한 번에 추론 또는 학습할 표본 수(`args.batch_size`); 영상 녹화 여부(`'record_video'`)에 True; 출력 mode(`'output_mode'`)에 'LIGHTWEIGHT'을 지정합니다.
                        'batch_size': args.batch_size, 'record_video': True, 'output_mode': 'LIGHTWEIGHT'}))
                    # 수신 또는 전송할 JSON 메시지 사전에 계산식 `await wait_for_result(ws, args.timeout)`의 결과를 저장합니다. 대상: `packet`.
                    packet = await wait_for_result(ws, args.timeout)
                    # 저장·종료 결과 요약에 계산식 `packet['summary']`의 결과를 저장합니다. 대상: `summary`.
                    summary = packet['summary']
                    # 프레임 csv에 `Path(summary['frame_csv'])` 호출 결과를 저장합니다. 대상: `frame_csv`.
                    frame_csv = Path(summary['frame_csv'])
                    # 이전 서버 응답과 호환하기 위해 실제 기록된 상태를 보조 정보로 읽습니다.
                    # `frame_csv.open(encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                    with frame_csv.open(encoding='utf-8-sig', newline='') as frames:
                        # `logged` 값에 `list(csv.DictReader(frames))` 호출 결과를 저장합니다. 대상: `logged`.
                        logged = list(csv.DictReader(frames))
                    # `last` 값에 `logged`이면 `logged[-1]`, 아니면 `{}`을 저장합니다. 대상: `last`.
                    last = logged[-1] if logged else {}
                    # `finished` 값에 계산식 `last.get('calib_step') == 'FINISHED'`의 결과를 저장합니다. 대상: `finished`.
                    finished = last.get('calib_step') == 'FINISHED'
                    # `row.update(result='FINISHED' if finished else 'INCOMPLETE', left_reps=last.get('left_rep_count', 0), right_reps=last.get('right_rep_count', 0), final_step=last.get('cali…`을 호출하여 해당 단계의 작업을 수행합니다.
                    row.update(result='FINISHED' if finished else 'INCOMPLETE',
                        # 구성 항목: 왼쪽 반복 횟수(`left_reps`)에 `last.get('left_rep_count', 0)` 호출 결과; 오른쪽 반복 횟수(`right_reps`)에 `last.get('right_rep_count', 0)` 호출 결과를 지정합니다.
                        left_reps=last.get('left_rep_count', 0), right_reps=last.get('right_rep_count', 0),
                        # 구성 항목: `final_step` 값(`final_step`)에 `last.get('calib_step', 'NO_FRAMES')` 호출 결과; 프레임 csv(`frame_csv`)에 `str(frame_csv)` 호출 결과를 지정합니다.
                        final_step=last.get('calib_step', 'NO_FRAMES'), frame_csv=str(frame_csv))
                    # `ass` 값에 `(args.subtitle_root / path.relative_to(args.video_root)).with_suffix('.ass')` 호출 결과를 저장합니다. 대상: `ass`.
                    ass = (args.subtitle_root / path.relative_to(args.video_root)).with_suffix('.ass')
                    # 길이에 `packet.get('source_duration_sec')` 호출 결과를 저장합니다. 대상: `duration`.
                    duration = packet.get('source_duration_sec')
                    # `row.update(write_calibration_ass(frame_csv, ass, duration_sec=duration))`을 호출하여 해당 단계의 작업을 수행합니다.
                    row.update(write_calibration_ass(frame_csv, ass, duration_sec=duration))
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(f"  {row['result']} | L={row['left_reps']} R={row['right_reps']} | {row['final_step']}", flush=True)
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(f'  ASS: {ass}', flush=True)
                # `(TimeoutError, websockets.exceptions.ConnectionClosed)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except (TimeoutError, websockets.exceptions.ConnectionClosed):
                    # 시간 초과 뒤에도 서버 작업이 진행 중일 수 있으므로 해당 연결에 다음 영상을 보내지 않습니다.
                    # 이 조건에서는 처리를 계속할 수 없으므로 `None` 예외를 발생시킵니다.
                    raise
                # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except Exception as exc:
                    # `row.update(result='ERROR', error=str(exc))`을 호출하여 해당 단계의 작업을 수행합니다.
                    row.update(result='ERROR', error=str(exc))
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(f'  ERROR: {exc}', flush=True)
                # `row['runtime_sec']` 값에 `round(time.monotonic() - started, 2)` 호출 결과를 저장합니다. 대상: `row['runtime_sec']`.
                row['runtime_sec'] = round(time.monotonic() - started, 2)
                # `results.append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
                results.append(row)
                # `writer.writerow(row)`을 호출하여 CSV에 행 하나를 기록합니다.
                writer.writerow(row)
                # `file.flush()`을 호출하여 해당 단계의 작업을 수행합니다.
                file.flush()
        # `counts` 값에 `Counter((row['result'] for row in results))` 호출 결과를 저장합니다. 대상: `counts`.
        counts = Counter(row['result'] for row in results)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"\nDONE: {len(results)} videos | FINISHED={counts['FINISHED']} "
              # 구성 항목: `flush` 값(`flush`)에 True을 지정합니다.
              f"INCOMPLETE={counts['INCOMPLETE']} ERROR={counts['ERROR']}", flush=True)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f'Results: {output / "results.csv"}', flush=True)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print('ASS files are beside the source videos by default; open a video in an ASS-capable player.', flush=True)
        # 현재 함수 처리를 끝내고 `counts['INCOMPLETE'] + counts['ERROR'] == 0`이면 `0`, 아니면 `1`을 호출한 쪽으로 반환합니다.
        return 0 if counts['INCOMPLETE'] + counts['ERROR'] == 0 else 1
    # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
    finally:
        # 조건 `ws is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if ws is not None:
            # `ws.close()`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await ws.close()
        # 조건 `process is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if process is not None:
            # `process.terminate()`을 호출하여 해당 단계의 작업을 수행합니다.
            process.terminate()
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `process.wait(timeout=5)`을 호출하여 해당 단계의 작업을 수행합니다.
                process.wait(timeout=5)
            # `subprocess.TimeoutExpired` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except subprocess.TimeoutExpired:
                # `process.kill()`을 호출하여 해당 단계의 작업을 수행합니다.
                process.kill()
                # `process.wait()`을 호출하여 해당 단계의 작업을 수행합니다.
                process.wait()
        # 조건 `log_file is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if log_file is not None:
            # `log_file.close()`을 호출하여 해당 단계의 작업을 수행합니다.
            log_file.close()


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def main():
    # `parser` 값에 `argparse.ArgumentParser(description=__doc__)` 호출 결과를 저장합니다. 대상: `parser`.
    parser = argparse.ArgumentParser(description=__doc__)
    # `parser.add_argument('--uri', default='ws://127.0.0.1:8080')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--uri', default='ws://127.0.0.1:8080')
    # `parser.add_argument('--video-root', type=Path, default=ROOT / 'test_video')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--video-root', type=Path, default=ROOT / 'test_video')
    # `parser.add_argument('--output', type=Path)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--output', type=Path)
    # `parser.add_argument('--subtitle-root', type=Path, help='ASS destination root (default: beside source videos in --video-root)')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--subtitle-root', type=Path,
                        # 구성 항목: `help` 값(`help`)에 'ASS destination root (default: beside source videos in --video-root)'을 지정합니다.
                        help='ASS destination root (default: beside source videos in --video-root)')
    # `parser.add_argument('--exercise', choices=list(EXERCISES), action='append')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--exercise', choices=list(EXERCISES), action='append')
    # `parser.add_argument('--limit-per-exercise', type=int)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--limit-per-exercise', type=int)
    # `parser.add_argument('--batch-size', type=int, default=8)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--batch-size', type=int, default=8)
    # `parser.add_argument('--timeout', type=float, default=600)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--timeout', type=float, default=600)
    # `parser.add_argument('--startup-timeout', type=float, default=180)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--startup-timeout', type=float, default=180)
    # `parser.add_argument('--no-start-server', action='store_true')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--no-start-server', action='store_true')
    # `args` 값에 `parser.parse_args()` 호출 결과를 저장합니다. 대상: `args`.
    args = parser.parse_args()
    # 조건 `args.batch_size <= 0 or args.timeout <= 0 or args.startup_timeout <= 0 or (args.limit_per_exercise is not None and args.limit_per_exercise <= 0)`을 만족하는 경우에만 아래 분기를 실행합니다.
    if (args.batch_size <= 0 or args.timeout <= 0 or args.startup_timeout <= 0
            # 앞줄에서 시작한 식·인수·자료 구조를 `or (args.limit_per_exercise is not None and args.limit_per_exercise <= 0)):` 내용으로 이어서 구성합니다.
            or (args.limit_per_exercise is not None and args.limit_per_exercise <= 0)):
        # `parser.error('Batch size, timeouts and limits must be positive')`을 호출하여 해당 단계의 작업을 수행합니다.
        parser.error('Batch size, timeouts and limits must be positive')
    # 영상 최상위 경로에 `args.video_root.resolve()` 호출 결과를 저장합니다. 대상: `args.video_root`.
    args.video_root = args.video_root.resolve()
    # subtitle 최상위 경로에 `(args.subtitle_root or args.video_root).resolve()` 호출 결과를 저장합니다. 대상: `args.subtitle_root`.
    args.subtitle_root = (args.subtitle_root or args.video_root).resolve()
    # `videos` 값에 `find_videos(args.video_root, args.exercise, args.limit_per_exercise)` 호출 결과를 저장합니다. 대상: `videos`.
    videos = find_videos(args.video_root, args.exercise, args.limit_per_exercise)
    # 조건 `not videos`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not videos:
        # `parser.error(f'No supported videos found in {args.video_root}')`을 호출하여 해당 단계의 작업을 수행합니다.
        parser.error(f'No supported videos found in {args.video_root}')
    # 생성할 결과 데이터 또는 경로에 `(args.output or ROOT / 'data/calibration_socket_tests' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()` 호출 결과를 저장합니다. 대상: `output`.
    output = (args.output or ROOT / 'data/calibration_socket_tests' /
              # 앞줄에서 시작한 식·인수·자료 구조를 `datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()` 내용으로 이어서 구성합니다.
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    # `output.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
    output.mkdir(parents=True, exist_ok=True)
    # 조건 `(output / 'results.csv').exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if (output / 'results.csv').exists():
        # `parser.error('Choose a new output directory; existing results will not be overwritten')`을 호출하여 해당 단계의 작업을 수행합니다.
        parser.error('Choose a new output directory; existing results will not be overwritten')
    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print(f'Videos: {len(videos)} | Output: {output}', flush=True)
    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print(f'ASS destination: {args.subtitle_root} (existing same-name ASS files are replaced)', flush=True)
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # 현재 함수 처리를 끝내고 `asyncio.run(run(args, output, videos))` 호출 결과를 호출한 쪽으로 반환합니다.
        return asyncio.run(run(args, output, videos))
    # `KeyboardInterrupt` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
    except KeyboardInterrupt:
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print('\nStopped by user.', flush=True)
        # 현재 함수 처리를 끝내고 130을 호출한 쪽으로 반환합니다.
        return 130
    # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
    except Exception as exc:
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f'FAILED: {exc}', file=sys.stderr, flush=True)
        # 현재 함수 처리를 끝내고 2을 호출한 쪽으로 반환합니다.
        return 2


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `sys.exit(main())`을 호출하여 해당 단계의 작업을 수행합니다.
    sys.exit(main())
