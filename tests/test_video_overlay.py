# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np
# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets

# `core.video_overlay`에서 `VideoOverlayJob`, `find_videos`, `infer_exercise`, `render_video` 기능을 가져옵니다.
from core.video_overlay import VideoOverlayJob, find_videos, infer_exercise, render_video


# 영상 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def video(path):
    # `path.parent.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # CSV 또는 영상 출력 객체에 `cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10.0, (120, 80))` 호출 결과를 저장합니다. 대상: `writer`.
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10., (120, 80))
    # `writer.isOpened()`이 참인지 확인하고, 거짓이면 단언 오류로 기대 조건 위반을 알립니다.
    assert writer.isOpened()
    # `range(12)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
    for _ in range(12):
        # `writer.write(np.zeros((80, 120, 3), np.uint8))`을 호출하여 파일 또는 영상에 데이터를 기록합니다.
        writer.write(np.zeros((80, 120, 3), np.uint8))
    # `writer.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
    writer.release()


# 로그 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def logs(path):
    # `rows` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `rows`.
    rows = []
    # `(0, 4, 8)`에서 항목을 차례로 꺼내 `index`에 넣고 아래 처리를 반복합니다.
    for index in (0, 4, 8):
        # `rows.append(dict(frame_idx=index, video_frame_idx=index, motion_name='biceps_curl', calib_step='', left_value=0.5, left_start_val=0, left_target_val=1, left_rep_count=in…`을 호출하여 항목 하나를 끝에 추가합니다.
        rows.append(dict(frame_idx=index, video_frame_idx=index, motion_name='biceps_curl',
                         # 구성 항목: 캘리브레이션 전체 진행 단계(`calib_step`)에 ''; 왼쪽 측정값(`left_value`)에 0.5; 왼쪽 시작 val(`left_start_val`)에 0; 왼쪽 목표 val(`left_target_val`)에 1을 지정합니다.
                         calib_step='', left_value=.5, left_start_val=0, left_target_val=1,
                         # 구성 항목: 왼쪽 반복 횟수(`left_rep_count`)에 계산식 `index // 4`의 결과; 왼쪽 횟수 updated(`left_count_updated`)에 계산식 `index > 0`의 결과를 지정합니다.
                         left_rep_count=index // 4, left_count_updated=index > 0,
                         # 구성 항목: 왼쪽 품질 등급(`left_quality`)에 'GOOD'; 왼쪽 total 점수(`left_total_score`)에 계산식 `index / 8`의 결과; 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 False을 지정합니다.
                         left_quality='GOOD', left_total_score=index / 8, is_occluded=False))
    # `path.open('w', encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=list(rows[0]))` 호출 결과를 저장합니다. 대상: `writer`.
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
        writer.writeheader()
        # `writer.writerows(rows)`을 호출하여 CSV에 여러 행을 기록합니다.
        writer.writerows(rows)


# `VideoOverlayTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class VideoOverlayTests(unittest.TestCase):
    # 회귀 테스트: `test_render_preserves_duration_and_source_and_adds_pixels` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_render_preserves_duration_and_source_and_adds_pixels(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 최상위 경로에 `Path(directory)` 호출 결과를 저장합니다. 대상: `root`.
            root = Path(directory)
            # `(source, output)` 값에 순서·구성을 지정한 `(root / 'session.mp4', root / 'session.overlay.mp4')`을 저장합니다. 대상: `(source, output)`.
            source, output = root / 'session.mp4', root / 'session.overlay.mp4'
            # `video(source)`을 호출하여 해당 단계의 작업을 수행합니다.
            video(source)
            # `original` 값에 `source.read_bytes()` 호출 결과를 저장합니다. 대상: `original`.
            original = source.read_bytes()
            # `logs(source.with_suffix('.csv'))`을 호출하여 해당 단계의 작업을 수행합니다.
            logs(source.with_suffix('.csv'))
            # 함수 또는 처리 단계의 결과에 `render_video(source, output, source.with_suffix('.csv'))` 호출 결과를 저장합니다. 대상: `result`.
            result = render_video(source, output, source.with_suffix('.csv'))
            # 검증문 `self.assertEqual(result['frames'], 12)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['frames'], 12)
            # 검증문 `self.assertEqual(source.read_bytes(), original)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(source.read_bytes(), original)
            # OpenCV 영상 입력 장치에 `cv2.VideoCapture(str(output))` 호출 결과를 저장합니다. 대상: `cap`.
            cap = cv2.VideoCapture(str(output))
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # 검증문 `self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 12)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 12)
                # 검증문 `self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(cap.get(cv2.CAP_PROP_FPS), 10)
                # `(ok, frame)` 값에 `cap.read()` 호출 결과를 저장합니다. 대상: `(ok, frame)`.
                ok, frame = cap.read()
                # 검증문 `self.assertTrue(ok)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertTrue(ok)
                # 검증문 `self.assertGreater(frame.mean(), 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertGreater(frame.mean(), 1)
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
                cap.release()

    # 회귀 테스트: `test_discovery_excludes_outputs_and_infers_motion` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_discovery_excludes_outputs_and_infers_motion(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 최상위 경로에 `Path(directory)` 호출 결과를 저장합니다. 대상: `root`.
            root = Path(directory)
            # `('biceps_curl/a.mp4', 'biceps_curl/a_overlay.mp4', 'a.mp4.overlay.mp4', '_overlays/old/a.mp4')`에서 항목을 차례로 꺼내 `relative`에 넣고 아래 처리를 반복합니다.
            for relative in ('biceps_curl/a.mp4', 'biceps_curl/a_overlay.mp4',
                             # 앞줄에서 시작한 식·인수·자료 구조를 `'a.mp4.overlay.mp4', '_overlays/old/a.mp4'):` 내용으로 이어서 구성합니다.
                             'a.mp4.overlay.mp4', '_overlays/old/a.mp4'):
                # 경로에 계산식 `root / relative`의 결과를 저장합니다. 대상: `path`.
                path = root / relative
                # `path.parent.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
                path.parent.mkdir(parents=True, exist_ok=True)
                # `path.touch()`을 호출하여 해당 단계의 작업을 수행합니다.
                path.touch()
            # 검증문 `self.assertEqual(find_videos(root), [root / 'biceps_curl/a.mp4'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(find_videos(root), [root / 'biceps_curl/a.mp4'])
            # 검증문 `self.assertEqual(infer_exercise(root / 'leg_extension/a.mp4'), 'knee_extension')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(infer_exercise(root / 'leg_extension/a.mp4'), 'knee_extension')

    # 회귀 테스트: `test_cancel_removes_partial_output_and_never_overwrites_existing` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_cancel_removes_partial_output_and_never_overwrites_existing(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 최상위 경로에 `Path(directory)` 호출 결과를 저장합니다. 대상: `root`.
            root = Path(directory)
            # `(source, output)` 값에 순서·구성을 지정한 `(root / 'source.mp4', root / 'result.mp4')`을 저장합니다. 대상: `(source, output)`.
            source, output = root / 'source.mp4', root / 'result.mp4'
            # `video(source)`을 호출하여 해당 단계의 작업을 수행합니다.
            video(source)
            # `logs(source.with_suffix('.csv'))`을 호출하여 해당 단계의 작업을 수행합니다.
            logs(source.with_suffix('.csv'))
            # 취소에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `cancel`.
            cancel = threading.Event()
            # `cancel.set()`을 호출하여 스레드 신호를 활성화합니다.
            cancel.set()
            # `self.assertRaises(InterruptedError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.assertRaises(InterruptedError):
                # `render_video(source, output, source.with_suffix('.csv'), cancel=cancel)`을 호출하여 해당 단계의 작업을 수행합니다.
                render_video(source, output, source.with_suffix('.csv'), cancel=cancel)
            # 검증문 `self.assertFalse(output.exists())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertFalse(output.exists())
            # `output.write_bytes(b'existing')`을 호출하여 해당 단계의 작업을 수행합니다.
            output.write_bytes(b'existing')
            # `self.assertRaises(FileExistsError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.assertRaises(FileExistsError):
                # `render_video(source, output, source.with_suffix('.csv'))`을 호출하여 해당 단계의 작업을 수행합니다.
                render_video(source, output, source.with_suffix('.csv'))
            # 검증문 `self.assertEqual(output.read_bytes(), b'existing')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(output.read_bytes(), b'existing')

    # 회귀 테스트: `test_logged_job_does_not_need_socket_and_handles_dotted_filenames` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_logged_job_does_not_need_socket_and_handles_dotted_filenames(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 최상위 경로에 `Path(directory)` 호출 결과를 저장합니다. 대상: `root`.
            root = Path(directory)
            # 읽을 원본 경로 또는 데이터에 계산식 `root / 'biceps_curl/session.take2.mp4'`의 결과를 저장합니다. 대상: `source`.
            source = root / 'biceps_curl/session.take2.mp4'
            # `video(source)`을 호출하여 해당 단계의 작업을 수행합니다.
            video(source)
            # `logs(source.with_suffix('.csv'))`을 호출하여 해당 단계의 작업을 수행합니다.
            logs(source.with_suffix('.csv'))
            # `job` 값에 `VideoOverlayJob(root, 'ws://unreachable', analyze_missing=False)` 호출 결과를 저장합니다. 대상: `job`.
            job = VideoOverlayJob(root, 'ws://unreachable', analyze_missing=False)
            # `job.start()`을 호출하여 해당 단계의 작업을 수행합니다.
            job.start()
            # `job._thread.join(5)`을 호출하여 백그라운드 스레드의 종료를 기다림합니다.
            job._thread.join(5)
            # 처리 단계에 `job.snapshot()` 호출 결과를 저장합니다. 대상: `state`.
            state = job.snapshot()
            # 검증문 `self.assertFalse(state['running'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertFalse(state['running'])
            # 검증문 `self.assertEqual(state['results'][0]['status'], 'COMPLETED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(state['results'][0]['status'], 'COMPLETED')
            # 검증문 `self.assertTrue(Path(state['results'][0]['output']).is_file())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(Path(state['results'][0]['output']).is_file())
            # 검증문 `self.assertEqual(find_videos(root), [source])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(find_videos(root), [source])


# `SocketOverlayTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class SocketOverlayTests(unittest.IsolatedAsyncioTestCase):
    # 회귀 테스트: `test_missing_logs_are_requested_from_socket_server` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_missing_logs_are_requested_from_socket_server(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 최상위 경로에 `Path(directory)` 호출 결과를 저장합니다. 대상: `root`.
            root = Path(directory)
            # 읽을 원본 경로 또는 데이터에 계산식 `root / 'biceps_curl/source.mp4'`의 결과를 저장합니다. 대상: `source`.
            source = root / 'biceps_curl/source.mp4'
            # `video(source)`을 호출하여 해당 단계의 작업을 수행합니다.
            video(source)
            # csv 경로에 계산식 `root / 'analysis.csv'`의 결과를 저장합니다. 대상: `csv_path`.
            csv_path = root / 'analysis.csv'
            # `logs(csv_path)`을 호출하여 해당 단계의 작업을 수행합니다.
            logs(csv_path)
            # 메시지에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `packets`.
            packets = []

            # `handler` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
            async def handler(ws):
                # `ws.send(json.dumps({'type': 'SERVER_STATUS', 'ready': True}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await ws.send(json.dumps({'type': 'SERVER_STATUS', 'ready': True}))
                # `packets.append(json.loads(await ws.recv()))`을 호출하여 항목 하나를 끝에 추가합니다.
                packets.append(json.loads(await ws.recv()))
                # `ws.send(json.dumps({'type': 'VIDEO_ANALYSIS_COMPLETED', 'summary': {'frame_csv': str(csv_path), 'skeleton_npz': str(root / 'absent.npz')}}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await ws.send(json.dumps({'type': 'VIDEO_ANALYSIS_COMPLETED',
                    # 구성 항목: 저장·종료 결과 요약(`'summary'`)에 키 'frame_csv', 'skeleton_npz'로 구성한 사전; 프레임 csv(`'frame_csv'`)에 `str(csv_path)` 호출 결과; `skeleton_npz` 값(`'skeleton_npz'`)에 `str(root / 'absent.npz')` 호출 결과를 지정합니다.
                    'summary': {'frame_csv': str(csv_path), 'skeleton_npz': str(root / 'absent.npz')}}))
                # `ws.wait_closed()`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await ws.wait_closed()

            # `websockets.serve(handler, '127.0.0.1', 0)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            async with websockets.serve(handler, '127.0.0.1', 0) as server:
                # WebSocket 접속 주소에 계산식 `f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'`의 결과를 저장합니다. 대상: `uri`.
                uri = f'ws://127.0.0.1:{server.sockets[0].getsockname()[1]}'
                # `job` 값에 `VideoOverlayJob(root, uri)` 호출 결과를 저장합니다. 대상: `job`.
                job = VideoOverlayJob(root, uri)
                # `job.start()`을 호출하여 해당 단계의 작업을 수행합니다.
                job.start()
                # `asyncio.to_thread(job._thread.join, 10)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await asyncio.to_thread(job._thread.join, 10)
                # 검증문 `self.assertFalse(job.snapshot()['running'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertFalse(job.snapshot()['running'])
                # 검증문 `self.assertEqual(job.snapshot()['results'][0]['status'], 'COMPLETED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(job.snapshot()['results'][0]['status'], 'COMPLETED')
                # 검증문 `self.assertEqual(packets[0]['video_path'], str(source))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(packets[0]['video_path'], str(source))
                # 검증문 `self.assertEqual(packets[0]['mode'], 'TEST')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(packets[0]['mode'], 'TEST')
                # 검증문 `self.assertTrue(packets[0]['record_video'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertTrue(packets[0]['record_video'])


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
