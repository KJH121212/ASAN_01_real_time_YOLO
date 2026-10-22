# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# `types`에서 `SimpleNamespace` 기능을 가져옵니다.
from types import SimpleNamespace
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# `unittest.mock`에서 `AsyncMock`, `Mock`, `patch` 기능을 가져옵니다.
from unittest.mock import AsyncMock, Mock, patch

# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np
# `websockets.exceptions`에서 `ConnectionClosedOK` 기능을 가져옵니다.
from websockets.exceptions import ConnectionClosedOK

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `network.socket_server`에서 `ExerciseSocketServer`, `ThreadedWebcam` 기능을 가져옵니다.
from network.socket_server import ExerciseSocketServer, ThreadedWebcam
# `network.session_commands`에서 `DEFAULT_PREPARATION_SECONDS`, `get_preparation_seconds` 기능을 가져옵니다.
from network.session_commands import DEFAULT_PREPARATION_SECONDS, get_preparation_seconds


# `CaptureTimestampTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class CaptureTimestampTests(unittest.TestCase):
    # 회귀 테스트: `test_phone_driver_can_fall_back_to_media_foundation` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_phone_driver_can_fall_back_to_media_foundation(self):
        # `(first, second)` 값에 순서·구성을 지정한 `(Mock(), Mock())`을 저장합니다. 대상: `(first, second)`.
        first, second = Mock(), Mock()
        # return 측정값에 False을 저장합니다. 대상: `first.isOpened.return_value`.
        first.isOpened.return_value = False
        # return 측정값에 True을 저장합니다. 대상: `second.isOpened.return_value`.
        second.isOpened.return_value = True
        # return 측정값에 순서·구성을 지정한 `(True, np.zeros((48, 24, 3), np.uint8))`을 저장합니다. 대상: `second.read.return_value`.
        second.read.return_value = (True, np.zeros((48, 24, 3), np.uint8))
        # `patch('network.socket_server.cv2.VideoCapture', side_effect=[first, second])`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch('network.socket_server.cv2.VideoCapture', side_effect=[first, second]) as opened:
            # `camera` 값에 `ThreadedWebcam(2)` 호출 결과를 저장합니다. 대상: `camera`.
            camera = ThreadedWebcam(2)
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 검증문 `self.assertIs(camera.cap, second)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertIs(camera.cap, second)
            # 검증문 `first.release.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            first.release.assert_called_once()
            # 검증문 `self.assertEqual(opened.call_args_list[0].args, (2, cv2.CAP_DSHOW))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(opened.call_args_list[0].args, (2, cv2.CAP_DSHOW))
            # 검증문 `self.assertEqual(opened.call_args_list[1].args, (2, cv2.CAP_MSMF))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(opened.call_args_list[1].args, (2, cv2.CAP_MSMF))
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `camera.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            camera.stop()

    # 회귀 테스트: `test_camera_warmup_retries_before_switching_backend` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_camera_warmup_retries_before_switching_backend(self):
        # OpenCV 영상 입력 장치에 `Mock()` 호출 결과를 저장합니다. 대상: `cap`.
        cap = Mock()
        # return 측정값에 True을 저장합니다. 대상: `cap.isOpened.return_value`.
        cap.isOpened.return_value = True
        # `cap.read.side_effect` 값에 순서·구성을 지정한 `[(False, None), (True, np.empty((0, 0, 3))), (True, np.zeros((48, 24, 3), np.uint8))]`을 저장합니다. 대상: `cap.read.side_effect`.
        cap.read.side_effect = [(False, None), (True, np.empty((0, 0, 3))),
                                # 앞줄에서 시작한 식·인수·자료 구조를 `(True, np.zeros((48, 24, 3), np.uint8))]` 내용으로 이어서 구성합니다.
                                (True, np.zeros((48, 24, 3), np.uint8))]
        # `patch('network.socket_server.cv2.VideoCapture', return_value=cap), patch('network.socket_server.time.sleep')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch('network.socket_server.cv2.VideoCapture', return_value=cap) as opened, \
                patch('network.socket_server.time.sleep'):
            # `camera` 값에 `ThreadedWebcam()` 호출 결과를 저장합니다. 대상: `camera`.
            camera = ThreadedWebcam()
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 검증문 `opened.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            opened.assert_called_once()
            # 검증문 `self.assertEqual(camera.read_sample()[1].shape, (48, 24, 3))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(camera.read_sample()[1].shape, (48, 24, 3))
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `camera.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            camera.stop()

    # 회귀 테스트: `test_no_camera_fails_explicitly_and_releases_both_backends` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_no_camera_fails_explicitly_and_releases_both_backends(self):
        # `caps` 값에 순서·구성을 지정한 `[Mock(), Mock()]`을 저장합니다. 대상: `caps`.
        caps = [Mock(), Mock()]
        # `caps`에서 항목을 차례로 꺼내 `cap`에 넣고 아래 처리를 반복합니다.
        for cap in caps:
            # return 측정값에 True을 저장합니다. 대상: `cap.isOpened.return_value`.
            cap.isOpened.return_value = True
            # return 측정값에 순서·구성을 지정한 `(False, None)`을 저장합니다. 대상: `cap.read.return_value`.
            cap.read.return_value = (False, None)
        # `patch('network.socket_server.cv2.VideoCapture', side_effect=caps), patch('network.socket_server.time.sleep'), self.assertRaisesRegex(RuntimeError, '카메라 3')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch('network.socket_server.cv2.VideoCapture', side_effect=caps), \
                patch('network.socket_server.time.sleep'), self.assertRaisesRegex(RuntimeError, '카메라 3'):
            # `ThreadedWebcam(3)`을 호출하여 해당 단계의 작업을 수행합니다.
            ThreadedWebcam(3)
        # `caps`에서 항목을 차례로 꺼내 `cap`에 넣고 아래 처리를 반복합니다.
        for cap in caps:
            # 검증문 `cap.release.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            cap.release.assert_called_once()

    # 회귀 테스트: `test_native_read_error_is_reported_to_the_server_loop` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_native_read_error_is_reported_to_the_server_loop(self):
        # OpenCV 영상 입력 장치에 `Mock()` 호출 결과를 저장합니다. 대상: `cap`.
        cap = Mock()
        # return 측정값에 True을 저장합니다. 대상: `cap.isOpened.return_value`.
        cap.isOpened.return_value = True
        # return 측정값에 순서·구성을 지정한 `(True, np.zeros((48, 24, 3), np.uint8))`을 저장합니다. 대상: `cap.read.return_value`.
        cap.read.return_value = (True, np.zeros((48, 24, 3), np.uint8))
        # `patch('network.socket_server.cv2.VideoCapture', return_value=cap)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch('network.socket_server.cv2.VideoCapture', return_value=cap):
            # `camera` 값에 `ThreadedWebcam()` 호출 결과를 저장합니다. 대상: `camera`.
            camera = ThreadedWebcam()
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 동작 시작 시각 또는 실행 여부에 True을 저장합니다. 대상: `camera.started`.
            camera.started = True
            # `cap.read.side_effect` 값에 `cv2.error('phone disconnected')` 호출 결과를 저장합니다. 대상: `cap.read.side_effect`.
            cap.read.side_effect = cv2.error('phone disconnected')
            # `camera.update()`을 호출하여 해당 단계의 작업을 수행합니다.
            camera.update()
            # `self.assertRaisesRegex(RuntimeError, 'phone disconnected')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.assertRaisesRegex(RuntimeError, 'phone disconnected'):
                # `camera.read_sample()`을 호출하여 해당 단계의 작업을 수행합니다.
                camera.read_sample()
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `camera.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            camera.stop()

    # 회귀 테스트: `test_capture_time_is_attached_at_read_and_remains_stable_for_consumers` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_capture_time_is_attached_at_read_and_remains_stable_for_consumers(self):
        # OpenCV BGR 영상 프레임에 `np.zeros((24, 32, 3), dtype=np.uint8)` 호출 결과를 저장합니다. 대상: `frame`.
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        # OpenCV 영상 입력 장치에 `Mock()` 호출 결과를 저장합니다. 대상: `cap`.
        cap = Mock()
        # return 측정값에 순서·구성을 지정한 `(True, frame)`을 저장합니다. 대상: `cap.read.return_value`.
        cap.read.return_value = (True, frame)
        # `patch('network.socket_server.cv2.VideoCapture', return_value=cap), patch('network.socket_server.time.perf_counter', return_value=10.25)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch('network.socket_server.cv2.VideoCapture', return_value=cap), \
                patch('network.socket_server.time.perf_counter', return_value=10.25):
            # `camera` 값에 `ThreadedWebcam()` 호출 결과를 저장합니다. 대상: `camera`.
            camera = ThreadedWebcam()
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `first` 값에 `camera.read_sample()` 호출 결과를 저장합니다. 대상: `first`.
            first = camera.read_sample()
            # `second` 값에 `camera.read_sample()` 호출 결과를 저장합니다. 대상: `second`.
            second = camera.read_sample()
            # 검증문 `self.assertEqual(first[2:], (10.25, 0))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(first[2:], (10.25, 0))
            # 검증문 `self.assertEqual(second[2:], first[2:])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(second[2:], first[2:])
            # `first[1][:]` 값에 255을 저장합니다. 대상: `first[1][:]`.
            first[1][:] = 255
            # 검증문 `self.assertEqual(int(second[1].max()), 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(int(second[1].max()), 0)

            # `capture_next` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
            def capture_next():
                # 동작 시작 시각 또는 실행 여부에 False을 저장합니다. 대상: `camera.started`.
                camera.started = False  # Run exactly one capture iteration.
                # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(True, np.full_like(frame, 128))`을 호출한 쪽으로 반환합니다.
                return True, np.full_like(frame, 128)

            # 동작 시작 시각 또는 실행 여부에 True을 저장합니다. 대상: `camera.started`.
            camera.started = True
            # `cap.read.side_effect` 값에 `capture_next` 값(`capture_next`)을 저장합니다. 대상: `cap.read.side_effect`.
            cap.read.side_effect = capture_next
            # `patch('network.socket_server.time.perf_counter', return_value=10.45)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with patch('network.socket_server.time.perf_counter', return_value=10.45):
                # `camera.update()`을 호출하여 해당 단계의 작업을 수행합니다.
                camera.update()
            # `latest` 값에 `camera.read_sample()` 호출 결과를 저장합니다. 대상: `latest`.
            latest = camera.read_sample()
            # 검증문 `self.assertEqual(latest[2:], (10.45, 1))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(latest[2:], (10.45, 1))
            # 검증문 `self.assertEqual(int(latest[1].max()), 128)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(int(latest[1].max()), 128)
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `camera.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            camera.stop()


# `LiveVideoTimingTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class LiveVideoTimingTests(unittest.IsolatedAsyncioTestCase):
    # 회귀 테스트: `test_live_initialization_uses_default_delay_and_zero_disables_gate` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_live_initialization_uses_default_delay_and_zero_disables_gate(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # `(('MAIN', None), ('CALIBRATION', None), ('TEST', 0))`에서 항목을 차례로 꺼내 `(mode, delay)`에 넣고 아래 처리를 반복합니다.
            for mode, delay in (('MAIN', None), ('CALIBRATION', None), ('TEST', 0)):
                # `self.subTest(mode=mode, delay=delay)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with self.subTest(mode=mode, delay=delay):
                    # 서버에 `ExerciseSocketServer(engine_factory=lambda report: object())` 호출 결과를 저장합니다. 대상: `server`.
                    server = ExerciseSocketServer(engine_factory=lambda report: object())
                    # 시작 camera에 `Mock()` 호출 결과를 저장합니다. 대상: `server.start_camera`.
                    server.start_camera = Mock()
                    # 수신 또는 전송할 JSON 메시지 사전에 키 'mode', 'log_root'로 구성한 사전을 저장합니다. 대상: `packet`.
                    packet = {'mode': mode, 'log_root': directory}
                    # 조건 `delay is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if delay is not None:
                        # `packet['preparation_seconds']` 값에 종료 요청 전에 기다리는 시간(초)(`delay`)을 저장합니다. 대상: `packet['preparation_seconds']`.
                        packet['preparation_seconds'] = delay
                    # `patch('network.socket_server.time.perf_counter', return_value=100.0)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                    with patch('network.socket_server.time.perf_counter', return_value=100.0):
                        # `server.init_live_session(packet)`을 호출하여 해당 단계의 작업을 수행합니다.
                        server.init_live_session(packet)
                        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                        try:
                            # 테스트에서 기대하는 결과에 `delay is None`이면 `DEFAULT_PREPARATION_SECONDS`, 아니면 `delay`을 저장합니다. 대상: `expected`.
                            expected = DEFAULT_PREPARATION_SECONDS if delay is None else delay
                            # 검증문 `self.assertEqual(server._session_started, 100.0 + expected)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                            self.assertEqual(server._session_started, 100.0 + expected)
                            # 검증문 `self.assertEqual(server._measurement_ready_at, 100.0 + expected if expected else None)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                            self.assertEqual(server._measurement_ready_at, 100.0 + expected if expected else None)
                            # 검증문 `server.start_camera.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                            server.start_camera.assert_called_once()
                            # 검증문 `self.assertEqual(server.controller.motion_engine.fsm_left.rep_count, 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                            self.assertEqual(server.controller.motion_engine.fsm_left.rep_count, 0)
                        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
                        finally:
                            # `server._save_session_files(status='STOPPED')`을 호출하여 해당 단계의 작업을 수행합니다.
                            server._save_session_files(status='STOPPED')

    # 회귀 테스트: `test_capture_timing_and_disconnect_reach_actual_mp4_and_npz` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_capture_timing_and_disconnect_reach_actual_mp4_and_npz(self):
        # `self.run_recording(preparation_seconds=0)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.run_recording(preparation_seconds=0)

    # 회귀 테스트: `test_preparation_frames_never_reach_inference_logs_or_video` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_preparation_frames_never_reach_inference_logs_or_video(self):
        # `self.run_recording(preparation_seconds=5)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.run_recording(preparation_seconds=5)

    # 실행 녹화 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def run_recording(self, preparation_seconds):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 서버에 `ExerciseSocketServer(engine_factory=lambda report: object())` 호출 결과를 저장합니다. 대상: `server`.
            server = ExerciseSocketServer(engine_factory=lambda report: object())
            # `manager` 값에 `server._prepare_data_manager({'patient_id': '0001', 'mode': 'MAIN', 'record_video': True, 'log_root': directory}, live_timing=True)` 호출 결과를 저장합니다. 대상: `manager`.
            manager = server._prepare_data_manager({
                # 구성 항목: 환자 식별 번호(`'patient_id'`)에 '0001'; 실행 모드(`'mode'`)에 'MAIN'; 영상 녹화 여부(`'record_video'`)에 True을 지정합니다.
                'patient_id': '0001', 'mode': 'MAIN', 'record_video': True,
                # 구성 항목: 세션 로그를 저장하는 최상위 폴더(`'log_root'`)에 폴더(`directory`); 실시간 시각 동기화(`live_timing`)에 True을 지정합니다.
                'log_root': directory}, live_timing=True)
            # 측정 세션 started에 계산식 `100.0 + preparation_seconds`의 결과를 저장합니다. 대상: `server._session_started`.
            server._session_started = 100.0 + preparation_seconds
            # 조건 `preparation_seconds`을 만족하는 경우에만 아래 분기를 실행합니다.
            if preparation_seconds:
                # measurement 준비 상태 at에 측정 세션 started(`server._session_started`)을 저장합니다. 대상: `server._measurement_ready_at`.
                server._measurement_ready_at = server._session_started
            # 현재 작업 설정에 키 'eval_config', 'default_thresholds'로 구성한 사전을 저장합니다. 대상: `config`.
            config = {'eval_config': {'metric_type': 'RELATIVE_Y', 'motion_direction': 'INCREASING'},
                      # 구성 항목: 기본 가동 범위 기준(`'default_thresholds'`)에 반복·조건식 `{side: {'start_val': 0.1, 'target_val': 0.9} for side in ('left', 'right')}`으로 구성한 값; 개인 가동 범위의 시작값(`'start_val'`)에 0.1; 개인 가동 범위의 목표값(`'target_val'`)에 0.9을 지정합니다.
                      'default_thresholds': {side: {'start_val': 0.1, 'target_val': 0.9}
                                             # 앞줄에서 시작한 식·인수·자료 구조를 `for side in ('left', 'right')}}` 내용으로 이어서 구성합니다.
                                             for side in ('left', 'right')}}
            # 생성 또는 검증할 분석 엔진에 `MotionEngine(config, mode='MAIN')` 호출 결과를 저장합니다. 대상: `engine`.
            engine = MotionEngine(config, mode='MAIN')
            # 운동 설정의 식별 이름에 'biceps_curl'을 저장합니다. 대상: `engine.exercise_name`.
            engine.exercise_name = 'biceps_curl'
            # 좌우 구분에 키 'val', 'rep_count'로 구성한 사전을 저장합니다. 대상: `side`.
            side = {'val': 0.2, 'rep_count': 0}
            # 캘리브레이션·본 운동 진행 제어기에 `SimpleNamespace(data_manager=manager, motion_engine=engine, calib_step='', last_left_data=side, last_right_data=side, process_main_frame=Mock(return_value=(side, side, F…` 호출 결과를 저장합니다. 대상: `server.controller`.
            server.controller = SimpleNamespace(
                # 구성 항목: 환자 정보 및 세션 로그 관리자(`data_manager`)에 `manager` 값(`manager`); 관절 측정값 계산 및 반복 판정 엔진(`motion_engine`)에 생성 또는 검증할 분석 엔진(`engine`); 캘리브레이션 전체 진행 단계(`calib_step`)에 ''을 지정합니다.
                data_manager=manager, motion_engine=engine, calib_step='',
                # 구성 항목: last 왼쪽 데이터(`last_left_data`)에 좌우 구분(`side`); last 오른쪽 데이터(`last_right_data`)에 좌우 구분(`side`)을 지정합니다.
                last_left_data=side, last_right_data=side,
                # 구성 항목: 처리 진입점 프레임(`process_main_frame`)에 `Mock(return_value=(side, side, False))` 호출 결과; return 측정값(`return_value`)에 순서·구성을 지정한 `(side, side, False)`을 지정합니다.
                process_main_frame=Mock(return_value=(side, side, False)))
            # 평활화 엔진에 `SimpleNamespace(update=lambda points, timestamp: None)` 호출 결과를 저장합니다. 대상: `server.filter_engine`.
            server.filter_engine = SimpleNamespace(update=lambda points, timestamp: None)
            # 사람 검출·관절 추출 AI 엔진에 `SimpleNamespace(extract_keypoints=Mock(return_value=None))` 호출 결과를 저장합니다. 대상: `server.skeleton_engine`.
            server.skeleton_engine = SimpleNamespace(extract_keypoints=Mock(return_value=None))
            # `inference` 값에 extract 관절(`server.skeleton_engine.extract_keypoints`)을 저장합니다. 대상: `inference`.
            inference = server.skeleton_engine.extract_keypoints
            # 백그라운드 AI 모델 준비 관리자에 `SimpleNamespace(start=lambda : None, snapshot=lambda : {'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})` 호출 결과를 저장합니다. 대상: `server.model_loader`.
            server.model_loader = SimpleNamespace(start=lambda: None, snapshot=lambda: {
                # 구성 항목: `type` 값(`'type'`)에 'SERVER_STATUS'; 모델 단계(`'model_state'`)에 'READY'; `phase` 값(`'phase'`)에 'READY'; 준비 상태(`'ready'`)에 True을 지정합니다.
                'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})
            # OpenCV BGR 영상 프레임에 `np.zeros((24, 32, 3), dtype=np.uint8)` 호출 결과를 저장합니다. 대상: `frame`.
            frame = np.zeros((24, 32, 3), dtype=np.uint8)
            # `origin` 값에 측정 세션 started(`server._session_started`)을 저장합니다. 대상: `origin`.
            origin = server._session_started
            # `server.camera` 값에 `SimpleNamespace(read_sample=Mock(side_effect=[(True, frame, origin + 0.5, 1), (True, frame, origin + 0.5, 1), (True, frame, origin + 0.7, 2), (True, frame, origin + 1.0,…` 호출 결과를 저장합니다. 대상: `server.camera`.
            server.camera = SimpleNamespace(read_sample=Mock(side_effect=[
                # 앞줄에서 시작한 식·인수·자료 구조를 `(True, frame, origin + 0.5, 1), (True, frame, origin + 0.5, 1),` 내용으로 이어서 구성합니다.
                (True, frame, origin + 0.5, 1), (True, frame, origin + 0.5, 1),
                # 구성 항목: 중지(`stop`)에 `Mock()` 호출 결과를 지정합니다.
                (True, frame, origin + 0.7, 2), (True, frame, origin + 1.0, 3)]), stop=Mock())
            # `camera` 값에 `server.camera` 값(`server.camera`)을 저장합니다. 대상: `camera`.
            camera = server.camera
            # 출력 mode에 'LIGHTWEIGHT'을 저장합니다. 대상: `server.output_mode`.
            server.output_mode = 'LIGHTWEIGHT'
            # `clock` 값에 키 'now'로 구성한 사전을 저장합니다. 대상: `clock`.
            clock = {'now': 100.0}
            # `steps` 값에 `iter(([100.0, 104.0] if preparation_seconds else []) + [origin + 0.5, origin + 0.6, origin + 0.7, origin + 1.0, origin + 2.0])` 호출 결과를 저장합니다. 대상: `steps`.
            steps = iter(([100.0, 104.0] if preparation_seconds else []) +
                         # 앞줄에서 시작한 식·인수·자료 구조를 `[origin + 0.5, origin + 0.6, origin + 0.7, origin + 1.0, origin + 2.0])` 내용으로 이어서 구성합니다.
                         [origin + 0.5, origin + 0.6, origin + 0.7, origin + 1.0, origin + 2.0])

            # 수신 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
            async def receive():
                # `clock['now']` 값에 `next(steps)` 호출 결과를 저장합니다. 대상: `clock['now']`.
                clock['now'] = next(steps)
                # 조건 `clock['now'] == origin + 2.0`을 만족하는 경우에만 아래 분기를 실행합니다.
                if clock['now'] == origin + 2.0:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `ConnectionClosedOK(None, None)` 예외를 발생시킵니다.
                    raise ConnectionClosedOK(None, None)
                # 이 조건에서는 처리를 계속할 수 없으므로 `asyncio.TimeoutError` 예외를 발생시킵니다.
                raise asyncio.TimeoutError

            # 현재 클라이언트의 WebSocket 연결에 `SimpleNamespace(remote_address=('test', 0), send=AsyncMock(), recv=receive)` 호출 결과를 저장합니다. 대상: `websocket`.
            websocket = SimpleNamespace(
                # 구성 항목: `remote_address` 값(`remote_address`)에 순서·구성을 지정한 `('test', 0)`; 전송(`send`)에 `AsyncMock()` 호출 결과를 지정합니다.
                remote_address=('test', 0), send=AsyncMock(),
                # 구성 항목: `recv` 값(`recv`)에 수신(`receive`)을 지정합니다.
                recv=receive)
            # 현재 서버의 측정 자원을 점유한 연결에 현재 클라이언트의 WebSocket 연결(`websocket`)을 저장합니다. 대상: `server._session_owner`.
            server._session_owner = websocket
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `patch('network.socket_server.time.perf_counter', side_effect=lambda : clock['now'])`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with patch('network.socket_server.time.perf_counter', side_effect=lambda: clock['now']):
                    # `server.handle_client(websocket)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                    await server.handle_client(websocket)
                # 동일한 캡처 프레임을 반복 조회해도 중복 추론·기록하지 않는지 확인합니다.
                # 검증문 `self.assertEqual(inference.call_count, 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(inference.call_count, 3)
                # 검증문 `self.assertEqual(camera.read_sample.call_count, 4)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(camera.read_sample.call_count, 4)
                # 메시지에 반복·조건식 `[json.loads(call.args[0]) for call in websocket.send.call_args_list]`으로 구성한 값을 저장합니다. 대상: `packets`.
                packets = [json.loads(call.args[0]) for call in websocket.send.call_args_list]
                # `countdowns` 값에 반복·조건식 `[packet['remaining_seconds'] for packet in packets if packet['type'] == 'SESSION_PREPARING']`으로 구성한 값을 저장합니다. 대상: `countdowns`.
                countdowns = [packet['remaining_seconds'] for packet in packets
                              # 앞줄에서 시작한 식·인수·자료 구조를 `if packet['type'] == 'SESSION_PREPARING']` 내용으로 이어서 구성합니다.
                              if packet['type'] == 'SESSION_PREPARING']
                # 검증문 `self.assertEqual(countdowns, [5, 1] if preparation_seconds else [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(countdowns, [5, 1] if preparation_seconds else [])
                # `np.load(manager.session_log.info['skeleton_npz'])`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with np.load(manager.session_log.info['skeleton_npz']) as data:
                    # 검증문 `np.testing.assert_allclose(data['timestamps'], [0.5, 0.7, 1.0])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    np.testing.assert_allclose(data['timestamps'], [0.5, 0.7, 1.0])
                    # 검증문 `np.testing.assert_array_equal(data['video_frame_idx'], [0, 6, 15])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    np.testing.assert_array_equal(data['video_frame_idx'], [0, 6, 15])
                # 세션 파일 위치와 상태를 담은 메타데이터에 JSON 문자열을 해석한 Python 값(`json.loads(Path(manager.session_log.info['manifest_json']).read_text(encoding='utf-8'))`)을 저장합니다. 대상: `manifest`.
                manifest = json.loads(Path(manager.session_log.info['manifest_json']).read_text(encoding='utf-8'))
                # 검증문 `self.assertEqual(manifest['status'], 'STOPPED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(manifest['status'], 'STOPPED')
                # 검증문 `self.assertAlmostEqual(manifest['video_duration_sec'], 1.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertAlmostEqual(manifest['video_duration_sec'], 1.5)
                # OpenCV 영상 입력 장치에 `cv2.VideoCapture(manifest['video_mp4'])` 호출 결과를 저장합니다. 대상: `cap`.
                cap = cv2.VideoCapture(manifest['video_mp4'])
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # 검증문 `self.assertTrue(cap.isOpened())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertTrue(cap.isOpened())
                    # 검증문 `self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 1.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS), 1.5)
                # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
                finally:
                    # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
                    cap.release()
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # `manager.finish_session([], status='TEST')`을 호출하여 해당 단계의 작업을 수행합니다.
                manager.finish_session([], status='TEST')

    # 회귀 테스트: `test_stop_during_preparation_saves_empty_logs_with_zero_duration` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_stop_during_preparation_saves_empty_logs_with_zero_duration(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 서버에 `ExerciseSocketServer(engine_factory=lambda report: object())` 호출 결과를 저장합니다. 대상: `server`.
            server = ExerciseSocketServer(engine_factory=lambda report: object())
            # `manager` 값에 `server._prepare_data_manager({'mode': 'MAIN', 'log_root': directory}, live_timing=True)` 호출 결과를 저장합니다. 대상: `manager`.
            manager = server._prepare_data_manager({
                # 구성 항목: 실행 모드(`'mode'`)에 'MAIN'; 세션 로그를 저장하는 최상위 폴더(`'log_root'`)에 폴더(`directory`); 실시간 시각 동기화(`live_timing`)에 True을 지정합니다.
                'mode': 'MAIN', 'log_root': directory}, live_timing=True)
            # 측정 세션 started에 105.0을 저장합니다. 대상: `server._session_started`.
            server._session_started = 105.0
            # measurement 준비 상태 at에 105.0을 저장합니다. 대상: `server._measurement_ready_at`.
            server._measurement_ready_at = 105.0
            # 생성 또는 검증할 분석 엔진에 `SimpleNamespace(fsm_left=SimpleNamespace(completed_reps_history=[]), fsm_right=SimpleNamespace(completed_reps_history=[]))` 호출 결과를 저장합니다. 대상: `engine`.
            engine = SimpleNamespace(fsm_left=SimpleNamespace(completed_reps_history=[]),
                                     # 구성 항목: 오른쪽 반복 판정 상태 기계(`fsm_right`)에 `SimpleNamespace(completed_reps_history=[])` 호출 결과; 완료된 반복별 분석 기록(`completed_reps_history`)에 순서·구성을 지정한 `[]`을 지정합니다.
                                     fsm_right=SimpleNamespace(completed_reps_history=[]))
            # 캘리브레이션·본 운동 진행 제어기에 `SimpleNamespace(data_manager=manager, motion_engine=engine)` 호출 결과를 저장합니다. 대상: `server.controller`.
            server.controller = SimpleNamespace(data_manager=manager, motion_engine=engine)
            # `server.camera` 값에 `SimpleNamespace(read_sample=Mock(), stop=Mock())` 호출 결과를 저장합니다. 대상: `server.camera`.
            server.camera = SimpleNamespace(read_sample=Mock(), stop=Mock())
            # `camera` 값에 `server.camera` 값(`server.camera`)을 저장합니다. 대상: `camera`.
            camera = server.camera
            # 백그라운드 AI 모델 준비 관리자에 `SimpleNamespace(start=lambda : None, snapshot=lambda : {'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})` 호출 결과를 저장합니다. 대상: `server.model_loader`.
            server.model_loader = SimpleNamespace(start=lambda: None, snapshot=lambda: {
                # 구성 항목: `type` 값(`'type'`)에 'SERVER_STATUS'; 모델 단계(`'model_state'`)에 'READY'; `phase` 값(`'phase'`)에 'READY'; 준비 상태(`'ready'`)에 True을 지정합니다.
                'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})
            # 현재 클라이언트의 WebSocket 연결에 `SimpleNamespace(remote_address=('test', 0), send=AsyncMock(), recv=AsyncMock(side_effect=[asyncio.TimeoutError, ConnectionClosedOK(None, None)]))` 호출 결과를 저장합니다. 대상: `websocket`.
            websocket = SimpleNamespace(remote_address=('test', 0), send=AsyncMock(),
                # 구성 항목: `recv` 값(`recv`)에 `AsyncMock(side_effect=[asyncio.TimeoutError, ConnectionClosedOK(None, None)])` 호출 결과; `side_effect` 값(`side_effect`)에 순서·구성을 지정한 `[asyncio.TimeoutError, ConnectionClosedOK(None, None)]`을 지정합니다.
                recv=AsyncMock(side_effect=[asyncio.TimeoutError, ConnectionClosedOK(None, None)]))
            # 현재 서버의 측정 자원을 점유한 연결에 현재 클라이언트의 WebSocket 연결(`websocket`)을 저장합니다. 대상: `server._session_owner`.
            server._session_owner = websocket
            # `patch('network.socket_server.time.perf_counter', return_value=100.0)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with patch('network.socket_server.time.perf_counter', return_value=100.0):
                # `server.handle_client(websocket)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await server.handle_client(websocket)
            # 검증문 `camera.read_sample.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            camera.read_sample.assert_not_called()
            # 검증문 `self.assertEqual(manager.session_log._summary['saved_frames'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(manager.session_log._summary['saved_frames'], 0)
            # 검증문 `self.assertEqual(manager.session_log._summary['duration_sec'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(manager.session_log._summary['duration_sec'], 0)


# `PreparationSettingTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class PreparationSettingTests(unittest.TestCase):
    # 회귀 테스트: `test_default_and_zero_disable` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_default_and_zero_disable(self):
        # 검증문 `self.assertEqual(get_preparation_seconds({}), DEFAULT_PREPARATION_SECONDS)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(get_preparation_seconds({}), DEFAULT_PREPARATION_SECONDS)
        # 검증문 `self.assertEqual(get_preparation_seconds({'preparation_seconds': 0}), 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(get_preparation_seconds({'preparation_seconds': 0}), 0)
        # 검증문 `self.assertEqual(get_preparation_seconds({'preparation_seconds': 2.5}), 2.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(get_preparation_seconds({'preparation_seconds': 2.5}), 2.5)

    # 회귀 테스트: `test_invalid_delay_is_rejected` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_invalid_delay_is_rejected(self):
        # `(-1, True, '5', None, float('nan'), float('inf'))`에서 항목을 차례로 꺼내 `value`에 넣고 아래 처리를 반복합니다.
        for value in (-1, True, '5', None, float('nan'), float('inf')):
            # `self.subTest(value=value), self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.subTest(value=value), self.assertRaises(ValueError):
                # `get_preparation_seconds({'preparation_seconds': value})`을 호출하여 해당 단계의 작업을 수행합니다.
                get_preparation_seconds({'preparation_seconds': value})


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
