# 한국어 코드 해설: WebSocket 명령을 받아 실시간 카메라 또는 영상 파일을 분석하고 상태·반복 이벤트 및 세션 로그를 생성합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [모듈 정보]
# 파일: network/socket_server.py
# 설명: 다음 입력 방식을 지원하는 WebSocket 측정 서버입니다.
# 모드 2(LIGHTWEIGHT): 실시간 웹캠을 분석하고 Unity용 측정값을 전송합니다.
# 모드 3(OFFLINE_VIDEO): 영상 파일을 GPU 배치 추론으로 분석합니다.
# ==============================================================================

# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os
os.environ["MMCV_WITH_OPS"] = "0"

import asyncio
import json
import sys
import threading
import time
_BOOT_STARTED = time.perf_counter()
import traceback
import cv2
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_file_path = os.path.abspath(__file__)
network_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(network_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.data_manager import DataManager, load_exercise_config
from core.motion_engine import MotionEngine
from core.model_loader import BackgroundModelLoader
from core.session_controller import SessionController
from utils.filters import OneEuroKeypointFilter
from utils.normalization import PoseNormalizer
from network.session_commands import COUNT_THRESHOLD_COMMANDS, handle_count_threshold_command, get_preparation_seconds, get_target_score


# `ThreadedWebcam` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ThreadedWebcam:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """실시간 웹캠 최신 프레임 캡처 스레드 (모드 2)"""
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, cam_idx=0):
        # OpenCV 영상 입력 장치에 None을 저장합니다. 대상: `self.cap`.
        self.cap = None
        # 휴대폰·가상 웹캠 드라이버는 Windows 입력 백엔드 중 하나만 지원할 수도 있습니다.
        # `(cv2.CAP_DSHOW, cv2.CAP_MSMF)`에서 항목을 차례로 꺼내 `backend`에 넣고 아래 처리를 반복합니다.
        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            # OpenCV 영상 입력 장치에 `cv2.VideoCapture(cam_idx, backend)` 호출 결과를 저장합니다. 대상: `cap`.
            cap = cv2.VideoCapture(cam_idx, backend)
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # 조건 `cap.isOpened()`을 만족하는 경우에만 아래 분기를 실행합니다.
                if cap.isOpened():
                    # `cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)`을 호출하여 스레드 신호를 활성화합니다.
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    # `range(5)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
                    for _ in range(5):
                        # `(grabbed, frame)` 값에 `cap.read()` 호출 결과를 저장합니다. 대상: `(grabbed, frame)`.
                        grabbed, frame = cap.read()
                        # 조건 `self._valid_frame(grabbed, frame)`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if self._valid_frame(grabbed, frame):
                            # `(self.cap, self.grabbed, self.frame)` 값에 순서·구성을 지정한 `(cap, grabbed, frame)`을 저장합니다. 대상: `(self.cap, self.grabbed, self.frame)`.
                            self.cap, self.grabbed, self.frame = cap, grabbed, frame
                            # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                            break
                        # `time.sleep(0.05)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다.
                        time.sleep(0.05)
            # `cv2.error` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except cv2.error as exc:
                # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                print(f'[WARN][CAMERA] Backend {backend} failed: {exc}', flush=True)
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # 조건 `self.cap is not cap`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self.cap is not cap:
                    # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
                    cap.release()
            # 조건 `self.cap is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.cap is not None:
                # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                break
        # 조건 `self.cap is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.cap is None:
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(f'카메라 {cam_idx}에서 영상을 읽을 수 없습니다 (DirectShow/Media Foundation). 휴대폰 웹캠 앱의 연결 상태와 카메라 번호를 확인하고, 카메라를 사용하는 다른 앱을 닫아 주세요.')` 예외를 발생시킵니다.
            raise RuntimeError(
                # 앞줄에서 시작한 식·인수·자료 구조를 `f'카메라 {cam_idx}에서 영상을 읽을 수 없습니다 (DirectShow/Media Foundation). '` 내용으로 이어서 구성합니다.
                f'카메라 {cam_idx}에서 영상을 읽을 수 없습니다 (DirectShow/Media Foundation). '
                # 앞줄에서 시작한 식·인수·자료 구조를 `'휴대폰 웹캠 앱의 연결 상태와 카메라 번호를 확인하고, 카메라를 사용하는 다른 앱을 닫아 주세요.')` 내용으로 이어서 구성합니다.
                '휴대폰 웹캠 앱의 연결 상태와 카메라 번호를 확인하고, 카메라를 사용하는 다른 앱을 닫아 주세요.')
        # 프레임을 획득한 단조 시각(초)에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `self.captured_at`.
        self.captured_at = time.perf_counter()
        # `self._last_valid_at` 값에 프레임을 획득한 단조 시각(초)(`self.captured_at`)을 저장합니다. 대상: `self._last_valid_at`.
        self._last_valid_at = self.captured_at
        # capture 오류에 None을 저장합니다. 대상: `self._capture_error`.
        self._capture_error = None
        # 캡처 스레드의 프레임 식별 번호에 0을 저장합니다. 대상: `self.frame_id`.
        self.frame_id = 0
        # 동작 시작 시각 또는 실행 여부에 False을 저장합니다. 대상: `self.started`.
        self.started = False
        # 캡처 결과와 시각을 함께 읽고 쓰기 위한 잠금에 스레드 공유 데이터용 잠금(`threading.Lock()`)을 저장합니다. 대상: `self.read_lock`.
        self.read_lock = threading.Lock()

    # 데코레이터 `staticmethod`를 바로 아래 함수·클래스에 적용합니다.
    @staticmethod
    # valid 프레임 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _valid_frame(grabbed, frame):
        # 현재 함수 처리를 끝내고 계산식 `grabbed and isinstance(frame, np.ndarray) and (frame.ndim == 3) and (frame.shape[0] > 0) and (frame.shape[1] > 0) and (frame.shape[2] == 3)`의 결과를 호출한 쪽으로 반환합니다.
        return (grabbed and isinstance(frame, np.ndarray) and frame.ndim == 3
                # 앞줄에서 시작한 식·인수·자료 구조를 `and frame.shape[0] > 0 and frame.shape[1] > 0 and frame.shape[2] == 3)` 내용으로 이어서 구성합니다.
                and frame.shape[0] > 0 and frame.shape[1] > 0 and frame.shape[2] == 3)

    # 시작 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start(self):
        # 조건 `self.started`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.started:
            # 현재 함수 처리를 끝내고 현재 객체(`self`)을 호출한 쪽으로 반환합니다.
            return self
        # 동작 시작 시각 또는 실행 여부에 True을 저장합니다. 대상: `self.started`.
        self.started = True
        # `self.thread` 값에 백그라운드 실행 스레드(`threading.Thread(target=self.update, daemon=True)`)을 저장합니다. 대상: `self.thread`.
        self.thread = threading.Thread(target=self.update, daemon=True)
        # `self.thread.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.thread.start()
        # 현재 함수 처리를 끝내고 현재 객체(`self`)을 호출한 쪽으로 반환합니다.
        return self

    # 갱신 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def update(self):
        # 조건 `self.started`이 유지되는 동안 아래 처리를 반복합니다.
        while self.started:
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `(grabbed, frame)` 값에 `self.cap.read()` 호출 결과를 저장합니다. 대상: `(grabbed, frame)`.
                grabbed, frame = self.cap.read()
            # `cv2.error` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except cv2.error as exc:
                # `self.read_lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with self.read_lock:
                    # capture 오류에 `str(exc)` 호출 결과를 저장합니다. 대상: `self._capture_error`.
                    self._capture_error = str(exc)
                # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                break
            # 유효한 영상 프레임을 읽었는지 여부에 `self._valid_frame(grabbed, frame)` 호출 결과를 저장합니다. 대상: `grabbed`.
            grabbed = self._valid_frame(grabbed, frame)
            # 프레임을 획득한 단조 시각(초)에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `captured_at`.
            captured_at = time.perf_counter()
            # `self.read_lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.read_lock:
                # 유효한 영상 프레임을 읽었는지 여부에 유효한 영상 프레임을 읽었는지 여부(`grabbed`)을 저장합니다. 대상: `self.grabbed`.
                self.grabbed = grabbed
                # OpenCV BGR 영상 프레임에 OpenCV BGR 영상 프레임(`frame`)을 저장합니다. 대상: `self.frame`.
                self.frame = frame
                # 프레임을 획득한 단조 시각(초)에 프레임을 획득한 단조 시각(초)(`captured_at`)을 저장합니다. 대상: `self.captured_at`.
                self.captured_at = captured_at
                # 캡처 스레드의 프레임 식별 번호에 `1`을 더해 누적합니다. 대상: `self.frame_id`.
                self.frame_id += 1
                # 조건 `grabbed`을 만족하는 경우에만 아래 분기를 실행합니다.
                if grabbed:
                    # `self._last_valid_at` 값에 프레임을 획득한 단조 시각(초)(`captured_at`)을 저장합니다. 대상: `self._last_valid_at`.
                    self._last_valid_at = captured_at
            # `time.sleep(0.005)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다.
            time.sleep(0.005)

    # 읽기 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def read(self):
        # `(grabbed, frame, _, _)` 값에 `self.read_sample()` 호출 결과를 저장합니다. 대상: `(grabbed, frame, _, _)`.
        grabbed, frame, _, _ = self.read_sample()
        # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(grabbed, frame)`을 호출한 쪽으로 반환합니다.
        return grabbed, frame

    # 잠금 안에서 영상·획득 시각·프레임 번호를 함께 복사합니다. 장시간 입력 단절은 오류로 알립니다.
    def read_sample(self):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Return one atomic frame/timestamp/id snapshot from the capture thread."""
        # `self.read_lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.read_lock:
            # 조건 `self._capture_error`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self._capture_error:
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(f'카메라 프레임 읽기 실패: {self._capture_error}')` 예외를 발생시킵니다.
                raise RuntimeError(f'카메라 프레임 읽기 실패: {self._capture_error}')
            # 조건 `not self.grabbed or self.frame is None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not self.grabbed or self.frame is None:
                # 조건 `time.perf_counter() - self._last_valid_at >= 5.0`을 만족하는 경우에만 아래 분기를 실행합니다.
                if time.perf_counter() - self._last_valid_at >= 5.0:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('카메라 영상이 5초 이상 끊겼습니다. 휴대폰 웹캠 연결을 확인해 주세요.')` 예외를 발생시킵니다.
                    raise RuntimeError('카메라 영상이 5초 이상 끊겼습니다. 휴대폰 웹캠 연결을 확인해 주세요.')
                # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(False, None, self.captured_at, self.frame_id)`을 호출한 쪽으로 반환합니다.
                return False, None, self.captured_at, self.frame_id
            # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(True, self.frame.copy(), self.captured_at, self.frame_id)`을 호출한 쪽으로 반환합니다.
            return True, self.frame.copy(), self.captured_at, self.frame_id

    # 중지 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def stop(self):
        # 동작 시작 시각 또는 실행 여부에 False을 저장합니다. 대상: `self.started`.
        self.started = False
        # 조건 `self.cap and self.cap.isOpened()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.cap and self.cap.isOpened():
            # `self.cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            self.cap.release()


# `ExerciseSocketServer` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ExerciseSocketServer:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, host="127.0.0.1", port=8080, engine_factory=None):
        # `self.host` 값에 `host` 값(`host`)을 저장합니다. 대상: `self.host`.
        self.host = host
        # `self.port` 값에 `port` 값(`port`)을 저장합니다. 대상: `self.port`.
        self.port = port

        # 사람 검출·관절 추출 AI 엔진에 None을 저장합니다. 대상: `self.skeleton_engine`.
        self.skeleton_engine = None
        # 백그라운드 AI 모델 준비 관리자에 `BackgroundModelLoader(engine_factory or self._create_skeleton_engine)` 호출 결과를 저장합니다. 대상: `self.model_loader`.
        self.model_loader = BackgroundModelLoader(engine_factory or self._create_skeleton_engine)
        # `self.bound_port` 값에 None을 저장합니다. 대상: `self.bound_port`.
        self.bound_port = None
        # 현재 서버의 측정 자원을 점유한 연결에 None을 저장합니다. 대상: `self._session_owner`.
        self._session_owner = None
        # 연결 종료 후 로그·카메라 정리가 끝났는지 확인하는 신호입니다.
        self._connection_done = {}
        # 현재 기록 중인 세션 로그에 None을 저장합니다. 대상: `self._active_log`.
        self._active_log = None
        # 실시간 캡처 시각을 기준으로 로그를 기록하는지 여부에 False을 저장합니다. 대상: `self._live_timing`.
        self._live_timing = False
        # `self._last_capture_id` 값에 None을 저장합니다. 대상: `self._last_capture_id`.
        self._last_capture_id = None
        # measurement 준비 상태 at에 None을 저장합니다. 대상: `self._measurement_ready_at`.
        self._measurement_ready_at = None
        # `self._last_preparation_remaining` 값에 None을 저장합니다. 대상: `self._last_preparation_remaining`.
        self._last_preparation_remaining = None

        # 골반 중심·몸통 길이 기준 좌표 변환기에 `PoseNormalizer(conf_threshold=0.35, invert_y=True)` 호출 결과를 저장합니다. 대상: `self.normalizer`.
        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)
        # `self.camera` 값에 None을 저장합니다. 대상: `self.camera`.
        self.camera = None
        # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
        self.controller = None
        # 평활화 엔진에 None을 저장합니다. 대상: `self.filter_engine`.
        self.filter_engine = None
        # 출력 mode에 'LIGHTWEIGHT'을 저장합니다. 대상: `self.output_mode`.
        self.output_mode = "LIGHTWEIGHT"

    # 데코레이터 `staticmethod`를 바로 아래 함수·클래스에 적용합니다.
    @staticmethod
    # 생성 skeleton 엔진 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _create_skeleton_engine(report_phase):
        # Torch·MMPose 가져오기와 모델 등록은 소켓 수신 시작 이후로 미룹니다.
        # 실제 준비 작업은 모델 로더 스레드에서 실행해 소켓 시작을 막지 않습니다.
        # `core.skeleton_engine`에서 `SkeletonEngine` 기능을 가져옵니다.
        from core.skeleton_engine import SkeletonEngine
        # 현재 함수 처리를 끝내고 `SkeletonEngine(yolo_interval=5, on_progress=report_phase)` 호출 결과를 호출한 쪽으로 반환합니다.
        return SkeletonEngine(yolo_interval=5, on_progress=report_phase)

    # 시작 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _start_session(self, websocket, packet):
        # 조건 `str(packet.get('output_mode', 'LIGHTWEIGHT')).upper() != 'LIGHTWEIGHT'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if str(packet.get('output_mode', 'LIGHTWEIGHT')).upper() != 'LIGHTWEIGHT':
            # `websocket.send(json.dumps({'type': 'ERROR', 'code': 'UNSUPPORTED_OUTPUT_MODE', 'message': 'VISUAL 모드는 제거되었습니다. LIGHTWEIGHT를 사용하고, 영상 분석은 video_path를 지정하세요.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps({'type': 'ERROR', 'code': 'UNSUPPORTED_OUTPUT_MODE',
                # 구성 항목: 메시지(`'message'`)에 'VISUAL 모드는 제거되었습니다. LIGHTWEIGHT를 사용하고, 영상 분석은 video_path를 지정하세요.'을 지정합니다.
                'message': 'VISUAL 모드는 제거되었습니다. LIGHTWEIGHT를 사용하고, 영상 분석은 video_path를 지정하세요.'}))
            # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
            return False
        # 조건 `self._session_owner is not None and self._session_owner is not websocket`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._session_owner is not None and self._session_owner is not websocket:
            # `websocket.send(json.dumps({'type': 'ERROR', 'code': 'SERVER_BUSY', 'message': '다른 연결에서 세션을 사용 중입니다.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps({"type": "ERROR", "code": "SERVER_BUSY",
                                              # 구성 항목: 메시지(`'message'`)에 '다른 연결에서 세션을 사용 중입니다.'을 지정합니다.
                                              "message": "다른 연결에서 세션을 사용 중입니다."}))
            # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
            return False
        # 현재 서버의 측정 자원을 점유한 연결에 현재 클라이언트의 WebSocket 연결(`websocket`)을 저장합니다. 대상: `self._session_owner`.
        self._session_owner = websocket
        # last 클라이언트 종료에 None을 저장합니다. 대상: `self._last_client_finish`.
        self._last_client_finish = None
        # 서버 PC에서 읽을 원본 영상 경로에 계산식 `packet.get('video_path') or packet.get('input_source')`의 결과를 저장합니다. 대상: `video_path`.
        video_path = packet.get("video_path") or packet.get("input_source")
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 조건 `self.controller`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.controller:
                # `self._save_session_files(status='STOPPED')`을 호출하여 해당 단계의 작업을 수행합니다.
                self._save_session_files(status="STOPPED")
                # `self.stop_camera()`을 호출하여 해당 단계의 작업을 수행합니다.
                self.stop_camera()
                # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
                self.controller = None
            # 조건 `video_path`을 만족하는 경우에만 아래 분기를 실행합니다.
            if video_path:
                # 조건 `not os.path.isfile(str(video_path))`을 만족하는 경우에만 아래 분기를 실행합니다.
                if not os.path.isfile(str(video_path)):
                    # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'비디오 파일을 찾을 수 없습니다: {video_path}')` 예외를 발생시킵니다.
                    raise ValueError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
                # `self.process_offline_video(websocket, packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await self.process_offline_video(websocket, packet)
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # `websocket.send(json.dumps({'type': 'SESSION_PREPARING', 'message': '카메라 연결 중입니다. 휴대폰 웹캠 앱의 연결 상태를 확인해 주세요.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await websocket.send(json.dumps({"type": "SESSION_PREPARING",
                    # 구성 항목: 메시지(`'message'`)에 '카메라 연결 중입니다. 휴대폰 웹캠 앱의 연결 상태를 확인해 주세요.'을 지정합니다.
                    "message": "카메라 연결 중입니다. 휴대폰 웹캠 앱의 연결 상태를 확인해 주세요."}))
                # `asyncio.to_thread(self.init_live_session, packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await asyncio.to_thread(self.init_live_session, packet)
                # `websocket.send(json.dumps({'type': 'SESSION_LOG_STARTED', **self.controller.data_manager.session_log.info}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await websocket.send(json.dumps({
                    # 구성 항목: `type` 값(`'type'`)에 'SESSION_LOG_STARTED'을 지정합니다.
                    "type": "SESSION_LOG_STARTED", **self.controller.data_manager.session_log.info}))
        # `ConnectionClosed` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except ConnectionClosed:
            # 이 조건에서는 처리를 계속할 수 없으므로 `None` 예외를 발생시킵니다.
            raise
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception as exc:
            # `traceback.print_exc()`을 호출하여 현재 예외의 상세 호출 경로를 출력합니다.
            traceback.print_exc()
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # 조건 `self.controller`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self.controller:
                    # `self._save_session_files(status='FAILED')`을 호출하여 해당 단계의 작업을 수행합니다.
                    self._save_session_files(status="FAILED")
                # 조건 `self._active_log`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif self._active_log:
                    # `self._active_log.finish([], status='FAILED')`을 호출하여 해당 단계의 작업을 수행합니다.
                    self._active_log.finish([], status="FAILED")
            # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except Exception:
                # `traceback.print_exc()`을 호출하여 현재 예외의 상세 호출 경로를 출력합니다.
                traceback.print_exc()
            # 현재 기록 중인 세션 로그에 None을 저장합니다. 대상: `self._active_log`.
            self._active_log = None
            # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
            self.controller = None
            # `self.stop_camera()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.stop_camera()
            # `websocket.send(json.dumps({'type': 'ERROR', 'code': 'VIDEO_ANALYSIS_FAILED' if video_path else 'SESSION_START_FAILED', 'message': f'세션 시작/분석 실패: {exc}'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps({
                # 구성 항목: `type` 값(`'type'`)에 'ERROR'; `code` 값(`'code'`)에 `video_path`이면 `'VIDEO_ANALYSIS_FAILED'`, 아니면 `'SESSION_START_FAILED'`을 지정합니다.
                "type": "ERROR", "code": "VIDEO_ANALYSIS_FAILED" if video_path else "SESSION_START_FAILED",
                # 구성 항목: 메시지(`'message'`)에 계산식 `f'세션 시작/분석 실패: {exc}'`의 결과를 지정합니다.
                "message": f"세션 시작/분석 실패: {exc}",
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            }))
        # 현재 함수 처리를 끝내고 True을 호출한 쪽으로 반환합니다.
        return True

    # 시작 camera 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start_camera(self, cam_idx=0):
        # 조건 `self.camera`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.camera:
            # `self.camera.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.camera.stop()
        # `self.camera` 값에 `ThreadedWebcam(cam_idx).start()` 호출 결과를 저장합니다. 대상: `self.camera`.
        self.camera = ThreadedWebcam(cam_idx).start()

    # 중지 camera 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def stop_camera(self):
        # 조건 `self.camera`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.camera:
            # `self.camera.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.camera.stop()
            # `self.camera` 값에 None을 저장합니다. 대상: `self.camera`.
            self.camera = None

    # 준비 데이터 manager 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _prepare_data_manager(self, data, video_fps=30.0, live_timing=False):
        # 실행 모드에 `data.get('mode', 'CALIBRATION')` 호출 결과를 저장합니다. 대상: `mode`.
        mode = data.get("mode", "CALIBRATION")
        # 조건 `'cal' in data and (not isinstance(data['cal'], bool) or data['cal'] != (mode == 'CALIBRATION'))`을 만족하는 경우에만 아래 분기를 실행합니다.
        if "cal" in data and (not isinstance(data["cal"], bool) or data["cal"] != (mode == "CALIBRATION")):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('cal must match mode == CALIBRATION')` 예외를 발생시킵니다.
            raise ValueError("cal must match mode == CALIBRATION")
        # `manager` 값에 `DataManager(player_id=data.get('patient_id', data.get('player_id', 'patient_1')), patient_name=data.get('patient_name', 'Unknown'), log_root=data.get('log_root'), patien…` 호출 결과를 저장합니다. 대상: `manager`.
        manager = DataManager(
            # 구성 항목: 환자 식별 번호의 기존 필드명(`player_id`)에 `data.get('patient_id', data.get('player_id', 'patient_1'))` 호출 결과를 지정합니다.
            player_id=data.get("patient_id", data.get("player_id", "0001")),
            # 구성 항목: `patient_name` 값(`patient_name`)에 `data.get('patient_name', 'Unknown')` 호출 결과; 세션 로그를 저장하는 최상위 폴더(`log_root`)에 `data.get('log_root')` 호출 결과를 지정합니다.
            patient_name=data.get("patient_name", "Unknown"), log_root=data.get("log_root"),
            # 구성 항목: 환자의 키·몸무게 등 부가 정보(`patient_metadata`)에 `data.get('patient_metadata')` 호출 결과를 지정합니다.
            patient_metadata=data.get("patient_metadata"),
            # 버전·등록 차수는 Unity가 사용하는 값을 그대로 전달합니다.
            data_version=data.get("data_version", "v1.0.0(0)"), registration=data.get("registration", 1))
        # `manager.begin_session(exercise_name=data.get('exercise_name', 'biceps_curl'), mode=mode, session=data.get('session'), round_number=data.get('round', 1), set_number=data.…`을 호출하여 해당 단계의 작업을 수행합니다.
        manager.begin_session(
            # 구성 항목: 운동 설정의 식별 이름(`exercise_name`)에 `data.get('exercise_name', 'biceps_curl')` 호출 결과; 실행 모드(`mode`)에 실행 모드(`mode`)을 지정합니다.
            exercise_name=data.get("exercise_name", "biceps_curl"), mode=mode,
            # 구성 항목: 측정 세션(`session`)에 `data.get('session')` 호출 결과; `round_number` 값(`round_number`)에 `data.get('round', 1)` 호출 결과; 설정 number(`set_number`)에 `data.get('set', 1)` 호출 결과를 지정합니다.
            session=data.get("session"), round_number=data.get("round", 1), set_number=data.get("set", 1),
            # 구성 항목: 영상 녹화 여부(`record_video`)에 True; 저장 영상의 초당 프레임 수(`video_fps`)에 저장 영상의 초당 프레임 수(`video_fps`)을 지정합니다.
            record_video=True, video_fps=video_fps,
            # 구성 항목: 게임 목표 점수(`target_score`)에 `data.get('target_score')` 호출 결과를 지정합니다.
            target_score=data.get("target_score"),
            # 플레이 전체 ID와 운동별 기록 ID를 카메라 파일의 메타데이터에 연결합니다.
            play_session_id=data.get("play_session_id"), exercise_record_id=data.get("exercise_record_id"),
            slot=data.get("slot", 1))
        # 현재 기록 중인 세션 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`manager.session_log`)을 저장합니다. 대상: `self._active_log`.
        self._active_log = manager.session_log
        # 측정 세션 started에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `self._session_started`.
        self._session_started = time.perf_counter()
        # 실시간 캡처 시각을 기준으로 로그를 기록하는지 여부에 실시간 시각 동기화(`live_timing`)을 저장합니다. 대상: `self._live_timing`.
        self._live_timing = live_timing
        # `self._last_capture_id` 값에 None을 저장합니다. 대상: `self._last_capture_id`.
        self._last_capture_id = None
        # measurement 준비 상태 at에 None을 저장합니다. 대상: `self._measurement_ready_at`.
        self._measurement_ready_at = None
        # 현재 함수 처리를 끝내고 `manager` 값(`manager`)을 호출한 쪽으로 반환합니다.
        return manager

    # init 실시간 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def init_live_session(self, data: dict):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """실시간 웹캠 세션 초기화 (모드 2)"""
        # 환자 식별 번호의 기존 필드명에 `data.get('player_id', 'patient_1')` 호출 결과를 저장합니다. 대상: `player_id`.
        player_id = data.get("player_id", "patient_1")
        # `patient_name` 값에 `data.get('patient_name', 'Unknown')` 호출 결과를 저장합니다. 대상: `patient_name`.
        patient_name = data.get("patient_name", "Unknown")
        # 운동 설정의 식별 이름에 `data.get('exercise_name', 'biceps_curl')` 호출 결과를 저장합니다. 대상: `exercise_name`.
        exercise_name = data.get("exercise_name", "biceps_curl")
        # 실행 모드에 `data.get('mode', 'CALIBRATION')` 호출 결과를 저장합니다. 대상: `mode`.
        mode = data.get("mode", "CALIBRATION")
        # 게임 목표 점수에 `'target_score' in data or 'target_reps' in data`이면 `get_target_score(data)`, 아니면 `10.0`을 저장합니다. 대상: `target_score`.
        target_score = get_target_score(data) if "target_score" in data or "target_reps" in data else 10.0
        # `cam_idx` 값에 `int(data.get('camera_index', 0))` 호출 결과를 저장합니다. 대상: `cam_idx`.
        cam_idx = int(data.get("camera_index", 0))
        # 출력 mode에 `data.get('output_mode', 'LIGHTWEIGHT').upper()` 호출 결과를 저장합니다. 대상: `self.output_mode`.
        self.output_mode = data.get("output_mode", "LIGHTWEIGHT").upper()
        # 조건 `self.output_mode != 'LIGHTWEIGHT'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.output_mode != "LIGHTWEIGHT":
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Only LIGHTWEIGHT output is supported')` 예외를 발생시킵니다.
            raise ValueError("Only LIGHTWEIGHT output is supported")
        # preparation 초에 `get_preparation_seconds(data)` 호출 결과를 저장합니다. 대상: `preparation_seconds`.
        preparation_seconds = get_preparation_seconds(data)

        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[DEBUG][LIVE_INIT] Mode: {mode} | Output: {self.output_mode} | Exercise: {exercise_name}")
        # 환자 정보 및 세션 로그 관리자에 `self._prepare_data_manager(data, live_timing=True)` 호출 결과를 저장합니다. 대상: `data_manager`.
        data_manager = self._prepare_data_manager(data, live_timing=True)
        # exercise 설정에 `load_exercise_config(exercise_name)` 호출 결과를 저장합니다. 대상: `exercise_config`.
        exercise_config = load_exercise_config(exercise_name)

        # 환자에게 저장된 좌우 개인 가동 범위에 None을 저장합니다. 대상: `custom_thresholds`.
        custom_thresholds = None
        # 조건 `mode in ['MAIN', 'TEST']`을 만족하는 경우에만 아래 분기를 실행합니다.
        if mode in ["MAIN", "TEST"]:
            # 환자에게 저장된 좌우 개인 가동 범위에 `data_manager.get_custom_threshold(exercise_name)` 호출 결과를 저장합니다. 대상: `custom_thresholds`.
            custom_thresholds = data_manager.get_custom_threshold(exercise_name)

        # 조건 `custom_thresholds and 'left' not in custom_thresholds and ('start_val' in custom_thresholds)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            # 환자에게 저장된 좌우 개인 가동 범위에 키 'left', 'right'로 구성한 사전을 저장합니다. 대상: `custom_thresholds`.
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        # effective 반복 횟수에 `mode == 'CALIBRATION'`이면 `3`, 아니면 `target_score`을 저장합니다. 대상: `effective_reps`.
        effective_reps = 3 if mode == "CALIBRATION" else target_score

        # 관절 측정값 계산 및 반복 판정 엔진에 `MotionEngine(config=exercise_config, custom_thresholds=custom_thresholds, mode=mode, target_reps=effective_reps, target_score=target_score)` 호출 결과를 저장합니다. 대상: `motion_engine`.
        motion_engine = MotionEngine(
            # 구성 항목: 현재 작업 설정(`config`)에 exercise 설정(`exercise_config`)을 지정합니다.
            config=exercise_config,
            # 구성 항목: 환자에게 저장된 좌우 개인 가동 범위(`custom_thresholds`)에 환자에게 저장된 좌우 개인 가동 범위(`custom_thresholds`)을 지정합니다.
            custom_thresholds=custom_thresholds,
            # 구성 항목: 실행 모드(`mode`)에 실행 모드(`mode`)을 지정합니다.
            mode=mode,
            # 구성 항목: 목표 반복 횟수(`target_reps`)에 effective 반복 횟수(`effective_reps`); 게임 목표 점수(`target_score`)에 게임 목표 점수(`target_score`)을 지정합니다.
            target_reps=effective_reps, target_score=target_score
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        )
        # `setattr(motion_engine, 'exercise_name', exercise_name)`을 호출하여 해당 단계의 작업을 수행합니다.
        setattr(motion_engine, "exercise_name", exercise_name)

        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(data_manager=data_manager, motion_engine=motion_engine)` 호출 결과를 저장합니다. 대상: `self.controller`.
        self.controller = SessionController(
            # 구성 항목: 환자 정보 및 세션 로그 관리자(`data_manager`)에 환자 정보 및 세션 로그 관리자(`data_manager`)을 지정합니다.
            data_manager=data_manager,
            # 구성 항목: 관절 측정값 계산 및 반복 판정 엔진(`motion_engine`)에 관절 측정값 계산 및 반복 판정 엔진(`motion_engine`)을 지정합니다.
            motion_engine=motion_engine
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        )
        # 평활화 엔진에 `OneEuroKeypointFilter()` 호출 결과를 저장합니다. 대상: `self.filter_engine`.
        self.filter_engine = OneEuroKeypointFilter()
        # `self._reset_raw_buffers()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._reset_raw_buffers()
        # `self.start_camera(cam_idx)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.start_camera(cam_idx)
        # 카메라를 실제 사용할 수 있게 된 뒤 준비 시간 카운트다운을 시작합니다.
        # 측정 세션 started에 계산식 `time.perf_counter() + preparation_seconds`의 결과를 저장합니다. 대상: `self._session_started`.
        self._session_started = time.perf_counter() + preparation_seconds
        # measurement 준비 상태 at에 `preparation_seconds`이면 `self._session_started`, 아니면 `None`을 저장합니다. 대상: `self._measurement_ready_at`.
        self._measurement_ready_at = self._session_started if preparation_seconds else None
        # `self._last_preparation_remaining` 값에 None을 저장합니다. 대상: `self._last_preparation_remaining`.
        self._last_preparation_remaining = None

    # measurement 준비 상태 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _measurement_ready(self, websocket):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Nonblocking gate: no inference, filtering, counting or recording during preparation."""
        # 조건 `self._measurement_ready_at is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._measurement_ready_at is None:
            # 현재 함수 처리를 끝내고 True을 호출한 쪽으로 반환합니다.
            return True
        # `remaining` 값에 계산식 `self._measurement_ready_at - time.perf_counter()`의 결과를 저장합니다. 대상: `remaining`.
        remaining = self._measurement_ready_at - time.perf_counter()
        # 조건 `remaining > 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if remaining > 0:
            # 초에 `int(np.ceil(remaining))` 호출 결과를 저장합니다. 대상: `seconds`.
            seconds = int(np.ceil(remaining))
            # 조건 `seconds != self._last_preparation_remaining`을 만족하는 경우에만 아래 분기를 실행합니다.
            if seconds != self._last_preparation_remaining:
                # `websocket.send(json.dumps({'type': 'SESSION_PREPARING', 'remaining_seconds': seconds, 'message': f'기본 자세를 취해 주세요. {seconds}초 후 측정을 시작합니다.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await websocket.send(json.dumps({
                    # 구성 항목: `type` 값(`'type'`)에 'SESSION_PREPARING'; remaining 초(`'remaining_seconds'`)에 초(`seconds`)을 지정합니다.
                    "type": "SESSION_PREPARING", "remaining_seconds": seconds,
                    # 구성 항목: 메시지(`'message'`)에 계산식 `f'기본 자세를 취해 주세요. {seconds}초 후 측정을 시작합니다.'`의 결과를 지정합니다.
                    "message": f"기본 자세를 취해 주세요. {seconds}초 후 측정을 시작합니다."}))
                # `self._last_preparation_remaining` 값에 초(`seconds`)을 저장합니다. 대상: `self._last_preparation_remaining`.
                self._last_preparation_remaining = seconds
            # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
            return False
        # measurement 준비 상태 at에 None을 저장합니다. 대상: `self._measurement_ready_at`.
        self._measurement_ready_at = None
        # `websocket.send(json.dumps({'type': 'SESSION_MEASUREMENT_STARTED', 'message': '준비 시간이 끝나 측정을 시작합니다.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await websocket.send(json.dumps({"type": "SESSION_MEASUREMENT_STARTED",
                                         # 구성 항목: 메시지(`'message'`)에 '준비 시간이 끝나 측정을 시작합니다.'을 지정합니다.
                                         "message": "준비 시간이 끝나 측정을 시작합니다."}))
        # 현재 함수 처리를 끝내고 True을 호출한 쪽으로 반환합니다.
        return True

    # 초기화 raw buffers 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _reset_raw_buffers(self):
        # 조건 `self.controller and self.controller.data_manager`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.controller and self.controller.data_manager:
            # `dm` 값에 환자 정보 및 세션 로그 관리자(`self.controller.data_manager`)을 저장합니다. 대상: `dm`.
            dm = self.controller.data_manager
            # `dm.buf_timestamps` 값에 프레임별 경과 시각 목록(`dm.session_log.timestamps`)을 저장합니다. 대상: `dm.buf_timestamps`.
            dm.buf_timestamps = dm.session_log.timestamps
            # buf 측정값에 측정값(`dm.session_log.values`)을 저장합니다. 대상: `dm.buf_values`.
            dm.buf_values = dm.session_log.values
            # buf raw 관절에 COCO 관절 좌표 목록(`dm.session_log.keypoints`)을 저장합니다. 대상: `dm.buf_raw_keypoints`.
            dm.buf_raw_keypoints = dm.session_log.keypoints

    # `_append_to_buffers` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _append_to_buffers(self, left_data, right_data, raw_keypoints, elapsed_sec=None,
                           # 이 줄의 입력 인수: `frame_idx` = 분석 프레임 번호; `is_occluded` = 필수 관절이 가려졌거나 유효하지 않은지 여부; `video_frame_idx` = 저장 영상에서 대응하는 프레임 번호. 기본값·형식은 아래 선언을 따릅니다.
                           frame_idx=None, is_occluded=False, video_frame_idx=-1):
        # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.controller.data_manager.session_log`)을 저장합니다. 대상: `log`.
        log = self.controller.data_manager.session_log
        # 생성 또는 검증할 분석 엔진에 관절 측정값 계산 및 반복 판정 엔진(`self.controller.motion_engine`)을 저장합니다. 대상: `engine`.
        engine = self.controller.motion_engine
        # 조건 `elapsed_sec is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if elapsed_sec is None:
            # `elapsed_sec` 값에 계산식 `time.perf_counter() - self._session_started`의 결과를 저장합니다. 대상: `elapsed_sec`.
            elapsed_sec = time.perf_counter() - self._session_started
        # 조건 `frame_idx is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if frame_idx is None:
            # 분석 프레임 번호에 `len(log.timestamps)` 호출 결과를 저장합니다. 대상: `frame_idx`.
            frame_idx = len(log.timestamps)
        # 조건 `is_occluded`을 만족하는 경우에만 아래 분기를 실행합니다.
        if is_occluded:
            # 왼쪽 최신 판정 결과에 키 'val', 'is_count_updated'로 구성한 사전을 저장합니다. 대상: `left_data`.
            left_data = {**left_data, "val": None, "is_count_updated": False}
            # 오른쪽 최신 판정 결과에 키 'val', 'is_count_updated'로 구성한 사전을 저장합니다. 대상: `right_data`.
            right_data = {**right_data, "val": None, "is_count_updated": False}
        # `log.append_frame(elapsed_sec, frame_idx, raw_keypoints, left_data, right_data, metric_type=engine.metric_type, thresholds=engine.get_count_thresholds(), threshold_versio…`을 호출하여 해당 단계의 작업을 수행합니다.
        log.append_frame(
            # 앞줄에서 시작한 식·인수·자료 구조를 `elapsed_sec, frame_idx, raw_keypoints, left_data, right_data,` 내용으로 이어서 구성합니다.
            elapsed_sec, frame_idx, raw_keypoints, left_data, right_data,
            # 구성 항목: 측정 방식(각도·좌표·길이·비율)(`metric_type`)에 측정 방식(각도·좌표·길이·비율)(`engine.metric_type`); 좌우 가동 범위 시작값·목표값(`thresholds`)에 `engine.get_count_thresholds()` 호출 결과를 지정합니다.
            metric_type=engine.metric_type, thresholds=engine.get_count_thresholds(),
            # 구성 항목: 현재 가동 범위 설정의 변경 번호(`threshold_version`)에 현재 가동 범위 설정의 변경 번호(`engine.threshold_version`); 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)을 지정합니다.
            threshold_version=engine.threshold_version, is_occluded=is_occluded,
            # 구성 항목: 캘리브레이션 전체 진행 단계(`calib_step`)에 `getattr(self.controller, 'calib_step', '')` 호출 결과; 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)에 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)을 지정합니다.
            calib_step=getattr(self.controller, "calib_step", ""), video_frame_idx=video_frame_idx)

    # 저장 측정 세션 파일 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _save_session_files(self, prefix="SESSION", status="COMPLETED"):
        # 조건 `not self.controller or not self.controller.data_manager`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.controller or not self.controller.data_manager:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 생성 또는 검증할 분석 엔진에 관절 측정값 계산 및 반복 판정 엔진(`self.controller.motion_engine`)을 저장합니다. 대상: `engine`.
        engine = self.controller.motion_engine
        # `rows` 값에 계산식 `engine.fsm_left.completed_reps_history + engine.fsm_right.completed_reps_history`의 결과를 저장합니다. 대상: `rows`.
        rows = engine.fsm_left.completed_reps_history + engine.fsm_right.completed_reps_history
        # `end_elapsed` 값에 `self._live_timing`이면 `max(0.0, time.perf_counter() - self._session_started)`, 아니면 `None`을 저장합니다. 대상: `end_elapsed`.
        end_elapsed = max(0.0, time.perf_counter() - self._session_started) if self._live_timing else None
        # 저장·종료 결과 요약에 `self.controller.data_manager.finish_session(rows, status=status, end_elapsed_sec=end_elapsed)` 호출 결과를 저장합니다. 대상: `summary`.
        summary = self.controller.data_manager.finish_session(rows, status=status, end_elapsed_sec=end_elapsed)
        # 현재 기록 중인 세션 로그에 None을 저장합니다. 대상: `self._active_log`.
        self._active_log = None
        # 현재 함수 처리를 끝내고 저장·종료 결과 요약(`summary`)을 호출한 쪽으로 반환합니다.
        return summary

    # 전송 반복 이벤트 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _send_rep_events(self, websocket, left, right, timestamp=None):
        # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.controller.data_manager.session_log`)을 저장합니다. 대상: `log`.
        log = self.controller.data_manager.session_log
        # 생성 또는 검증할 분석 엔진에 관절 측정값 계산 및 반복 판정 엔진(`self.controller.motion_engine`)을 저장합니다. 대상: `engine`.
        engine = self.controller.motion_engine
        # `(('left', left), ('right', right))`에서 항목을 차례로 꺼내 `(side, data)`에 넣고 아래 처리를 반복합니다.
        for side, data in (("left", left), ("right", right)):
            # 조건 `not data.get('is_count_updated')`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not data.get("is_count_updated"):
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # `websocket.send(json.dumps({'type': 'REP_COMPLETED', 'session_id': log.session_id, 'exercise_name': engine.exercise_name, 'mode': engine.mode, 'side': side, 'rep_id': dat…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps({
                # 구성 항목: `type` 값(`'type'`)에 'REP_COMPLETED'; 현재 측정 세션 식별자(`'session_id'`)에 현재 측정 세션 식별자(`log.session_id`)을 지정합니다.
                "type": "REP_COMPLETED", "session_id": log.session_id,
                # Unity의 운동 상세·게임 이벤트와 직접 연결할 수 있는 공통 ID입니다.
                "play_session_id": log.play_session_id, "exercise_record_id": log.exercise_record_id,
                "sensor_file_id": log.sensor_file_id,
                # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 운동 설정의 식별 이름(`engine.exercise_name`); 실행 모드(`'mode'`)에 실행 모드(`engine.mode`)을 지정합니다.
                "exercise_name": engine.exercise_name, "mode": engine.mode,
                # 구성 항목: 좌우 구분(`'side'`)에 좌우 구분(`side`); 반복 id(`'rep_id'`)에 계산식 `data['rep_count']`의 결과; 동작 품질 등급(`'quality'`)에 계산식 `data['quality']`의 결과를 지정합니다.
                "side": side, "rep_id": data["rep_count"], "quality": data["quality"],
                # 구성 항목: 현재 가동 범위 설정의 변경 번호(`'threshold_version'`)에 현재 가동 범위 설정의 변경 번호(`engine.threshold_version`)을 지정합니다.
                "threshold_version": engine.threshold_version,
                # 구성 항목: 입력 표본 또는 세션 기준 시각(초)(`'timestamp'`)에 입력 표본 또는 세션 기준 시각(초)(`timestamp`); `val` 값(`'val'`)에 `data.get('val')` 호출 결과를 지정합니다.
                "timestamp": timestamp, "val": data.get("val"),
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            }))

    # 종료 클라이언트 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _finish_client_session(self, websocket, packet):
        # 명령과 응답을 연결하는 요청 식별자에 `packet.get('request_id')` 호출 결과를 저장합니다. 대상: `request_id`.
        request_id = packet.get("request_id")
        # 오류 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def error(code, message):
            # 현재 함수 처리를 끝내고 키 'type', 'command', 'code', 'request_id', 'message'로 구성한 사전을 호출한 쪽으로 반환합니다.
            return {"type": "ERROR", "command": "CMD_STOP_SESSION", "code": code,
                    # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 명령과 응답을 연결하는 요청 식별자(`request_id`); 메시지(`'message'`)에 메시지(`message`)을 지정합니다.
                    "request_id": request_id, "message": message}
        # `previous` 값에 `getattr(self, '_last_client_finish', None)` 호출 결과를 저장합니다. 대상: `previous`.
        previous = getattr(self, "_last_client_finish", None)
        # 조건 `self._session_owner is not websocket or not self.controller`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._session_owner is not websocket or not self.controller:
            # 조건 `previous and previous[0] is websocket and (self._session_owner is None)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if previous and previous[0] is websocket and self._session_owner is None:
                # `response` 값에 키 'request_id'로 구성한 사전을 저장합니다. 대상: `response`.
                response = {**previous[1], "request_id": request_id}
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # `response` 값에 `error('NO_ACTIVE_SESSION', 'This connection does not own an active session')` 호출 결과를 저장합니다. 대상: `response`.
                response = error("NO_ACTIVE_SESSION", "This connection does not own an active session")
            # `websocket.send(json.dumps(response))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps(response))
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return
        # 현재 측정 세션 식별자에 현재 측정 세션 식별자(`self.controller.data_manager.session_log.session_id`)을 저장합니다. 대상: `session_id`.
        session_id = self.controller.data_manager.session_log.session_id
        # 조건 `packet.get('session_id') not in (None, session_id)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if packet.get("session_id") not in (None, session_id):
            # `websocket.send(json.dumps(error('SESSION_MISMATCH', 'session_id does not match the active session')))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps(error("SESSION_MISMATCH", "session_id does not match the active session")))
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return
        # 조건 `self.controller.mode == 'CALIBRATION' and packet.get('reason') == 'TARGET_REACHED'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.controller.mode == "CALIBRATION" and packet.get("reason") == "TARGET_REACHED":
            # `websocket.send(json.dumps(error('INVALID_MODE', 'Calibration completion is managed by the server')))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps(error("INVALID_MODE", "Calibration completion is managed by the server")))
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return
        # 상태에 `packet.get('reason') == 'TARGET_REACHED'`이면 `'COMPLETED'`, 아니면 `'STOPPED'`을 저장합니다. 대상: `status`.
        status = "COMPLETED" if packet.get("reason") == "TARGET_REACHED" else "STOPPED"
        # `self._record_client_result(packet)`을 호출하여 해당 단계의 작업을 수행합니다.
        self._record_client_result(packet)
        # 저장·종료 결과 요약에 `self._save_session_files(status=status)` 호출 결과를 저장합니다. 대상: `summary`.
        summary = self._save_session_files(status=status)
        # `response` 값에 키 'type', 'session_id', 'request_id', 'reason', 'status', 'summary'로 구성한 사전을 저장합니다. 대상: `response`.
        response = {"type": "SESSION_FINISHED", "session_id": session_id,
                    # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 명령과 응답을 연결하는 요청 식별자(`request_id`); `reason` 값(`'reason'`)에 `packet.get('reason', 'USER_STOP')` 호출 결과를 지정합니다.
                    "request_id": request_id, "reason": packet.get("reason", "USER_STOP"),
                    # 구성 항목: 상태(`'status'`)에 상태(`status`); 저장·종료 결과 요약(`'summary'`)에 저장·종료 결과 요약(`summary`)을 지정합니다.
                    "status": status, "summary": summary}
        # `self.stop_camera()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.stop_camera()
        # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
        self.controller = None
        # 현재 서버의 측정 자원을 점유한 연결에 None을 저장합니다. 대상: `self._session_owner`.
        self._session_owner = None
        # last 클라이언트 종료에 순서·구성을 지정한 `(websocket, response)`을 저장합니다. 대상: `self._last_client_finish`.
        self._last_client_finish = (websocket, response)
        # `websocket.send(json.dumps(response))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await websocket.send(json.dumps(response))

    # 기록 클라이언트 결과 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _record_client_result(self, packet):
        # 클라이언트 게임 결과는 선택적인 로그 부가 정보이며 서버의 자동 종료 규칙이 아닙니다.
        # 함수 또는 처리 단계의 결과에 `packet.get('client_result')` 호출 결과를 저장합니다. 대상: `result`.
        result = packet.get("client_result")
        # 조건 `not isinstance(result, dict)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not isinstance(result, dict):
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return
        # 목표에 `result.get('target_score')` 호출 결과를 저장합니다. 대상: `target`.
        target = result.get("target_score")
        # 관절 신뢰도 배열 또는 좌우 게임 점수(사용 위치에 따름)에 `result.get('scores')` 호출 결과를 저장합니다. 대상: `scores`.
        scores = result.get("scores")
        # 측정값에 계산식 `[target] + ([scores.get(s) for s in ('left', 'right')] if isinstance(scores, dict) else [None])`의 결과를 저장합니다. 대상: `values`.
        values = [target] + ([scores.get(s) for s in ("left", "right")] if isinstance(scores, dict) else [None])
        # 조건 `not all((type(v) in (int, float) and np.isfinite(v) and (v >= 0) for v in values))`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in values):
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return
        # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.controller.data_manager.session_log`)을 저장합니다. 대상: `log`.
        log = self.controller.data_manager.session_log
        # `log.info['client_result']` 값에 함수 또는 처리 단계의 결과(`result`)을 저장합니다. 대상: `log.info['client_result']`.
        log.info["client_result"] = result
        # `log.info['target_score']` 값에 목표(`target`)을 저장합니다. 대상: `log.info['target_score']`.
        log.info["target_score"] = target

    # poll 영상 중지 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _poll_video_stop(self, websocket):
        # handle_client가 파일 분석 완료를 기다리므로 분석 함수에서 중지 명령을 직접 읽습니다.
        # 조건 `not hasattr(websocket, 'recv')`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not hasattr(websocket, "recv"):
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.001))`)을 저장합니다. 대상: `packet`.
            packet = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.001))
        # `asyncio.TimeoutError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except asyncio.TimeoutError:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 조건 `packet.get('type') == 'CMD_STOP_SESSION'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if packet.get("type") == "CMD_STOP_SESSION":
            # 현재 측정 세션 식별자에 현재 측정 세션 식별자(`self.controller.data_manager.session_log.session_id`)을 저장합니다. 대상: `session_id`.
            session_id = self.controller.data_manager.session_log.session_id
            # 조건 `packet.get('session_id') in (None, session_id)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if packet.get("session_id") in (None, session_id):
                # 현재 함수 처리를 끝내고 수신 또는 전송할 JSON 메시지 사전(`packet`)을 호출한 쪽으로 반환합니다.
                return packet
        # `websocket.send(json.dumps({'type': 'ERROR', 'command': packet.get('type'), 'request_id': packet.get('request_id'), 'code': 'VIDEO_COMMAND_REJECTED', 'message': 'Only a s…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await websocket.send(json.dumps({"type": "ERROR", "command": packet.get("type"),
            # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 `packet.get('request_id')` 호출 결과; `code` 값(`'code'`)에 'VIDEO_COMMAND_REJECTED'을 지정합니다.
            "request_id": packet.get("request_id"), "code": "VIDEO_COMMAND_REJECTED",
            # 구성 항목: 메시지(`'message'`)에 'Only a stop command for the active session is supported during video analysis'을 지정합니다.
            "message": "Only a stop command for the active session is supported during video analysis"}))
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return None

    # ==========================================================================
    # [모드 3] 비디오 분석 방식 (OFFLINE_VIDEO): 배치 GPU 가속 검증
    # ==========================================================================
    # 서버 로컬 영상을 배치 추론하고 프레임 시각에 맞춰 동작 판정·로그·이벤트를 생성합니다.
    async def process_offline_video(self, websocket, data: dict):
        # 서버 PC에서 읽을 원본 영상 경로에 계산식 `data.get('video_path') or data.get('input_source')`의 결과를 저장합니다. 대상: `video_path`.
        video_path = data.get("video_path") or data.get("input_source")
        # 환자 식별 번호의 기존 필드명에 `data.get('player_id', 'patient_1')` 호출 결과를 저장합니다. 대상: `player_id`.
        player_id = data.get("player_id", "patient_1")
        # `patient_name` 값에 `data.get('patient_name', 'Unknown')` 호출 결과를 저장합니다. 대상: `patient_name`.
        patient_name = data.get("patient_name", "Unknown")
        # 운동 설정의 식별 이름에 `data.get('exercise_name', 'biceps_curl')` 호출 결과를 저장합니다. 대상: `exercise_name`.
        exercise_name = data.get("exercise_name", "biceps_curl")
        # 실행 모드에 `data.get('mode', 'CALIBRATION')` 호출 결과를 저장합니다. 대상: `mode`.
        mode = data.get("mode", "CALIBRATION")
        # 게임 목표 점수에 `'target_score' in data or 'target_reps' in data`이면 `get_target_score(data)`, 아니면 `10.0`을 저장합니다. 대상: `target_score`.
        target_score = get_target_score(data) if "target_score" in data or "target_reps" in data else 10.0
        # 한 번에 추론 또는 학습할 표본 수에 `int(data.get('batch_size', 16))` 호출 결과를 저장합니다. 대상: `batch_size`.
        batch_size = int(data.get("batch_size", 16))
        # 조건 `batch_size <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if batch_size <= 0:
            # `websocket.send(json.dumps({'type': 'ERROR', 'code': 'INVALID_BATCH_SIZE', 'message': 'batch_size must be greater than zero'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps({"type": "ERROR", "code": "INVALID_BATCH_SIZE",
                                              # 구성 항목: 메시지(`'message'`)에 'batch_size must be greater than zero'을 지정합니다.
                                              "message": "batch_size must be greater than zero"}))
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return

        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"\n[MODE 3: VIDEO_BATCH] Processing '{video_path}' (Batch: {batch_size}, Mode: {mode})")

        # `websocket.send(json.dumps({'type': 'VIDEO_STATUS', 'status': 'PROCESSING', 'message': '비디오를 고속 배치 분석 중입니다. 잠시만 기다려주세요...'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await websocket.send(json.dumps({
            # 구성 항목: `type` 값(`'type'`)에 'VIDEO_STATUS'을 지정합니다.
            "type": "VIDEO_STATUS",
            # 구성 항목: 상태(`'status'`)에 'PROCESSING'을 지정합니다.
            "status": "PROCESSING",
            # 구성 항목: 메시지(`'message'`)에 '비디오를 고속 배치 분석 중입니다. 잠시만 기다려주세요...'을 지정합니다.
            "message": "비디오를 고속 배치 분석 중입니다. 잠시만 기다려주세요..."
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }))

        # OpenCV 영상 입력 장치에 `cv2.VideoCapture(video_path)` 호출 결과를 저장합니다. 대상: `cap`.
        cap = cv2.VideoCapture(video_path)
        # 조건 `not cap.isOpened()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not cap.isOpened():
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()
            # `websocket.send(json.dumps({'type': 'ERROR', 'message': f'비디오 파일을 열 수 없습니다: {video_path}'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps({"type": "ERROR", "message": f"비디오 파일을 열 수 없습니다: {video_path}"}))
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return

        # 영상의 초당 프레임 수에 `cap.get(cv2.CAP_PROP_FPS)` 호출 결과를 저장합니다. 대상: `fps`.
        fps = cap.get(cv2.CAP_PROP_FPS)
        # 영상의 초당 프레임 수에 `np.isfinite(fps) and fps > 0`이면 `fps`, 아니면 `30.0`을 저장합니다. 대상: `fps`.
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        # total 프레임 묶음에 `int(cap.get(cv2.CAP_PROP_FRAME_COUNT))` 호출 결과를 저장합니다. 대상: `total_frames`.
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # 프레임 길이에 계산식 `1.0 / fps`의 결과를 저장합니다. 대상: `frame_duration`.
        frame_duration = 1.0 / fps

        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 환자 정보 및 세션 로그 관리자에 `self._prepare_data_manager(data, video_fps=fps)` 호출 결과를 저장합니다. 대상: `data_manager`.
            data_manager = self._prepare_data_manager(data, video_fps=fps)
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception:
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()
            # 이 조건에서는 처리를 계속할 수 없으므로 `None` 예외를 발생시킵니다.
            raise
        # `websocket.send(json.dumps({'type': 'SESSION_LOG_STARTED', **data_manager.session_log.info}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await websocket.send(json.dumps({"type": "SESSION_LOG_STARTED", **data_manager.session_log.info}))
        # exercise 설정에 `load_exercise_config(exercise_name)` 호출 결과를 저장합니다. 대상: `exercise_config`.
        exercise_config = load_exercise_config(exercise_name)
        # 환자에게 저장된 좌우 개인 가동 범위에 `mode in ['MAIN', 'TEST']`이면 `data_manager.get_custom_threshold(exercise_name)`, 아니면 `None`을 저장합니다. 대상: `custom_thresholds`.
        custom_thresholds = data_manager.get_custom_threshold(exercise_name) if mode in ["MAIN", "TEST"] else None
        # 조건 `custom_thresholds and 'left' not in custom_thresholds and ('start_val' in custom_thresholds)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            # 환자에게 저장된 좌우 개인 가동 범위에 키 'left', 'right'로 구성한 사전을 저장합니다. 대상: `custom_thresholds`.
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        # effective 반복 횟수에 `mode == 'CALIBRATION'`이면 `3`, 아니면 `target_score`을 저장합니다. 대상: `effective_reps`.
        effective_reps = 3 if mode == "CALIBRATION" else target_score
        # 관절 측정값 계산 및 반복 판정 엔진에 `MotionEngine(config=exercise_config, custom_thresholds=custom_thresholds, mode=mode, target_reps=effective_reps, target_score=target_score)` 호출 결과를 저장합니다. 대상: `motion_engine`.
        motion_engine = MotionEngine(
            # 구성 항목: 현재 작업 설정(`config`)에 exercise 설정(`exercise_config`)을 지정합니다.
            config=exercise_config,
            # 구성 항목: 환자에게 저장된 좌우 개인 가동 범위(`custom_thresholds`)에 환자에게 저장된 좌우 개인 가동 범위(`custom_thresholds`)을 지정합니다.
            custom_thresholds=custom_thresholds,
            # 구성 항목: 실행 모드(`mode`)에 실행 모드(`mode`)을 지정합니다.
            mode=mode,
            # 구성 항목: 목표 반복 횟수(`target_reps`)에 effective 반복 횟수(`effective_reps`); 게임 목표 점수(`target_score`)에 게임 목표 점수(`target_score`)을 지정합니다.
            target_reps=effective_reps, target_score=target_score
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        )
        # `setattr(motion_engine, 'exercise_name', exercise_name)`을 호출하여 해당 단계의 작업을 수행합니다.
        setattr(motion_engine, "exercise_name", exercise_name)

        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(data_manager=data_manager, motion_engine=motion_engine)` 호출 결과를 저장합니다. 대상: `self.controller`.
        self.controller = SessionController(data_manager=data_manager, motion_engine=motion_engine)
        # 평활화 엔진에 `OneEuroKeypointFilter()` 호출 결과를 저장합니다. 대상: `self.filter_engine`.
        self.filter_engine = OneEuroKeypointFilter()
        # `self._reset_raw_buffers()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._reset_raw_buffers()

        # Step 1: 전체 프레임 GPU 고속 배치 키포인트 추출
        # all extracted 관절에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `all_extracted_keypoints`.
        all_extracted_keypoints = []
        # 한 번에 추론할 영상 프레임 목록에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `batch_frames`.
        batch_frames = []
        # decoded 프레임 묶음에 0을 저장합니다. 대상: `decoded_frames`.
        decoded_frames = 0
        # 중지 request에 None을 저장합니다. 대상: `stop_request`.
        stop_request = None
        # t 시작에 시스템 현재 시각(초)(`time.time()`)을 저장합니다. 대상: `t_start`.
        t_start = time.time()
        # last 진행률에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `last_progress`.
        last_progress = time.perf_counter()

        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:
                # `(ret, frame)` 값에 `cap.read()` 호출 결과를 저장합니다. 대상: `(ret, frame)`.
                ret, frame = cap.read()
                # 조건 `not ret or frame is None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if not ret or frame is None:
                    # 조건 `batch_frames`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if batch_frames:
                        # kpts 일괄 처리에 계산식 `await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)`의 결과를 저장합니다. 대상: `kpts_batch`.
                        kpts_batch = await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)
                        # `all_extracted_keypoints.extend(kpts_batch)`을 호출하여 여러 항목을 순서대로 추가합니다.
                        all_extracted_keypoints.extend(kpts_batch)
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break

                # 배치 추론에 걸린 실제 시간이 아닌 원본 영상의 프레임 시각을 사용합니다.
                # `data_manager.session_log.write_video_frame(frame, decoded_frames * frame_duration)`을 호출하여 해당 단계의 작업을 수행합니다.
                data_manager.session_log.write_video_frame(frame, decoded_frames * frame_duration)
                # decoded 프레임 묶음에 `1`을 더해 누적합니다. 대상: `decoded_frames`.
                decoded_frames += 1
                # `batch_frames.append(frame)`을 호출하여 항목 하나를 끝에 추가합니다.
                batch_frames.append(frame)
                # 조건 `len(batch_frames) >= batch_size`을 만족하는 경우에만 아래 분기를 실행합니다.
                if len(batch_frames) >= batch_size:
                    # kpts 일괄 처리에 계산식 `await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)`의 결과를 저장합니다. 대상: `kpts_batch`.
                    kpts_batch = await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)
                    # `all_extracted_keypoints.extend(kpts_batch)`을 호출하여 여러 항목을 순서대로 추가합니다.
                    all_extracted_keypoints.extend(kpts_batch)
                    # `batch_frames.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
                    batch_frames.clear()
                    # 중지 request에 계산식 `await self._poll_video_stop(websocket)`의 결과를 저장합니다. 대상: `stop_request`.
                    stop_request = await self._poll_video_stop(websocket)
                    # 조건 `stop_request`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if stop_request:
                        # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                        break
                    # 조건 `time.perf_counter() - last_progress >= 1.0`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if time.perf_counter() - last_progress >= 1.0:
                        # `websocket.send(json.dumps({'type': 'VIDEO_PROGRESS', 'phase': 'EXTRACTING', 'processed_frames': decoded_frames, 'total_frames': total_frames}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                        await websocket.send(json.dumps({
                            # 구성 항목: `type` 값(`'type'`)에 'VIDEO_PROGRESS'; `phase` 값(`'phase'`)에 'EXTRACTING'을 지정합니다.
                            "type": "VIDEO_PROGRESS", "phase": "EXTRACTING",
                            # 구성 항목: processed 프레임 묶음(`'processed_frames'`)에 decoded 프레임 묶음(`decoded_frames`); total 프레임 묶음(`'total_frames'`)에 total 프레임 묶음(`total_frames`)을 지정합니다.
                            "processed_frames": decoded_frames, "total_frames": total_frames}))
                        # last 진행률에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `last_progress`.
                        last_progress = time.perf_counter()
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[MODE 3: VIDEO_BATCH] Extracted {len(all_extracted_keypoints)} frames in {time.time() - t_start:.2f}s")

        # Step 2: 프레임 누락 없는 순차 FSM 판정 (양측 3회 완수 검사)
        # 단계 대기 시간이 시작된 시각에 0.0을 저장합니다. 대상: `self.controller.timer_start`.
        self.controller.timer_start = 0.0

        # `enumerate(all_extracted_keypoints)`에서 항목을 차례로 꺼내 `(frame_idx, raw_kpts)`에 넣고 아래 처리를 반복합니다.
        for frame_idx, raw_kpts in enumerate(all_extracted_keypoints):
            # 조건 `stop_request`을 만족하는 경우에만 아래 분기를 실행합니다.
            if stop_request:
                # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                break
            # virtual 시각에 계산식 `frame_idx * frame_duration`의 결과를 저장합니다. 대상: `virtual_time`.
            virtual_time = frame_idx * frame_duration
            # `filtered_kpts` 값에 `self.filter_engine.update(raw_kpts, timestamp=virtual_time)` 호출 결과를 저장합니다. 대상: `filtered_kpts`.
            filtered_kpts = self.filter_engine.update(raw_kpts, timestamp=virtual_time)
            # 필수 관절이 가려졌거나 유효하지 않은지 여부에 `filtered_kpts`이면 `self.controller.check_occlusion(filtered_kpts)`, 아니면 `True`을 저장합니다. 대상: `is_occluded`.
            is_occluded = self.controller.check_occlusion(filtered_kpts) if filtered_kpts else True

            # 조건 `mode == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if mode == "CALIBRATION":
                # `cal_res` 값에 `self.controller.process_calibration_frame(filtered_kpts, is_occluded, now=virtual_time)` 호출 결과를 저장합니다. 대상: `cal_res`.
                cal_res = self.controller.process_calibration_frame(filtered_kpts, is_occluded, now=virtual_time)
                # `self._append_to_buffers(self.controller.last_left_data, self.controller.last_right_data, raw_kpts, elapsed_sec=virtual_time, frame_idx=frame_idx, is_occluded=is_occluded…`을 호출하여 해당 단계의 작업을 수행합니다.
                self._append_to_buffers(self.controller.last_left_data, self.controller.last_right_data,
                    # 구성 항목: `elapsed_sec` 값(`elapsed_sec`)에 virtual 시각(`virtual_time`); 분석 프레임 번호(`frame_idx`)에 분석 프레임 번호(`frame_idx`); 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)을 지정합니다.
                    raw_kpts, elapsed_sec=virtual_time, frame_idx=frame_idx, is_occluded=is_occluded,
                    # 구성 항목: 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)에 `data_manager.session_log.record_video`이면 `frame_idx`, 아니면 `-1`을 지정합니다.
                    video_frame_idx=frame_idx if data_manager.session_log.record_video else -1)

                # 조건 `cal_res['calib_step'] == 'FINISHED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if cal_res["calib_step"] == "FINISHED":
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(f"[MODE 3: VIDEO_BATCH] 양측 캘리브레이션 3회 완수 완료 (Frame: {frame_idx}/{total_frames})")
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break

            # 조건 `mode in ['MAIN', 'TEST']`을 만족하는 경우에만 아래 분기를 실행합니다.
            elif mode in ["MAIN", "TEST"]:
                # `(left_data, right_data, is_finished)` 값에 `self.controller.process_main_frame(filtered_kpts, is_occluded, now=virtual_time)` 호출 결과를 저장합니다. 대상: `(left_data, right_data, is_finished)`.
                left_data, right_data, is_finished = self.controller.process_main_frame(filtered_kpts, is_occluded, now=virtual_time)
                # `self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=virtual_time, frame_idx=frame_idx, is_occluded=is_occluded, video_frame_idx=frame_idx if data_manage…`을 호출하여 해당 단계의 작업을 수행합니다.
                self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=virtual_time,
                    # 구성 항목: 분석 프레임 번호(`frame_idx`)에 분석 프레임 번호(`frame_idx`); 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)을 지정합니다.
                    frame_idx=frame_idx, is_occluded=is_occluded,
                    # 구성 항목: 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)에 `data_manager.session_log.record_video`이면 `frame_idx`, 아니면 `-1`을 지정합니다.
                    video_frame_idx=frame_idx if data_manager.session_log.record_video else -1)

                # `self._send_rep_events(websocket, left_data, right_data, virtual_time)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await self._send_rep_events(websocket, left_data, right_data, virtual_time)
            # 중지 request에 계산식 `await self._poll_video_stop(websocket)`의 결과를 저장합니다. 대상: `stop_request`.
            stop_request = await self._poll_video_stop(websocket)

        # Step 3: 파일 저장 및 완료 결과 일괄 전송
        # `prefix` 값에 `mode == 'CALIBRATION'`이면 `'CALIB'`, 아니면 `'SESSION'`을 저장합니다. 대상: `prefix`.
        prefix = "CALIB" if mode == "CALIBRATION" else "SESSION"
        # 캘리브레이션 complete에 계산식 `mode == 'CALIBRATION' and self.controller.calib_step == 'FINISHED'`의 결과를 저장합니다. 대상: `calibration_complete`.
        calibration_complete = mode == "CALIBRATION" and self.controller.calib_step == "FINISHED"
        # `custom_th` 값에 `calibration_complete`이면 `self.controller.compute_and_save_thresholds(exercise_name)`, 아니면 `None`을 저장합니다. 대상: `custom_th`.
        custom_th = self.controller.compute_and_save_thresholds(exercise_name) if calibration_complete else None
        # `final_step` 값에 `mode == 'CALIBRATION'`이면 `self.controller.calib_step`, 아니면 `None`을 저장합니다. 대상: `final_step`.
        final_step = self.controller.calib_step if mode == "CALIBRATION" else None
        # 조건 `stop_request`을 만족하는 경우에만 아래 분기를 실행합니다.
        if stop_request:
            # `self._record_client_result(stop_request)`을 호출하여 해당 단계의 작업을 수행합니다.
            self._record_client_result(stop_request)
        # 요약 meta에 `self._save_session_files(prefix=prefix, status='COMPLETED' if stop_request and stop_request.get('reason') == 'TARGET_REACHED' else 'STOPPED' if stop_request else 'INCOMP…` 호출 결과를 저장합니다. 대상: `summary_meta`.
        summary_meta = self._save_session_files(
            # 구성 항목: `prefix` 값(`prefix`)에 `prefix` 값(`prefix`); 상태(`status`)에 `stop_request and stop_request.get('reason') == 'TARGET_REACHED'`이면 `'COMPLETED'`, 아니면 `'STOPPED' if stop_request else 'INCOMPLETE' if mode == 'CALIBRATION' and (not calibration_complete) else 'COMPLETED'`을 지정합니다.
            prefix=prefix, status=("COMPLETED" if stop_request and stop_request.get("reason") == "TARGET_REACHED"
                                   # 앞줄에서 시작한 식·인수·자료 구조를 `else "STOPPED" if stop_request` 내용으로 이어서 구성합니다.
                                   else "STOPPED" if stop_request
                                   # 앞줄에서 시작한 식·인수·자료 구조를 `else "INCOMPLETE" if mode == "CALIBRATION" and not calibration_complete else "COMPLETED"))` 내용으로 이어서 구성합니다.
                                   else "INCOMPLETE" if mode == "CALIBRATION" and not calibration_complete else "COMPLETED"))

        # `payload` 값에 키 'type', 'session_id', 'request_id', 'reason', 'mode', 'video_path', 'total_frames_analyzed', 'source_duration…로 구성한 사전을 저장합니다. 대상: `payload`.
        payload = {
            # 구성 항목: `type` 값(`'type'`)에 `stop_request`이면 `'SESSION_FINISHED'`, 아니면 `'VIDEO_ANALYSIS_COMPLETED'`을 지정합니다.
            "type": "SESSION_FINISHED" if stop_request else "VIDEO_ANALYSIS_COMPLETED",
            # 구성 항목: 현재 측정 세션 식별자(`'session_id'`)에 현재 측정 세션 식별자(`data_manager.session_log.session_id`)을 지정합니다.
            "session_id": data_manager.session_log.session_id,
            # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 `stop_request`이면 `stop_request.get('request_id')`, 아니면 `None`을 지정합니다.
            "request_id": stop_request.get("request_id") if stop_request else None,
            # 구성 항목: `reason` 값(`'reason'`)에 `stop_request`이면 `stop_request.get('reason', 'USER_STOP')`, 아니면 `'END_OF_VIDEO'`을 지정합니다.
            "reason": stop_request.get("reason", "USER_STOP") if stop_request else "END_OF_VIDEO",
            # 구성 항목: 실행 모드(`'mode'`)에 실행 모드(`mode`)을 지정합니다.
            "mode": mode,
            # 구성 항목: 서버 PC에서 읽을 원본 영상 경로(`'video_path'`)에 서버 PC에서 읽을 원본 영상 경로(`video_path`)을 지정합니다.
            "video_path": video_path,
            # 구성 항목: total 프레임 묶음 analyzed(`'total_frames_analyzed'`)에 `len(all_extracted_keypoints)` 호출 결과를 지정합니다.
            "total_frames_analyzed": len(all_extracted_keypoints),
            # 구성 항목: 원본 길이 sec(`'source_duration_sec'`)에 계산식 `decoded_frames * frame_duration`의 결과를 지정합니다.
            "source_duration_sec": decoded_frames * frame_duration,
            # 구성 항목: 저장·종료 결과 요약(`'summary'`)에 요약 meta(`summary_meta`)을 지정합니다.
            "summary": summary_meta
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
        # 조건 `mode == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if mode == "CALIBRATION":
            # `payload['thresholds']` 값에 `custom_th` 값(`custom_th`)을 저장합니다. 대상: `payload['thresholds']`.
            payload["thresholds"] = custom_th
            # `payload['calibration_completed']` 값에 캘리브레이션 complete(`calibration_complete`)을 저장합니다. 대상: `payload['calibration_completed']`.
            payload["calibration_completed"] = calibration_complete
            # `payload['calib_step']` 값에 `final_step` 값(`final_step`)을 저장합니다. 대상: `payload['calib_step']`.
            payload["calib_step"] = final_step

        # 조건 `stop_request`을 만족하는 경우에만 아래 분기를 실행합니다.
        if stop_request:
            # `payload['status']` 값에 `stop_request.get('reason') == 'TARGET_REACHED'`이면 `'COMPLETED'`, 아니면 `'STOPPED'`을 저장합니다. 대상: `payload['status']`.
            payload["status"] = "COMPLETED" if stop_request.get("reason") == "TARGET_REACHED" else "STOPPED"
        # `websocket.send(json.dumps(payload))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
        await websocket.send(json.dumps(payload))
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[MODE 3: VIDEO_BATCH] Completed successfully.")
        # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
        self.controller = None

    # ==========================================================================
    # 메인 웹소켓 루프 (모드 2 실시간 처리)
    # ==========================================================================
    # 이전 측정 연결의 종료와 자원 정리 완료를 확인하는 복구 명령입니다.
    async def reset_session_connection(self, websocket, packet):
        # 새 화면의 연결에서 이전 측정 연결만 닫습니다. 모델과 수신 소켓은 유지합니다.
        owner = self._session_owner
        # 같은 측정 연결에서는 일반 중지 명령을 사용해야 응답 연결이 끊기지 않습니다.
        if owner is websocket:
            await websocket.send(json.dumps({"type": "ERROR", "code": "RESET_FROM_MONITOR_REQUIRED",
                "message": "새 연결에서 복구하거나 현재 연결에서 CMD_STOP_SESSION을 사용하세요."}))
            return
        # 연결 핸들러의 finally가 로그 저장과 카메라 해제를 마칠 때까지 기다립니다.
        done = self._connection_done.get(id(owner))
        if owner is not None:
            try:
                # 연결을 닫아 기존 핸들러가 스스로 안전한 종료 경로로 들어가게 합니다.
                await asyncio.wait_for(owner.close(code=4001, reason="Session reset requested"), 5.0)
                # 추론 작업 도중 공유 controller를 강제로 지우지 않습니다.
                if done is not None:
                    await asyncio.wait_for(done.wait(), 15.0)
                # 종료 신호가 없는 외부 연결은 완료를 확인할 수 없으므로 성공으로 알리지 않습니다.
                if self._session_owner is owner:
                    raise asyncio.TimeoutError()
            except asyncio.TimeoutError:
                await websocket.send(json.dumps({"type": "ERROR", "code": "SESSION_RESET_TIMEOUT",
                    "message": "이전 세션 정리가 아직 끝나지 않았습니다. 잠시 후 다시 복구하세요."}))
                return
        # 이전 연결의 실제 정리가 완료된 뒤에만 새 세션을 시작해도 된다고 알립니다.
        await websocket.send(json.dumps({"type": "SERVER_SESSION_RESET", "reset": owner is not None,
            "request_id": packet.get("request_id"), "message": "이전 세션 연결 정리를 완료했습니다."}))

    async def handle_client(self, websocket):
        # 이 연결의 종료 정리 완료 신호를 등록합니다.
        connection_done = asyncio.Event()
        self._connection_done[id(websocket)] = connection_done
        # 클라이언트 address에 `websocket.remote_address` 값(`websocket.remote_address`)을 저장합니다. 대상: `client_address`.
        client_address = websocket.remote_address
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[DEBUG][NETWORK] Client connected: {client_address}")

        # prev 시각에 시스템 현재 시각(초)(`time.time()`)을 저장합니다. 대상: `prev_time`.
        prev_time = time.time()
        # 대기 측정 세션에 None을 저장합니다. 대상: `pending_session`.
        pending_session = None
        # 대기 명령에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `pending_commands`.
        pending_commands = []
        # last 모델 상태에 None을 저장합니다. 대상: `last_model_status`.
        last_model_status = None
        # end 상태에 'STOPPED'을 저장합니다. 대상: `end_status`.
        end_status = "STOPPED"
        # `self.model_loader.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.model_loader.start()

        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:
                # 상태에 `self.model_loader.snapshot()` 호출 결과를 저장합니다. 대상: `status`.
                status = self.model_loader.snapshot()
                # 상태 key에 순서·구성을 지정한 `(status['model_state'], status['phase'], status.get('model_error'))`을 저장합니다. 대상: `status_key`.
                status_key = (status["model_state"], status["phase"], status.get("model_error"))
                # 조건 `status_key != last_model_status`을 만족하는 경우에만 아래 분기를 실행합니다.
                if status_key != last_model_status:
                    # `websocket.send(json.dumps(status))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await websocket.send(json.dumps(status))
                    # last 모델 상태에 상태 key(`status_key`)을 저장합니다. 대상: `last_model_status`.
                    last_model_status = status_key
                # 조건 `pending_session and status['model_state'] == 'FAILED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if pending_session and status["model_state"] == "FAILED":
                    # `websocket.send(json.dumps({'type': 'ERROR', 'code': 'MODEL_LOAD_FAILED', 'fatal': True, 'message': f"모델 준비 실패: {status.get('model_error')}"}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await websocket.send(json.dumps({
                        # 구성 항목: `type` 값(`'type'`)에 'ERROR'; `code` 값(`'code'`)에 'MODEL_LOAD_FAILED'; `fatal` 값(`'fatal'`)에 True을 지정합니다.
                        "type": "ERROR", "code": "MODEL_LOAD_FAILED", "fatal": True,
                        # 구성 항목: 메시지(`'message'`)에 계산식 `f"모델 준비 실패: {status.get('model_error')}"`의 결과를 지정합니다.
                        "message": f"모델 준비 실패: {status.get('model_error')}",
                    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                    }))
                    # 대기 측정 세션에 None을 저장합니다. 대상: `pending_session`.
                    pending_session = None
                    # `pending_commands.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
                    pending_commands.clear()
                # 조건 `pending_session and status['ready']`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif pending_session and status["ready"]:
                    # 사람 검출·관절 추출 AI 엔진에 생성 또는 검증할 분석 엔진(`self.model_loader.engine`)을 저장합니다. 대상: `self.skeleton_engine`.
                    self.skeleton_engine = self.model_loader.engine
                    # `(packet_to_start, pending_session)` 값에 순서·구성을 지정한 `(pending_session, None)`을 저장합니다. 대상: `(packet_to_start, pending_session)`.
                    packet_to_start, pending_session = pending_session, None
                    # `accepted` 값에 계산식 `await self._start_session(websocket, packet_to_start)`의 결과를 저장합니다. 대상: `accepted`.
                    accepted = await self._start_session(websocket, packet_to_start)
                    # 조건 `accepted`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if accepted:
                        # `pending_commands`에서 항목을 차례로 꺼내 `command`에 넣고 아래 처리를 반복합니다.
                        for command in pending_commands:
                            # `websocket.send(json.dumps(handle_count_threshold_command(self.controller, command)))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                            await websocket.send(json.dumps(handle_count_threshold_command(self.controller, command)))
                    # `pending_commands.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
                    pending_commands.clear()
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `raw_msg` 값에 계산식 `await asyncio.wait_for(websocket.recv(), timeout=0.01)`의 결과를 저장합니다. 대상: `raw_msg`.
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(raw_msg)`)을 저장합니다. 대상: `packet`.
                    packet = json.loads(raw_msg)
                    # `pkt_type` 값에 `packet.get('type')` 호출 결과를 저장합니다. 대상: `pkt_type`.
                    pkt_type = packet.get("type")

                    # 조건 `pkt_type == 'CMD_SET_SESSION'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if pkt_type == "CMD_SET_SESSION":
                        # 대기 측정 세션에 수신 또는 전송할 JSON 메시지 사전(`packet`)을 저장합니다. 대상: `pending_session`.
                        pending_session = packet
                        # `pending_commands.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
                        pending_commands.clear()
                        # 조건 `not self.model_loader.snapshot()['ready']`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if not self.model_loader.snapshot()["ready"]:
                            # `websocket.send(json.dumps({'type': 'SESSION_PENDING', 'message': '모델 준비가 끝나면 요청한 세션을 시작합니다.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                            await websocket.send(json.dumps({
                                # 구성 항목: `type` 값(`'type'`)에 'SESSION_PENDING'을 지정합니다.
                                "type": "SESSION_PENDING",
                                # 구성 항목: 메시지(`'message'`)에 '모델 준비가 끝나면 요청한 세션을 시작합니다.'을 지정합니다.
                                "message": "모델 준비가 끝나면 요청한 세션을 시작합니다.",
                            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                            }))

                    # 조건 `pkt_type == 'CMD_STOP_SESSION'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif pkt_type == "CMD_STOP_SESSION":
                        # 조건 `pending_session and self._session_owner in (None, websocket)`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if pending_session and self._session_owner in (None, websocket):
                            # 대기 측정 세션에 None을 저장합니다. 대상: `pending_session`.
                            pending_session = None
                            # `pending_commands.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
                            pending_commands.clear()
                            # `websocket.send(json.dumps({'type': 'SESSION_FINISHED', 'status': 'STOPPED', 'reason': 'USER_STOP', 'request_id': packet.get('request_id'), 'summary': None}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                            await websocket.send(json.dumps({"type": "SESSION_FINISHED", "status": "STOPPED",
                                # 구성 항목: `reason` 값(`'reason'`)에 'USER_STOP'; 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 `packet.get('request_id')` 호출 결과; 저장·종료 결과 요약(`'summary'`)에 None을 지정합니다.
                                "reason": "USER_STOP", "request_id": packet.get("request_id"), "summary": None}))
                        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                        else:
                            # `self._finish_client_session(websocket, packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                            await self._finish_client_session(websocket, packet)

                    # 조건 `pkt_type == 'CMD_GET_SERVER_STATUS'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif pkt_type == "CMD_RESET_SESSION":
                        # 별도 연결에서 요청한 명시적 복구 명령입니다. 자동으로 다른 세션을 끊지는 않습니다.
                        await self.reset_session_connection(websocket, packet)

                    elif pkt_type == "CMD_GET_SERVER_STATUS":
                        # `websocket.send(json.dumps(self.model_loader.snapshot()))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                        await websocket.send(json.dumps(self.model_loader.snapshot()))

                    # 조건 `pkt_type == 'CMD_STOP_CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif pkt_type == "CMD_STOP_CALIBRATION":
                        # 조건 `self._session_owner is websocket and self.controller and (self.controller.mode == 'CALIBRATION')`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if self._session_owner is websocket and self.controller and self.controller.mode == "CALIBRATION":
                            # 캘리브레이션 전체 진행 단계에 'FINISHED'을 저장합니다. 대상: `self.controller.calib_step`.
                            self.controller.calib_step = "FINISHED"

                    # 조건 `pkt_type in COUNT_THRESHOLD_COMMANDS`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif pkt_type in COUNT_THRESHOLD_COMMANDS:
                        # 조건 `pending_session`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if pending_session:
                            # 조건 `len(pending_commands) < 16`을 만족하는 경우에만 아래 분기를 실행합니다.
                            if len(pending_commands) < 16:
                                # `pending_commands.append(packet)`을 호출하여 항목 하나를 끝에 추가합니다.
                                pending_commands.append(packet)
                        # 조건 `self._session_owner is not None and self._session_owner is not websocket`을 만족하는 경우에만 아래 분기를 실행합니다.
                        elif self._session_owner is not None and self._session_owner is not websocket:
                            # `websocket.send(json.dumps({'type': 'ERROR', 'code': 'SERVER_BUSY', 'command': pkt_type, 'request_id': packet.get('request_id'), 'message': '다른 연결에서 세션을 사용 중입니다.'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                            await websocket.send(json.dumps({"type": "ERROR", "code": "SERVER_BUSY",
                                # 구성 항목: 명령(`'command'`)에 `pkt_type` 값(`pkt_type`); 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 `packet.get('request_id')` 호출 결과를 지정합니다.
                                "command": pkt_type, "request_id": packet.get("request_id"),
                                # 구성 항목: 메시지(`'message'`)에 '다른 연결에서 세션을 사용 중입니다.'을 지정합니다.
                                "message": "다른 연결에서 세션을 사용 중입니다."}))
                        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                        else:
                            # `response` 값에 `handle_count_threshold_command(self.controller, packet)` 호출 결과를 저장합니다. 대상: `response`.
                            response = handle_count_threshold_command(self.controller, packet)
                            # `websocket.send(json.dumps(response))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                            await websocket.send(json.dumps(response))

                # `asyncio.TimeoutError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except asyncio.TimeoutError:
                    # 이 분기에는 추가 동작이 없으며 문법적으로 필요한 빈 블록만 유지합니다.
                    pass

                # 조건 `self._session_owner is websocket and self.controller and self.camera`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self._session_owner is websocket and self.controller and self.camera:
                    # 조건 `not await self._measurement_ready(websocket)`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if not await self._measurement_ready(websocket):
                        # `asyncio.sleep(0.01)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
                        await asyncio.sleep(0.01)
                        # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                        continue
                    # `(ret, frame, sample_time, capture_id)` 값에 `self.camera.read_sample()` 호출 결과를 저장합니다. 대상: `(ret, frame, sample_time, capture_id)`.
                    ret, frame, sample_time, capture_id = self.camera.read_sample()

                    # 조건 `ret and frame is not None and (capture_id != self._last_capture_id) and (sample_time >= self._session_started)`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if (ret and frame is not None and capture_id != self._last_capture_id
                            # 앞줄에서 시작한 식·인수·자료 구조를 `and sample_time >= self._session_started):` 내용으로 이어서 구성합니다.
                            and sample_time >= self._session_started):
                        # `self._last_capture_id` 값에 `capture_id` 값(`capture_id`)을 저장합니다. 대상: `self._last_capture_id`.
                        self._last_capture_id = capture_id
                        # 로그 elapsed에 계산식 `sample_time - self._session_started`의 결과를 저장합니다. 대상: `log_elapsed`.
                        log_elapsed = sample_time - self._session_started
                        # 영상 index에 `self.controller.data_manager.session_log.write_video_frame(frame, log_elapsed)` 호출 결과를 저장합니다. 대상: `video_index`.
                        video_index = self.controller.data_manager.session_log.write_video_frame(frame, log_elapsed)
                        # curr 시각에 시스템 현재 시각(초)(`time.time()`)을 저장합니다. 대상: `curr_time`.
                        curr_time = time.time()
                        # 시각 delta에 계산식 `curr_time - prev_time`의 결과를 저장합니다. 대상: `time_delta`.
                        time_delta = curr_time - prev_time
                        # 영상의 초당 프레임 수에 `time_delta > 0`이면 `round(1.0 / time_delta, 1)`, 아니면 `30.0`을 저장합니다. 대상: `fps`.
                        fps = round(1.0 / time_delta, 1) if time_delta > 0 else 30.0
                        # prev 시각에 curr 시각(`curr_time`)을 저장합니다. 대상: `prev_time`.
                        prev_time = curr_time

                        # 영상뿐 아니라 평활화·로그에도 카메라에서 프레임을 얻은 시각을 사용합니다.
                        # `raw_kpts` 값에 계산식 `await asyncio.to_thread(self.skeleton_engine.extract_keypoints, frame)`의 결과를 저장합니다. 대상: `raw_kpts`.
                        raw_kpts = await asyncio.to_thread(self.skeleton_engine.extract_keypoints, frame)
                        # `filtered_kpts` 값에 `self.filter_engine.update(raw_kpts, timestamp=sample_time)` 호출 결과를 저장합니다. 대상: `filtered_kpts`.
                        filtered_kpts = self.filter_engine.update(raw_kpts, timestamp=sample_time)
                        # `yolo_detected` 값에 계산식 `filtered_kpts is not None and len(filtered_kpts) > 0`의 결과를 저장합니다. 대상: `yolo_detected`.
                        yolo_detected = filtered_kpts is not None and len(filtered_kpts) > 0
                        # 필수 관절이 가려졌거나 유효하지 않은지 여부에 `yolo_detected`이면 `self.controller.check_occlusion(filtered_kpts)`, 아니면 `True`을 저장합니다. 대상: `is_occluded`.
                        is_occluded = self.controller.check_occlusion(filtered_kpts) if yolo_detected else True

                        # `current_mode` 값에 실행 모드(`self.controller.motion_engine.mode`)을 저장합니다. 대상: `current_mode`.
                        current_mode = self.controller.motion_engine.mode
                        # 왼쪽 최신 판정 결과에 last 왼쪽 데이터(`self.controller.last_left_data`)을 저장합니다. 대상: `left_data`.
                        left_data = self.controller.last_left_data
                        # 오른쪽 최신 판정 결과에 last 오른쪽 데이터(`self.controller.last_right_data`)을 저장합니다. 대상: `right_data`.
                        right_data = self.controller.last_right_data
                        # calib 반복 횟수 왼쪽에 0을 저장합니다. 대상: `calib_reps_left`.
                        calib_reps_left = 0
                        # calib 반복 횟수 오른쪽에 0을 저장합니다. 대상: `calib_reps_right`.
                        calib_reps_right = 0
                        # calib 반복 횟수에 0을 저장합니다. 대상: `calib_rep_count`.
                        calib_rep_count = 0

                        # --- CALIBRATION 진행 (양측 모두 3회) ---
                        # 조건 `current_mode == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if current_mode == "CALIBRATION":
                            # `cal_res` 값에 `self.controller.process_calibration_frame(filtered_kpts, is_occluded, now=sample_time)` 호출 결과를 저장합니다. 대상: `cal_res`.
                            cal_res = self.controller.process_calibration_frame(filtered_kpts, is_occluded, now=sample_time)
                            # calib 반복 횟수에 계산식 `cal_res['calib_rep_count']`의 결과를 저장합니다. 대상: `calib_rep_count`.
                            calib_rep_count = cal_res["calib_rep_count"]
                            # calib 반복 횟수 왼쪽에 계산식 `cal_res['calib_reps_left']`의 결과를 저장합니다. 대상: `calib_reps_left`.
                            calib_reps_left = cal_res["calib_reps_left"]
                            # calib 반복 횟수 오른쪽에 계산식 `cal_res['calib_reps_right']`의 결과를 저장합니다. 대상: `calib_reps_right`.
                            calib_reps_right = cal_res["calib_reps_right"]

                            # `(left_data, right_data)` 값에 순서·구성을 지정한 `(self.controller.last_left_data, self.controller.last_right_data)`을 저장합니다. 대상: `(left_data, right_data)`.
                            left_data, right_data = self.controller.last_left_data, self.controller.last_right_data
                            # `self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=log_elapsed, is_occluded=is_occluded, video_frame_idx=video_index)`을 호출하여 해당 단계의 작업을 수행합니다.
                            self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=log_elapsed,
                                # 구성 항목: 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`); 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)에 영상 index(`video_index`)을 지정합니다.
                                is_occluded=is_occluded, video_frame_idx=video_index)

                            # 조건 `cal_res['calib_step'] == 'FINISHED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                            if cal_res["calib_step"] == "FINISHED":
                                # `ex_name` 값에 `getattr(self.controller.motion_engine, 'exercise_name', 'biceps_curl')` 호출 결과를 저장합니다. 대상: `ex_name`.
                                ex_name = getattr(self.controller.motion_engine, "exercise_name", "biceps_curl")
                                # `custom_th` 값에 `self.controller.compute_and_save_thresholds(ex_name)` 호출 결과를 저장합니다. 대상: `custom_th`.
                                custom_th = self.controller.compute_and_save_thresholds(ex_name)
                                # calib 요약에 `self._save_session_files(prefix='CALIB')` 호출 결과를 저장합니다. 대상: `calib_summary`.
                                calib_summary = self._save_session_files(prefix="CALIB")

                                # `websocket.send(json.dumps({'type': 'CALIBRATION_FINISHED', 'player_id': self.controller.data_manager.player_id, 'thresholds': custom_th, 'summary': calib_summary}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                                await websocket.send(json.dumps({
                                    # 구성 항목: `type` 값(`'type'`)에 'CALIBRATION_FINISHED'을 지정합니다.
                                    "type": "CALIBRATION_FINISHED",
                                    # 구성 항목: 환자 식별 번호의 기존 필드명(`'player_id'`)에 환자 식별 번호의 기존 필드명(`self.controller.data_manager.player_id`)을 지정합니다.
                                    "player_id": self.controller.data_manager.player_id,
                                    # 구성 항목: 좌우 가동 범위 시작값·목표값(`'thresholds'`)에 `custom_th` 값(`custom_th`)을 지정합니다.
                                    "thresholds": custom_th,
                                    # 구성 항목: 저장·종료 결과 요약(`'summary'`)에 calib 요약(`calib_summary`)을 지정합니다.
                                    "summary": calib_summary
                                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                                }))
                                # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
                                self.controller = None
                                # `self.stop_camera()`을 호출하여 해당 단계의 작업을 수행합니다.
                                self.stop_camera()
                                # 현재 서버의 측정 자원을 점유한 연결에 None을 저장합니다. 대상: `self._session_owner`.
                                self._session_owner = None
                                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                                continue

                        # --- TEST / MAIN 진행 ---
                        # 조건 `current_mode in ['MAIN', 'TEST']`을 만족하는 경우에만 아래 분기를 실행합니다.
                        elif current_mode in ["MAIN", "TEST"]:
                            # `(left_data, right_data, is_finished)` 값에 `self.controller.process_main_frame(filtered_kpts, is_occluded)` 호출 결과를 저장합니다. 대상: `(left_data, right_data, is_finished)`.
                            left_data, right_data, is_finished = self.controller.process_main_frame(filtered_kpts, is_occluded)
                            # `self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=log_elapsed, is_occluded=is_occluded, video_frame_idx=video_index)`을 호출하여 해당 단계의 작업을 수행합니다.
                            self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=log_elapsed,
                                # 구성 항목: 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`); 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)에 영상 index(`video_index`)을 지정합니다.
                                is_occluded=is_occluded, video_frame_idx=video_index)

                            # `self._send_rep_events(websocket, left_data, right_data, log_elapsed)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                            await self._send_rep_events(websocket, left_data, right_data, log_elapsed)

                        # --- 이미지 없이 수치만 전송 ---
                        # `pose_payload` 값에 키 'type', 'mode', 'left', 'right', 'fps', 'is_occluded', 'calib_step', 'calib_rep_count', 'calib_reps_left', 'c…로 구성한 사전을 저장합니다. 대상: `pose_payload`.
                        pose_payload = {
                            # 구성 항목: `type` 값(`'type'`)에 'POSE_UPDATE'을 지정합니다.
                            "type": "POSE_UPDATE",
                            # 구성 항목: 실행 모드(`'mode'`)에 `current_mode` 값(`current_mode`)을 지정합니다.
                            "mode": current_mode,
                            # 구성 항목: 왼쪽(`'left'`)에 왼쪽 최신 판정 결과(`left_data`)을 지정합니다.
                            "left": left_data,
                            # 구성 항목: 오른쪽(`'right'`)에 오른쪽 최신 판정 결과(`right_data`)을 지정합니다.
                            "right": right_data,
                            # 구성 항목: 영상의 초당 프레임 수(`'fps'`)에 영상의 초당 프레임 수(`fps`)을 지정합니다.
                            "fps": fps,
                            # 구성 항목: 필수 관절이 가려졌거나 유효하지 않은지 여부(`'is_occluded'`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)을 지정합니다.
                            "is_occluded": is_occluded,
                            # 구성 항목: 캘리브레이션 전체 진행 단계(`'calib_step'`)에 `self.controller`이면 `self.controller.calib_step`, 아니면 `'FINISHED'`을 지정합니다.
                            "calib_step": self.controller.calib_step if self.controller else "FINISHED",
                            # 구성 항목: calib 반복 횟수(`'calib_rep_count'`)에 calib 반복 횟수(`calib_rep_count`)을 지정합니다.
                            "calib_rep_count": calib_rep_count,
                            # 구성 항목: calib 반복 횟수 왼쪽(`'calib_reps_left'`)에 calib 반복 횟수 왼쪽(`calib_reps_left`)을 지정합니다.
                            "calib_reps_left": calib_reps_left,
                            # 구성 항목: calib 반복 횟수 오른쪽(`'calib_reps_right'`)에 calib 반복 횟수 오른쪽(`calib_reps_right`)을 지정합니다.
                            "calib_reps_right": calib_reps_right,
                            # 구성 항목: 좌우 각각 필요한 캘리브레이션 횟수(`'target_calib_reps'`)에 3을 지정합니다.
                            "target_calib_reps": 3,
                        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                        }

                        # `pose_payload.update(self.controller.motion_engine.count_threshold_status())`을 호출하여 해당 단계의 작업을 수행합니다.
                        pose_payload.update(self.controller.motion_engine.count_threshold_status())
                        # `websocket.send(json.dumps(pose_payload))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                        await websocket.send(json.dumps(pose_payload))

                # `asyncio.sleep(0.001)`을 호출하여 지정한 시간만큼 다음 처리를 늦춤합니다. 완료될 때까지 비동기로 기다립니다.
                await asyncio.sleep(0.001)

        # `ConnectionClosed` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except ConnectionClosed:
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"[DEBUG][NETWORK] Client disconnected: {client_address}")
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception as e:
            # end 상태에 'FAILED'을 저장합니다. 대상: `end_status`.
            end_status = "FAILED"
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"[ERROR][SERVER] Exception: {e}")
            # `traceback.print_exc()`을 호출하여 현재 예외의 상세 호출 경로를 출력합니다.
            traceback.print_exc()
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `websocket.send(json.dumps({'type': 'ERROR', 'code': 'SESSION_PROCESSING_FAILED', 'message': str(e)}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await websocket.send(json.dumps({"type": "ERROR", "code": "SESSION_PROCESSING_FAILED",
                                                  # 구성 항목: 메시지(`'message'`)에 `str(e)` 호출 결과를 지정합니다.
                                                  "message": str(e)}))
            # `ConnectionClosed` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except ConnectionClosed:
                # 이 분기에는 추가 동작이 없으며 문법적으로 필요한 빈 블록만 유지합니다.
                pass
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # 조건 `self._session_owner is websocket`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self._session_owner is websocket:
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # 조건 `self._active_log is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if self._active_log is not None:
                        # 조건 `self.controller`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if self.controller:
                            # `self._save_session_files(status=end_status)`을 호출하여 해당 단계의 작업을 수행합니다.
                            self._save_session_files(status=end_status)
                        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                        else:
                            # `self._active_log.finish([], status=end_status)`을 호출하여 해당 단계의 작업을 수행합니다.
                            self._active_log.finish([], status=end_status)
                # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except Exception:
                    # `traceback.print_exc()`을 호출하여 현재 예외의 상세 호출 경로를 출력합니다.
                    traceback.print_exc()
                # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
                finally:
                    # 현재 기록 중인 세션 로그에 None을 저장합니다. 대상: `self._active_log`.
                    self._active_log = None
                    # `self.stop_camera()`을 호출하여 해당 단계의 작업을 수행합니다.
                    self.stop_camera()
                    # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
                    self.controller = None
                    # 현재 서버의 측정 자원을 점유한 연결에 None을 저장합니다. 대상: `self._session_owner`.
                    self._session_owner = None

            # 소유권이 없는 상태 조회 연결도 등록 신호를 제거하여 연결 객체가 누적되지 않게 합니다.
            self._connection_done.pop(id(websocket), None)
            # 카메라·로그 정리까지 끝난 시점에 복구 요청을 기다리는 연결을 깨웁니다.
            connection_done.set()

    # 실행 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def run(self):
        # `websockets.serve(self.handle_client, self.host, self.port)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.serve(self.handle_client, self.host, self.port) as listener:
            # `self.bound_port` 값에 계산식 `listener.sockets[0].getsockname()[1]`의 결과를 저장합니다. 대상: `self.bound_port`.
            self.bound_port = listener.sockets[0].getsockname()[1]
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"[DEBUG][SERVER] ExerciseSocketServer listening on ws://{self.host}:{self.bound_port}", flush=True)
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"[STARTUP] Socket listening after {time.perf_counter() - _BOOT_STARTED:.3f}s; models load in background", flush=True)
            # `self.model_loader.start()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.model_loader.start()
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `asyncio.Future()`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await asyncio.Future()
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # `task` 값에 `self.model_loader.task` 값(`self.model_loader.task`)을 저장합니다. 대상: `task`.
                task = self.model_loader.task
                # 조건 `task is not None and (not task.done())`을 만족하는 경우에만 아래 분기를 실행합니다.
                if task is not None and not task.done():
                    # `task.cancel()`을 호출하여 해당 단계의 작업을 수행합니다.
                    task.cancel()
                    # `asyncio.gather(task, return_exceptions=True)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                    await asyncio.gather(task, return_exceptions=True)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == "__main__":
    # 서버에 `ExerciseSocketServer()` 호출 결과를 저장합니다. 대상: `server`.
    server = ExerciseSocketServer()
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # `asyncio.run(server.run())`을 호출하여 해당 단계의 작업을 수행합니다.
        asyncio.run(server.run())
    # `KeyboardInterrupt` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
    except KeyboardInterrupt:
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print("\n[DEBUG][SERVER] Server stopped by user.")
