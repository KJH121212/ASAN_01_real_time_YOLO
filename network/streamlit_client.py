# 한국어 코드 해설: 화면 재실행과 독립된 스레드에서 WebSocket 연결을 유지하고 명령·이벤트·최신 상태를 교환합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Persistent WebSocket transport for a Streamlit session (no Streamlit calls).

Widgets rerun the UI script. Keeping networking in a worker lets those reruns
change ROM through the original connection without restarting measurement.
"""

# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `copy` 모듈을 불러옵니다.
import copy
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# 이 파일에서 사용할 `queue` 모듈을 불러옵니다.
import queue
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time
# 이 파일에서 사용할 `uuid` 모듈을 불러옵니다.
import uuid

# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets

# `network.client_progress`에서 `ClientProgress` 기능을 가져옵니다.
from network.client_progress import ClientProgress


# 종료 메시지에 순서·구성을 지정한 `{'SESSION_FINISHED', 'CALIBRATION_FINISHED', 'VIDEO_ANALYSIS_COMPLETED'}`을 저장합니다. 대상: `FINISH_PACKETS`.
FINISH_PACKETS = {"SESSION_FINISHED", "CALIBRATION_FINISHED", "VIDEO_ANALYSIS_COMPLETED"}


# `StreamlitSessionClient` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class StreamlitSessionClient:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, uri, config, recorder=None, ui_heartbeat_timeout=None):
        # WebSocket 접속 주소에 WebSocket 접속 주소(`uri`)을 저장합니다. 대상: `self.uri`.
        self.uri = uri
        # 현재 작업 설정에 `copy.deepcopy(config)` 호출 결과를 저장합니다. 대상: `self.config`.
        self.config = copy.deepcopy(config)
        # `self.recorder` 값에 `recorder` 값(`recorder`)을 저장합니다. 대상: `self.recorder`.
        self.recorder = recorder
        # 조건 `ui_heartbeat_timeout is not None and (not math.isfinite(ui_heartbeat_timeout) or ui_heartbeat_timeout <= 0)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if ui_heartbeat_timeout is not None and (not math.isfinite(ui_heartbeat_timeout) or ui_heartbeat_timeout <= 0):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('ui_heartbeat_timeout must be finite and positive')` 예외를 발생시킵니다.
            raise ValueError('ui_heartbeat_timeout must be finite and positive')
        # ui 화면 생존 신호 timeout에 ui 화면 생존 신호 timeout(`ui_heartbeat_timeout`)을 저장합니다. 대상: `self._ui_heartbeat_timeout`.
        self._ui_heartbeat_timeout = ui_heartbeat_timeout
        # last ui 화면 생존 신호에 시스템 시계 변경에 영향받지 않는 시각(초)(`time.monotonic()`)을 저장합니다. 대상: `self._last_ui_heartbeat`.
        self._last_ui_heartbeat = time.monotonic()
        # 현재 측정 세션 식별자에 `uuid.uuid4().hex` 값(`uuid.uuid4().hex`)을 저장합니다. 대상: `self.session_id`.
        self.session_id = uuid.uuid4().hex
        # 화면에서 통신 스레드로 보내는 명령 큐에 `queue.Queue()` 호출 결과를 저장합니다. 대상: `self._commands`.
        self._commands = queue.Queue()
        # 통신 스레드에서 화면으로 보내는 이벤트 큐에 `queue.Queue()` 호출 결과를 저장합니다. 대상: `self._events`.
        self._events = queue.Queue()
        # 작업 중단 요청 신호에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `self._stop`.
        self._stop = threading.Event()
        # 백그라운드 통신 작업의 실행 신호에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `self._running`.
        self._running = threading.Event()
        # 스레드 사이의 공유 상태를 보호하는 잠금에 스레드 공유 데이터용 잠금(`threading.Lock()`)을 저장합니다. 대상: `self._lock`.
        self._lock = threading.Lock()
        # 백그라운드 작업 스레드에 None을 저장합니다. 대상: `self._thread`.
        self._thread = None
        # `self._pose` 값에 None을 저장합니다. 대상: `self._pose`.
        self._pose = None
        # 가동 범위 기준에 None을 저장합니다. 대상: `self._thresholds`.
        self._thresholds = None
        # 서버 상태에 None을 저장합니다. 대상: `self._server_status`.
        self._server_status = None
        # 로그 info에 None을 저장합니다. 대상: `self._log_info`.
        self._log_info = None
        # 서버 상태 at에 None을 저장합니다. 대상: `self._server_status_at`.
        self._server_status_at = None
        # 상태에 'IDLE'을 저장합니다. 대상: `self._status`.
        self._status = "IDLE"
        # 메시지 횟수에 0을 저장합니다. 대상: `self._packet_count`.
        self._packet_count = 0
        # 진행률에 `ClientProgress(self.config)` 호출 결과를 저장합니다. 대상: `self._progress`.
        self._progress = ClientProgress(self.config)
        # 종료 request에 None을 저장합니다. 대상: `self._finish_request`.
        self._finish_request = None
        # 종료 deadline에 None을 저장합니다. 대상: `self._finish_deadline`.
        self._finish_deadline = None

    # 데코레이터 `property`를 바로 아래 함수·클래스에 적용합니다.
    @property
    # `running` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def running(self):
        # 현재 함수 처리를 끝내고 `self._running.is_set()` 호출 결과를 호출한 쪽으로 반환합니다.
        return self._running.is_set()

    # 데코레이터 `property`를 바로 아래 함수·클래스에 적용합니다.
    @property
    # can edit 가동 범위 기준 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def can_edit_thresholds(self):
        # 현재 함수 처리를 끝내고 계산식 `self.running and (not self._stop.is_set()) and (self._finish_request is None) and (self._server_status is None or self._server_status.get('ready', False)) and (not self.…`의 결과를 호출한 쪽으로 반환합니다.
        return (self.running and not self._stop.is_set() and self._finish_request is None
                # 앞줄에서 시작한 식·인수·자료 구조를 `and (self._server_status is None or self._server_status.get("ready", False))` 내용으로 이어서 구성합니다.
                and (self._server_status is None or self._server_status.get("ready", False))
                # 앞줄에서 시작한 식·인수·자료 구조를 `and not self.config.get("video_path")` 내용으로 이어서 구성합니다.
                and not self.config.get("video_path")
                # 앞줄에서 시작한 식·인수·자료 구조를 `and self.config.get("mode") in {"MAIN", "TEST"})` 내용으로 이어서 구성합니다.
                and self.config.get("mode") in {"MAIN", "TEST"})

    # 시작 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start(self):
        # 조건 `self._thread is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._thread is not None:
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Create a new client for each session')` 예외를 발생시킵니다.
            raise RuntimeError("Create a new client for each session")
        # `self._running.set()`을 호출하여 스레드 신호를 활성화합니다.
        self._running.set()
        # `self.touch_ui()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.touch_ui()
        # 상태에 'CONNECTING'을 저장합니다. 대상: `self._status`.
        self._status = "CONNECTING"
        # 백그라운드 작업 스레드에 백그라운드 실행 스레드(`threading.Thread(target=self._worker, daemon=True)`)을 저장합니다. 대상: `self._thread`.
        self._thread = threading.Thread(target=self._worker, daemon=True)
        # `self._thread.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._thread.start()

    # Streamlit 화면이 살아 있음을 단조 시각으로 기록합니다. 통신 스레드의 자동 갱신으로 대신하지 않습니다.
    def touch_ui(self):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Called by the Streamlit fragment, never by the networking worker."""
        # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self._lock:
            # last ui 화면 생존 신호에 시스템 시계 변경에 영향받지 않는 시각(초)(`time.monotonic()`)을 저장합니다. 대상: `self._last_ui_heartbeat`.
            self._last_ui_heartbeat = time.monotonic()

    # 중지 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def stop(self, timeout=6.0):
        # 조건 `self.running and (not self._stop.is_set())`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.running and not self._stop.is_set():
            # `self.request_finish('USER_STOP')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.request_finish("USER_STOP")
        # 조건 `self._thread is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._thread is not None:
            # `self._thread.join(timeout=timeout)`을 호출하여 백그라운드 스레드의 종료를 기다림합니다.
            self._thread.join(timeout=timeout)
            # 조건 `self._thread.is_alive()`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self._thread.is_alive():
                # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with self._lock:
                    # 상태에 'ERROR'을 저장합니다. 대상: `self._status`.
                    self._status = "ERROR"
                # `self._events.put({'type': 'CLIENT_ERROR', 'message': 'Session stop timed out; server log saving is unconfirmed'})`을 호출하여 다른 작업에서 처리할 항목을 큐에 등록합니다.
                self._events.put({"type": "CLIENT_ERROR",
                    # 구성 항목: 메시지(`'message'`)에 'Session stop timed out; server log saving is unconfirmed'을 지정합니다.
                    "message": "Session stop timed out; server log saving is unconfirmed"})
                # `self._stop.set()`을 호출하여 스레드 신호를 활성화합니다.
                self._stop.set()
                # `self._thread.join(timeout=1.0)`을 호출하여 백그라운드 스레드의 종료를 기다림합니다.
                self._thread.join(timeout=1.0)

    # request 종료 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def request_finish(self, reason="TARGET_REACHED"):
        # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self._lock:
            # 조건 `not self.running or self._finish_request is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not self.running or self._finish_request is not None:
                # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                return
            # 종료 request에 `uuid.uuid4().hex` 값(`uuid.uuid4().hex`)을 저장합니다. 대상: `self._finish_request`.
            self._finish_request = uuid.uuid4().hex
            # 종료 deadline에 계산식 `time.monotonic() + 5.0`의 결과를 저장합니다. 대상: `self._finish_deadline`.
            self._finish_deadline = time.monotonic() + 5.0
            # 상태에 'STOPPING'을 저장합니다. 대상: `self._status`.
            self._status = "STOPPING"
            # 현재 측정 세션 식별자에 `(self._log_info or {}).get('session_id')` 호출 결과를 저장합니다. 대상: `session_id`.
            session_id = (self._log_info or {}).get("session_id")
        # 수신 또는 전송할 JSON 메시지 사전에 키 'type', 'reason', 'request_id'로 구성한 사전을 저장합니다. 대상: `packet`.
        packet = {"type": "CMD_STOP_SESSION", "reason": reason,
                  # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 종료 request(`self._finish_request`)을 지정합니다.
                  "request_id": self._finish_request}
        # 조건 `self._progress.enabled`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._progress.enabled:
            # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self._lock:
                # `packet['client_result']` 값에 `self._progress.snapshot()` 호출 결과를 저장합니다. 대상: `packet['client_result']`.
                packet["client_result"] = self._progress.snapshot()
        # 조건 `session_id`을 만족하는 경우에만 아래 분기를 실행합니다.
        if session_id:
            # `packet['session_id']` 값에 현재 측정 세션 식별자(`session_id`)을 저장합니다. 대상: `packet['session_id']`.
            packet["session_id"] = session_id
        # `self._commands.put(json.dumps(packet))`을 호출하여 다른 작업에서 처리할 항목을 큐에 등록합니다.
        self._commands.put(json.dumps(packet))

    # 전송 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def send(self, packet):
        # 조건 `not self.running or self._stop.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.running or self._stop.is_set():
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('No active connection')` 예외를 발생시킵니다.
            raise RuntimeError("No active connection")
        # 화면의 후속 편집이 대기 중인 명령을 바꾸지 않도록 복사·검증 후 큐에 넣습니다.
        # `encoded` 값에 전송·저장용 JSON 문자열(`json.dumps(packet, allow_nan=False)`)을 저장합니다. 대상: `encoded`.
        encoded = json.dumps(packet, allow_nan=False)
        # `self._commands.put(encoded)`을 호출하여 다른 작업에서 처리할 항목을 큐에 등록합니다.
        self._commands.put(encoded)

    # 가동 범위 기준 명령 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def threshold_command(self, thresholds=None):
        # 수신 또는 전송할 JSON 메시지 사전에 키 'type', 'exercise_name', 'request_id'로 구성한 사전을 저장합니다. 대상: `packet`.
        packet = {
            # 구성 항목: `type` 값(`'type'`)에 `thresholds is None`이면 `'CMD_GET_COUNT_THRESHOLD'`, 아니면 `'CMD_UPDATE_COUNT_THRESHOLD'`을 지정합니다.
            "type": "CMD_GET_COUNT_THRESHOLD" if thresholds is None else "CMD_UPDATE_COUNT_THRESHOLD",
            # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 계산식 `self.config['exercise_name']`의 결과를 지정합니다.
            "exercise_name": self.config["exercise_name"],
            # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 `uuid.uuid4().hex` 값(`uuid.uuid4().hex`)을 지정합니다.
            "request_id": uuid.uuid4().hex,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
        # 조건 `thresholds is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if thresholds is not None:
            # 조건 `not self.can_edit_thresholds`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not self.can_edit_thresholds:
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('ROM updates require a live MAIN/TEST session')` 예외를 발생시킵니다.
                raise RuntimeError("ROM updates require a live MAIN/TEST session")
            # `packet['thresholds']` 값에 좌우 가동 범위 시작값·목표값(`thresholds`)을 저장합니다. 대상: `packet['thresholds']`.
            packet["thresholds"] = thresholds
        # `self.send(packet)`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다.
        self.send(packet)
        # 현재 함수 처리를 끝내고 계산식 `packet['request_id']`의 결과를 호출한 쪽으로 반환합니다.
        return packet["request_id"]

    # `snapshot` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def snapshot(self):
        # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self._lock:
            # 함수 또는 처리 단계의 결과에 `copy.deepcopy({'pose': self._pose, 'threshold_status': self._thresholds, 'server_status': self._server_status, 'log_info': self._log_info, 'status': self._status, 'packe…` 호출 결과를 저장합니다. 대상: `result`.
            result = copy.deepcopy({
                # 구성 항목: `pose` 값(`'pose'`)에 `self._pose` 값(`self._pose`)을 지정합니다.
                "pose": self._pose,
                # 구성 항목: 가동 범위 기준 상태(`'threshold_status'`)에 가동 범위 기준(`self._thresholds`)을 지정합니다.
                "threshold_status": self._thresholds,
                # 구성 항목: 서버 상태(`'server_status'`)에 서버 상태(`self._server_status`)을 지정합니다.
                "server_status": self._server_status,
                # 구성 항목: 로그 info(`'log_info'`)에 로그 info(`self._log_info`)을 지정합니다.
                "log_info": self._log_info,
                # 구성 항목: 상태(`'status'`)에 상태(`self._status`)을 지정합니다.
                "status": self._status,
                # 구성 항목: 메시지 횟수(`'packet_count'`)에 메시지 횟수(`self._packet_count`)을 지정합니다.
                "packet_count": self._packet_count,
                # 구성 항목: 진행률(`'progress'`)에 `self._progress.snapshot()` 호출 결과를 지정합니다.
                "progress": self._progress.snapshot(),
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            })
            # 조건 `result['server_status'] and result['server_status'].get('model_state') == 'LOADING'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if result["server_status"] and result["server_status"].get("model_state") == "LOADING":
                # `result['server_status']['elapsed_seconds']` 값에 `time.monotonic() - self._server_status_at`을 더해 누적합니다. 대상: `result['server_status']['elapsed_seconds']`.
                result["server_status"]["elapsed_seconds"] += time.monotonic() - self._server_status_at
            # 조건 `result['pose'] and self.config.get('mode') in {'MAIN', 'TEST'}`을 만족하는 경우에만 아래 분기를 실행합니다.
            if result["pose"] and self.config.get("mode") in {"MAIN", "TEST"}:
                # `('left', 'right')`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
                for side in ("left", "right"):
                    # `result['pose'].setdefault(side, {})['total_score']` 값에 계산식 `result['progress']['scores'][side]`의 결과를 저장합니다. 대상: `result['pose'].setdefault(side, {})['total_score']`.
                    result["pose"].setdefault(side, {})["total_score"] = result["progress"]["scores"][side]
            # 현재 함수 처리를 끝내고 함수 또는 처리 단계의 결과(`result`)을 호출한 쪽으로 반환합니다.
            return result

    # drain 이벤트 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def drain_events(self):
        # 이벤트에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `events`.
        events = []
        # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
        while True:
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `events.append(self._events.get_nowait())`을 호출하여 항목 하나를 끝에 추가합니다.
                events.append(self._events.get_nowait())
            # `queue.Empty` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except queue.Empty:
                # 현재 함수 처리를 끝내고 이벤트(`events`)을 호출한 쪽으로 반환합니다.
                return events

    # 작업 스레드 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _worker(self):
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `asyncio.run(self._run())`을 호출하여 해당 단계의 작업을 수행합니다.
            asyncio.run(self._run())
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception as exc:
            # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self._lock:
                # 상태에 'ERROR'을 저장합니다. 대상: `self._status`.
                self._status = "ERROR"
            # `self._events.put({'type': 'CLIENT_ERROR', 'message': str(exc)})`을 호출하여 다른 작업에서 처리할 항목을 큐에 등록합니다.
            self._events.put({"type": "CLIENT_ERROR", "message": str(exc)})
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # 조건 `self.recorder`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.recorder:
                # `self.recorder.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
                self.recorder.stop()
            # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self._lock:
                # 조건 `self._status not in {'FINISHED', 'ERROR'}`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self._status not in {"FINISHED", "ERROR"}:
                    # 상태에 'STOPPED'을 저장합니다. 대상: `self._status`.
                    self._status = "STOPPED"
            # `self._running.clear()`을 호출하여 누적 내용을 비워 초기화합니다.
            self._running.clear()

    # 실행 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _run(self):
        # `websockets.connect(self.uri, open_timeout=5, close_timeout=1)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri, open_timeout=5, close_timeout=1) as ws:
            # 조건 `self._stop.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self._stop.is_set():
                # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                return
            # 조건 `self.recorder`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.recorder:
                # `self.recorder.start()`을 호출하여 해당 단계의 작업을 수행합니다.
                self.recorder.start()
            # 목표와 점수 규칙은 클라이언트가 관리하므로 서버 시작 설정에는 보내지 않습니다.
            # 시작 설정에 반복·조건식 `{key: value for (key, value) in self.config.items() if key not in {'target_score', 'target_reps', 'finish_delay_seconds', 'quality_points'}}`으로 구성한 값을 저장합니다. 대상: `start_config`.
            start_config = {key: value for key, value in self.config.items()
                            # 앞줄에서 시작한 식·인수·자료 구조를 `if key not in {"target_score", "target_reps", "finish_delay_seconds", "quality_points"}}` 내용으로 이어서 구성합니다.
                            if key not in {"target_score", "target_reps", "finish_delay_seconds", "quality_points"}}
            # 조건 `self.config.get('mode') == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self.config.get("mode") == "CALIBRATION":
                # 시작 config['target reps']에 3을 저장합니다. 대상: `start_config['target_reps']`.
                start_config["target_reps"] = 3
            # `ws.send(json.dumps({'type': 'CMD_SET_SESSION', **start_config}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await ws.send(json.dumps({"type": "CMD_SET_SESSION", **start_config}))
            # 조건 `not self.config.get('video_path')`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not self.config.get("video_path"):
                # `ws.send(json.dumps({'type': 'CMD_GET_COUNT_THRESHOLD', 'exercise_name': self.config['exercise_name'], 'request_id': uuid.uuid4().hex}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                await ws.send(json.dumps({
                    # 구성 항목: `type` 값(`'type'`)에 'CMD_GET_COUNT_THRESHOLD'을 지정합니다.
                    "type": "CMD_GET_COUNT_THRESHOLD",
                    # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 계산식 `self.config['exercise_name']`의 결과를 지정합니다.
                    "exercise_name": self.config["exercise_name"],
                    # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 `uuid.uuid4().hex` 값(`uuid.uuid4().hex`)을 지정합니다.
                    "request_id": uuid.uuid4().hex,
                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                }))
            # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self._lock:
                # 상태에 'RUNNING'을 저장합니다. 대상: `self._status`.
                self._status = "RUNNING"

            # 조건 `not self._stop.is_set()`이 유지되는 동안 아래 처리를 반복합니다.
            while not self._stop.is_set():
                # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with self._lock:
                    # `ui_expired` 값에 계산식 `self._ui_heartbeat_timeout is not None and time.monotonic() - self._last_ui_heartbeat >= self._ui_heartbeat_timeout`의 결과를 저장합니다. 대상: `ui_expired`.
                    ui_expired = (self._ui_heartbeat_timeout is not None and
                                  # 앞줄에서 시작한 식·인수·자료 구조를 `time.monotonic() - self._last_ui_heartbeat >= self._ui_heartbeat_timeout)` 내용으로 이어서 구성합니다.
                                  time.monotonic() - self._last_ui_heartbeat >= self._ui_heartbeat_timeout)
                # 조건 `ui_expired and self._finish_request is None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if ui_expired and self._finish_request is None:
                    # `self.request_finish('UI_DISCONNECTED')`을 호출하여 해당 단계의 작업을 수행합니다.
                    self.request_finish('UI_DISCONNECTED')
                # 조건 `self._progress.finish_due(time.monotonic()) and self._finish_request is None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self._progress.finish_due(time.monotonic()) and self._finish_request is None:
                    # `self.request_finish('TARGET_REACHED')`을 호출하여 해당 단계의 작업을 수행합니다.
                    self.request_finish("TARGET_REACHED")
                # 조건 `self._finish_deadline is not None and time.monotonic() >= self._finish_deadline`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self._finish_deadline is not None and time.monotonic() >= self._finish_deadline:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `TimeoutError('Server did not acknowledge CMD_STOP_SESSION; log saving is unconfirmed')` 예외를 발생시킵니다.
                    raise TimeoutError("Server did not acknowledge CMD_STOP_SESSION; log saving is unconfirmed")
                # `range(16)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
                for _ in range(16):
                    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                    try:
                        # 명령에 `self._commands.get_nowait()` 호출 결과를 저장합니다. 대상: `command`.
                        command = self._commands.get_nowait()
                    # `queue.Empty` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                    except queue.Empty:
                        # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                        break
                    # `ws.send(command)`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await ws.send(command)
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # 메시지에 계산식 `await asyncio.wait_for(ws.recv(), timeout=0.05)`의 결과를 저장합니다. 대상: `message`.
                    message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                # `asyncio.TimeoutError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except asyncio.TimeoutError:
                    # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                    continue
                # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(message)`)을 저장합니다. 대상: `packet`.
                packet = json.loads(message)
                # 메시지 type에 `packet.get('type')` 호출 결과를 저장합니다. 대상: `packet_type`.
                packet_type = packet.get("type")
                # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with self._lock:
                    # 메시지 횟수에 `1`을 더해 누적합니다. 대상: `self._packet_count`.
                    self._packet_count += 1
                    # 조건 `packet_type == 'REP_COMPLETED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet_type == "REP_COMPLETED":
                        # `self._progress.accept(packet, time.monotonic())`을 호출하여 해당 단계의 작업을 수행합니다.
                        self._progress.accept(packet, time.monotonic())
                    # 조건 `packet_type == 'POSE_UPDATE'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet_type == "POSE_UPDATE":
                        # `self._pose` 값에 수신 또는 전송할 JSON 메시지 사전(`packet`)을 저장합니다. 대상: `self._pose`.
                        self._pose = packet
                        # 상태에 'RUNNING'을 저장합니다. 대상: `self._status`.
                        self._status = "RUNNING"
                    # 조건 `packet_type == 'SERVER_STATUS'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif packet_type == "SERVER_STATUS":
                        # 서버 상태에 수신 또는 전송할 JSON 메시지 사전(`packet`)을 저장합니다. 대상: `self._server_status`.
                        self._server_status = packet
                        # 서버 상태 at에 시스템 시계 변경에 영향받지 않는 시각(초)(`time.monotonic()`)을 저장합니다. 대상: `self._server_status_at`.
                        self._server_status_at = time.monotonic()
                        # 조건 `packet.get('model_state') == 'LOADING'`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if packet.get("model_state") == "LOADING":
                            # 상태에 'PREPARING'을 저장합니다. 대상: `self._status`.
                            self._status = "PREPARING"
                    # 조건 `packet_type == 'SESSION_PENDING'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif packet_type == "SESSION_PENDING":
                        # 상태에 'PREPARING'을 저장합니다. 대상: `self._status`.
                        self._status = "PREPARING"
                    # 조건 `packet_type == 'SESSION_PREPARING'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif packet_type == "SESSION_PREPARING":
                        # 상태에 'PREPARING'을 저장합니다. 대상: `self._status`.
                        self._status = "PREPARING"
                    # 조건 `packet_type == 'SESSION_MEASUREMENT_STARTED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif packet_type == "SESSION_MEASUREMENT_STARTED":
                        # 상태에 'RUNNING'을 저장합니다. 대상: `self._status`.
                        self._status = "RUNNING"
                    # 조건 `packet_type == 'VIDEO_STATUS'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif packet_type == "VIDEO_STATUS":
                        # 상태에 'RUNNING'을 저장합니다. 대상: `self._status`.
                        self._status = "RUNNING"
                    # 조건 `packet_type == 'SESSION_LOG_STARTED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif packet_type == "SESSION_LOG_STARTED":
                        # 로그 info에 수신 또는 전송할 JSON 메시지 사전(`packet`)을 저장합니다. 대상: `self._log_info`.
                        self._log_info = packet
                        # `self._progress.bind_session(packet.get('session_id'))`을 호출하여 해당 단계의 작업을 수행합니다.
                        self._progress.bind_session(packet.get("session_id"))
                    # 조건 `packet_type in FINISH_PACKETS and (packet.get('summary') or {}).get('directory')`을 만족하는 경우에만 아래 분기를 실행합니다.
                    elif packet_type in FINISH_PACKETS and (packet.get("summary") or {}).get("directory"):
                        # 로그 info에 계산식 `packet['summary']`의 결과를 저장합니다. 대상: `self._log_info`.
                        self._log_info = packet["summary"]
                    # 조건 `packet_type in {'POSE_UPDATE', 'COUNT_THRESHOLD_STATUS', 'COUNT_THRESHOLD_UPDATED'}`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet_type in {"POSE_UPDATE", "COUNT_THRESHOLD_STATUS", "COUNT_THRESHOLD_UPDATED"}:
                        # 조건 `'thresholds' in packet`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if "thresholds" in packet:
                            # 가동 범위 기준에 반복·조건식 `{key: packet.get(key) for key in ('exercise_name', 'metric_type', 'motion_direction', 'threshold_version', 'thresholds')}`으로 구성한 값을 저장합니다. 대상: `self._thresholds`.
                            self._thresholds = {
                                # 앞줄에서 시작한 식·인수·자료 구조를 `key: packet.get(key) for key in (` 내용으로 이어서 구성합니다.
                                key: packet.get(key) for key in (
                                    # 앞줄에서 시작한 식·인수·자료 구조를 `"exercise_name", "metric_type", "motion_direction",` 내용으로 이어서 구성합니다.
                                    "exercise_name", "metric_type", "motion_direction",
                                    # 앞줄에서 시작한 식·인수·자료 구조를 `"threshold_version", "thresholds",` 내용으로 이어서 구성합니다.
                                    "threshold_version", "thresholds",
                                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                                )
                            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                            }
                # 조건 `self._finish_request is not None and packet_type not in FINISH_PACKETS`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self._finish_request is not None and packet_type not in FINISH_PACKETS:
                    # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                    with self._lock:
                        # 상태에 'STOPPING'을 저장합니다. 대상: `self._status`.
                        self._status = "STOPPING"
                # 조건 `packet_type == 'POSE_UPDATE'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet_type == "POSE_UPDATE":
                    # 조건 `self.recorder`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if self.recorder:
                        # `self.recorder.record_pose(packet)`을 호출하여 해당 단계의 작업을 수행합니다.
                        self.recorder.record_pose(packet)
                # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                else:
                    # `self._events.put(packet)`을 호출하여 다른 작업에서 처리할 항목을 큐에 등록합니다.
                    self._events.put(packet)
                # 조건 `packet_type in FINISH_PACKETS`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet_type in FINISH_PACKETS:
                    # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                    with self._lock:
                        # 상태에 'FINISHED'을 저장합니다. 대상: `self._status`.
                        self._status = "FINISHED"
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break
                # 조건 `packet_type == 'ERROR' and (packet.get('fatal') or not packet.get('command'))`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet_type == "ERROR" and (packet.get("fatal") or not packet.get("command")):
                    # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                    with self._lock:
                        # 상태에 'ERROR'을 저장합니다. 대상: `self._status`.
                        self._status = "ERROR"
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break
