# 한국어 코드 해설: 소켓 수신을 먼저 시작할 수 있도록 AI 모델을 작업 스레드에서 한 번 로드하고 준비 상태를 공유합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Load AI dependencies/weights once without delaying the listening socket."""

# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time
# 이 파일에서 사용할 `traceback` 모듈을 불러옵니다.
import traceback


# `PHASE_MESSAGES` 값에 키 'NOT_STARTED', 'IMPORTING', 'DEVICE_SETUP', 'LOADING_YOLO', 'LOADING_POSE', 'READY', 'FAILED'로 구성한 사전을 저장합니다. 대상: `PHASE_MESSAGES`.
PHASE_MESSAGES = {
    # 구성 항목: 하지 않음 started(`'NOT_STARTED'`)에 '모델 준비 대기 중입니다.'을 지정합니다.
    'NOT_STARTED': '모델 준비 대기 중입니다.',
    # 구성 항목: `IMPORTING` 값(`'IMPORTING'`)에 'AI 라이브러리를 불러오는 중입니다.'을 지정합니다.
    'IMPORTING': 'AI 라이브러리를 불러오는 중입니다.',
    # 구성 항목: device 준비(`'DEVICE_SETUP'`)에 '연산 장치를 확인하는 중입니다.'을 지정합니다.
    'DEVICE_SETUP': '연산 장치를 확인하는 중입니다.',
    # 구성 항목: `LOADING_YOLO` 값(`'LOADING_YOLO'`)에 '사람 검출 모델을 불러오는 중입니다.'을 지정합니다.
    'LOADING_YOLO': '사람 검출 모델을 불러오는 중입니다.',
    # 구성 항목: `LOADING_POSE` 값(`'LOADING_POSE'`)에 '관절 추출 모델을 불러오는 중입니다.'을 지정합니다.
    'LOADING_POSE': '관절 추출 모델을 불러오는 중입니다.',
    # 구성 항목: 준비 상태(`'READY'`)에 '모델 준비가 완료되었습니다.'을 지정합니다.
    'READY': '모델 준비가 완료되었습니다.',
    # 구성 항목: `FAILED` 값(`'FAILED'`)에 '모델 준비에 실패했습니다.'을 지정합니다.
    'FAILED': '모델 준비에 실패했습니다.',
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
}


# `BackgroundModelLoader` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class BackgroundModelLoader:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, factory):
        # `self.factory` 값에 `factory` 값(`factory`)을 저장합니다. 대상: `self.factory`.
        self.factory = factory
        # 생성 또는 검증할 분석 엔진에 None을 저장합니다. 대상: `self.engine`.
        self.engine = None
        # `self.task` 값에 None을 저장합니다. 대상: `self.task`.
        self.task = None
        # 스레드 사이의 공유 상태를 보호하는 잠금에 스레드 공유 데이터용 잠금(`threading.Lock()`)을 저장합니다. 대상: `self._lock`.
        self._lock = threading.Lock()
        # 단계에 'NOT_STARTED'을 저장합니다. 대상: `self._state`.
        self._state = 'NOT_STARTED'
        # `self._phase` 값에 'NOT_STARTED'을 저장합니다. 대상: `self._phase`.
        self._phase = 'NOT_STARTED'
        # 오류에 None을 저장합니다. 대상: `self._error`.
        self._error = None
        # `self._started` 값에 None을 저장합니다. 대상: `self._started`.
        self._started = None
        # `self._phase_started` 값에 None을 저장합니다. 대상: `self._phase_started`.
        self._phase_started = None
        # `self._elapsed` 값에 0.0을 저장합니다. 대상: `self._elapsed`.
        self._elapsed = 0.0
        # `self._timings` 값에 빈 사전을 저장합니다. 대상: `self._timings`.
        self._timings = {}

    # 시작 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start(self):
        # 조건 `self.task is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.task is None:
            # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self._lock:
                # `self._started` 값에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `self._started`.
                self._started = time.perf_counter()
                # 단계에 'LOADING'을 저장합니다. 대상: `self._state`.
                self._state = 'LOADING'
            # `self.report_phase('IMPORTING')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.report_phase('IMPORTING')
            # `self.task` 값에 별도로 진행하도록 등록한 비동기 작업(`asyncio.create_task(self._load())`)을 저장합니다. 대상: `self.task`.
            self.task = asyncio.create_task(self._load())
        # 현재 함수 처리를 끝내고 `self.task` 값(`self.task`)을 호출한 쪽으로 반환합니다.
        return self.task

    # `report_phase` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def report_phase(self, phase):
        # 이번 처리에 사용할 현재 시각(초)에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `now`.
        now = time.perf_counter()
        # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self._lock:
            # 조건 `phase == self._phase`을 만족하는 경우에만 아래 분기를 실행합니다.
            if phase == self._phase:
                # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
                return
            # 조건 `self._phase_started is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self._phase_started is not None:
                # `self._timings[self._phase]` 값에 계산식 `now - self._phase_started`의 결과를 저장합니다. 대상: `self._timings[self._phase]`.
                self._timings[self._phase] = now - self._phase_started
            # `self._phase` 값에 `phase` 값(`phase`)을 저장합니다. 대상: `self._phase`.
            self._phase = phase
            # `self._phase_started` 값에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self._phase_started`.
            self._phase_started = now
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f'[STARTUP] {phase}: {PHASE_MESSAGES.get(phase, phase)}', flush=True)

    # 불러오기 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _load(self):
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 생성 또는 검증할 분석 엔진에 계산식 `await asyncio.to_thread(self.factory, self.report_phase)`의 결과를 저장합니다. 대상: `engine`.
            engine = await asyncio.to_thread(self.factory, self.report_phase)
            # 조건 `engine is None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if engine is None:
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Model factory returned no engine')` 예외를 발생시킵니다.
                raise RuntimeError('Model factory returned no engine')
            # `self.report_phase('READY')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.report_phase('READY')
            # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self._lock:
                # 생성 또는 검증할 분석 엔진에 생성 또는 검증할 분석 엔진(`engine`)을 저장합니다. 대상: `self.engine`.
                self.engine = engine
                # 단계에 'READY'을 저장합니다. 대상: `self._state`.
                self._state = 'READY'
                # `self._elapsed` 값에 계산식 `time.perf_counter() - self._started`의 결과를 저장합니다. 대상: `self._elapsed`.
                self._elapsed = time.perf_counter() - self._started
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f'[STARTUP] Models ready in {self._elapsed:.3f}s', flush=True)
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception as exc:
            # `self.report_phase('FAILED')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.report_phase('FAILED')
            # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self._lock:
                # 단계에 'FAILED'을 저장합니다. 대상: `self._state`.
                self._state = 'FAILED'
                # 오류에 `str(exc)` 호출 결과를 저장합니다. 대상: `self._error`.
                self._error = str(exc)
                # `self._elapsed` 값에 계산식 `time.perf_counter() - self._started`의 결과를 저장합니다. 대상: `self._elapsed`.
                self._elapsed = time.perf_counter() - self._started
            # `traceback.print_exc()`을 호출하여 현재 예외의 상세 호출 경로를 출력합니다.
            traceback.print_exc()

    # `snapshot` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def snapshot(self):
        # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self._lock:
            # `timings` 값에 `dict(self._timings)` 호출 결과를 저장합니다. 대상: `timings`.
            timings = dict(self._timings)
            # `elapsed` 값에 `self._elapsed` 값(`self._elapsed`)을 저장합니다. 대상: `elapsed`.
            elapsed = self._elapsed
            # 조건 `self._state == 'LOADING'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self._state == 'LOADING':
                # 이번 처리에 사용할 현재 시각(초)에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `now`.
                now = time.perf_counter()
                # `elapsed` 값에 계산식 `now - self._started`의 결과를 저장합니다. 대상: `elapsed`.
                elapsed = now - self._started
                # `timings[self._phase]` 값에 계산식 `now - self._phase_started`의 결과를 저장합니다. 대상: `timings[self._phase]`.
                timings[self._phase] = now - self._phase_started
            # 현재 함수 처리를 끝내고 키 'type', 'model_state', 'phase', 'ready', 'message', 'model_error', 'elapsed_seconds', 'phase_seconds'로 구성한 사전을 호출한 쪽으로 반환합니다.
            return {
                # 구성 항목: `type` 값(`'type'`)에 'SERVER_STATUS'; 모델 단계(`'model_state'`)에 단계(`self._state`)을 지정합니다.
                'type': 'SERVER_STATUS', 'model_state': self._state,
                # 구성 항목: `phase` 값(`'phase'`)에 `self._phase` 값(`self._phase`); 준비 상태(`'ready'`)에 계산식 `self._state == 'READY'`의 결과를 지정합니다.
                'phase': self._phase, 'ready': self._state == 'READY',
                # 구성 항목: 메시지(`'message'`)에 `PHASE_MESSAGES.get(self._phase, self._phase)` 호출 결과를 지정합니다.
                'message': PHASE_MESSAGES.get(self._phase, self._phase),
                # 구성 항목: 모델 오류(`'model_error'`)에 오류(`self._error`); elapsed 초(`'elapsed_seconds'`)에 `round(elapsed, 3)` 호출 결과를 지정합니다.
                'model_error': self._error, 'elapsed_seconds': round(elapsed, 3),
                # 구성 항목: phase 초(`'phase_seconds'`)에 반복·조건식 `{phase: round(seconds, 3) for (phase, seconds) in timings.items()}`으로 구성한 값을 지정합니다.
                'phase_seconds': {phase: round(seconds, 3) for phase, seconds in timings.items()},
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            }
