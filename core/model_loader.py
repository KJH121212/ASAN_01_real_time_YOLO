"""Load AI dependencies/weights once without delaying the listening socket."""

import asyncio
import threading
import time
import traceback


PHASE_MESSAGES = {
    'NOT_STARTED': '모델 준비 대기 중입니다.',
    'IMPORTING': 'AI 라이브러리를 불러오는 중입니다.',
    'DEVICE_SETUP': '연산 장치를 확인하는 중입니다.',
    'LOADING_YOLO': '사람 검출 모델을 불러오는 중입니다.',
    'LOADING_POSE': '관절 추출 모델을 불러오는 중입니다.',
    'READY': '모델 준비가 완료되었습니다.',
    'FAILED': '모델 준비에 실패했습니다.',
}


class BackgroundModelLoader:
    def __init__(self, factory):
        self.factory = factory
        self.engine = None
        self.task = None
        self._lock = threading.Lock()
        self._state = 'NOT_STARTED'
        self._phase = 'NOT_STARTED'
        self._error = None
        self._started = None
        self._phase_started = None
        self._elapsed = 0.0
        self._timings = {}

    def start(self):
        if self.task is None:
            with self._lock:
                self._started = time.perf_counter()
                self._state = 'LOADING'
            self.report_phase('IMPORTING')
            self.task = asyncio.create_task(self._load())
        return self.task

    def report_phase(self, phase):
        now = time.perf_counter()
        with self._lock:
            if phase == self._phase:
                return
            if self._phase_started is not None:
                self._timings[self._phase] = now - self._phase_started
            self._phase = phase
            self._phase_started = now
        print(f'[STARTUP] {phase}: {PHASE_MESSAGES.get(phase, phase)}', flush=True)

    async def _load(self):
        try:
            engine = await asyncio.to_thread(self.factory, self.report_phase)
            if engine is None:
                raise RuntimeError('Model factory returned no engine')
            self.report_phase('READY')
            with self._lock:
                self.engine = engine
                self._state = 'READY'
                self._elapsed = time.perf_counter() - self._started
            print(f'[STARTUP] Models ready in {self._elapsed:.3f}s', flush=True)
        except Exception as exc:
            self.report_phase('FAILED')
            with self._lock:
                self._state = 'FAILED'
                self._error = str(exc)
                self._elapsed = time.perf_counter() - self._started
            traceback.print_exc()

    def snapshot(self):
        with self._lock:
            timings = dict(self._timings)
            elapsed = self._elapsed
            if self._state == 'LOADING':
                now = time.perf_counter()
                elapsed = now - self._started
                timings[self._phase] = now - self._phase_started
            return {
                'type': 'SERVER_STATUS', 'model_state': self._state,
                'phase': self._phase, 'ready': self._state == 'READY',
                'message': PHASE_MESSAGES.get(self._phase, self._phase),
                'model_error': self._error, 'elapsed_seconds': round(elapsed, 3),
                'phase_seconds': {phase: round(seconds, 3) for phase, seconds in timings.items()},
            }
