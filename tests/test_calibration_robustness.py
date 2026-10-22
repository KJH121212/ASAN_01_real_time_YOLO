# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# `unittest.mock`에서 `Mock` 기능을 가져옵니다.
from unittest.mock import Mock

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `CalibrationCycleDetector`, `SessionController` 기능을 가져옵니다.
from core.session_controller import CalibrationCycleDetector, SessionController


# `Signal` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class Signal:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, detector, fps=10):
        # `(self.detector, self.now, self.fps)` 값에 순서·구성을 지정한 `(detector, 0.0, fps)`을 저장합니다. 대상: `(self.detector, self.now, self.fps)`.
        self.detector, self.now, self.fps = detector, 0., fps

    # `hold` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def hold(self, value, seconds):
        # `range(math.ceil(seconds * self.fps))`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(math.ceil(seconds * self.fps)):
            # `self.detector.update(value, self.now)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.detector.update(value, self.now)
            # 이번 처리에 사용할 현재 시각(초)에 `1 / self.fps`을 더해 누적합니다. 대상: `self.now`.
            self.now += 1 / self.fps


# `CalibrationRobustnessTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class CalibrationRobustnessTests(unittest.TestCase):
    # 회귀 테스트: `test_calibrates_at_low_high_and_variable_fps_in_both_directions` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_calibrates_at_low_high_and_variable_fps_in_both_directions(self):
        # `(2, 5, 10, 15, 30)`에서 항목을 차례로 꺼내 `fps`에 넣고 아래 처리를 반복합니다.
        for fps in (2, 5, 10, 15, 30):
            # `(-1, 1)`에서 항목을 차례로 꺼내 `sign`에 넣고 아래 처리를 반복합니다.
            for sign in (-1, 1):
                # `self.subTest(fps=fps, sign=sign)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with self.subTest(fps=fps, sign=sign):
                    # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector(direction='AUTO')` 호출 결과를 저장합니다. 대상: `detector`.
                    detector = CalibrationCycleDetector(direction='AUTO')
                    # `signal` 값에 `Signal(detector, fps)` 호출 결과를 저장합니다. 대상: `signal`.
                    signal = Signal(detector, fps)
                    # `signal.hold(0.0, max(1.0, 3 / fps))`을 호출하여 해당 단계의 작업을 수행합니다.
                    signal.hold(0., max(1., 3 / fps))
                    # 검증문 `self.assertTrue(detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertTrue(detector.ready)
                    # `range(3)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
                    for _ in range(3):
                        # `signal.hold(sign * 0.2, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
                        signal.hold(sign * .2, .6)
                        # `signal.hold(sign * 0.04, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
                        signal.hold(sign * .04, .6)
                    # 검증문 `self.assertTrue(detector.consistent)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertTrue(detector.consistent)
                    # 검증문 `self.assertEqual(len(detector.peaks), 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertEqual(len(detector.peaks), 3)
                    # 검증문 `self.assertAlmostEqual(detector.thresholds()['target_val'], sign * 0.17)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertAlmostEqual(detector.thresholds()['target_val'], sign * .17)
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector()
        # 이번 처리에 사용할 현재 시각(초)에 0.0을 저장합니다. 대상: `now`.
        now = 0.
        # `range(15)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(15):
            # `detector.update(0.0, now)`을 호출하여 해당 단계의 작업을 수행합니다.
            detector.update(0., now)
            # 이번 처리에 사용할 현재 시각(초)에 `(0.08, 0.15, 0.2)[_ % 3]`을 더해 누적합니다. 대상: `now`.
            now += (.08, .15, .2)[_ % 3]
        # 검증문 `self.assertTrue(detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(detector.ready)
        # `range(3)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(3):
            # `(0.2,) * 6 + (0.04,) * 6`에서 항목을 차례로 꺼내 `value`에 넣고 아래 처리를 반복합니다.
            for value in (.2,) * 6 + (.04,) * 6:
                # `detector.update(value, now)`을 호출하여 해당 단계의 작업을 수행합니다.
                detector.update(value, now)
                # 이번 처리에 사용할 현재 시각(초)에 `(0.08, 0.15, 0.2)[_ % 3]`을 더해 누적합니다. 대상: `now`.
                now += (.08, .15, .2)[_ % 3]
        # 검증문 `self.assertTrue(detector.consistent)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(detector.consistent)

    # 회귀 테스트: `test_baseline_requires_elapsed_rest_and_multiple_samples` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_baseline_requires_elapsed_rest_and_multiple_samples(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector()
        # `detector.update(0, 0)`을 호출하여 해당 단계의 작업을 수행합니다.
        detector.update(0, 0)
        # `detector.update(0, 0.5)`을 호출하여 해당 단계의 작업을 수행합니다.
        detector.update(0, .5)
        # 검증문 `self.assertFalse(detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(detector.ready)
        # `detector.update(0, 1.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        detector.update(0, 1.)
        # 검증문 `self.assertTrue(detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(detector.ready)
        # `moving` 값에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `moving`.
        moving = CalibrationCycleDetector()
        # `range(100)`에서 항목을 차례로 꺼내 `index`에 넣고 아래 처리를 반복합니다.
        for index in range(100):
            # `moving.update(index * 0.1, index / 10)`을 호출하여 해당 단계의 작업을 수행합니다.
            moving.update(index * .1, index / 10)
        # 검증문 `self.assertFalse(moving.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(moving.ready)

    # 회귀 테스트: `test_return_tolerance_does_not_count_rest_plateau_again` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_return_tolerance_does_not_count_rest_plateau_again(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector(metric_type='ANGLE', min_rise=8.0, baseline_range=4.0)` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector(metric_type='ANGLE', min_rise=8., baseline_range=4.)
        # `signal` 값에 `Signal(detector)` 호출 결과를 저장합니다. 대상: `signal`.
        signal = Signal(detector)
        # `signal.hold(90, 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(90, 1)
        # `range(1, 4)`에서 항목을 차례로 꺼내 `count`에 넣고 아래 처리를 반복합니다.
        for count in range(1, 4):
            # `signal.hold(120, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
            signal.hold(120, .6)
            # `signal.hold(96, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
            signal.hold(96, .6)
            # 검증문 `self.assertEqual(len(detector.peaks), count)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(len(detector.peaks), count)
            # `signal.hold(96, 2)`을 호출하여 해당 단계의 작업을 수행합니다.
            signal.hold(96, 2)
            # 검증문 `self.assertEqual(len(detector.peaks), count)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(len(detector.peaks), count)
        # 검증문 `self.assertTrue(detector.consistent)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(detector.consistent)

    # 회귀 테스트: `test_short_loss_preserves_completed_cycles_but_long_loss_restarts` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_short_loss_preserves_completed_cycles_but_long_loss_restarts(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector()
        # `signal` 값에 `Signal(detector)` 호출 결과를 저장합니다. 대상: `signal`.
        signal = Signal(detector)
        # `signal.hold(0, 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, 1)
        # `signal.hold(0.2, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(.2, .6)
        # `signal.hold(0, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, .6)
        # 검증문 `self.assertEqual(len(detector.peaks), 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(detector.peaks), 1)
        # `signal.hold(None, 0.3)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(None, .3)
        # 검증문 `self.assertTrue(detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(detector.ready)
        # 검증문 `self.assertEqual(len(detector.peaks), 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(detector.peaks), 1)
        # `signal.hold(0, 0.3)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, .3)
        # `signal.hold(None, 2.3)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(None, 2.3)
        # 검증문 `self.assertFalse(detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(detector.ready)
        # 검증문 `self.assertEqual(len(detector.peaks), 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(detector.peaks), 0)

    # 회귀 테스트: `test_spikes_and_interrupted_holds_are_not_counted` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_spikes_and_interrupted_holds_are_not_counted(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector()
        # `signal` 값에 `Signal(detector)` 호출 결과를 저장합니다. 대상: `signal`.
        signal = Signal(detector)
        # `signal.hold(0, 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, 1)
        # `signal.hold(0.2, 0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(.2, .1)
        # `signal.hold(0, 0.5)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, .5)
        # 검증문 `self.assertEqual(detector.peaks, [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(detector.peaks, [])
        # `signal.hold(0.2, 0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(.2, .1)
        # `signal.hold(None, 0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(None, .1)
        # `signal.hold(0, 0.5)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, .5)
        # 검증문 `self.assertEqual(detector.peaks, [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(detector.peaks, [])

    # 회귀 테스트: `test_inconsistent_cycles_require_more_consistent_motion` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_inconsistent_cycles_require_more_consistent_motion(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector()
        # `signal` 값에 `Signal(detector)` 호출 결과를 저장합니다. 대상: `signal`.
        signal = Signal(detector)
        # `signal.hold(0, 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, 1)
        # `(0.09, 0.4, 0.4)`에서 항목을 차례로 꺼내 `peak`에 넣고 아래 처리를 반복합니다.
        for peak in (.09, .4, .4):
            # `signal.hold(peak, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
            signal.hold(peak, .6)
            # `signal.hold(0, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
            signal.hold(0, .6)
        # 검증문 `self.assertFalse(detector.consistent)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(detector.consistent)
        # 검증문 `self.assertEqual(detector.status()['state'], 'NEED_CONSISTENT_CYCLES')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(detector.status()['state'], 'NEED_CONSISTENT_CYCLES')
        # `signal.hold(0.4, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(.4, .6)
        # `signal.hold(0, 0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
        signal.hold(0, .6)
        # 검증문 `self.assertTrue(detector.consistent)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(detector.consistent)

    # 회귀 테스트: `test_controller_finishes_and_saves_after_short_occlusion` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_controller_finishes_and_saves_after_short_occlusion(self):
        # 현재 작업 설정에 키 'eval_config', 'default_thresholds'로 구성한 사전을 저장합니다. 대상: `config`.
        config = {'eval_config': {'metric_type': 'COORDINATE', 'motion_direction': 'INCREASING',
                                  # 구성 항목: `axis` 값(`'axis'`)에 'y'; 목표 kpt(`'target_kpt'`)에 키 'left', 'right'로 구성한 사전; 왼쪽(`'left'`)에 9; 오른쪽(`'right'`)에 10을 지정합니다.
                                  'axis': 'y', 'target_kpt': {'left': 9, 'right': 10}},
                  # 구성 항목: 기본 가동 범위 기준(`'default_thresholds'`)에 반복·조건식 `{side: {'start_val': 0.0, 'target_val': 1.0} for side in ('left', 'right')}`으로 구성한 값; 개인 가동 범위의 시작값(`'start_val'`)에 0.0; 개인 가동 범위의 목표값(`'target_val'`)에 1.0을 지정합니다.
                  'default_thresholds': {side: {'start_val': 0., 'target_val': 1.} for side in ('left', 'right')}}
        # `manager` 값에 `Mock()` 호출 결과를 저장합니다. 대상: `manager`.
        manager = Mock()
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(manager, MotionEngine(config, mode='CALIBRATION'))` 호출 결과를 저장합니다. 대상: `controller`.
        controller = SessionController(manager, MotionEngine(config, mode='CALIBRATION'))
        # `controller.normalizer.normalize` 값에 계산식 `lambda points: points`의 결과를 저장합니다. 대상: `controller.normalizer.normalize`.
        controller.normalizer.normalize = lambda points: points
        # 이번 처리에 사용할 현재 시각(초)에 0.0을 저장합니다. 대상: `now`.
        now = 0.

        # `hold` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def hold(value, frames):
            # `now`을 현재 함수의 새 지역 변수가 아닌 바깥 범위 변수로 사용합니다.
            nonlocal now
            # `range(frames)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
            for _ in range(frames):
                # 계산에 사용할 관절 또는 좌표 묶음에 반복·조건식 `[dict(id=i, x=0.0, y=value if i in (9, 10) else 0.0, score=1.0) for i in range(17)]`으로 구성한 값을 저장합니다. 대상: `points`.
                points = [dict(id=i, x=0., y=value if i in (9, 10) else 0., score=1.) for i in range(17)]
                # `controller.process_calibration_frame(None if value is None else points, value is None, now=now)`을 호출하여 해당 단계의 작업을 수행합니다.
                controller.process_calibration_frame(None if value is None else points, value is None, now=now)
                # 이번 처리에 사용할 현재 시각(초)에 `0.1`을 더해 누적합니다. 대상: `now`.
                now += .1

        # `hold(0, 10)`을 호출하여 해당 단계의 작업을 수행합니다.
        hold(0, 10)
        # `hold(0.2, 6)`을 호출하여 해당 단계의 작업을 수행합니다.
        hold(.2, 6)
        # `hold(0.04, 6)`을 호출하여 해당 단계의 작업을 수행합니다.
        hold(.04, 6)
        # `hold(None, 2)`을 호출하여 해당 단계의 작업을 수행합니다.
        hold(None, 2)
        # 검증문 `self.assertEqual(controller.calib_reps_left, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.calib_reps_left, 1)
        # 검증문 `self.assertEqual(controller.calib_step, 'PAUSED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.calib_step, 'PAUSED')
        # `hold(0, 4)`을 호출하여 해당 단계의 작업을 수행합니다.
        hold(0, 4)
        # `range(2)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(2):
            # `hold(0.2, 6)`을 호출하여 해당 단계의 작업을 수행합니다.
            hold(.2, 6)
            # `hold(0.04, 6)`을 호출하여 해당 단계의 작업을 수행합니다.
            hold(.04, 6)
        # `hold(0.04, 15)`을 호출하여 해당 단계의 작업을 수행합니다.
        hold(.04, 15)
        # 검증문 `self.assertEqual(controller.calib_step, 'FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.calib_step, 'FINISHED')
        # 검증문 `self.assertEqual(controller.calib_reps_right, 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.calib_reps_right, 3)
        # 검증문 `self.assertEqual(len(controller.motion_engine.fsm_left.completed_reps_history), 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(controller.motion_engine.fsm_left.completed_reps_history), 3)
        # 좌우 가동 범위 시작값·목표값에 `controller.compute_and_save_thresholds('test')` 호출 결과를 저장합니다. 대상: `thresholds`.
        thresholds = controller.compute_and_save_thresholds('test')
        # 검증문 `self.assertAlmostEqual(thresholds['left']['target_val'], 0.17)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(thresholds['left']['target_val'], .17)
        # 검증문 `manager.save_custom_threshold.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        manager.save_custom_threshold.assert_called_once()


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
