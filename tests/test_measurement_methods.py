# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `CalibrationCycleDetector`, `SessionController` 기능을 가져옵니다.
from core.session_controller import CalibrationCycleDetector, SessionController


# `MeasurementMethodsTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class MeasurementMethodsTests(unittest.TestCase):
    # 생성 또는 검증할 분석 엔진 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def engine(self, metric, **settings):
        # 현재 함수 처리를 끝내고 `MotionEngine({'eval_config': {'metric_type': metric, **settings}, 'default_thresholds': {s: {'start_val': 0.0, 'target_val': 1.0} for s in ('left', 'right')}})` 호출 결과를 호출한 쪽으로 반환합니다.
        return MotionEngine({'eval_config': {'metric_type': metric, **settings},
                             # 구성 항목: 기본 가동 범위 기준(`'default_thresholds'`)에 반복·조건식 `{s: {'start_val': 0.0, 'target_val': 1.0} for s in ('left', 'right')}`으로 구성한 값; 개인 가동 범위의 시작값(`'start_val'`)에 0.0; 개인 가동 범위의 목표값(`'target_val'`)에 1.0을 지정합니다.
                             'default_thresholds': {s: {'start_val': 0., 'target_val': 1.}
                                                    # 앞줄에서 시작한 식·인수·자료 구조를 `for s in ('left', 'right')}})` 내용으로 이어서 구성합니다.
                                                    for s in ('left', 'right')}})

    # 회귀 테스트: `test_four_measurement_functions` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_four_measurement_functions(self):
        # 계산에 사용할 관절 또는 좌표 묶음에 반복·조건식 `{i: dict(id=i, x=x, y=y, score=1.0) for (i, x, y) in [(1, 0.0, 1.0), (2, 0.0, 0.0), (3, 1.0, 0.0)]}`으로 구성한 값을 저장합니다. 대상: `points`.
        points = {i: dict(id=i, x=x, y=y, score=1.)
                  # 앞줄에서 시작한 식·인수·자료 구조를 `for i, x, y in [(1, 0., 1.), (2, 0., 0.), (3, 1., 0.)]}` 내용으로 이어서 구성합니다.
                  for i, x, y in [(1, 0., 1.), (2, 0., 0.), (3, 1., 0.)]}
        # 각도에 `self.engine('ANGLE', primary_kpts={'left': [1, 2, 3]})` 호출 결과를 저장합니다. 대상: `angle`.
        angle = self.engine('ANGLE', primary_kpts={'left': [1, 2, 3]})
        # 검증문 `self.assertAlmostEqual(angle._compute_side_value('left', points), 90.0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(angle._compute_side_value('left', points), 90.)
        # 좌표에 `self.engine('COORDINATE', target_kpt={'left': 1}, reference_kpt={'left': 3}, axis='x')` 호출 결과를 저장합니다. 대상: `coordinate`.
        coordinate = self.engine('COORDINATE', target_kpt={'left': 1}, reference_kpt={'left': 3}, axis='x')
        # 검증문 `self.assertAlmostEqual(coordinate._compute_side_value('left', points), -1.0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(coordinate._compute_side_value('left', points), -1.)
        # 길이에 `self.engine('LENGTH', primary_kpts={'left': [1, 3]})` 호출 결과를 저장합니다. 대상: `length`.
        length = self.engine('LENGTH', primary_kpts={'left': [1, 3]})
        # 검증문 `self.assertAlmostEqual(length._compute_side_value('left', points), 2 ** 0.5, places=4)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(length._compute_side_value('left', points), 2**.5, places=4)

    # 비율 엔진 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def ratio_engine(self):
        # 현재 함수 처리를 끝내고 `self.engine('RATIO', numerator_kpts={'left': [2, 3], 'right': [2, 3]}, denominator_kpts={'left': [1, 2], 'right': [1, 2]})` 호출 결과를 호출한 쪽으로 반환합니다.
        return self.engine('RATIO', numerator_kpts={'left': [2, 3], 'right': [2, 3]},
                           # 구성 항목: `denominator_kpts` 값(`denominator_kpts`)에 키 'left', 'right'로 구성한 사전; 왼쪽(`'left'`)에 순서·구성을 지정한 `[1, 2]`; 오른쪽(`'right'`)에 순서·구성을 지정한 `[1, 2]`을 지정합니다.
                           denominator_kpts={'left': [1, 2], 'right': [1, 2]})

    # 회귀 테스트: `test_ratio_cancels_scale_translation_and_rotation` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_ratio_cancels_scale_translation_and_rotation(self):
        # 이 파일에서 사용할 `math` 모듈을 불러옵니다.
        import math
        # 생성 또는 검증할 분석 엔진에 `self.ratio_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = self.ratio_engine()
        # `(0.01, 0.5, 1.0, 100.0)`에서 항목을 차례로 꺼내 `scale`에 넣고 아래 처리를 반복합니다.
        for scale in (.01, .5, 1., 100.):
            # `(0.0, 0.7, 2.0)`에서 항목을 차례로 꺼내 `angle`에 넣고 아래 처리를 반복합니다.
            for angle in (0., .7, 2.):
                # 계산에 사용할 관절 또는 좌표 묶음에 반복·조건식 `{i: dict(id=i, x=7.0 - scale * y * math.sin(angle), y=-3.0 + scale * y * math.cos(angle), score=1.0) for (i, y) in ((1, 0.0), (2, 2.0), (3, 5.0))}`으로 구성한 값을 저장합니다. 대상: `points`.
                points = {i: dict(id=i, x=7.-scale*y*math.sin(angle),
                                  # 구성 항목: `y` 값(`y`)에 계산식 `-3.0 + scale * y * math.cos(angle)`의 결과; 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`score`)에 1.0을 지정합니다.
                                  y=-3.+scale*y*math.cos(angle), score=1.)
                          # 앞줄에서 시작한 식·인수·자료 구조를 `for i, y in ((1, 0.), (2, 2.), (3, 5.))}` 내용으로 이어서 구성합니다.
                          for i, y in ((1, 0.), (2, 2.), (3, 5.))}
                # `('left', 'right')`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
                for side in ('left', 'right'):
                    # 검증문 `self.assertAlmostEqual(engine._compute_side_value(side, points), 1.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertAlmostEqual(engine._compute_side_value(side, points), 1.5)

    # 회귀 테스트: `test_ratio_rejects_invalid_joints_and_collapsed_denominator` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_ratio_rejects_invalid_joints_and_collapsed_denominator(self):
        # `copy`에서 `deepcopy` 기능을 가져옵니다.
        from copy import deepcopy
        # 생성 또는 검증할 분석 엔진에 `self.ratio_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = self.ratio_engine()
        # `good` 값에 반복·조건식 `{i: dict(id=i, x=0.0, y=y, score=1.0) for (i, y) in ((1, 0.0), (2, 2.0), (3, 5.0))}`으로 구성한 값을 저장합니다. 대상: `good`.
        good = {i: dict(id=i, x=0., y=y, score=1.)
                # 앞줄에서 시작한 식·인수·자료 구조를 `for i, y in ((1, 0.), (2, 2.), (3, 5.))}` 내용으로 이어서 구성합니다.
                for i, y in ((1, 0.), (2, 2.), (3, 5.))}
        # `(1, 2, 3)`에서 항목을 차례로 꺼내 `joint`에 넣고 아래 처리를 반복합니다.
        for joint in (1, 2, 3):
            # `(('score', 0.1), ('x', float('nan')), ('y', float('inf')))`에서 항목을 차례로 꺼내 `(field, value)`에 넣고 아래 처리를 반복합니다.
            for field, value in (('score', .1), ('x', float('nan')), ('y', float('inf'))):
                # 계산에 사용할 관절 또는 좌표 묶음에 `deepcopy(good)` 호출 결과를 저장합니다. 대상: `points`.
                points = deepcopy(good)
                # `points[joint][field]` 값에 측정값(`value`)을 저장합니다. 대상: `points[joint][field]`.
                points[joint][field] = value
                # 검증문 `self.assertIsNone(engine._compute_side_value('left', points))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertIsNone(engine._compute_side_value('left', points))
            # 계산에 사용할 관절 또는 좌표 묶음에 `deepcopy(good)` 호출 결과를 저장합니다. 대상: `points`.
            points = deepcopy(good)
            # `points[joint]` 항목을 제거합니다.
            del points[joint]
            # 검증문 `self.assertIsNone(engine._compute_side_value('left', points))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertIsNone(engine._compute_side_value('left', points))
        # `(0.0, 1e-08)`에서 항목을 차례로 꺼내 `delta`에 넣고 아래 처리를 반복합니다.
        for delta in (0., 1e-8):
            # 계산에 사용할 관절 또는 좌표 묶음에 `deepcopy(good)` 호출 결과를 저장합니다. 대상: `points`.
            points = deepcopy(good)
            # `points[2]['y']` 값에 `delta` 값(`delta`)을 저장합니다. 대상: `points[2]['y']`.
            points[2]['y'] = delta
            # 검증문 `self.assertIsNone(engine._compute_side_value('left', points))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertIsNone(engine._compute_side_value('left', points))

    # 회귀 테스트: `test_ratio_rejects_old_coordinate_calibration_and_negative_rom` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_ratio_rejects_old_coordinate_calibration_and_negative_rom(self):
        # 생성 또는 검증할 분석 엔진에 `self.ratio_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = self.ratio_engine()
        # `(None, 'COORDINATE', 'LENGTH')`에서 항목을 차례로 꺼내 `old_metric`에 넣고 아래 처리를 반복합니다.
        for old_metric in (None, 'COORDINATE', 'LENGTH'):
            # 이전에 반복·조건식 `{s: dict(start_val=-1.0, target_val=-0.5, metric_type=old_metric) for s in ('left', 'right')}`으로 구성한 값을 저장합니다. 대상: `old`.
            old = {s: dict(start_val=-1., target_val=-.5, metric_type=old_metric)
                   # 앞줄에서 시작한 식·인수·자료 구조를 `for s in ('left', 'right')}` 내용으로 이어서 구성합니다.
                   for s in ('left', 'right')}
            # `self.assertRaisesRegex(ValueError, 'CALIBRATION')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.assertRaisesRegex(ValueError, 'CALIBRATION'):
                # `MotionEngine(engine.config, custom_thresholds=old)`을 호출하여 해당 단계의 작업을 수행합니다.
                MotionEngine(engine.config, custom_thresholds=old)
        # `self.assertRaisesRegex(ValueError, 'nonnegative')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            # `engine.update_count_thresholds({'left': {'start_val': -0.1}})`을 호출하여 해당 단계의 작업을 수행합니다.
            engine.update_count_thresholds({'left': {'start_val': -.1}})

    # 회귀 테스트: `test_ratio_requires_two_valid_pairs_for_each_side` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_ratio_requires_two_valid_pairs_for_each_side(self):
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `self.engine('RATIO')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.engine('RATIO')

    # 회귀 테스트: `test_unknown_metric_is_rejected` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_unknown_metric_is_rejected(self):
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `self.engine('OTHER')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.engine('OTHER')

    # 회귀 테스트: `test_common_detector_for_all_units_and_both_directions` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_common_detector_for_all_units_and_both_directions(self):
        # `[('ANGLE', 90.0, 30.0, 8.0), ('COORDINATE', -0.8, 0.2, 0.08), ('LENGTH', 1.0, 0.2, 0.08), ('RATIO', 1.0, 0.2, 0.08)]`에서 항목을 차례로 꺼내 `(metric, baseline, amplitude, minimum)`에 넣고 아래 처리를 반복합니다.
        for metric, baseline, amplitude, minimum in [('ANGLE', 90., 30., 8.),
                                                    # 앞줄에서 시작한 식·인수·자료 구조를 `('COORDINATE', -.8, .2, .08),` 내용으로 이어서 구성합니다.
                                                    ('COORDINATE', -.8, .2, .08),
                                                    # 앞줄에서 시작한 식·인수·자료 구조를 `('LENGTH', 1., .2, .08),` 내용으로 이어서 구성합니다.
                                                    ('LENGTH', 1., .2, .08),
                                                    # 앞줄에서 시작한 식·인수·자료 구조를 `('RATIO', 1., .2, .08)]:` 내용으로 이어서 구성합니다.
                                                    ('RATIO', 1., .2, .08)]:
            # `(-1.0, 1.0)`에서 항목을 차례로 꺼내 `sign`에 넣고 아래 처리를 반복합니다.
            for sign in (-1., 1.):
                # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector(metric_type=metric, direction='AUTO', min_rise=minimum, baseline_range=minimum * 0.5)` 호출 결과를 저장합니다. 대상: `detector`.
                detector = CalibrationCycleDetector(metric_type=metric, direction='AUTO', min_rise=minimum,
                                                     # 구성 항목: 기본 자세에서 허용하는 측정값 흔들림 폭(`baseline_range`)에 계산식 `minimum * 0.5`의 결과를 지정합니다.
                                                     baseline_range=minimum*.5)
                # 이번 처리에 사용할 현재 시각(초)에 0.0을 저장합니다. 대상: `now`.
                now = 0.
                # `feed` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
                def feed(value, frames):
                    # `now`을 현재 함수의 새 지역 변수가 아닌 바깥 범위 변수로 사용합니다.
                    nonlocal now
                    # `range(frames)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
                    for _ in range(frames):
                        # `detector.update(value, now)`을 호출하여 해당 단계의 작업을 수행합니다.
                        detector.update(value, now)
                        # 이번 처리에 사용할 현재 시각(초)에 `1 / 30`을 더해 누적합니다. 대상: `now`.
                        now += 1/30
                # `feed(baseline, 30)`을 호출하여 해당 단계의 작업을 수행합니다.
                feed(baseline, 30)
                # `range(3)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
                for _ in range(3):
                    # `feed(baseline + sign * amplitude, 12)`을 호출하여 해당 단계의 작업을 수행합니다.
                    feed(baseline + sign*amplitude, 12)
                    # `feed(baseline, 12)`을 호출하여 해당 단계의 작업을 수행합니다.
                    feed(baseline, 12)
                # 검증문 `self.assertTrue(detector.consistent)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertTrue(detector.consistent)
                # `th` 값에 `detector.thresholds()` 호출 결과를 저장합니다. 대상: `th`.
                th = detector.thresholds()
                # 검증문 `self.assertAlmostEqual(th['target_val'], baseline + sign * 0.85 * amplitude)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertAlmostEqual(th['target_val'], baseline + sign*.85*amplitude)
                # 검증문 `self.assertEqual(th['metric_type'], metric)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(th['metric_type'], metric)

    # 회귀 테스트: `test_auto_direction_not_committed_by_a_spike` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_auto_direction_not_committed_by_a_spike(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector(direction='AUTO')` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector(direction='AUTO')
        # `range(30)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
        for i in range(30):
            # `detector.update(0.0, i / 30)`을 호출하여 해당 단계의 작업을 수행합니다.
            detector.update(0., i/30)
        # `detector.update(-0.2, 1.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        detector.update(-.2, 1.)
        # `detector.update(0.0, 1.033)`을 호출하여 해당 단계의 작업을 수행합니다.
        detector.update(0., 1.033)
        # 검증문 `self.assertEqual(detector.direction, 'AUTO')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(detector.direction, 'AUTO')
        # `range(15)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
        for i in range(15):
            # `detector.update(0.2, 1.1 + i / 30)`을 호출하여 해당 단계의 작업을 수행합니다.
            detector.update(.2, 1.1+i/30)
        # `range(15)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
        for i in range(15):
            # `detector.update(0.0, 1.6 + i / 30)`을 호출하여 해당 단계의 작업을 수행합니다.
            detector.update(0., 1.6+i/30)
        # 검증문 `self.assertEqual(detector.direction, 'INCREASING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(detector.direction, 'INCREASING')
        # 검증문 `self.assertEqual(len(detector.peaks), 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(detector.peaks), 1)

    # 회귀 테스트: `test_derivative_uses_elapsed_seconds` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_derivative_uses_elapsed_seconds(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `detector`.
        detector = CalibrationCycleDetector()
        # `detector.update(0.0, 0.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        detector.update(0., 0.)
        # `detector.update(0.1, 0.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        detector.update(.1, .2)
        # 검증문 `self.assertAlmostEqual(detector.velocity, 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(detector.velocity, .5)

    # 회귀 테스트: `test_main_restores_each_calibrated_direction` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_main_restores_each_calibrated_direction(self):
        # 생성 또는 검증할 분석 엔진에 `self.engine('COORDINATE')` 호출 결과를 저장합니다. 대상: `engine`.
        engine = self.engine('COORDINATE')
        # 좌우 가동 범위 시작값·목표값에 반복·조건식 `{s: dict(start_val=0.0, target_val=sign * 0.2, metric_type='COORDINATE', motion_direction='DECREASING' if sign < 0 else 'INCREASING') for (s, sign) in [('left', -1), ('r…`으로 구성한 값을 저장합니다. 대상: `thresholds`.
        thresholds = {s: dict(start_val=0., target_val=sign*.2, metric_type='COORDINATE',
                             # 구성 항목: 측정값 증가·감소 방향(`motion_direction`)에 `sign < 0`이면 `'DECREASING'`, 아니면 `'INCREASING'`을 지정합니다.
                             motion_direction='DECREASING' if sign < 0 else 'INCREASING')
                      # 앞줄에서 시작한 식·인수·자료 구조를 `for s, sign in [('left', -1), ('right', 1)]}` 내용으로 이어서 구성합니다.
                      for s, sign in [('left', -1), ('right', 1)]}
        # 진입점에 `MotionEngine(engine.config, custom_thresholds=thresholds)` 호출 결과를 저장합니다. 대상: `main`.
        main = MotionEngine(engine.config, custom_thresholds=thresholds)
        # 검증문 `self.assertTrue(main.fsm_left.is_decreasing)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(main.fsm_left.is_decreasing)
        # 검증문 `self.assertFalse(main.fsm_right.is_decreasing)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(main.fsm_right.is_decreasing)
        # `main.update_count_thresholds({'left': {'target_val': -0.3}})`을 호출하여 해당 단계의 작업을 수행합니다.
        main.update_count_thresholds({'left': {'target_val': -.3}})
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `main.update_count_thresholds({'left': {'target_val': 0.3}})`을 호출하여 해당 단계의 작업을 수행합니다.
            main.update_count_thresholds({'left': {'target_val': .3}})


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
