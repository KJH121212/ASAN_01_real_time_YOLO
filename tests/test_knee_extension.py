# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Regression coverage for knee-extension normalized length calibration and main sessions."""

# 이 파일에서 사용할 `contextlib` 모듈을 불러옵니다.
import contextlib
# 이 파일에서 사용할 `io` 모듈을 불러옵니다.
import io
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `CalibrationCycleDetector` 기능을 가져옵니다.
from core.session_controller import CalibrationCycleDetector
# `core.session_controller`에서 `SessionController` 기능을 가져옵니다.
from core.session_controller import SessionController
# `utils.normalization`에서 `PoseNormalizer` 기능을 가져옵니다.
from utils.normalization import PoseNormalizer


# `KneeExtensionTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class KneeExtensionTests(unittest.TestCase):
    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 경로에 계산식 `Path(__file__).resolve().parents[1] / 'configs/exercise_kpt_config_coco.json'`의 결과를 저장합니다. 대상: `path`.
        path = Path(__file__).resolve().parents[1] / 'configs/exercise_kpt_config_coco.json'
        # `path.open(encoding='utf-8')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with path.open(encoding='utf-8') as file:
            # 현재 작업 설정에 계산식 `json.load(file)['exercises']['knee_extension']`의 결과를 저장합니다. 대상: `self.config`.
            self.config = json.load(file)['exercises']['knee_extension']
        # 생성 또는 검증할 분석 엔진에 `MotionEngine(self.config, mode='CALIBRATION', target_reps=3)` 호출 결과를 저장합니다. 대상: `self.engine`.
        self.engine = MotionEngine(self.config, mode='CALIBRATION', target_reps=3)

    # `pose` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def pose(self, scale=1.0):
        # 계산에 사용할 관절 또는 좌표 묶음에 반복·조건식 `[{'id': i, 'x': 0.5, 'y': 0.5, 'score': 1.0} for i in range(17)]`으로 구성한 값을 저장합니다. 대상: `points`.
        points = [{'id': i, 'x': 0.5, 'y': 0.5, 'score': 1.0} for i in range(17)]
        # `[(5, 0.4, 0.2), (6, 0.6, 0.2), (11, 0.4, 0.5), (12, 0.6, 0.5), (13, 0.4, 0.6), (15, 0.4, 0.84)]`에서 항목을 차례로 꺼내 `(i, x, y)`에 넣고 아래 처리를 반복합니다.
        for i, x, y in [(5, .4, .2), (6, .6, .2), (11, .4, .5),
                        # 앞줄에서 시작한 식·인수·자료 구조를 `(12, .6, .5), (13, .4, .6), (15, .4, .84)]:` 내용으로 이어서 구성합니다.
                        (12, .6, .5), (13, .4, .6), (15, .4, .84)]:
            # `points[i].update(x=x * scale, y=y * scale)`을 호출하여 해당 단계의 작업을 수행합니다.
            points[i].update(x=x * scale, y=y * scale)
        # 현재 함수 처리를 끝내고 계산에 사용할 관절 또는 좌표 묶음(`points`)을 호출한 쪽으로 반환합니다.
        return points

    # 회귀 테스트: `test_length_is_independent_of_image_scale` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_length_is_independent_of_image_scale(self):
        # 골반 중심·몸통 길이 기준 좌표 변환기에 `PoseNormalizer(invert_y=True)` 호출 결과를 저장합니다. 대상: `normalizer`.
        normalizer = PoseNormalizer(invert_y=True)
        # `(1.0, 2.0)`에서 항목을 차례로 꺼내 `scale`에 넣고 아래 처리를 반복합니다.
        for scale in (1.0, 2.0):
            # 계산에 사용할 관절 또는 좌표 묶음에 `normalizer.normalize(self.pose(scale))` 호출 결과를 저장합니다. 대상: `points`.
            points = normalizer.normalize(self.pose(scale))
            # 측정값에 `self.engine._compute_side_value('left', {p['id']: p for p in points})` 호출 결과를 저장합니다. 대상: `value`.
            value = self.engine._compute_side_value('left', {p['id']: p for p in points})
            # 검증문 `self.assertAlmostEqual(value, 0.8, places=3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertAlmostEqual(value, .8, places=3)

    # 회귀 테스트: `test_length_uses_torso_normalization` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_length_uses_torso_normalization(self):
        # 골반 중심·몸통 길이 기준 좌표 변환기에 `PoseNormalizer(invert_y=True)` 호출 결과를 저장합니다. 대상: `normalizer`.
        normalizer = PoseNormalizer(invert_y=True)
        # `(0.1, 0.2, 0.35, 0.4)`에서 항목을 차례로 꺼내 `shoulder_y`에 넣고 아래 처리를 반복합니다.
        for shoulder_y in (.1, .2, .35, .4):
            # 계산에 사용할 관절 또는 좌표 묶음에 `self.pose()` 호출 결과를 저장합니다. 대상: `points`.
            points = self.pose()
            # `(5, 6)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
            for i in (5, 6):
                # `points[i]['y']` 값에 `shoulder_y` 값(`shoulder_y`)을 저장합니다. 대상: `points[i]['y']`.
                points[i]['y'] = shoulder_y
            # 신체 기준으로 정규화한 관절 목록에 `normalizer.normalize(points)` 호출 결과를 저장합니다. 대상: `normalized`.
            normalized = normalizer.normalize(points)
            # 측정값에 `self.engine._compute_side_value('left', {p['id']: p for p in normalized})` 호출 결과를 저장합니다. 대상: `value`.
            value = self.engine._compute_side_value('left', {p['id']: p for p in normalized})
            # 검증문 `self.assertAlmostEqual(value, 0.24 / (0.5 - shoulder_y), places=3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertAlmostEqual(value, .24 / (.5 - shoulder_y), places=3)

    # 회귀 테스트: `test_length_requires_only_knee_and_ankle` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_length_requires_only_knee_and_ankle(self):
        # 검증문 `self.assertEqual(self.engine.metric_type, 'LENGTH')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.engine.metric_type, 'LENGTH')
        # `(('left', 13, 15), ('right', 14, 16))`에서 항목을 차례로 꺼내 `(side, knee, ankle)`에 넣고 아래 처리를 반복합니다.
        for side, knee, ankle in (('left', 13, 15), ('right', 14, 16)):
            # 계산에 사용할 관절 또는 좌표 묶음에 키 knee, ankle로 구성한 사전을 저장합니다. 대상: `points`.
            points = {knee: dict(x=0., y=0., score=1.),
                      # 구성 항목: `ankle` 값(`ankle`)에 `dict(x=0.3, y=0.4, score=1.0)` 호출 결과; `x` 값(`x`)에 0.3; `y` 값(`y`)에 0.4; 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`score`)에 1.0을 지정합니다.
                      ankle: dict(x=.3, y=.4, score=1.)}
            # 검증문 `self.assertEqual(self.engine._compute_side_value(side, points), 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(self.engine._compute_side_value(side, points), .5)
            # `points[ankle]['score']` 값에 0.1을 저장합니다. 대상: `points[ankle]['score']`.
            points[ankle]['score'] = .1
            # 검증문 `self.assertIsNone(self.engine._compute_side_value(side, points))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertIsNone(self.engine._compute_side_value(side, points))

    # 회귀 테스트: `test_old_ratio_calibration_requires_recalibration` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_old_ratio_calibration_requires_recalibration(self):
        # 이전에 반복·조건식 `{side: dict(start_val=2.0, target_val=1.0, metric_type='RATIO') for side in ('left', 'right')}`으로 구성한 값을 저장합니다. 대상: `old`.
        old = {side: dict(start_val=2., target_val=1., metric_type='RATIO')
               # 앞줄에서 시작한 식·인수·자료 구조를 `for side in ('left', 'right')}` 내용으로 이어서 구성합니다.
               for side in ('left', 'right')}
        # `self.assertRaisesRegex(ValueError, 'CALIBRATION')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaisesRegex(ValueError, 'CALIBRATION'):
            # `MotionEngine(self.config, custom_thresholds=old)`을 호출하여 해당 단계의 작업을 수행합니다.
            MotionEngine(self.config, custom_thresholds=old)

    # 회귀 테스트: `test_occlusion_checks_body_and_length_joints` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_occlusion_checks_body_and_length_joints(self):
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(motion_engine=self.engine)` 호출 결과를 저장합니다. 대상: `controller`.
        controller = SessionController(motion_engine=self.engine)
        # `(11, 12, 13, 14, 15, 16)`에서 항목을 차례로 꺼내 `joint`에 넣고 아래 처리를 반복합니다.
        for joint in (11, 12, 13, 14, 15, 16):
            # 계산에 사용할 관절 또는 좌표 묶음에 `self.pose()` 호출 결과를 저장합니다. 대상: `points`.
            points = self.pose()
            # `points[joint]['score']` 값에 0.1을 저장합니다. 대상: `points[joint]['score']`.
            points[joint]['score'] = .1
            # 검증문 `self.assertTrue(controller.check_occlusion(points))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(controller.check_occlusion(points))

    # 회귀 테스트: `test_missing_ankle_does_not_produce_a_measurement` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_missing_ankle_does_not_produce_a_measurement(self):
        # 계산에 사용할 관절 또는 좌표 묶음에 `self.pose()` 호출 결과를 저장합니다. 대상: `points`.
        points = self.pose()
        # `points[15]['score']` 값에 0.1을 저장합니다. 대상: `points[15]['score']`.
        points[15]['score'] = .1
        # 검증문 `self.assertIsNone(self.engine._compute_side_value('left', {p['id']: p for p in points}))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(self.engine._compute_side_value('left', {p['id']: p for p in points}))

    # 회귀 테스트: `test_legacy_distance_thresholds_require_recalibration` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_legacy_distance_thresholds_require_recalibration(self):
        # 이전에 반복·조건식 `{side: dict(start_val=1.0, target_val=0.5) for side in ('left', 'right')}`으로 구성한 값을 저장합니다. 대상: `old`.
        old = {side: dict(start_val=1., target_val=.5) for side in ('left', 'right')}
        # `self.assertRaisesRegex(ValueError, 'CALIBRATION')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaisesRegex(ValueError, 'CALIBRATION'):
            # `MotionEngine(self.config, custom_thresholds=old)`을 호출하여 해당 단계의 작업을 수행합니다.
            MotionEngine(self.config, custom_thresholds=old)

    # 회귀 테스트: `test_controller_completes_and_personal_thresholds_work_in_main` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_controller_completes_and_personal_thresholds_work_in_main(self):
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(motion_engine=self.engine)` 호출 결과를 저장합니다. 대상: `controller`.
        controller = SessionController(motion_engine=self.engine)
        # 이번 처리에 사용할 현재 시각(초)에 0.0을 저장합니다. 대상: `now`.
        now = 0.
        # `feed` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def feed(rise, seconds):
            # `now`을 현재 함수의 새 지역 변수가 아닌 바깥 범위 변수로 사용합니다.
            nonlocal now
            # `range(round(seconds * 30))`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
            for _ in range(round(seconds * 30)):
                # `pts` 값에 `self.pose()` 호출 결과를 저장합니다. 대상: `pts`.
                pts = self.pose()
                # `((13, 15, 0.4), (14, 16, 0.6))`에서 항목을 차례로 꺼내 `(knee, ankle, x)`에 넣고 아래 처리를 반복합니다.
                for knee, ankle, x in ((13, 15, .4), (14, 16, .6)):
                    # `pts[knee].update(x=x, y=0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
                    pts[knee].update(x=x, y=.6)
                    # `pts[ankle].update(x=x, y=0.84 - rise * 0.3)`을 호출하여 해당 단계의 작업을 수행합니다.
                    pts[ankle].update(x=x, y=.84 - rise * .3)
                # `controller.process_calibration_frame(pts, False, now=now)`을 호출하여 해당 단계의 작업을 수행합니다.
                controller.process_calibration_frame(pts, False, now=now)
                # 이번 처리에 사용할 현재 시각(초)에 `1 / 30`을 더해 누적합니다. 대상: `now`.
                now += 1/30
        # `feed(0, 2.4)`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(0, 2.4)
        # 검증문 `self.assertEqual(controller.calib_step, 'COLLECTING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.calib_step, 'COLLECTING')
        # `range(3)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(3):
            # `feed(0.25, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
            feed(.25, .4)
            # `feed(0, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
            feed(0, .4)
        # `feed(0, 1.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(0, 1.2)
        # 검증문 `self.assertEqual(controller.calib_step, 'FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.calib_step, 'FINISHED')
        # 좌우 가동 범위 시작값·목표값에 `controller.compute_and_save_thresholds('knee_extension')` 호출 결과를 저장합니다. 대상: `thresholds`.
        thresholds = controller.compute_and_save_thresholds('knee_extension')
        # 진입점에 `MotionEngine(self.config, custom_thresholds=thresholds)` 호출 결과를 저장합니다. 대상: `main`.
        main = MotionEngine(self.config, custom_thresholds=thresholds)
        # `contextlib.redirect_stdout(io.StringIO())`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with contextlib.redirect_stdout(io.StringIO()):
            # `(0.8, 0.55, 0.55, 0.8)`에서 항목을 차례로 꺼내 `v`에 넣고 아래 처리를 반복합니다.
            for v in (.8, .55, .55, .8):
                # `main.fsm_left.update(v)`을 호출하여 해당 단계의 작업을 수행합니다.
                main.fsm_left.update(v)
        # 검증문 `self.assertEqual(main.fsm_left.rep_count, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(main.fsm_left.rep_count, 1)


# `AnkleCalibrationTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class AnkleCalibrationTests(unittest.TestCase):
    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `CalibrationCycleDetector()` 호출 결과를 저장합니다. 대상: `self.detector`.
        self.detector = CalibrationCycleDetector()
        # 이번 처리에 사용할 현재 시각(초)에 0.0을 저장합니다. 대상: `self.now`.
        self.now = 0.

    # `feed` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def feed(self, value, seconds):
        # 이벤트에 0을 저장합니다. 대상: `events`.
        events = 0
        # `range(round(seconds * 30))`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(round(seconds * 30)):
            # 이벤트에 `self.detector.update(value, self.now)`을 더해 누적합니다. 대상: `events`.
            events += self.detector.update(value, self.now)
            # 이번 처리에 사용할 현재 시각(초)에 `1 / 30`을 더해 누적합니다. 대상: `self.now`.
            self.now += 1/30
        # 현재 함수 처리를 끝내고 이벤트(`events`)을 호출한 쪽으로 반환합니다.
        return events

    # 회귀 테스트: `test_personal_rest_and_three_complete_cycles` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_personal_rest_and_three_complete_cycles(self):
        # `(-1.2, -0.5)`에서 항목을 차례로 꺼내 `rest`에 넣고 아래 처리를 반복합니다.
        for rest in (-1.2, -.5):
            # `self.detector.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.detector.reset()
            # `self.feed(rest, 2.2)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(rest, 2.2)
            # 검증문 `self.assertTrue(self.detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(self.detector.ready)
            # `range(3)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
            for _ in range(3):
                # `self.feed(rest + 0.2, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
                self.feed(rest + .2, .4)
                # `self.feed(rest, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
                self.feed(rest, .4)
            # 검증문 `self.assertTrue(self.detector.consistent)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(self.detector.consistent)
            # 검증문 `self.assertAlmostEqual(self.detector.thresholds()['target_val'], rest + 0.17)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertAlmostEqual(self.detector.thresholds()['target_val'], rest + .17)

    # 회귀 테스트: `test_stationary_jitter_and_single_spike_do_not_count` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_stationary_jitter_and_single_spike_do_not_count(self):
        # `self.feed(-1, 2.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, 2.2)
        # `range(50)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(50):
            # `self.feed(-0.99, 0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(-.99, .1)
            # `self.feed(-1.01, 0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(-1.01, .1)
        # `self.feed(-0.6, 1 / 30)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-.6, 1/30)
        # `self.feed(-1, 0.5)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, .5)
        # 검증문 `self.assertEqual(self.detector.peaks, [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.detector.peaks, [])

    # 회귀 테스트: `test_ascent_without_return_does_not_count` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_ascent_without_return_does_not_count(self):
        # `self.feed(-1, 2.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, 2.2)
        # `self.feed(-0.7, 2)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-.7, 2)
        # 검증문 `self.assertEqual(self.detector.peaks, [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.detector.peaks, [])

    # 회귀 테스트: `test_noise_increases_activation_above_common_floor` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_noise_increases_activation_above_common_floor(self):
        # `range(80)`에서 항목을 차례로 꺼내 `index`에 넣고 아래 처리를 반복합니다.
        for index in range(80):
            # `self.feed(-1 + 0.018 * math.sin(index * 2 * math.pi / 15), 1 / 30)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(-1 + .018 * math.sin(index * 2 * math.pi / 15), 1/30)
        # 검증문 `self.assertTrue(self.detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.detector.ready)
        # 검증문 `self.assertGreater(self.detector.threshold, self.detector.min_rise)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertGreater(self.detector.threshold, self.detector.min_rise)
        # `self.feed(-0.92, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-.92, .4)
        # `self.feed(-1, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, .4)
        # 검증문 `self.assertEqual(self.detector.peaks, [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.detector.peaks, [])

    # 회귀 테스트: `test_time_gap_does_not_complete_an_old_cycle` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_time_gap_does_not_complete_an_old_cycle(self):
        # `self.feed(-1, 2.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, 2.2)
        # `self.feed(-0.7, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-.7, .4)
        # 이번 처리에 사용할 현재 시각(초)에 `2`을 더해 누적합니다. 대상: `self.now`.
        self.now += 2
        # `self.feed(-1, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, .4)
        # 검증문 `self.assertEqual(self.detector.peaks, [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.detector.peaks, [])

    # 회귀 테스트: `test_lower_activation_does_not_tighten_preparation_stability` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_lower_activation_does_not_tighten_preparation_stability(self):
        # 좌우 캘리브레이션 판정기 모음에 반복·조건식 `[CalibrationCycleDetector(min_rise=rise) for rise in (0.08, 0.025)]`으로 구성한 값을 저장합니다. 대상: `detectors`.
        detectors = [CalibrationCycleDetector(min_rise=rise) for rise in (.08, .025)]
        # `range(80)`에서 항목을 차례로 꺼내 `index`에 넣고 아래 처리를 반복합니다.
        for index in range(80):
            # 측정값에 계산식 `-1 + 0.014 * math.sin(index * 2 * math.pi / 15)`의 결과를 저장합니다. 대상: `value`.
            value = -1 + .014 * math.sin(index * 2 * math.pi / 15)
            # `detectors`에서 항목을 차례로 꺼내 `detector`에 넣고 아래 처리를 반복합니다.
            for detector in detectors:
                # `detector.update(value, index / 30)`을 호출하여 해당 단계의 작업을 수행합니다.
                detector.update(value, index/30)
        # 검증문 `self.assertTrue(all((d.ready for d in detectors)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(all(d.ready for d in detectors))
        # 검증문 `self.assertEqual(detectors[0].baseline_at, detectors[1].baseline_at)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(detectors[0].baseline_at, detectors[1].baseline_at)

    # 회귀 테스트: `test_moving_preparation_does_not_establish_rest` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_moving_preparation_does_not_establish_rest(self):
        # `range(30)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(30):
            # `self.feed(-1, 0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(-1, .1)
            # `self.feed(-0.7, 0.1)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(-.7, .1)
        # 검증문 `self.assertFalse(self.detector.ready)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(self.detector.ready)

    # 회귀 테스트: `test_missing_data_cancels_cycle` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_missing_data_cancels_cycle(self):
        # `self.feed(-1, 2.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, 2.2)
        # `self.feed(-0.7, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-.7, .4)
        # `self.feed(None, 0.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(None, .2)
        # `self.feed(-1, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, .4)
        # 검증문 `self.assertEqual(self.detector.peaks, [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.detector.peaks, [])

    # 회귀 테스트: `test_inconsistent_repetitions_cannot_save_thresholds` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_inconsistent_repetitions_cannot_save_thresholds(self):
        # `self.feed(-1, 2.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.feed(-1, 2.2)
        # `(0.1, 0.4, 0.1)`에서 항목을 차례로 꺼내 `amplitude`에 넣고 아래 처리를 반복합니다.
        for amplitude in (.1, .4, .1):
            # `self.feed(-1 + amplitude, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(-1 + amplitude, .4)
            # `self.feed(-1, 0.4)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.feed(-1, .4)
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `self.detector.thresholds()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.detector.thresholds()


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
