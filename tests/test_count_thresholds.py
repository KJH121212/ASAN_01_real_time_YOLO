# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `copy` 모듈을 불러옵니다.
import copy
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `SessionController` 기능을 가져옵니다.
from core.session_controller import SessionController
# `network.session_commands`에서 `handle_count_threshold_command` 기능을 가져옵니다.
from network.session_commands import handle_count_threshold_command
# `network.threshold_test_client`에서 `make_command` 기능을 가져옵니다.
from network.threshold_test_client import make_command


# 생성 엔진 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def make_engine(direction="INCREASING", metric="RELATIVE_Y", target_reps=10):
    # `(start, target)` 값에 `direction == 'INCREASING'`이면 `(0, 100)`, 아니면 `(180, 20)`을 저장합니다. 대상: `(start, target)`.
    start, target = (0, 100) if direction == "INCREASING" else (180, 20)
    # 현재 함수 처리를 끝내고 `MotionEngine({'exercise_name': 'test_motion', 'eval_config': {'metric_type': metric, 'motion_direction': direction}, 'default_thresholds': {side: {'start_val': start, 't…` 호출 결과를 호출한 쪽으로 반환합니다.
    return MotionEngine({
        # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 'test_motion'을 지정합니다.
        "exercise_name": "test_motion",
        # 구성 항목: eval 설정(`'eval_config'`)에 키 'metric_type', 'motion_direction'로 구성한 사전; 측정 방식(각도·좌표·길이·비율)(`'metric_type'`)에 `metric` 값(`metric`); 측정값 증가·감소 방향(`'motion_direction'`)에 운동 방향(`direction`)을 지정합니다.
        "eval_config": {"metric_type": metric, "motion_direction": direction},
        # 구성 항목: 기본 가동 범위 기준(`'default_thresholds'`)에 반복·조건식 `{side: {'start_val': start, 'target_val': target} for side in ('left', 'right')}`으로 구성한 값을 지정합니다.
        "default_thresholds": {
            # 구성 항목: 개인 가동 범위의 시작값(`'start_val'`)에 시작(`start`); 개인 가동 범위의 목표값(`'target_val'`)에 목표(`target`)을 지정합니다.
            side: {"start_val": start, "target_val": target} for side in ("left", "right")
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        },
    # 구성 항목: 목표 반복 횟수(`target_reps`)에 목표 반복 횟수(`target_reps`)을 지정합니다.
    }, target_reps=target_reps)


# `feed` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def feed(fsm, values):
    # 현재 함수 처리를 끝내고 반복·조건식 `[fsm.update(value) for value in values]`으로 구성한 값을 호출한 쪽으로 반환합니다.
    return [fsm.update(value) for value in values]


# `CountThresholdTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class CountThresholdTests(unittest.TestCase):
    # 회귀 테스트: `test_reduced_rom_accepts_previously_insufficient_motion` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_reduced_rom_accepts_previously_insufficient_motion(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # 한쪽 동작의 반복 판정 상태 기계에 왼쪽 반복 판정 상태 기계(`engine.fsm_left`)을 저장합니다. 대상: `fsm`.
        fsm = engine.fsm_left
        # `feed(fsm, [0, 20, 30, 20, 0])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(fsm, [0, 20, 30, 20, 0])
        # 검증문 `self.assertEqual(fsm.rep_count, 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.rep_count, 0)  # 30 < 40% of the original 100.
        # `engine.update_count_thresholds({'left': {'target_val': 50}})`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds({"left": {"target_val": 50}})
        # `feed(fsm, [0, 20, 30, 20])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(fsm, [0, 20, 30, 20])
        # 검증문 `self.assertEqual(fsm.rep_count, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.rep_count, 1)  # Now 60%: GOOD.
        # 검증문 `self.assertEqual(fsm.last_quality, 'GOOD')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.last_quality, "GOOD")
        # 검증문 `self.assertEqual(fsm.completed_reps_history[-1]['target_val'], 50)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.completed_reps_history[-1]['target_val'], 50)
        # 검증문 `self.assertEqual(fsm.completed_reps_history[-1]['threshold_version'], 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.completed_reps_history[-1]['threshold_version'], 1)

    # 회귀 테스트: `test_edit_mid_motion_discards_old_peak_and_waits_for_return` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_edit_mid_motion_discards_old_peak_and_waits_for_return(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # 한쪽 동작의 반복 판정 상태 기계에 왼쪽 반복 판정 상태 기계(`engine.fsm_left`)을 저장합니다. 대상: `fsm`.
        fsm = engine.fsm_left
        # `feed(fsm, [0, 20, 70])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(fsm, [0, 20, 70])
        # 검증문 `self.assertEqual(fsm.state, 'PUSHING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.state, "PUSHING")
        # `engine.update_count_thresholds({'left': {'target_val': 50}})`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds({"left": {"target_val": 50}})
        # `results` 값에 `feed(fsm, [60, 40, 30, 10])` 호출 결과를 저장합니다. 대상: `results`.
        results = feed(fsm, [60, 40, 30, 10])
        # 검증문 `self.assertFalse(any((row['is_count_updated'] for row in results)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(any(row['is_count_updated'] for row in results))
        # 검증문 `self.assertEqual(fsm.state, 'READY')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.state, "READY")
        # `feed(fsm, [20, 40, 30])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(fsm, [20, 40, 30])
        # 검증문 `self.assertEqual(fsm.rep_count, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.rep_count, 1)

    # 회귀 테스트: `test_completed_reps_quality_history_and_other_side_are_preserved` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_completed_reps_quality_history_and_other_side_are_preserved(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # `feed(engine.fsm_left, [0, 20, 80, 60])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [0, 20, 80, 60])
        # `feed(engine.fsm_right, [0, 20, 70])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_right, [0, 20, 70])
        # `history` 값에 `copy.deepcopy(engine.fsm_left.completed_reps_history)` 호출 결과를 저장합니다. 대상: `history`.
        history = copy.deepcopy(engine.fsm_left.completed_reps_history)
        # `other_side` 값에 `copy.deepcopy(engine.fsm_right.__dict__)` 호출 결과를 저장합니다. 대상: `other_side`.
        other_side = copy.deepcopy(engine.fsm_right.__dict__)
        # `changed` 값에 `engine.update_count_thresholds({'left': {'target_val': 50}})` 호출 결과를 저장합니다. 대상: `changed`.
        changed = engine.update_count_thresholds({"left": {"target_val": 50}})
        # 검증문 `self.assertEqual(changed, ['left'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(changed, ["left"])
        # 검증문 `self.assertEqual(engine.fsm_left.rep_count, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.rep_count, 1)
        # 검증문 `self.assertEqual(engine.fsm_left.last_quality, 'PERFECT')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.last_quality, "PERFECT")
        # 검증문 `self.assertEqual(engine.fsm_left.completed_reps_history, history)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.completed_reps_history, history)
        # 검증문 `self.assertEqual(engine.fsm_right.__dict__, other_side)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_right.__dict__, other_side)
        # `feed(engine.fsm_left, [0, 20, 40, 30])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [0, 20, 40, 30])
        # 검증문 `self.assertEqual(engine.fsm_left.rep_count, 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.rep_count, 2)

    # 회귀 테스트: `test_decreasing_direction_reduces_rom_by_raising_target` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_decreasing_direction_reduces_rom_by_raising_target(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine('DECREASING', 'ANGLE')` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine("DECREASING", "ANGLE")
        # `feed(engine.fsm_left, [180, 150, 135, 150, 180])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [180, 150, 135, 150, 180])
        # 검증문 `self.assertEqual(engine.fsm_left.rep_count, 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.rep_count, 0)
        # `engine.update_count_thresholds({'left': {'target_val': 100}})`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds({"left": {"target_val": 100}})
        # `feed(engine.fsm_left, [180, 150, 135, 150])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [180, 150, 135, 150])
        # 검증문 `self.assertEqual(engine.fsm_left.rep_count, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.rep_count, 1)
        # 검증문 `self.assertEqual(engine.fsm_left.last_quality, 'GOOD')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.last_quality, "GOOD")

    # 회귀 테스트: `test_invalid_second_side_does_not_change_first_side` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_invalid_second_side_does_not_change_first_side(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # 이전에 `copy.deepcopy(engine.__dict__)` 호출 결과를 저장합니다. 대상: `before`.
        before = copy.deepcopy(engine.__dict__)
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `engine.update_count_thresholds({'left': {'target_val': 50}, 'right': {'target_val': -1}})`을 호출하여 해당 단계의 작업을 수행합니다.
            engine.update_count_thresholds({"left": {"target_val": 50},
                                            # 구성 항목: 오른쪽(`'right'`)에 키 'target_val'로 구성한 사전; 개인 가동 범위의 목표값(`'target_val'`)에 계산식 `-1`의 결과를 지정합니다.
                                            "right": {"target_val": -1}})
        # 검증문 `self.assertEqual(engine.get_count_thresholds(), {side: {'start_val': 0, 'target_val': 100} for side in ('left', 'right')})`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.get_count_thresholds(), {
            # 구성 항목: 개인 가동 범위의 시작값(`'start_val'`)에 0; 개인 가동 범위의 목표값(`'target_val'`)에 100을 지정합니다.
            side: {"start_val": 0, "target_val": 100} for side in ("left", "right")})
        # 검증문 `self.assertEqual(engine.threshold_version, 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.threshold_version, 0)
        # 검증문 `self.assertEqual(engine.fsm_left.__dict__, before['fsm_left'].__dict__)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.__dict__, before['fsm_left'].__dict__)

    # 회귀 테스트: `test_invalid_values_shapes_and_direction_are_rejected` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_invalid_values_shapes_and_direction_are_rejected(self):
        # 유효하지 않은에 순서·구성을 지정한 `[None, [], {}, {'both': {'target_val': 50}}, {'left': {}}, {'left': {'typo': 50}}, {'left': None}]`을 저장합니다. 대상: `invalid`.
        invalid = [None, [], {}, {'both': {'target_val': 50}}, {'left': {}},
                   # 구성 항목: 왼쪽(`'left'`)에 키 'typo'로 구성한 사전; 왼쪽(`'left'`)에 None; `typo` 값(`'typo'`)에 50을 지정합니다.
                   {'left': {'typo': 50}}, {'left': None}]
        # 유효하지 않은에 `[{'left': {'target_val': v}} for v in (0, -1, True, '50', None, float('nan'), float('inf'), 10 ** 400)]`을 더해 누적합니다. 대상: `invalid`.
        invalid += [{'left': {'target_val': v}} for v in
                    # 앞줄에서 시작한 식·인수·자료 구조를 `(0, -1, True, '50', None, float('nan'), float('inf'), 10 ** 400)]` 내용으로 이어서 구성합니다.
                    (0, -1, True, '50', None, float('nan'), float('inf'), 10 ** 400)]
        # `invalid`에서 항목을 차례로 꺼내 `patch`에 넣고 아래 처리를 반복합니다.
        for patch in invalid:
            # `self.subTest(patch=patch), self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                # `make_engine().update_count_thresholds(patch)`을 호출하여 해당 단계의 작업을 수행합니다.
                make_engine().update_count_thresholds(patch)
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `make_engine('DECREASING').update_count_thresholds({'right': {'target_val': 190}})`을 호출하여 해당 단계의 작업을 수행합니다.
            make_engine('DECREASING').update_count_thresholds({'right': {'target_val': 190}})
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `make_engine(metric='ANGLE').update_count_thresholds({'left': {'target_val': 181}})`을 호출하여 해당 단계의 작업을 수행합니다.
            make_engine(metric='ANGLE').update_count_thresholds({'left': {'target_val': 181}})
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `make_engine(metric='DISTANCE').update_count_thresholds({'left': {'start_val': -1}})`을 호출하여 해당 단계의 작업을 수행합니다.
            make_engine(metric='DISTANCE').update_count_thresholds({'left': {'start_val': -1}})

    # 회귀 테스트: `test_signed_height_and_partial_updates_are_supported` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_signed_height_and_partial_updates_are_supported(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # `engine.update_count_thresholds({'left': {'start_val': -1, 'target_val': 0}})`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds({'left': {'start_val': -1, 'target_val': 0}})
        # 검증문 `self.assertEqual(engine.get_count_thresholds()['left'], {'start_val': -1, 'target_val': 0})`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.get_count_thresholds()['left'], {'start_val': -1, 'target_val': 0})
        # `engine.update_count_thresholds({'left': {'target_val': -0.5}})`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds({'left': {'target_val': -0.5}})
        # 검증문 `self.assertEqual(engine.fsm_left.start_val, -1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.start_val, -1)

    # 회귀 테스트: `test_identical_request_does_not_reset_current_motion` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_identical_request_does_not_reset_current_motion(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # `patch` 값에 키 'left'로 구성한 사전을 저장합니다. 대상: `patch`.
        patch = {'left': {'target_val': 50}}
        # `engine.update_count_thresholds(patch)`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds(patch)
        # `feed(engine.fsm_left, [0, 20, 40])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [0, 20, 40])
        # 이전에 `copy.deepcopy(engine.fsm_left.__dict__)` 호출 결과를 저장합니다. 대상: `before`.
        before = copy.deepcopy(engine.fsm_left.__dict__)
        # 검증문 `self.assertEqual(engine.update_count_thresholds(patch), [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.update_count_thresholds(patch), [])
        # 검증문 `self.assertEqual(engine.threshold_version, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.threshold_version, 1)
        # 검증문 `self.assertEqual(engine.fsm_left.__dict__, before)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.__dict__, before)

    # 회귀 테스트: `test_main_side_continues_after_legacy_target_and_update` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_main_side_continues_after_legacy_target_and_update(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine(target_reps=1)` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine(target_reps=1)
        # `feed(engine.fsm_left, [0, 20, 80, 60])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [0, 20, 80, 60])
        # `engine.update_count_thresholds({'left': {'target_val': 50}})`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds({'left': {'target_val': 50}})
        # 검증문 `self.assertEqual(engine.fsm_left.state, 'RETURNING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.state, 'RETURNING')
        # `feed(engine.fsm_left, [0, 20, 40, 30])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [0, 20, 40, 30])
        # 검증문 `self.assertEqual(engine.fsm_left.rep_count, 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.rep_count, 2)

    # 회귀 테스트: `test_status_is_a_copy_and_both_sides_share_one_revision` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_status_is_a_copy_and_both_sides_share_one_revision(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # `engine.update_count_thresholds({side: {'target_val': 50} for side in ('left', 'right')})`을 호출하여 해당 단계의 작업을 수행합니다.
        engine.update_count_thresholds({side: {'target_val': 50} for side in ('left', 'right')})
        # 검증문 `self.assertEqual(engine.threshold_version, 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.threshold_version, 1)
        # 상태에 `engine.count_threshold_status()` 호출 결과를 저장합니다. 대상: `status`.
        status = engine.count_threshold_status()
        # `status['thresholds']['left']['target_val']` 값에 999을 저장합니다. 대상: `status['thresholds']['left']['target_val']`.
        status['thresholds']['left']['target_val'] = 999
        # 검증문 `self.assertEqual(engine.fsm_left.target_val, 50)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.target_val, 50)
        # 검증문 `self.assertEqual(engine.fsm_left.threshold_version, engine.fsm_right.threshold_version)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.threshold_version, engine.fsm_right.threshold_version)


# `CountThresholdCommandTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class CountThresholdCommandTests(unittest.TestCase):
    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `self.engine`.
        self.engine = make_engine()
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(motion_engine=self.engine)` 호출 결과를 저장합니다. 대상: `self.controller`.
        self.controller = SessionController(motion_engine=self.engine)

    # 명령 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def command(self, **overrides):
        # 수신 또는 전송할 JSON 메시지 사전에 키 'type', 'exercise_name', 'request_id', 'thresholds'로 구성한 사전을 저장합니다. 대상: `packet`.
        packet = {'type': 'CMD_UPDATE_COUNT_THRESHOLD', 'exercise_name': 'test_motion',
                  # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 'request-1'; 좌우 가동 범위 시작값·목표값(`'thresholds'`)에 키 'left'로 구성한 사전; 왼쪽(`'left'`)에 키 'target_val'로 구성한 사전; 개인 가동 범위의 목표값(`'target_val'`)에 50을 지정합니다.
                  'request_id': 'request-1', 'thresholds': {'left': {'target_val': 50}}}
        # `packet.update(overrides)`을 호출하여 해당 단계의 작업을 수행합니다.
        packet.update(overrides)
        # Unity·서버가 수행하는 것과 같은 JSON 변환을 거쳐 통신 호환성을 검증합니다.
        # 현재 함수 처리를 끝내고 `handle_count_threshold_command(self.controller, json.loads(json.dumps(packet)))` 호출 결과를 호출한 쪽으로 반환합니다.
        return handle_count_threshold_command(self.controller, json.loads(json.dumps(packet)))

    # 회귀 테스트: `test_update_acknowledges_request_and_refreshes_cached_state` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_update_acknowledges_request_and_refreshes_cached_state(self):
        # `feed(self.engine.fsm_left, [0, 20, 70])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(self.engine.fsm_left, [0, 20, 70])
        # `self.controller.last_left_data.update(val=70, state='PUSHING', is_count_updated=True)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.controller.last_left_data.update(val=70, state='PUSHING', is_count_updated=True)
        # `response` 값에 `self.command()` 호출 결과를 저장합니다. 대상: `response`.
        response = self.command()
        # 검증문 `self.assertEqual(response['type'], 'COUNT_THRESHOLD_UPDATED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['type'], 'COUNT_THRESHOLD_UPDATED')
        # 검증문 `self.assertEqual(response['request_id'], 'request-1')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['request_id'], 'request-1')
        # 검증문 `self.assertEqual(response['threshold_version'], 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['threshold_version'], 1)
        # 검증문 `self.assertEqual(response['thresholds']['left']['target_val'], 50)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['thresholds']['left']['target_val'], 50)
        # 검증문 `self.assertEqual(self.controller.last_left_data['state'], 'RETURNING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.controller.last_left_data['state'], 'RETURNING')
        # 검증문 `self.assertFalse(self.controller.last_left_data['is_count_updated'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(self.controller.last_left_data['is_count_updated'])
        # 검증문 `self.assertEqual(self.controller.mode, 'MAIN')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.controller.mode, 'MAIN')

    # 회귀 테스트: `test_query_does_not_modify_motion` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_query_does_not_modify_motion(self):
        # `feed(self.engine.fsm_left, [0, 20, 70])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(self.engine.fsm_left, [0, 20, 70])
        # `response` 값에 `self.command(type='CMD_GET_COUNT_THRESHOLD')` 호출 결과를 저장합니다. 대상: `response`.
        response = self.command(type='CMD_GET_COUNT_THRESHOLD')
        # 검증문 `self.assertEqual(response['type'], 'COUNT_THRESHOLD_STATUS')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['type'], 'COUNT_THRESHOLD_STATUS')
        # 검증문 `self.assertEqual(self.engine.fsm_left.state, 'PUSHING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.engine.fsm_left.state, 'PUSHING')
        # 검증문 `self.assertEqual(self.engine.threshold_version, 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.engine.threshold_version, 0)

    # 회귀 테스트: `test_errors_leave_session_usable` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_errors_leave_session_usable(self):
        # 검증문 `self.assertEqual(self.command(exercise_name='another_game')['code'], 'EXERCISE_MISMATCH')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.command(exercise_name='another_game')['code'], 'EXERCISE_MISMATCH')
        # 검증문 `self.assertEqual(self.command(exercise_name=None)['code'], 'EXERCISE_MISMATCH')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.command(exercise_name=None)['code'], 'EXERCISE_MISMATCH')
        # 검증문 `self.assertEqual(self.command(thresholds={})['code'], 'INVALID_THRESHOLD')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.command(thresholds={})['code'], 'INVALID_THRESHOLD')
        # 검증문 `self.assertEqual(self.command()['type'], 'COUNT_THRESHOLD_UPDATED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.command()['type'], 'COUNT_THRESHOLD_UPDATED')

    # 회귀 테스트: `test_no_session_and_calibration_are_rejected` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_no_session_and_calibration_are_rejected(self):
        # `response` 값에 `handle_count_threshold_command(None, {'type': 'CMD_UPDATE_COUNT_THRESHOLD'})` 호출 결과를 저장합니다. 대상: `response`.
        response = handle_count_threshold_command(None, {'type': 'CMD_UPDATE_COUNT_THRESHOLD'})
        # 검증문 `self.assertEqual(response['code'], 'NO_ACTIVE_SESSION')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['code'], 'NO_ACTIVE_SESSION')
        # 실행 모드에 'CALIBRATION'을 저장합니다. 대상: `self.engine.mode`.
        self.engine.mode = 'CALIBRATION'
        # 검증문 `self.assertEqual(self.command()['code'], 'INVALID_MODE')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.command()['code'], 'INVALID_MODE')
        # 검증문 `self.assertEqual(self.command(type='CMD_GET_COUNT_THRESHOLD')['type'], 'COUNT_THRESHOLD_STATUS')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.command(type='CMD_GET_COUNT_THRESHOLD')['type'], 'COUNT_THRESHOLD_STATUS')

    # 회귀 테스트: `test_console_client_builds_unity_compatible_packets` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_console_client_builds_unity_compatible_packets(self):
        # `response` 값에 `handle_count_threshold_command(self.controller, make_command('both 0 50', 'test_motion'))` 호출 결과를 저장합니다. 대상: `response`.
        response = handle_count_threshold_command(self.controller, make_command('both 0 50', 'test_motion'))
        # 검증문 `self.assertEqual(response['changed_sides'], ['left', 'right'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['changed_sides'], ['left', 'right'])
        # 검증문 `self.assertEqual(self.engine.fsm_right.target_val, 50)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.engine.fsm_right.target_val, 50)
        # `response` 값에 `handle_count_threshold_command(self.controller, make_command('show', 'test_motion'))` 호출 결과를 저장합니다. 대상: `response`.
        response = handle_count_threshold_command(self.controller, make_command('show', 'test_motion'))
        # 검증문 `self.assertEqual(response['type'], 'COUNT_THRESHOLD_STATUS')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['type'], 'COUNT_THRESHOLD_STATUS')
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `make_command('left bad 50', 'test_motion')`을 호출하여 해당 단계의 작업을 수행합니다.
            make_command('left bad 50', 'test_motion')


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
