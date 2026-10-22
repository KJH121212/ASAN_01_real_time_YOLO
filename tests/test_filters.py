# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# `core.motion_engine`에서 `SideFSM` 기능을 가져옵니다.
from core.motion_engine import SideFSM
# `utils.filters`에서 `OneEuroFilter`, `OneEuroKeypointFilter` 기능을 가져옵니다.
from utils.filters import OneEuroFilter, OneEuroKeypointFilter


# `joint` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def joint(joint_id=9, x=0.5, y=0.5, score=0.9):
    # 현재 함수 처리를 끝내고 `dict(id=joint_id, x=x, y=y, score=score)` 호출 결과를 호출한 쪽으로 반환합니다.
    return dict(id=joint_id, x=x, y=y, score=score)


# `OneEuroFilterTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class OneEuroFilterTests(unittest.TestCase):
    # 회귀 테스트: `test_low_pass_matches_known_step_with_variable_dt` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_low_pass_matches_known_step_with_variable_dt(self):
        # `filt` 값에 `OneEuroFilter(beta=0)` 호출 결과를 저장합니다. 대상: `filt`.
        filt = OneEuroFilter(beta=0)
        # 테스트에서 기대하는 결과에 0.0을 저장합니다. 대상: `expected`.
        expected = 0.0
        # 검증문 `self.assertEqual(filt.update(0.0, 0.0), expected)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update(0.0, 0.0), expected)
        # previous 시각에 0.0을 저장합니다. 대상: `previous_time`.
        previous_time = 0.0
        # `(0.02, 0.07, 0.1, 0.18)`에서 항목을 차례로 꺼내 `timestamp`에 넣고 아래 처리를 반복합니다.
        for timestamp in (0.02, 0.07, 0.10, 0.18):
            # `r` 값에 계산식 `2 * math.pi * (timestamp - previous_time)`의 결과를 저장합니다. 대상: `r`.
            r = 2 * math.pi * (timestamp - previous_time)
            # 테스트에서 기대하는 결과에 `r / (1 + r) * (1 - expected)`을 더해 누적합니다. 대상: `expected`.
            expected += r / (1 + r) * (1 - expected)
            # 검증문 `self.assertAlmostEqual(filt.update(1.0, timestamp), expected)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertAlmostEqual(filt.update(1.0, timestamp), expected)
            # previous 시각에 입력 표본 또는 세션 기준 시각(초)(`timestamp`)을 저장합니다. 대상: `previous_time`.
            previous_time = timestamp

    # 회귀 테스트: `test_static_jitter_is_reduced` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_static_jitter_is_reduced(self):
        # `filt` 값에 `OneEuroFilter()` 호출 결과를 저장합니다. 대상: `filt`.
        filt = OneEuroFilter()
        # `(raw, smoothed)` 값에 순서·구성을 지정한 `([], [])`을 저장합니다. 대상: `(raw, smoothed)`.
        raw, smoothed = [], []
        # `range(240)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
        for i in range(240):
            # 측정값에 계산식 `0.5 + (0.01 if i % 2 else -0.01)`의 결과를 저장합니다. 대상: `value`.
            value = 0.5 + (0.01 if i % 2 else -0.01)
            # 생성할 결과 데이터 또는 경로에 `filt.update(value, i / 30)` 호출 결과를 저장합니다. 대상: `output`.
            output = filt.update(value, i / 30)
            # 조건 `i > 30`을 만족하는 경우에만 아래 분기를 실행합니다.
            if i > 30:
                # `raw.append((value - 0.5) ** 2)`을 호출하여 항목 하나를 끝에 추가합니다.
                raw.append((value - 0.5) ** 2)
                # `smoothed.append((output - 0.5) ** 2)`을 호출하여 항목 하나를 끝에 추가합니다.
                smoothed.append((output - 0.5) ** 2)
        # 검증문 `self.assertLess(sum(smoothed), sum(raw) * 0.1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertLess(sum(smoothed), sum(raw) * 0.1)

    # 회귀 테스트: `test_speed_adaptation_reduces_lag` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_speed_adaptation_reduces_lag(self):
        # `fixed` 값에 `OneEuroFilter(beta=0)` 호출 결과를 저장합니다. 대상: `fixed`.
        fixed = OneEuroFilter(beta=0)
        # `adaptive` 값에 `OneEuroFilter(beta=1)` 호출 결과를 저장합니다. 대상: `adaptive`.
        adaptive = OneEuroFilter(beta=1)
        # `fixed.update(0.0, 0.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        fixed.update(0.0, 0.0)
        # `adaptive.update(0.0, 0.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        adaptive.update(0.0, 0.0)
        # 검증문 `self.assertGreater(adaptive.update(1.0, 1 / 30), fixed.update(1.0, 1 / 30))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertGreater(adaptive.update(1.0, 1 / 30), fixed.update(1.0, 1 / 30))

    # 회귀 테스트: `test_discontinuous_timestamps_start_fresh` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_discontinuous_timestamps_start_fresh(self):
        # `filt` 값에 `OneEuroFilter()` 호출 결과를 저장합니다. 대상: `filt`.
        filt = OneEuroFilter()
        # `filt.update(0.0, 0.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        filt.update(0.0, 0.0)
        # 검증문 `self.assertEqual(filt.update(0.7, 1.0), 0.7)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update(0.7, 1.0), 0.7)
        # 검증문 `self.assertEqual(filt.update(0.2, 1.0), 0.2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update(0.2, 1.0), 0.2)
        # 검증문 `self.assertEqual(filt.update(0.9, 0.0), 0.9)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update(0.9, 0.0), 0.9)
        # `filt.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
        filt.reset()
        # 검증문 `self.assertEqual(filt.update(0.1, 0.1), 0.1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update(0.1, 0.1), 0.1)

    # 회귀 테스트: `test_invalid_parameters_and_samples_are_rejected` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_invalid_parameters_and_samples_are_rejected(self):
        # `({'min_cutoff': 0}, {'beta': -1}, {'d_cutoff': 0}, {'max_gap': 0}, {'beta': float('nan')})`에서 항목을 차례로 꺼내 `kwargs`에 넣고 아래 처리를 반복합니다.
        for kwargs in ({'min_cutoff': 0}, {'beta': -1}, {'d_cutoff': 0},
                       # 구성 항목: 필터 이력을 이어 사용할 최대 입력 간격(초)(`'max_gap'`)에 0; 움직임 속도에 따른 차단 주파수 증가 계수(`'beta'`)에 `float('nan')` 호출 결과를 지정합니다.
                       {'max_gap': 0}, {'beta': float('nan')}):
            # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.assertRaises(ValueError):
                # `OneEuroFilter(**kwargs)`을 호출하여 해당 단계의 작업을 수행합니다.
                OneEuroFilter(**kwargs)
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `OneEuroFilter().update(float('nan'), 0)`을 호출하여 해당 단계의 작업을 수행합니다.
            OneEuroFilter().update(float('nan'), 0)

    # 회귀 테스트: `test_smoothed_noisy_cycles_count_once_per_repetition` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_smoothed_noisy_cycles_count_once_per_repetition(self):
        # `filt` 값에 `OneEuroFilter()` 호출 결과를 저장합니다. 대상: `filt`.
        filt = OneEuroFilter()
        # 한쪽 동작의 반복 판정 상태 기계에 `SideFSM('left', 'INCREASING', {'start_val': 0, 'target_val': 1})` 호출 결과를 저장합니다. 대상: `fsm`.
        fsm = SideFSM('left', 'INCREASING', {'start_val': 0, 'target_val': 1})
        # `cycle` 값에 계산식 `[i / 60 for i in range(61)] + [1 - i / 60 for i in range(1, 61)] + [0] * 30`의 결과를 저장합니다. 대상: `cycle`.
        cycle = ([i / 60 for i in range(61)]
                 # 앞줄에서 시작한 식·인수·자료 구조를 `+ [1 - i / 60 for i in range(1, 61)] + [0] * 30)` 내용으로 이어서 구성합니다.
                 + [1 - i / 60 for i in range(1, 61)] + [0] * 30)
        # 시각과 측정값의 안정 구간 표본에 계산식 `[0] * 30 + cycle * 2`의 결과를 저장합니다. 대상: `samples`.
        samples = [0] * 30 + cycle * 2
        # `updates` 값에 0을 저장합니다. 대상: `updates`.
        updates = 0
        # `enumerate(samples)`에서 항목을 차례로 꺼내 `(i, value)`에 넣고 아래 처리를 반복합니다.
        for i, value in enumerate(samples):
            # `noisy` 값에 계산식 `value + (0.005 if i % 2 else -0.005)`의 결과를 저장합니다. 대상: `noisy`.
            noisy = value + (0.005 if i % 2 else -0.005)
            # 함수 또는 처리 단계의 결과에 `fsm.update(filt.update(noisy, i / 30))` 호출 결과를 저장합니다. 대상: `result`.
            result = fsm.update(filt.update(noisy, i / 30))
            # `updates` 값에 `result['is_count_updated']`을 더해 누적합니다. 대상: `updates`.
            updates += result['is_count_updated']
        # 검증문 `self.assertEqual(fsm.rep_count, 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.rep_count, 2)
        # 검증문 `self.assertEqual(updates, 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(updates, 2)


# `OneEuroKeypointFilterTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class OneEuroKeypointFilterTests(unittest.TestCase):
    # 회귀 테스트: `test_joints_follow_id_and_input_is_preserved` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_joints_follow_id_and_input_is_preserved(self):
        # `filt` 값에 `OneEuroKeypointFilter(beta=0)` 호출 결과를 저장합니다. 대상: `filt`.
        filt = OneEuroKeypointFilter(beta=0)
        # `filt.update([joint(9, x=0.1), joint(10, x=0.9)], 0)`을 호출하여 해당 단계의 작업을 수행합니다.
        filt.update([joint(9, x=0.1), joint(10, x=0.9)], 0)
        # 계산에 사용할 관절 또는 좌표 묶음에 순서·구성을 지정한 `[joint(10, x=0.8), joint(9, x=0.2)]`을 저장합니다. 대상: `points`.
        points = [joint(10, x=0.8), joint(9, x=0.2)]
        # 생성할 결과 데이터 또는 경로에 `filt.update(points, 1 / 30)` 호출 결과를 저장합니다. 대상: `output`.
        output = filt.update(points, 1 / 30)
        # 검증문 `self.assertGreater(output[0]['x'], 0.8)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertGreater(output[0]['x'], 0.8)
        # 검증문 `self.assertLess(output[1]['x'], 0.2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertLess(output[1]['x'], 0.2)
        # 검증문 `self.assertEqual(points[0]['x'], 0.8)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(points[0]['x'], 0.8)
        # 검증문 `self.assertEqual(output[0]['score'], points[0]['score'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(output[0]['score'], points[0]['score'])

    # 회귀 테스트: `test_low_confidence_and_missing_joints_reset_history` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_low_confidence_and_missing_joints_reset_history(self):
        # `filt` 값에 `OneEuroKeypointFilter()` 호출 결과를 저장합니다. 대상: `filt`.
        filt = OneEuroKeypointFilter()
        # `filt.update([joint(x=0.1)], 0)`을 호출하여 해당 단계의 작업을 수행합니다.
        filt.update([joint(x=0.1)], 0)
        # `low` 값에 `joint(x=0.9, score=0.1)` 호출 결과를 저장합니다. 대상: `low`.
        low = joint(x=0.9, score=0.1)
        # 검증문 `self.assertEqual(filt.update([low], 1 / 30), [low])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update([low], 1 / 30), [low])
        # 검증문 `self.assertEqual(filt.update([joint(x=0.8)], 2 / 30)[0]['x'], 0.8)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update([joint(x=0.8)], 2 / 30)[0]['x'], 0.8)
        # `filt.update([joint(10)], 3 / 30)`을 호출하여 해당 단계의 작업을 수행합니다.
        filt.update([joint(10)], 3 / 30)
        # 검증문 `self.assertEqual(filt.update([joint(x=0.2)], 4 / 30)[0]['x'], 0.2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update([joint(x=0.2)], 4 / 30)[0]['x'], 0.2)
        # 검증문 `self.assertIsNone(filt.update(None, 5 / 30))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(filt.update(None, 5 / 30))
        # 검증문 `self.assertEqual(filt.update([joint(x=0.7)], 6 / 30)[0]['x'], 0.7)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update([joint(x=0.7)], 6 / 30)[0]['x'], 0.7)
        # `filt.reset()`을 호출하여 해당 단계의 작업을 수행합니다.
        filt.reset()
        # 검증문 `self.assertEqual(filt.update([joint(x=0.3)], 7 / 30)[0]['x'], 0.3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update([joint(x=0.3)], 7 / 30)[0]['x'], 0.3)

    # 회귀 테스트: `test_nonfinite_joints_are_invalid_and_do_not_poison_recovery` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_nonfinite_joints_are_invalid_and_do_not_poison_recovery(self):
        # `filt` 값에 `OneEuroKeypointFilter()` 호출 결과를 저장합니다. 대상: `filt`.
        filt = OneEuroKeypointFilter()
        # `filt.update([joint()], 0)`을 호출하여 해당 단계의 작업을 수행합니다.
        filt.update([joint()], 0)
        # 생성할 결과 데이터 또는 경로에 계산식 `filt.update([joint(x=float('nan'))], 1 / 30)[0]`의 결과를 저장합니다. 대상: `output`.
        output = filt.update([joint(x=float('nan'))], 1 / 30)[0]
        # 검증문 `self.assertEqual(output['score'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(output['score'], 0)
        # 검증문 `self.assertTrue(math.isfinite(output['x']))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(math.isfinite(output['x']))
        # 검증문 `self.assertEqual(filt.update([joint(x=0.8)], 2 / 30)[0]['x'], 0.8)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(filt.update([joint(x=0.8)], 2 / 30)[0]['x'], 0.8)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
