# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# `core.data_manager`에서 `DataManager` 기능을 가져옵니다.
from core.data_manager import DataManager
# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `core.session_controller`에서 `SessionController` 기능을 가져옵니다.
from core.session_controller import SessionController
# `network.session_commands`에서 `get_target_score` 기능을 가져옵니다.
from network.session_commands import get_target_score
# `tests.test_count_thresholds`에서 `make_engine`, `feed` 기능을 가져옵니다.
from tests.test_count_thresholds import make_engine, feed


# `ScoringTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ScoringTests(unittest.TestCase):
    # 회귀 테스트: `test_quality_weights_and_score_events_are_awarded_once` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_quality_weights_and_score_events_are_awarded_once(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine()` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine()
        # 한쪽 동작의 반복 판정 상태 기계에 왼쪽 반복 판정 상태 기계(`engine.fsm_left`)을 저장합니다. 대상: `fsm`.
        fsm = engine.fsm_left
        # `((40, 'BAD', 0.25, 1), (60, 'GOOD', 0.5, 2), (80, 'PERFECT', 1, 3))`에서 항목을 차례로 꺼내 `(peak, quality, delta, code)`에 넣고 아래 처리를 반복합니다.
        for peak, quality, delta, code in ((40, 'BAD', 0.25, 1), (60, 'GOOD', 0.5, 2), (80, 'PERFECT', 1, 3)):
            # 함수 또는 처리 단계의 결과에 계산식 `feed(fsm, [0, 20, peak, peak - 10])[-1]`의 결과를 저장합니다. 대상: `result`.
            result = feed(fsm, [0, 20, peak, peak - 10])[-1]
            # 검증문 `self.assertEqual(result['quality'], quality)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['quality'], quality)
            # 검증문 `self.assertEqual(result['score_delta'], delta)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['score_delta'], delta)
            # 검증문 `self.assertEqual(result['score'], code)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['score'], code)
            # 검증문 `self.assertEqual(fsm.completed_reps_history[-1]['score_delta'], delta)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(fsm.completed_reps_history[-1]['score_delta'], delta)
            # `idle` 값에 계산식 `feed(fsm, [peak - 10, 0])[-1]`의 결과를 저장합니다. 대상: `idle`.
            idle = feed(fsm, [peak - 10, 0])[-1]
            # 검증문 `self.assertEqual(idle['score_delta'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(idle['score_delta'], 0)
            # 검증문 `self.assertEqual(idle['score'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(idle['score'], 0)
        # 검증문 `self.assertEqual(fsm.rep_count, 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.rep_count, 3)
        # 검증문 `self.assertEqual(fsm.total_score, 1.75)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.total_score, 1.75)
        # 검증문 `self.assertEqual(fsm.update(None)['total_score'], 1.75)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.update(None)['total_score'], 1.75)

    # 회귀 테스트: `test_server_continues_counting_beyond_legacy_score_target` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_server_continues_counting_beyond_legacy_score_target(self):
        # 한쪽 동작의 반복 판정 상태 기계에 왼쪽 반복 판정 상태 기계(`make_engine(target_reps=1).fsm_left`)을 저장합니다. 대상: `fsm`.
        fsm = make_engine(target_reps=1).fsm_left
        # `range(1, 9)`에서 항목을 차례로 꺼내 `count`에 넣고 아래 처리를 반복합니다.
        for count in range(1, 9):
            # `feed(fsm, [0, 20, 40, 30, 0])`을 호출하여 해당 단계의 작업을 수행합니다.
            feed(fsm, [0, 20, 40, 30, 0])
            # 검증문 `self.assertEqual(fsm.rep_count, count)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(fsm.rep_count, count)
            # 검증문 `self.assertFalse(fsm.target_reached)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertFalse(fsm.target_reached)
        # `fsm.replace_thresholds(0, 50, 1)`을 호출하여 해당 단계의 작업을 수행합니다.
        fsm.replace_thresholds(0, 50, 1)
        # 검증문 `self.assertEqual(fsm.state, 'RETURNING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(fsm.state, 'RETURNING')

    # 회귀 테스트: `test_rom_update_preserves_score_and_rearms_unfinished_side` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_rom_update_preserves_score_and_rearms_unfinished_side(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine(target_reps=1)` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine(target_reps=1)
        # `feed(engine.fsm_left, [0, 20, 40, 30])`을 호출하여 해당 단계의 작업을 수행합니다.
        feed(engine.fsm_left, [0, 20, 40, 30])
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(motion_engine=engine)` 호출 결과를 저장합니다. 대상: `controller`.
        controller = SessionController(motion_engine=engine)
        # `controller.update_count_thresholds({'left': {'target_val': 50}})`을 호출하여 해당 단계의 작업을 수행합니다.
        controller.update_count_thresholds({'left': {'target_val': 50}})
        # 검증문 `self.assertEqual(engine.fsm_left.total_score, 0.25)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.total_score, 0.25)
        # 검증문 `self.assertEqual(engine.fsm_left.state, 'RETURNING')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.state, 'RETURNING')
        # 검증문 `self.assertEqual(controller.last_left_data['total_score'], 0.25)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.last_left_data['total_score'], 0.25)
        # 검증문 `self.assertEqual(controller.last_left_data['score_delta'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(controller.last_left_data['score_delta'], 0)

    # 회귀 테스트: `test_calibration_still_finishes_after_three_repetitions` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_calibration_still_finishes_after_three_repetitions(self):
        # 현재 작업 설정에 현재 작업 설정(`make_engine().config`)을 저장합니다. 대상: `config`.
        config = make_engine().config
        # 생성 또는 검증할 분석 엔진에 `MotionEngine(config, mode='CALIBRATION', target_reps=3, target_score=3)` 호출 결과를 저장합니다. 대상: `engine`.
        engine = MotionEngine(config, mode='CALIBRATION', target_reps=3, target_score=3)
        # `range(4)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(4):
            # `feed(engine.fsm_left, [0, 20, 40, 30, 0])`을 호출하여 해당 단계의 작업을 수행합니다.
            feed(engine.fsm_left, [0, 20, 40, 30, 0])
        # 검증문 `self.assertEqual(engine.fsm_left.rep_count, 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.rep_count, 3)
        # 검증문 `self.assertEqual(engine.fsm_left.total_score, 0.75)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(engine.fsm_left.total_score, 0.75)
        # 검증문 `self.assertTrue(engine.fsm_left.target_reached)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(engine.fsm_left.target_reached)

    # 회귀 테스트: `test_controller_never_ends_on_score_or_elapsed_time` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_controller_never_ends_on_score_or_elapsed_time(self):
        # 생성 또는 검증할 분석 엔진에 `make_engine(target_reps=1)` 호출 결과를 저장합니다. 대상: `engine`.
        engine = make_engine(target_reps=1)
        # `engine.config['eval_config']['target_kpt']` 값에 키 'left', 'right'로 구성한 사전을 저장합니다. 대상: `engine.config['eval_config']['target_kpt']`.
        engine.config['eval_config']['target_kpt'] = {'left': 9, 'right': 10}
        # 캘리브레이션·본 운동 진행 제어기에 `SessionController(motion_engine=engine)` 호출 결과를 저장합니다. 대상: `controller`.
        controller = SessionController(motion_engine=engine)
        # `controller.normalizer.normalize` 값에 계산식 `lambda points: points`의 결과를 저장합니다. 대상: `controller.normalizer.normalize`.
        controller.normalizer.normalize = lambda points: points
        # 이번 처리에 사용할 현재 시각(초)에 100.0을 저장합니다. 대상: `now`.
        now = 100.0
        # `range(4)`에서 항목을 차례로 꺼내 `repetition`에 넣고 아래 처리를 반복합니다.
        for repetition in range(4):
            # `(0, 20, 40, 30, 0)`에서 항목을 차례로 꺼내 `value`에 넣고 아래 처리를 반복합니다.
            for value in (0, 20, 40, 30, 0):
                # 계산에 사용할 관절 또는 좌표 묶음에 순서·구성을 지정한 `[{'id': 9, 'x': 0, 'y': value, 'score': 1}, {'id': 10, 'x': 0, 'y': 0, 'score': 1}]`을 저장합니다. 대상: `points`.
                points = [{'id': 9, 'x': 0, 'y': value, 'score': 1},
                          # 구성 항목: `id` 값(`'id'`)에 10; `x` 값(`'x'`)에 0; `y` 값(`'y'`)에 0; 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 1을 지정합니다.
                          {'id': 10, 'x': 0, 'y': 0, 'score': 1}]
                # `(left, right, finished)` 값에 `controller.process_main_frame(points, False, now=now)` 호출 결과를 저장합니다. 대상: `(left, right, finished)`.
                left, right, finished = controller.process_main_frame(points, False, now=now)
                # 검증문 `self.assertFalse(finished)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertFalse(finished)
                # 이번 처리에 사용할 현재 시각(초)에 `0.1`을 더해 누적합니다. 대상: `now`.
                now += 0.1
            # 조건 `repetition < 3`을 만족하는 경우에만 아래 분기를 실행합니다.
            if repetition < 3:
                # 검증문 `self.assertIsNone(controller.test_finish_timer)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertIsNone(controller.test_finish_timer)
        # 검증문 `self.assertEqual(left['rep_count'], 4)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(left['rep_count'], 4)
        # 검증문 `self.assertEqual(left['total_score'], 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(left['total_score'], 1)
        # 검증문 `self.assertEqual(right['total_score'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(right['total_score'], 0)
        # 검증문 `self.assertFalse(controller.process_main_frame(points, False, now=now + 100)[2])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertFalse(controller.process_main_frame(points, False, now=now + 100)[2])
        # `hidden` 값에 계산식 `controller.process_main_frame(None, True, now=now)[0]`의 결과를 저장합니다. 대상: `hidden`.
        hidden = controller.process_main_frame(None, True, now=now)[0]
        # 검증문 `self.assertEqual(hidden['score_delta'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(hidden['score_delta'], 0)
        # 검증문 `self.assertEqual(hidden['score'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(hidden['score'], 0)

    # 회귀 테스트: `test_request_prefers_target_score_and_supports_legacy_target_reps` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_request_prefers_target_score_and_supports_legacy_target_reps(self):
        # 검증문 `self.assertEqual(get_target_score({'target_score': 1.25, 'target_reps': 40}), 1.25)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(get_target_score({'target_score': 1.25, 'target_reps': 40}), 1.25)
        # 검증문 `self.assertEqual(get_target_score({'target_reps': 40}), 40)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(get_target_score({'target_reps': 40}), 40)
        # `(0, -1, True, float('nan'), float('inf'), '40')`에서 항목을 차례로 꺼내 `invalid`에 넣고 아래 처리를 반복합니다.
        for invalid in (0, -1, True, float('nan'), float('inf'), '40'):
            # `self.subTest(value=invalid), self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                # `get_target_score({'target_score': invalid})`을 호출하여 해당 단계의 작업을 수행합니다.
                get_target_score({'target_score': invalid})

    # 회귀 테스트: `test_frame_npz_rep_and_daily_logs_store_score` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_frame_npz_rep_and_daily_logs_store_score(self):
        # 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
        import csv
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # `manager` 값에 `DataManager('test_score', log_root=directory)` 호출 결과를 저장합니다. 대상: `manager`.
            manager = DataManager('test_score', log_root=directory)
            # `manager.begin_session('test_motion', 'MAIN', target_score=1)`을 호출하여 해당 단계의 작업을 수행합니다.
            manager.begin_session('test_motion', 'MAIN', target_score=1)
            # 생성 또는 검증할 분석 엔진에 `make_engine(target_reps=1)` 호출 결과를 저장합니다. 대상: `engine`.
            engine = make_engine(target_reps=1)
            # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`manager.session_log`)을 저장합니다. 대상: `log`.
            log = manager.session_log
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `enumerate((0, 20, 60, 50))`에서 항목을 차례로 꺼내 `(index, value)`에 넣고 아래 처리를 반복합니다.
                for index, value in enumerate((0, 20, 60, 50)):
                    # 왼쪽에 `engine.fsm_left.update(value)` 호출 결과를 저장합니다. 대상: `left`.
                    left = engine.fsm_left.update(value)
                    # `log.append_frame(index / 10, index, None, left, {}, 'RELATIVE_Y', engine.get_count_thresholds())`을 호출하여 해당 단계의 작업을 수행합니다.
                    log.append_frame(index / 10, index, None, left, {}, 'RELATIVE_Y', engine.get_count_thresholds())
                # 저장·종료 결과 요약에 `manager.finish_session(engine.fsm_left.completed_reps_history)` 호출 결과를 저장합니다. 대상: `summary`.
                summary = manager.finish_session(engine.fsm_left.completed_reps_history)
                # 검증문 `self.assertEqual(summary['left_reps'], 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(summary['left_reps'], 1)
                # 검증문 `self.assertEqual(summary['left_total_score'], 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(summary['left_total_score'], 0.5)
                # 검증문 `self.assertEqual(summary['target_score'], 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(summary['target_score'], 1)
                # `np.load(summary['skeleton_npz'])`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with np.load(summary['skeleton_npz']) as data:
                    # 검증문 `np.testing.assert_array_equal(data['total_scores'][:, 0], [0, 0, 0, 0.5])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    np.testing.assert_array_equal(data['total_scores'][:, 0], [0, 0, 0, 0.5])
                    # 검증문 `np.testing.assert_array_equal(data['score_deltas'][:, 0], [0, 0, 0, 0.5])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    np.testing.assert_array_equal(data['score_deltas'][:, 0], [0, 0, 0, 0.5])
                # `('frame_csv', 'reps_csv', 'gamedata_csv')`에서 항목을 차례로 꺼내 `key`에 넣고 아래 처리를 반복합니다.
                for key in ('frame_csv', 'reps_csv', 'gamedata_csv'):
                    # `Path(summary[key]).open(encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                    with Path(summary[key]).open(encoding='utf-8-sig', newline='') as file:
                        # `row` 값에 계산식 `list(csv.DictReader(file))[-1]`의 결과를 저장합니다. 대상: `row`.
                        row = list(csv.DictReader(file))[-1]
                    # `field` 값에 `key == 'reps_csv'`이면 `'total_score'`, 아니면 `'left_total_score'`을 저장합니다. 대상: `field`.
                    field = 'total_score' if key == 'reps_csv' else 'left_total_score'
                    # 검증문 `self.assertEqual(float(row[field]), 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertEqual(float(row[field]), 0.5)
                # 검증문 `self.assertEqual(json.loads(Path(summary['manifest_json']).read_text(encoding='utf-8'))['left_total_score'], 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(json.loads(Path(summary['manifest_json']).read_text(encoding='utf-8'))['left_total_score'], 0.5)
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # `manager.finish_session([])`을 호출하여 해당 단계의 작업을 수행합니다.
                manager.finish_session([])


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
