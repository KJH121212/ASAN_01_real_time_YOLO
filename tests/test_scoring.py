import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.data_manager import DataManager
from core.motion_engine import MotionEngine
from core.session_controller import SessionController
from network.session_commands import get_target_score
from tests.test_count_thresholds import make_engine, feed


class ScoringTests(unittest.TestCase):
    def test_quality_weights_and_score_events_are_awarded_once(self):
        engine = make_engine()
        fsm = engine.fsm_left
        for peak, quality, delta, code in ((40, 'BAD', 0.25, 1), (60, 'GOOD', 0.5, 2), (80, 'PERFECT', 1, 3)):
            result = feed(fsm, [0, 20, peak, peak - 10])[-1]
            self.assertEqual(result['quality'], quality)
            self.assertEqual(result['score_delta'], delta)
            self.assertEqual(result['score'], code)
            self.assertEqual(fsm.completed_reps_history[-1]['score_delta'], delta)
            idle = feed(fsm, [peak - 10, 0])[-1]
            self.assertEqual(idle['score_delta'], 0)
            self.assertEqual(idle['score'], 0)
        self.assertEqual(fsm.rep_count, 3)
        self.assertEqual(fsm.total_score, 1.75)
        self.assertEqual(fsm.update(None)['total_score'], 1.75)

    def test_server_continues_counting_beyond_legacy_score_target(self):
        fsm = make_engine(target_reps=1).fsm_left
        for count in range(1, 9):
            feed(fsm, [0, 20, 40, 30, 0])
            self.assertEqual(fsm.rep_count, count)
            self.assertFalse(fsm.target_reached)
        fsm.replace_thresholds(0, 50, 1)
        self.assertEqual(fsm.state, 'RETURNING')

    def test_rom_update_preserves_score_and_rearms_unfinished_side(self):
        engine = make_engine(target_reps=1)
        feed(engine.fsm_left, [0, 20, 40, 30])
        controller = SessionController(motion_engine=engine)
        controller.update_count_thresholds({'left': {'target_val': 50}})
        self.assertEqual(engine.fsm_left.total_score, 0.25)
        self.assertEqual(engine.fsm_left.state, 'RETURNING')
        self.assertEqual(controller.last_left_data['total_score'], 0.25)
        self.assertEqual(controller.last_left_data['score_delta'], 0)

    def test_calibration_still_finishes_after_three_repetitions(self):
        config = make_engine().config
        engine = MotionEngine(config, mode='CALIBRATION', target_reps=3, target_score=3)
        for _ in range(4):
            feed(engine.fsm_left, [0, 20, 40, 30, 0])
        self.assertEqual(engine.fsm_left.rep_count, 3)
        self.assertEqual(engine.fsm_left.total_score, 0.75)
        self.assertTrue(engine.fsm_left.target_reached)

    def test_controller_never_ends_on_score_or_elapsed_time(self):
        engine = make_engine(target_reps=1)
        engine.config['eval_config']['target_kpt'] = {'left': 9, 'right': 10}
        controller = SessionController(motion_engine=engine)
        controller.normalizer.normalize = lambda points: points
        now = 100.0
        for repetition in range(4):
            for value in (0, 20, 40, 30, 0):
                points = [{'id': 9, 'x': 0, 'y': value, 'score': 1},
                          {'id': 10, 'x': 0, 'y': 0, 'score': 1}]
                left, right, finished = controller.process_main_frame(points, False, now=now)
                self.assertFalse(finished)
                now += 0.1
            if repetition < 3:
                self.assertIsNone(controller.test_finish_timer)
        self.assertEqual(left['rep_count'], 4)
        self.assertEqual(left['total_score'], 1)
        self.assertEqual(right['total_score'], 0)
        self.assertFalse(controller.process_main_frame(points, False, now=now + 100)[2])
        hidden = controller.process_main_frame(None, True, now=now)[0]
        self.assertEqual(hidden['score_delta'], 0)
        self.assertEqual(hidden['score'], 0)

    def test_request_prefers_target_score_and_supports_legacy_target_reps(self):
        self.assertEqual(get_target_score({'target_score': 1.25, 'target_reps': 40}), 1.25)
        self.assertEqual(get_target_score({'target_reps': 40}), 40)
        for invalid in (0, -1, True, float('nan'), float('inf'), '40'):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                get_target_score({'target_score': invalid})

    def test_frame_npz_rep_and_daily_logs_store_score(self):
        import csv
        with tempfile.TemporaryDirectory() as directory:
            manager = DataManager('test_score', log_root=directory)
            manager.begin_session('test_motion', 'MAIN', target_score=1)
            engine = make_engine(target_reps=1)
            log = manager.session_log
            try:
                for index, value in enumerate((0, 20, 60, 50)):
                    left = engine.fsm_left.update(value)
                    log.append_frame(index / 10, index, None, left, {}, 'RELATIVE_Y', engine.get_count_thresholds())
                summary = manager.finish_session(engine.fsm_left.completed_reps_history)
                self.assertEqual(summary['left_reps'], 1)
                self.assertEqual(summary['left_total_score'], 0.5)
                self.assertEqual(summary['target_score'], 1)
                with np.load(summary['skeleton_npz']) as data:
                    np.testing.assert_array_equal(data['total_scores'][:, 0], [0, 0, 0, 0.5])
                    np.testing.assert_array_equal(data['score_deltas'][:, 0], [0, 0, 0, 0.5])
                for key in ('frame_csv', 'reps_csv', 'gamedata_csv'):
                    with Path(summary[key]).open(encoding='utf-8-sig', newline='') as file:
                        row = list(csv.DictReader(file))[-1]
                    field = 'total_score' if key == 'reps_csv' else 'left_total_score'
                    self.assertEqual(float(row[field]), 0.5)
                self.assertEqual(json.loads(Path(summary['manifest_json']).read_text(encoding='utf-8'))['left_total_score'], 0.5)
            finally:
                manager.finish_session([])


if __name__ == '__main__':
    unittest.main()
