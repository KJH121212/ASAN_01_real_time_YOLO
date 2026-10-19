import math
import unittest

from core.motion_engine import SideFSM
from utils.filters import OneEuroFilter, OneEuroKeypointFilter


def joint(joint_id=9, x=0.5, y=0.5, score=0.9):
    return dict(id=joint_id, x=x, y=y, score=score)


class OneEuroFilterTests(unittest.TestCase):
    def test_low_pass_matches_known_step_with_variable_dt(self):
        filt = OneEuroFilter(beta=0)
        expected = 0.0
        self.assertEqual(filt.update(0.0, 0.0), expected)
        previous_time = 0.0
        for timestamp in (0.02, 0.07, 0.10, 0.18):
            r = 2 * math.pi * (timestamp - previous_time)
            expected += r / (1 + r) * (1 - expected)
            self.assertAlmostEqual(filt.update(1.0, timestamp), expected)
            previous_time = timestamp

    def test_static_jitter_is_reduced(self):
        filt = OneEuroFilter()
        raw, smoothed = [], []
        for i in range(240):
            value = 0.5 + (0.01 if i % 2 else -0.01)
            output = filt.update(value, i / 30)
            if i > 30:
                raw.append((value - 0.5) ** 2)
                smoothed.append((output - 0.5) ** 2)
        self.assertLess(sum(smoothed), sum(raw) * 0.1)

    def test_speed_adaptation_reduces_lag(self):
        fixed = OneEuroFilter(beta=0)
        adaptive = OneEuroFilter(beta=1)
        fixed.update(0.0, 0.0)
        adaptive.update(0.0, 0.0)
        self.assertGreater(adaptive.update(1.0, 1 / 30), fixed.update(1.0, 1 / 30))

    def test_discontinuous_timestamps_start_fresh(self):
        filt = OneEuroFilter()
        filt.update(0.0, 0.0)
        self.assertEqual(filt.update(0.7, 1.0), 0.7)
        self.assertEqual(filt.update(0.2, 1.0), 0.2)
        self.assertEqual(filt.update(0.9, 0.0), 0.9)
        filt.reset()
        self.assertEqual(filt.update(0.1, 0.1), 0.1)

    def test_invalid_parameters_and_samples_are_rejected(self):
        for kwargs in ({'min_cutoff': 0}, {'beta': -1}, {'d_cutoff': 0},
                       {'max_gap': 0}, {'beta': float('nan')}):
            with self.assertRaises(ValueError):
                OneEuroFilter(**kwargs)
        with self.assertRaises(ValueError):
            OneEuroFilter().update(float('nan'), 0)

    def test_smoothed_noisy_cycles_count_once_per_repetition(self):
        filt = OneEuroFilter()
        fsm = SideFSM('left', 'INCREASING', {'start_val': 0, 'target_val': 1})
        cycle = ([i / 60 for i in range(61)]
                 + [1 - i / 60 for i in range(1, 61)] + [0] * 30)
        samples = [0] * 30 + cycle * 2
        updates = 0
        for i, value in enumerate(samples):
            noisy = value + (0.005 if i % 2 else -0.005)
            result = fsm.update(filt.update(noisy, i / 30))
            updates += result['is_count_updated']
        self.assertEqual(fsm.rep_count, 2)
        self.assertEqual(updates, 2)


class OneEuroKeypointFilterTests(unittest.TestCase):
    def test_joints_follow_id_and_input_is_preserved(self):
        filt = OneEuroKeypointFilter(beta=0)
        filt.update([joint(9, x=0.1), joint(10, x=0.9)], 0)
        points = [joint(10, x=0.8), joint(9, x=0.2)]
        output = filt.update(points, 1 / 30)
        self.assertGreater(output[0]['x'], 0.8)
        self.assertLess(output[1]['x'], 0.2)
        self.assertEqual(points[0]['x'], 0.8)
        self.assertEqual(output[0]['score'], points[0]['score'])

    def test_low_confidence_and_missing_joints_reset_history(self):
        filt = OneEuroKeypointFilter()
        filt.update([joint(x=0.1)], 0)
        low = joint(x=0.9, score=0.1)
        self.assertEqual(filt.update([low], 1 / 30), [low])
        self.assertEqual(filt.update([joint(x=0.8)], 2 / 30)[0]['x'], 0.8)
        filt.update([joint(10)], 3 / 30)
        self.assertEqual(filt.update([joint(x=0.2)], 4 / 30)[0]['x'], 0.2)
        self.assertIsNone(filt.update(None, 5 / 30))
        self.assertEqual(filt.update([joint(x=0.7)], 6 / 30)[0]['x'], 0.7)
        filt.reset()
        self.assertEqual(filt.update([joint(x=0.3)], 7 / 30)[0]['x'], 0.3)

    def test_nonfinite_joints_are_invalid_and_do_not_poison_recovery(self):
        filt = OneEuroKeypointFilter()
        filt.update([joint()], 0)
        output = filt.update([joint(x=float('nan'))], 1 / 30)[0]
        self.assertEqual(output['score'], 0)
        self.assertTrue(math.isfinite(output['x']))
        self.assertEqual(filt.update([joint(x=0.8)], 2 / 30)[0]['x'], 0.8)


if __name__ == '__main__':
    unittest.main()
