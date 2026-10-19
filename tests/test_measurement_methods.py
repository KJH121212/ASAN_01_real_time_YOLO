import unittest

from core.motion_engine import MotionEngine
from core.session_controller import CalibrationCycleDetector, SessionController


class MeasurementMethodsTests(unittest.TestCase):
    def engine(self, metric, **settings):
        return MotionEngine({'eval_config': {'metric_type': metric, **settings},
                             'default_thresholds': {s: {'start_val': 0., 'target_val': 1.}
                                                    for s in ('left', 'right')}})

    def test_four_measurement_functions(self):
        points = {i: dict(id=i, x=x, y=y, score=1.)
                  for i, x, y in [(1, 0., 1.), (2, 0., 0.), (3, 1., 0.)]}
        angle = self.engine('ANGLE', primary_kpts={'left': [1, 2, 3]})
        self.assertAlmostEqual(angle._compute_side_value('left', points), 90.)
        coordinate = self.engine('COORDINATE', target_kpt={'left': 1}, reference_kpt={'left': 3}, axis='x')
        self.assertAlmostEqual(coordinate._compute_side_value('left', points), -1.)
        length = self.engine('LENGTH', primary_kpts={'left': [1, 3]})
        self.assertAlmostEqual(length._compute_side_value('left', points), 2**.5, places=4)

    def ratio_engine(self):
        return self.engine('RATIO', numerator_kpts={'left': [2, 3], 'right': [2, 3]},
                           denominator_kpts={'left': [1, 2], 'right': [1, 2]})

    def test_ratio_cancels_scale_translation_and_rotation(self):
        import math
        engine = self.ratio_engine()
        for scale in (.01, .5, 1., 100.):
            for angle in (0., .7, 2.):
                points = {i: dict(id=i, x=7.-scale*y*math.sin(angle),
                                  y=-3.+scale*y*math.cos(angle), score=1.)
                          for i, y in ((1, 0.), (2, 2.), (3, 5.))}
                for side in ('left', 'right'):
                    self.assertAlmostEqual(engine._compute_side_value(side, points), 1.5)

    def test_ratio_rejects_invalid_joints_and_collapsed_denominator(self):
        from copy import deepcopy
        engine = self.ratio_engine()
        good = {i: dict(id=i, x=0., y=y, score=1.)
                for i, y in ((1, 0.), (2, 2.), (3, 5.))}
        for joint in (1, 2, 3):
            for field, value in (('score', .1), ('x', float('nan')), ('y', float('inf'))):
                points = deepcopy(good)
                points[joint][field] = value
                self.assertIsNone(engine._compute_side_value('left', points))
            points = deepcopy(good)
            del points[joint]
            self.assertIsNone(engine._compute_side_value('left', points))
        for delta in (0., 1e-8):
            points = deepcopy(good)
            points[2]['y'] = delta
            self.assertIsNone(engine._compute_side_value('left', points))

    def test_ratio_rejects_old_coordinate_calibration_and_negative_rom(self):
        engine = self.ratio_engine()
        for old_metric in (None, 'COORDINATE', 'LENGTH'):
            old = {s: dict(start_val=-1., target_val=-.5, metric_type=old_metric)
                   for s in ('left', 'right')}
            with self.assertRaisesRegex(ValueError, 'CALIBRATION'):
                MotionEngine(engine.config, custom_thresholds=old)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            engine.update_count_thresholds({'left': {'start_val': -.1}})

    def test_ratio_requires_two_valid_pairs_for_each_side(self):
        with self.assertRaises(ValueError):
            self.engine('RATIO')

    def test_unknown_metric_is_rejected(self):
        with self.assertRaises(ValueError):
            self.engine('OTHER')

    def test_common_detector_for_all_units_and_both_directions(self):
        for metric, baseline, amplitude, minimum in [('ANGLE', 90., 30., 8.),
                                                    ('COORDINATE', -.8, .2, .08),
                                                    ('LENGTH', 1., .2, .08),
                                                    ('RATIO', 1., .2, .08)]:
            for sign in (-1., 1.):
                detector = CalibrationCycleDetector(metric_type=metric, direction='AUTO', min_rise=minimum,
                                                     baseline_range=minimum*.5)
                now = 0.
                def feed(value, frames):
                    nonlocal now
                    for _ in range(frames):
                        detector.update(value, now)
                        now += 1/30
                feed(baseline, 30)
                for _ in range(3):
                    feed(baseline + sign*amplitude, 12)
                    feed(baseline, 12)
                self.assertTrue(detector.consistent)
                th = detector.thresholds()
                self.assertAlmostEqual(th['target_val'], baseline + sign*.85*amplitude)
                self.assertEqual(th['metric_type'], metric)

    def test_auto_direction_not_committed_by_a_spike(self):
        detector = CalibrationCycleDetector(direction='AUTO')
        for i in range(30):
            detector.update(0., i/30)
        detector.update(-.2, 1.)
        detector.update(0., 1.033)
        self.assertEqual(detector.direction, 'AUTO')
        for i in range(15):
            detector.update(.2, 1.1+i/30)
        for i in range(15):
            detector.update(0., 1.6+i/30)
        self.assertEqual(detector.direction, 'INCREASING')
        self.assertEqual(len(detector.peaks), 1)

    def test_derivative_uses_elapsed_seconds(self):
        detector = CalibrationCycleDetector()
        detector.update(0., 0.)
        detector.update(.1, .2)
        self.assertAlmostEqual(detector.velocity, .5)

    def test_main_restores_each_calibrated_direction(self):
        engine = self.engine('COORDINATE')
        thresholds = {s: dict(start_val=0., target_val=sign*.2, metric_type='COORDINATE',
                             motion_direction='DECREASING' if sign < 0 else 'INCREASING')
                      for s, sign in [('left', -1), ('right', 1)]}
        main = MotionEngine(engine.config, custom_thresholds=thresholds)
        self.assertTrue(main.fsm_left.is_decreasing)
        self.assertFalse(main.fsm_right.is_decreasing)
        main.update_count_thresholds({'left': {'target_val': -.3}})
        with self.assertRaises(ValueError):
            main.update_count_thresholds({'left': {'target_val': .3}})


if __name__ == '__main__':
    unittest.main()
