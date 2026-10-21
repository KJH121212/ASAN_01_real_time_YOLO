import math
import unittest
from unittest.mock import Mock

from core.motion_engine import MotionEngine
from core.session_controller import CalibrationCycleDetector, SessionController


class Signal:
    def __init__(self, detector, fps=10):
        self.detector, self.now, self.fps = detector, 0., fps

    def hold(self, value, seconds):
        for _ in range(math.ceil(seconds * self.fps)):
            self.detector.update(value, self.now)
            self.now += 1 / self.fps


class CalibrationRobustnessTests(unittest.TestCase):
    def test_calibrates_at_low_high_and_variable_fps_in_both_directions(self):
        for fps in (2, 5, 10, 15, 30):
            for sign in (-1, 1):
                with self.subTest(fps=fps, sign=sign):
                    detector = CalibrationCycleDetector(direction='AUTO')
                    signal = Signal(detector, fps)
                    signal.hold(0., max(1., 3 / fps))
                    self.assertTrue(detector.ready)
                    for _ in range(3):
                        signal.hold(sign * .2, .6)
                        signal.hold(sign * .04, .6)
                    self.assertTrue(detector.consistent)
                    self.assertEqual(len(detector.peaks), 3)
                    self.assertAlmostEqual(detector.thresholds()['target_val'], sign * .17)
        detector = CalibrationCycleDetector()
        now = 0.
        for _ in range(15):
            detector.update(0., now)
            now += (.08, .15, .2)[_ % 3]
        self.assertTrue(detector.ready)
        for _ in range(3):
            for value in (.2,) * 6 + (.04,) * 6:
                detector.update(value, now)
                now += (.08, .15, .2)[_ % 3]
        self.assertTrue(detector.consistent)

    def test_baseline_requires_elapsed_rest_and_multiple_samples(self):
        detector = CalibrationCycleDetector()
        detector.update(0, 0)
        detector.update(0, .5)
        self.assertFalse(detector.ready)
        detector.update(0, 1.)
        self.assertTrue(detector.ready)
        moving = CalibrationCycleDetector()
        for index in range(100):
            moving.update(index * .1, index / 10)
        self.assertFalse(moving.ready)

    def test_return_tolerance_does_not_count_rest_plateau_again(self):
        detector = CalibrationCycleDetector(metric_type='ANGLE', min_rise=8., baseline_range=4.)
        signal = Signal(detector)
        signal.hold(90, 1)
        for count in range(1, 4):
            signal.hold(120, .6)
            signal.hold(96, .6)
            self.assertEqual(len(detector.peaks), count)
            signal.hold(96, 2)
            self.assertEqual(len(detector.peaks), count)
        self.assertTrue(detector.consistent)

    def test_short_loss_preserves_completed_cycles_but_long_loss_restarts(self):
        detector = CalibrationCycleDetector()
        signal = Signal(detector)
        signal.hold(0, 1)
        signal.hold(.2, .6)
        signal.hold(0, .6)
        self.assertEqual(len(detector.peaks), 1)
        signal.hold(None, .3)
        self.assertTrue(detector.ready)
        self.assertEqual(len(detector.peaks), 1)
        signal.hold(0, .3)
        signal.hold(None, 2.3)
        self.assertFalse(detector.ready)
        self.assertEqual(len(detector.peaks), 0)

    def test_spikes_and_interrupted_holds_are_not_counted(self):
        detector = CalibrationCycleDetector()
        signal = Signal(detector)
        signal.hold(0, 1)
        signal.hold(.2, .1)
        signal.hold(0, .5)
        self.assertEqual(detector.peaks, [])
        signal.hold(.2, .1)
        signal.hold(None, .1)
        signal.hold(0, .5)
        self.assertEqual(detector.peaks, [])

    def test_inconsistent_cycles_require_more_consistent_motion(self):
        detector = CalibrationCycleDetector()
        signal = Signal(detector)
        signal.hold(0, 1)
        for peak in (.09, .4, .4):
            signal.hold(peak, .6)
            signal.hold(0, .6)
        self.assertFalse(detector.consistent)
        self.assertEqual(detector.status()['state'], 'NEED_CONSISTENT_CYCLES')
        signal.hold(.4, .6)
        signal.hold(0, .6)
        self.assertTrue(detector.consistent)

    def test_controller_finishes_and_saves_after_short_occlusion(self):
        config = {'eval_config': {'metric_type': 'COORDINATE', 'motion_direction': 'INCREASING',
                                  'axis': 'y', 'target_kpt': {'left': 9, 'right': 10}},
                  'default_thresholds': {side: {'start_val': 0., 'target_val': 1.} for side in ('left', 'right')}}
        manager = Mock()
        controller = SessionController(manager, MotionEngine(config, mode='CALIBRATION'))
        controller.normalizer.normalize = lambda points: points
        now = 0.

        def hold(value, frames):
            nonlocal now
            for _ in range(frames):
                points = [dict(id=i, x=0., y=value if i in (9, 10) else 0., score=1.) for i in range(17)]
                controller.process_calibration_frame(None if value is None else points, value is None, now=now)
                now += .1

        hold(0, 10)
        hold(.2, 6)
        hold(.04, 6)
        hold(None, 2)
        self.assertEqual(controller.calib_reps_left, 1)
        self.assertEqual(controller.calib_step, 'PAUSED')
        hold(0, 4)
        for _ in range(2):
            hold(.2, 6)
            hold(.04, 6)
        hold(.04, 15)
        self.assertEqual(controller.calib_step, 'FINISHED')
        self.assertEqual(controller.calib_reps_right, 3)
        self.assertEqual(len(controller.motion_engine.fsm_left.completed_reps_history), 3)
        thresholds = controller.compute_and_save_thresholds('test')
        self.assertAlmostEqual(thresholds['left']['target_val'], .17)
        manager.save_custom_threshold.assert_called_once()


if __name__ == '__main__':
    unittest.main()
