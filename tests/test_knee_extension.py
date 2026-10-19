"""Regression coverage for knee-extension length-ratio calibration and main sessions."""

import contextlib
import io
import json
import math
from pathlib import Path
import unittest

from core.motion_engine import MotionEngine
from core.session_controller import CalibrationCycleDetector
from core.session_controller import SessionController
from utils.normalization import PoseNormalizer


class KneeExtensionTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).resolve().parents[1] / 'configs/exercise_kpt_config_coco.json'
        with path.open(encoding='utf-8') as file:
            self.config = json.load(file)['exercises']['knee_extension']
        self.engine = MotionEngine(self.config, mode='CALIBRATION', target_reps=3)

    def pose(self, scale=1.0):
        points = [{'id': i, 'x': 0.5, 'y': 0.5, 'score': 1.0} for i in range(17)]
        for i, x, y in [(5, .4, .2), (6, .6, .2), (11, .4, .5),
                        (12, .6, .5), (13, .4, .6), (15, .4, .84)]:
            points[i].update(x=x * scale, y=y * scale)
        return points

    def test_length_ratio_is_independent_of_image_scale(self):
        normalizer = PoseNormalizer(invert_y=True)
        for scale in (1.0, 2.0):
            points = normalizer.normalize(self.pose(scale))
            value = self.engine._compute_side_value('left', {p['id']: p for p in points})
            self.assertAlmostEqual(value, 2.4, places=3)

    def test_torso_changes_do_not_change_ratio(self):
        normalizer = PoseNormalizer(invert_y=True)
        for shoulder_y in (.1, .2, .35, .4):
            points = self.pose()
            for i in (5, 6):
                points[i]['y'] = shoulder_y
            normalized = normalizer.normalize(points)
            value = self.engine._compute_side_value('left', {p['id']: p for p in normalized})
            self.assertAlmostEqual(value, 2.4, places=3)

    def test_occlusion_checks_ratio_joints(self):
        controller = SessionController(motion_engine=self.engine)
        for joint in (11, 12, 13, 14, 15, 16):
            points = self.pose()
            points[joint]['score'] = .1
            self.assertTrue(controller.check_occlusion(points))

    def test_missing_ankle_does_not_produce_a_measurement(self):
        points = self.pose()
        points[15]['score'] = .1
        self.assertIsNone(self.engine._compute_side_value('left', {p['id']: p for p in points}))

    def test_legacy_distance_thresholds_require_recalibration(self):
        old = {side: dict(start_val=1., target_val=.5) for side in ('left', 'right')}
        with self.assertRaisesRegex(ValueError, 'CALIBRATION'):
            MotionEngine(self.config, custom_thresholds=old)

    def test_controller_completes_and_personal_thresholds_work_in_main(self):
        controller = SessionController(motion_engine=self.engine)
        now = 0.
        def feed(rise, seconds):
            nonlocal now
            for _ in range(round(seconds * 30)):
                pts = self.pose()
                for knee, ankle, x in ((13, 15, .4), (14, 16, .6)):
                    pts[knee].update(x=x, y=.6)
                    pts[ankle].update(x=x, y=.84 - rise * .3)
                controller.process_calibration_frame(pts, False, now=now)
                now += 1/30
        feed(0, 2.4)
        self.assertEqual(controller.calib_step, 'COLLECTING')
        for _ in range(3):
            feed(.25, .4)
            feed(0, .4)
        feed(0, 1.2)
        self.assertEqual(controller.calib_step, 'FINISHED')
        thresholds = controller.compute_and_save_thresholds('knee_extension')
        main = MotionEngine(self.config, custom_thresholds=thresholds)
        with contextlib.redirect_stdout(io.StringIO()):
            for v in (2.4, 1.65, 1.65, 2.4):
                main.fsm_left.update(v)
        self.assertEqual(main.fsm_left.rep_count, 1)


class AnkleCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.detector = CalibrationCycleDetector()
        self.now = 0.

    def feed(self, value, seconds):
        events = 0
        for _ in range(round(seconds * 30)):
            events += self.detector.update(value, self.now)
            self.now += 1/30
        return events

    def test_personal_rest_and_three_complete_cycles(self):
        for rest in (-1.2, -.5):
            self.detector.reset()
            self.feed(rest, 2.2)
            self.assertTrue(self.detector.ready)
            for _ in range(3):
                self.feed(rest + .2, .4)
                self.feed(rest, .4)
            self.assertTrue(self.detector.consistent)
            self.assertAlmostEqual(self.detector.thresholds()['target_val'], rest + .17)

    def test_stationary_jitter_and_single_spike_do_not_count(self):
        self.feed(-1, 2.2)
        for _ in range(50):
            self.feed(-.99, .1)
            self.feed(-1.01, .1)
        self.feed(-.6, 1/30)
        self.feed(-1, .5)
        self.assertEqual(self.detector.peaks, [])

    def test_ascent_without_return_does_not_count(self):
        self.feed(-1, 2.2)
        self.feed(-.7, 2)
        self.assertEqual(self.detector.peaks, [])

    def test_noise_increases_activation_above_common_floor(self):
        for index in range(80):
            self.feed(-1 + .018 * math.sin(index * 2 * math.pi / 15), 1/30)
        self.assertTrue(self.detector.ready)
        self.assertGreater(self.detector.threshold, self.detector.min_rise)
        self.feed(-.92, .4)
        self.feed(-1, .4)
        self.assertEqual(self.detector.peaks, [])

    def test_time_gap_does_not_complete_an_old_cycle(self):
        self.feed(-1, 2.2)
        self.feed(-.7, .4)
        self.now += 2
        self.feed(-1, .4)
        self.assertEqual(self.detector.peaks, [])

    def test_lower_activation_does_not_tighten_preparation_stability(self):
        detectors = [CalibrationCycleDetector(min_rise=rise) for rise in (.08, .025)]
        for index in range(80):
            value = -1 + .014 * math.sin(index * 2 * math.pi / 15)
            for detector in detectors:
                detector.update(value, index/30)
        self.assertTrue(all(d.ready for d in detectors))
        self.assertEqual(detectors[0].baseline_at, detectors[1].baseline_at)

    def test_moving_preparation_does_not_establish_rest(self):
        for _ in range(30):
            self.feed(-1, .1)
            self.feed(-.7, .1)
        self.assertFalse(self.detector.ready)

    def test_missing_data_cancels_cycle(self):
        self.feed(-1, 2.2)
        self.feed(-.7, .4)
        self.feed(None, .2)
        self.feed(-1, .4)
        self.assertEqual(self.detector.peaks, [])

    def test_inconsistent_repetitions_cannot_save_thresholds(self):
        self.feed(-1, 2.2)
        for amplitude in (.1, .4, .1):
            self.feed(-1 + amplitude, .4)
            self.feed(-1, .4)
        with self.assertRaises(ValueError):
            self.detector.thresholds()


if __name__ == '__main__':
    unittest.main()
