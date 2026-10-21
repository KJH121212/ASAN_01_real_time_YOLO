import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from core.data_manager import DataManager
from core.motion_engine import MotionEngine
from utils.image_geometry import letterbox_preview, resize_within
from utils.normalization import PoseNormalizer


def pose(width, height):
    cx, cy = width / 2, height / 2
    pixels = {5: (-40, -100), 6: (40, -100), 11: (-30, 0), 12: (30, 0),
              7: (-100, -50), 9: (-20, -10)}
    return [dict(id=i, x=(cx + pixels.get(i, (0, 0))[0]) / width,
                 y=(cy + pixels.get(i, (0, 0))[1]) / height, score=1.,
                 image_aspect_ratio=width / height) for i in range(17)]


class PortraitInputTests(unittest.TestCase):
    def test_same_pixel_geometry_has_same_normalized_pose_and_angle_in_all_aspects(self):
        normalizer = PoseNormalizer(invert_y=True)
        engine = MotionEngine({'eval_config': {'metric_type': 'ANGLE',
                             'primary_kpts': {'left': [5, 7, 9]}}})
        reference = None
        for width, height in ((1920, 1080), (1080, 1920), (640, 640)):
            points = pose(width, height)
            original = copy.deepcopy(points)
            normalized = normalizer.normalize(points)
            self.assertEqual(points, original)
            matrix = np.asarray([[p['x'], p['y']] for p in normalized])
            value = engine._compute_side_value('left', {p['id']: p for p in normalized})
            if reference is None:
                reference = matrix, value
            else:
                np.testing.assert_allclose(matrix, reference[0], atol=0.0002)
                self.assertAlmostEqual(value, reference[1], delta=0.03)

    def test_portrait_preview_keeps_proportions_and_moves_skeleton_with_image(self):
        frame = np.full((1920, 1080, 3), 255, np.uint8)
        self.assertEqual(resize_within(frame, 400, 300).shape[:2], (300, 169))
        points = [dict(id=5, x=0, y=0, score=1)]
        canvas, remapped = letterbox_preview(frame, points)
        self.assertEqual(canvas.shape, (480, 640, 3))
        self.assertEqual(int(canvas[:, :185].max()), 0)
        self.assertEqual(int(canvas[:, 185:455].min()), 255)
        self.assertAlmostEqual(remapped[0]['x'] * 640, 185)
        self.assertEqual(points[0]['x'], 0)

    def test_npz_retains_aspect_ratio_needed_to_reconstruct_measurements(self):
        with tempfile.TemporaryDirectory() as folder:
            manager = DataManager('portrait', log_root=folder)
            info = manager.begin_session('shoulder_press', 'MAIN')
            log = manager.session_log
            thresholds = {side: {'start_val': 45, 'target_val': 160} for side in ('left', 'right')}
            try:
                log.append_frame(0, 0, pose(1080, 1920), {}, {}, 'ANGLE', thresholds)
                manager.finish_session([])
                with np.load(info['skeleton_npz']) as data:
                    self.assertAlmostEqual(float(data['image_aspect_ratio'][0]), 1080 / 1920)
                    self.assertEqual(data['keypoints'].shape, (1, 17, 3))
            finally:
                manager.finish_session([])


if __name__ == '__main__':
    unittest.main()
