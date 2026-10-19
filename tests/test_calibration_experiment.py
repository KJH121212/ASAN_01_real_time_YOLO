import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tests.evaluate_calibration_videos import CaptureEngine, analyze_frames


class CalibrationExperimentTests(unittest.TestCase):
    def test_eof_is_not_success_even_when_both_counts_are_three(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'frames.csv'
            rows = []
            for index, (step, reps) in enumerate((('COLLECTING', 2), ('COOLDOWN', 3))):
                row = dict(timestamp_ms=index * 100, is_occluded='False', calib_step=step)
                for side in ('left', 'right'):
                    row.update({f'{side}_rep_count': reps, f'{side}_value': 0.5,
                                f'{side}_state': 'WAITING'})
                rows.append(row)
            with path.open('w', encoding='utf-8-sig', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = analyze_frames(path)
            self.assertFalse(result['finished'])
            self.assertEqual(result['left_final_reps'], 3)
            self.assertEqual(result['final_step'], 'COOLDOWN')

    def test_raw_cache_preserves_missing_frames_and_joint_ids(self):
        capture = CaptureEngine(None)
        capture.raw = [None, [{'id': 9, 'x': 0.2, 'y': 0.7, 'score': 0.9}]]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'raw.npz'
            capture.save(path, 30)
            with np.load(path) as data:
                self.assertEqual(data['keypoints'].shape, (2, 17, 3))
                self.assertTrue(np.isnan(data['keypoints'][0, :, :2]).all())
                np.testing.assert_allclose(data['keypoints'][1, 9], [0.2, 0.7, 0.9])
                self.assertEqual(float(data['fps']), 30)


if __name__ == '__main__':
    unittest.main()
