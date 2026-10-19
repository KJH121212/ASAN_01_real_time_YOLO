"""Exercise the real batch methods with deterministic detector/pose outputs."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2  # Import once outside the isolated sys.modules patches (native module).


ROOT = Path(__file__).resolve().parents[1]


def load_model_free_module(relative_path, extra_modules=None):
    # Isolate the application's optional GPU bootstrap and its global patches.
    # The class/method bodies still come directly from the production files.
    torch = ModuleType('torch')
    torch.load = lambda *args, **kwargs: None
    torch.no_grad = lambda: lambda method: method
    serialization = ModuleType('torch.serialization')
    serialization.add_safe_globals = lambda *args: None
    serialization.load = torch.load
    torch.serialization = serialization
    inference = ModuleType('mmpose.apis.inference')
    inference.init_model = Mock()
    inference.inference_topdown = Mock()
    pose_utils = ModuleType('mmpose.utils')
    pose_utils.register_all_modules = Mock()
    dataset = ModuleType('mmengine.dataset')
    dataset.Compose = lambda config: lambda data: data
    dataset.pseudo_collate = lambda data: data
    registry = ModuleType('mmengine.registry')
    registry.init_default_scope = Mock()
    ultralytics = ModuleType('ultralytics')
    ultralytics.YOLO = Mock()
    modules = {'torch': torch, 'torch.serialization': serialization,
               'pycocotools': ModuleType('pycocotools'),
               'mmpose.apis.inference': inference, 'mmpose.utils': pose_utils,
               'mmengine.dataset': dataset, 'mmengine.registry': registry,
               'ultralytics': ultralytics}
    modules.update(extra_modules or {})
    spec = importlib.util.spec_from_file_location('_isolated_test_module', ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


def detection(boxes, scores):
    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values, dtype=np.float32)

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Boxes:
        xyxy = Tensor(boxes)
        conf = Tensor(scores)

        def __len__(self):
            return len(boxes)

    return SimpleNamespace(boxes=Boxes())


def pose(x=50, y=25):
    return SimpleNamespace(pred_instances=SimpleNamespace(
        keypoints=np.array([[[x, y]] * 17], dtype=np.float32),
        keypoint_scores=np.array([[0.9] * 17], dtype=np.float32)))


class SkeletonBatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_model_free_module('core/skeleton_engine.py')

    def setUp(self):
        self.engine = self.module.SkeletonEngine.__new__(self.module.SkeletonEngine)
        self.engine.device = 'cpu'
        self.engine.cached_center_bbox = np.array([1, 2, 3, 4])
        self.engine._batch_pose_pipeline = None
        self.engine.pose_model = Mock()
        self.engine.pose_model.cfg = SimpleNamespace(
            get=lambda key, default=None: None,
            test_dataloader=SimpleNamespace(dataset=SimpleNamespace(pipeline=[])))
        self.engine.pose_model.dataset_meta = {'dataset_name': 'coco'}
        self.engine.pose_model.test_step.side_effect = lambda samples: [pose() for _ in samples]
        self.engine.detector = Mock()
        self.frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
        self.person = detection([[50, 10, 150, 90]], [0.9])

    def test_detector_and_pose_each_run_once_for_multiple_frames(self):
        self.engine.detector.return_value = [self.person] * 3
        result = self.engine.extract_keypoints_batch(self.frames)
        self.assertEqual(len(result), 3)
        self.engine.detector.assert_called_once()
        self.engine.pose_model.test_step.assert_called_once()
        samples = self.engine.pose_model.test_step.call_args.args[0]
        self.assertEqual(len(samples), 3)
        self.assertEqual(samples[0]['bbox'].shape, (1, 4))
        self.assertEqual(result[0][0], {'id': 0, 'x': 0.25, 'y': 0.25, 'score': 0.9})
        self.assertEqual(len(result[0]), 17)

    def test_missing_frames_and_people_keep_original_indices(self):
        self.engine.detector.return_value = [self.person, detection([], []), self.person]
        self.engine.pose_model.test_step.return_value = [pose(20, 10), pose(80, 40)]
        self.engine.pose_model.test_step.side_effect = None
        result = self.engine.extract_keypoints_batch([self.frames[0], None, self.frames[1], self.frames[2]])
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0][0]['x'], 0.1)
        self.assertIsNone(result[1])
        self.assertIsNone(result[2])
        self.assertEqual(result[3][0]['x'], 0.4)

    def test_center_person_selected_independently_for_each_frame(self):
        self.engine.detector.return_value = [detection(
            [[0, 0, 20, 20], [80, 10, 120, 90]], [0.99, 0.8])]
        self.engine.extract_keypoints_batch(self.frames[:1])
        sample = self.engine.pose_model.test_step.call_args.args[0][0]
        np.testing.assert_array_equal(sample['bbox'], [[80, 10, 120, 90]])
        np.testing.assert_array_equal(self.engine.cached_center_bbox, [1, 2, 3, 4])

    def test_empty_and_missing_only_batches_do_not_call_models(self):
        self.assertEqual(self.engine.extract_keypoints_batch([]), [])
        self.assertEqual(self.engine.extract_keypoints_batch([None, None]), [None, None])
        self.engine.detector.assert_not_called()
        self.engine.pose_model.test_step.assert_not_called()

    def test_no_person_does_not_use_cached_box_or_infer_full_frame(self):
        self.engine.detector.return_value = [detection([[0, 0, 100, 100]], [0.3])]
        self.assertEqual(self.engine.extract_keypoints_batch(self.frames[:1]), [None])
        self.engine.pose_model.test_step.assert_not_called()

    def test_pipeline_rejection_keeps_missing_slot(self):
        self.engine.detector.return_value = [self.person] * 2
        self.engine._batch_pose_pipeline = Mock(side_effect=[None, {'valid': True}])
        result = self.engine.extract_keypoints_batch(self.frames[:2])
        self.assertIsNone(result[0])
        self.assertIsNotNone(result[1])
        self.assertEqual(len(self.engine.pose_model.test_step.call_args.args[0]), 1)

    def test_model_length_mismatches_fail_instead_of_shifting_frames(self):
        self.engine.detector.return_value = [self.person]
        with self.assertRaisesRegex(RuntimeError, 'YOLO batch result count'):
            self.engine.extract_keypoints_batch(self.frames)
        self.engine.pose_model.test_step.side_effect = None
        self.engine.pose_model.test_step.return_value = []
        with self.assertRaisesRegex(RuntimeError, 'RTMPose batch result count'):
            self.engine.extract_keypoints_batch(self.frames[:1])


if __name__ == '__main__':
    unittest.main()
