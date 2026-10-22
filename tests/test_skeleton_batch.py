# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Exercise the real batch methods with deterministic detector/pose outputs."""

# 이 파일에서 사용할 `importlib.util` 모듈을 불러옵니다.
import importlib.util
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys
# `types`에서 `ModuleType`, `SimpleNamespace` 기능을 가져옵니다.
from types import ModuleType, SimpleNamespace
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# `unittest.mock`에서 `Mock`, `patch` 기능을 가져옵니다.
from unittest.mock import Mock, patch

# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np
# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2  # Import once outside the isolated sys.modules patches (native module).


# 최상위 경로에 계산식 `Path(__file__).resolve().parents[1]`의 결과를 저장합니다. 대상: `ROOT`.
ROOT = Path(__file__).resolve().parents[1]


# 불러오기 모델 free module 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def load_model_free_module(relative_path, extra_modules=None):
    # 테스트에서는 실제 GPU 초기화와 전역 패치의 영향을 분리합니다.
    # 테스트할 클래스와 함수 본문은 실제 운영 파일에서 그대로 가져옵니다.
    # `torch` 값에 `ModuleType('torch')` 호출 결과를 저장합니다. 대상: `torch`.
    torch = ModuleType('torch')
    # 불러오기에 계산식 `lambda *args, **kwargs: None`의 결과를 저장합니다. 대상: `torch.load`.
    torch.load = lambda *args, **kwargs: None
    # `torch.no_grad` 값에 계산식 `lambda : lambda method: method`의 결과를 저장합니다. 대상: `torch.no_grad`.
    torch.no_grad = lambda: lambda method: method
    # `serialization` 값에 `ModuleType('torch.serialization')` 호출 결과를 저장합니다. 대상: `serialization`.
    serialization = ModuleType('torch.serialization')
    # `serialization.add_safe_globals` 값에 계산식 `lambda *args: None`의 결과를 저장합니다. 대상: `serialization.add_safe_globals`.
    serialization.add_safe_globals = lambda *args: None
    # 불러오기에 불러오기(`torch.load`)을 저장합니다. 대상: `serialization.load`.
    serialization.load = torch.load
    # `torch.serialization` 값에 `serialization` 값(`serialization`)을 저장합니다. 대상: `torch.serialization`.
    torch.serialization = serialization
    # `inference` 값에 `ModuleType('mmpose.apis.inference')` 호출 결과를 저장합니다. 대상: `inference`.
    inference = ModuleType('mmpose.apis.inference')
    # init 모델에 `Mock()` 호출 결과를 저장합니다. 대상: `inference.init_model`.
    inference.init_model = Mock()
    # `inference.inference_topdown` 값에 `Mock()` 호출 결과를 저장합니다. 대상: `inference.inference_topdown`.
    inference.inference_topdown = Mock()
    # `pose_utils` 값에 `ModuleType('mmpose.utils')` 호출 결과를 저장합니다. 대상: `pose_utils`.
    pose_utils = ModuleType('mmpose.utils')
    # `pose_utils.register_all_modules` 값에 `Mock()` 호출 결과를 저장합니다. 대상: `pose_utils.register_all_modules`.
    pose_utils.register_all_modules = Mock()
    # `dataset` 값에 `ModuleType('mmengine.dataset')` 호출 결과를 저장합니다. 대상: `dataset`.
    dataset = ModuleType('mmengine.dataset')
    # `dataset.Compose` 값에 계산식 `lambda config: lambda data: data`의 결과를 저장합니다. 대상: `dataset.Compose`.
    dataset.Compose = lambda config: lambda data: data
    # `dataset.pseudo_collate` 값에 계산식 `lambda data: data`의 결과를 저장합니다. 대상: `dataset.pseudo_collate`.
    dataset.pseudo_collate = lambda data: data
    # `registry` 값에 `ModuleType('mmengine.registry')` 호출 결과를 저장합니다. 대상: `registry`.
    registry = ModuleType('mmengine.registry')
    # init 기본 scope에 `Mock()` 호출 결과를 저장합니다. 대상: `registry.init_default_scope`.
    registry.init_default_scope = Mock()
    # `ultralytics` 값에 `ModuleType('ultralytics')` 호출 결과를 저장합니다. 대상: `ultralytics`.
    ultralytics = ModuleType('ultralytics')
    # `ultralytics.YOLO` 값에 `Mock()` 호출 결과를 저장합니다. 대상: `ultralytics.YOLO`.
    ultralytics.YOLO = Mock()
    # `modules` 값에 키 'torch', 'torch.serialization', 'pycocotools', 'mmpose.apis.inference', 'mmpose.utils', 'mmengine.dataset', '…로 구성한 사전을 저장합니다. 대상: `modules`.
    modules = {'torch': torch, 'torch.serialization': serialization,
               # 구성 항목: `pycocotools` 값(`'pycocotools'`)에 `ModuleType('pycocotools')` 호출 결과를 지정합니다.
               'pycocotools': ModuleType('pycocotools'),
               # 구성 항목: `mmpose.apis.inference` 값(`'mmpose.apis.inference'`)에 `inference` 값(`inference`); `mmpose.utils` 값(`'mmpose.utils'`)에 `pose_utils` 값(`pose_utils`)을 지정합니다.
               'mmpose.apis.inference': inference, 'mmpose.utils': pose_utils,
               # 구성 항목: `mmengine.dataset` 값(`'mmengine.dataset'`)에 `dataset` 값(`dataset`); `mmengine.registry` 값(`'mmengine.registry'`)에 `registry` 값(`registry`)을 지정합니다.
               'mmengine.dataset': dataset, 'mmengine.registry': registry,
               # 구성 항목: `ultralytics` 값(`'ultralytics'`)에 `ultralytics` 값(`ultralytics`)을 지정합니다.
               'ultralytics': ultralytics}
    # `modules.update(extra_modules or {})`을 호출하여 해당 단계의 작업을 수행합니다.
    modules.update(extra_modules or {})
    # `spec` 값에 `importlib.util.spec_from_file_location('_isolated_test_module', ROOT / relative_path)` 호출 결과를 저장합니다. 대상: `spec`.
    spec = importlib.util.spec_from_file_location('_isolated_test_module', ROOT / relative_path)
    # `module` 값에 `importlib.util.module_from_spec(spec)` 호출 결과를 저장합니다. 대상: `module`.
    module = importlib.util.module_from_spec(spec)
    # `patch.dict(sys.modules, modules)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with patch.dict(sys.modules, modules):
        # `spec.loader.exec_module(module)`을 호출하여 해당 단계의 작업을 수행합니다.
        spec.loader.exec_module(module)
    # 현재 함수 처리를 끝내고 `module` 값(`module`)을 호출한 쪽으로 반환합니다.
    return module


# `detection` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def detection(boxes, scores):
    # `Tensor` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
    class Tensor:
        # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
        def __init__(self, values):
            # 측정값에 `np.asarray(values, dtype=np.float32)` 호출 결과를 저장합니다. 대상: `self.values`.
            self.values = np.asarray(values, dtype=np.float32)

        # `cpu` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def cpu(self):
            # 현재 함수 처리를 끝내고 현재 객체(`self`)을 호출한 쪽으로 반환합니다.
            return self

        # `numpy` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def numpy(self):
            # 현재 함수 처리를 끝내고 측정값(`self.values`)을 호출한 쪽으로 반환합니다.
            return self.values

    # `Boxes` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
    class Boxes:
        # `xyxy` 값에 `Tensor(boxes)` 호출 결과를 저장합니다. 대상: `xyxy`.
        xyxy = Tensor(boxes)
        # `conf` 값에 `Tensor(scores)` 호출 결과를 저장합니다. 대상: `conf`.
        conf = Tensor(scores)

        # `__len__` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def __len__(self):
            # 현재 함수 처리를 끝내고 `len(boxes)` 호출 결과를 호출한 쪽으로 반환합니다.
            return len(boxes)

    # 현재 함수 처리를 끝내고 `SimpleNamespace(boxes=Boxes())` 호출 결과를 호출한 쪽으로 반환합니다.
    return SimpleNamespace(boxes=Boxes())


# `pose` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def pose(x=50, y=25):
    # 현재 함수 처리를 끝내고 `SimpleNamespace(pred_instances=SimpleNamespace(keypoints=np.array([[[x, y]] * 17], dtype=np.float32), keypoint_scores=np.array([[0.9] * 17], dtype=np.float32)))` 호출 결과를 호출한 쪽으로 반환합니다.
    return SimpleNamespace(pred_instances=SimpleNamespace(
        # 구성 항목: COCO 관절 좌표 목록(`keypoints`)에 `np.array([[[x, y]] * 17], dtype=np.float32)` 호출 결과; `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
        keypoints=np.array([[[x, y]] * 17], dtype=np.float32),
        # 구성 항목: `keypoint_scores` 값(`keypoint_scores`)에 `np.array([[0.9] * 17], dtype=np.float32)` 호출 결과; `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
        keypoint_scores=np.array([[0.9] * 17], dtype=np.float32)))


# `SkeletonBatchTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class SkeletonBatchTests(unittest.TestCase):
    # 데코레이터 `classmethod`를 바로 아래 함수·클래스에 적용합니다.
    @classmethod
    # 설정 up class 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def setUpClass(cls):
        # `cls.module` 값에 `load_model_free_module('core/skeleton_engine.py')` 호출 결과를 저장합니다. 대상: `cls.module`.
        cls.module = load_model_free_module('core/skeleton_engine.py')

    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 생성 또는 검증할 분석 엔진에 `self.module.SkeletonEngine.__new__(self.module.SkeletonEngine)` 호출 결과를 저장합니다. 대상: `self.engine`.
        self.engine = self.module.SkeletonEngine.__new__(self.module.SkeletonEngine)
        # `self.engine.device` 값에 'cpu'을 저장합니다. 대상: `self.engine.device`.
        self.engine.device = 'cpu'
        # `self.engine.cached_center_bbox` 값에 `np.array([1, 2, 3, 4])` 호출 결과를 저장합니다. 대상: `self.engine.cached_center_bbox`.
        self.engine.cached_center_bbox = np.array([1, 2, 3, 4])
        # 일괄 처리 pose pipeline에 None을 저장합니다. 대상: `self.engine._batch_pose_pipeline`.
        self.engine._batch_pose_pipeline = None
        # pose 모델에 `Mock()` 호출 결과를 저장합니다. 대상: `self.engine.pose_model`.
        self.engine.pose_model = Mock()
        # `self.engine.pose_model.cfg` 값에 `SimpleNamespace(get=lambda key, default=None: None, test_dataloader=SimpleNamespace(dataset=SimpleNamespace(pipeline=[])))` 호출 결과를 저장합니다. 대상: `self.engine.pose_model.cfg`.
        self.engine.pose_model.cfg = SimpleNamespace(
            # 이 줄의 입력 인수: `key` = `key` 값; `default` = 기본. 기본값·형식은 아래 선언을 따릅니다.
            get=lambda key, default=None: None,
            # 구성 항목: 테스트 데이터 읽기 설정(`test_dataloader`)에 `SimpleNamespace(dataset=SimpleNamespace(pipeline=[]))` 호출 결과; `dataset` 값(`dataset`)에 `SimpleNamespace(pipeline=[])` 호출 결과; `pipeline` 값(`pipeline`)에 순서·구성을 지정한 `[]`을 지정합니다.
            test_dataloader=SimpleNamespace(dataset=SimpleNamespace(pipeline=[])))
        # `self.engine.pose_model.dataset_meta` 값에 키 'dataset_name'로 구성한 사전을 저장합니다. 대상: `self.engine.pose_model.dataset_meta`.
        self.engine.pose_model.dataset_meta = {'dataset_name': 'coco'}
        # `self.engine.pose_model.test_step.side_effect` 값에 계산식 `lambda samples: [pose() for _ in samples]`의 결과를 저장합니다. 대상: `self.engine.pose_model.test_step.side_effect`.
        self.engine.pose_model.test_step.side_effect = lambda samples: [pose() for _ in samples]
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `Mock()` 호출 결과를 저장합니다. 대상: `self.engine.detector`.
        self.engine.detector = Mock()
        # 영상 프레임 묶음에 반복·조건식 `[np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]`으로 구성한 값을 저장합니다. 대상: `self.frames`.
        self.frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(3)]
        # `self.person` 값에 `detection([[50, 10, 150, 90]], [0.9])` 호출 결과를 저장합니다. 대상: `self.person`.
        self.person = detection([[50, 10, 150, 90]], [0.9])

    # 회귀 테스트: `test_detector_and_pose_each_run_once_for_multiple_frames` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_detector_and_pose_each_run_once_for_multiple_frames(self):
        # return 측정값에 계산식 `[self.person] * 3`의 결과를 저장합니다. 대상: `self.engine.detector.return_value`.
        self.engine.detector.return_value = [self.person] * 3
        # 함수 또는 처리 단계의 결과에 `self.engine.extract_keypoints_batch(self.frames)` 호출 결과를 저장합니다. 대상: `result`.
        result = self.engine.extract_keypoints_batch(self.frames)
        # 검증문 `self.assertEqual(len(result), 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(result), 3)
        # 검증문 `self.engine.detector.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.engine.detector.assert_called_once()
        # 검증문 `self.engine.pose_model.test_step.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.engine.pose_model.test_step.assert_called_once()
        # 시각과 측정값의 안정 구간 표본에 계산식 `self.engine.pose_model.test_step.call_args.args[0]`의 결과를 저장합니다. 대상: `samples`.
        samples = self.engine.pose_model.test_step.call_args.args[0]
        # 검증문 `self.assertEqual(len(samples), 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(samples), 3)
        # 검증문 `self.assertEqual(samples[0]['bbox'].shape, (1, 4))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(samples[0]['bbox'].shape, (1, 4))
        # 검증문 `self.assertEqual(result[0][0], {'id': 0, 'x': 0.25, 'y': 0.25, 'score': 0.9, 'image_aspect_ratio': 2.0})`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result[0][0], {'id': 0, 'x': 0.25, 'y': 0.25, 'score': 0.9,
                                      # 구성 항목: 입력 영상의 가로/세로 비율(`'image_aspect_ratio'`)에 2.0을 지정합니다.
                                      'image_aspect_ratio': 2.0})
        # 검증문 `self.assertEqual(len(result[0]), 17)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(result[0]), 17)

    # 회귀 테스트: `test_missing_frames_and_people_keep_original_indices` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_missing_frames_and_people_keep_original_indices(self):
        # return 측정값에 순서·구성을 지정한 `[self.person, detection([], []), self.person]`을 저장합니다. 대상: `self.engine.detector.return_value`.
        self.engine.detector.return_value = [self.person, detection([], []), self.person]
        # return 측정값에 순서·구성을 지정한 `[pose(20, 10), pose(80, 40)]`을 저장합니다. 대상: `self.engine.pose_model.test_step.return_value`.
        self.engine.pose_model.test_step.return_value = [pose(20, 10), pose(80, 40)]
        # `self.engine.pose_model.test_step.side_effect` 값에 None을 저장합니다. 대상: `self.engine.pose_model.test_step.side_effect`.
        self.engine.pose_model.test_step.side_effect = None
        # 함수 또는 처리 단계의 결과에 `self.engine.extract_keypoints_batch([self.frames[0], None, self.frames[1], self.frames[2]])` 호출 결과를 저장합니다. 대상: `result`.
        result = self.engine.extract_keypoints_batch([self.frames[0], None, self.frames[1], self.frames[2]])
        # 검증문 `self.assertEqual(len(result), 4)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(result), 4)
        # 검증문 `self.assertEqual(result[0][0]['x'], 0.1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result[0][0]['x'], 0.1)
        # 검증문 `self.assertIsNone(result[1])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(result[1])
        # 검증문 `self.assertIsNone(result[2])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(result[2])
        # 검증문 `self.assertEqual(result[3][0]['x'], 0.4)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result[3][0]['x'], 0.4)

    # 회귀 테스트: `test_center_person_selected_independently_for_each_frame` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_center_person_selected_independently_for_each_frame(self):
        # return 측정값에 순서·구성을 지정한 `[detection([[0, 0, 20, 20], [80, 10, 120, 90]], [0.99, 0.8])]`을 저장합니다. 대상: `self.engine.detector.return_value`.
        self.engine.detector.return_value = [detection(
            # 앞줄에서 시작한 식·인수·자료 구조를 `[[0, 0, 20, 20], [80, 10, 120, 90]], [0.99, 0.8])]` 내용으로 이어서 구성합니다.
            [[0, 0, 20, 20], [80, 10, 120, 90]], [0.99, 0.8])]
        # `self.engine.extract_keypoints_batch(self.frames[:1])`을 호출하여 해당 단계의 작업을 수행합니다.
        self.engine.extract_keypoints_batch(self.frames[:1])
        # `sample` 값에 계산식 `self.engine.pose_model.test_step.call_args.args[0][0]`의 결과를 저장합니다. 대상: `sample`.
        sample = self.engine.pose_model.test_step.call_args.args[0][0]
        # 검증문 `np.testing.assert_array_equal(sample['bbox'], [[80, 10, 120, 90]])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        np.testing.assert_array_equal(sample['bbox'], [[80, 10, 120, 90]])
        # 검증문 `np.testing.assert_array_equal(self.engine.cached_center_bbox, [1, 2, 3, 4])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        np.testing.assert_array_equal(self.engine.cached_center_bbox, [1, 2, 3, 4])

    # 회귀 테스트: `test_empty_and_missing_only_batches_do_not_call_models` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_empty_and_missing_only_batches_do_not_call_models(self):
        # 검증문 `self.assertEqual(self.engine.extract_keypoints_batch([]), [])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.engine.extract_keypoints_batch([]), [])
        # 검증문 `self.assertEqual(self.engine.extract_keypoints_batch([None, None]), [None, None])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.engine.extract_keypoints_batch([None, None]), [None, None])
        # 검증문 `self.engine.detector.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.engine.detector.assert_not_called()
        # 검증문 `self.engine.pose_model.test_step.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.engine.pose_model.test_step.assert_not_called()

    # 회귀 테스트: `test_no_person_does_not_use_cached_box_or_infer_full_frame` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_no_person_does_not_use_cached_box_or_infer_full_frame(self):
        # return 측정값에 순서·구성을 지정한 `[detection([[0, 0, 100, 100]], [0.3])]`을 저장합니다. 대상: `self.engine.detector.return_value`.
        self.engine.detector.return_value = [detection([[0, 0, 100, 100]], [0.3])]
        # 검증문 `self.assertEqual(self.engine.extract_keypoints_batch(self.frames[:1]), [None])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.engine.extract_keypoints_batch(self.frames[:1]), [None])
        # 검증문 `self.engine.pose_model.test_step.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.engine.pose_model.test_step.assert_not_called()

    # 회귀 테스트: `test_pipeline_rejection_keeps_missing_slot` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_pipeline_rejection_keeps_missing_slot(self):
        # return 측정값에 계산식 `[self.person] * 2`의 결과를 저장합니다. 대상: `self.engine.detector.return_value`.
        self.engine.detector.return_value = [self.person] * 2
        # 일괄 처리 pose pipeline에 `Mock(side_effect=[None, {'valid': True}])` 호출 결과를 저장합니다. 대상: `self.engine._batch_pose_pipeline`.
        self.engine._batch_pose_pipeline = Mock(side_effect=[None, {'valid': True}])
        # 함수 또는 처리 단계의 결과에 `self.engine.extract_keypoints_batch(self.frames[:2])` 호출 결과를 저장합니다. 대상: `result`.
        result = self.engine.extract_keypoints_batch(self.frames[:2])
        # 검증문 `self.assertIsNone(result[0])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(result[0])
        # 검증문 `self.assertIsNotNone(result[1])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNotNone(result[1])
        # 검증문 `self.assertEqual(len(self.engine.pose_model.test_step.call_args.args[0]), 1)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(self.engine.pose_model.test_step.call_args.args[0]), 1)

    # 회귀 테스트: `test_model_length_mismatches_fail_instead_of_shifting_frames` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_model_length_mismatches_fail_instead_of_shifting_frames(self):
        # return 측정값에 순서·구성을 지정한 `[self.person]`을 저장합니다. 대상: `self.engine.detector.return_value`.
        self.engine.detector.return_value = [self.person]
        # `self.assertRaisesRegex(RuntimeError, 'YOLO batch result count')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaisesRegex(RuntimeError, 'YOLO batch result count'):
            # `self.engine.extract_keypoints_batch(self.frames)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.engine.extract_keypoints_batch(self.frames)
        # `self.engine.pose_model.test_step.side_effect` 값에 None을 저장합니다. 대상: `self.engine.pose_model.test_step.side_effect`.
        self.engine.pose_model.test_step.side_effect = None
        # return 측정값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.engine.pose_model.test_step.return_value`.
        self.engine.pose_model.test_step.return_value = []
        # `self.assertRaisesRegex(RuntimeError, 'RTMPose batch result count')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaisesRegex(RuntimeError, 'RTMPose batch result count'):
            # `self.engine.extract_keypoints_batch(self.frames[:1])`을 호출하여 해당 단계의 작업을 수행합니다.
            self.engine.extract_keypoints_batch(self.frames[:1])


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
