# 한국어 코드 해설: YOLO로 대상자를 찾고 RTMPose로 COCO 17개 관절을 추출합니다. 단일 프레임과 영상 배치 추론을 지원합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [파일 정보]
# 파일명: core/skeleton_engine.py
# 설명: YOLOv8 및 RTMPose 기반 실시간 관절 포인트 추출 엔진 (mmcv 에러 완벽 우회 적용)
# ==============================================================================

# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os  # 운영체제 환경 변수 설정을 위한 내장 모듈 로드
# environ['mmcv 포함 ops']에 '0'을 저장합니다. 대상: `os.environ['MMCV_WITH_OPS']`.
os.environ["MMCV_WITH_OPS"] = "0"  # mmcv의 C++ 확장 모듈 로드를 원천적으로 차단

# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys  # 파이썬 런타임 환경 제어를 위한 내장 모듈 로드
# `unittest.mock`에서 `MagicMock` 기능을 가져옵니다.
from unittest.mock import MagicMock  # 존재하지 않는 모듈을 가짜로 만들기 위한 모킹 클래스 로드

# `mock_ext` 값에 `MagicMock()` 호출 결과를 저장합니다. 대상: `mock_ext`.
mock_ext = MagicMock()  # 호출되어도 에러를 발생시키지 않는 가짜 객체 생성
# `mock_ext.__spec__` 값에 `MagicMock()` 호출 결과를 저장합니다. 대상: `mock_ext.__spec__`.
mock_ext.__spec__ = MagicMock()  # 임포트 시스템의 스펙 검사 우회를 위한 속성 추가
# `sys.modules['mmcv._ext']` 값에 `mock_ext` 값(`mock_ext`)을 저장합니다. 대상: `sys.modules['mmcv._ext']`.
sys.modules['mmcv._ext'] = mock_ext  # 문제가 되는 mmcv C++ 확장 모듈을 가짜 객체로 대체
# `sys.modules['mmcv.ops.multi_scale_deform_attn']` 값에 `mock_ext` 값(`mock_ext`)을 저장합니다. 대상: `sys.modules['mmcv.ops.multi_scale_deform_attn']`.
sys.modules['mmcv.ops.multi_scale_deform_attn'] = mock_ext  # 트랜스포머 어텐션 모듈 임포트 에러 우회
# `sys.modules['mmcv.ops.active_rotated_filter']` 값에 `mock_ext` 값(`mock_ext`)을 저장합니다. 대상: `sys.modules['mmcv.ops.active_rotated_filter']`.
sys.modules['mmcv.ops.active_rotated_filter'] = mock_ext  # 로테이티드 필터 모듈 임포트 에러 우회

# 이 파일에서 사용할 `pycocotools` 모듈을 불러옵니다.
import pycocotools  # mmpose 내부의 xtcocotools 종속성 에러를 방지하기 위한 코코툴스 로드
# `sys.modules['xtcocotools']` 값에 `pycocotools` 값(`pycocotools`)을 저장합니다. 대상: `sys.modules['xtcocotools']`.
sys.modules['xtcocotools'] = pycocotools  # xtcocotools를 pycocotools로 매핑하여 강제 로드

# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path  # 경로 탐색을 위한 모듈 로드
# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2  # 이미지 처리를 위한 OpenCV 라이브러리 로드
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np  # 배열 연산을 위한 NumPy 라이브러리 로드
# 이 파일에서 사용할 `torch` 모듈을 불러옵니다.
import torch  # 딥러닝 모델 구동을 위한 PyTorch 로드

# PyTorch 보안 정책(weights_only=True) 우회 패치
# original torch 불러오기에 불러오기(`torch.load`)을 저장합니다. 대상: `_original_torch_load`.
_original_torch_load = torch.load  # 원본 로드 함수 백업

# patched torch 불러오기 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def _patched_torch_load(*args, **kwargs):  # 경고를 끄기 위한 래퍼 함수 정의
    # `kwargs['weights_only']` 값에 False을 저장합니다. 대상: `kwargs['weights_only']`.
    kwargs['weights_only'] = False  # 보안 로드 기능을 강제로 해제
    # 현재 함수 처리를 끝내고 `_original_torch_load(*args, **kwargs)` 호출 결과를 호출한 쪽으로 반환합니다.
    return _original_torch_load(*args, **kwargs)  # 패치된 상태로 원본 함수 실행

# 불러오기에 patched torch 불러오기(`_patched_torch_load`)을 저장합니다. 대상: `torch.load`.
torch.load = _patched_torch_load  # 전역 패치 적용

# `mmpose.apis.inference`에서 `init_model`, `inference_topdown` 기능을 가져옵니다.
from mmpose.apis.inference import init_model, inference_topdown  # MMPose 추론 API 로드
# `mmpose.utils`에서 `register_all_modules` 기능을 가져옵니다.
from mmpose.utils import register_all_modules  # 커스텀 모듈 레지스트리 등록 함수 로드
# `mmengine.dataset`에서 `Compose`, `pseudo_collate` 기능을 가져옵니다.
from mmengine.dataset import Compose, pseudo_collate
# `mmengine.registry`에서 `init_default_scope` 기능을 가져옵니다.
from mmengine.registry import init_default_scope
# `ultralytics`에서 `YOLO` 기능을 가져옵니다.
from ultralytics import YOLO  # 사람 객체 탐지를 위한 YOLO 로드

# `register_all_modules()`을 호출하여 해당 단계의 작업을 수행합니다.
register_all_modules()  # MMPose 모듈 안전 등록


# `SkeletonEngine` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class SkeletonEngine:  # 실시간 관절 포인트 추출을 전담하는 메인 AI 엔진 클래스 선언

    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, device: str = None, yolo_interval: int = 5, on_progress=None):
        # `report` 값에 계산식 `on_progress or (lambda phase: None)`의 결과를 저장합니다. 대상: `report`.
        report = on_progress or (lambda phase: None)
        # `report('DEVICE_SETUP')`을 호출하여 해당 단계의 작업을 수행합니다.
        report('DEVICE_SETUP')
        # 조건 `device is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if device is None:  # 명시적으로 디바이스가 전달되지 않은 경우
            # `self.device` 값에 `torch.cuda.is_available()`이면 `'cuda:0'`, 아니면 `'cpu'`을 저장합니다. 대상: `self.device`.
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # GPU 가용성을 확인하여 자동으로 디바이스 할당
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:  # 디바이스가 외부에서 명시된 경우
            # `self.device` 값에 `device` 값(`device`)을 저장합니다. 대상: `self.device`.
            self.device = device  # 지정된 연산 장치로 확정 적용

        # 최상위 경로 dir에 `Path(__file__).resolve().parent.parent` 값(`Path(__file__).resolve().parent.parent`)을 저장합니다. 대상: `root_dir`.
        root_dir = Path(__file__).resolve().parent.parent  # 현재 파일 기준 프로젝트 최상위 루트 디렉터리 경로 탐색
        # yolo 경로에 `str(root_dir / 'configs' / 'checkpoints' / 'yolov8n.pt')` 호출 결과를 저장합니다. 대상: `yolo_path`.
        yolo_path = str(root_dir / "configs" / "checkpoints" / "yolov8n.pt")  # YOLOv8 가중치 파일의 절대 경로 조립
        # 설정 경로에 `str(root_dir / 'configs' / 'checkpoints' / 'RTMPose_config.py')` 호출 결과를 저장합니다. 대상: `config_path`.
        config_path = str(root_dir / "configs" / "checkpoints" / "RTMPose_config.py")  # RTMPose 모델 구조 설정 파일의 절대 경로 조립
        # checkpoint 경로에 `str(root_dir / 'configs' / 'checkpoints' / 'RTMPose.pth')` 호출 결과를 저장합니다. 대상: `checkpoint_path`.
        checkpoint_path = str(root_dir / "configs" / "checkpoints" / "RTMPose.pth")  # RTMPose 학습 가중치 파일의 절대 경로 조립

        # `report('LOADING_YOLO')`을 호출하여 해당 단계의 작업을 수행합니다.
        report('LOADING_YOLO')
        # 사람 검출 모델 또는 단일 동작 판정기(사용 위치에 따름)에 `YOLO(yolo_path)` 호출 결과를 저장합니다. 대상: `self.detector`.
        self.detector = YOLO(yolo_path)  # 지정된 경로의 가중치로 YOLOv8 사람 탐지기 객체 생성 및 메모리 로드
        # `report('LOADING_POSE')`을 호출하여 해당 단계의 작업을 수행합니다.
        report('LOADING_POSE')
        # pose 모델에 `init_model(config_path, checkpoint_path, device=self.device)` 호출 결과를 저장합니다. 대상: `self.pose_model`.
        self.pose_model = init_model(config_path, checkpoint_path, device=self.device)  # 지정된 디바이스 메모리에 RTMPose 모델 초기화
        
        # `self.yolo_interval` 값에 `yolo_interval` 값(`yolo_interval`)을 저장합니다. 대상: `self.yolo_interval`.
        self.yolo_interval = yolo_interval  # YOLO 추론을 건너뛰어 성능을 최적화할 프레임 간격 저장
        # 프레임 횟수에 0을 저장합니다. 대상: `self.frame_count`.
        self.frame_count = 0  # 시스템에서 누적 처리된 프레임 수를 기록할 카운터 초기화
        # `self.cached_center_bbox` 값에 None을 저장합니다. 대상: `self.cached_center_bbox`.
        self.cached_center_bbox = None  # 직전 프레임에서 확정된 대상자 바운딩 박스 영역 캐시 초기화
        # 일괄 처리 pose pipeline에 None을 저장합니다. 대상: `self._batch_pose_pipeline`.
        self._batch_pose_pipeline = None

    # 데코레이터 `staticmethod`를 바로 아래 함수·클래스에 적용합니다.
    @staticmethod
    # `_select_center_bbox` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _select_center_bbox(detection, frame):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Select one confident person per frame, without borrowing another frame's bbox."""
        # 조건 `detection.boxes is None or len(detection.boxes) == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if detection.boxes is None or len(detection.boxes) == 0:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # `boxes` 값에 `detection.boxes.xyxy.cpu().numpy()` 호출 결과를 저장합니다. 대상: `boxes`.
        boxes = detection.boxes.xyxy.cpu().numpy()
        # 관절 신뢰도 배열 또는 좌우 게임 점수(사용 위치에 따름)에 `detection.boxes.conf.cpu().numpy()` 호출 결과를 저장합니다. 대상: `scores`.
        scores = detection.boxes.conf.cpu().numpy()
        # `(h, w)` 값에 계산식 `frame.shape[:2]`의 결과를 저장합니다. 대상: `(h, w)`.
        h, w = frame.shape[:2]
        # `candidates` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `candidates`.
        candidates = []
        # `zip(boxes, scores)`에서 항목을 차례로 꺼내 `(box, score)`에 넣고 아래 처리를 반복합니다.
        for box, score in zip(boxes, scores):
            # 조건 `score <= 0.4 or not np.isfinite(score) or (not np.all(np.isfinite(box)))`을 만족하는 경우에만 아래 분기를 실행합니다.
            if score <= 0.4 or not np.isfinite(score) or not np.all(np.isfinite(box)):
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # 조건 `box[2] <= box[0] or box[3] <= box[1]`을 만족하는 경우에만 아래 분기를 실행합니다.
            if box[2] <= box[0] or box[3] <= box[1]:
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # `distance` 값에 계산식 `((box[0] + box[2]) / 2 - w / 2) ** 2 + ((box[1] + box[3]) / 2 - h / 2) ** 2`의 결과를 저장합니다. 대상: `distance`.
            distance = ((box[0] + box[2]) / 2 - w / 2) ** 2 + ((box[1] + box[3]) / 2 - h / 2) ** 2
            # `candidates.append((distance, box))`을 호출하여 항목 하나를 끝에 추가합니다.
            candidates.append((distance, box))
        # 조건 `not candidates`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not candidates:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 현재 함수 처리를 끝내고 `np.asarray(min(candidates, key=lambda candidate: candidate[0])[1], dtype=np.float32)` 호출 결과를 호출한 쪽으로 반환합니다.
        return np.asarray(min(candidates, key=lambda candidate: candidate[0])[1], dtype=np.float32)

    # 데코레이터 `staticmethod`를 바로 아래 함수·클래스에 적용합니다.
    @staticmethod
    # 형식 변환 관절 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _format_keypoints(pose_result, frame):
        # `(h, w)` 값에 계산식 `frame.shape[:2]`의 결과를 저장합니다. 대상: `(h, w)`.
        h, w = frame.shape[:2]
        # `predicted` 값에 `pose_result.pred_instances` 값(`pose_result.pred_instances`)을 저장합니다. 대상: `predicted`.
        predicted = pose_result.pred_instances
        # 조건 `len(predicted.keypoints) == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if len(predicted.keypoints) == 0:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # `(kpts, scores)` 값에 순서·구성을 지정한 `(predicted.keypoints[0], predicted.keypoint_scores[0])`을 저장합니다. 대상: `(kpts, scores)`.
        kpts, scores = predicted.keypoints[0], predicted.keypoint_scores[0]
        # 현재 함수 처리를 끝내고 반복·조건식 `[{'id': i, 'x': round(float(kpts[i][0] / w), 4), 'y': round(float(kpts[i][1] / h), 4), 'score': round(float(scores[i]), 3), 'image_aspect_ratio': w / h} for i in range(1…`으로 구성한 값을 호출한 쪽으로 반환합니다.
        return [{"id": i, "x": round(float(kpts[i][0] / w), 4),
                 # 구성 항목: `y` 값(`'y'`)에 `round(float(kpts[i][1] / h), 4)` 호출 결과; 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 `round(float(scores[i]), 3)` 호출 결과를 지정합니다.
                 "y": round(float(kpts[i][1] / h), 4), "score": round(float(scores[i]), 3),
                 # 구성 항목: 입력 영상의 가로/세로 비율(`'image_aspect_ratio'`)에 계산식 `w / h`의 결과를 지정합니다.
                 "image_aspect_ratio": w / h}
                # 앞줄에서 시작한 식·인수·자료 구조를 `for i in range(17)]` 내용으로 이어서 구성합니다.
                for i in range(17)]

    # 데코레이터 `torch.no_grad()`를 바로 아래 함수·클래스에 적용합니다.
    @torch.no_grad()
    # 영상 프레임 묶음의 사람 검출과 자세 추론을 실행하고 입력 순서에 대응하는 관절 목록을 반환합니다.
    def extract_keypoints_batch(self, frames: list) -> list:
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Batch YOLO and RTMPose while preserving one result slot per input.

        Missing frames/people produce None at their original index. Offline
        detection is independent of the realtime bbox cache and batch size.
        """
        # 조건 `not frames`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not frames:
            # 현재 함수 처리를 끝내고 순서·구성을 지정한 `[]`을 호출한 쪽으로 반환합니다.
            return []
        # 생성할 결과 데이터 또는 경로에 계산식 `[None] * len(frames)`의 결과를 저장합니다. 대상: `output`.
        output = [None] * len(frames)
        # `valid_indices` 값에 반복·조건식 `[i for (i, frame) in enumerate(frames) if frame is not None]`으로 구성한 값을 저장합니다. 대상: `valid_indices`.
        valid_indices = [i for i, frame in enumerate(frames) if frame is not None]
        # 조건 `not valid_indices`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not valid_indices:
            # 현재 함수 처리를 끝내고 생성할 결과 데이터 또는 경로(`output`)을 호출한 쪽으로 반환합니다.
            return output
        # `images` 값에 반복·조건식 `[frames[i] for i in valid_indices]`으로 구성한 값을 저장합니다. 대상: `images`.
        images = [frames[i] for i in valid_indices]
        # `detections` 값에 `self.detector(images, classes=[0], imgsz=320, verbose=False, device=self.device)` 호출 결과를 저장합니다. 대상: `detections`.
        detections = self.detector(images, classes=[0], imgsz=320, verbose=False, device=self.device)
        # 조건 `len(detections) != len(images)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if len(detections) != len(images):
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('YOLO batch result count does not match the input frame count')` 예외를 발생시킵니다.
            raise RuntimeError("YOLO batch result count does not match the input frame count")

        # 설치된 MMPose의 inference_topdown과 같은 전처리를 사용합니다.
        # 서로 다른 프레임의 사람 영역을 하나의 model.test_step 배치로 묶습니다.
        # `scope` 값에 `self.pose_model.cfg.get('default_scope', 'mmpose')` 호출 결과를 저장합니다. 대상: `scope`.
        scope = self.pose_model.cfg.get('default_scope', 'mmpose')
        # 조건 `scope is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if scope is not None:
            # `init_default_scope(scope)`을 호출하여 해당 단계의 작업을 수행합니다.
            init_default_scope(scope)
        # 조건 `self._batch_pose_pipeline is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._batch_pose_pipeline is None:
            # 일괄 처리 pose pipeline에 `Compose(self.pose_model.cfg.test_dataloader.dataset.pipeline)` 호출 결과를 저장합니다. 대상: `self._batch_pose_pipeline`.
            self._batch_pose_pipeline = Compose(self.pose_model.cfg.test_dataloader.dataset.pipeline)
        # `(samples, sample_indices)` 값에 순서·구성을 지정한 `([], [])`을 저장합니다. 대상: `(samples, sample_indices)`.
        samples, sample_indices = [], []
        # `zip(valid_indices, detections)`에서 항목을 차례로 꺼내 `(index, detection)`에 넣고 아래 처리를 반복합니다.
        for index, detection in zip(valid_indices, detections):
            # `bbox` 값에 `self._select_center_bbox(detection, frames[index])` 호출 결과를 저장합니다. 대상: `bbox`.
            bbox = self._select_center_bbox(detection, frames[index])
            # 조건 `bbox is None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if bbox is None:
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # 데이터에 `dict(img=frames[index], bbox=bbox[None], bbox_score=np.ones(1, dtype=np.float32))` 호출 결과를 저장합니다. 대상: `data`.
            data = dict(img=frames[index], bbox=bbox[None], bbox_score=np.ones(1, dtype=np.float32))
            # `data.update(self.pose_model.dataset_meta)`을 호출하여 해당 단계의 작업을 수행합니다.
            data.update(self.pose_model.dataset_meta)
            # `sample` 값에 `self._batch_pose_pipeline(data)` 호출 결과를 저장합니다. 대상: `sample`.
            sample = self._batch_pose_pipeline(data)
            # 조건 `sample is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if sample is not None:
                # `samples.append(sample)`을 호출하여 항목 하나를 끝에 추가합니다.
                samples.append(sample)
                # `sample_indices.append(index)`을 호출하여 항목 하나를 끝에 추가합니다.
                sample_indices.append(index)
        # 조건 `samples`을 만족하는 경우에만 아래 분기를 실행합니다.
        if samples:
            # `poses` 값에 `self.pose_model.test_step(pseudo_collate(samples))` 호출 결과를 저장합니다. 대상: `poses`.
            poses = self.pose_model.test_step(pseudo_collate(samples))
            # 조건 `len(poses) != len(sample_indices)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if len(poses) != len(sample_indices):
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('RTMPose batch result count does not match the detected frame count')` 예외를 발생시킵니다.
                raise RuntimeError("RTMPose batch result count does not match the detected frame count")
            # `zip(sample_indices, poses)`에서 항목을 차례로 꺼내 `(index, pose)`에 넣고 아래 처리를 반복합니다.
            for index, pose in zip(sample_indices, poses):
                # `output[index]` 값에 `self._format_keypoints(pose, frames[index])` 호출 결과를 저장합니다. 대상: `output[index]`.
                output[index] = self._format_keypoints(pose, frames[index])
        # 현재 함수 처리를 끝내고 생성할 결과 데이터 또는 경로(`output`)을 호출한 쪽으로 반환합니다.
        return output

    # 데코레이터 `torch.no_grad()`를 바로 아래 함수·클래스에 적용합니다.
    @torch.no_grad()  # 추론 과정에서 역전파용 기울기 연산을 차단하여 메모리를 절약하고 속도를 극대화
    # 실시간 단일 프레임에서 대상자의 COCO 관절을 추출합니다.
    def extract_keypoints(self, frame: np.ndarray) -> list:  # RGB/BGR 이미지 배열을 받아 관절 좌표 리스트를 반환하는 메서드
        # 조건 `frame is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if frame is None:  # 입력된 이미지 프레임 변수가 비어있는 경우
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None  # 추론할 대상이 없으므로 None 반환 후 종료

        # `(h, w, _)` 값에 배열 차원(`frame.shape`)을 저장합니다. 대상: `(h, w, _)`.
        h, w, _ = frame.shape  # 연산을 위해 프레임의 세로, 가로, 채널 수 추출
        # 프레임 횟수에 `1`을 더해 누적합니다. 대상: `self.frame_count`.
        self.frame_count += 1  # 엔진 처리 프레임 카운터 증가

        # 조건 `self.cached_center_bbox is None or self.frame_count % self.yolo_interval == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.cached_center_bbox is None or (self.frame_count % self.yolo_interval == 0):  # 갱신 주기가 되었는지 조건 검사
            # `det_results` 값에 계산식 `self.detector(frame, classes=[0], imgsz=320, verbose=False, device=self.device)[0]`의 결과를 저장합니다. 대상: `det_results`.
            det_results = self.detector(frame, classes=[0], imgsz=320, verbose=False, device=self.device)[0]  # 사람 한정 추론 수행
            
            # 조건 `len(det_results.boxes) > 0`을 만족하는 경우에만 아래 분기를 실행합니다.
            if len(det_results.boxes) > 0:  # 화면 내에 감지된 사람 객체가 1명 이상 존재하는 경우
                # `boxes` 값에 `det_results.boxes.xyxy.cpu().numpy()` 호출 결과를 저장합니다. 대상: `boxes`.
                boxes = det_results.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표를 CPU 메모리의 NumPy 배열로 복사
                # `confidences` 값에 `det_results.boxes.conf.cpu().numpy()` 호출 결과를 저장합니다. 대상: `confidences`.
                confidences = det_results.boxes.conf.cpu().numpy()  # 박스별 감지 신뢰도 점수 배열 복사

                # `(cam_center_x, cam_center_y)` 값에 순서·구성을 지정한 `(w / 2.0, h / 2.0)`을 저장합니다. 대상: `(cam_center_x, cam_center_y)`.
                cam_center_x, cam_center_y = w / 2.0, h / 2.0  # 카메라 화면 정중앙의 X, Y 픽셀 좌표 기준점 연산
                # `min_dist` 값에 `float('inf')` 호출 결과를 저장합니다. 대상: `min_dist`.
                min_dist = float('inf')  # 최소 거리 비교를 위한 무한대 값 초기화
                # `best_box` 값에 None을 저장합니다. 대상: `best_box`.
                best_box = None  # 타겟 대상의 바운딩 박스를 담을 변수 초기화

                # `zip(boxes, confidences)`에서 항목을 차례로 꺼내 `(box, conf)`에 넣고 아래 처리를 반복합니다.
                for box, conf in zip(boxes, confidences):  # 검출된 모든 박스 좌표와 신뢰도 점수를 순회
                    # 조건 `conf > 0.4`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if conf > 0.4:  # 노이즈를 걸러내기 위해 신뢰도가 0.4를 초과하는 객체만 취급
                        # `box_center_x` 값에 계산식 `(box[0] + box[2]) / 2.0`의 결과를 저장합니다. 대상: `box_center_x`.
                        box_center_x = (box[0] + box[2]) / 2.0  # 현재 검사 중인 박스의 중앙 X 좌표 연산
                        # `box_center_y` 값에 계산식 `(box[1] + box[3]) / 2.0`의 결과를 저장합니다. 대상: `box_center_y`.
                        box_center_y = (box[1] + box[3]) / 2.0  # 현재 검사 중인 박스의 중앙 Y 좌표 연산
                        # `dist` 값에 계산식 `(box_center_x - cam_center_x) ** 2 + (box_center_y - cam_center_y) ** 2`의 결과를 저장합니다. 대상: `dist`.
                        dist = (box_center_x - cam_center_x) ** 2 + (box_center_y - cam_center_y) ** 2  # 중심 거리 제곱 연산
                        
                        # 조건 `dist < min_dist`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if dist < min_dist:  # 화면 중앙에 가장 가까운 경우
                            # `min_dist` 값에 `dist` 값(`dist`)을 저장합니다. 대상: `min_dist`.
                            min_dist = dist  # 최소 거리 기준값 갱신
                            # `best_box` 값에 `box` 값(`box`)을 저장합니다. 대상: `best_box`.
                            best_box = box  # 최적 대상자 박스로 덮어쓰기

                # 조건 `best_box is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if best_box is not None:  # 유효한 중앙 대상자 박스를 찾은 경우
                    # `self.cached_center_bbox` 값에 `best_box` 값(`best_box`)을 저장합니다. 대상: `self.cached_center_bbox`.
                    self.cached_center_bbox = best_box  # 다음 프레임 재사용을 위해 캐시에 저장

        # 조건 `self.cached_center_bbox is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.cached_center_bbox is None:  # 사람을 찾지 못했거나 캐시가 비어있는 경우
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None  # 추출 실패로 간주하고 None 반환

        # `bboxes_np` 값에 `np.array([self.cached_center_bbox], dtype=np.float32)` 호출 결과를 저장합니다. 대상: `bboxes_np`.
        bboxes_np = np.array([self.cached_center_bbox], dtype=np.float32)  # MMPose 입력 규격에 맞게 2D NumPy 배열로 변환
        # `pose_results` 값에 `inference_topdown(self.pose_model, frame, bboxes_np)` 호출 결과를 저장합니다. 대상: `pose_results`.
        pose_results = inference_topdown(self.pose_model, frame, bboxes_np)  # 바운딩 박스 영역 내에서 관절 추론 수행

        # 조건 `len(pose_results) == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if len(pose_results) == 0:  # 추론 결과가 비어있는 경우
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None  # 추출 실패로 간주하고 None 반환

        # `pred_instances` 값에 `pose_results[0].pred_instances` 값(`pose_results[0].pred_instances`)을 저장합니다. 대상: `pred_instances`.
        pred_instances = pose_results[0].pred_instances  # 예측한 인스턴스 세부 데이터 추출
        # `kpts` 값에 계산식 `pred_instances.keypoints[0]`의 결과를 저장합니다. 대상: `kpts`.
        kpts = pred_instances.keypoints[0]  # 검출된 대상자의 17개 관절 픽셀 좌표 획득
        # 관절 신뢰도 배열 또는 좌우 게임 점수(사용 위치에 따름)에 계산식 `pred_instances.keypoint_scores[0]`의 결과를 저장합니다. 대상: `scores`.
        scores = pred_instances.keypoint_scores[0]  # 검출된 17개 관절의 신뢰도 점수 획득

        # COCO 관절 좌표 목록에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `keypoints`.
        keypoints = []  # 반환할 최종 관절 리스트 구조체 초기화
        # `range(17)`에서 항목을 차례로 꺼내 `coco_id`에 넣고 아래 처리를 반복합니다.
        for coco_id in range(17):  # 17개 관절을 순회
            # `(px, py)` 값에 계산식 `kpts[coco_id]`의 결과를 저장합니다. 대상: `(px, py)`.
            px, py = kpts[coco_id]  # 현재 관절의 픽셀 좌표 분리
            # 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)에 계산식 `scores[coco_id]`의 결과를 저장합니다. 대상: `score`.
            score = scores[coco_id]  # 현재 관절의 신뢰도 분리
            
            # `keypoints.append({'id': coco_id, 'x': round(float(px / w), 4), 'y': round(float(py / h), 4), 'score': round(float(score), 3), 'image_aspect_ratio': w / h})`을 호출하여 항목 하나를 끝에 추가합니다.
            keypoints.append({  # 딕셔너리 형태로 포장하여 등록
                # 구성 항목: `id` 값(`'id'`)에 `coco_id` 값(`coco_id`)을 지정합니다.
                "id": coco_id,  # 관절 식별 번호 할당
                # 구성 항목: `x` 값(`'x'`)에 `round(float(px / w), 4)` 호출 결과를 지정합니다.
                "x": round(float(px / w), 4),  # 해상도로 나누어 정규 비율 좌표로 변환
                # 구성 항목: `y` 값(`'y'`)에 `round(float(py / h), 4)` 호출 결과를 지정합니다.
                "y": round(float(py / h), 4),  # 해상도로 나누어 정규 비율 좌표로 변환
                # 구성 항목: 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 `round(float(score), 3)` 호출 결과를 지정합니다.
                "score": round(float(score), 3),  # 신뢰도 점수를 반올림하여 저장
                # 구성 항목: 입력 영상의 가로/세로 비율(`'image_aspect_ratio'`)에 계산식 `w / h`의 결과를 지정합니다.
                "image_aspect_ratio": w / h,
            # 앞줄에서 시작한 식·인수·자료 구조를 `}) # 단일 관절 데이터 등록 완료` 내용으로 이어서 구성합니다.
            })  # 단일 관절 데이터 등록 완료

        # 현재 함수 처리를 끝내고 COCO 관절 좌표 목록(`keypoints`)을 호출한 쪽으로 반환합니다.
        return keypoints  # 원본 비율 좌표가 담긴 리스트 반환
