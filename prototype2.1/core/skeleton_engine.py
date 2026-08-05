import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# ==============================================================================
# 1. CPU 연산 자원 제한 설정
# Multi-threading 병목 현상을 방지하고 비동기 소켓 서버 환경에서 지연 시간을 최적화하기 위해 
# PyTorch, OpenCV 및 OpenMP/MKL 스레드 개수를 2개로 제한합니다.
# ==============================================================================
torch.set_num_threads(2)
cv2.setNumThreads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# ==============================================================================
# 2. xtcocotools 라이브러리 호환성 패치
# 일부 RTMPose 설정 파일이 xtcocotools를 참조할 때 기존 pycocotools로 대체되도록 모듈을 매핑합니다.
# ==============================================================================
import pycocotools
sys.modules['xtcocotools'] = pycocotools

from mmpose.apis.inference import init_model, inference_topdown
from mmpose.utils import register_all_modules

# MMPose 구동에 필요한 전처리, 후처리 모듈 및 모델 레지스트리를 전역 등록합니다.
register_all_modules()

from ultralytics import YOLO


class SkeletonEngine:
    """
    YOLOv8n(사람 검출)과 RTMPose(관절 추정) 모델을 결합한 스켈레톤 추출 엔진 Class입니다.
    
    비디오 프레임 내에서 화면 중앙에 가장 가까운 1인을 타겟팅하며, 
    YOLO 연산 주기를 조절하는 프레임 스킵 기법을 통해 연산 속도를 최적화합니다.
    """
    def __init__(self, device: str = None, yolo_interval: int = 3):
        """
        SkeletonEngine 클래스 초기화
        
        Args:
            device (str, optional): 연산에 사용할 장치 ('cuda:0' 또는 'cpu'). None일 경우 자동 선택.
            yolo_interval (int): YOLO 사람 검출을 수행할 프레임 간격 (기본값: 3프레임마다 1회 연산).
        """
        # CUDA 사용 가능 여부를 점검하여 추론 장치(GPU/CPU)를 설정합니다.
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[SkeletonEngine] Device 설정 완료: {self.device}")

        # 프로젝트 최상위 디렉토리(Root) 기준으로 가중치 및 설정 파일 경로를 절대 경로로 로드합니다.
        root_dir = Path(__file__).resolve().parent.parent.parent
        yolo_path = str(root_dir / "configs" / "checkpoints" / "yolov8n.pt")
        config_path = str(root_dir / "configs" / "checkpoints" / "RTMPose_config.py")
        checkpoint_path = str(root_dir / "configs" / "checkpoints" / "RTMPose.pth")

        # 1. YOLOv8n 사람 검출 모델 로드
        print("[SkeletonEngine] Detector(YOLOv8n) 모델 로드 중...")
        self.detector = YOLO(yolo_path)

        # 2. RTMPose 관절 추정 모델 초기화 및 VRAM 로드
        print("[SkeletonEngine] RTMPose 모델 초기화 중...")
        self.pose_model = init_model(config_path, checkpoint_path, device=self.device)
        print("[SkeletonEngine] 모든 모델 메모리 로드 완료.")

        # 3. 프레임 스킵 제어 및 BBox 캐싱을 위한 내부 변수 초기화
        self.yolo_interval = yolo_interval  # YOLO 감지 주기
        self.frame_count = 0                # 수신된 전체 프레임 카운터
        self.cached_center_bbox = None      # 직전에 감지된 화면 중앙 인물의 바운딩 박스 좌표 (x1, y1, x2, y2)

    @torch.no_grad()
    def extract_keypoints(self, frame: np.ndarray) -> list:
        """
        입력 프레임에서 화면 중앙에 가장 가까운 피험자를 감지하고 17개 관절 좌표(COCO 포맷)를 추출합니다.

        Args:
            frame (np.ndarray): OpenCV BGR 이미지 프레임 (H, W, C)

        Returns:
            list: 17개 관절의 정규화된 좌표(x, y: 0.0~1.0)와 신뢰도(score)를 담은 딕셔너리 리스트.
                  사람이 감지되지 않으면 None 반환.
        """
        if frame is None:
            return None

        h, w, _ = frame.shape
        self.frame_count += 1

        # =========================================================================
        # [단계 1] YOLO 검출 연산 (지정된 프레임 주기마다 또는 저장된 BBox가 없을 때 실행)
        # =========================================================================
        if self.cached_center_bbox is None or (self.frame_count % self.yolo_interval == 0):
            # classes=[0](사람만 검출), imgsz=320(입력 해상도 축소)로 설정하여 속도를 최적화합니다.
            det_results = self.detector(frame, classes=[0], imgsz=320, verbose=False, device=self.device)[0]

            if len(det_results.boxes) > 0:
                boxes = det_results.boxes.xyxy.cpu().numpy()
                confidences = det_results.boxes.conf.cpu().numpy()

                # 화면 중앙 좌표 계산
                cam_center_x, cam_center_y = w / 2.0, h / 2.0
                min_dist = float('inf')
                best_box = None

                # 검출된 인물들 중 화면 중심점과 가장 가까운 대상을 탐색합니다.
                for box, conf in zip(boxes, confidences):
                    if conf > 0.4:  # BBox 신뢰도 점수가 0.4 이상인 경우만 처리
                        box_center_x = (box[0] + box[2]) / 2.0
                        box_center_y = (box[1] + box[3]) / 2.0

                        # 중심점 간 유클리드 거리의 제곱 계산
                        dist = (box_center_x - cam_center_x) ** 2 + (box_center_y - cam_center_y) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_box = box

                # 화면 중앙에 가장 가까운 인물 BBox로 캐시를 갱신합니다.
                if best_box is not None:
                    self.cached_center_bbox = best_box

        # 감지된 바운딩 박스가 없는 경우 관절 추정을 수행하지 않고 종료합니다.
        if self.cached_center_bbox is None:
            return None

        # =========================================================================
        # [단계 2] RTMPose 관절 추정 연산 (선택된 BBox 영역 대상)
        # =========================================================================
        bboxes_np = np.array([self.cached_center_bbox], dtype=np.float32)
        pose_results = inference_topdown(self.pose_model, frame, bboxes_np)

        if len(pose_results) == 0:
            return None

        # 추론 결과에서 키포인트 좌표 및 신뢰도 점수를 추출합니다.
        pred_instances = pose_results[0].pred_instances
        kpts = pred_instances.keypoints[0]
        scores = pred_instances.keypoint_scores[0]

        # =========================================================================
        # [단계 3] COCO 17개 Keypoint 좌표 정규화 (비율 0.0 ~ 1.0) 및 데이터 구성
        # =========================================================================
        keypoints = []
        for coco_id in range(17):
            px, py = kpts[coco_id]
            score = scores[coco_id]

            keypoints.append({
                "id": coco_id,
                "x": round(float(px / w), 4),     # 프레임 너비 기준 정규화 X 좌표 (0.0~1.0)
                "y": round(float(py / h), 4),     # 프레임 높이 기준 정규화 Y 좌표 (0.0~1.0)
                "score": round(float(score), 3)   # 관절 인식 신뢰도 점수
            })

        return keypoints

    def release(self):
        """
        엔진 사용 종료 시 자원 해제 및 상태 알림
        """
        print("[SkeletonEngine] 자원 해제 완료.")