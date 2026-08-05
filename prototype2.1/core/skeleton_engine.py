"""
==============================================================================
[프로그램 개요] AI 관절 추정 및 인체 검출 엔진 (skeleton_engine.py)
==============================================================================
"""

import os
import sys
import torch
import cv2
import numpy as np
from pathlib import Path

# ==============================================================================
# 1. CPU 연산 자원 제한 설정 (비동기 서버 지연시간 최적화)
# ==============================================================================
torch.set_num_threads(2)
cv2.setNumThreads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# ==============================================================================
# 2. xtcocotools 라이브러리 호환성 패치 및 MMPose 모듈 등록
# ==============================================================================
import pycocotools
sys.modules['xtcocotools'] = pycocotools

from mmpose.apis.inference import init_model, inference_topdown
from mmpose.utils import register_all_modules

register_all_modules()

from ultralytics import YOLO
from utils.normalization import normalize_pelvis_centered


# ==============================================================================
# 3. SkeletonEngine 메인 클래스 정의
# ==============================================================================
class SkeletonEngine:
    def __init__(self, device: str = None, yolo_interval: int = 3):
        if device is None:
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"[SkeletonEngine] Device 설정 완료: {self.device}")

        root_dir = Path(__file__).resolve().parent.parent.parent
        yolo_path = str(root_dir / "configs" / "checkpoints" / "yolov8n.pt")
        config_path = str(root_dir / "configs" / "checkpoints" / "RTMPose_config.py")
        checkpoint_path = str(root_dir / "configs" / "checkpoints" / "RTMPose.pth")

        print("[SkeletonEngine] Detector(YOLOv8n) 모델 로드 중...")
        self.detector = YOLO(yolo_path)

        print("[SkeletonEngine] RTMPose 모델 초기화 중...")
        self.pose_model = init_model(config_path, checkpoint_path, device=self.device)
        print("[SkeletonEngine] 모든 모델 메모리 로드 완료.")

        self.yolo_interval = yolo_interval
        self.frame_count = 0
        self.cached_center_bbox = None

    @torch.no_grad()
    def extract_keypoints(self, frame: np.ndarray) -> list:
        if frame is None:
            return None

        h, w, _ = frame.shape
        self.frame_count += 1

        # [단계 1] YOLO 인체 검출 연산
        if self.cached_center_bbox is None or (self.frame_count % self.yolo_interval == 0):
            det_results = self.detector(frame, classes=[0], imgsz=320, verbose=False, device=self.device)[0]

            if len(det_results.boxes) > 0:
                boxes = det_results.boxes.xyxy.cpu().numpy()
                confidences = det_results.boxes.conf.cpu().numpy()

                cam_center_x, cam_center_y = w / 2.0, h / 2.0
                min_dist = float('inf')
                best_box = None

                for box, conf in zip(boxes, confidences):
                    if conf > 0.4:
                        box_center_x = (box[0] + box[2]) / 2.0
                        box_center_y = (box[1] + box[3]) / 2.0

                        dist = (box_center_x - cam_center_x) ** 2 + (box_center_y - cam_center_y) ** 2
                        if dist < min_dist:
                            min_dist = dist
                            best_box = box

                if best_box is not None:
                    self.cached_center_bbox = best_box

        if self.cached_center_bbox is None:
            return None

        # [단계 2] RTMPose 관절 추정 연산
        bboxes_np = np.array([self.cached_center_bbox], dtype=np.float32)
        pose_results = inference_topdown(self.pose_model, frame, bboxes_np)

        if len(pose_results) == 0:
            return None

        pred_instances = pose_results[0].pred_instances
        kpts = pred_instances.keypoints[0]
        scores = pred_instances.keypoint_scores[0]

        # [단계 3] COCO 17개 관절 좌표 정규화 및 골반 중심 좌표 변환
        keypoints = []
        for coco_id in range(17):
            px, py = kpts[coco_id]
            score = scores[coco_id]

            keypoints.append({
                "id": coco_id,
                "x": round(float(px / w), 4),
                "y": round(float(py / h), 4),
                "score": round(float(score), 3)
            })

        # 골반 중점 (0,0) 원점 및 상체 길이 1.0 기준 재정규화 적용
        if len(keypoints) == 17:
            keypoints = normalize_pelvis_centered(keypoints)

        return keypoints

    def release(self):
        print("[SkeletonEngine] 자원 해제 완료.")