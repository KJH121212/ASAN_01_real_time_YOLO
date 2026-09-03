# ==============================================================================
# [파일 정보]
# 파일명: core/skeleton_engine.py
# 설명: YOLOv8 및 RTMPose 기반 실시간 관절 포인트 추출 엔진 (mmcv 에러 완벽 우회 적용)
# ==============================================================================

import os  # 운영체제 환경 변수 설정을 위한 내장 모듈 로드
os.environ["MMCV_WITH_OPS"] = "0"  # mmcv의 C++ 확장 모듈 로드를 원천적으로 차단

import sys  # 파이썬 런타임 환경 제어를 위한 내장 모듈 로드
from unittest.mock import MagicMock  # 존재하지 않는 모듈을 가짜로 만들기 위한 모킹 클래스 로드

mock_ext = MagicMock()  # 호출되어도 에러를 발생시키지 않는 가짜 객체 생성
mock_ext.__spec__ = MagicMock()  # 임포트 시스템의 스펙 검사 우회를 위한 속성 추가
sys.modules['mmcv._ext'] = mock_ext  # 문제가 되는 mmcv C++ 확장 모듈을 가짜 객체로 대체
sys.modules['mmcv.ops.multi_scale_deform_attn'] = mock_ext  # 트랜스포머 어텐션 모듈 임포트 에러 우회
sys.modules['mmcv.ops.active_rotated_filter'] = mock_ext  # 로테이티드 필터 모듈 임포트 에러 우회

import pycocotools  # mmpose 내부의 xtcocotools 종속성 에러를 방지하기 위한 코코툴스 로드
sys.modules['xtcocotools'] = pycocotools  # xtcocotools를 pycocotools로 매핑하여 강제 로드

from pathlib import Path  # 경로 탐색을 위한 모듈 로드
import cv2  # 이미지 처리를 위한 OpenCV 라이브러리 로드
import numpy as np  # 배열 연산을 위한 NumPy 라이브러리 로드
import torch  # 딥러닝 모델 구동을 위한 PyTorch 로드

# PyTorch 보안 정책(weights_only=True) 우회 패치
_original_torch_load = torch.load  # 원본 로드 함수 백업

def _patched_torch_load(*args, **kwargs):  # 경고를 끄기 위한 래퍼 함수 정의
    kwargs['weights_only'] = False  # 보안 로드 기능을 강제로 해제
    return _original_torch_load(*args, **kwargs)  # 패치된 상태로 원본 함수 실행

torch.load = _patched_torch_load  # 전역 패치 적용

from mmpose.apis.inference import init_model, inference_topdown  # MMPose 추론 API 로드
from mmpose.utils import register_all_modules  # 커스텀 모듈 레지스트리 등록 함수 로드
from ultralytics import YOLO  # 사람 객체 탐지를 위한 YOLO 로드

register_all_modules()  # MMPose 모듈 안전 등록


class SkeletonEngine:  # 실시간 관절 포인트 추출을 전담하는 메인 AI 엔진 클래스 선언

    def __init__(self, device: str = None, yolo_interval: int = 5):  # 엔진 초기화 생성자 및 기본 프레임 주기 설정
        if device is None:  # 명시적으로 디바이스가 전달되지 않은 경우
            self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'  # GPU 가용성을 확인하여 자동으로 디바이스 할당
        else:  # 디바이스가 외부에서 명시된 경우
            self.device = device  # 지정된 연산 장치로 확정 적용

        root_dir = Path(__file__).resolve().parent.parent  # 현재 파일 기준 프로젝트 최상위 루트 디렉터리 경로 탐색
        yolo_path = str(root_dir / "configs" / "checkpoints" / "yolov8n.pt")  # YOLOv8 가중치 파일의 절대 경로 조립
        config_path = str(root_dir / "configs" / "checkpoints" / "RTMPose_config.py")  # RTMPose 모델 구조 설정 파일의 절대 경로 조립
        checkpoint_path = str(root_dir / "configs" / "checkpoints" / "RTMPose.pth")  # RTMPose 학습 가중치 파일의 절대 경로 조립

        self.detector = YOLO(yolo_path)  # 지정된 경로의 가중치로 YOLOv8 사람 탐지기 객체 생성 및 메모리 로드
        self.pose_model = init_model(config_path, checkpoint_path, device=self.device)  # 지정된 디바이스 메모리에 RTMPose 모델 초기화
        
        self.yolo_interval = yolo_interval  # YOLO 추론을 건너뛰어 성능을 최적화할 프레임 간격 저장
        self.frame_count = 0  # 시스템에서 누적 처리된 프레임 수를 기록할 카운터 초기화
        self.cached_center_bbox = None  # 직전 프레임에서 확정된 대상자 바운딩 박스 영역 캐시 초기화

    @torch.no_grad()  # 추론 과정에서 역전파용 기울기 연산을 차단하여 메모리를 절약하고 속도를 극대화
    def extract_keypoints(self, frame: np.ndarray) -> list:  # RGB/BGR 이미지 배열을 받아 관절 좌표 리스트를 반환하는 메서드
        if frame is None:  # 입력된 이미지 프레임 변수가 비어있는 경우
            return None  # 추론할 대상이 없으므로 None 반환 후 종료

        h, w, _ = frame.shape  # 연산을 위해 프레임의 세로, 가로, 채널 수 추출
        self.frame_count += 1  # 엔진 처리 프레임 카운터 증가

        if self.cached_center_bbox is None or (self.frame_count % self.yolo_interval == 0):  # 갱신 주기가 되었는지 조건 검사
            det_results = self.detector(frame, classes=[0], imgsz=320, verbose=False, device=self.device)[0]  # 사람 한정 추론 수행
            
            if len(det_results.boxes) > 0:  # 화면 내에 감지된 사람 객체가 1명 이상 존재하는 경우
                boxes = det_results.boxes.xyxy.cpu().numpy()  # 바운딩 박스 좌표를 CPU 메모리의 NumPy 배열로 복사
                confidences = det_results.boxes.conf.cpu().numpy()  # 박스별 감지 신뢰도 점수 배열 복사

                cam_center_x, cam_center_y = w / 2.0, h / 2.0  # 카메라 화면 정중앙의 X, Y 픽셀 좌표 기준점 연산
                min_dist = float('inf')  # 최소 거리 비교를 위한 무한대 값 초기화
                best_box = None  # 타겟 대상의 바운딩 박스를 담을 변수 초기화

                for box, conf in zip(boxes, confidences):  # 검출된 모든 박스 좌표와 신뢰도 점수를 순회
                    if conf > 0.4:  # 노이즈를 걸러내기 위해 신뢰도가 0.4를 초과하는 객체만 취급
                        box_center_x = (box[0] + box[2]) / 2.0  # 현재 검사 중인 박스의 중앙 X 좌표 연산
                        box_center_y = (box[1] + box[3]) / 2.0  # 현재 검사 중인 박스의 중앙 Y 좌표 연산
                        dist = (box_center_x - cam_center_x) ** 2 + (box_center_y - cam_center_y) ** 2  # 중심 거리 제곱 연산
                        
                        if dist < min_dist:  # 화면 중앙에 가장 가까운 경우
                            min_dist = dist  # 최소 거리 기준값 갱신
                            best_box = box  # 최적 대상자 박스로 덮어쓰기

                if best_box is not None:  # 유효한 중앙 대상자 박스를 찾은 경우
                    self.cached_center_bbox = best_box  # 다음 프레임 재사용을 위해 캐시에 저장

        if self.cached_center_bbox is None:  # 사람을 찾지 못했거나 캐시가 비어있는 경우
            return None  # 추출 실패로 간주하고 None 반환

        bboxes_np = np.array([self.cached_center_bbox], dtype=np.float32)  # MMPose 입력 규격에 맞게 2D NumPy 배열로 변환
        pose_results = inference_topdown(self.pose_model, frame, bboxes_np)  # 바운딩 박스 영역 내에서 관절 추론 수행

        if len(pose_results) == 0:  # 추론 결과가 비어있는 경우
            return None  # 추출 실패로 간주하고 None 반환

        pred_instances = pose_results[0].pred_instances  # 예측한 인스턴스 세부 데이터 추출
        kpts = pred_instances.keypoints[0]  # 검출된 대상자의 17개 관절 픽셀 좌표 획득
        scores = pred_instances.keypoint_scores[0]  # 검출된 17개 관절의 신뢰도 점수 획득

        keypoints = []  # 반환할 최종 관절 리스트 구조체 초기화
        for coco_id in range(17):  # 17개 관절을 순회
            px, py = kpts[coco_id]  # 현재 관절의 픽셀 좌표 분리
            score = scores[coco_id]  # 현재 관절의 신뢰도 분리
            
            keypoints.append({  # 딕셔너리 형태로 포장하여 등록
                "id": coco_id,  # 관절 식별 번호 할당
                "x": round(float(px / w), 4),  # 해상도로 나누어 정규 비율 좌표로 변환
                "y": round(float(py / h), 4),  # 해상도로 나누어 정규 비율 좌표로 변환
                "score": round(float(score), 3)  # 신뢰도 점수를 반올림하여 저장
            })  # 단일 관절 데이터 등록 완료

        return keypoints  # 원본 비율 좌표가 담긴 리스트 반환