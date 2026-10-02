# ==============================================================================
# [Module Information]
# File: utils/overlay_renderer.py
# Description: Canvas-synchronized skeleton renderer supporting both normalized
#              image coordinates (0.0~1.0) and Cartesian normalized pose space.
# ==============================================================================

import cv2
import numpy as np

# COCO 17 관절 기준 주요 뼈대 연결 페어 (총 12개)
SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체 어깨 및 양팔
    (5, 11), (6, 12), (11, 12),               # 몸통 및 골반 코어
    (11, 13), (13, 15), (12, 14), (14, 16)   # 하지 허벅지 및 종아리
]


class OverlayRenderer:
    """
    영상 프레임 위에 뼈대 연결선과 관절 노드를 종횡비 왜곡 없이 렌더링하는 클래스.
    원본 비율 좌표계(0.0~1.0)와 정규화 데카르트 좌표계 입력을 모두 지원합니다.
    """

    def __init__(self, conf_threshold: float = 0.35):
        """
        Args:
            conf_threshold: 화면에 표시할 관절의 최하 신뢰도 컷오프
        """
        self.conf_threshold = conf_threshold
        print(f"[DEBUG][RENDERER_INIT] OverlayRenderer initialized. Confidence Threshold: {self.conf_threshold}")

    def draw_skeleton(
        self,
        canvas: np.ndarray,
        keypoints: list,
        is_occluded: bool = False,
        is_cartesian_space: bool = False
    ) -> np.ndarray:
        """
        캔버스 상에 관절과 뼈대를 중첩하여 렌더링합니다.

        Args:
            canvas: (H, W, 3) 크기의 OpenCV BGR 이미지 행렬
            keypoints: 17개 관절 딕셔너리 리스트
            is_occluded: 가림 감지 상태 (True일 경우 경고 색상 적용)
            is_cartesian_space: 입력 데이터가 골반 중심 데카르트 좌표계인지 여부
        Returns:
            np.ndarray: 뼈대가 시각화된 BGR 이미지 행렬
        """
        if canvas is None or not keypoints:
            return canvas

        canvas_h, canvas_w = canvas.shape[:2]
        kpt_dict = {}

        for kp in keypoints:
            score = kp.get("score", 0.0)
            if score >= self.conf_threshold:
                raw_x = float(kp.get("x", 0.0))
                raw_y = float(kp.get("y", 0.0))

                if is_cartesian_space:
                    # 골반 중심 (0, 0), 상향(+Y)인 데카르트 좌표를 캔버스 화면 중앙으로 사영
                    # 화면 중앙을 원점으로 배치하고, 몸통 길이 1.0을 캔버스 높이의 30%로 스케일 매핑
                    viewport_scale = canvas_h * 0.3
                    px = int(canvas_w / 2.0 + (raw_x * viewport_scale))
                    py = int(canvas_h / 2.0 - (raw_y * viewport_scale))  # +Y가 화면 위쪽이 되도록 반전
                else:
                    # 일반 정규 비율 좌표계 (0.0 ~ 1.0)
                    clamped_x = min(max(raw_x, 0.0), 1.0)
                    clamped_y = min(max(raw_y, 0.0), 1.0)
                    px = int(clamped_x * canvas_w)
                    py = int(clamped_y * canvas_h)

                kpt_dict[kp["id"]] = (px, py)

        # 상태에 따른 BGR 색상 팔레트 지정
        line_color = (0, 0, 255) if is_occluded else (0, 255, 255)       # 가림: Red, 정상: Yellow
        joint_color = (0, 80, 255) if is_occluded else (0, 165, 255)     # 가림: Dark Orange, 정상: Bright Orange

        # 1. 뼈대 연결선 렌더링
        for p1_id, p2_id in SKELETON_CONNECTIONS:
            if p1_id in kpt_dict and p2_id in kpt_dict:
                cv2.line(canvas, kpt_dict[p1_id], kpt_dict[p2_id], line_color, 2, cv2.LINE_AA)

        # 2. 신체 주요 12개 관절 포인트 렌더링 (얼굴 0~4번 제외)
        for j_id, (px, py) in kpt_dict.items():
            if j_id >= 5:
                # 캔버스 밖으로 벗어난 좌표는 그리기 생략
                if 0 <= px < canvas_w and 0 <= py < canvas_h:
                    cv2.circle(canvas, (px, py), 4, joint_color, -1, cv2.LINE_AA)

        return canvas