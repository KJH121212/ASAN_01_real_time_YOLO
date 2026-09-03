# ==============================================================================
# [파일 정보]
# 파일명: utils/overlay_renderer.py
# 작성자: 개발자 (Developer)
# 설명: 실시간 관절 좌표 시각화 및 영상 해상도 종횡비 자동 보정 단일 패스(Single-Pass) 렌더러
# ==============================================================================

# ------------------------------------------------------------------------------
# [코드 설명]
# 본 모듈은 SkeletonEngine에서 0.0~1.0 비율로 정규화된 17개 관절 좌표를 입력받아
# 캔버스의 실제 가로(W)와 세로(H) 해상도에 맞춰 각각 독립적으로 스케일링을 수행합니다.
# 단일 scale 변수로 인해 발생하던 종횡비 왜곡(Y축 밀림/어긋남)을 원천 차단하고
# 신체 뼈대 연결선과 관절점을 원본 영상에 오차 없이 정확하게 오버레이합니다.
# ------------------------------------------------------------------------------

import cv2  # 이미지 픽셀 조작 및 도형 렌더링을 위한 OpenCV 라이브러리 로드
import numpy as np  # 고속 배열 연산 및 좌표 클램핑 처리를 위한 NumPy 라이브러리 로드

SKELETON_CONNECTIONS = [  # COCO 17 관절 기준 신체 주요 뼈대 연결선 정의
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체 어깨 및 팔 관절 연결선
    (5, 11), (6, 12), (11, 12),  # 척추 및 골반 몸통 박스 연결선
    (11, 13), (13, 15), (12, 14), (14, 16)  # 하체 허벅지 및 종아리 연결선
]  # 총 12개 주요 뼈대 연결 페어 리스트 구성 완료


class OverlayRenderer:  # 관절 좌표를 영상 위에 정밀하게 중첩 렌더링하는 전담 클래스 선언

    def __init__(self, conf_threshold: float = 0.35):  # 렌더러 초기화 생성자 정의
        self.conf_threshold = conf_threshold  # 관절을 화면에 표시할 최소 감지 신뢰도 임계값 저장

    def draw_skeleton(self, canvas: np.ndarray, keypoints: list, is_occluded: bool = False) -> np.ndarray:  # 캔버스 크기 자동 동기화 렌더링 메서드 정의
        if not keypoints or canvas is None:  # 관절 데이터가 없거나 캔버스가 비어있는 경우
            return canvas  # 추가 연산 없이 입력된 원본 캔버스 즉시 반환

        canvas_h, canvas_w = canvas.shape[:2]  # 전달받은 캔버스의 실제 세로(H) 및 가로(W) 픽셀 해상도 추출
        kpt_dict = {}  # 픽셀 좌표를 O(1)로 조회하기 위한 해시 딕셔너리 초기화

        for kp in keypoints:  # 17개 관절 포인트 순회
            score = kp.get("score", 0.0)  # 관절 감지 신뢰도 추출
            if score >= self.conf_threshold:  # 신뢰도가 임계값 이상인 유효 관절만 필터링
                norm_x = min(max(float(kp.get("x", 0.0)), 0.0), 1.0)  # X 비율 좌표를 0.0~1.0 범위로 클램핑하여 오버플로우 방지
                norm_y = min(max(float(kp.get("y", 0.0)), 0.0), 1.0)  # Y 비율 좌표를 0.0~1.0 범위로 클램핑하여 오버플로우 방지
                px = int(norm_x * canvas_w)  # 캔버스의 실제 가로폭(canvas_w)을 곱해 정밀 픽셀 X 좌표 산출
                py = int(norm_y * canvas_h)  # 캔버스의 실제 세로높이(canvas_h)를 곱해 정밀 픽셀 Y 좌표 산출
                kpt_dict[kp["id"]] = (px, py)  # 관절 ID를 키로 하여 변환된 정수 픽셀 좌표 튜플 저장

        line_color = (0, 0, 255) if is_occluded else (0, 255, 255)  # 전신 가림 시 붉은색, 정상 시 노란색으로 선 색상 분기
        for p1_id, p2_id in SKELETON_CONNECTIONS:  # 사전 정의된 뼈대 연결 인덱스 순회
            if p1_id in kpt_dict and p2_id in kpt_dict:  # 시작 관절과 끝 관절이 모두 화면에 존재하는 경우
                cv2.line(canvas, kpt_dict[p1_id], kpt_dict[p2_id], line_color, 2, cv2.LINE_AA)  # 앤티앨리어싱을 적용하여 두 점 사이에 뼈대 선분 렌더링

        joint_color = (0, 100, 255) if is_occluded else (0, 165, 255)  # 전신 가림 시 진한 주황, 정상 시 밝은 주황으로 관절 색상 분기
        for j_id, (px, py) in kpt_dict.items():  # 유효하게 변환된 모든 관절 좌표 순회
            if j_id >= 5:  # 얼굴 부위를 제외한 신체 12개 관절만 타겟팅
                cv2.circle(canvas, (px, py), 4, joint_color, -1, cv2.LINE_AA)  # 관절 위치에 앤티앨리어싱이 적용된 채워진 원형 포인트 렌더링

        return canvas  # 뼈대와 관절 오버레이가 완료된 최종 캔버스 행렬 반환