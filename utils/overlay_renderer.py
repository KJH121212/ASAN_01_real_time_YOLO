# 한국어 코드 해설: 영상 위에 관절 연결선과 동작 측정 정보를 그리는 렌더링 도구를 제공합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [모듈 정보]
# 파일: utils/overlay_renderer.py
# 설명: 표시 화면에 맞춰 스켈레톤을 그리는 렌더러입니다.
# 0~1 영상 좌표와 신체 정규화 직교 좌표를 지원합니다.
# ==============================================================================

# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# COCO 17 관절 기준 주요 뼈대 연결 페어 (총 12개)
# `SKELETON_CONNECTIONS` 값에 순서·구성을 지정한 `[(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]`을 저장합니다. 대상: `SKELETON_CONNECTIONS`.
SKELETON_CONNECTIONS = [
    # 앞줄에서 시작한 식·인수·자료 구조를 `(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), # 상체 어깨 및 양팔` 내용으로 이어서 구성합니다.
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체 어깨 및 양팔
    # 앞줄에서 시작한 식·인수·자료 구조를 `(5, 11), (6, 12), (11, 12), # 몸통 및 골반 코어` 내용으로 이어서 구성합니다.
    (5, 11), (6, 12), (11, 12),               # 몸통 및 골반 코어
    # 앞줄에서 시작한 식·인수·자료 구조를 `(11, 13), (13, 15), (12, 14), (14, 16) # 하지 허벅지 및 종아리` 내용으로 이어서 구성합니다.
    (11, 13), (13, 15), (12, 14), (14, 16)   # 하지 허벅지 및 종아리
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]


# `OverlayRenderer` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class OverlayRenderer:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """
    영상 프레임 위에 뼈대 연결선과 관절 노드를 종횡비 왜곡 없이 렌더링하는 클래스.
    원본 비율 좌표계(0.0~1.0)와 정규화 데카르트 좌표계 입력을 모두 지원합니다.
    """

    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, conf_threshold: float = 0.35):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """
        Args:
            conf_threshold: 화면에 표시할 관절의 최하 신뢰도 컷오프
        """
        # 관절 검출 결과를 유효하게 사용할 최소 신뢰도에 관절 검출 결과를 유효하게 사용할 최소 신뢰도(`conf_threshold`)을 저장합니다. 대상: `self.conf_threshold`.
        self.conf_threshold = conf_threshold
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[DEBUG][RENDERER_INIT] OverlayRenderer initialized. Confidence Threshold: {self.conf_threshold}")

    # `draw_skeleton` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def draw_skeleton(
        # 이 줄의 입력 인수: `self` = 현재 객체. 기본값·형식은 아래 선언을 따릅니다.
        self,
        # 이 줄의 입력 인수: `canvas` = `canvas` 값. 기본값·형식은 아래 선언을 따릅니다.
        canvas: np.ndarray,
        # 이 줄의 입력 인수: `keypoints` = COCO 관절 좌표 목록. 기본값·형식은 아래 선언을 따릅니다.
        keypoints: list,
        # 이 줄의 입력 인수: `is_occluded` = 필수 관절이 가려졌거나 유효하지 않은지 여부. 기본값·형식은 아래 선언을 따릅니다.
        is_occluded: bool = False,
        # 이 줄의 입력 인수: `is_cartesian_space` = 여부 cartesian space. 기본값·형식은 아래 선언을 따릅니다.
        is_cartesian_space: bool = False
    # 앞줄에서 시작한 식·인수·자료 구조를 `) -> np.ndarray:` 내용으로 이어서 구성합니다.
    ) -> np.ndarray:
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
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
        # 조건 `canvas is None or not keypoints`을 만족하는 경우에만 아래 분기를 실행합니다.
        if canvas is None or not keypoints:
            # 현재 함수 처리를 끝내고 `canvas` 값(`canvas`)을 호출한 쪽으로 반환합니다.
            return canvas

        # `(canvas_h, canvas_w)` 값에 계산식 `canvas.shape[:2]`의 결과를 저장합니다. 대상: `(canvas_h, canvas_w)`.
        canvas_h, canvas_w = canvas.shape[:2]
        # `kpt_dict` 값에 빈 사전을 저장합니다. 대상: `kpt_dict`.
        kpt_dict = {}

        # `keypoints`에서 항목을 차례로 꺼내 `kp`에 넣고 아래 처리를 반복합니다.
        for kp in keypoints:
            # 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)에 `kp.get('score', 0.0)` 호출 결과를 저장합니다. 대상: `score`.
            score = kp.get("score", 0.0)
            # 조건 `score >= self.conf_threshold`을 만족하는 경우에만 아래 분기를 실행합니다.
            if score >= self.conf_threshold:
                # `raw_x` 값에 `float(kp.get('x', 0.0))` 호출 결과를 저장합니다. 대상: `raw_x`.
                raw_x = float(kp.get("x", 0.0))
                # `raw_y` 값에 `float(kp.get('y', 0.0))` 호출 결과를 저장합니다. 대상: `raw_y`.
                raw_y = float(kp.get("y", 0.0))

                # 조건 `is_cartesian_space`을 만족하는 경우에만 아래 분기를 실행합니다.
                if is_cartesian_space:
                    # 골반 중심 (0, 0), 상향(+Y)인 데카르트 좌표를 캔버스 화면 중앙으로 사영
                    # 화면 중앙을 원점으로 배치하고, 몸통 길이 1.0을 캔버스 높이의 30%로 스케일 매핑
                    # `viewport_scale` 값에 계산식 `canvas_h * 0.3`의 결과를 저장합니다. 대상: `viewport_scale`.
                    viewport_scale = canvas_h * 0.3
                    # `px` 값에 `int(canvas_w / 2.0 + raw_x * viewport_scale)` 호출 결과를 저장합니다. 대상: `px`.
                    px = int(canvas_w / 2.0 + (raw_x * viewport_scale))
                    # `py` 값에 `int(canvas_h / 2.0 - raw_y * viewport_scale)` 호출 결과를 저장합니다. 대상: `py`.
                    py = int(canvas_h / 2.0 - (raw_y * viewport_scale))  # +Y가 화면 위쪽이 되도록 반전
                # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                else:
                    # 일반 정규 비율 좌표계 (0.0 ~ 1.0)
                    # `clamped_x` 값에 `min(max(raw_x, 0.0), 1.0)` 호출 결과를 저장합니다. 대상: `clamped_x`.
                    clamped_x = min(max(raw_x, 0.0), 1.0)
                    # `clamped_y` 값에 `min(max(raw_y, 0.0), 1.0)` 호출 결과를 저장합니다. 대상: `clamped_y`.
                    clamped_y = min(max(raw_y, 0.0), 1.0)
                    # `px` 값에 `int(clamped_x * canvas_w)` 호출 결과를 저장합니다. 대상: `px`.
                    px = int(clamped_x * canvas_w)
                    # `py` 값에 `int(clamped_y * canvas_h)` 호출 결과를 저장합니다. 대상: `py`.
                    py = int(clamped_y * canvas_h)

                # `kpt_dict[kp['id']]` 값에 순서·구성을 지정한 `(px, py)`을 저장합니다. 대상: `kpt_dict[kp['id']]`.
                kpt_dict[kp["id"]] = (px, py)

        # 상태에 따른 BGR 색상 팔레트 지정
        # `line_color` 값에 `is_occluded`이면 `(0, 0, 255)`, 아니면 `(0, 255, 255)`을 저장합니다. 대상: `line_color`.
        line_color = (0, 0, 255) if is_occluded else (0, 255, 255)       # 가림: Red, 정상: Yellow
        # `joint_color` 값에 `is_occluded`이면 `(0, 80, 255)`, 아니면 `(0, 165, 255)`을 저장합니다. 대상: `joint_color`.
        joint_color = (0, 80, 255) if is_occluded else (0, 165, 255)     # 가림: Dark Orange, 정상: Bright Orange

        # 1. 뼈대 연결선 렌더링
        # `SKELETON_CONNECTIONS`에서 항목을 차례로 꺼내 `(p1_id, p2_id)`에 넣고 아래 처리를 반복합니다.
        for p1_id, p2_id in SKELETON_CONNECTIONS:
            # 조건 `p1_id in kpt_dict and p2_id in kpt_dict`을 만족하는 경우에만 아래 분기를 실행합니다.
            if p1_id in kpt_dict and p2_id in kpt_dict:
                # `cv2.line(canvas, kpt_dict[p1_id], kpt_dict[p2_id], line_color, 2, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
                cv2.line(canvas, kpt_dict[p1_id], kpt_dict[p2_id], line_color, 2, cv2.LINE_AA)

        # 2. 신체 주요 12개 관절 포인트 렌더링 (얼굴 0~4번 제외)
        # `kpt_dict.items()`에서 항목을 차례로 꺼내 `(j_id, (px, py))`에 넣고 아래 처리를 반복합니다.
        for j_id, (px, py) in kpt_dict.items():
            # 조건 `j_id >= 5`을 만족하는 경우에만 아래 분기를 실행합니다.
            if j_id >= 5:
                # 캔버스 밖으로 벗어난 좌표는 그리기 생략
                # 조건 `0 <= px < canvas_w and 0 <= py < canvas_h`을 만족하는 경우에만 아래 분기를 실행합니다.
                if 0 <= px < canvas_w and 0 <= py < canvas_h:
                    # `cv2.circle(canvas, (px, py), 4, joint_color, -1, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
                    cv2.circle(canvas, (px, py), 4, joint_color, -1, cv2.LINE_AA)

        # 현재 함수 처리를 끝내고 `canvas` 값(`canvas`)을 호출한 쪽으로 반환합니다.
        return canvas