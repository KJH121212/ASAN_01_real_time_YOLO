# ==============================================================================
# [Module Information]
# File: utils/normalization.py
# Description: Torso-scaled isotropic normalization centering the pelvis at (0, 0)
#              maintaining standard screen coordinate system (Downward is +Y).
# ==============================================================================

import numpy as np


class PoseNormalizer:
    """
    골반 중심을 (0, 0)으로 설정하고 몸통(어깨-골반 거리) 길이를 1.0으로 스케일링하는
    등방성(Isotropic) 2D 포즈 정규화 클래스.
    화면 좌표계(위쪽 0, 아래쪽 +Y)를 그대로 유지하여 팔을 위로 올릴 시 Y값이 감소합니다.
    """

    # COCO 관절 인덱스 정의
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self, conf_threshold: float = 0.35, invert_y: bool = False):
        """
        Args:
            conf_threshold: 기준 관절(골반/어깨)의 최소 감지 신뢰도 임계값
            invert_y: False일 경우 화면 기본 좌표계 유지 (상향 이동 시 Y값 감소)
        """
        self.conf_threshold = conf_threshold
        self.invert_y = invert_y
        print(f"[DEBUG][NORM_INIT] PoseNormalizer configured. ConfThresh: {self.conf_threshold}, InvertY: {self.invert_y}")

    def normalize(self, keypoints: list) -> list:
        """
        17개 관절 딕셔너리 리스트를 받아 신체 중심 정규화 좌표계로 변환합니다.

        Args:
            keypoints: [{'id': int, 'x': float, 'y': float, 'score': float}, ...]
        Returns:
            list: 정규화된 관절 좌표 딕셔너리 리스트 (변환 불가 시 None 반환)
        """
        if not keypoints or len(keypoints) < 17:
            print("[WARN][NORM] Input keypoints list is empty or invalid. Skipping normalization.")
            return None

        # 고속 벡터 연산을 위해 NumPy 배열로 변환: Shape (17, 3) -> [x, y, score]
        kpt_matrix = np.array(
            [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in keypoints],
            dtype=np.float32
        )
        coords = kpt_matrix[:, :2]
        scores = kpt_matrix[:, 2]

        # 필수 기준 관절 유효 감지 여부 검증
        has_hips = (scores[self.LEFT_HIP] >= self.conf_threshold and scores[self.RIGHT_HIP] >= self.conf_threshold)
        has_shoulders = (scores[self.LEFT_SHOULDER] >= self.conf_threshold and scores[self.RIGHT_SHOULDER] >= self.conf_threshold)

        if not has_hips:
            # 골반이 가려진 경우 정규화 기준 원점을 확정할 수 없으므로 무효 처리
            print(f"[WARN][NORM] Pelvis keypoints occluded. Left Hip Score: {scores[self.LEFT_HIP]:.2f}, Right Hip Score: {scores[self.RIGHT_HIP]:.2f}")
            return None

        # 1. 골반 중심 원점(Origin) 계산
        hip_center = (coords[self.LEFT_HIP] + coords[self.RIGHT_HIP]) / 2.0

        # 2. 몸통 길이(Torso Length) 척도 계산
        if has_shoulders:
            shoulder_center = (coords[self.LEFT_SHOULDER] + coords[self.RIGHT_SHOULDER]) / 2.0
            torso_length = float(np.linalg.norm(shoulder_center - hip_center))
        else:
            # 어깨 결측 시 좌우 골반 너비를 기반으로 몸통 길이 대체 추정 (인체 비율 약 1.5배 보정)
            hip_width = float(np.linalg.norm(coords[self.LEFT_HIP] - coords[self.RIGHT_HIP]))
            torso_length = hip_width * 1.5
            print(f"[DEBUG][NORM] Shoulders missing. Estimated torso length from hip width: {torso_length:.4f}")

        # 0으로 나누기 방어
        if torso_length < 1e-4:
            print("[WARN][NORM] Measured scale factor is near zero. Falling back to 1.0.")
            torso_length = 1.0

        # 3. 원점 평행이동 및 등방성 스케일링 수행
        norm_coords = (coords - hip_center) / torso_length

        # 4. Y축 반전 비활성화 (화면 좌표계 유지)
        if self.invert_y:
            norm_coords[:, 1] = -norm_coords[:, 1]

        # 5. 기존 파이프라인 호환 딕셔너리 구조체로 복원
        normalized_list = []
        for i in range(17):
            normalized_list.append({
                "id": i,
                "x": round(float(norm_coords[i, 0]), 4),
                "y": round(float(norm_coords[i, 1]), 4),
                "score": round(float(scores[i]), 3)
            })

        return normalized_list


# 함수형 인터페이스 호환용 래퍼
def normalize_pelvis_centered(keypoints: list, invert_y: bool = False) -> list:
    normalizer = PoseNormalizer(invert_y=invert_y)
    return normalizer.normalize(keypoints)