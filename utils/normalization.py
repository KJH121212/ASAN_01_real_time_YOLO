# 설명: 골반 중심을 원점으로 하고 몸통 길이를 1로 맞추는 등방성 정규화입니다.
# 기본 화면 좌표는 아래쪽이 +Y이며 invert_y=True이면 위쪽이 +Y가 됩니다.

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

    # 관절을 골반 중심 원점·몸통 길이 단위로 변환합니다. 유효한 기준 관절이 없으면 None을 반환합니다.
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
        # 가로·세로가 다른 영상에서 x/w와 y/h는 서로 다른 척도입니다.
        # 관절 각도나 거리를 계산하기 전에 두 축을 영상 높이 기준 단위로 맞춥니다.
        # 영상 비율 정보가 없는 기존 호출은 기본 비율 1로 기존 좌표를 유지합니다.
        aspect_ratio = float(keypoints[0].get('image_aspect_ratio', 1.0))
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError('image_aspect_ratio must be finite and positive')
        # x/w와 y/h의 축별 척도가 다르므로 x에 w/h를 곱해 두 축을 영상 높이 단위로 맞춥니다. 이 보정이 없으면 세로 영상의 각도·길이가 왜곡됩니다.
        coords[:, 0] *= aspect_ratio

        # 필수 기준 관절 유효 감지 여부 검증
        has_hips = (scores[self.LEFT_HIP] >= self.conf_threshold and scores[self.RIGHT_HIP] >= self.conf_threshold)
        has_shoulders = (scores[self.LEFT_SHOULDER] >= self.conf_threshold and scores[self.RIGHT_SHOULDER] >= self.conf_threshold)

        if not has_hips:
            # 골반이 가려진 경우 정규화 기준 원점을 확정할 수 없으므로 무효 처리
            print(f"[WARN][NORM] Pelvis keypoints occluded. Left Hip Score: {scores[self.LEFT_HIP]:.2f}, Right Hip Score: {scores[self.RIGHT_HIP]:.2f}")
            return None

        # 1. 골반 중심 원점(Origin) 계산
        # 좌우 골반 좌표의 평균을 신체 원점으로 삼아 화면 내 위치가 달라져도 같은 자세를 비교할 수 있게 합니다.
        hip_center = (coords[self.LEFT_HIP] + coords[self.RIGHT_HIP]) / 2.0

        # 2. 몸통 길이(Torso Length) 척도 계산
        if has_shoulders:
            shoulder_center = (coords[self.LEFT_SHOULDER] + coords[self.RIGHT_SHOULDER]) / 2.0
            # 어깨 중심과 골반 중심 사이의 2D 거리를 신체 크기 척도로 사용합니다. 이후 이 길이를 1로 맞추므로 카메라 거리의 영향을 줄입니다.
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
        # 모든 관절에서 골반 중심을 빼고 같은 몸통 길이로 나눕니다. x·y에 같은 배율을 적용하므로 보정된 영상의 각도와 비율을 보존합니다.
        norm_coords = (coords - hip_center) / torso_length

        # 4. Y축 반전 비활성화 (화면 좌표계 유지)
        if self.invert_y:
            # invert_y=True일 때 y 부호를 뒤집습니다. 영상의 아래쪽 양수 좌표가 위쪽 양수 좌표로 바뀌며, 현재 세션 제어기는 이 옵션을 켭니다.
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
