"""골반 중심을 원점(0, 0)으로 하고 몸통 길이를 1.0으로 맞추는 23개 관절 등방성 정규화 모듈."""

import numpy as np

NUM_KEYPOINTS = 23


class PoseNormalizer:
    """골반 중심 원점·몸통 길이 기준 2D 포즈 정규화 클래스."""

    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_HIP = 11
    RIGHT_HIP = 12

    def __init__(self, conf_threshold: float = 0.35, invert_y: bool = False):
        self.conf_threshold = conf_threshold
        self.invert_y = invert_y

    def normalize(self, keypoints: list) -> list:
        """관절 딕셔너리 리스트(23kpt 또는 17kpt)를 신체 정규화 좌표계로 변환한다."""
        if not keypoints or len(keypoints) < 17:
            return None

        n_kpts = len(keypoints)
        kpt_matrix = np.array(
            [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in keypoints],
            dtype=np.float32,
        )
        coords = kpt_matrix[:, :2]
        scores = kpt_matrix[:, 2]

        aspect_ratio = float(keypoints[0].get("image_aspect_ratio", 1.0))
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            raise ValueError("image_aspect_ratio must be finite and positive")
        coords[:, 0] *= aspect_ratio

        has_hips = (
            scores[self.LEFT_HIP] >= self.conf_threshold
            and scores[self.RIGHT_HIP] >= self.conf_threshold
        )
        has_shoulders = (
            scores[self.LEFT_SHOULDER] >= self.conf_threshold
            and scores[self.RIGHT_SHOULDER] >= self.conf_threshold
        )
        if not has_hips:
            return None

        hip_center = (coords[self.LEFT_HIP] + coords[self.RIGHT_HIP]) / 2.0
        if has_shoulders:
            shoulder_center = (coords[self.LEFT_SHOULDER] + coords[self.RIGHT_SHOULDER]) / 2.0
            torso_length = float(np.linalg.norm(shoulder_center - hip_center))
        else:
            hip_width = float(np.linalg.norm(coords[self.LEFT_HIP] - coords[self.RIGHT_HIP]))
            torso_length = hip_width * 1.5

        if torso_length < 1e-4:
            torso_length = 1.0

        norm_coords = (coords - hip_center) / torso_length
        if self.invert_y:
            norm_coords[:, 1] = -norm_coords[:, 1]

        return [
            {
                "id": i,
                "x": round(float(norm_coords[i, 0]), 4),
                "y": round(float(norm_coords[i, 1]), 4),
                "score": round(float(scores[i]), 3),
            }
            for i in range(n_kpts)
        ]
