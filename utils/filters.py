# ==============================================================================
# [Module Information]
# File: utils/filters.py
# Description: Real-time temporal smoothing filter with Euclidean jump clamping
#              and Exponential Moving Average (EMA) for joint jitter reduction.
# ==============================================================================

import numpy as np


class RealtimeEMAFilter:
    """
    프레임 간 비정상적인 관절 튐(Jitter) 현상을 유클리디안 거리 기준으로 클램핑하고
    지수 이동 평균(EMA)을 통해 시계열 잔떨림을 완화하는 실시간 필터.
    """

    def __init__(self, max_jump: float = 0.15, alpha: float = 0.6):
        """
        Args:
            max_jump: 단일 프레임 간 허용되는 최대 유클리디안 변위 거리 (정규 비율 좌표 기준)
            alpha: 현재 프레임 반영 비율 (1.0에 가까울수록 민감, 낮을수록 스무딩 강화)
        """
        self.max_jump = max_jump
        self.alpha = alpha
        self.prev_kpts = None  # 직전 프레임 유효 관절 배열 Shape: (17, 2)
        print(f"[DEBUG][FILTER_INIT] EMA Filter active. Max Jump: {self.max_jump}, Alpha: {self.alpha}")

    def update(self, current_kpts: list) -> list:
        """
        17개 관절 좌표 리스트에 필터를 적용하여 노이즈가 제거된 리스트를 반환합니다.

        Args:
            current_kpts: [{'id': int, 'x': float, 'y': float, 'score': float}, ...]
        Returns:
            list: 평활화된 관절 좌표 리스트
        """
        if not current_kpts or len(current_kpts) < 17:
            return current_kpts

        # 관절 배열 추출: Shape (17, 3) -> [x, y, score]
        kpt_array = np.array(
            [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in current_kpts],
            dtype=np.float32
        )
        curr_coords = kpt_array[:, :2]
        scores = kpt_array[:, 2]

        # 1. 최초 진입 시 초기화 버퍼 생성
        if self.prev_kpts is None:
            self.prev_kpts = curr_coords.copy()
            print(f"[DEBUG][FILTER] Initialized history buffer. Joint count: {len(current_kpts)}")
            return current_kpts

        smoothed_list = []
        jitter_detected_count = 0

        # 2. 17개 관절별 독립 시계열 필터링 수행
        for i in range(17):
            curr_pt = curr_coords[i].copy()
            prev_pt = self.prev_kpts[i].copy()
            score = scores[i]

            # 신뢰도가 매우 낮거나 좌표가 0인 결측 상황 처리
            if score < 0.15 or np.all(curr_pt == 0.0):
                curr_pt = prev_pt
            else:
                displacement = float(np.linalg.norm(curr_pt - prev_pt))
                # 급격한 좌표 튐(Jitter) 감지 시 벡터 클램핑
                if displacement > self.max_jump:
                    jitter_detected_count += 1
                    curr_pt = prev_pt + (curr_pt - prev_pt) * (self.max_jump / displacement)

                # EMA(지수 이동 평균) 가중치 적용
                curr_pt = self.alpha * curr_pt + (1.0 - self.alpha) * prev_pt

            # 다음 프레임 연산을 위해 캐시 최신화
            self.prev_kpts[i] = curr_pt

            smoothed_list.append({
                "id": i,
                "x": round(float(curr_pt[0]), 4),
                "y": round(float(curr_pt[1]), 4),
                "score": round(float(score), 3)
            })

        if jitter_detected_count > 0:
            print(f"[DEBUG][FILTER] Clamped {jitter_detected_count} jittering joints in current frame.")

        return smoothed_list

    def reset(self):
        """연속성이 단절되거나 세션이 변경될 때 이전 프레임 캐시를 초기화합니다."""
        self.prev_kpts = None
        print("[DEBUG][FILTER] Filter history cache flushed.")