# ==============================================================================
# [파일 정보]
# 파일명: utils/filters.py
# 작성자: 개발자 (Developer)
# 설명: 관절 튐 현상 방지(Clamping) 및 지수 이동 평균(EMA) 기반 실시간 시계열 평활화 필터
# ==============================================================================

import numpy as np  # 벡터 거리 계산 및 수치 조작을 위한 라이브러리 로드


class RealtimeEMAFilter:  # 스트리밍 환경에서 매 프레임 좌표의 이상치를 다듬는 실시간 필터 클래스

    def __init__(self, max_jump: float = 0.15, alpha: float = 0.6):  # 필터 생성자
        self.max_jump = max_jump  # 프레임 간 허용되는 최대 유클리드 이동 거리 임계값 저장
        self.alpha = alpha  # 현재 프레임 반영 비율 (1-alpha는 이전 프레임 반영 비율)
        self.prev_kpts = None  # 직전 프레임의 유효 관절 상태를 저장할 내부 변수 할당

    def update(self, current_kpts: list) -> list:  # 매 프레임 수신되는 17개 관절 좌표를 정제하는 메서드
        if not current_kpts or len(current_kpts) < 17:  # 빈 좌표가 들어올 경우
            return current_kpts  # 처리 불가하므로 원본 즉시 반환

        kpt_array = np.array([[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in current_kpts])  # 수학적 연산을 위해 (17, 3) 2D NumPy 행렬로 변환

        if self.prev_kpts is None:  # 첫 프레임 진입 상태라면
            self.prev_kpts = kpt_array.copy()  # 현재 행렬을 직전 유효 데이터로 초기화 기록
            return current_kpts  # 첫 데이터는 가공 없이 원형 반환

        smoothed_list = []  # 정제된 딕셔너리 구조체들을 담을 새로운 리스트 할당
        for i in range(17):  # 17개의 관절을 개별적으로 순회
            curr = kpt_array[i, :2]  # i번째 관절의 현재 (X, Y) 좌표 추출
            prev = self.prev_kpts[i, :2]  # i번째 관절의 직전 프레임 유효 (X, Y) 좌표 추출
            score = kpt_array[i, 2]  # 현재 감지 신뢰도 유지

            if score < 0.1 or np.all(curr == 0):  # 감지가 유실되었거나 좌표가 0인 경우
                curr = prev  # 값이 비정상적이므로 직전 좌표로 덮어쓰기 (클램핑)
            else:  # 감지가 유효한 경우 이상치 검사 진행
                distance = np.linalg.norm(curr - prev)  # 이전 위치와 현재 위치 사이의 절대 거리 연산
                if distance > self.max_jump:  # 지정된 임계 폭 이상으로 좌표가 급격히 튄 경우 (Jitter)
                    curr = prev + (curr - prev) * (self.max_jump / distance)  # 허용된 최대 폭까지만 이동하도록 벡터 길이 강제 축소

                curr = self.alpha * curr + (1.0 - self.alpha) * prev  # 이상치 제거 후 EMA(지수 이동 평균) 가중치 필터 적용하여 잔떨림 감쇄

            self.prev_kpts[i, :2] = curr  # 다음 프레임 연산을 위해 정제된 현재 좌표를 내부 캐시에 갱신
            smoothed_list.append({  # 기존 자료 구조 규격에 맞게 딕셔너리 포장
                "id": i,  # 관절 ID
                "x": round(float(curr[0]), 4),  # 정제 및 반올림된 X 좌표
                "y": round(float(curr[1]), 4),  # 정제 및 반올림된 Y 좌표
                "score": round(float(score), 3)  # 신뢰도 보존
            })  # 결과 리스트에 등록 완료

        return smoothed_list  # 노이즈가 제거된 깨끗한 관절 리스트 반환