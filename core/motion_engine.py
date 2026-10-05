# ==============================================================================
# [파일 정보]
# 파일명: core/motion_engine.py
# 작성자: 개발자 (Developer)
# 설명: 관절 좌표 기반 실시간 수치(Angle/Relative Y) 연산 및 유한 상태 머신(FSM) 운동 카운팅 모듈
# ==============================================================================

# ------------------------------------------------------------------------------
# [코드 설명]
# 본 모듈은 SkeletonEngine에서 추출된 17개 관절 픽셀 좌표를 입력받아,
# 사용자의 설정(config)에 따라 3점 내적 각도(Angle) 또는 1점 수직 변위(Relative Y)를 산출합니다.
# 계산된 수치는 좌우 각각 독립적으로 동작하는 SideFSM(유한 상태 머신)에 전달되어,
# 준비(READY) -> 수축(PUSHING) -> 완료/대기(WAITING)의 상태 전이를 거치며 운동 횟수와 품질을 평가합니다.
# ------------------------------------------------------------------------------

import math  # 삼각함수 연산 및 아크코사인 결과를 각도(Degree)로 변환하기 위한 수학 모듈 로드
import numpy as np  # 벡터 생성, 다차원 배열 연산 및 내적 계산을 위한 NumPy 라이브러리 로드


class SideFSM:
    def __init__(self, side_name: str, motion_direction: str, thresholds: dict, target_reps: int = 10):
        self.side = side_name
        self.is_decreasing = (motion_direction == "DECREASING")
        self.target_reps = target_reps

        self.state = "READY"
        self.rep_count = 0
        self.peak_reached = False

        # 기본 임계값 안전 마진 할당
        self.start_val = thresholds.get("start_val", 0.15 if not self.is_decreasing else 0.80)
        self.target_val = thresholds.get("target_val", 0.80 if not self.is_decreasing else 0.15)

        self.min_val_recorded = 999.0
        self.max_val_recorded = -999.0
        self.last_quality = "READY"
        self.completed_reps_history = []

    def evaluate_quality(self, peak_val: float) -> str:
        total_range = abs(self.target_val - self.start_val)
        if total_range == 0:
            return "GOOD"

        achieved_range = abs(peak_val - self.start_val)
        ratio = achieved_range / total_range

        if ratio >= 0.70:
            return "PERFECT"
        elif ratio >= 0.50:
            return "GOOD"
        elif ratio >= 0.30:
            return "BAD"
        else:
            return "INVALID"

    def update(self, current_val: float) -> dict:
        if self.rep_count >= self.target_reps:
            self.state = "WAITING"
            return {
                "state": "WAITING",
                "rep_count": self.rep_count,
                "progress_ratio": 1.0,
                "quality": "FINISHED",
                "is_count_updated": False
            }

        if current_val is None:
            return {
                "state": self.state,
                "rep_count": self.rep_count,
                "progress_ratio": 0.0,
                "quality": self.last_quality,
                "is_count_updated": False
            }

        self.min_val_recorded = min(self.min_val_recorded, current_val)
        self.max_val_recorded = max(self.max_val_recorded, current_val)

        val_range = abs(self.target_val - self.start_val)
        if val_range > 0:
            # 게이지 바와 동일하게 시작점 기준 절대 이동 비율 산출
            progress_ratio = float(np.clip(abs(current_val - self.start_val) / val_range, 0.0, 1.0))
        else:
            progress_ratio = 0.0

        progress_ratio = round(progress_ratio, 2)
        is_count_updated = False

        # 1. 수축 시작 (READY -> PUSHING)
        if self.state == "READY":
            if progress_ratio >= 0.20:
                self.state = "PUSHING"

        # 2. 수축 진행 및 복귀 카운트 (PUSHING -> READY)
        elif self.state == "PUSHING":
            # 35% 이상 올라가면 유효 동작으로 인정
            if progress_ratio >= 0.35:
                self.peak_reached = True

            # 복귀 기준을 0.25로 여유 있게 완화하여 오차 흡수
            if progress_ratio <= 0.25:
                if self.peak_reached:
                    peak_val = self.min_val_recorded if self.is_decreasing else self.max_val_recorded
                    quality = self.evaluate_quality(peak_val)

                    # 35% 이상 가동 후 복귀 시 무효 판정 방지 (최소 BAD 인정)
                    if quality == "INVALID" and self.peak_reached:
                        quality = "BAD"

                    self.rep_count += 1
                    is_count_updated = True
                    self.last_quality = quality

                    self.completed_reps_history.append({
                        "side": self.side,
                        "rep_num": self.rep_count,
                        "duration_sec": 0.0,
                        "min_angle": round(self.min_val_recorded, 2),
                        "max_angle": round(self.max_val_recorded, 2),
                        "achieved_rom": round(abs(self.max_val_recorded - self.min_val_recorded), 2),
                        "quality": quality,
                    })

                    print(f"\n[FSM COUNT UP] {self.side.upper()} Rep: {self.rep_count}/{self.target_reps} | Quality: {quality} | Peak: {peak_val:.2f}")

                # 극값 및 상태 리셋
                self.min_val_recorded = 999.0
                self.max_val_recorded = -999.0
                self.peak_reached = False
                self.state = "WAITING" if self.rep_count >= self.target_reps else "READY"

        return {
            "state": self.state,
            "rep_count": self.rep_count,
            "progress_ratio": progress_ratio,
            "quality": self.last_quality,
            "is_count_updated": is_count_updated
        }

class MotionEngine:  # 좌우 FSM을 통괄하고 관절 좌표에서 수학적 측정값을 도출하는 메인 엔진 클래스

    def __init__(self, config: dict, custom_thresholds: dict = None, mode: str = "MAIN", target_reps: int = 10):  # 모션 엔진 초기화 생성자
        self.config = config  # 외부에서 주입받은 운동별 설정 JSON 딕셔너리 저장
        eval_cfg = config.get("eval_config", {})  # 설정에서 평가 세부 조건 항목만 분리
        
        self.metric_type = eval_cfg.get("metric_type", "ANGLE")  # 측정 방식을 결정하는 타입 값 저장 (ANGLE 또는 RELATIVE_Y)
        self.motion_direction = eval_cfg.get("motion_direction", "DECREASING")  # 수축 시 값이 작아지는지 커지는지 여부 저장
        self.mode = mode  # 엔진의 현재 구동 목적 저장 (CALIBRATION 또는 MAIN)

        default_th = config.get("default_thresholds", {})  # 환자 기록이 없을 때 사용할 설정 파일의 기본 임계값 로드
        th_left = custom_thresholds.get("left") if custom_thresholds else default_th.get("left", {})  # 좌측 맞춤 데이터가 있으면 우선 적용, 없으면 기본값 할당
        th_right = custom_thresholds.get("right") if custom_thresholds else default_th.get("right", {})  # 우측 맞춤 데이터가 있으면 우선 적용, 없으면 기본값 할당

        self.fsm_left = SideFSM("left", self.motion_direction, th_left, target_reps)  # 생성된 임계값으로 왼쪽 팔/다리 전용 FSM 인스턴스 구축
        self.fsm_right = SideFSM("right", self.motion_direction, th_right, target_reps)  # 생성된 임계값으로 오른쪽 팔/다리 전용 FSM 인스턴스 구축

    def _calculate_angle(self, p1: dict, p2: dict, p3: dict) -> float:  # 3개의 관절 좌표점 사이의 내각을 계산하는 내부 메서드
        v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])  # 중앙점(p2)을 기점으로 첫 번째 점(p1)을 향하는 방향 벡터 생성
        v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])  # 중앙점(p2)을 기점으로 세 번째 점(p3)을 향하는 방향 벡터 생성

        norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)  # 생성된 두 벡터의 유클리디안 스칼라 길이(크기) 산출
        if norm_v1 == 0 or norm_v2 == 0:  # 관절이 겹쳐서 벡터 길이가 0이 된 경우 0으로 나누기 에러(ZeroDivision) 방지
            return 0.0  # 계산 불가 시 0도 반환

        cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)  # 두 벡터의 내적을 통해 Cosine 값을 도출하고, 수학적 허용 범위(-1.0 ~ 1.0)로 제한
        return round(math.degrees(np.arccos(cosine)), 2)  # 아크코사인 역함수로 라디안 값을 구한 뒤 일반 각도(Degree)로 변환하고 소수점 2자리 반올림 반환

    def _compute_side_value(self, side: str, kpt_map: dict) -> float:  # 운동 종류에 맞게 각도 또는 위치 변위 측정값을 반환하는 메서드
        eval_cfg = self.config.get("eval_config", {})  # 설정 파일에서 평가 조건 딕셔너리 로드

        if self.metric_type == "RELATIVE_Y":  # 측정 타입이 특정 관절의 Y축 상하 수직 이동을 추적하는 형태일 경우 (예: 이두 컬)
            target_id = eval_cfg.get("target_kpt", {}).get(side)  # 추적 대상으로 지정된 관절 번호(ID) 추출
            target_kp = kpt_map.get(target_id)  # 프레임 내 해당 ID의 관절 객체 정보 확인
            if target_kp and target_kp.get("score", 0.0) >= 0.35:  # 해당 관절이 존재하고 감지 신뢰도가 0.35 이상으로 유효한 경우
                return round(float(target_kp["y"]), 3)  # 해당 관절의 Y 좌표값을 소수점 3자리로 반올림하여 수치로 반환
            return None  # 관절이 화면에서 가려졌을 경우 결측치(None) 반환

        else:  # 측정 타입이 3개 관절의 굽힘 정도를 추적하는 각도 형태일 경우 (예: 스쿼트, 익스텐션)
            p_indices = eval_cfg.get("primary_kpts", {}).get(side, [])  # 각도를 형성하는 3개의 필수 관절 번호 배열 추출
            pts = [kpt_map.get(i) for i in p_indices if kpt_map.get(i)]  # 해당 번호를 가진 실제 관절 객체들을 수집하여 리스트화
            if len(pts) == 3 and all(p.get("score", 0.0) >= 0.4 for p in pts):  # 3개의 관절이 모두 존재하고, 3점 모두 신뢰도 0.4 이상으로 뚜렷하게 보이는 경우
                return self._calculate_angle(pts[0], pts[1], pts[2])  # 3점을 이용해 각도를 계산하고 결과 반환
            return None  # 하나라도 가려지거나 흐릿하면 부정확한 연산을 막기 위해 결측치(None) 반환

    def process_keypoints(self, keypoints: list) -> dict:  # 매 프레임 외부에서 관절 리스트를 받아 FSM을 통과시키는 최종 진입점 메서드
        if not keypoints:  # 전송받은 관절 리스트가 비어있는 경우
            return None  # 처리할 데이터가 없으므로 None 반환

        kpt_map = {kp["id"]: kp for kp in keypoints}  # O(1) 시간 복잡도로 빠르게 검색하기 위해 관절 ID를 키로 하는 딕셔너리로 변환

        val_left = self._compute_side_value("left", kpt_map)  # 좌측 신체에 해당하는 모션 수치(각도/변위) 산출
        val_right = self._compute_side_value("right", kpt_map)  # 우측 신체에 해당하는 모션 수치(각도/변위) 산출

        res_left = self.fsm_left.update(val_left)  # 산출된 수치를 좌측 상태 머신에 주입하고 갱신된 상태 패킷 수령
        res_right = self.fsm_right.update(val_right)  # 산출된 수치를 우측 상태 머신에 주입하고 갱신된 상태 패킷 수령

        return {  # 좌우 엔진의 분석 결과를 하나로 통합하여 반환
            "mode": self.mode,  # 현재 동작 중인 세션 모드 정보 첨부
            "left": {**res_left, "val": val_left},  # 좌측 FSM 결과 패킷에 현재 측정된 실시간 수치(val) 병합
            "right": {**res_right, "val": val_right}  # 우측 FSM 결과 패킷에 현재 측정된 실시간 수치(val) 병합
        }  # 완성된 최종 딕셔너리 구조체 리턴