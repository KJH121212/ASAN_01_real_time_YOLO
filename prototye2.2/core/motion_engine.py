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


class SideFSM:  # 좌측 또는 우측 단일 측면의 운동 진행 상태를 추적하는 상태 머신 클래스

    def __init__(self, side_name: str, motion_direction: str, thresholds: dict, target_reps: int = 10):  # 상태 머신 초기화 생성자
        self.side = side_name  # 현재 객체가 담당하는 신체 측면 명칭 ("left" 또는 "right") 저장
        self.is_decreasing = (motion_direction == "DECREASING")  # 운동 진행 시 수치가 감소하는 형태인지 논리값으로 저장
        self.target_reps = target_reps  # 현재 세션에서 달성해야 할 목표 반복 횟수 저장

        self.state = "READY"  # FSM의 초기 상태를 동작 준비 완료인 "READY" 상태로 지정
        self.rep_count = 0  # 성공적으로 수행한 누적 운동 횟수를 0으로 초기화
        self.peak_reached = False  # 1회차 운동 중 유효한 최대 가동 지점에 도달했는지 확인하는 플래그 초기화

        self.start_val = thresholds.get("start_val", 160.0 if self.is_decreasing else 90.0)  # JSON 설정 기반 동작 시작점 임계값 할당
        self.target_val = thresholds.get("target_val", 50.0 if self.is_decreasing else 160.0)  # JSON 설정 기반 동작 목표점 임계값 할당

        self.min_val_recorded = 999.0  # 단일 회차 내 가장 작게 측정된 수치를 추적하기 위해 무한대 값으로 초기화
        self.max_val_recorded = -999.0  # 단일 회차 내 가장 크게 측정된 수치를 추적하기 위해 음의 무한대 값으로 초기화
        self.last_quality = "READY"  # 사용자 화면에 표시할 직전 회차의 수행 품질 문자열 초기화

        self.completed_reps_history = []  # 성공적으로 완료된 회차들의 상세 측정 기록(ROM 등)을 누적할 빈 리스트 할당

    def evaluate_quality(self, peak_val: float) -> str:  # 달성된 극값(Peak)을 바탕으로 운동 수행 품질을 4단계로 분류하는 메서드
        total_range = abs(self.target_val - self.start_val)  # 환자에게 요구되는 전체 목표 가동 범위 크기 산출
        if total_range == 0:  # 설정 오류 등으로 목표 가동 범위가 0이 되어 분모가 0이 되는 상황 예외 처리
            return "GOOD"  # 시스템 크래시를 방지하기 위해 기본 유효 등급 반환

        achieved_range = abs(peak_val - self.start_val)  # 시작점으로부터 환자가 실제로 도달한 가동 범위 크기 산출
        ratio = achieved_range / total_range  # 목표치 대비 환자의 실제 달성 비율 계산

        if ratio >= 0.80:  # 달성 비율이 80% 이상인 훌륭한 수행일 경우
            return "PERFECT"  # 최우수 품질 등급 문자열 반환
        elif ratio >= 0.60:  # 달성 비율이 60% 이상인 양호한 수행일 경우
            return "GOOD"  # 우수 품질 등급 문자열 반환
        elif ratio >= 0.40:  # 달성 비율이 40% 이상인 다소 부족한 수행일 경우
            return "BAD"  # 미흡 품질 등급 문자열 반환
        else:  # 달성 비율이 40% 미만으로 움직임이 거의 없는 경우
            return "INVALID"  # 카운트를 인정하지 않는 무효 등급 반환

    def update(self, current_val: float) -> dict:  # 매 프레임마다 산출된 최신 관절 수치를 받아 상태를 전이하는 메서드
        if self.rep_count >= self.target_reps:  # 현재 성공 횟수가 목표 횟수에 이미 도달한 경우
            self.state = "WAITING"  # 추가적인 횟수 증가를 막고 대기 상태로 고정
            return {"state": "WAITING", "rep_count": self.rep_count, "progress_ratio": 1.0, "quality": "FINISHED", "is_count_updated": False}  # 100% 완료 상태 패킷 반환

        if current_val is None:  # 카메라 화면을 벗어나는 등 관절 수치가 입력되지 않은 결측 상황일 경우
            return {"state": self.state, "rep_count": self.rep_count, "progress_ratio": 0.0, "quality": self.last_quality, "is_count_updated": False}  # 기존 상태와 횟수를 그대로 유지하여 반환

        self.min_val_recorded = min(self.min_val_recorded, current_val)  # 들어온 수치 중 가장 작은 값을 지속적으로 갱신
        self.max_val_recorded = max(self.max_val_recorded, current_val)  # 들어온 수치 중 가장 큰 값을 지속적으로 갱신

        val_range = abs(self.target_val - self.start_val)  # 현재 설정된 운동의 전체 가동 범위 크기 계산
        progress_ratio = round(float(np.clip(abs(current_val - self.start_val) / val_range, 0.0, 1.0)), 2) if val_range != 0 else 0.0  # 시작점 대비 현재 위치를 0.0~1.0 사이의 백분율로 제한하여 산출
        is_count_updated = False  # 현재 프레임에서 운동 횟수가 증가했는지를 클라이언트에 알릴 논리 플래그

        if self.state == "READY":  # FSM이 다음 동작을 기다리는 준비 상태인 경우
            if progress_ratio > 0.2:  # 사용자가 시작점으로부터 20% 이상 확실하게 움직임을 시작한 경우
                self.state = "PUSHING"  # 상태를 수축 진행 중(PUSHING)으로 전이

        elif self.state == "PUSHING":  # FSM이 사용자의 근육 수축 동작을 추적 중인 상태인 경우
            if progress_ratio >= 0.40:  # 사용자가 유효 판정 최소 기준인 40% 지점을 돌파한 경우
                self.peak_reached = True  # 이번 횟수를 무효가 아닌 유효한 동작으로 인정하는 플래그 활성화

            if progress_ratio <= 0.15:  # 힘을 빼고 다시 시작점 방향(15% 지점 이하)으로 복귀한 경우
                if self.peak_reached:  # 단순히 깔짝거린 것이 아니라 40% 이상 깊이 도달하고 돌아온 정상 동작일 경우
                    peak_val = self.min_val_recorded if self.is_decreasing else self.max_val_recorded  # 수치 증감 방향에 따라 기록된 극값 중 진짜 피크값 결정
                    quality = self.evaluate_quality(peak_val)  # 도출된 피크값으로 이번 회차의 수행 품질 등급 평가

                    if quality in ["PERFECT", "GOOD", "BAD"]:  # 수행 품질이 최하 등급인 무효(INVALID) 판정이 아닌 경우
                        self.rep_count += 1  # 정상적인 운동 1회 수행으로 인정하여 카운트 증가
                        is_count_updated = True  # 화면 UI에 축하 이펙트를 띄울 수 있도록 갱신 플래그 활성화
                        self.last_quality = quality  # 사용자에게 텍스트로 보여주기 위해 평가된 품질을 캐시에 저장

                        self.completed_reps_history.append({  # 파일 저장을 위해 성공한 회차의 상세 메타데이터 누적
                            "side": self.side,  # 신체 측면 정보
                            "rep_num": self.rep_count,  # 누적된 회차 번호
                            "duration_sec": 0.0,  # 시간 측정값 (필요 시 외부 타임스탬프와 연동 가능)
                            "min_angle": round(self.min_val_recorded, 2),  # 1회 동작 중 기록된 최소 꺾임 수치
                            "max_angle": round(self.max_val_recorded, 2),  # 1회 동작 중 기록된 최대 꺾임 수치
                            "achieved_rom": round(abs(self.max_val_recorded - self.min_val_recorded), 2),  # 실제 달성한 순수 가동 범위
                            "quality": quality,  # 최종 부여된 품질 등급
                        })  # 딕셔너리 정보 리스트 삽입 완료

                self.min_val_recorded, self.max_val_recorded = 999.0, -999.0  # 다음 회차 측정을 위해 극값 추적 변수 초기화
                self.peak_reached = False  # 유효 동작 달성 플래그도 초기화
                self.state = "WAITING" if self.rep_count >= self.target_reps else "READY"  # 목표를 다 채웠으면 대기 상태로, 아니면 다음 준비 상태로 전이

        return {"state": self.state, "rep_count": self.rep_count, "progress_ratio": progress_ratio, "quality": self.last_quality, "is_count_updated": is_count_updated}  # 조립된 현재 상태 패킷 반환


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