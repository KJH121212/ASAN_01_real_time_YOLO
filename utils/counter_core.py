import numpy as np
import operator
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))  # 상위 디렉토리 경로 추가
from utils.config_loader import load_exercise_configs

# ==========================================
# ⚙️ 1. 연산자 매핑 딕셔너리
# ==========================================
# YAML 설정 파일에서 읽어온 문자열 기호('<', '>')를 
# 파이썬의 실제 비교 연산자 함수로 변환해주는 딕셔너리입니다.
OPERATORS = {
    '<': operator.lt,   # 작다
    '<=': operator.le,  # 작거나 같다
    '>': operator.gt,   # 크다
    '>=': operator.ge   # 크거나 같다
}

# ==========================================
# 🧠 2. 범용 카운터 코어 엔진 클래스
# ==========================================
class UniversalRepetitionCounter:
    def __init__(self, exercise_name, camera_angle):
        """
        초기화 함수: 동작 이름(예: biceps_curl)과 카메라 각도(예: frontal)를 받아
        해당하는 YAML 설정을 로드하고 카운터 상태를 초기화합니다.
        """
        # 공백 등을 대비해 소문자 및 언더스코어 처리 (예: Biceps Curl -> biceps_curl)
        self.exercise_name = exercise_name.lower().replace(" ", "_")
        self.camera_angle = camera_angle.lower()
        
        # 1. config_loader를 통해 모든 운동의 YAML 설정을 가져옴
        all_configs = load_exercise_configs()
        ex_config = all_configs.get(self.exercise_name, {})
        view_config = ex_config.get(self.camera_angle, {})
        
        # 설정이 없으면 에러 발생 (안전 장치)
        if not view_config:
            raise ValueError(f"[오류] 지원하지 않는 설정입니다: [{self.exercise_name} - {self.camera_angle}]")
            
        # 2. 로드된 설정값을 클래스 변수로 할당
        self.calc_method = view_config['calc_method']   # 'angle' 또는 'y_distance'
        self.joints_dict = view_config['joints']        # 좌/우 관절 인덱스 맵
        self.sm_config = view_config['state_machine']   # 상태 전이 임계값 및 조건
        
        # 3. 좌/우(left, right) 독립적인 카운팅 상태 초기화
        self.sides = list(self.joints_dict.keys())
        self.counts = {side: 0 for side in self.sides}                               # 누적 횟수
        self.states = {side: self.sm_config['start_state'] for side in self.sides}   # 현재 상태 (relax/flexion)

    def calculate_2d_angle(self, a, b, c):
        """
        [내부 헬퍼 함수] 세 점(a, b, c)의 좌표를 받아 중심점(b)에서의 2D 내각을 계산합니다.
        """
        a, b, c = np.array(a), np.array(b), np.array(c)
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        ang = np.abs(radians * 180.0 / np.pi)
        return 360 - ang if ang > 180.0 else ang

    def process_frame(self, keypoints):
        """
        [메인 실행 함수] 영상의 1프레임에 해당하는 키포인트 배열(17, 3)을 입력받아
        현재 자세의 상태를 평가하고 카운트를 업데이트합니다.
        
        Returns:
            current_metrics (dict): 현재 프레임의 측정값 (각도 또는 Y-거리)
            count_events (dict): 현재 프레임에서 카운트가 올라갔는지 여부 (True/False)
        """
        current_metrics = {side: None for side in self.sides}
        count_events = {side: False for side in self.sides}
        
        for side in self.sides:
            joints = self.joints_dict[side]
            
            try:
                # ----------------------------------------------------
                # [단계 1] 관절 가시성(Visibility) 체크 최적화
                # ----------------------------------------------------
                # y_distance는 joints[0]과 joints[1]만 확인하면 됨
                if self.calc_method == 'y_distance':
                    if keypoints[joints[0]][2] == 0 or keypoints[joints[1]][2] == 0:
                        continue
                # angle은 세 점(joints[0, 1, 2]) 모두 확인
                elif self.calc_method == 'angle':
                    if any(keypoints[j][2] == 0 for j in joints[:3]):
                        continue

                # ----------------------------------------------------
                # [단계 2] 척도(Metric) 계산
                # ----------------------------------------------------
                if self.calc_method == 'angle':
                    pt1, pt2, pt3 = keypoints[joints[0]][:2], keypoints[joints[1]][:2], keypoints[joints[2]][:2]
                    metric = self.calculate_2d_angle(pt1, pt2, pt3)
                elif self.calc_method == 'y_distance':
                    # 어깨(joints[0])와 손목(joints[1])의 수직 거리
                    pt1, pt2 = keypoints[joints[0]][:2], keypoints[joints[1]][:2]
                    metric = abs(pt1[1] - pt2[1])
                else:
                    continue
                
                current_metrics[side] = metric
                
                # ----------------------------------------------------
                # [단계 3] 상태 머신(State Machine) 평가 및 전이
                # ----------------------------------------------------
                cur_state = self.states[side]
                start_st = self.sm_config['start_state']     # 예: 'relax'
                act_st = self.sm_config['active_state']      # 예: 'flexion'
                trig_act = self.sm_config['trigger_active']  # 수축 진입 조건
                trig_start = self.sm_config['trigger_start'] # 이완(완료) 진입 조건
                
                # (1) 이완(Relax) 상태에서 -> 수축(Flexion) 임계값을 넘었는지 확인
                if cur_state == start_st:
                    # 설정 파일에 명시된 부등호 기호('<')를 가져와서 비교
                    if OPERATORS[trig_act['operator']](metric, trig_act['threshold']):
                        self.states[side] = act_st # 상태 업데이트
                        
                # (2) 수축(Flexion) 상태에서 -> 다시 이완(Relax) 임계값을 넘었는지 확인 (1회 완료)
                elif cur_state == act_st:
                    if OPERATORS[trig_start['operator']](metric, trig_start['threshold']):
                        self.states[side] = start_st  # 상태 초기화
                        self.counts[side] += 1        # 🌟 카운트 1 증가
                        count_events[side] = True     # 외부 알림용 이벤트 마커 발생
                        
            except IndexError:
                # 관절 배열 인덱스를 벗어나는 비정상 프레임인 경우 무시
                continue

        return current_metrics, count_events