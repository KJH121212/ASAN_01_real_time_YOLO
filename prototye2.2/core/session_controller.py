# ==============================================================================
# [파일 정보]
# 파일명: core/session_controller.py
# 작성자: 개발자 (Developer)
# 설명: 캘리브레이션 및 본 운동(Test) 세션의 가림(Occlusion) 처리 및 상태 전이 총괄 제어기
# 주요 기능:
#    1. 12개 주요 신체 관절 신뢰도 기반 실시간 전신 가림(Occlusion) 판별
#    2. 캘리브레이션 모드: 가림 발생 시 버퍼 초기화 및 전신 확인 단계로 강제 롤백
#    3. 본 운동(Test) 모드: 가림 발생 시 FSM 연산 일시 정지 및 이전 진행률/카운트 캐시 유지
#    4. 캘리브레이션 종료 시 맞춤 임계값 자동 산출 및 DataManager 영구 저장 연동
# ==============================================================================

# ------------------------------------------------------------------------------
# [코드 설명]
# 본 클래스는 소켓 서버 내 복잡한 조건 분기문을 캡슐화하여, 현재 모드(CALIBRATION / MAIN)에 맞춰
# 프레임 단위의 관절 데이터 처리, 타이머 제어, 임계값 자동 산출 및 상태 유지를 전담합니다.
# ------------------------------------------------------------------------------

import time  # 시간 측정 및 카운트다운 타이머 제어를 위한 time 모듈 로드
import numpy as np  # 백분위수 계산 및 배열 데이터 처리를 위한 NumPy 라이브러리 로드


class SessionController:  # 세션 수명 주기 및 가림 상태 제어 전용 클래스 선언

  def __init__(
      self, data_manager=None, motion_engine=None, calib_duration: float = 10.0
  ):  # 제어기 초기화 생성자 정의
    self.data_manager = data_manager  # 데이터 영구 저장을 담당하는 DataManager 인스턴스 저장
    self.motion_engine = motion_engine  # 운동 평가 및 카운팅을 수행하는 MotionEngine 인스턴스 저장
    self.calib_duration = float(calib_duration)  # 캘리브레이션 데이터 수집 지속 시간(초) 설정

    self.mode = "CALIBRATION"  # 현재 동작 세션 모드 ("CALIBRATION" 또는 "MAIN")
    self.calib_step = "FULL_BODY_CHECK"  # 캘리브레이션 내부 단계 상태 ("FULL_BODY_CHECK", "COUNTDOWN", "COLLECTING", "FINISHED")
    self.timer_start = 0.0  # 카운트다운 및 수집 시간 측정을 위한 기준 타임스탬프

    self.calib_vals_left = []  # 캘리브레이션 중 수집되는 좌측 모션 수치 리스트
    self.calib_vals_right = []  # 캘리브레이션 중 수집되는 우측 모션 수치 리스트

    self.last_left_data = {
        "val": None,
        "rep_count": 0,
        "progress_ratio": 0.0,
        "state": "READY",
        "quality": "CALIBRATING",
    }  # 좌측 상태 캐시
    self.last_right_data = {
        "val": None,
        "rep_count": 0,
        "progress_ratio": 0.0,
        "state": "READY",
        "quality": "CALIBRATING",
    }  # 우측 상태 캐시

  def check_occlusion(
      self, keypoints: list
  ) -> bool:  # 12개 주요 신체 관절 신뢰도 기반 전신 가림 여부 검사 메서드
    if (
        not keypoints or len(keypoints) < 17
    ):  # 키포인트 배열이 비어있거나 17개 미만인 경우
      return True  # 관절 유실로 인한 전신 가림(True) 반환

    kpt_map = {
        kp["id"]: kp.get("score", 0.0) for kp in keypoints
    }  # 관절 ID를 키로 하는 신뢰도 맵핑 생성
    required_ids = [
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
    ]  # 어깨부터 발목까지 12개 주요 관절 ID 리스트 정의
    
    return not all(
        kpt_map.get(i, 0.0) >= 0.35 for i in required_ids
    )  # 하나라도 신뢰도 0.35 미만이면 가림(True) 반환

  def process_calibration_frame(
      self, keypoints: list, is_occluded: bool
  ) -> dict:  # 캘리브레이션 단계별 상태 전이 및 수집 메서드
    now = time.time()  # 현재 시각 타임스탬프 획득
    remaining_sec = 0.0  # 남은 타이머 시간 초기화

    if is_occluded:  # [요구사항 반영] 수집 중 가림 발생 시 처음부터 다시 시작
      self.calib_step = "FULL_BODY_CHECK"  # 상태를 전신 확인 단계로 롤백
      self.timer_start = 0.0  # 타이머 초기화
      self.calib_vals_left.clear()  # 기존 수집된 좌측 불완전 데이터 폐기
      self.calib_vals_right.clear()  # 기존 수집된 우측 불완전 데이터 폐기
      return {
          "calib_step": self.calib_step,
          "remaining_sec": 0.0,
      }  # 롤백 상태 반환

    if (
        self.calib_step == "FULL_BODY_CHECK"
    ):  # 전신이 안정적으로 감지된 대기 상태일 때
      self.calib_step = "COUNTDOWN"  # 3초 카운트다운 단계로 전환
      self.timer_start = now  # 카운트다운 시작 시각 기록

    elif self.calib_step == "COUNTDOWN":  # 3초 카운트다운 진행 중일 때
      elapsed = now - self.timer_start  # 경과 시간 계산
      remaining_sec = max(0.0, 3.0 - elapsed)  # 3초 카운트다운 잔여 시간 연산
      if elapsed >= 3.0:  # 3초가 경과하면
        self.calib_step = "COLLECTING"  # 본 데이터 수집 단계로 전환
        self.timer_start = now  # 수집 지속 시간 측정을 위해 타이머 리셋

    elif self.calib_step == "COLLECTING":  # 본 데이터 수집 진행 중일 때
      elapsed = now - self.timer_start  # 수집 경과 시간 계산
      remaining_sec = max(
          0.0, self.calib_duration - elapsed
      )  # 수집 잔여 시간 연산

      if self.motion_engine and keypoints:  # 모션 엔진 및 관절 데이터가 유효한 경우
        kpt_map = {
            kp["id"]: kp for kp in keypoints
        }  # 관절 ID 딕셔너리 매핑 생성
        vl = self.motion_engine._compute_side_value(
            "left", kpt_map
        )  # 좌측 관절 수치(각도/변위) 산출
        vr = self.motion_engine._compute_side_value(
            "right", kpt_map
        )  # 우측 관절 수치(각도/변위) 산출
        if vl is not None:  # 유효 수치일 경우
          self.calib_vals_left.append(vl)  # 좌측 수집 리스트에 추가
        if vr is not None:  # 유효 수치일 경우
          self.calib_vals_right.append(vr)  # 우측 수집 리스트에 추가

      if elapsed >= self.calib_duration:  # 설정된 수집 시간이 완료된 경우
        self.calib_step = "FINISHED"  # 캘리브레이션 완료 상태로 전이

    return {
        "calib_step": self.calib_step,
        "remaining_sec": round(remaining_sec, 1),
    }  # 현재 캘리브레이션 상태 패킷 반환

  def compute_and_save_thresholds(
      self, exercise_name: str
  ) -> dict:  # 5%/95% 백분위수 기반 85% 가동범위 임계값 산출 및 저장 메서드
    def _calc(values: list) -> dict:  # 단일 측면 임계값 연산 서브 함수 정의
      if (
          not values or len(values) < 10
      ):  # 수집 데이터가 너무 적을 경우 예외 처리
        return {
            "start_val": 160.0,
            "target_val": 60.0,
        }  # 기본 폴백 임계값 반환

      q_min = float(
          np.percentile(values, 5)
      )  # 하위 5% 백분위수를 통한 최소 극값(이상치 제거) 추출
      q_max = float(
          np.percentile(values, 95)
      )  # 상위 5% 백분위수를 통한 최대 극값(이상치 제거) 추출

      if (
          self.motion_engine
          and self.motion_engine.motion_direction == "DECREASING"
      ):  # 수치 감소형 운동일 경우
        return {
            "start_val": round(q_max, 1),  # 최대 이완 상태를 시작점으로 지정
            "target_val": round(
                q_max - (q_max - q_min) * 0.85, 1
            ),  # 85% 수축 지점을 목표값으로 산출
        }  # 임계값 반환
      else:  # 수치 증가형 운동일 경우
        return {
            "start_val": round(q_min, 1),  # 최소 상태를 시작점으로 지정
            "target_val": round(
                q_min + (q_max - q_min) * 0.85, 1
            ),  # 85% 수축 지점을 목표값으로 산출
        }  # 임계값 반환

    custom_th = {
        "left": _calc(self.calib_vals_left),  # 좌측 수집 데이터 기반 임계값 도출
        "right": _calc(self.calib_vals_right),  # 우측 수집 데이터 기반 임계값 도출
    }  # 양측 임계값 딕셔너리 구성

    if (
        self.data_manager
    ):  # 데이터 매니저가 연동되어 있는 경우
      self.data_manager.save_custom_threshold(
          exercise_name, custom_th, is_confirmed=True
      )  # JSON 파일에 임계값 영구 저장

    return custom_th  # 산출된 임계값 딕셔너리 반환

  def process_main_frame(
      self, keypoints: list, is_occluded: bool
  ) -> tuple:  # 본 운동(Test) 실시간 카운팅 및 캐시 유지 메서드
    if (
        is_occluded or not keypoints
    ):  # [요구사항 반영] 본 운동 중 가림 발생 시 FSM 연산을 멈추고 직전 상태 캐시 유지
      return (
          self.last_left_data.copy(),
          self.last_right_data.copy(),
          False,
      )  # 이전 상태 그대로 반환

    result = self.motion_engine.process_keypoints(
        keypoints
    )  # 가림이 없을 때만 FSM 모션 분석 파이프라인 가동
    if not result:  # 분석 결과가 비어있는 경우
      return (
          self.last_left_data.copy(),
          self.last_right_data.copy(),
          False,
      )  # 캐시 반환

    left_data, right_data = (
        result["left"],
        result["right"],
    )  # 최신 분석 패킷 분리
    self.last_left_data = (
        left_data.copy()
    )  # 다음 가림 상황을 대비하여 좌측 캐시 최신화
    self.last_right_data = (
        right_data.copy()
    )  # 다음 가림 상황을 대비하여 우측 캐시 최신화

    target_reps = (
        self.motion_engine.fsm_left.target_reps
        if self.motion_engine
        else 10  # 목표 반복 횟수 조회
    )
    is_finished = (
        left_data["rep_count"] >= target_reps
        or right_data["rep_count"] >= target_reps  # 편측 운동을 고려한 OR 조건 종료 판정
    )

    return left_data, right_data, is_finished  # 최신 분석 결과 및 세션 완료 여부 반환