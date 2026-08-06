import math  # 삼각함수 연산 (cos -> degree 변환) 모듈[cite: 11]
import time  # 실시간 소요 시간 측정용 모듈[cite: 11]
import numpy as np  # 벡터 연산 및 값 제한(clip)을 위한 NumPy 라이브러리[cite: 11]


# ==============================================================================
# [클래스 1] SideFSM : 좌/우 개별 측면의 유한 상태 머신 (운동 카운팅 및 ROM 평가)[cite: 11]
# ==============================================================================
class SideFSM:

  def __init__(
      self,
      side_name: str,  # 측정 측면 구분 ("left" 또는 "right")[cite: 11]
      motion_direction: str,  # 운동 수치 변화 방향 ("DECREASING" 또는 "INCREASING")[cite: 11]
      thresholds: dict,  # 임계값 설정 (start_val, target_val)[cite: 11]
      is_calibration: bool = False,  # 시범 동작 측정 모드 여부[cite: 11]
      target_reps: int = 3,  # 목표 반복 횟수[cite: 11]
  ):
    # --- [기본 세션 정보 초기화] ---
    self.side = side_name  # 측면 이름 저장[cite: 11]
    self.is_decreasing = (
        motion_direction == "DECREASING"
    )  # 수치 감소형 운동 여부 플래그[cite: 11]
    self.is_calibration = is_calibration  # 시범 모드 여부 저장[cite: 11]
    self.target_reps = target_reps  # 목표 횟수 저장[cite: 11]

    # --- [상태 및 카운팅 변수 초기화] ---
    self.state = "READY"  # FSM 초기 상태 ("READY", "PUSHING", "WAITING")[cite: 11]
    self.rep_count = 0  # 성공한 반복 횟수 카운터[cite: 11]
    self.peak_reached = (
        False  # 최대 가동범위(Peak) 도달 여부 확인 플래그[cite: 11]
    )

    # --- [임계값 설정 (설정값이 없으면 기본값 적용)] ---
    self.start_val = thresholds.get(
        "start_val", 160.0 if self.is_decreasing else 90.0
    )  # 동작 시작 기준 수치[cite: 11]
    self.target_val = thresholds.get(
        "target_val", 50.0 if self.is_decreasing else 160.0
    )  # 동작 목표 기준 수치[cite: 11]

    # --- [회차별 관절 수치 극값 기록용 변수] ---
    self.min_val_recorded = 999.0  # 해당 회차 내 최솟값 추적 변수[cite: 11]
    self.max_val_recorded = -999.0  # 해당 회차 내 최댓값 추적 변수[cite: 11]

    # --- [관절 2D 이동 궤적 범주 기록용 변수] ---
    self.min_x, self.max_x = 999.0, -999.0  # X축 이동 최소/최대 범위[cite: 11]
    self.min_y, self.max_y = 999.0, -999.0  # Y축 이동 최소/최대 범위[cite: 11]

    # --- [성능 및 성과 기록 변수] ---
    self.rep_start_time = None  # 단일 회차 시작 시각[cite: 11]
    self.last_rep_peak = None  # 직전 회차에서 도달한 피크값[cite: 11]
    self.last_quality = (
        "CALIBRATING" if is_calibration else "READY"
    )  # 직전 회차의 수행 품질[cite: 11]

    # --- [데이터 저장용 히스토리 리스트] ---
    self.calibration_history = (
        []
    )  # 시범 동작 측정 결과 저장용 히스토리[cite: 11]
    self.completed_reps_history = (
        []
    )  # 완료된 회차별 상세 측정 기록 (CSV 저장용)[cite: 11]

  # ------------------------------------------------------------------------------
  # [메서드 1-1] evaluate_quality : 피크 달성도 기준 수행 품질(Quality) 평가[cite: 11]
  # ------------------------------------------------------------------------------
  def evaluate_quality(self, peak_val: float) -> str:
    # 시범 동작 모드일 때는 품질을 항상 "CALIBRATING"으로 통일[cite: 11]
    if self.is_calibration:
      return "CALIBRATING"

    # 설정된 전체 목표 가동 범주 계산[cite: 11]
    total_range = abs(self.target_val - self.start_val)
    if total_range == 0:  # 예외 처리: 목표 범위가 0일 경우[cite: 11]
      return "GOOD"

    # 실제 도달한 가동 범주 계산[cite: 11]
    achieved_range = abs(peak_val - self.start_val)
    ratio = achieved_range / total_range  # 목표 대비 달성 비율[cite: 11]

    # 달성 비율별 수행 품질 등급 판정[cite: 11]
    if ratio >= 0.80:  # 80% 이상 달성[cite: 11]
      return "PERFECT"
    elif ratio >= 0.60:  # 60% 이상 달성[cite: 11]
      return "GOOD"
    elif ratio >= 0.40:  # 40% 이상 달성[cite: 11]
      return "BAD"
    else:  # 40% 미만 달성 (미달)[cite: 11]
      return "INVALID"

  # ------------------------------------------------------------------------------
  # [메서드 1-2] update : 매 프레임 수치 기반 상태 전이 및 카운팅 처리[cite: 11]
  # ------------------------------------------------------------------------------
  def update(self, current_val: float, primary_pts: list = None) -> dict:
    # [상황 1] 목표 횟수를 이미 채운 경우 -> WAITING 상태 유지[cite: 11]
    if self.rep_count >= self.target_reps:
      self.state = "WAITING"
      return {
          "state": "WAITING",
          "rep_count": self.rep_count,
          "progress_ratio": 1.0,
          "is_count_updated": False,
          "quality": (
              "FINISHED" if not self.is_calibration else "CALIBRATING"
          ),
      }

    # [상황 2] 관절 유실로 수치가 입력되지 않는 경우 (None)[cite: 11]
    if current_val is None:
      return {
          "state": self.state,
          "rep_count": self.rep_count,
          "progress_ratio": 0.0,
          "is_count_updated": False,
          "quality": self.last_quality,
      }

    # --- [극값 및 이동 궤적 실시간 갱신] ---
    now_time = time.time()  # 현재 시간 수집[cite: 11]
    self.min_val_recorded = min(
        self.min_val_recorded, current_val
    )  # 최소 수치 기록[cite: 11]
    self.max_val_recorded = max(
        self.max_val_recorded, current_val
    )  # 최대 수치 기록[cite: 11]

    # 전달받은 주요 관절 위치 좌표의 최소/최대 영역 갱신[cite: 11]
    if primary_pts:
      for pt in primary_pts:
        if pt and "x" in pt and "y" in pt:
          self.min_x = min(self.min_x, pt["x"])  # X 최소[cite: 11]
          self.max_x = max(self.max_x, pt["x"])  # X 최대[cite: 11]
          self.min_y = min(self.min_y, pt["y"])  # Y 최소[cite: 11]
          self.max_y = max(self.max_y, pt["y"])  # Y 최대[cite: 11]

    # --- [현재 진행률(Progress Ratio) 계산] ---
    val_range = abs(self.target_val - self.start_val)  # 전체 가동 폭[cite: 11]
    progress_ratio = (
        round(
            float(
                np.clip(
                    abs(current_val - self.start_val) / val_range, 0.0, 1.0
                )  # 0.0 ~ 1.0 사이로 클리핑[cite: 11]
            ),
            2,
        )
        if val_range != 0
        else 0.0
    )
    is_count_updated = False  # 현 프레임 카운트 증가 여부[cite: 11]

    # --- [FSM 상태 전이 로직] ---
    # [상태 A] READY : 준비 상태에서 동작 시작(20% 이상 진입) 감지[cite: 11]
    if self.state == "READY":
      if progress_ratio > 0.2:  # 시작 구간(20%) 이탈 시[cite: 11]
        self.state = "PUSHING"  # 운동 수축 진행 상태로 전환[cite: 11]
        self.rep_start_time = now_time  # 회차 시작 시간 수집[cite: 11]

    # [상태 B] PUSHING : 동작 수축 및 원위치 복귀 감지[cite: 11]
    elif self.state == "PUSHING":
      # 가동범위 40% 이상 도달 시 유효 피크 인정[cite: 11]
      if progress_ratio >= 0.40:
        self.peak_reached = True  # 피크 도달 플래그 세팅[cite: 11]

      # 시작점 원위치 복귀(15% 이하) 감지 시 완료 판정[cite: 11]
      if progress_ratio <= 0.15:
        if self.peak_reached:  # 유효 피크를 찍고 돌아온 경우만 인정[cite: 11]
          # 운동 방향에 따라 최고 도달값 결정 (DECREASING: 최솟값, INCREASING: 최댓값)[cite: 11]
          peak_val = (
              self.min_val_recorded
              if self.is_decreasing
              else self.max_val_recorded
          )
          # 수행 품질 평가[cite: 11]
          quality = (
              "CALIBRATING"
              if self.is_calibration
              else self.evaluate_quality(peak_val)
          )

          # INVALID(40% 미달)가 아니면 성공 카운트 등록[cite: 11]
          if quality in ["PERFECT", "GOOD", "BAD", "CALIBRATING"]:
            self.rep_count += 1  # 횟수 1회 증가[cite: 11]
            is_count_updated = True  # 카운트 갱신 플래그 ON[cite: 11]
            self.last_rep_peak = peak_val  # 피크값 저장[cite: 11]
            self.last_quality = quality  # 품질 저장[cite: 11]

            # 소요 시간 및 달성 가동범위(ROM) 계산[cite: 11]
            duration_sec = (
                round(now_time - self.rep_start_time, 2)
                if self.rep_start_time
                else 0.0
            )
            achieved_rom = round(
                abs(self.max_val_recorded - self.min_val_recorded), 2
            )

            # 완료 회차 세부 데이터 기록[cite: 11]
            self.completed_reps_history.append({
                "side": self.side,
                "rep_num": self.rep_count,
                "duration_sec": duration_sec,
                "min_angle": round(self.min_val_recorded, 2),
                "max_angle": round(self.max_val_recorded, 2),
                "achieved_rom": achieved_rom,
                "min_x": round(self.min_x, 3) if self.min_x != 999.0 else 0.0,
                "max_x": round(self.max_x, 3) if self.max_x != -999.0 else 0.0,
                "min_y": round(self.min_y, 3) if self.min_y != 999.0 else 0.0,
                "max_y": round(self.max_y, 3) if self.max_y != -999.0 else 0.0,
                "quality": quality,
            })

            # 시범 동작 분석용 히스토리 기록[cite: 11]
            self.calibration_history.append({
                "min_val": self.min_val_recorded,
                "max_val": self.max_val_recorded,
            })

        # --- [다음 회차를 위한 임시 극값 및 플래그 리셋] ---
        self.min_val_recorded, self.max_val_recorded = (
            999.0,
            -999.0,
        )  # 측정 수치 리셋[cite: 11]
        self.min_x, self.max_x, self.min_y, self.max_y = (
            999.0,
            -999.0,
            999.0,
            -999.0,
        )  # 바운딩 리셋[cite: 11]
        self.peak_reached = False  # 피크 플래그 리셋[cite: 11]

        # 목표 횟수 채움 여부에 따라 다음 상태 결정[cite: 11]
        if self.rep_count >= self.target_reps:
          self.state = "WAITING"  # 세션 종료 대기[cite: 11]
        else:
          self.state = "READY"  # 다음 회차 준비[cite: 11]

    # 최종 상태 객체 반환[cite: 11]
    return {
        "state": self.state,
        "rep_count": self.rep_count,
        "progress_ratio": progress_ratio,
        "is_count_updated": is_count_updated,
        "quality": self.last_quality,
    }


# ==============================================================================
# [클래스 2] MotionEngine : 양측 FSM 총괄 제어 및 파이프라인 엔진[cite: 11]
# ==============================================================================
class MotionEngine:

  def __init__(
      self,
      config: dict,  # 운동 설정 JSON 객체[cite: 11]
      custom_thresholds: dict = None,  # 환자 개인 맞춤 임계값[cite: 11]
      mode: str = "CALIBRATION",  # 동작 모드 ("CALIBRATION" 또는 "MAIN")[cite: 11]
      target_reps: int = 10,  # 목표 운동 횟수[cite: 11]
  ):
    # --- [운동 기본 설정 파싱] ---
    self.config = config  # 전체 구성 저장[cite: 11]
    self.exercise_name = config.get(
        "exercise_name", "unknown"
    )  # 운동 종목명[cite: 11]

    eval_cfg = config.get("eval_config", {})  # 평가 환경 설정[cite: 11]
    self.metric_type = eval_cfg.get(
        "metric_type", "ANGLE"
    )  # 측정 방식 ("RELATIVE_Y" 또는 "ANGLE")[cite: 11]
    self.motion_direction = eval_cfg.get(
        "motion_direction", "DECREASING"
    )  # 수치 변화 방향[cite: 11]
    self.mode = mode  # 엔진 가동 모드[cite: 11]

    default_th = config.get(
        "default_thresholds", {}
    )  # JSON 내 기본 임계값[cite: 11]

    # --- [모드에 따른 임계값 및 목표 횟수 분기 할당] ---
    if self.mode == "CALIBRATION":
      th_left = default_th.get("left", {})  # 시범 모드: 기본 왼쪽 임계값[cite: 11]
      th_right = default_th.get(
          "right", {}
      )  # 시범 모드: 기본 오른쪽 임계값[cite: 11]
      reps_limit = 3  # 시범 모드 고정 3회[cite: 11]
    else:
      # 본 운동 모드: 환자 맞춤 데이터 우선 적용 (없을 경우 기본값)[cite: 11]
      th_left = (
          custom_thresholds.get("left")
          if custom_thresholds
          else default_th.get("left", {})
      )
      th_right = (
          custom_thresholds.get("right")
          if custom_thresholds
          else default_th.get("right", {})
      )
      reps_limit = target_reps  # 사용자 지정 목표 횟수[cite: 11]

    # --- [좌/우 독립 FSM 객체 생성] ---
    is_calib = self.mode == "CALIBRATION"
    self.fsm_left = SideFSM(
        "left",
        self.motion_direction,
        th_left,
        is_calibration=is_calib,
        target_reps=reps_limit,
    )  # 좌측 FSM[cite: 11]
    self.fsm_right = SideFSM(
        "right",
        self.motion_direction,
        th_right,
        is_calibration=is_calib,
        target_reps=reps_limit,
    )  # 우측 FSM[cite: 11]

  # ------------------------------------------------------------------------------
  # [메서드 2-1] _calculate_angle : 3개 관절 2D 벡터 내적 기반 각도 계산[cite: 11]
  # ------------------------------------------------------------------------------
  def _calculate_angle(self, p1: dict, p2: dict, p3: dict) -> float:
    # 관절 p2를 중심으로 하는 두 벡터 v1, v2 생성[cite: 11]
    v1 = np.array([p1["x"] - p2["x"], p1["y"] - p2["y"]])  # p2 -> p1 벡터[cite: 11]
    v2 = np.array([p3["x"] - p2["x"], p3["y"] - p2["y"]])  # p2 -> p3 벡터[cite: 11]

    # 각 벡터의 크기(길이) 계산[cite: 11]
    norm_v1, norm_v2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if norm_v1 == 0 or norm_v2 == 0:  # 예외 처리: 길이 0 분모 방지[cite: 11]
      return 0.0

    # 내적 공식을 통한 코사인 값 산출 및 -1.0~1.0 범주 제한[cite: 11]
    cosine = np.clip(np.dot(v1, v2) / (norm_v1 * norm_v2), -1.0, 1.0)
    # 아크코사인 변환 후 라디안을 육십분법 각도(Degree)로 반올림 반환[cite: 11]
    return round(math.degrees(np.arccos(cosine)), 2)

  # ------------------------------------------------------------------------------
  # [메서드 2-2] _get_primary_pts : 평가 기준 주요 관절 포인트 추출[cite: 11]
  # ------------------------------------------------------------------------------
  def _get_primary_pts(self, side: str, kpt_map: dict) -> list:
    eval_cfg = self.config.get("eval_config", {})  # 평가 구성 로드[cite: 11]
    metric_type = eval_cfg.get("metric_type", "ANGLE")  # 지표 타입 확인[cite: 11]

    # 상대 좌표 방식(RELATIVE_Y)일 경우 단일 타겟 관절 점만 추출 반환[cite: 11]
    if metric_type == "RELATIVE_Y":
      target_id = eval_cfg.get("target_kpt", {}).get(
          side
      )  # 타겟 관절 ID (손목/발목)[cite: 11, 14]
      target_kp = kpt_map.get(target_id)  # 키포인트 조회[cite: 11]
      return [target_kp] if target_kp else []  # 리스트로 반환[cite: 11]

    # 각도 방식(ANGLE)일 경우 관절 3개 리스트 추출 반환[cite: 11]
    else:
      p_indices = eval_cfg.get("primary_kpts", {}).get(
          side, []
      )  # 관절 ID 3개[cite: 11, 14]
      return [
          kpt_map.get(i) for i in p_indices if kpt_map.get(i)
      ]  # 키포인트 객체 매핑[cite: 11]

  # ------------------------------------------------------------------------------
  # [메서드 2-3] _compute_side_value : 측정 타입별 수치 산출 (핵심)[cite: 11]
  # ------------------------------------------------------------------------------
  def _compute_side_value(self, side: str, kpt_map: dict) -> float:
    eval_cfg = self.config.get("eval_config", {})  # 구성 로드[cite: 11]
    metric_type = eval_cfg.get("metric_type", "ANGLE")  # 타입 체크[cite: 11]

    # ==============================================================================
    # [측정 방식 1] RELATIVE_Y : Y축 상대 좌표 추적 방식 (특정 관절 수직 이동량)[cite: 11]
    # ==============================================================================
    # - 적용 운동: 이두 굴곡(Biceps Curl), 숄더 프레스, SLR(하지 직거상) 등[cite: 14]
    # - 산출 수치: 골반 중점(0,0) 스케일 기반 손목/발목의 Y축 위치값 (-2.0 ~ +1.5)[cite: 14]
    # - 동작 특징: 위로 올라갈수록 Y값이 감소, 아래로 내려갈수록 Y값이 증가[cite: 14]
    if metric_type == "RELATIVE_Y":
      # JSON 설정에서 측정할 대상 관절 ID 추출 (예: 왼쪽 손목 9번, 오른쪽 손목 10번)[cite: 11, 14]
      target_id = eval_cfg.get("target_kpt", {}).get(side)
      target_kp = kpt_map.get(target_id)

      # 관절 인식 신뢰도(score)가 0.35 이상일 때만 정규화 Y좌표 반환[cite: 11]
      if target_kp and target_kp.get("score", 0.0) >= 0.35:
        return round(float(target_kp["y"]), 3)
      return None

    # ==============================================================================
    # [측정 방식 2] ANGLE : 3개 관절 사이의 각도 계산 방식 (관절 굴곡/신전 각도)[cite: 11]
    # ==============================================================================
    # - 적용 운동: 슬관절 신전(Knee Extension), 동적 브릿지, 클램쉘 등[cite: 14]
    # - 산출 수치: 3개 관절 벡터 사이의 내적을 이용한 60°~180° 관절 각도(Degree)[cite: 14]
    # - 동작 특징: 굽히거나 펼 때의 순수 관절 꺾임 각도 측정[cite: 14]
    else:
      # JSON 설정에서 각도를 형성하는 3개 관절 점 추출 (예: 어깨-팔꿈치-손목)[cite: 11, 14]
      pts = self._get_primary_pts(side, kpt_map)

      # 3개 관절 모두 신뢰도가 0.4 이상으로 확실히 감지되었을 때 각도 계산[cite: 11]
      if len(pts) == 3 and all(p.get("score", 0.0) >= 0.4 for p in pts):
        return self._calculate_angle(pts[0], pts[1], pts[2])
      return None

  # ------------------------------------------------------------------------------
  # [메서드 2-4] process_keypoints : 실시간 키포인트 파이프라인 처리[cite: 11]
  # ------------------------------------------------------------------------------
  def process_keypoints(self, keypoints: list) -> dict:
    if not keypoints:  # 키포인트 프레임 입력 누락 시 처리 안 함[cite: 11]
      return None

    # COCO 관절 리스트를 ID 기준 딕셔너리로 빠른 매핑 구조 변환[cite: 11]
    kpt_map = {kp["id"]: kp for kp in keypoints}

    # 좌/우 측정값 구하기[cite: 11]
    val_left = self._compute_side_value("left", kpt_map)
    val_right = self._compute_side_value("right", kpt_map)

    # 좌/우 주요 관절 추출[cite: 11]
    pts_left = self._get_primary_pts("left", kpt_map)
    pts_right = self._get_primary_pts("right", kpt_map)

    # 좌/우 개별 FSM 업데이트 진행[cite: 11]
    res_left = self.fsm_left.update(val_left, pts_left)
    res_right = self.fsm_right.update(val_right, pts_right)

    # 시범 동작 모드일 때 양측 3회 완성 시 맞춤 임계값 산출[cite: 11]
    new_custom_thresholds = None
    if self.mode == "CALIBRATION":
      if (
          len(self.fsm_left.calibration_history) >= 3
          and len(self.fsm_right.calibration_history) >= 3
      ):
        new_custom_thresholds = (
            self._generate_bilateral_thresholds()
        )  # 맞춤 Threshold 생성[cite: 11]

    # 최종 결과 보낼 패킷 형성[cite: 11]
    return {
        "mode": self.mode,
        "left": {
            "val": val_left,
            "rep_count": res_left["rep_count"],
            "progress_ratio": res_left["progress_ratio"],
            "state": res_left["state"],
            "quality": res_left["quality"],
            "is_updated": res_left["is_count_updated"],
        },
        "right": {
            "val": val_right,
            "rep_count": res_right["rep_count"],
            "progress_ratio": res_right["progress_ratio"],
            "state": res_right["state"],
            "quality": res_right["quality"],
            "is_updated": res_right["is_count_updated"],
        },
        "new_custom_thresholds": new_custom_thresholds,
    }

  # ------------------------------------------------------------------------------
  # [메서드 2-5] _generate_bilateral_thresholds : 시범동작 평균 기반 맞춤 임계값 계산[cite: 11]
  # ------------------------------------------------------------------------------
  def _generate_bilateral_thresholds(self) -> dict:

    # 한쪽 측면의 3회 시범 동작 기록 평균값을 통한 start/target 산출 내장함수[cite: 11]
    def _calc_side_th(history: list) -> dict:
      h_len = len(history) if history else 1
      avg_min = (
          sum(h["min_val"] for h in history) / h_len
      )  # 3회 동작 중 최솟값들의 평균[cite: 11]
      avg_max = (
          sum(h["max_val"] for h in history) / h_len
      )  # 3회 동작 중 최댓값들의 평균[cite: 11]

      # 수치 감소 운동일 때 (DECREASING): 시작값=최대, 목표값=85% 지점[cite: 11]
      if self.motion_direction == "DECREASING":
        return {
            "start_val": round(avg_max, 1),
            "target_val": round(avg_max - (avg_max - avg_min) * 0.85, 1),
        }
      # 수치 증가 운동일 때 (INCREASING): 시작값=최소, 목표값=85% 지점[cite: 11]
      else:
        return {
            "start_val": round(avg_min, 1),
            "target_val": round(avg_min + (avg_max - avg_min) * 0.85, 1),
        }

    # 양측 좌/우 수치 결과 반환[cite: 11]
    return {
        "left": _calc_side_th(self.fsm_left.calibration_history),
        "right": _calc_side_th(self.fsm_right.calibration_history),
    }