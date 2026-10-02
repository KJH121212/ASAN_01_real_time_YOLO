# ==============================================================================
# [파일 정보]
# 파일명: core/session_controller.py
# 설명: 3회 반복 기반 캘리브레이션 수집 및 본 운동(TEST) 세션 흐름 제어기
# ==============================================================================

import os
from pathlib import Path
import sys
import time
import numpy as np

# 프로젝트 루트 경로 확인 및 시스템 패스 등록
current_file_path = Path(__file__).resolve()
core_dir = current_file_path.parent
project_root = core_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 올바른 정규화기 모듈 임포트 (utils/normalization.py)
from utils.normalization import PoseNormalizer


class SessionController:
    """
    전신 가림(Occlusion) 판별, 3회 동작 기반 캘리브레이션 FSM 수집 및
    본 운동 세션의 실시간 평가를 총괄 관리하는 오케스트레이터 클래스.
    """

    def __init__(self, data_manager=None, motion_engine=None):
        """
        세션 제어기 초기화.
        
        Args:
            data_manager: 임계값 및 궤적 파일 저장을 전담하는 DataManager 인스턴스
            motion_engine: 관절 수치 연산 및 FSM 평가를 담당하는 MotionEngine 인스턴스
        """
        self.data_manager = data_manager
        self.motion_engine = motion_engine
        self.target_calib_reps = 3  # 3회 고정 캘리브레이션 규격

        self.mode = "CALIBRATION"
        self.calib_step = "FULL_BODY_CHECK"
        self.timer_start = 0.0

        # 데카르트 정규화기 인스턴스화 (골반 중심 0,0, 상향 +Y 반전)
        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)

        # 3회 반복 동안 수집되는 관절 측정값 시계열 버퍼
        self.calib_vals_left = []
        self.calib_vals_right = []
        self.calib_rep_count = 0

        # 가림 상태 대비 직전 프레임 캐시 구조체
        self.last_left_data = {
            "val": None,
            "rep_count": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "CALIBRATING",
        }
        self.last_right_data = {
            "val": None,
            "rep_count": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "CALIBRATING",
        }

        print(f"[DEBUG][CONTROLLER] Initialized successfully. Target Calib Reps: {self.target_calib_reps}")

    def check_occlusion(self, keypoints: list) -> bool:
        """
        어깨부터 발목까지 주요 12개 관절의 신뢰도를 검증하여 신체 가림 여부를 판별합니다.
        
        Args:
            keypoints: 17개 관절 딕셔너리 리스트
            
        Returns:
            bool: 하나라도 신뢰도 0.35 미만일 경우 True (가림 발생)
        """
        if not keypoints or len(keypoints) < 17:
            return True

        kpt_map = {kp["id"]: kp.get("score", 0.0) for kp in keypoints}
        required_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        
        is_occluded = not all(kpt_map.get(i, 0.0) >= 0.35 for i in required_ids)
        return is_occluded

    def process_calibration_frame(self, raw_keypoints: list, is_occluded: bool) -> dict:
        """
        시간 기반이 아닌 실시간 3회 동작 완료 시점까지 관절 데이터를 수집하고 단계를 전이합니다.
        
        Args:
            raw_keypoints: 비전 추론 모델에서 추출된 17개 관절 리스트
            is_occluded: 신체 가림 판별 플래그
            
        Returns:
            dict: 캘리브레이션 세부 단계 및 달성 횟수 패킷
        """
        now = time.time()

        # 가림 발생 시 데이터 무결성을 위해 카운트 및 수집 버퍼를 리셋하고 대기 단계로 롤백
        if is_occluded:
            if self.calib_step != "FULL_BODY_CHECK":
                print("[WARN][CALIB] Occlusion detected. Resetting calibration progress to FULL_BODY_CHECK.")
            self.calib_step = "FULL_BODY_CHECK"
            self.timer_start = 0.0
            self.calib_vals_left.clear()
            self.calib_vals_right.clear()
            self.calib_rep_count = 0
            if self.motion_engine:
                self.motion_engine.fsm_left.rep_count = 0
                self.motion_engine.fsm_right.rep_count = 0
                self.motion_engine.fsm_left.state = "READY"
                self.motion_engine.fsm_right.state = "READY"
            return {
                "calib_step": self.calib_step,
                "calib_rep_count": 0,
                "target_calib_reps": self.target_calib_reps
            }

        # 1. 전신 감지 대기 단계
        if self.calib_step == "FULL_BODY_CHECK":
            self.calib_step = "COUNTDOWN"
            self.timer_start = now
            print("[DEBUG][CALIB] Full body detected. Starting 3-second countdown.")

        # 2. 동작 개시 전 3초 카운트다운
        elif self.calib_step == "COUNTDOWN":
            elapsed = now - self.timer_start
            if elapsed >= 3.0:
                self.calib_step = "COLLECTING"
                self.timer_start = now
                self.calib_rep_count = 0
                print(f"[DEBUG][CALIB] Countdown finished. Collecting {self.target_calib_reps} active repetitions.")

        # 3. 3회 동작 수행 및 시계열 수치 수집 단계
        elif self.calib_step == "COLLECTING":
            if self.motion_engine and raw_keypoints:
                # 관절 좌표를 골반 중심 데카르트 좌표계로 정규화
                norm_keypoints = self.normalizer.normalize(raw_keypoints)
                if norm_keypoints:
                    kpt_map = {kp["id"]: kp for kp in norm_keypoints}
                    vl = self.motion_engine._compute_side_value("left", kpt_map)
                    vr = self.motion_engine._compute_side_value("right", kpt_map)

                    if vl is not None:
                        self.calib_vals_left.append(vl)
                    if vr is not None:
                        self.calib_vals_right.append(vr)

                    # FSM 상태 머신 업데이트 및 횟수 평가
                    res_l = self.motion_engine.fsm_left.update(vl)
                    res_r = self.motion_engine.fsm_right.update(vr)

                    # 좌우 측면 중 최댓값 기준으로 유효 반복 횟수 갱신
                    current_reps = max(res_l["rep_count"], res_r["rep_count"])
                    if current_reps > self.calib_rep_count:
                        self.calib_rep_count = current_reps
                        print(f"[DEBUG][CALIB] Completed Rep: {self.calib_rep_count}/{self.target_calib_reps}")

                    # 3회 반복 완수 시 완료 상태로 전이
                    if self.calib_rep_count >= self.target_calib_reps:
                        self.calib_step = "FINISHED"
                        print("[DEBUG][CALIB] 3 Repetitions fulfilled. Stage set to FINISHED.")

        return {
            "calib_step": self.calib_step,
            "calib_rep_count": self.calib_rep_count,
            "target_calib_reps": self.target_calib_reps
        }

    def compute_and_save_thresholds(self, exercise_name: str) -> dict:
        """
        3회 반복 수집 데이터에서 이상치를 배제한 뒤 85% 가동 범위 임계값을 산출하여 저장합니다.
        """
        def _calc_side(values: list, side: str) -> dict:
            def_th = self.motion_engine.config.get("default_thresholds", {}).get(side, {})
            fb_start = def_th.get("start_val", 0.0)
            fb_target = def_th.get("target_val", 1.0)

            if not values or len(values) < 15:
                print(f"[WARN][CALIB] Insufficient data frames ({len(values)}). Using fallback values.")
                return {"start_val": fb_start, "target_val": fb_target}

            # 5%, 95% 백분위수를 사용하여 측정 노이즈 제거
            p_min = float(np.percentile(values, 5))
            p_max = float(np.percentile(values, 95))

            if self.motion_engine and self.motion_engine.motion_direction == "DECREASING":
                # 수축 시 값이 감소하는 운동
                return {
                    "start_val": round(p_max, 2),
                    "target_val": round(p_max - (p_max - p_min) * 0.85, 2)
                }
            else:
                # 수축 시 값이 증가하는 운동
                return {
                    "start_val": round(p_min, 2),
                    "target_val": round(p_min + (p_max - p_min) * 0.85, 2)
                }

        custom_th = {
            "left": _calc_side(self.calib_vals_left, "left"),
            "right": _calc_side(self.calib_vals_right, "right")
        }

        print(f"[DEBUG][POST_PROCESS] Computed Custom Thresholds for '{exercise_name}': {custom_th}")

        if self.data_manager:
            self.data_manager.save_custom_threshold(exercise_name, custom_th, is_confirmed=True)

        return custom_th

    def process_main_frame(self, raw_keypoints: list, is_occluded: bool) -> tuple:
        """
        본 운동(TEST) 세션 프레임 처리. 가림 시 이전 캐시를 유지하고 정상 시 FSM을 평가합니다.
        """
        if is_occluded or not raw_keypoints:
            return self.last_left_data.copy(), self.last_right_data.copy(), False

        norm_keypoints = self.normalizer.normalize(raw_keypoints)
        if not norm_keypoints:
            return self.last_left_data.copy(), self.last_right_data.copy(), False

        result = self.motion_engine.process_keypoints(norm_keypoints)
        if not result:
            return self.last_left_data.copy(), self.last_right_data.copy(), False

        left_data, right_data = result["left"], result["right"]
        self.last_left_data = left_data.copy()
        self.last_right_data = right_data.copy()

        target_reps = self.motion_engine.fsm_left.target_reps if self.motion_engine else 10
        is_finished = (left_data["rep_count"] >= target_reps or right_data["rep_count"] >= target_reps)

        return left_data, right_data, is_finished