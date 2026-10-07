# ==============================================================================
# File: core/session_controller.py
# Description: 3회 반복 기반 캘리브레이션 및 실시간 데이터 캐시 동기화 제어기
# ==============================================================================

import os
from pathlib import Path
import sys
import time
import numpy as np

current_file_path = Path(__file__).resolve()
core_dir = current_file_path.parent
project_root = core_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.normalization import PoseNormalizer


class SessionController:

    def __init__(self, data_manager=None, motion_engine=None):
        self.data_manager = data_manager
        self.motion_engine = motion_engine
        self.target_calib_reps = 3

        self.mode = "CALIBRATION"
        self.calib_step = "FULL_BODY_CHECK"
        self.timer_start = 0.0

        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)

        self.calib_vals_left = []
        self.calib_vals_right = []
        self.calib_rep_count = 0

        # [추가] TEST 모드 목표 달성 후 쿨다운 타이머
        self.test_finish_timer = None

        self.last_left_data = {
            "val": 0.0,
            "rep_count": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "CALIBRATING",
        }
        self.last_right_data = {
            "val": 0.0,
            "rep_count": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "CALIBRATING",
        }
        print(f"[DEBUG][CONTROLLER] Initialized successfully. Target Calib Reps: {self.target_calib_reps}")

    def check_occlusion(self, keypoints: list) -> bool:
        if not keypoints or len(keypoints) < 17:
            return True
        kpt_map = {kp["id"]: kp.get("score", 0.0) for kp in keypoints}
        # 상체 및 골반 8개 관절 위주 검사
        required_ids = [5, 6, 7, 8, 9, 10, 11, 12]
        return not all(kpt_map.get(i, 0.0) >= 0.35 for i in required_ids)

    def process_calibration_frame(self, raw_keypoints: list, is_occluded: bool) -> dict:
        now = time.time()

        if is_occluded:
            if self.calib_step not in ["FULL_BODY_CHECK", "FINISHED"]:
                print("[WARN][CALIB] Occlusion detected. Resetting to FULL_BODY_CHECK.")
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

        # 1. 감지 대기
        if self.calib_step == "FULL_BODY_CHECK":
            self.calib_step = "COUNTDOWN"
            self.timer_start = now
            print("[DEBUG][CALIB] Full body detected. Starting 3-second countdown.")

        # 2. 3초 카운트다운
        elif self.calib_step == "COUNTDOWN":
            elapsed = now - self.timer_start
            if elapsed >= 3.0:
                self.calib_step = "COLLECTING"
                self.timer_start = now
                self.calib_rep_count = 0
                print(f"[DEBUG][CALIB] Countdown finished. Collecting {self.target_calib_reps} active repetitions.")

        # 3. 데이터 수집 및 1초 쿨다운 제어
        if self.motion_engine and raw_keypoints:
            norm_keypoints = self.normalizer.normalize(raw_keypoints)
            if norm_keypoints:
                kpt_map = {kp["id"]: kp for kp in norm_keypoints}
                vl = self.motion_engine._compute_side_value("left", kpt_map)
                vr = self.motion_engine._compute_side_value("right", kpt_map)

                res_l = self.motion_engine.fsm_left.update(vl)
                res_r = self.motion_engine.fsm_right.update(vr)

                # 실시간 패킷 캐시 갱신
                self.last_left_data = {**res_l, "val": vl}
                self.last_right_data = {**res_r, "val": vr}

                # 수집 중일 때
                if self.calib_step == "COLLECTING":
                    if vl is not None:
                        self.calib_vals_left.append(vl)
                    if vr is not None:
                        self.calib_vals_right.append(vr)

                    current_reps = max(res_l["rep_count"], res_r["rep_count"])
                    if current_reps > self.calib_rep_count:
                        self.calib_rep_count = current_reps
                        print(f"[CALIB HIT] Rep Count: {self.calib_rep_count}/{self.target_calib_reps}")

                    # 3회 달성 시 즉시 끄지 않고 1초 COOLDOWN 돌입
                    if self.calib_rep_count >= self.target_calib_reps:
                        self.calib_step = "COOLDOWN"
                        self.timer_start = now
                        print(f"[CALIB COMPLETE] 3회 달성! 1초간 마무리 프레임을 유지합니다...")

                # 3회 달성 후 1.0초 동안 추가 프레임 계속 전송
                elif self.calib_step == "COOLDOWN":
                    if vl is not None:
                        self.calib_vals_left.append(vl)
                    if vr is not None:
                        self.calib_vals_right.append(vr)

                    if now - self.timer_start >= 1.0:
                        self.calib_step = "FINISHED"
                        print("[CALIB FINISHED] 1초 쿨다운 종료. 세션을 완료합니다.")

        return {
            "calib_step": self.calib_step,
            "calib_rep_count": self.calib_rep_count,
            "target_calib_reps": self.target_calib_reps
        }
    def compute_and_save_thresholds(self, exercise_name: str) -> dict:
        def _calc_side(values: list, side: str) -> dict:
            def_th = self.motion_engine.config.get("default_thresholds", {}).get(side, {})
            fb_start = def_th.get("start_val", 0.15)
            fb_target = def_th.get("target_val", 0.80)

            if not values or len(values) < 15:
                return {"start_val": fb_start, "target_val": fb_target}

            p_min = float(np.percentile(values, 5))
            p_max = float(np.percentile(values, 95))

            if self.motion_engine and self.motion_engine.motion_direction == "DECREASING":
                return {
                    "start_val": round(p_max, 2),
                    "target_val": round(p_max - (p_max - p_min) * 0.85, 2)
                }
            else:
                return {
                    "start_val": round(p_min, 2),
                    "target_val": round(p_min + (p_max - p_min) * 0.85, 2)
                }

        custom_th = {
            "left": _calc_side(self.calib_vals_left, "left"),
            "right": _calc_side(self.calib_vals_right, "right")
        }

        if self.data_manager:
            self.data_manager.save_custom_threshold(exercise_name, custom_th, is_confirmed=True)

        return custom_th

    def process_main_frame(self, raw_keypoints: list, is_occluded: bool) -> tuple:
        """
        본 운동(TEST) 세션 프레임 처리.
        목표 횟수 도달 즉시 종료하지 않고 1.5초간 완수 상태 및 영상 송출을 유지합니다.
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
        reps_reached = (left_data["rep_count"] >= target_reps or right_data["rep_count"] >= target_reps)

        is_finished = False

        # 목표 횟수 달성 시점 감지
        if reps_reached:
            now = time.time()
            if self.test_finish_timer is None:
                self.test_finish_timer = now
                print(f"[DEBUG][TEST] 목표 횟수({target_reps}회) 달성! 1.5초 쿨다운 유지 후 종료합니다.")

            # 1.5초 동안 프레임을 계속 송출하고 대기
            if now - self.test_finish_timer >= 1.5:
                is_finished = True
                print("[DEBUG][TEST] 1.5초 쿨다운 종료. 세션 데이터를 저장합니다.")

        return left_data, right_data, is_finished

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