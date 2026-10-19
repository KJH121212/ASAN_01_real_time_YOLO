# ==============================================================================
# [Module Information]
# File: app.py
# Description: Streamlit client for AI rehabilitation motion tracking.
#              Supports 3 verification test modes:
#              1) Visual Mode: Webcam + Full Skeleton & Image Stream (HUD)
#              2) Lightweight Mode: Webcam + Pure Metrics (Unity Preview)
#              3) Offline Video Mode: High-throughput batch test for recorded files
# ==============================================================================

import base64
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import sys
<<<<<<< HEAD
import time
=======
import threading
import traceback
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
import cv2
import numpy as np
import streamlit as st

current_file_path = os.path.abspath(__file__)
root_dir = os.path.dirname(current_file_path)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.overlay_renderer import OverlayRenderer
from network.streamlit_client import FINISH_PACKETS, StreamlitSessionClient
from core.session_log import default_log_root
from network.session_commands import DEFAULT_PREPARATION_SECONDS


def get_root_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_exercise_config() -> tuple[list[str], dict[str, str]]:
    search_paths = [
        os.path.join(get_root_dir(), "exercise_kpt_config_coco.json"),
        os.path.join(get_root_dir(), "configs", "exercise_kpt_config_coco.json"),
        os.path.join(get_root_dir(), "data", "exercise_kpt_config_coco.json"),
    ]
<<<<<<< HEAD
=======

    for p in search_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                exercises = cfg_data.get("exercises", {})
                ex_keys = list(exercises.keys())
                display_map = {
                    k: exercises[k].get("display_name_kr", k) for k in ex_keys
                }
                return ex_keys, display_map
            except Exception as e:
                st.warning(f"설정 파일 파싱 중 오류 발생: {e}")

    fallback_keys = ["biceps_curl", "shoulder_press", "clamshell", "slr", "hip_knee_flexion", "knee_extension"]
    return fallback_keys, {k: k for k in fallback_keys}


class AsyncVideoRecorder:
    def __init__(self, output_path: str, fps: int = 30, size: tuple = (1280, 720)):
        self.output_path = output_path
        self.fps = fps
        self.size = size
        self.queue = queue.Queue(maxsize=150)
        self.is_recording = False
        self.thread = None

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, float(fps), size)

        if not self.writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.output_path = output_path.replace(".mp4", ".avi")
            self.writer = cv2.VideoWriter(self.output_path, fourcc, float(fps), size)

    def start(self):
        self.is_recording = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while self.is_recording or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.05)
                if (frame.shape[1], frame.shape[0]) != self.size:
                    frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_LINEAR)
                self.writer.write(frame)
                self.queue.task_done()
            except queue.Empty:
                continue
        self.writer.release()

    def record_frame(self, frame: np.ndarray):
        if self.is_recording:
            try:
                self.queue.put_nowait(frame)
            except queue.Full:
                pass

    def stop(self):
        self.is_recording = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)


@st.cache_data(ttl=10)
def detect_available_cameras(max_tested: int = 4) -> list[int]:
    available = []
    for idx in range(max_tested):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            available.append(idx)
            cap.release()
    return available if available else [0]


def draw_hud_overlay(
    canvas: np.ndarray,
    current_mode: str,
    left_info: dict,
    right_info: dict,
    target_reps: int,
    calib_step: str = "",
    calib_reps_left: int = 0,
    calib_reps_right: int = 0,
    is_occluded: bool = False,
    is_recording: bool = False,
) -> np.ndarray:
    h, w, _ = canvas.shape

    # 1. 상단 메인 HUD 배너
    top_bar_h = 60
    sub_overlay = canvas.copy()
    cv2.rectangle(sub_overlay, (0, 0), (w, top_bar_h), (10, 10, 10), -1)
    cv2.addWeighted(sub_overlay, 0.70, canvas, 0.30, 0, canvas)
    cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (80, 80, 80), 1)

    if current_mode == "CALIBRATION":
        l_cnt = calib_reps_left
        r_cnt = calib_reps_right
        title_txt = f"CALIB [{calib_step}]  L: {l_cnt}/{target_reps} | R: {r_cnt}/{target_reps}"
        cv2.putText(canvas, title_txt, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)
    else:
        l_cnt = left_info.get("rep_count", 0) if isinstance(left_info, dict) else 0
        r_cnt = right_info.get("rep_count", 0) if isinstance(right_info, dict) else 0
        title_txt = f"REPS  L: {l_cnt}/{target_reps} | R: {r_cnt}/{target_reps}"
        cv2.putText(canvas, title_txt, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 100), 2, cv2.LINE_AA)

    if is_recording:
        cv2.circle(canvas, (w - 75, 30), 7, (0, 0, 255), -1)
        cv2.putText(canvas, "REC", (w - 60, 36), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    # 2. 좌/우 수직 ROM 게이지 바
    bar_w = 26
    bar_h = 180
    y_bottom = h - 50
    y_top = y_bottom - bar_h

    def safe_extract(info):
        if not isinstance(info, dict):
            return 0.0, 0.0
        val = float(info.get("val", 0.0) or 0.0)
        ratio = float(info.get("progress_ratio", 0.0) or 0.0)
        return min(max(ratio, 0.0), 1.0), val

    l_ratio, val_l = safe_extract(left_info)
    r_ratio, val_r = safe_extract(right_info)

    gauges = [
        (l_ratio, val_l, 25, f"L ({l_cnt}/{target_reps})", (255, 160, 20)),
        (r_ratio, val_r, w - 25 - bar_w, f"R ({r_cnt}/{target_reps})", (20, 220, 255))
    ]

    for ratio, val, x_pos, label, bar_color in gauges:
        bg_pad = 8
        card_overlay = canvas.copy()
        cv2.rectangle(card_overlay, (x_pos - bg_pad, y_top - 25), (x_pos + bar_w + bg_pad, y_bottom + 42), (20, 20, 20), -1)
        cv2.addWeighted(card_overlay, 0.60, canvas, 0.40, 0, canvas)
        cv2.rectangle(canvas, (x_pos, y_top), (x_pos + bar_w, y_bottom), (60, 60, 60), -1)

        filled_h = int(bar_h * ratio)
        if filled_h > 0:
            cv2.rectangle(canvas, (x_pos, y_bottom - filled_h), (x_pos + bar_w, y_bottom), bar_color, -1)

        cv2.rectangle(canvas, (x_pos, y_top), (x_pos + bar_w, y_bottom), (220, 220, 220), 2)
        cv2.putText(canvas, label, (x_pos - 12, y_top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{int(ratio * 100)}%", (x_pos - 2, y_bottom + 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{val:.1f}", (x_pos - 4, y_bottom + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

    # 3. 가림 경고
    if is_occluded:
        warn_box = canvas.copy()
        cv2.rectangle(warn_box, (40, h // 2 - 35), (w - 40, h // 2 + 35), (0, 0, 180), -1)
        cv2.addWeighted(warn_box, 0.8, canvas, 0.2, 0, canvas)
        cv2.putText(canvas, "OCCLUSION DETECTED / PAUSED", (w // 2 - 190, h // 2 + 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    return canvas


async def run_client_session(
    uri: str, config: dict, placeholders: dict, renderer: OverlayRenderer, recorder: AsyncVideoRecorder = None
):
    try:
        if recorder:
            recorder.start()

        async with websockets.connect(uri) as ws:
            cmd_packet = {
                "type": "CMD_SET_SESSION",
                "player_id": config["player_id"],
                "patient_name": config["patient_name"],
                "exercise_name": config["exercise_name"],
                "mode": config["mode"],
                "target_reps": config["target_reps"],
                "output_mode": config.get("output_mode", "VISUAL"),
                "camera_index": config.get("camera_index", 0),
                "video_path": config.get("video_path"),
                "batch_size": config.get("batch_size", 16),
            }
            await ws.send(json.dumps(cmd_packet))
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0

    for p in search_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    cfg_data = json.load(f)
                exercises = cfg_data.get("exercises", {})
                ex_keys = list(exercises.keys())
                display_map = {
                    k: exercises[k].get("display_name_kr", k) for k in ex_keys
                }
                return ex_keys, display_map
            except Exception as e:
                st.warning(f"설정 파일 파싱 중 오류 발생: {e}")

<<<<<<< HEAD
    fallback_keys = ["biceps_curl", "shoulder_press", "clamshell", "slr", "hip_knee_flexion", "knee_extension"]
    return fallback_keys, {k: k for k in fallback_keys}




def draw_hud_overlay(
    canvas: np.ndarray,
    current_mode: str,
    left_info: dict,
    right_info: dict,
    target_reps: float,
    calib_step: str = "",
    calib_reps_left: int = 0,
    calib_reps_right: int = 0,
    is_occluded: bool = False,
    is_recording: bool = False,
) -> np.ndarray:
    h, w, _ = canvas.shape

    # 1. 상단 메인 HUD 배너
    top_bar_h = 60
    sub_overlay = canvas.copy()
    cv2.rectangle(sub_overlay, (0, 0), (w, top_bar_h), (10, 10, 10), -1)
    cv2.addWeighted(sub_overlay, 0.70, canvas, 0.30, 0, canvas)
    cv2.line(canvas, (0, top_bar_h), (w, top_bar_h), (80, 80, 80), 1)

    if current_mode == "CALIBRATION":
        l_cnt = calib_reps_left
        r_cnt = calib_reps_right
        title_txt = f"CALIB [{calib_step}]  L: {l_cnt}/{target_reps} | R: {r_cnt}/{target_reps}"
        cv2.putText(canvas, title_txt, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 230, 255), 2, cv2.LINE_AA)
    else:
        l_cnt = left_info.get("total_score", 0) if isinstance(left_info, dict) else 0
        r_cnt = right_info.get("total_score", 0) if isinstance(right_info, dict) else 0
        title_txt = f"SCORE  L: {l_cnt:g}/{target_reps:g} | R: {r_cnt:g}/{target_reps:g}"
        cv2.putText(canvas, title_txt, (20, 42), cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 100), 2, cv2.LINE_AA)

    if is_recording:
        cv2.circle(canvas, (w - 75, 30), 7, (0, 0, 255), -1)
        cv2.putText(canvas, "REC", (w - 60, 36), cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)

    # 2. 좌/우 수직 ROM 게이지 바
    bar_w = 26
    bar_h = 180
    y_bottom = h - 50
    y_top = y_bottom - bar_h

    def safe_extract(info):
        if not isinstance(info, dict):
            return 0.0, 0.0
        val = float(info.get("val", 0.0) or 0.0)
        ratio = float(info.get("progress_ratio", 0.0) or 0.0)
        return min(max(ratio, 0.0), 1.0), val

    l_ratio, val_l = safe_extract(left_info)
    r_ratio, val_r = safe_extract(right_info)

    gauges = [
        (l_ratio, val_l, 25, f"L ({l_cnt}/{target_reps})", (255, 160, 20)),
        (r_ratio, val_r, w - 25 - bar_w, f"R ({r_cnt}/{target_reps})", (20, 220, 255))
    ]

    for ratio, val, x_pos, label, bar_color in gauges:
        bg_pad = 8
        card_overlay = canvas.copy()
        cv2.rectangle(card_overlay, (x_pos - bg_pad, y_top - 25), (x_pos + bar_w + bg_pad, y_bottom + 42), (20, 20, 20), -1)
        cv2.addWeighted(card_overlay, 0.60, canvas, 0.40, 0, canvas)
        cv2.rectangle(canvas, (x_pos, y_top), (x_pos + bar_w, y_bottom), (60, 60, 60), -1)

        filled_h = int(bar_h * ratio)
        if filled_h > 0:
            cv2.rectangle(canvas, (x_pos, y_bottom - filled_h), (x_pos + bar_w, y_bottom), bar_color, -1)

        cv2.rectangle(canvas, (x_pos, y_top), (x_pos + bar_w, y_bottom), (220, 220, 220), 2)
        cv2.putText(canvas, label, (x_pos - 12, y_top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{int(ratio * 100)}%", (x_pos - 2, y_bottom + 18), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{val:.1f}", (x_pos - 4, y_bottom + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)
=======
                    # ----------------------------------------------------------
                    # [모드 3] 비디오 배치 분석 결과 수신
                    # ----------------------------------------------------------
                    if pkt_type == "VIDEO_STATUS":
                        placeholders["alert"].info(f"⏳ {data.get('message')}")
                        continue

                    if pkt_type == "VIDEO_ANALYSIS_COMPLETED":
                        summary = data.get("summary", {})
                        if config["mode"] == "CALIBRATION":
                            th = data.get("thresholds", {})
                            st.session_state.last_calib_thresholds = th
                            placeholders["alert"].success(
                                f"✅ 비디오 캘리브레이션 분석 완료! 총 {data.get('total_frames_analyzed')} 프레임 분석 완료."
                            )
                        else:
                            placeholders["alert"].success(
                                f"🎉 비디오 운동 분석 완료! 총 {summary.get('saved_reps_count', 0)}회 달성 기록됨."
                            )
                        st.session_state.is_running = False
                        break

                    # ----------------------------------------------------------
                    # [모드 1 & 2 공통] 실시간 세션 완료 이벤트
                    # ----------------------------------------------------------
                    if pkt_type in ["SESSION_FINISHED", "CALIBRATION_FINISHED"]:
                        summary = data.get("summary", {})
                        if pkt_type == "CALIBRATION_FINISHED":
                            th = data.get("thresholds", {})
                            st.session_state.last_calib_thresholds = th
                            placeholders["alert"].success("양측 3회 캘리브레이션 완수! 맞춤 임계값이 성공적으로 저장되었습니다.")
                        else:
                            saved_reps = summary.get("saved_reps_count", config["target_reps"])
                            placeholders["alert"].success(f"본 운동 세션 완료! 총 {saved_reps}회 달성 완료")
                        st.session_state.is_running = False
                        break

                    # ----------------------------------------------------------
                    # [실시간 모드] 포즈 및 수치 데이터 처리
                    # ----------------------------------------------------------
                    if pkt_type == "POSE_UPDATE":
                        current_mode = data.get("mode", config["mode"])
                        left_info = data.get("left", {}) or {}
                        right_info = data.get("right", {}) or {}
                        is_occluded = data.get("is_occluded", False)
                        calib_step = data.get("calib_step", "")
                        
                        calib_reps_left = data.get("calib_reps_left", left_info.get("rep_count", 0))
                        calib_reps_right = data.get("calib_reps_right", right_info.get("rep_count", 0))
                        target_reps = 3 if current_mode == "CALIBRATION" else config["target_reps"]
                        fps_val = data.get("fps", 0.0)

                        # --- 모드 1 (VISUAL): 카메라 화면 및 스켈레톤 HUD 렌더링 ---
                        if config["output_mode"] == "VISUAL":
                            img_b64 = data.get("frame_b64")
                            canvas = np.zeros((480, 640, 3), dtype=np.uint8)

                            if img_b64:
                                img_bytes = base64.b64decode(img_b64)
                                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                                if frame is not None:
                                    if recorder:
                                        recorder.record_frame(frame)
                                    canvas = cv2.resize(frame, (640, 480))

                            keypoints = data.get("keypoints", [])
                            canvas = renderer.draw_skeleton(
                                canvas=canvas,
                                keypoints=keypoints,
                                is_occluded=is_occluded,
                                is_cartesian_space=False,
                            )
                            canvas = draw_hud_overlay(
                                canvas=canvas,
                                current_mode=current_mode,
                                left_info=left_info,
                                right_info=right_info,
                                target_reps=target_reps,
                                calib_step=calib_step,
                                calib_reps_left=calib_reps_left,
                                calib_reps_right=calib_reps_right,
                                is_occluded=is_occluded,
                                is_recording=(recorder is not None),
                            )

                            rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                            placeholders["video"].image(rgb_canvas, channels="RGB")

                        # --- 모드 2 (LIGHTWEIGHT): 이미지 없이 수치 카드 고속 갱신 ---
                        else:
                            l_cnt = calib_reps_left if current_mode == "CALIBRATION" else left_info.get("rep_count", 0)
                            r_cnt = calib_reps_right if current_mode == "CALIBRATION" else right_info.get("rep_count", 0)
                            l_ratio = int((left_info.get("progress_ratio", 0.0) or 0.0) * 100)
                            r_ratio = int((right_info.get("progress_ratio", 0.0) or 0.0) * 100)

                            with placeholders["metrics"]:
                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("왼쪽 (Left)", f"{l_cnt} / {target_reps}", f"{left_info.get('quality', 'READY')}")
                                c2.metric("Left ROM", f"{l_ratio}%", f"{left_info.get('state', 'READY')}")
                                c3.metric("오른쪽 (Right)", f"{r_cnt} / {target_reps}", f"{right_info.get('quality', 'READY')}")
                                c4.metric("Right ROM", f"{r_ratio}%", f"{right_info.get('state', 'READY')}")

                        # 상단 알림 배너 갱신
                        if current_mode == "CALIBRATION":
                            if is_occluded:
                                placeholders["alert"].error("가림 감지: 신체가 화면 중앙에 온전히 나오도록 서주세요.")
                            elif calib_step == "COLLECTING":
                                placeholders["alert"].info(
                                    f"측정 진행 중 (양측 각각 3회) ➔ 왼쪽: {calib_reps_left}/{target_reps} | 오른쪽: {calib_reps_right}/{target_reps}"
                                )
                            elif calib_step == "COOLDOWN":
                                placeholders["alert"].success("양측 3회 달성 완수! 임계값 저장 중...")
                        else:
                            l_cnt = left_info.get("rep_count", 0)
                            r_cnt = right_info.get("rep_count", 0)
                            if is_occluded:
                                placeholders["alert"].error("신체 가림 발생: 동작 측정이 일시 중단되었습니다.")
                            elif l_cnt >= target_reps and r_cnt >= target_reps:
                                placeholders["alert"].success(f"양측 목표 완수! (L: {l_cnt}, R: {r_cnt})")
                            else:
                                placeholders["alert"].empty()

                        rec_status = " | REC: ON" if recorder else " | REC: OFF"
                        placeholders["fps"].caption(
                            f"FPS: {fps_val:.1f} | Mode: {current_mode} ({config['output_mode']}){rec_status} | Packets: {packet_count}"
                        )
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0

    # 3. 가림 경고
    if is_occluded:
        warn_box = canvas.copy()
        cv2.rectangle(warn_box, (40, h // 2 - 35), (w - 40, h // 2 + 35), (0, 0, 180), -1)
        cv2.addWeighted(warn_box, 0.8, canvas, 0.2, 0, canvas)
        cv2.putText(canvas, "OCCLUSION DETECTED / PAUSED", (w // 2 - 190, h // 2 + 8), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

<<<<<<< HEAD
    return canvas


def render_pose_update(data, config, placeholders, renderer, packet_count=0, is_recording=False):
    current_mode = data.get("mode", config["mode"])
    left_info = data.get("left", {}) or {}
    right_info = data.get("right", {}) or {}
    is_occluded = data.get("is_occluded", False)
    calib_step = data.get("calib_step", "")

    calib_reps_left = data.get("calib_reps_left", left_info.get("rep_count", 0))
    calib_reps_right = data.get("calib_reps_right", right_info.get("rep_count", 0))
    target_reps = 3 if current_mode == "CALIBRATION" else config.get('target_score', config.get('target_reps', 40))
    fps_val = data.get("fps", 0.0)

    # --- 모드 1 (VISUAL): 카메라 화면 및 스켈레톤 HUD 렌더링 ---
    if config["output_mode"] == "VISUAL":
        img_b64 = data.get("frame_b64")
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

        if img_b64:
            img_bytes = base64.b64decode(img_b64)
            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                canvas = cv2.resize(frame, (640, 480))

        keypoints = data.get("keypoints", [])
        canvas = renderer.draw_skeleton(
            canvas=canvas,
            keypoints=keypoints,
            is_occluded=is_occluded,
            is_cartesian_space=False,
        )
        canvas = draw_hud_overlay(
            canvas=canvas,
            current_mode=current_mode,
            left_info=left_info,
            right_info=right_info,
            target_reps=target_reps,
            calib_step=calib_step,
            calib_reps_left=calib_reps_left,
            calib_reps_right=calib_reps_right,
            is_occluded=is_occluded,
            is_recording=is_recording,
        )

        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        placeholders["video"].image(rgb_canvas, channels="RGB")

    # --- 모드 2 (LIGHTWEIGHT): 이미지 없이 수치 카드 고속 갱신 ---
    else:
        l_cnt = calib_reps_left if current_mode == "CALIBRATION" else left_info.get("total_score", 0)
        r_cnt = calib_reps_right if current_mode == "CALIBRATION" else right_info.get("total_score", 0)
        l_ratio = int((left_info.get("progress_ratio", 0.0) or 0.0) * 100)
        r_ratio = int((right_info.get("progress_ratio", 0.0) or 0.0) * 100)

        with placeholders["metrics"]:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("왼쪽 (Left)", f"{l_cnt:g} / {target_reps:g}" + ('점' if current_mode != 'CALIBRATION' else '회'),
                      f"{left_info.get('quality', 'READY')} · {left_info.get('rep_count', 0)}회")
            c2.metric("Left ROM", f"{l_ratio}%", f"{left_info.get('state', 'READY')}")
            c3.metric("오른쪽 (Right)", f"{r_cnt:g} / {target_reps:g}" + ('점' if current_mode != 'CALIBRATION' else '회'),
                      f"{right_info.get('quality', 'READY')} · {right_info.get('rep_count', 0)}회")
            c4.metric("Right ROM", f"{r_ratio}%", f"{right_info.get('state', 'READY')}")

    # 상단 알림 배너 갱신
    if current_mode == "CALIBRATION":
        if is_occluded:
            placeholders["alert"].error("가림 감지: 신체가 화면 중앙에 온전히 나오도록 서주세요.")
        elif calib_step == "COLLECTING":
            placeholders["alert"].info(
                f"측정 진행 중 (양측 각각 3회) ➔ 왼쪽: {calib_reps_left}/{target_reps} | 오른쪽: {calib_reps_right}/{target_reps}"
            )
        elif calib_step == "COOLDOWN":
            placeholders["alert"].success("양측 3회 달성 완수! 임계값 저장 중...")
    else:
        l_cnt = left_info.get("total_score", 0)
        r_cnt = right_info.get("total_score", 0)
        if is_occluded:
            placeholders["alert"].error("신체 가림 발생: 동작 측정이 일시 중단되었습니다.")
        elif l_cnt > target_reps or r_cnt > target_reps:
            placeholders["alert"].success(f"목표 점수 달성! (L: {l_cnt:g}, R: {r_cnt:g})")
        else:
            placeholders["alert"].empty()

    rec_status = " | REC: ON" if is_recording else " | REC: OFF"
    placeholders["fps"].caption(
        f"FPS: {fps_val:.1f} | Mode: {current_mode} ({config['output_mode']}){rec_status} | Packets: {packet_count}"
    )



def render_rom_editor(client, status):
    editable = client.can_edit_thresholds
    if not status or not status.get("thresholds"):
        st.info("세션을 시작하면 서버의 가동 범위를 불러옵니다.")
        return
    prefix = f"rom_{client.session_id}"
    if st.session_state.get("rom_editor_session") != client.session_id:
        for side, values in status["thresholds"].items():
            for field, value in values.items():
                st.session_state[f"{prefix}_{side}_{field}"] = float(value)
        st.session_state.rom_editor_session = client.session_id

    metric = status.get("metric_type")
    unit = {"ANGLE": "각도 (°)", "COORDINATE": "몸통 길이 기준 상대 높이",
            "LENGTH": "거리 지표", "RATIO": "길이 비율 (Length Ratio)"}.get(metric, metric)
    st.caption(f"단위: {unit} · 현재 세션에 적용")
    st.caption("변경 후 시작 자세로 돌아오면 다음 반복부터 새 범위로 판정합니다.")
    if not editable:
        st.info("가동 범위 수정은 실시간 MAIN / TEST 측정 중에 사용할 수 있습니다.")

    selected = st.radio("수정 대상", ["both", "left", "right"],
                        format_func=lambda side: {"both": "양쪽", "left": "왼쪽", "right": "오른쪽"}[side],
                        horizontal=True, key=f"{prefix}_side", disabled=not editable)
    sides = ("left", "right") if selected == "both" else (selected,)
    sync_col, query_col = st.columns(2)
    if sync_col.button("현재 적용값 불러오기", key=f"{prefix}_load"):
        for side, values in status["thresholds"].items():
            for field, value in values.items():
                st.session_state[f"{prefix}_{side}_{field}"] = float(value)
    if query_col.button("서버 적용값 조회", key=f"{prefix}_query",
                        disabled=not client.running or bool(client.config.get("video_path"))):
        try:
            client.threshold_command()
            st.session_state.rom_notice = ("info", "서버에 현재 가동 범위를 조회했습니다.")
        except (RuntimeError, ValueError) as exc:
            st.session_state.rom_notice = ("error", str(exc))

    with st.form(f"{prefix}_form"):
        columns = st.columns(2)
        draft = {}
        for column, side, label in zip(columns, ("left", "right"), ("왼쪽", "오른쪽")):
            with column:
                st.markdown(f"**{label}**")
                kwargs = {"step": 1.0 if metric == "ANGLE" else 0.01, "format": "%.3f",
                          "disabled": not editable or side not in sides}
                if metric in {"ANGLE", "LENGTH", "RATIO"}:
                    kwargs["min_value"] = 0.0
                if metric == "ANGLE":
                    kwargs["max_value"] = 180.0
                start = st.number_input(f"{label} 시작값", key=f"{prefix}_{side}_start_val", **kwargs)
                target = st.number_input(f"{label} 목표값", key=f"{prefix}_{side}_target_val", **kwargs)
                draft[side] = {"start_val": start, "target_val": target}
        submitted = st.form_submit_button(
            "가동 범위 적용", type="primary",
            disabled=not editable or bool(st.session_state.get("pending_rom_request")))
    if submitted:
        try:
            patch = {side: {key: float(value) for key, value in draft[side].items()} for side in sides}
            request_id = client.threshold_command(patch)
            st.session_state.pending_rom_request = request_id
            st.session_state.pending_rom_at = time.monotonic()
            st.session_state.rom_notice = ("info", "가동 범위 수정 요청을 보냈습니다. 서버 응답을 기다립니다.")
        except (RuntimeError, TypeError, ValueError) as exc:
            st.session_state.rom_notice = ("error", f"입력값을 확인하세요: {exc}")


@st.fragment(run_every=0.2)
def refresh_session(client, placeholders, renderer):
    pending_before = st.session_state.get("pending_rom_request")
    snapshot = client.snapshot()
    for packet in client.drain_events():
        packet_type = packet.get("type")
        if packet_type in {"COUNT_THRESHOLD_STATUS", "COUNT_THRESHOLD_UPDATED"}:
            if packet_type == "COUNT_THRESHOLD_UPDATED":
                if packet.get("request_id") == st.session_state.get("pending_rom_request"):
                    st.session_state.pending_rom_request = None
                message = (f"가동 범위 적용 완료 · 버전 {packet.get('threshold_version')}. "
                           "시작 자세로 복귀한 뒤 다시 움직여 주세요."
                           if packet.get("changed_sides") else "기존 값과 같아 변경하지 않았습니다.")
                st.session_state.rom_notice = ("success", message)
            else:
                st.session_state.rom_notice = ("info", "서버의 현재 가동 범위를 확인했습니다.")
        elif packet_type == "SESSION_PREPARING":
            st.session_state.session_notice = ("info", packet["message"])
        elif packet_type == "SESSION_MEASUREMENT_STARTED":
            st.session_state.session_notice = None
        elif packet_type == "VIDEO_STATUS":
            st.session_state.session_notice = ("info", packet.get("message", "영상 분석 중입니다."))
        elif packet_type in FINISH_PACKETS:
            summary = packet.get("summary") or {}
            scores = snapshot["progress"]["scores"]
            if packet.get("status") == "STOPPED":
                st.session_state.session_notice = ("info", "세션을 중지하고 기록을 저장했습니다.")
                continue
            if packet.get('calibration_completed') is False:
                st.session_state.session_notice = ('warning', '영상이 끝났지만 캘리브레이션은 미완료입니다. 개인 가동 범위는 저장하지 않았습니다.')
                continue
            if packet.get("thresholds"):
                st.session_state.last_calib_thresholds = packet["thresholds"]
            message = ("캘리브레이션 완료. 맞춤 가동 범위를 저장했습니다."
                       if "CALIBRATION" in packet_type or client.config["mode"] == "CALIBRATION"
                       else f"분석 완료 · 왼쪽 {scores['left']:g}점 / "
                            f"오른쪽 {scores['right']:g}점 · "
                            f"반복 기록 {summary.get('saved_reps_count', 0)}개")
            st.session_state.session_notice = ("success", message)
        elif packet_type in {"ERROR", "CLIENT_ERROR"}:
            message = packet.get("message", "서버 통신 오류")
            if packet.get("command") in {"CMD_GET_COUNT_THRESHOLD", "CMD_UPDATE_COUNT_THRESHOLD"}:
                if packet.get("request_id") == st.session_state.get("pending_rom_request"):
                    st.session_state.pending_rom_request = None
                st.session_state.rom_notice = ("error", message)
            else:
                st.session_state.session_notice = ("error", message)

    if (st.session_state.get("pending_rom_request")
            and time.monotonic() - st.session_state.get("pending_rom_at", 0) > 10):
        st.session_state.pending_rom_request = None
        st.session_state.rom_notice = ("warning", "수정 응답이 지연되고 있습니다. 서버 적용값을 조회해 확인하세요.")

    pose = snapshot["pose"]
    if pose:
        render_pose_update(pose, client.config, placeholders, renderer,
                           snapshot["packet_count"], client.config.get("record_video", False))
    else:
        placeholders["fps"].caption(f"연결 상태: {snapshot['status']}")
        server_status = snapshot.get("server_status")
        if server_status and server_status.get("model_state") == "LOADING":
            elapsed = server_status.get("elapsed_seconds", 0)
            placeholders["alert"].info(
                f"서버 연결됨 · {server_status.get('message')} ({elapsed:.1f}초) "
                "준비가 끝나면 세션을 자동으로 시작합니다.")
    notice = st.session_state.get("session_notice")
    if notice:
        getattr(placeholders["alert"], notice[0])(notice[1])
    rom_notice = st.session_state.get("rom_notice")
    if rom_notice:
        getattr(placeholders["rom_feedback"], rom_notice[0])(rom_notice[1])
    log_info = snapshot.get("log_info")
    if log_info:
        with placeholders["log_paths"].container():
            st.caption("서버의 실제 로그 저장 위치")
            st.code(log_info["directory"], language=None)
            st.caption(f"CSV: {Path(log_info['frame_csv']).name} · NPZ: {Path(log_info['skeleton_npz']).name}")
            if log_info.get("video_mp4"):
                st.caption(f"MP4: {Path(log_info['video_mp4']).name}")

    status = snapshot["threshold_status"]
    with placeholders["applied_rom"].container():
        if status and status.get("thresholds"):
            st.caption(f"서버 적용값 · 버전 {status.get('threshold_version')} · {status.get('metric_type')}")
            st.table([{"측면": "왼쪽" if side == "left" else "오른쪽",
                       "시작값": values["start_val"], "목표값": values["target_val"],
                       "가동 범위": abs(values["target_val"] - values["start_val"])}
                      for side, values in status["thresholds"].items()])
        else:
            st.caption("서버 적용값 대기 중")

    # Seed the editor once. Later polling doesn't rerun widgets or networking.
    if status and st.session_state.get("rom_editor_session") != client.session_id:
        st.rerun()
    if pending_before and not st.session_state.get("pending_rom_request"):
        st.rerun()
    if st.session_state.get("is_running") != client.running:
        st.session_state.is_running = client.running
        st.session_state.pending_rom_request = None
        st.rerun()
=======
    except ConnectionRefusedError:
        placeholders["alert"].error("소켓 서버(127.0.0.1:8080)에 연결할 수 없습니다. 터미널에서 서버를 먼저 실행하세요.")
        st.session_state.is_running = False
    except Exception as e:
        traceback.print_exc()
        placeholders["alert"].error(f"통신 오류 발생: {e}")
        st.session_state.is_running = False
    finally:
        if recorder:
            recorder.stop()
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0


def main():
    st.set_page_config(page_title="AI Rehabilitation Motion Studio", layout="wide")
    st.title("AI Rehabilitation Motion System")
<<<<<<< HEAD
    st.caption("세션을 시작하고, 측정 중 환자에게 맞게 좌우 가동 범위를 조정합니다.")
    for key, value in {"is_running": False, "last_calib_thresholds": None,
                       "pending_rom_request": None, "session_notice": None,
                       "rom_notice": None}.items():
        if key not in st.session_state:
            st.session_state[key] = value
    client = st.session_state.get("session_client")
    running = client.running if client else False
    st.session_state.is_running = running
    exercise_keys, display_map = load_exercise_config()
    default_video_path = os.path.join(get_root_dir(), "data", "test", "biceps_curl", "patient_1", "calibration.mp4")

    with st.sidebar:
        st.header("세션 설정")
        uri = st.text_input("Socket Server", value="ws://127.0.0.1:8080", disabled=running)
        test_mode_choice = st.radio(
            "검증 방식 선택",
            ["모드 1: 기존 시각화 (Webcam + Image)",
             "모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)",
             "모드 3: 오프라인 비디오 분석 (Video Batch 검증)"],
            index=0, disabled=running)
        player_id = st.text_input("Patient ID", value="patient_1", disabled=running)
        patient_name = st.text_input("Patient Name", value="김지후", disabled=running)
        log_root = st.text_input("로그 저장 폴더 (서버 기준)", value=str(default_log_root()), disabled=running,
                                 help="분석 결과를 저장할 폴더입니다. 영상 파일 경로는 Video File Path에 입력하세요.")
        st.caption("환자 / 날짜 / 운동 폴더로 저장합니다. 기존 data 폴더는 이동하지 않습니다.")
        session_column, round_column, set_column = st.columns(3)
        session_number = session_column.number_input("Session", min_value=0, value=1, step=1, disabled=running)
        round_number = round_column.number_input("Round", min_value=0, value=1, step=1, disabled=running)
        set_number = set_column.number_input("Set", min_value=0, value=1, step=1, disabled=running)
        exercise_name = st.selectbox("Exercise Name", exercise_keys,
                                    format_func=lambda key: display_map.get(key, key), disabled=running)
        mode = st.radio("Execution Mode", ["CALIBRATION", "TEST", "MAIN"], index=1, disabled=running)
        if mode == "CALIBRATION":
            if exercise_name == 'knee_extension':
                st.info("무릎 펴기: 양발을 내린 채 2초 이상 정지한 뒤, 좌우 각각 3회 천천히 펴고 완전히 내려주세요. 움직임이 불안정하면 준비 자세를 계속 측정합니다.")
            st.info("양측 각각 3회로 개인 맞춤 가동 범위를 측정합니다.")
            target_reps = 3
        else:
            target_reps = st.number_input("목표 점수 (Target Score)", min_value=0.25, max_value=1000.0,
                                         value=40.0, step=0.25, disabled=running)
            st.caption("Perfect +1점 · Good +0.5점 · Bad +0.25점 · 한쪽이 목표 초과 시 1.5초 후 종료")
        chosen_source = None
        batch_size = 16
        camera_index = 0
        preparation_seconds = DEFAULT_PREPARATION_SECONDS
        output_mode = "LIGHTWEIGHT" if "모드 2" in test_mode_choice else "VISUAL"
        if "모드 3" in test_mode_choice:
            video_path = st.text_input("Video File Path", value=default_video_path, disabled=running)
            chosen_source = str(Path(video_path).resolve()) if video_path else None
            batch_size = st.select_slider("Batch Size", [8, 16, 32], value=16, disabled=running)
            st.caption("오프라인 분석 중에는 가동 범위를 수정할 수 없습니다.")
        else:
            # Camera ownership stays on the socket server across UI reruns.
            camera_choice = st.selectbox(
                "카메라 선택 (Camera)", options=[0, 1, 2, 3, 4, 5, "manual"],
                key="camera_choice", disabled=running,
                format_func=lambda value: (
                    "직접 입력 (Custom Index)" if value == "manual"
                    else f"카메라 {value} (Camera {value})"),
                help="서버 PC의 카메라 장치 번호를 선택하세요. 기본 번호는 0입니다.")
            if camera_choice == "manual":
                camera_index = st.number_input(
                    "카메라 번호 (Camera Index)", min_value=0, value=0, step=1,
                    key="custom_camera_index", disabled=running)
            else:
                camera_index = camera_choice
            st.caption("서버 PC의 카메라를 사용합니다. 연결된 장치에 따라 번호가 달라집니다.")
            preparation_seconds = st.number_input(
                "측정 전 준비 시간 (초)", min_value=0.0, value=DEFAULT_PREPARATION_SECONDS,
                step=1.0, disabled=running, help="0으로 설정하면 준비 시간 없이 바로 측정합니다.")
        save_video = st.toggle("영상 녹화 저장", value=False, key="save_video_toggle",
                               disabled=running, help="소켓 서버에서 원본 프레임을 같은 세션 이름의 MP4로 기록합니다.")
        if not running:
=======
    st.markdown("---")

    renderer = OverlayRenderer(conf_threshold=0.35)

    if "is_running" not in st.session_state:
        st.session_state.is_running = False
    if "last_calib_thresholds" not in st.session_state:
        st.session_state.last_calib_thresholds = None

    default_video_path = r"C:\Users\kjh\code\ASAN_01_real_time_YOLO\data\test\biceps_curl\patient_1\calibration.mp4"
    exercise_keys, exercise_display_map = load_exercise_config()

    with st.sidebar:
        st.header("1. Test Pipeline Mode")
        test_mode_choice = st.radio(
            "검증 방식 선택",
            [
                "모드 1: 기존 시각화 (Webcam + Image)",
                "모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)",
                "모드 3: 오프라인 비디오 분석 (Video Batch 검증)",
            ],
            index=0,
        )

        st.markdown("---")
        st.header("2. Session Settings")

        player_id = st.text_input("Patient ID", value="patient_1")
        patient_name = st.text_input("Patient Name", value="김지후")

        exercise_name = st.selectbox(
            "Exercise Name",
            options=exercise_keys,
            index=0,
            format_func=lambda k: exercise_display_map.get(k, k),
        )

        mode = st.radio("Execution Mode", ["CALIBRATION", "TEST"], index=0)

        if mode == "CALIBRATION":
            st.info("양측 모두 3회씩 동작을 완료하여 개인 맞춤형 임계값을 산출합니다.")
            target_reps = 3
        else:
            target_reps = st.number_input(
                "Target Repetitions (TEST)",
                min_value=1,
                max_value=50,
                value=5,
                step=1,
            )

        input_source = None
        camera_index = 0
        output_mode = "VISUAL"

        if "모드 1" in test_mode_choice:
            output_mode = "VISUAL"
            available_cams = detect_available_cameras(max_tested=4)
            camera_index = st.selectbox(
                "Camera Device",
                options=available_cams,
                index=0,
                format_func=lambda idx: (f"Camera {idx} (Default)" if idx == 0 else f"Camera {idx}"),
            )
        elif "모드 2" in test_mode_choice:
            output_mode = "LIGHTWEIGHT"
            available_cams = detect_available_cameras(max_tested=4)
            camera_index = st.selectbox(
                "Camera Device",
                options=available_cams,
                index=0,
                format_func=lambda idx: (f"Camera {idx} (Default)" if idx == 0 else f"Camera {idx}"),
            )
            st.caption("ℹ️ 이미지를 전송하지 않고 초경량 수치만 수신합니다 (Unity 테스트용).")
        else:
            input_source = st.text_input("Video File Path", value=default_video_path)
            batch_size = st.select_slider("Batch Size", options=[8, 16, 32], value=16)

        st.markdown("---")
        st.subheader("Recording Settings")
        save_video = st.toggle(
            "영상 녹화 저장 (Save 720p Video)",
            value=False,
            key="save_video_toggle",
            disabled=("모드 2" in test_mode_choice or "모드 3" in test_mode_choice),
            help="모드 1에서만 지원됩니다.",
        )

        st.markdown("---")

        if not st.session_state.is_running:
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
            if st.button("세션 시작 (Start Session)", type="primary"):
                if "모드 3" in test_mode_choice and (not chosen_source or not os.path.isfile(chosen_source)):
                    st.error("분석할 영상 파일 경로를 확인하세요.")
                elif log_root and Path(log_root).expanduser().is_file():
                    st.error("로그 저장 폴더에는 영상 파일이 아닌 폴더를 입력하세요. 영상 경로는 Video File Path에 입력하세요.")
                else:
                    cfg = dict(player_id=player_id, patient_name=patient_name,
                               exercise_name=exercise_name, mode=mode, target_reps=target_reps,
                               target_score=float(target_reps), finish_delay_seconds=1.5,
                               quality_points={"PERFECT": 1.0, "GOOD": 0.5, "BAD": 0.25},
                               output_mode=output_mode, camera_index=int(camera_index),
                               video_path=chosen_source, batch_size=batch_size, log_root=log_root,
                               session=int(session_number), round=int(round_number), set=int(set_number),
                               cal=mode == "CALIBRATION", record_video=save_video,
                               preparation_seconds=float(preparation_seconds))
                    client = StreamlitSessionClient(uri, cfg)
                    st.session_state.session_client = client
                    st.session_state.session_notice = None
                    st.session_state.rom_notice = None
                    st.session_state.pending_rom_request = None
                    client.start()
                    st.rerun()
        elif st.button("세션 중지 (Stop Session)"):
            client.stop()
            st.session_state.is_running = client.running
            st.session_state.pending_rom_request = None
            st.session_state.session_notice = (
                ("error", "종료 응답을 확인하지 못했습니다. 서버의 기록 저장 상태를 확인하세요.")
                if client.snapshot()["status"] == "ERROR"
                else ("info", "세션을 중지하고 기록 저장을 완료했습니다."))
            st.rerun()
        if st.session_state.last_calib_thresholds:
            st.caption("최근 캘리브레이션 결과")
            st.json(st.session_state.last_calib_thresholds)

<<<<<<< HEAD
    live_column, controls_column = st.columns([3, 2])
    with live_column:
=======
    _, col_center, _ = st.columns([1, 6, 1])
    with col_center:
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0
        alert_box = st.empty()
        video_box = st.empty()
        metrics_box = st.empty()
        fps_box = st.empty()
<<<<<<< HEAD
        log_paths_box = st.empty()
        if not client:
            alert_box.info("소켓 서버 실행 후 세션을 시작하세요.")
    with controls_column:
        st.subheader("가동 범위 조정")
        applied_box = st.empty()
        feedback_box = st.empty()
        if client:
            render_rom_editor(client, client.snapshot()["threshold_status"])
        else:
            st.info("실시간 TEST 또는 MAIN 세션에서 시작값과 목표값을 수정할 수 있습니다.")
    if client:
        refresh_session(client, {
            "alert": alert_box, "video": video_box, "metrics": metrics_box, "fps": fps_box,
            "applied_rom": applied_box, "rom_feedback": feedback_box,
            "log_paths": log_paths_box,
        }, OverlayRenderer(conf_threshold=0.35))
=======

    placeholders = {
        "video": video_box,
        "metrics": metrics_box,
        "alert": alert_box,
        "fps": fps_box,
    }

    if st.session_state.is_running:
        chosen_source = None
        if "모드 3" in test_mode_choice and input_source:
            chosen_source = str(Path(input_source).resolve())
            if os.path.exists(chosen_source):
                st.video(chosen_source)

        cfg = {
            "player_id": player_id,
            "patient_name": patient_name,
            "exercise_name": exercise_name,
            "mode": mode,
            "target_reps": int(target_reps),
            "output_mode": output_mode,
            "camera_index": int(camera_index),
            "video_path": chosen_source,
            "batch_size": 16 if "모드 3" not in test_mode_choice else batch_size,
        }

        recorder = None
        if st.session_state.get("save_video_toggle", False) and output_mode == "VISUAL":
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_filename = f"{mode.lower()}_{player_id}_{exercise_name}_{timestamp}.mp4"
            save_path = os.path.join(get_root_dir(), "recordings", save_filename)
            recorder = AsyncVideoRecorder(output_path=save_path, fps=30, size=(1280, 720))

        asyncio.run(
            run_client_session("ws://127.0.0.1:8080", cfg, placeholders, renderer, recorder=recorder)
        )
>>>>>>> b6c8a449ef52f485f67941f73fc858f4e076adb0


if __name__ == "__main__":
    main()
