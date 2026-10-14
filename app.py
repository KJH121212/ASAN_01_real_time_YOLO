# ==============================================================================
# [Module Information]
# File: app.py
# Description: Streamlit client for AI rehabilitation motion tracking.
#              Supports 3 verification test modes:
#              1) Visual Mode: Webcam + Full Skeleton & Image Stream (HUD)
#              2) Lightweight Mode: Webcam + Pure Metrics (Unity Preview)
#              3) Offline Video Mode: High-throughput batch test for recorded files
# ==============================================================================

import asyncio
import base64
from datetime import datetime
import json
import os
from pathlib import Path
import queue
import sys
import threading
import traceback
import cv2
import numpy as np
import streamlit as st
import websockets

current_file_path = os.path.abspath(__file__)
root_dir = os.path.dirname(current_file_path)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from utils.overlay_renderer import OverlayRenderer


def get_root_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


@st.cache_data
def load_exercise_config() -> tuple[list[str], dict[str, str]]:
    search_paths = [
        os.path.join(get_root_dir(), "exercise_kpt_config_coco.json"),
        os.path.join(get_root_dir(), "configs", "exercise_kpt_config_coco.json"),
        os.path.join(get_root_dir(), "data", "exercise_kpt_config_coco.json"),
    ]

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

            packet_count = 0
            while st.session_state.get("is_running", False):
                try:
                    raw_res = await asyncio.wait_for(ws.recv(), timeout=0.005)
                    data = json.loads(raw_res)
                    pkt_type = data.get("type")
                    packet_count += 1

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

                except asyncio.TimeoutError:
                    pass

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


def main():
    st.set_page_config(page_title="AI Rehabilitation Motion Studio", layout="wide")
    st.title("AI Rehabilitation Motion System")
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
            if st.button("세션 시작 (Start Session)", type="primary"):
                st.session_state.is_running = True
                st.rerun()
        else:
            if st.button("세션 중지 (Stop Session)"):
                st.session_state.is_running = False
                st.rerun()

        if st.session_state.last_calib_thresholds:
            st.markdown("---")
            st.caption("최근 산출된 맞춤 ROM 임계값")
            st.json(st.session_state.last_calib_thresholds)

    _, col_center, _ = st.columns([1, 6, 1])
    with col_center:
        alert_box = st.empty()
        video_box = st.empty()
        metrics_box = st.empty()
        fps_box = st.empty()

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


if __name__ == "__main__":
    main()