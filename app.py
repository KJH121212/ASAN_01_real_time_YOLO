# ==============================================================================
# [Module Information]
# File: app.py
# Description: Streamlit frontend controller supporting 3-repetition calibration,
#              real-time pose overlay, and HUD synchronizations.
# ==============================================================================

import asyncio
import base64
import json
import os
import socket
import subprocess
import sys
import time
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


def is_port_in_use(port: int = 8080, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def ensure_socket_server(base_dir: str):
    """소켓 서버 포트 점유 여부를 검사하고 미실행 시 백그라운드로 기동합니다."""
    if is_port_in_use(8080):
        print("[DEBUG][SERVER] Port 8080 is active. Skipping process spawn.")
        return

    if "server_process" not in st.session_state or st.session_state.server_process is None:
        server_script = os.path.join(base_dir, "network", "socket_server.py")
        print(f"[DEBUG][SERVER] Launching backend server script: {server_script}")
        try:
            process = subprocess.Popen([sys.executable, server_script])
            st.session_state.server_process = process
            print(f"[DEBUG][SERVER] Subprocess PID: {process.pid}. Waiting 3.0s for weight loading...")
            time.sleep(3.0)
        except Exception as e:
            print(f"[ERROR][SERVER] Failed to execute socket server: {e}")
            traceback.print_exc()


def toggle_session():
    st.session_state.is_running = not st.session_state.is_running
    print(f"[DEBUG][UI] Session toggle button pressed. Running state: {st.session_state.is_running}")


async def run_unity_client_session(uri: str, config: dict, placeholders: dict, renderer: OverlayRenderer):
    """WebSocket 스트리밍 연결 및 3회 캘리브레이션 / 본 운동 HUD 동기화 루프"""
    print(f"[DEBUG][WS] Connecting to socket server: {uri}")

    try:
        async with websockets.connect(uri) as ws:
            print("[DEBUG][WS] Connection established successfully.")

            # 시간 파라미터 제외, 3회 고정 캘리브레이션 규격 패킷 발송
            cmd_packet = {
                "type": "CMD_SET_SESSION",
                "player_id": config["player_id"],
                "patient_name": config["patient_name"],
                "exercise_name": config["exercise_name"],
                "mode": config["mode"],
                "target_reps": config["target_reps"],
                "camera_index": config["camera_index"],
            }
            await ws.send(json.dumps(cmd_packet))
            print(f"[DEBUG][WS] Initialized session with packet: {cmd_packet}")

            packet_count = 0
            while st.session_state.get("is_running", False):
                try:
                    raw_res = await asyncio.wait_for(ws.recv(), timeout=0.03)
                    data = json.loads(raw_res)
                    pkt_type = data.get("type")
                    packet_count += 1

                    # 1. 완료 이벤트 수신 처리
                    if pkt_type in ["SESSION_FINISHED", "CALIBRATION_FINISHED"]:
                        print(f"[DEBUG][WS] Received session termination signal: {pkt_type}")
                        summary = data.get("summary", {})
                        placeholders["alert"].success(
                            f"Session Completed: {pkt_type} | Saved Records: {summary.get('saved_reps_count', 0)}"
                        )
                        st.session_state.is_running = False
                        break

                    # 2. 실시간 프레임 패킷 수신 처리
                    if pkt_type == "POSE_UPDATE":
                        img_b64 = data.get("frame_b64")
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

                        if img_b64:
                            img_bytes = base64.b64decode(img_b64)
                            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                canvas = cv2.resize(frame, (640, 480))

                        keypoints = data.get("keypoints", [])
                        is_occluded = data.get("is_occluded", False)
                        current_mode = data.get("mode", config["mode"])

                        # 뼈대 오버레이 시각화
                        canvas = renderer.draw_skeleton(
                            canvas=canvas,
                            keypoints=keypoints,
                            is_occluded=is_occluded,
                            is_cartesian_space=False
                        )

                        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                        placeholders["video"].image(rgb_canvas, channels="RGB")

                        # 캘리브레이션 3회 수행 UI 분기
                        if current_mode == "CALIBRATION":
                            calib_step = data.get("calib_step", "")
                            calib_rep_count = data.get("calib_rep_count", 0)
                            target_calib_reps = data.get("target_calib_reps", 3)

                            if is_occluded:
                                placeholders["alert"].error("Occlusion Detected: Stand in camera center to reset.")
                            elif calib_step == "FULL_BODY_CHECK":
                                placeholders["alert"].info("Standby: Please stand still facing the camera.")
                            elif calib_step == "COUNTDOWN":
                                placeholders["alert"].warning("Preparation: Get ready to perform 3 full repetitions.")
                            elif calib_step == "COLLECTING":
                                placeholders["alert"].success(
                                    f"Calibrating ROM: Perform 3 full reps! (Progress: {calib_rep_count} / {target_calib_reps})"
                                )

                        # 본 운동(TEST) 세션 UI 분기
                        else:
                            if is_occluded:
                                placeholders["alert"].error("Occlusion Alert: Keep entire body in frame. (FSM frozen)")
                            else:
                                placeholders["alert"].empty()

                            left_info = data.get("left", {})
                            right_info = data.get("right", {})

                            placeholders["left_metric"].metric(
                                label="Left Reps",
                                value=f"{left_info.get('rep_count', 0)} / {config['target_reps']}",
                                delta=f"Quality: {left_info.get('quality', '-')}"
                            )

                            placeholders["right_metric"].metric(
                                label="Right Reps",
                                value=f"{right_info.get('rep_count', 0)} / {config['target_reps']}",
                                delta=f"Quality: {right_info.get('quality', '-')}"
                            )

                            l_ratio = min(max(float(left_info.get("progress_ratio", 0.0)), 0.0), 1.0)
                            r_ratio = min(max(float(right_info.get("progress_ratio", 0.0)), 0.0), 1.0)

                            placeholders["left_progress"].progress(l_ratio, text=f"Left ROM: {int(l_ratio * 100)}%")
                            placeholders["right_progress"].progress(r_ratio, text=f"Right ROM: {int(r_ratio * 100)}%")

                        fps_val = data.get("fps", 0.0)
                        placeholders["fps"].caption(
                            f"Status: Connected | FPS: {fps_val:.1f} | Mode: {current_mode} | Packets: {packet_count}"
                        )

                except asyncio.TimeoutError:
                    pass

    except websockets.exceptions.ConnectionClosed:
        print("[WARN][WS] Remote server closed connection.")
        placeholders["alert"].warning("WebSocket server closed connection.")
        st.session_state.is_running = False
    except Exception as e:
        print(f"[ERROR][WS] Pipeline error: {e}")
        traceback.print_exc()
        placeholders["alert"].error(f"Socket connection failure: {e}")
        st.session_state.is_running = False


def main():
    st.set_page_config(page_title="AI Rehab Unity Simulator", layout="wide")
    st.title("AI Rehabilitation Motion Frontend Simulator")
    st.markdown("---")

    base_root = get_root_dir()
    renderer = OverlayRenderer(conf_threshold=0.35)

    if "is_running" not in st.session_state:
        st.session_state.is_running = False

    with st.sidebar:
        st.header("Session Control Panel")

        player_id = st.text_input("Patient ID", value="patient_1")
        patient_name = st.text_input("Patient Name", value="Kim Jihoo")

        exercise_list = [
            "biceps_curl",
            "shoulder_press",
            "clamshell",
            "slr",
            "hip_knee_flexion",
            "knee_extension"
        ]
        exercise_name = st.selectbox("Exercise Name", exercise_list, index=0)

        mode = st.radio("Session Mode", ["CALIBRATION", "TEST"], index=0)

        # 캘리브레이션은 3회 고정이므로 슬라이더 대신 고정 정보 표시
        if mode == "CALIBRATION":
            st.info("Calibration Mode: Perform exactly 3 repetitions.")
            target_reps = 3
        else:
            target_reps = st.number_input("Target Repetitions (TEST only)", min_value=1, max_value=50, value=5, step=1)

        camera_index = st.number_input("Camera Device Index", min_value=0, max_value=5, value=0, step=1)
        st.markdown("---")

        if not st.session_state.is_running:
            st.button("Start Session", type="primary", on_click=toggle_session)
        else:
            st.button("Stop Session", on_click=toggle_session)

    col_view, col_stats = st.columns([3, 2])

    with col_view:
        st.subheader("Visual Skeleton Viewport")
        alert_box = st.empty()
        video_box = st.empty()
        fps_box = st.empty()

    with col_stats:
        st.subheader("Kinematic HUD")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            left_metric_box = st.empty()
        with col_m2:
            right_metric_box = st.empty()

        st.markdown("##### Real-Time Range of Motion (ROM)")
        left_prog_box = st.empty()
        right_prog_box = st.empty()

    placeholders = {
        "video": video_box,
        "alert": alert_box,
        "fps": fps_box,
        "left_metric": left_metric_box,
        "right_metric": right_metric_box,
        "left_progress": left_prog_box,
        "right_progress": right_prog_box,
    }

    if st.session_state.is_running:
        ensure_socket_server(base_root)

        cfg = {
            "player_id": player_id,
            "patient_name": patient_name,
            "exercise_name": exercise_name,
            "mode": mode,
            "target_reps": int(target_reps),
            "camera_index": int(camera_index),
        }

        asyncio.run(run_unity_client_session("ws://127.0.0.1:8080", cfg, placeholders, renderer))


if __name__ == "__main__":
    main()