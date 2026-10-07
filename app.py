# ==============================================================================
# [Module Information]
# File: app.py
# Description: Pure Streamlit client frontend for AI rehabilitation motion tracking.
#              Displays clean skeleton video on viewport and delegates all HUD
#              and dynamic ROM gauges to the right side panel.
# ==============================================================================

import asyncio
import base64
import json
import os
from pathlib import Path
import sys
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


async def run_client_session(uri: str, config: dict, placeholders: dict, renderer: OverlayRenderer):
    """WebSocket 실시간 스트리밍 및 CALIBRATION / TEST 통제 루프"""
    try:
        async with websockets.connect(uri) as ws:
            cmd_packet = {
                "type": "CMD_SET_SESSION",
                "player_id": config["player_id"],
                "patient_name": config["patient_name"],
                "exercise_name": config["exercise_name"],
                "mode": config["mode"],
                "target_reps": config["target_reps"],
                "camera_index": config["camera_index"],
                "input_source": config.get("input_source")
            }
            await ws.send(json.dumps(cmd_packet))

            packet_count = 0
            while st.session_state.get("is_running", False):
                try:
                    raw_res = await asyncio.wait_for(ws.recv(), timeout=0.005)
                    data = json.loads(raw_res)
                    pkt_type = data.get("type")
                    packet_count += 1

                    # 1. 세션 완료 이벤트 처리
                    if pkt_type in ["SESSION_FINISHED", "CALIBRATION_FINISHED"]:
                        summary = data.get("summary", {})
                        if pkt_type == "CALIBRATION_FINISHED":
                            th = data.get("thresholds", {})
                            st.session_state.last_calib_thresholds = th
                            placeholders["alert"].success(
                                "캘리브레이션 3회 완수! 맞춤 임계값이 성공적으로 저장되었습니다."
                            )
                        else:
                            placeholders["alert"].success(
                                f"본 운동 세션 완료! 총 {summary.get('saved_reps_count', config['target_reps'])}회 달성 완료"
                            )
                        st.session_state.is_running = False
                        break

                    # 2. 실시간 포즈 업데이트 처리
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
                        left_info = data.get("left", {})
                        right_info = data.get("right", {})

                        # 영상에는 순수하게 스켈레톤 선과 관절 포인트만 렌더링 (HUD/Bar 오버레이 완전 제거)
                        canvas = renderer.draw_skeleton(
                            canvas=canvas,
                            keypoints=keypoints,
                            is_occluded=is_occluded,
                            is_cartesian_space=False
                        )

                        val_l = left_info.get("val")
                        val_r = right_info.get("val")

                        # ------------------------------------------------------
                        # [오른쪽 패널 HUD 및 상단 알림창 업데이트]
                        # ------------------------------------------------------
                        if current_mode == "CALIBRATION":
                            calib_step = data.get("calib_step", "")
                            calib_reps = data.get("calib_rep_count", 0)
                            target_reps = data.get("target_calib_reps", 3)

                            placeholders["left_metric"].metric(
                                label="캘리브레이션 횟수",
                                value=f"{calib_reps} / {target_reps}",
                                delta=f"상태: {calib_step}"
                            )
                            placeholders["right_metric"].metric(
                                label="실시간 수치 (L / R)",
                                value=f"{val_l or 0.0:.2f} / {val_r or 0.0:.2f}",
                                delta=f"FSM: L[{left_info.get('state', '-')}] R[{right_info.get('state', '-')}]"
                            )

                            if is_occluded:
                                placeholders["alert"].error("가림 감지: 신체가 화면 중앙에 온전히 나오도록 서주세요.")
                            elif calib_step == "FULL_BODY_CHECK":
                                placeholders["alert"].info("정면을 응시하고 준비 자세를 유지해주세요.")
                            elif calib_step == "COUNTDOWN":
                                placeholders["alert"].warning("카운트다운: 잠시 후 3회 동작 수집을 시작합니다.")
                            elif calib_step == "COLLECTING":
                                placeholders["alert"].success(f"측정 진행 중: 전체 가동 범위로 3회 반복하세요! ({calib_reps}/{target_reps})")
                            elif calib_step == "COOLDOWN":
                                placeholders["alert"].success("측정 완료! 3회 달성 성공 (잠시 후 결과가 저장됩니다)")

                        else:  # TEST 모드
                            l_cnt = left_info.get("rep_count", 0)
                            r_cnt = right_info.get("rep_count", 0)
                            rep_val = max(l_cnt, r_cnt)
                            target_reps = config["target_reps"]

                            placeholders["left_metric"].metric(
                                label="Left Reps",
                                value=f"{l_cnt} / {target_reps}",
                                delta=f"등급: {left_info.get('quality', '-')}"
                            )
                            placeholders["right_metric"].metric(
                                label="Right Reps",
                                value=f"{r_cnt} / {target_reps}",
                                delta=f"등급: {right_info.get('quality', '-')}"
                            )

                            if is_occluded:
                                placeholders["alert"].error("가림 감지: 신체 주요 관절이 가려져 FSM이 일시 중지되었습니다.")
                            elif rep_val >= target_reps:
                                placeholders["alert"].success(f"🎉 목표 완수! 총 {target_reps}회를 모두 달성했습니다. (데이터 정리 중...)")
                            else:
                                placeholders["alert"].empty()

                        # 화면 스트림 렌더링
                        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                        placeholders["video"].image(rgb_canvas, channels="RGB")

                        # 오른쪽 사이드 패널의 Real-Time ROM 진행 바 갱신
                        l_ratio = min(max(float(left_info.get("progress_ratio", 0.0)), 0.0), 1.0)
                        r_ratio = min(max(float(right_info.get("progress_ratio", 0.0)), 0.0), 1.0)
                        placeholders["left_progress"].progress(l_ratio, text=f"Left ROM: {int(l_ratio * 100)}% (수치: {val_l or 0.0:.2f})")
                        placeholders["right_progress"].progress(r_ratio, text=f"Right ROM: {int(r_ratio * 100)}% (수치: {val_r or 0.0:.2f})")

                        fps_val = data.get("fps", 0.0)
                        placeholders["fps"].caption(f"FPS: {fps_val:.1f} | Mode: {current_mode} | Packets: {packet_count}")

                except asyncio.TimeoutError:
                    pass

    except ConnectionRefusedError:
        placeholders["alert"].error("소켓 서버(127.0.0.1:8080)에 연결할 수 없습니다. 먼저 별도 터미널에서 python network/socket_server.py 를 실행하세요.")
        st.session_state.is_running = False
    except Exception as e:
        traceback.print_exc()
        placeholders["alert"].error(f"서버 통신 오류: {e}")
        st.session_state.is_running = False


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

    with st.sidebar:
        st.header("Session Settings")

        player_id = st.text_input("Patient ID", value="patient_1")
        patient_name = st.text_input("Patient Name", value="김지후")

        exercise_list = ["biceps_curl", "shoulder_press", "clamshell", "slr", "knee_extension"]
        exercise_name = st.selectbox("Exercise Name", exercise_list, index=0)

        mode = st.radio("Execution Mode", ["CALIBRATION", "TEST"], index=0)

        if mode == "CALIBRATION":
            st.info("고정 3회 동작을 통해 개인 맞춤형 ROM 임계값을 산출합니다.")
            target_reps = 3
        else:
            target_reps = st.number_input("Target Repetitions (TEST)", min_value=1, max_value=50, value=5, step=1)

        source_type = st.radio("Source Type", ["Video File Path", "Live Webcam"], index=0)

        input_source = None
        camera_index = 0
        if source_type == "Live Webcam":
            camera_index = st.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
        else:
            input_source = st.text_input("Video File Path", value=default_video_path)

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

    col_view, col_stats = st.columns([3, 2])

    with col_view:
        st.subheader("실시간 모션 뷰포트 (Skeleton Only)")
        alert_box = st.empty()
        video_box = st.empty()
        fps_box = st.empty()

    with col_stats:
        st.subheader("운동 성과 및 생체 역학 HUD")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            left_metric_box = st.empty()
        with col_m2:
            right_metric_box = st.empty()

        st.markdown("##### Real-Time ROM Gauge")
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
        chosen_source = None
        if source_type == "Video File Path" and input_source:
            chosen_source = str(Path(input_source).resolve())

        cfg = {
            "player_id": player_id,
            "patient_name": patient_name,
            "exercise_name": exercise_name,
            "mode": mode,
            "target_reps": int(target_reps),
            "camera_index": int(camera_index),
            "input_source": chosen_source,
        }

        asyncio.run(run_client_session("ws://127.0.0.1:8080", cfg, placeholders, renderer))


if __name__ == "__main__":
    main()