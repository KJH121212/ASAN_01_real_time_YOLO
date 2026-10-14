# ==============================================================================
# [Module Information]
# File: network/socket_server.py
# Description: WebSocket streaming server supporting:
#              Mode 1 (VISUAL): Realtime webcam with full image/skeleton stream
#              Mode 2 (LIGHTWEIGHT): Realtime webcam with pure metrics (Unity target)
#              Mode 3 (OFFLINE_VIDEO): High-throughput GPU batch processing for video files
# ==============================================================================

import os
os.environ["MMCV_WITH_OPS"] = "0"

import asyncio
import base64
from datetime import datetime
import json
from pathlib import Path
import sys
import threading
import time
import traceback
import cv2
import numpy as np
import torch
import websockets

try:
    from torch.serialization import add_safe_globals
    add_safe_globals([np.core.multiarray._reconstruct, np.ndarray])
except ImportError:
    pass

_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

try:
    import torch.serialization
    torch.serialization.load = _patched_torch_load
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import pycocotools
sys.modules['xtcocotools'] = pycocotools

current_file_path = os.path.abspath(__file__)
network_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(network_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.data_manager import DataManager, load_exercise_config
from core.motion_engine import MotionEngine
from core.skeleton_engine import SkeletonEngine
from core.session_controller import SessionController
from utils.filters import RealtimeEMAFilter
from utils.normalization import PoseNormalizer


class ThreadedWebcam:
    """실시간 웹캠 최신 프레임 캡처 스레드 (모드 1, 모드 2 공용)"""
    def __init__(self, cam_idx=0):
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


class ExerciseSocketServer:
    def __init__(self, host="127.0.0.1", port=8080):
        self.host = host
        self.port = port

        print("[DEBUG][INIT] Instantiating SkeletonEngine...")
        load_start = time.perf_counter()
        self.skeleton_engine = SkeletonEngine(yolo_interval=5)
        print(f"[DEBUG][INIT] SkeletonEngine loaded in {(time.perf_counter() - load_start) * 1000.0:.2f} ms")

        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)
        self.camera = None
        self.controller = None
        self.filter_engine = None
        self.last_frame_b64 = None
        self.output_mode = "VISUAL"

    def start_camera(self, cam_idx=0):
        if self.camera:
            self.camera.stop()
        self.camera = ThreadedWebcam(cam_idx).start()

    def stop_camera(self):
        if self.camera:
            self.camera.stop()
            self.camera = None

    def init_live_session(self, data: dict):
        """실시간 웹캠 세션 공통 초기화 (모드 1 & 모드 2)"""
        player_id = data.get("player_id", "patient_1")
        patient_name = data.get("patient_name", "Unknown")
        exercise_name = data.get("exercise_name", "biceps_curl")
        mode = data.get("mode", "CALIBRATION")
        target_reps = int(data.get("target_reps", 10))
        cam_idx = int(data.get("camera_index", 0))
        self.output_mode = data.get("output_mode", "VISUAL").upper()

        print(f"[DEBUG][LIVE_INIT] Mode: {mode} | Output: {self.output_mode} | Exercise: {exercise_name}")
        self.start_camera(cam_idx)

        data_manager = DataManager(player_id=player_id, patient_name=patient_name)
        exercise_config = load_exercise_config(exercise_name)

        custom_thresholds = None
        if mode in ["MAIN", "TEST"]:
            custom_thresholds = data_manager.get_custom_threshold(exercise_name)

        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        effective_reps = 3 if mode == "CALIBRATION" else target_reps

        motion_engine = MotionEngine(
            config=exercise_config,
            custom_thresholds=custom_thresholds,
            mode=mode,
            target_reps=effective_reps
        )
        setattr(motion_engine, "exercise_name", exercise_name)

        self.controller = SessionController(
            data_manager=data_manager,
            motion_engine=motion_engine
        )
        self.filter_engine = RealtimeEMAFilter(max_jump=0.15, alpha=0.6)
        self._reset_raw_buffers()

    def _reset_raw_buffers(self):
        if self.controller and self.controller.data_manager:
            dm = self.controller.data_manager
            dm.buf_timestamps = []
            dm.buf_values = []
            dm.buf_raw_keypoints = []

    def _append_to_buffers(self, left_data: dict, right_data: dict, raw_keypoints: list, elapsed_sec: float = None):
        if not raw_keypoints or len(raw_keypoints) < 17:
            return

        raw_kpt_matrix = [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in raw_keypoints]
        if elapsed_sec is None:
            elapsed_sec = round(time.time() - self.controller.timer_start, 3)

        dm = self.controller.data_manager
        dm.buf_timestamps.append(elapsed_sec)
        dm.buf_values.append([left_data.get("val") or 0.0, right_data.get("val") or 0.0])
        dm.buf_raw_keypoints.append(raw_kpt_matrix)

    def _save_session_files(self, prefix: str = "SESSION") -> dict:
        dm = self.controller.data_manager
        if not dm or not getattr(dm, "buf_timestamps", []):
            return None

        session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        exercise_name = getattr(self.controller.motion_engine, "exercise_name", "unknown")

        saved_reps = 0
        if prefix == "SESSION":
            left_reps = self.controller.motion_engine.fsm_left.completed_reps_history
            right_reps = self.controller.motion_engine.fsm_right.completed_reps_history
            all_rep_rows = [{**r, "session_id": f"SESSION_{session_id}", "timestamp": timestamp_str} for r in left_reps + right_reps]
            dm.save_rep_details_csv(all_rep_rows)
            saved_reps = len(all_rep_rows)

        dm.save_raw_trajectory_npz(
            session_id=session_id,
            exercise_name=exercise_name,
            timestamps=dm.buf_timestamps,
            values=dm.buf_values,
            raw_keypoints=dm.buf_raw_keypoints,
            prefix=prefix
        )

        return {
            "session_id": f"{prefix}_{session_id}",
            "prefix": prefix,
            "saved_frames": len(dm.buf_timestamps),
            "saved_reps_count": saved_reps
        }

    # ==========================================================================
    # [모드 3] 비디오 분석 방식 (OFFLINE_VIDEO): 배치 GPU 가속 검증
    # ==========================================================================
    async def process_offline_video(self, websocket, data: dict):
        video_path = data.get("video_path") or data.get("input_source")
        player_id = data.get("player_id", "patient_1")
        patient_name = data.get("patient_name", "Unknown")
        exercise_name = data.get("exercise_name", "biceps_curl")
        mode = data.get("mode", "CALIBRATION")
        target_reps = int(data.get("target_reps", 10))
        batch_size = int(data.get("batch_size", 16))

        print(f"\n[MODE 3: VIDEO_BATCH] Processing '{video_path}' (Batch: {batch_size}, Mode: {mode})")

        await websocket.send(json.dumps({
            "type": "VIDEO_STATUS",
            "status": "PROCESSING",
            "message": "비디오를 고속 배치 분석 중입니다. 잠시만 기다려주세요..."
        }))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            await websocket.send(json.dumps({"type": "ERROR", "message": f"비디오 파일을 열 수 없습니다: {video_path}"}))
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_duration = 1.0 / fps

        data_manager = DataManager(player_id=player_id, patient_name=patient_name)
        exercise_config = load_exercise_config(exercise_name)
        custom_thresholds = data_manager.get_custom_threshold(exercise_name) if mode in ["MAIN", "TEST"] else None
        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        effective_reps = 3 if mode == "CALIBRATION" else target_reps
        motion_engine = MotionEngine(
            config=exercise_config,
            custom_thresholds=custom_thresholds,
            mode=mode,
            target_reps=effective_reps
        )
        setattr(motion_engine, "exercise_name", exercise_name)

        self.controller = SessionController(data_manager=data_manager, motion_engine=motion_engine)
        self._reset_raw_buffers()

        # Step 1: 전체 프레임 GPU 고속 배치 키포인트 추출
        all_extracted_keypoints = []
        batch_frames = []
        t_start = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if batch_frames:
                    kpts_batch = await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)
                    all_extracted_keypoints.extend(kpts_batch)
                break

            batch_frames.append(frame)
            if len(batch_frames) >= batch_size:
                kpts_batch = await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)
                all_extracted_keypoints.extend(kpts_batch)
                batch_frames.clear()

        cap.release()
        print(f"[MODE 3: VIDEO_BATCH] Extracted {len(all_extracted_keypoints)} frames in {time.time() - t_start:.2f}s")

        # Step 2: 프레임 누락 없는 순차 FSM 판정 (양측 3회 완수 검사)
        self.controller.timer_start = 0.0

        for frame_idx, raw_kpts in enumerate(all_extracted_keypoints):
            virtual_time = frame_idx * frame_duration
            is_occluded = self.controller.check_occlusion(raw_kpts) if raw_kpts else True

            if mode == "CALIBRATION":
                cal_res = self.controller.process_calibration_frame(raw_kpts, is_occluded, now=virtual_time)
                if cal_res["calib_step"] in ["COLLECTING", "COOLDOWN"] and not is_occluded and raw_kpts:
                    vl = self.controller.calib_vals_left[-1] if self.controller.calib_vals_left else 0.0
                    vr = self.controller.calib_vals_right[-1] if self.controller.calib_vals_right else 0.0
                    self._append_to_buffers({"val": vl}, {"val": vr}, raw_kpts, elapsed_sec=virtual_time)

                if cal_res["calib_step"] == "FINISHED":
                    print(f"[MODE 3: VIDEO_BATCH] 양측 캘리브레이션 3회 완수 완료 (Frame: {frame_idx}/{total_frames})")
                    break

            elif mode in ["MAIN", "TEST"]:
                left_data, right_data, is_finished = self.controller.process_main_frame(raw_kpts, is_occluded, now=virtual_time)
                if not is_occluded and raw_kpts:
                    self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=virtual_time)

                if is_finished:
                    print(f"[MODE 3: VIDEO_BATCH] Target reps 완수 완료 (Frame: {frame_idx}/{total_frames})")
                    break

        # Step 3: 파일 저장 및 완료 결과 일괄 전송
        prefix = "CALIB" if mode == "CALIBRATION" else "SESSION"
        summary_meta = self._save_session_files(prefix=prefix)

        payload = {
            "type": "VIDEO_ANALYSIS_COMPLETED",
            "mode": mode,
            "video_path": video_path,
            "total_frames_analyzed": len(all_extracted_keypoints),
            "summary": summary_meta
        }
        if mode == "CALIBRATION":
            payload["thresholds"] = self.controller.compute_and_save_thresholds(exercise_name)

        await websocket.send(json.dumps(payload))
        print(f"[MODE 3: VIDEO_BATCH] Completed successfully.")
        self.controller = None

    # ==========================================================================
    # 메인 웹소켓 루프 (모드 1 & 모드 2 실시간 분기 처리)
    # ==========================================================================
    async def handle_client(self, websocket):
        client_address = websocket.remote_address
        print(f"[DEBUG][NETWORK] Client connected: {client_address}")

        prev_time = time.time()
        frame_counter = 0

        try:
            while True:
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    packet = json.loads(raw_msg)
                    pkt_type = packet.get("type")

                    if pkt_type == "CMD_SET_SESSION":
                        video_path = packet.get("video_path") or packet.get("input_source")
                        if video_path and os.path.isfile(str(video_path)):
                            await self.process_offline_video(websocket, packet)
                            continue
                        else:
                            self.init_live_session(packet)

                    elif pkt_type == "CMD_STOP_CALIBRATION":
                        if self.controller and self.controller.mode == "CALIBRATION":
                            self.controller.calib_step = "FINISHED"

                except asyncio.TimeoutError:
                    pass

                if self.controller and self.camera:
                    ret, frame = self.camera.read()

                    if ret and frame is not None:
                        curr_time = time.time()
                        time_delta = curr_time - prev_time
                        fps = round(1.0 / time_delta, 1) if time_delta > 0 else 30.0
                        prev_time = curr_time
                        frame_counter += 1

                        if self.output_mode == "VISUAL" and frame_counter % 2 == 0:
                            def encode_frame(img):
                                small = cv2.resize(img, (400, 300))
                                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])
                                return base64.b64encode(buf).decode("utf-8")
                            self.last_frame_b64 = await asyncio.to_thread(encode_frame, frame)

                        raw_kpts = await asyncio.to_thread(self.skeleton_engine.extract_keypoints, frame)
                        filtered_kpts = self.filter_engine.update(raw_kpts)
                        yolo_detected = filtered_kpts is not None and len(filtered_kpts) > 0
                        is_occluded = self.controller.check_occlusion(filtered_kpts) if yolo_detected else True

                        current_mode = self.controller.motion_engine.mode
                        left_data = self.controller.last_left_data
                        right_data = self.controller.last_right_data
                        calib_reps_left = 0
                        calib_reps_right = 0
                        calib_rep_count = 0

                        # --- CALIBRATION 진행 (양측 모두 3회) ---
                        if current_mode == "CALIBRATION":
                            cal_res = self.controller.process_calibration_frame(filtered_kpts, is_occluded)
                            calib_rep_count = cal_res["calib_rep_count"]
                            calib_reps_left = cal_res["calib_reps_left"]
                            calib_reps_right = cal_res["calib_reps_right"]

                            if cal_res["calib_step"] in ["COLLECTING", "COOLDOWN"] and not is_occluded and raw_kpts:
                                vl = self.controller.calib_vals_left[-1] if self.controller.calib_vals_left else 0.0
                                vr = self.controller.calib_vals_right[-1] if self.controller.calib_vals_right else 0.0
                                self._append_to_buffers({"val": vl}, {"val": vr}, raw_kpts)

                            if cal_res["calib_step"] == "FINISHED":
                                ex_name = getattr(self.controller.motion_engine, "exercise_name", "biceps_curl")
                                calib_summary = self._save_session_files(prefix="CALIB")
                                custom_th = self.controller.compute_and_save_thresholds(ex_name)

                                await websocket.send(json.dumps({
                                    "type": "CALIBRATION_FINISHED",
                                    "player_id": self.controller.data_manager.player_id,
                                    "thresholds": custom_th,
                                    "summary": calib_summary
                                }))
                                self.controller = None
                                continue

                        # --- TEST / MAIN 진행 ---
                        elif current_mode in ["MAIN", "TEST"]:
                            left_data, right_data, is_finished = self.controller.process_main_frame(filtered_kpts, is_occluded)
                            if not is_occluded and raw_kpts:
                                self._append_to_buffers(left_data, right_data, raw_kpts)

                            if is_finished:
                                test_summary = self._save_session_files(prefix="SESSION")
                                await websocket.send(json.dumps({
                                    "type": "SESSION_FINISHED",
                                    "summary": test_summary
                                }))
                                self.controller = None
                                continue

                        # --- 패킷 송출 (모드 1 vs 모드 2) ---
                        if self.output_mode == "VISUAL":
                            pose_payload = {
                                "type": "POSE_UPDATE",
                                "mode": current_mode,
                                "left": left_data,
                                "right": right_data,
                                "keypoints": filtered_kpts if yolo_detected else [],
                                "frame_b64": self.last_frame_b64,
                                "fps": fps,
                                "yolo_detected": yolo_detected,
                                "is_occluded": is_occluded,
                                "calib_step": self.controller.calib_step if self.controller else "FINISHED",
                                "calib_rep_count": calib_rep_count,
                                "calib_reps_left": calib_reps_left,
                                "calib_reps_right": calib_reps_right,
                                "target_calib_reps": 3,
                            }
                        else:
                            pose_payload = {
                                "type": "POSE_UPDATE",
                                "mode": current_mode,
                                "left": left_data,
                                "right": right_data,
                                "fps": fps,
                                "is_occluded": is_occluded,
                                "calib_step": self.controller.calib_step if self.controller else "FINISHED",
                                "calib_rep_count": calib_rep_count,
                                "calib_reps_left": calib_reps_left,
                                "calib_reps_right": calib_reps_right,
                                "target_calib_reps": 3,
                            }

                        await websocket.send(json.dumps(pose_payload))

                await asyncio.sleep(0.001)

        except websockets.exceptions.ConnectionClosed:
            print(f"[DEBUG][NETWORK] Client disconnected: {client_address}")
            if self.controller:
                prefix = "CALIB" if self.controller.mode == "CALIBRATION" else "SESSION"
                self._save_session_files(prefix=prefix)
        except Exception as e:
            print(f"[ERROR][SERVER] Exception: {e}")
            traceback.print_exc()
        finally:
            self.stop_camera()

    async def run(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"[DEBUG][SERVER] ExerciseSocketServer listening on ws://{self.host}:{self.port}")
            await asyncio.Future()


if __name__ == "__main__":
    server = ExerciseSocketServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[DEBUG][SERVER] Server stopped by user.")