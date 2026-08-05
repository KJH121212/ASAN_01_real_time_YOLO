import os
import sys
from unittest.mock import MagicMock

mock_ext = MagicMock()
mock_ext.__spec__ = MagicMock()
sys.modules['mmcv._ext'] = mock_ext

import pycocotools
sys.modules['xtcocotools'] = pycocotools

current_file_path = os.path.abspath(__file__)
prototype_dir = os.path.dirname(os.path.dirname(current_file_path))
root_dir = os.path.dirname(prototype_dir)

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if prototype_dir not in sys.path:
    sys.path.insert(1, prototype_dir)

import torch
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import asyncio
import websockets
import json
import cv2
import time
import base64
from datetime import datetime

from core.skeleton_engine import SkeletonEngine
from core.motion_engine import MotionEngine
from core.data_manager import DataManager, load_exercise_config


class ExerciseSocketServer:
    def __init__(self, host="127.0.0.1", port=8080):
        self.host = host
        self.port = port
        
        print("[SocketServer] SkeletonEngine AI 모델 로드 중...")
        self.skeleton_engine = SkeletonEngine(yolo_interval=3)
        
        self.cap = None
        self.camera_index = 0
        self.active_session = False
        
        self.data_manager = None
        self.motion_engine = None
        self.current_exercise = "shoulder_press"
        self.player_id = "patient_1"
        self.patient_name = "미지정"
        self.target_reps = 10
        self.session_start_time = None

        self.buf_timestamps = []
        self.buf_values = []
        self.buf_keypoints = []

        self.target_fps = 30
        self.frame_delay = 1.0 / self.target_fps

    def start_camera(self, camera_index: int = 0):
        if self.cap is not None and self.camera_index != camera_index:
            self.stop_camera()

        if self.cap is None or not self.cap.isOpened():
            self.camera_index = camera_index
            self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not self.cap.isOpened():
                self.camera_index = 0
                self.cap = cv2.VideoCapture(0)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None

    def init_session(self, player_id: str, patient_name: str, exercise_name: str, mode: str = "CALIBRATION", target_reps: int = 10, camera_index: int = 0):
        self.player_id = player_id
        self.patient_name = patient_name
        self.current_exercise = exercise_name
        self.target_reps = target_reps
        self.session_start_time = time.time()
        
        self.buf_timestamps = []
        self.buf_values = []
        self.buf_keypoints = []

        self.start_camera(camera_index)
        self.data_manager = DataManager(player_id=player_id, patient_name=patient_name)
        
        exercise_config = load_exercise_config(exercise_name)
        custom_thresholds = self.data_manager.get_custom_threshold(exercise_name)
        
        self.motion_engine = MotionEngine(exercise_config, custom_thresholds)
        if mode == "MAIN" and custom_thresholds:
            self.motion_engine.mode = "MAIN"
        else:
            self.motion_engine.mode = "CALIBRATION"
            
        self.active_session = True

    async def handle_client(self, websocket):
        print(f"[SocketServer] 클라이언트 접속: {websocket.remote_address}")
        prev_time = time.time()

        try:
            while True:
                loop_start = time.time()

                # Step 1. 클라이언트 명령 수신
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    data = json.loads(message)
                    cmd_type = data.get("type")
                    
                    if cmd_type == "CMD_SET_SESSION":
                        self.init_session(
                            player_id=data.get("player_id", "patient_1"),
                            patient_name=data.get("patient_name", "미지정"),
                            exercise_name=data.get("exercise_name", "shoulder_press"),
                            mode=data.get("mode", "CALIBRATION"),
                            target_reps=data.get("target_reps", 10),
                            camera_index=data.get("camera_index", 0)
                        )

                    elif cmd_type == "CMD_CONFIRM_THRESHOLD":
                        final_th = data.get("final_thresholds")
                        if self.data_manager and final_th:
                            self.data_manager.save_custom_threshold(self.current_exercise, final_th, is_confirmed=True)
                            exercise_config = load_exercise_config(self.current_exercise)
                            self.motion_engine = MotionEngine(exercise_config, final_th)
                            self.motion_engine.mode = "MAIN"

                except asyncio.TimeoutError:
                    pass

                # Step 2. AI 추론 및 데이터 송신 (YOLO 실패 여부와 무관하게 프레임 전송)
                if self.active_session and self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        curr_time = time.time()
                        fps = round(1.0 / (curr_time - prev_time), 1) if (curr_time - prev_time) > 0 else 30.0
                        prev_time = curr_time

                        keypoints = self.skeleton_engine.extract_keypoints(frame)
                        yolo_detected = keypoints is not None and len(keypoints) > 0

                        max_rep = 0
                        max_progress = 0.0
                        is_updated = False
                        current_mode = self.motion_engine.mode if self.motion_engine else "CALIBRATION"

                        if yolo_detected and self.motion_engine:
                            result = self.motion_engine.process_keypoints(keypoints)
                            
                            if result:
                                current_mode = result["mode"]
                                max_rep = max(result["left"]["rep_count"], result["right"]["rep_count"])
                                max_progress = max(result["left"]["progress_ratio"], result["right"]["progress_ratio"])
                                is_updated = result["left"]["is_updated"] or result["right"]["is_updated"]

                                if result["mode"] == "MAIN":
                                    kpt_matrix = [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in keypoints]
                                    self.buf_timestamps.append(round(time.time() - self.session_start_time, 3))
                                    self.buf_values.append([result["left"]["val"] or 0.0, result["right"]["val"] or 0.0])
                                    self.buf_keypoints.append(kpt_matrix)

                                if result.get("new_custom_thresholds"):
                                    new_th = result["new_custom_thresholds"]
                                    self.data_manager.save_custom_threshold(self.current_exercise, new_th, is_confirmed=False)
                                    await websocket.send(json.dumps({
                                        "type": "CALIBRATION_FINISHED",
                                        "player_id": self.player_id,
                                        "exercise_name": self.current_exercise,
                                        "recommended_thresholds": new_th
                                    }))

                        # 비디오 프레임 인코딩 (디버깅용)
                        _, img_buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                        frame_b64 = base64.b64encode(img_buffer).decode('utf-8')

                        payload = {
                            "type": "POSE_UPDATE",
                            "mode": current_mode,
                            "rep_count": max_rep,
                            "progress_ratio": max_progress,
                            "keypoints": keypoints if yolo_detected else [],
                            "fps": fps,
                            "quality": "DETECTED" if yolo_detected else "SEARCHING",
                            "is_updated": is_updated,
                            "yolo_detected": yolo_detected,
                            "frame_b64": frame_b64
                        }
                        await websocket.send(json.dumps(payload))

                        if yolo_detected and self.motion_engine and current_mode == "MAIN":
                            if result["left"]["rep_count"] >= self.target_reps and result["right"]["rep_count"] >= self.target_reps:
                                duration = round(time.time() - self.session_start_time, 1)
                                session_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                                
                                summary = {
                                    "session_id": session_id,
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "player_id": self.player_id,
                                    "patient_name": self.patient_name,
                                    "exercise_name": self.current_exercise,
                                    "target_reps": self.target_reps,
                                    "left_completed_reps": result["left"]["rep_count"],
                                    "right_completed_reps": result["right"]["rep_count"],
                                    "total_duration_sec": duration
                                }
                                
                                self.data_manager.save_test_session_summary(summary)
                                self.data_manager.save_trajectory_npz(
                                    session_id=session_id,
                                    exercise_name=self.current_exercise,
                                    timestamps=self.buf_timestamps,
                                    values=self.buf_values,
                                    keypoints=self.buf_keypoints
                                )
                                
                                await websocket.send(json.dumps({"type": "SESSION_FINISHED", "summary": summary}))
                                self.active_session = False

                elapsed = time.time() - loop_start
                await asyncio.sleep(max(0.001, self.frame_delay - elapsed))

        except websockets.exceptions.ConnectionClosed:
            print("[SocketServer] 클라이언트 연결 끊김")
        finally:
            self.stop_camera()

    async def run(self):
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"[SocketServer] AI Headless 서버 구동 중 (ws://{self.host}:{self.port})")
            await asyncio.Future()


if __name__ == "__main__":
    server = ExerciseSocketServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[SocketServer] 서버 종료")