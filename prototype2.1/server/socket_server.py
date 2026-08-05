import asyncio
import base64
from datetime import datetime
import json
import os
import sys
import time
import traceback
from unittest.mock import MagicMock
import cv2
import torch
import websockets

mock_ext = MagicMock()
mock_ext.__spec__ = MagicMock()
sys.modules['mmcv._ext'] = mock_ext

import pycocotools

sys.modules['xtcocotools'] = pycocotools

current_file_path = os.path.abspath(__file__)
server_dir = os.path.dirname(current_file_path)
prototype_dir = os.path.dirname(server_dir)
root_dir = os.path.dirname(prototype_dir)

if prototype_dir not in sys.path:
  sys.path.insert(0, prototype_dir)
if root_dir not in sys.path:
  sys.path.insert(1, root_dir)

_original_torch_load = torch.load


def _patched_torch_load(*args, **kwargs):
  kwargs['weights_only'] = False
  return _original_torch_load(*args, **kwargs)


torch.load = _patched_torch_load

from core.data_manager import DataManager, load_exercise_config
from core.motion_engine import MotionEngine
from core.skeleton_engine import SkeletonEngine


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
    self.current_exercise = "biceps_curl"
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

  def init_session(
      self,
      player_id: str,
      patient_name: str,
      exercise_name: str,
      mode: str = "CALIBRATION",
      target_reps: int = 10,
      camera_index: int = 0,
  ):
    self.player_id = player_id
    self.patient_name = patient_name
    self.current_exercise = exercise_name
    self.target_reps = target_reps
    self.session_start_time = time.time()

    self.buf_timestamps = []
    self.buf_values = []
    self.buf_keypoints = []

    self.start_camera(camera_index)
    self.data_manager = DataManager(
        player_id=player_id, patient_name=patient_name
    )

    exercise_config = load_exercise_config(exercise_name)
    custom_thresholds = self.data_manager.get_custom_threshold(exercise_name)

    self.motion_engine = MotionEngine(
        exercise_config,
        custom_thresholds,
        mode=mode,
        target_reps=target_reps,
    )
    self.active_session = True

  def _save_current_session_data(self):
    if not self.data_manager or len(self.buf_timestamps) == 0:
      return None

    session_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    left_reps = self.motion_engine.fsm_left.completed_reps_history
    right_reps = self.motion_engine.fsm_right.completed_reps_history

    all_rep_rows = []
    for r in left_reps + right_reps:
      row_copy = r.copy()
      row_copy["session_id"] = session_id
      row_copy["timestamp"] = timestamp_str
      all_rep_rows.append(row_copy)

    self.data_manager.save_rep_details_csv(all_rep_rows)
    self.data_manager.save_trajectory_npz(
        session_id=session_id,
        exercise_name=self.current_exercise,
        timestamps=self.buf_timestamps,
        values=self.buf_values,
        keypoints=self.buf_keypoints,
    )

    return {
        "session_id": session_id,
        "patient_id": self.player_id,
        "saved_reps_count": len(all_rep_rows),
    }

  async def handle_client(self, websocket):
    print(f"[SocketServer] 클라이언트 접속: {websocket.remote_address}")
    prev_time = time.time()

    try:
      while True:
        loop_start = time.time()

        try:
          message = await asyncio.wait_for(websocket.recv(), timeout=0.001)
          data = json.loads(message)
          if data.get("type") == "CMD_SET_SESSION":
            self.init_session(
                player_id=data.get("player_id", "patient_1"),
                patient_name=data.get("patient_name", "미지정"),
                exercise_name=data.get("exercise_name", "biceps_curl"),
                mode=data.get("mode", "CALIBRATION"),
                target_reps=data.get("target_reps", 10),
                camera_index=data.get("camera_index", 0),
            )
        except asyncio.TimeoutError:
          pass

        if self.active_session and self.cap and self.cap.isOpened():
          ret, frame = self.cap.read()
          frame_b64 = None

          if ret:
            curr_time = time.time()
            fps = (
                round(1.0 / (curr_time - prev_time), 1)
                if (curr_time - prev_time) > 0
                else 30.0
            )
            prev_time = curr_time

            small_frame = cv2.resize(frame, (480, 360))
            _, img_buffer = cv2.imencode(
                ".jpg", small_frame, [cv2.IMWRITE_JPEG_QUALITY, 40]
            )
            frame_b64 = base64.b64encode(img_buffer).decode("utf-8")

            keypoints = await asyncio.to_thread(
                self.skeleton_engine.extract_keypoints, frame
            )
            yolo_detected = keypoints is not None and len(keypoints) > 0

            current_mode = (
                self.motion_engine.mode
                if self.motion_engine
                else "CALIBRATION"
            )
            left_data = {
                "val": None,
                "rep_count": 0,
                "progress_ratio": 0.0,
                "state": "READY",
                "quality": "CALIBRATING",
            }
            right_data = {
                "val": None,
                "rep_count": 0,
                "progress_ratio": 0.0,
                "state": "READY",
                "quality": "CALIBRATING",
            }

            if yolo_detected and self.motion_engine:
              result = self.motion_engine.process_keypoints(keypoints)
              if result:
                current_mode = result["mode"]
                left_data = result["left"]
                right_data = result["right"]

                if result["mode"] == "MAIN":
                  kpt_matrix = [
                      [kp["x"], kp["y"], kp.get("score", 0.0)]
                      for kp in keypoints
                  ]
                  self.buf_timestamps.append(
                      round(time.time() - self.session_start_time, 3)
                  )
                  self.buf_values.append(
                      [left_data["val"] or 0.0, right_data["val"] or 0.0]
                  )
                  self.buf_keypoints.append(kpt_matrix)

                if result.get("new_custom_thresholds"):
                  new_th = result["new_custom_thresholds"]
                  self.data_manager.save_custom_threshold(
                      self.current_exercise, new_th, is_confirmed=True
                  )
                  await websocket.send(
                      json.dumps({
                          "type": "CALIBRATION_FINISHED",
                          "player_id": self.player_id,
                          "recommended_thresholds": new_th,
                      })
                  )
                  self.active_session = False

            payload = {
                "type": "POSE_UPDATE",
                "mode": current_mode,
                "left": left_data,
                "right": right_data,
                "keypoints": keypoints if yolo_detected else [],
                "fps": fps,
                "yolo_detected": yolo_detected,
                "frame_b64": frame_b64,
            }
            await websocket.send(json.dumps(payload))

            if yolo_detected and self.motion_engine and current_mode == "MAIN":
              if (
                  left_data["rep_count"] >= self.target_reps
                  and right_data["rep_count"] >= self.target_reps
              ):
                summary = self._save_current_session_data()
                if summary:
                  await websocket.send(
                      json.dumps(
                          {"type": "SESSION_FINISHED", "summary": summary}
                      )
                  )
                self.active_session = False

        elapsed = time.time() - loop_start
        await asyncio.sleep(max(0.001, self.frame_delay - elapsed))

    except websockets.exceptions.ConnectionClosed:
      print("[SocketServer] 클라이언트 연결 종료")
      if self.active_session and len(self.buf_timestamps) > 0:
        self._save_current_session_data()
    except Exception as e:
      print(f"[SocketServer Error] {e}")
      traceback.print_exc()
    finally:
      self.stop_camera()

  async def run(self):
    async with websockets.serve(self.handle_client, self.host, self.port):
      print(
          f"[SocketServer] AI Headless 서버 구동 중 (ws://{self.host}:{self.port})"
      )
      await asyncio.Future()


if __name__ == "__main__":
  server = ExerciseSocketServer()
  try:
    asyncio.run(server.run())
  except KeyboardInterrupt:
    print("\n[SocketServer] 서버 종료")