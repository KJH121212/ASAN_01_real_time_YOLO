# ==============================================================================
# [Module Information]
# File: network/socket_server.py
# Description: WebSocket streaming server supporting a strictly 3-repetition
#              calibration workflow, multi-source input (Webcam, Video File,
#              Image Sequence Directory), and saving unadulterated RAW keypoints.
# ==============================================================================

import os
os.environ["MMCV_WITH_OPS"] = "0"

import asyncio
import base64
from datetime import datetime
from glob import glob
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


class ThreadedCamera:
    """
    웹캠 인덱스, 단일 동영상 파일(MP4 등), 이미지 시퀀스 디렉터리를
    모두 지원하며 백그라운드 스레드에서 최신 1프레임을 공급하는 리더.
    """

    def __init__(self, source=0, target_fps=30.0):
        self.source = source
        self.target_fps = target_fps
        self.frame_delay = 1.0 / target_fps

        self.mode = "CAM"  # "CAM", "VIDEO", "IMAGE_DIR"
        self.image_files = []
        self.image_idx = 0
        self.cap = None

        # 1. 이미지 디렉터리 경로 검사
        if isinstance(source, str) and os.path.isdir(source):
            self.mode = "IMAGE_DIR"
            exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
            for ext in exts:
                self.image_files.extend(glob(os.path.join(source, ext)))
            self.image_files.sort()
            print(f"[DEBUG][SOURCE] Image Directory mode: {len(self.image_files)} frames found in '{source}'")

            if self.image_files:
                self.frame = cv2.imread(self.image_files[0])
                self.grabbed = self.frame is not None
            else:
                self.grabbed, self.frame = False, None

        # 2. 동영상 파일 경로 검사
        elif isinstance(source, str) and os.path.isfile(source):
            self.mode = "VIDEO"
            self.cap = cv2.VideoCapture(source)
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self.frame_delay = 1.0 / fps
            self.grabbed, self.frame = self.cap.read()
            print(f"[DEBUG][SOURCE] Video File mode: '{source}' (FPS: {1.0 / self.frame_delay:.1f})")

        # 3. 실시간 물리 웹캠 모드 (기본)
        else:
            self.mode = "CAM"
            cam_idx = int(source) if str(source).isdigit() else 0
            self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.grabbed, self.frame = self.cap.read()
            self.frame_delay = 0.005
            print(f"[DEBUG][SOURCE] Live Webcam mode (Index: {cam_idx}), Initial Grab: {self.grabbed}")

        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        print(f"[DEBUG][SOURCE] Capture worker thread active for mode: {self.mode}")
        return self

    def update(self):
        while self.started:
            start_t = time.time()

            if self.mode == "IMAGE_DIR":
                if not self.image_files:
                    time.sleep(0.03)
                    continue

                self.image_idx = (self.image_idx + 1) % len(self.image_files)
                frame = cv2.imread(self.image_files[self.image_idx])
                grabbed = frame is not None

                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame

            elif self.mode == "VIDEO":
                grabbed, frame = self.cap.read()
                # 영상이 끝나면 0번 프레임으로 되감아 무한 루프 반복
                if not grabbed or frame is None:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    grabbed, frame = self.cap.read()

                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame

            elif self.mode == "CAM":
                grabbed, frame = self.cap.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame

            # FPS 동기화 슬립
            elapsed = time.time() - start_t
            sleep_time = self.frame_delay - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def read(self):
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self):
        self.started = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        print(f"[DEBUG][SOURCE] Source ({self.mode}) resource disposed.")


class ExerciseSocketServer:
    """
    3회 반복 기반 캘리브레이션 및 실시간 모션 측정을 전담하는 WebSocket 서버 클래스.
    웹캠, 비디오 파일, 이미지 디렉터리 입력을 모두 지원합니다.
    """

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

    def start_camera(self, source=0):
        if self.camera:
            self.camera.stop()
        self.camera = ThreadedCamera(source).start()

    def stop_camera(self):
        if self.camera:
            self.camera.stop()
            self.camera = None

    def init_session(self, data: dict):
        """세션 파라미터를 파싱하여 3회 캘리브레이션 또는 본 운동 엔진을 구성합니다."""
        player_id = data.get("player_id", "patient_1")
        patient_name = data.get("patient_name", "Unknown")
        exercise_name = data.get("exercise_name", "biceps_curl")
        mode = data.get("mode", "CALIBRATION")
        target_reps = int(data.get("target_reps", 10))

        # [B안 지원 분기] input_source, video_path, image_dir을 순차 확인 후 기본 camera_index로 폴백
        source = (
            data.get("input_source")
            or data.get("video_path")
            or data.get("image_dir")
        )
        if source is None:
            source = int(data.get("camera_index", 0))

        print(f"[DEBUG][SESSION] Initializing session -> Player: {player_id}, Exercise: {exercise_name}, Mode: {mode}, Source: {source}")
        self.start_camera(source)

        data_manager = DataManager(player_id=player_id, patient_name=patient_name)
        exercise_config = load_exercise_config(exercise_name)

        # TEST 모드인 경우에만 기존 결과 JSON에서 임계값 로드
        custom_thresholds = None
        if mode in ["MAIN", "TEST"]:
            custom_thresholds = data_manager.get_custom_threshold(exercise_name)
            print(f"[DEBUG][SESSION] Loaded custom threshold profile: {custom_thresholds}")

        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        # 캘리브레이션 세션일 경우 FSM 내부 타겟 횟수를 3회로 강제 지정
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

        # 순수 원본 텐서 저장을 위한 버퍼 초기화
        self._reset_raw_buffers()
        print(f"[DEBUG][SESSION] SessionController configured. Effective Target Reps: {effective_reps}")

    def _reset_raw_buffers(self):
        """인메모리 관절 좌표 버퍼 플러시"""
        if self.controller and self.controller.data_manager:
            dm = self.controller.data_manager
            dm.buf_timestamps = []
            dm.buf_values = []
            dm.buf_raw_keypoints = []
            print("[DEBUG][BUFFER] Memory trajectory buffers reset.")

    def _append_to_buffers(self, left_data: dict, right_data: dict, raw_keypoints: list):
        """
        후처리가 배제된 순수 원본 17개 관절 텐서를 메모리 버퍼에 적재합니다.
        """
        if not raw_keypoints or len(raw_keypoints) < 17:
            return

        raw_kpt_matrix = [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in raw_keypoints]
        elapsed_sec = round(time.time() - self.controller.timer_start, 3)

        dm = self.controller.data_manager
        dm.buf_timestamps.append(elapsed_sec)
        dm.buf_values.append([left_data.get("val") or 0.0, right_data.get("val") or 0.0])
        dm.buf_raw_keypoints.append(raw_kpt_matrix)

    def _save_session_files(self, prefix: str = "SESSION") -> dict:
        """
        세션 종료 시 원본 궤적 NPZ 및 통계 CSV를 영구 보존합니다.
        """
        dm = self.controller.data_manager
        if not dm or not getattr(dm, "buf_timestamps", []):
            print(f"[DEBUG][IO] No frames collected for prefix {prefix}. Skipping save.")
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

        summary_meta = {
            "session_id": f"{prefix}_{session_id}",
            "prefix": prefix,
            "saved_frames": len(dm.buf_timestamps),
            "saved_reps_count": saved_reps
        }
        print(f"[DEBUG][IO] Trajectory archive committed: {summary_meta}")
        return summary_meta

    async def handle_client(self, websocket):
        """웹소켓 실시간 스트리밍 및 제어 루프"""
        client_address = websocket.remote_address
        print(f"[DEBUG][NETWORK] Client connected: {client_address}")

        prev_time = time.time()
        frame_counter = 0

        try:
            while True:
                # 1. 제어 패킷 논블로킹 수신
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                    packet = json.loads(raw_msg)
                    pkt_type = packet.get("type")
                    print(f"[DEBUG][NETWORK] Received command: {pkt_type}")

                    if pkt_type == "CMD_SET_SESSION":
                        self.init_session(packet)
                    elif pkt_type == "CMD_STOP_CALIBRATION":
                        if self.controller and self.controller.mode == "CALIBRATION":
                            print("[DEBUG][CALIB] Forced termination requested by client.")
                            self.controller.calib_step = "FINISHED"
                except asyncio.TimeoutError:
                    pass

                # 2. 비전 분석 및 상태 전이 파이프라인
                if self.controller and self.camera:
                    ret, frame = self.camera.read()

                    if ret and frame is not None:
                        curr_time = time.time()
                        time_delta = curr_time - prev_time
                        fps = round(1.0 / time_delta, 1) if time_delta > 0 else 30.0
                        prev_time = curr_time
                        frame_counter += 1

                        # 실시간 화면 송출용 저용량 JPEG 압축 (2프레임 당 1회)
                        if frame_counter % 2 == 0:
                            def encode_frame(img):
                                small = cv2.resize(img, (400, 300))
                                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])
                                return base64.b64encode(buf).decode("utf-8")

                            self.last_frame_b64 = await asyncio.to_thread(encode_frame, frame)

                        # Step 1: 비전 모델 관절 추론 (후처리 전 순수 원본 좌표)
                        raw_kpts = await asyncio.to_thread(self.skeleton_engine.extract_keypoints, frame)

                        # Step 2: 실시간 화면 렌더링용 평활화
                        filtered_kpts = self.filter_engine.update(raw_kpts)
                        yolo_detected = filtered_kpts is not None and len(filtered_kpts) > 0

                        # Step 3: 가림 여부 검사
                        is_occluded = self.controller.check_occlusion(filtered_kpts) if yolo_detected else True

                        current_mode = self.controller.motion_engine.mode
                        left_data = self.controller.last_left_data
                        right_data = self.controller.last_right_data
                        calib_rep_count = 0

                        # ------------------------------------------------------
                        # [MODE 1] CALIBRATION: 고정 3회 동작 수집 파이프라인
                        # ------------------------------------------------------
                        if current_mode == "CALIBRATION":
                            cal_res = self.controller.process_calibration_frame(filtered_kpts, is_occluded)
                            calib_rep_count = cal_res["calib_rep_count"]

                            # 수집 중 가림이 없을 때 [순수 원본 텐서]를 누적
                            if cal_res["calib_step"] == "COLLECTING" and not is_occluded and raw_kpts:
                                vl = self.controller.calib_vals_left[-1] if self.controller.calib_vals_left else 0.0
                                vr = self.controller.calib_vals_right[-1] if self.controller.calib_vals_right else 0.0
                                self._append_to_buffers({"val": vl}, {"val": vr}, raw_kpts)

                            # 3회 반복 완수 시 저장 및 후처리 실행
                            if cal_res["calib_step"] == "FINISHED":
                                print("[DEBUG][CALIB] 3 Repetitions reached. Saving RAW NPZ and post-processing thresholds...")
                                ex_name = getattr(self.controller.motion_engine, "exercise_name", "biceps_curl")

                                # 1. 순수 원본 궤적 저장 (CALIB 접두사)
                                calib_summary = self._save_session_files(prefix="CALIB")

                                # 2. 백분위수 기반 임계값 후처리 및 JSON 저장
                                custom_th = self.controller.compute_and_save_thresholds(ex_name)

                                finish_payload = {
                                    "type": "CALIBRATION_FINISHED",
                                    "player_id": self.controller.data_manager.player_id,
                                    "thresholds": custom_th,
                                    "summary": calib_summary
                                }
                                await websocket.send(json.dumps(finish_payload))
                                print(f"[DEBUG][CALIB] Completion packet sent: {finish_payload}")

                                self.controller = None
                                continue

                        # ------------------------------------------------------
                        # [MODE 2] TEST / MAIN: FSM 횟수 측정 및 목표 달성 저장
                        # ------------------------------------------------------
                        elif current_mode in ["MAIN", "TEST"]:
                            left_data, right_data, is_finished = self.controller.process_main_frame(filtered_kpts, is_occluded)

                            if not is_occluded and raw_kpts:
                                self._append_to_buffers(left_data, right_data, raw_kpts)

                            if is_finished:
                                print("[DEBUG][TEST] Target reps reached. Saving session logs...")
                                test_summary = self._save_session_files(prefix="SESSION")

                                finish_payload = {
                                    "type": "SESSION_FINISHED",
                                    "summary": test_summary
                                }
                                await websocket.send(json.dumps(finish_payload))
                                print(f"[DEBUG][TEST] Session finished packet sent: {test_summary}")

                                self.controller = None
                                continue

                        # 실시간 화면 동기화 패킷 송출
                        pose_payload = {
                            "type": "POSE_UPDATE",
                            "mode": current_mode,
                            "left": left_data,
                            "right": right_data,
                            "keypoints": filtered_kpts if yolo_detected else [],
                            "fps": fps,
                            "yolo_detected": yolo_detected,
                            "is_occluded": is_occluded,
                            "frame_b64": self.last_frame_b64,
                            "calib_step": self.controller.calib_step if self.controller else "FINISHED",
                            "calib_rep_count": calib_rep_count,
                            "target_calib_reps": 3,
                        }
                        await websocket.send(json.dumps(pose_payload))

                await asyncio.sleep(0.001)

        except websockets.exceptions.ConnectionClosed:
            print(f"[DEBUG][NETWORK] Client disconnected: {client_address}")
            if self.controller:
                prefix = "CALIB" if self.controller.mode == "CALIBRATION" else "SESSION"
                print(f"[DEBUG][IO] Flushing buffers before teardown. Prefix: {prefix}")
                self._save_session_files(prefix=prefix)
        except Exception as e:
            print(f"[ERROR][SERVER] Exception in main event loop: {e}")
            traceback.print_exc()
        finally:
            self.stop_camera()

    async def run(self):
        """서버 리스너 기동"""
        async with websockets.serve(self.handle_client, self.host, self.port):
            print(f"[DEBUG][SERVER] ExerciseSocketServer active on ws://{self.host}:{self.port}")
            await asyncio.Future()


if __name__ == "__main__":
    server = ExerciseSocketServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[DEBUG][SERVER] Stopped by KeyboardInterrupt.")