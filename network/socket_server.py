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
import json
import sys
import threading
import time
_BOOT_STARTED = time.perf_counter()
import traceback
import cv2
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

current_file_path = os.path.abspath(__file__)
network_dir = os.path.dirname(current_file_path)
root_dir = os.path.dirname(network_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from core.data_manager import DataManager, load_exercise_config
from core.motion_engine import MotionEngine
from core.model_loader import BackgroundModelLoader
from core.session_controller import SessionController
from utils.filters import OneEuroKeypointFilter
from utils.normalization import PoseNormalizer
from network.session_commands import COUNT_THRESHOLD_COMMANDS, handle_count_threshold_command, get_preparation_seconds, get_target_score


class ThreadedWebcam:
    """실시간 웹캠 최신 프레임 캡처 스레드 (모드 1, 모드 2 공용)"""
    def __init__(self, cam_idx=0):
        self.cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.grabbed, self.frame = self.cap.read()
        self.captured_at = time.perf_counter()
        self.frame_id = 0
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
            captured_at = time.perf_counter()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
                self.captured_at = captured_at
                self.frame_id += 1
            time.sleep(0.005)

    def read(self):
        grabbed, frame, _, _ = self.read_sample()
        return grabbed, frame

    def read_sample(self):
        """Return one atomic frame/timestamp/id snapshot from the capture thread."""
        with self.read_lock:
            if not self.grabbed or self.frame is None:
                return False, None, self.captured_at, self.frame_id
            return True, self.frame.copy(), self.captured_at, self.frame_id

    def stop(self):
        self.started = False
        if self.cap and self.cap.isOpened():
            self.cap.release()


class ExerciseSocketServer:
    def __init__(self, host="127.0.0.1", port=8080, engine_factory=None):
        self.host = host
        self.port = port

        self.skeleton_engine = None
        self.model_loader = BackgroundModelLoader(engine_factory or self._create_skeleton_engine)
        self.bound_port = None
        self._session_owner = None
        self._active_log = None
        self._live_timing = False
        self._last_capture_id = None
        self._measurement_ready_at = None
        self._last_preparation_remaining = None

        self.normalizer = PoseNormalizer(conf_threshold=0.35, invert_y=True)
        self.camera = None
        self.controller = None
        self.filter_engine = None
        self.last_frame_b64 = None
        self.output_mode = "VISUAL"

    @staticmethod
    def _create_skeleton_engine(report_phase):
        # Importing Torch/MMPose and registering models is deliberately deferred
        # until AFTER the server is listening, then runs in the loader thread.
        from core.skeleton_engine import SkeletonEngine
        return SkeletonEngine(yolo_interval=5, on_progress=report_phase)

    async def _start_session(self, websocket, packet):
        if self._session_owner is not None and self._session_owner is not websocket:
            await websocket.send(json.dumps({"type": "ERROR", "code": "SERVER_BUSY",
                                              "message": "다른 연결에서 세션을 사용 중입니다."}))
            return False
        self._session_owner = websocket
        self._last_client_finish = None
        video_path = packet.get("video_path") or packet.get("input_source")
        try:
            if self.controller:
                self._save_session_files(status="STOPPED")
                self.stop_camera()
                self.controller = None
            if video_path:
                if not os.path.isfile(str(video_path)):
                    raise ValueError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
                await self.process_offline_video(websocket, packet)
            else:
                self.init_live_session(packet)
                await websocket.send(json.dumps({
                    "type": "SESSION_LOG_STARTED", **self.controller.data_manager.session_log.info}))
        except ConnectionClosed:
            raise
        except Exception as exc:
            traceback.print_exc()
            try:
                if self.controller:
                    self._save_session_files(status="FAILED")
                elif self._active_log:
                    self._active_log.finish([], status="FAILED")
            except Exception:
                traceback.print_exc()
            self._active_log = None
            self.controller = None
            self.stop_camera()
            await websocket.send(json.dumps({
                "type": "ERROR", "code": "VIDEO_ANALYSIS_FAILED" if video_path else "SESSION_START_FAILED",
                "message": f"세션 시작/분석 실패: {exc}",
            }))
        return True

    def start_camera(self, cam_idx=0):
        if self.camera:
            self.camera.stop()
        self.camera = ThreadedWebcam(cam_idx).start()

    def stop_camera(self):
        if self.camera:
            self.camera.stop()
            self.camera = None

    def _prepare_data_manager(self, data, video_fps=30.0, live_timing=False):
        mode = data.get("mode", "CALIBRATION")
        if "cal" in data and (not isinstance(data["cal"], bool) or data["cal"] != (mode == "CALIBRATION")):
            raise ValueError("cal must match mode == CALIBRATION")
        manager = DataManager(
            player_id=data.get("patient_id", data.get("player_id", "patient_1")),
            patient_name=data.get("patient_name", "Unknown"), log_root=data.get("log_root"),
            patient_metadata=data.get("patient_metadata"))
        manager.begin_session(
            exercise_name=data.get("exercise_name", "biceps_curl"), mode=mode,
            session=data.get("session"), round_number=data.get("round", 1), set_number=data.get("set", 1),
            record_video=data.get("record_video", False), video_fps=video_fps,
            target_score=data.get("target_score"))
        self._active_log = manager.session_log
        self._session_started = time.perf_counter()
        self._live_timing = live_timing
        self._last_capture_id = None
        self._measurement_ready_at = None
        return manager

    def init_live_session(self, data: dict):
        """실시간 웹캠 세션 공통 초기화 (모드 1 & 모드 2)"""
        player_id = data.get("player_id", "patient_1")
        patient_name = data.get("patient_name", "Unknown")
        exercise_name = data.get("exercise_name", "biceps_curl")
        mode = data.get("mode", "CALIBRATION")
        target_score = get_target_score(data) if "target_score" in data or "target_reps" in data else 10.0
        cam_idx = int(data.get("camera_index", 0))
        self.output_mode = data.get("output_mode", "VISUAL").upper()
        preparation_seconds = get_preparation_seconds(data)

        print(f"[DEBUG][LIVE_INIT] Mode: {mode} | Output: {self.output_mode} | Exercise: {exercise_name}")
        data_manager = self._prepare_data_manager(data, live_timing=True)
        exercise_config = load_exercise_config(exercise_name)

        custom_thresholds = None
        if mode in ["MAIN", "TEST"]:
            custom_thresholds = data_manager.get_custom_threshold(exercise_name)

        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        effective_reps = 3 if mode == "CALIBRATION" else target_score

        motion_engine = MotionEngine(
            config=exercise_config,
            custom_thresholds=custom_thresholds,
            mode=mode,
            target_reps=effective_reps, target_score=target_score
        )
        setattr(motion_engine, "exercise_name", exercise_name)

        self.controller = SessionController(
            data_manager=data_manager,
            motion_engine=motion_engine
        )
        self.filter_engine = OneEuroKeypointFilter()
        self._reset_raw_buffers()
        self.start_camera(cam_idx)
        # Start the preparation clock only after the camera is available.
        self._session_started = time.perf_counter() + preparation_seconds
        self._measurement_ready_at = self._session_started if preparation_seconds else None
        self._last_preparation_remaining = None

    async def _measurement_ready(self, websocket):
        """Nonblocking gate: no inference, filtering, counting or recording during preparation."""
        if self._measurement_ready_at is None:
            return True
        remaining = self._measurement_ready_at - time.perf_counter()
        if remaining > 0:
            seconds = int(np.ceil(remaining))
            if seconds != self._last_preparation_remaining:
                await websocket.send(json.dumps({
                    "type": "SESSION_PREPARING", "remaining_seconds": seconds,
                    "message": f"기본 자세를 취해 주세요. {seconds}초 후 측정을 시작합니다."}))
                self._last_preparation_remaining = seconds
            return False
        self._measurement_ready_at = None
        await websocket.send(json.dumps({"type": "SESSION_MEASUREMENT_STARTED",
                                         "message": "준비 시간이 끝나 측정을 시작합니다."}))
        return True

    def _reset_raw_buffers(self):
        if self.controller and self.controller.data_manager:
            dm = self.controller.data_manager
            dm.buf_timestamps = dm.session_log.timestamps
            dm.buf_values = dm.session_log.values
            dm.buf_raw_keypoints = dm.session_log.keypoints

    def _append_to_buffers(self, left_data, right_data, raw_keypoints, elapsed_sec=None,
                           frame_idx=None, is_occluded=False, video_frame_idx=-1):
        log = self.controller.data_manager.session_log
        engine = self.controller.motion_engine
        if elapsed_sec is None:
            elapsed_sec = time.perf_counter() - self._session_started
        if frame_idx is None:
            frame_idx = len(log.timestamps)
        if is_occluded:
            left_data = {**left_data, "val": None, "is_count_updated": False}
            right_data = {**right_data, "val": None, "is_count_updated": False}
        log.append_frame(
            elapsed_sec, frame_idx, raw_keypoints, left_data, right_data,
            metric_type=engine.metric_type, thresholds=engine.get_count_thresholds(),
            threshold_version=engine.threshold_version, is_occluded=is_occluded,
            calib_step=getattr(self.controller, "calib_step", ""), video_frame_idx=video_frame_idx)

    def _save_session_files(self, prefix="SESSION", status="COMPLETED"):
        if not self.controller or not self.controller.data_manager:
            return None
        engine = self.controller.motion_engine
        rows = engine.fsm_left.completed_reps_history + engine.fsm_right.completed_reps_history
        end_elapsed = max(0.0, time.perf_counter() - self._session_started) if self._live_timing else None
        summary = self.controller.data_manager.finish_session(rows, status=status, end_elapsed_sec=end_elapsed)
        self._active_log = None
        return summary

    async def _send_rep_events(self, websocket, left, right, timestamp=None):
        log = self.controller.data_manager.session_log
        engine = self.controller.motion_engine
        for side, data in (("left", left), ("right", right)):
            if not data.get("is_count_updated"):
                continue
            await websocket.send(json.dumps({
                "type": "REP_COMPLETED", "session_id": log.session_id,
                "exercise_name": engine.exercise_name, "mode": engine.mode,
                "side": side, "rep_id": data["rep_count"], "quality": data["quality"],
                "threshold_version": engine.threshold_version,
                "timestamp": timestamp, "val": data.get("val"),
            }))

    async def _finish_client_session(self, websocket, packet):
        request_id = packet.get("request_id")
        def error(code, message):
            return {"type": "ERROR", "command": "CMD_STOP_SESSION", "code": code,
                    "request_id": request_id, "message": message}
        previous = getattr(self, "_last_client_finish", None)
        if self._session_owner is not websocket or not self.controller:
            if previous and previous[0] is websocket and self._session_owner is None:
                response = {**previous[1], "request_id": request_id}
            else:
                response = error("NO_ACTIVE_SESSION", "This connection does not own an active session")
            await websocket.send(json.dumps(response))
            return
        session_id = self.controller.data_manager.session_log.session_id
        if packet.get("session_id") not in (None, session_id):
            await websocket.send(json.dumps(error("SESSION_MISMATCH", "session_id does not match the active session")))
            return
        if self.controller.mode == "CALIBRATION" and packet.get("reason") == "TARGET_REACHED":
            await websocket.send(json.dumps(error("INVALID_MODE", "Calibration completion is managed by the server")))
            return
        status = "COMPLETED" if packet.get("reason") == "TARGET_REACHED" else "STOPPED"
        self._record_client_result(packet)
        summary = self._save_session_files(status=status)
        response = {"type": "SESSION_FINISHED", "session_id": session_id,
                    "request_id": request_id, "reason": packet.get("reason", "USER_STOP"),
                    "status": status, "summary": summary}
        self.stop_camera()
        self.controller = None
        self._session_owner = None
        self._last_client_finish = (websocket, response)
        await websocket.send(json.dumps(response))

    def _record_client_result(self, packet):
        # Client game results are optional log metadata, never a server stop rule.
        result = packet.get("client_result")
        if not isinstance(result, dict):
            return
        target = result.get("target_score")
        scores = result.get("scores")
        values = [target] + ([scores.get(s) for s in ("left", "right")] if isinstance(scores, dict) else [None])
        if not all(type(v) in (int, float) and np.isfinite(v) and v >= 0 for v in values):
            return
        log = self.controller.data_manager.session_log
        log.info["client_result"] = result
        log.info["target_score"] = target

    async def _poll_video_stop(self, websocket):
        # Offline analysis is awaited by handle_client, so it reads stop commands here.
        if not hasattr(websocket, "recv"):
            return None
        try:
            packet = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.001))
        except asyncio.TimeoutError:
            return None
        if packet.get("type") == "CMD_STOP_SESSION":
            session_id = self.controller.data_manager.session_log.session_id
            if packet.get("session_id") in (None, session_id):
                return packet
        await websocket.send(json.dumps({"type": "ERROR", "command": packet.get("type"),
            "request_id": packet.get("request_id"), "code": "VIDEO_COMMAND_REJECTED",
            "message": "Only a stop command for the active session is supported during video analysis"}))
        return None

    # ==========================================================================
    # [모드 3] 비디오 분석 방식 (OFFLINE_VIDEO): 배치 GPU 가속 검증
    # ==========================================================================
    async def process_offline_video(self, websocket, data: dict):
        video_path = data.get("video_path") or data.get("input_source")
        player_id = data.get("player_id", "patient_1")
        patient_name = data.get("patient_name", "Unknown")
        exercise_name = data.get("exercise_name", "biceps_curl")
        mode = data.get("mode", "CALIBRATION")
        target_score = get_target_score(data) if "target_score" in data or "target_reps" in data else 10.0
        batch_size = int(data.get("batch_size", 16))
        if batch_size <= 0:
            await websocket.send(json.dumps({"type": "ERROR", "code": "INVALID_BATCH_SIZE",
                                              "message": "batch_size must be greater than zero"}))
            return

        print(f"\n[MODE 3: VIDEO_BATCH] Processing '{video_path}' (Batch: {batch_size}, Mode: {mode})")

        await websocket.send(json.dumps({
            "type": "VIDEO_STATUS",
            "status": "PROCESSING",
            "message": "비디오를 고속 배치 분석 중입니다. 잠시만 기다려주세요..."
        }))

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            await websocket.send(json.dumps({"type": "ERROR", "message": f"비디오 파일을 열 수 없습니다: {video_path}"}))
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_duration = 1.0 / fps

        try:
            data_manager = self._prepare_data_manager(data, video_fps=fps)
        except Exception:
            cap.release()
            raise
        await websocket.send(json.dumps({"type": "SESSION_LOG_STARTED", **data_manager.session_log.info}))
        exercise_config = load_exercise_config(exercise_name)
        custom_thresholds = data_manager.get_custom_threshold(exercise_name) if mode in ["MAIN", "TEST"] else None
        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        effective_reps = 3 if mode == "CALIBRATION" else target_score
        motion_engine = MotionEngine(
            config=exercise_config,
            custom_thresholds=custom_thresholds,
            mode=mode,
            target_reps=effective_reps, target_score=target_score
        )
        setattr(motion_engine, "exercise_name", exercise_name)

        self.controller = SessionController(data_manager=data_manager, motion_engine=motion_engine)
        self.filter_engine = OneEuroKeypointFilter()
        self._reset_raw_buffers()

        # Step 1: 전체 프레임 GPU 고속 배치 키포인트 추출
        all_extracted_keypoints = []
        batch_frames = []
        decoded_frames = 0
        stop_request = None
        t_start = time.time()
        last_progress = time.perf_counter()

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    if batch_frames:
                        kpts_batch = await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)
                        all_extracted_keypoints.extend(kpts_batch)
                    break

                # File time, never wall-clock batch processing time.
                data_manager.session_log.write_video_frame(frame, decoded_frames * frame_duration)
                decoded_frames += 1
                batch_frames.append(frame)
                if len(batch_frames) >= batch_size:
                    kpts_batch = await asyncio.to_thread(self.skeleton_engine.extract_keypoints_batch, batch_frames)
                    all_extracted_keypoints.extend(kpts_batch)
                    batch_frames.clear()
                    stop_request = await self._poll_video_stop(websocket)
                    if stop_request:
                        break
                    if time.perf_counter() - last_progress >= 1.0:
                        await websocket.send(json.dumps({
                            "type": "VIDEO_PROGRESS", "phase": "EXTRACTING",
                            "processed_frames": decoded_frames, "total_frames": total_frames}))
                        last_progress = time.perf_counter()
        finally:
            cap.release()
        print(f"[MODE 3: VIDEO_BATCH] Extracted {len(all_extracted_keypoints)} frames in {time.time() - t_start:.2f}s")

        # Step 2: 프레임 누락 없는 순차 FSM 판정 (양측 3회 완수 검사)
        self.controller.timer_start = 0.0

        for frame_idx, raw_kpts in enumerate(all_extracted_keypoints):
            if stop_request:
                break
            virtual_time = frame_idx * frame_duration
            filtered_kpts = self.filter_engine.update(raw_kpts, timestamp=virtual_time)
            is_occluded = self.controller.check_occlusion(filtered_kpts) if filtered_kpts else True

            if mode == "CALIBRATION":
                cal_res = self.controller.process_calibration_frame(filtered_kpts, is_occluded, now=virtual_time)
                self._append_to_buffers(self.controller.last_left_data, self.controller.last_right_data,
                    raw_kpts, elapsed_sec=virtual_time, frame_idx=frame_idx, is_occluded=is_occluded,
                    video_frame_idx=frame_idx if data_manager.session_log.record_video else -1)

                if cal_res["calib_step"] == "FINISHED":
                    print(f"[MODE 3: VIDEO_BATCH] 양측 캘리브레이션 3회 완수 완료 (Frame: {frame_idx}/{total_frames})")
                    break

            elif mode in ["MAIN", "TEST"]:
                left_data, right_data, is_finished = self.controller.process_main_frame(filtered_kpts, is_occluded, now=virtual_time)
                self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=virtual_time,
                    frame_idx=frame_idx, is_occluded=is_occluded,
                    video_frame_idx=frame_idx if data_manager.session_log.record_video else -1)

                await self._send_rep_events(websocket, left_data, right_data, virtual_time)
            stop_request = await self._poll_video_stop(websocket)

        # Step 3: 파일 저장 및 완료 결과 일괄 전송
        prefix = "CALIB" if mode == "CALIBRATION" else "SESSION"
        calibration_complete = mode == "CALIBRATION" and self.controller.calib_step == "FINISHED"
        custom_th = self.controller.compute_and_save_thresholds(exercise_name) if calibration_complete else None
        final_step = self.controller.calib_step if mode == "CALIBRATION" else None
        if stop_request:
            self._record_client_result(stop_request)
        summary_meta = self._save_session_files(
            prefix=prefix, status=("COMPLETED" if stop_request and stop_request.get("reason") == "TARGET_REACHED"
                                   else "STOPPED" if stop_request
                                   else "INCOMPLETE" if mode == "CALIBRATION" and not calibration_complete else "COMPLETED"))

        payload = {
            "type": "SESSION_FINISHED" if stop_request else "VIDEO_ANALYSIS_COMPLETED",
            "session_id": data_manager.session_log.session_id,
            "request_id": stop_request.get("request_id") if stop_request else None,
            "reason": stop_request.get("reason", "USER_STOP") if stop_request else "END_OF_VIDEO",
            "mode": mode,
            "video_path": video_path,
            "total_frames_analyzed": len(all_extracted_keypoints),
            "source_duration_sec": decoded_frames * frame_duration,
            "summary": summary_meta
        }
        if mode == "CALIBRATION":
            payload["thresholds"] = custom_th
            payload["calibration_completed"] = calibration_complete
            payload["calib_step"] = final_step

        if stop_request:
            payload["status"] = "COMPLETED" if stop_request.get("reason") == "TARGET_REACHED" else "STOPPED"
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
        pending_session = None
        pending_commands = []
        last_model_status = None
        end_status = "STOPPED"
        self.model_loader.start()

        try:
            while True:
                status = self.model_loader.snapshot()
                status_key = (status["model_state"], status["phase"], status.get("model_error"))
                if status_key != last_model_status:
                    await websocket.send(json.dumps(status))
                    last_model_status = status_key
                if pending_session and status["model_state"] == "FAILED":
                    await websocket.send(json.dumps({
                        "type": "ERROR", "code": "MODEL_LOAD_FAILED", "fatal": True,
                        "message": f"모델 준비 실패: {status.get('model_error')}",
                    }))
                    pending_session = None
                    pending_commands.clear()
                elif pending_session and status["ready"]:
                    self.skeleton_engine = self.model_loader.engine
                    packet_to_start, pending_session = pending_session, None
                    accepted = await self._start_session(websocket, packet_to_start)
                    if accepted:
                        for command in pending_commands:
                            await websocket.send(json.dumps(handle_count_threshold_command(self.controller, command)))
                    pending_commands.clear()
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    packet = json.loads(raw_msg)
                    pkt_type = packet.get("type")

                    if pkt_type == "CMD_SET_SESSION":
                        pending_session = packet
                        pending_commands.clear()
                        if not self.model_loader.snapshot()["ready"]:
                            await websocket.send(json.dumps({
                                "type": "SESSION_PENDING",
                                "message": "모델 준비가 끝나면 요청한 세션을 시작합니다.",
                            }))

                    elif pkt_type == "CMD_STOP_SESSION":
                        if pending_session and self._session_owner in (None, websocket):
                            pending_session = None
                            pending_commands.clear()
                            await websocket.send(json.dumps({"type": "SESSION_FINISHED", "status": "STOPPED",
                                "reason": "USER_STOP", "request_id": packet.get("request_id"), "summary": None}))
                        else:
                            await self._finish_client_session(websocket, packet)

                    elif pkt_type == "CMD_GET_SERVER_STATUS":
                        await websocket.send(json.dumps(self.model_loader.snapshot()))

                    elif pkt_type == "CMD_STOP_CALIBRATION":
                        if self._session_owner is websocket and self.controller and self.controller.mode == "CALIBRATION":
                            self.controller.calib_step = "FINISHED"

                    elif pkt_type in COUNT_THRESHOLD_COMMANDS:
                        if pending_session:
                            if len(pending_commands) < 16:
                                pending_commands.append(packet)
                        elif self._session_owner is not None and self._session_owner is not websocket:
                            await websocket.send(json.dumps({"type": "ERROR", "code": "SERVER_BUSY",
                                "command": pkt_type, "request_id": packet.get("request_id"),
                                "message": "다른 연결에서 세션을 사용 중입니다."}))
                        else:
                            response = handle_count_threshold_command(self.controller, packet)
                            await websocket.send(json.dumps(response))

                except asyncio.TimeoutError:
                    pass

                if self._session_owner is websocket and self.controller and self.camera:
                    if not await self._measurement_ready(websocket):
                        await asyncio.sleep(0.01)
                        continue
                    ret, frame, sample_time, capture_id = self.camera.read_sample()

                    if (ret and frame is not None and capture_id != self._last_capture_id
                            and sample_time >= self._session_started):
                        self._last_capture_id = capture_id
                        log_elapsed = sample_time - self._session_started
                        video_index = self.controller.data_manager.session_log.write_video_frame(frame, log_elapsed)
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

                        # Use the camera timestamp for filtering/logs as well as video.
                        raw_kpts = await asyncio.to_thread(self.skeleton_engine.extract_keypoints, frame)
                        filtered_kpts = self.filter_engine.update(raw_kpts, timestamp=sample_time)
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

                            left_data, right_data = self.controller.last_left_data, self.controller.last_right_data
                            self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=log_elapsed,
                                is_occluded=is_occluded, video_frame_idx=video_index)

                            if cal_res["calib_step"] == "FINISHED":
                                ex_name = getattr(self.controller.motion_engine, "exercise_name", "biceps_curl")
                                custom_th = self.controller.compute_and_save_thresholds(ex_name)
                                calib_summary = self._save_session_files(prefix="CALIB")

                                await websocket.send(json.dumps({
                                    "type": "CALIBRATION_FINISHED",
                                    "player_id": self.controller.data_manager.player_id,
                                    "thresholds": custom_th,
                                    "summary": calib_summary
                                }))
                                self.controller = None
                                self.stop_camera()
                                self._session_owner = None
                                continue

                        # --- TEST / MAIN 진행 ---
                        elif current_mode in ["MAIN", "TEST"]:
                            left_data, right_data, is_finished = self.controller.process_main_frame(filtered_kpts, is_occluded)
                            self._append_to_buffers(left_data, right_data, raw_kpts, elapsed_sec=log_elapsed,
                                is_occluded=is_occluded, video_frame_idx=video_index)

                            await self._send_rep_events(websocket, left_data, right_data, log_elapsed)

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

                        pose_payload.update(self.controller.motion_engine.count_threshold_status())
                        await websocket.send(json.dumps(pose_payload))

                await asyncio.sleep(0.001)

        except ConnectionClosed:
            print(f"[DEBUG][NETWORK] Client disconnected: {client_address}")
        except Exception as e:
            end_status = "FAILED"
            print(f"[ERROR][SERVER] Exception: {e}")
            traceback.print_exc()
            try:
                await websocket.send(json.dumps({"type": "ERROR", "code": "SESSION_PROCESSING_FAILED",
                                                  "message": str(e)}))
            except ConnectionClosed:
                pass
        finally:
            if self._session_owner is websocket:
                try:
                    if self._active_log is not None:
                        if self.controller:
                            self._save_session_files(status=end_status)
                        else:
                            self._active_log.finish([], status=end_status)
                except Exception:
                    traceback.print_exc()
                finally:
                    self._active_log = None
                    self.stop_camera()
                    self.controller = None
                    self._session_owner = None

    async def run(self):
        async with websockets.serve(self.handle_client, self.host, self.port) as listener:
            self.bound_port = listener.sockets[0].getsockname()[1]
            print(f"[DEBUG][SERVER] ExerciseSocketServer listening on ws://{self.host}:{self.bound_port}", flush=True)
            print(f"[STARTUP] Socket listening after {time.perf_counter() - _BOOT_STARTED:.3f}s; models load in background", flush=True)
            self.model_loader.start()
            try:
                await asyncio.Future()
            finally:
                task = self.model_loader.task
                if task is not None and not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)


if __name__ == "__main__":
    server = ExerciseSocketServer()
    try:
        asyncio.run(server.run())
    except KeyboardInterrupt:
        print("\n[DEBUG][SERVER] Server stopped by user.")
