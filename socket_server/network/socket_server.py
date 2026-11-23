"""23개 관절(COCO-23) 실시간 측정 · 영상 미리보기(VISUAL) · 오프라인 영상 배치 분석(OFFLINE_VIDEO) 통합 WebSocket 서버."""

import asyncio
import base64
from collections import deque
import json
import math
from pathlib import Path
import threading
import time
import traceback

import cv2
import numpy as np
import websockets
from websockets.exceptions import ConnectionClosed

from core.data_manager import DataManager, load_exercise_config
from core.motion_engine import MotionEngine, config_for_mode
from core.session_controller import SessionController
from network.camera import (
    ThreadedWebcam,
    configure_local_and_gopro_no_proxy,
    downscale_frame_if_needed,
    scan_available_cameras,
)
from utils.filters import OneEuroKeypointFilter
from utils.pose_overlay import render_overlay

PREVIEW_SIZE = (640, 360)
PREVIEW_FPS = 25
PREVIEW_JPEG_QUALITY = 55
DEFAULT_PREPARATION_SECONDS = 3.0
COUNT_THRESHOLD_COMMANDS = {"CMD_GET_COUNT_THRESHOLD", "CMD_UPDATE_COUNT_THRESHOLD"}

PHASE_MESSAGES = {
    "NOT_STARTED": "모델 준비 대기 중입니다.",
    "IMPORTING": "AI 라이브러리를 불러오는 중입니다.",
    "DEVICE_SETUP": "연산 장치를 확인하는 중입니다.",
    "LOADING_YOLO": "사람 검출 모델을 불러오는 중입니다.",
    "LOADING_POSE": "23관절 추출 모델(RTMPose-COCO23)을 불러오는 중입니다.",
    "READY": "모델 준비가 완료되었습니다.",
    "FAILED": "모델 준비에 실패했습니다.",
}


def available_output_modes():
    """서버가 지원하는 세 가지 출력 모드를 반환한다."""
    return ("VISUAL", "LIGHTWEIGHT", "OFFLINE_VIDEO")


def resolve_output_mode(packet):
    """요청 패킷에서 출력 모드(VISUAL / LIGHTWEIGHT / OFFLINE_VIDEO)를 결정한다."""
    output = str(packet.get("output_mode", "LIGHTWEIGHT")).upper()
    if output not in {"VISUAL", "LIGHTWEIGHT", "OFFLINE_VIDEO"}:
        raise ValueError("output_mode must be VISUAL, LIGHTWEIGHT or OFFLINE_VIDEO")
    video = packet.get("video_path") or packet.get("input_source")
    if video:
        if output == "VISUAL":
            raise ValueError("영상 파일은 OFFLINE_VIDEO 모드를 사용하세요.")
        output = "OFFLINE_VIDEO"
    elif output == "OFFLINE_VIDEO":
        raise ValueError("OFFLINE_VIDEO 모드에는 video_path가 필요합니다.")
    return output


def get_target_score(packet):
    value = packet.get("target_score", packet.get("target_reps", 10))
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError("target_score must be a finite positive number")
    return float(value)


def get_target_reps(packet):
    value = packet.get("target_reps")
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError("target_reps must be a positive integer")
    return value


def get_preparation_seconds(packet):
    value = packet.get("preparation_seconds", DEFAULT_PREPARATION_SECONDS)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("preparation_seconds must be a finite nonnegative number")
    return float(value)


def handle_count_threshold_command(controller, packet: dict) -> dict:
    """실행 중인 세션의 카운트 임계값 조회·수정 명령을 처리한다."""
    request_id = packet.get("request_id")

    def error(code, message):
        return {
            "type": "ERROR",
            "request_id": request_id,
            "command": packet.get("type"),
            "code": code,
            "message": message,
        }

    command = packet.get("type")
    if command not in COUNT_THRESHOLD_COMMANDS:
        return error("UNKNOWN_COMMAND", "Unsupported count threshold command")
    if controller is None or controller.motion_engine is None:
        return error("NO_ACTIVE_SESSION", "Start a session before accessing count thresholds")
    engine = controller.motion_engine
    if packet.get("exercise_name") != engine.exercise_name:
        return error("EXERCISE_MISMATCH", "exercise_name must match the active session")

    changed_sides = []
    if command == "CMD_UPDATE_COUNT_THRESHOLD":
        if engine.mode != "MAIN":
            return error("INVALID_MODE", "Count threshold updates require MAIN mode")
        try:
            changed_sides = controller.update_count_thresholds(packet.get("thresholds"))
        except ValueError as exc:
            return error("INVALID_THRESHOLD", str(exc))

    return {
        "type": "COUNT_THRESHOLD_UPDATED" if command == "CMD_UPDATE_COUNT_THRESHOLD" else "COUNT_THRESHOLD_STATUS",
        "request_id": request_id,
        **engine.count_threshold_status(),
        "changed_sides": changed_sides,
        "states": {"left": engine.fsm_left.state, "right": engine.fsm_right.state},
        "scope": "SESSION",
    }


class BackgroundModelLoader:
    """서버 소켓 개방을 지연시키지 않도록 YOLO + RTMPose(23kpt) 모델을 백그라운드에서 로드한다."""

    def __init__(self, factory):
        self.factory = factory
        self.engine = None
        self.task = None
        self._lock = threading.Lock()
        self._state = "NOT_STARTED"
        self._phase = "NOT_STARTED"
        self._error = None
        self._started = None
        self._phase_started = None
        self._elapsed = 0.0
        self._timings = {}

    def start(self):
        if self.task is None:
            with self._lock:
                self._started = time.perf_counter()
                self._state = "LOADING"
            self.report_phase("IMPORTING")
            self.task = asyncio.create_task(self._load())
        return self.task

    def report_phase(self, phase):
        now = time.perf_counter()
        with self._lock:
            if phase == self._phase:
                return
            if self._phase_started is not None:
                self._timings[self._phase] = now - self._phase_started
            self._phase = phase
            self._phase_started = now
        print(f"[STARTUP] {phase}: {PHASE_MESSAGES.get(phase, phase)}", flush=True)

    async def _load(self):
        try:
            engine = await asyncio.to_thread(self.factory, self.report_phase)
            if engine is None:
                raise RuntimeError("Model factory returned no engine")
            self.report_phase("READY")
            with self._lock:
                self.engine = engine
                self._state = "READY"
                self._elapsed = time.perf_counter() - self._started
            print(f"[STARTUP] Models ready in {self._elapsed:.3f}s", flush=True)
        except Exception as exc:
            self.report_phase("FAILED")
            with self._lock:
                self._state = "FAILED"
                self._error = str(exc)
                self._elapsed = time.perf_counter() - self._started
            traceback.print_exc()

    def snapshot(self):
        with self._lock:
            timings = dict(self._timings)
            elapsed = self._elapsed
            if self._state == "LOADING":
                now = time.perf_counter()
                elapsed = now - self._started
                timings[self._phase] = now - self._phase_started
            return {
                "type": "SERVER_STATUS",
                "model_state": self._state,
                "phase": self._phase,
                "ready": self._state == "READY",
                "message": PHASE_MESSAGES.get(self._phase, self._phase),
                "model_error": self._error,
                "elapsed_seconds": round(elapsed, 3),
                "phase_seconds": {phase: round(seconds, 3) for phase, seconds in timings.items()},
            }


def encode_preview(frame, keypoints, pose, exercise_name, elapsed_sec):
    """미리보기 전용 축소 프레임에 23관절 스켈레톤과 진행바를 그려 JPEG로 인코딩한다."""
    height, width = frame.shape[:2]
    target_w, target_h = (PREVIEW_SIZE[1], PREVIEW_SIZE[0]) if height > width else PREVIEW_SIZE
    scale = min(target_w / width, target_h / height, 1.0)
    if scale < 1:
        frame = cv2.resize(
            frame,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    image = render_overlay(frame, keypoints, pose, exercise_name, elapsed_sec)
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    if not ok:
        raise RuntimeError("미리보기 JPEG 인코딩에 실패했습니다.")
    return {
        "type": "VIDEO_FRAME",
        "encoding": "jpeg",
        "image_base64": base64.b64encode(encoded).decode("ascii"),
        "width": image.shape[1],
        "height": image.shape[0],
        "timestamp": elapsed_sec,
    }


async def send_preview(server, websocket, frame, keypoints, pose, elapsed_sec):
    """VISUAL 모드에서 최대 10 FPS로 미리보기 프레임을 전송한다."""
    if server.output_mode != "VISUAL":
        return
    now = time.perf_counter()
    if now - server._last_preview_at < 1 / PREVIEW_FPS:
        return
    server._last_preview_at = now
    exercise = server.controller.motion_engine.exercise_name
    packet = await asyncio.to_thread(encode_preview, frame, keypoints, pose, exercise, elapsed_sec)
    await websocket.send(json.dumps(packet))


def missing_joints(server, keypoints):
    """현재 운동 평가에 필수인 관절이 화면에서 빠졌는지 확인한다."""
    if server.controller is None:
        return False
    engine = server.controller.motion_engine
    config = engine.config["eval_config"]
    required = {5, 6, 11, 12} if engine.use_normalization else set()
    for field in ("target_kpt", "reference_kpt", "primary_kpts"):
        for ids in config.get(field, {}).values():
            required.update(ids if isinstance(ids, list) else [ids])
    visible = {
        p["id"]
        for p in keypoints or []
        if p.get("score", 0) >= 0.35
        and p.get("x") is not None
        and p.get("y") is not None
        and math.isfinite(p["x"])
        and math.isfinite(p["y"])
        and 0 <= p["x"] <= 1
        and 0 <= p["y"] <= 1
    }
    return not required.issubset(visible)


async def watch_camera(server, websocket, packet):
    """측정 세션과 독립적으로 카메라 미리보기 스트림을 전송한다."""
    camera_index = int(packet.get("camera_index", 0))
    camera_url = (packet.get("camera_url") or "").strip() or None
    max_width = int(packet.get("camera_max_width", 1280))
    max_height = int(packet.get("camera_max_height", 720))
    server._preview_watchers += 1
    try:
        if server.controller and (
            server._camera_url != camera_url
            or (camera_url is None and server._camera_index not in (None, camera_index))
        ):
            await websocket.send(json.dumps({"type": "PREVIEW_STATUS", "warning": "측정 중인 카메라와 설정이 다릅니다."}))
            return
        while True:
            started = time.perf_counter()
            camera_failed = True
            try:
                await asyncio.to_thread(server.start_camera, camera_index, max_width, max_height, camera_url)
                ok, frame, captured_at, _ = server.camera.read_sample()
                if not ok:
                    raise RuntimeError("카메라에서 영상을 받지 못하고 있습니다. 연결을 확인해 주세요.")
                if time.perf_counter() - captured_at > 4.0:
                    raise RuntimeError("카메라 영상 갱신이 중단되었습니다. 연결을 복구하고 있습니다.")
                camera_failed = False
                pose = dict(server._preview_pose)
                points = server._preview_keypoints
                warning, notice = "", ""
                hide_measurement = False
                model = server.model_loader.snapshot()
                if not model["ready"]:
                    hide_measurement = True
                    notice = model.get("message", "모델을 준비하고 있습니다.")
                    if model["model_state"] == "FAILED":
                        warning = "모델 준비에 실패했습니다. 서버 상태를 확인해 주세요."
                elif server.controller is None:
                    hide_measurement = True
                    notice = "측정이 진행 중이 아닙니다. 현재 카메라 영상만 표시합니다."
                elif getattr(server, "_preparation_active", False):
                    hide_measurement = True
                    if server._measurement_ready_at and time.perf_counter() < server._measurement_ready_at:
                        remaining_sec = max(1, int(math.ceil(server._measurement_ready_at - time.perf_counter())))
                        notice = f"관절 인식 완료! ({remaining_sec}초) 후 측정을 시작합니다."
                    else:
                        notice = "카메라 및 관절 인식을 대기하고 있습니다. 화면에 자세를 잡아 주세요."
                elif time.perf_counter() - server._preview_updated_at > 1.0:
                    hide_measurement = True
                    warning = "측정 결과를 기다리고 있습니다. 카메라를 향해 자세를 유지해 주세요."
                    points = []
                elif pose.get("is_occluded") or missing_joints(server, points):
                    warning = "관절이 보이지 않습니다. 운동에 필요한 관절이 화면에 모두 보이도록 자세를 조정해 주세요."
                if server.controller is None or not model["ready"]:
                    points = []
                if hide_measurement:
                    if not getattr(server, "_preparation_active", False):
                        points = []
                    pose = {
                        **pose,
                        "is_occluded": False,
                        **{
                            side: {**pose.get(side, {}), "val": None, "progress_ratio": None}
                            for side in ("left", "right")
                        },
                    }
                controller = server.controller
                exercise = controller.motion_engine.exercise_name if controller else packet.get("exercise_name", "")
                elapsed = max(0.0, captured_at - getattr(server, "_session_started", captured_at))
                result = await asyncio.to_thread(encode_preview, frame, points, pose, exercise, elapsed)
                result.update(warning=warning, notice=notice)
                await websocket.send(json.dumps(result))
            except ConnectionClosed:
                raise
            except Exception as exc:
                await websocket.send(json.dumps({"type": "PREVIEW_STATUS", "warning": str(exc)}))
                if camera_failed:
                    with server._camera_lock:
                        if server.camera:
                            server.camera.stop()
                            server.camera = None
                            server._camera_index = None
                await asyncio.sleep(0.5)
            await asyncio.sleep(max(0.0, 1 / PREVIEW_FPS - (time.perf_counter() - started)))
    finally:
        server._preview_watchers -= 1
        if server.controller is None:
            server.stop_camera()


async def _poll_video_stop(server, websocket):
    """비디오 배치 분석 중 중단 명령(CMD_STOP_SESSION)이 들어왔는지 비차단으로 확인한다."""
    if not hasattr(websocket, "recv"):
        return None
    try:
        packet = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.001))
    except asyncio.TimeoutError:
        return None
    if packet.get("type") == "CMD_STOP_SESSION":
        session_id = server.controller.data_manager.session_log.session_id
        if packet.get("session_id") in (None, session_id):
            return packet
    return None


async def analyze_video(server, websocket, data: dict):
    """녹화 영상 파일에서 23개 관절 좌표와 측정 신호를 배치로 추출해 NPZ·CSV로 저장한다."""
    video_path = data.get("video_path") or data.get("input_source")
    mode = data.get("mode", "CALIBRATION")
    batch_size = int(data.get("batch_size", 16))
    if not video_path or not Path(video_path).is_file():
        await websocket.send(
            json.dumps({
                "type": "ERROR",
                "code": "VIDEO_NOT_FOUND",
                "message": f"유효하지 않은 비디오 경로입니다: {video_path}",
            })
        )
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        await websocket.send(
            json.dumps({
                "type": "ERROR",
                "code": "VIDEO_OPEN_FAILED",
                "message": f"비디오 파일을 열 수 없습니다: {video_path}",
            })
        )
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration = 1.0 / fps

    try:
        data_manager = server._create_session(data, video_fps=fps)
    except Exception:
        cap.release()
        raise

    data_manager.session_log.info["source_video"] = str(Path(video_path).resolve())
    if not data_manager.session_log.record_video:
        data_manager.session_log.info["video_mp4"] = str(Path(video_path).resolve())

    await websocket.send(json.dumps({"type": "SESSION_LOG_STARTED", **data_manager.session_log.info}))
    all_extracted_keypoints = []
    batch_frames = []
    decoded_frames = 0
    stop_request = None
    last_progress = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if batch_frames:
                    kpts_batch = await asyncio.to_thread(
                        server.skeleton_engine.extract_keypoints_batch, batch_frames
                    )
                    all_extracted_keypoints.extend(kpts_batch)
                break

            frame = downscale_frame_if_needed(frame, 1280, 720)
            if data_manager.session_log.record_video:
                data_manager.session_log.write_video_frame(frame, decoded_frames * frame_duration)
            decoded_frames += 1
            batch_frames.append(frame)

            if len(batch_frames) >= batch_size:
                kpts_batch = await asyncio.to_thread(
                    server.skeleton_engine.extract_keypoints_batch, batch_frames
                )
                all_extracted_keypoints.extend(kpts_batch)
                batch_frames.clear()

                stop_request = await _poll_video_stop(server, websocket)
                if stop_request:
                    break

                if time.perf_counter() - last_progress >= 0.4:
                    await websocket.send(
                        json.dumps({
                            "type": "VIDEO_PROGRESS",
                            "phase": "EXTRACTING",
                            "processed_frames": decoded_frames,
                            "total_frames": total_frames,
                        })
                    )
                    last_progress = time.perf_counter()
    finally:
        cap.release()

    server.controller.timer_start = 0.0
    for frame_idx, raw_kpts in enumerate(all_extracted_keypoints):
        if stop_request:
            break
        virtual_time = frame_idx * frame_duration
        filtered_kpts = server.filter_engine.update(raw_kpts, timestamp=virtual_time)
        is_occluded = server.controller.check_occlusion(filtered_kpts) if filtered_kpts else True

        normalized = (
            server.controller.measurement_points(filtered_kpts)
            if filtered_kpts and not is_occluded
            else None
        )
        points = {p["id"]: p for p in normalized} if normalized else {}
        val_left = server.controller.motion_engine._compute_side_value("left", points)
        val_right = server.controller.motion_engine._compute_side_value("right", points)

        left_data = {
            "val": val_left,
            "rep_count": 0,
            "total_score": 0.0,
            "score_delta": 0.0,
            "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "EXTRACTED",
            "is_count_updated": False,
        }
        right_data = {
            "val": val_right,
            "rep_count": 0,
            "total_score": 0.0,
            "score_delta": 0.0,
            "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "EXTRACTED",
            "is_count_updated": False,
        }
        server.controller.last_left_data = left_data
        server.controller.last_right_data = right_data
        server._append_to_buffers(
            left_data,
            right_data,
            raw_kpts,
            elapsed_sec=virtual_time,
            frame_idx=frame_idx,
            is_occluded=is_occluded,
            video_frame_idx=frame_idx,
        )
        stop_request = await _poll_video_stop(server, websocket)
        if frame_idx % 30 == 0 or frame_idx == len(all_extracted_keypoints) - 1:
            await websocket.send(
                json.dumps({
                    "type": "VIDEO_PROGRESS",
                    "phase": "ANALYZING",
                    "processed_frames": frame_idx + 1,
                    "total_frames": len(all_extracted_keypoints),
                })
            )

    if stop_request:
        server._record_client_result(stop_request)
    status = "COMPLETED"
    if stop_request:
        status = "COMPLETED" if stop_request.get("reason") == "TARGET_REACHED" else "STOPPED"
    summary_meta = server._save_session_files(status=status)

    payload = {
        "type": "SESSION_FINISHED" if stop_request else "VIDEO_ANALYSIS_COMPLETED",
        "session_id": data_manager.session_log.session_id,
        "request_id": stop_request.get("request_id") if stop_request else None,
        "reason": stop_request.get("reason", "USER_STOP") if stop_request else "END_OF_VIDEO",
        "mode": mode,
        "video_path": video_path,
        "total_frames_analyzed": len(all_extracted_keypoints),
        "source_duration_sec": decoded_frames * frame_duration,
        "summary": summary_meta,
    }
    if stop_request:
        payload["status"] = status
    await websocket.send(json.dumps(payload))
    server.controller = None


class ExerciseSocketServer:
    """운동 측정 및 스트리밍을 총괄하는 WebSocket 서버."""

    def __init__(self, host="127.0.0.1", port=8080, model_loader=None):
        configure_local_and_gopro_no_proxy()
        self.host = host
        self.port = port
        self.model_loader = model_loader or BackgroundModelLoader(self._build_skeleton_engine)
        self.skeleton_engine = None
        self.filter_engine = None
        self.controller = None
        self.camera = None
        self._camera_index = None
        self._camera_url = None
        self._camera_max_width = 1280
        self._camera_max_height = 720
        self._camera_lock = threading.Lock()
        self._preview_watchers = 0
        self._preview_pose = {}
        self._preview_keypoints = []
        self._preview_updated_at = 0.0
        self.output_mode = "LIGHTWEIGHT"
        self.preview_separate = False
        self._last_preview_at = 0.0
        self._session_owner = None
        self._last_client_finish = None
        self._connection_done = {}
        self._active_log = None
        self._session_started = 0.0
        self._live_timing = False
        self._last_capture_id = None
        self._preparation_seconds = DEFAULT_PREPARATION_SECONDS
        self._preparation_active = False
        self._measurement_ready_at = None
        self._last_preparation_remaining = None
        self._fps_timestamps = deque(maxlen=60)

    @staticmethod
    def _build_skeleton_engine(report_phase):
        from core.skeleton_engine import SkeletonEngine
        return SkeletonEngine(yolo_interval=10, on_progress=report_phase)

    def start_camera(self, camera_index=0, max_width=1280, max_height=720, camera_url=None):
        camera_url = (camera_url or "").strip() or None
        with self._camera_lock:
            if (
                self.camera
                and self._camera_index == camera_index
                and self._camera_url == camera_url
                and self._camera_max_width == max_width
                and self._camera_max_height == max_height
            ):
                return
            if self.camera:
                self.camera.stop()
                self.camera = None
                self._camera_index = None
                self._camera_url = None
            self.camera = ThreadedWebcam(
                cam_idx=camera_index,
                max_width=max_width,
                max_height=max_height,
                camera_url=camera_url,
            ).start()
            self._camera_index = camera_index
            self._camera_url = camera_url
            self._camera_max_width = max_width
            self._camera_max_height = max_height

    def stop_camera(self, force=False):
        """카메라를 해제한다. force=False일 때는 재연결 지연을 없애기 위해 연결을 계속 유지한다."""
        if not force:
            with self._camera_lock:
                if self.camera:
                    self.camera.set_orientation_lock(False)
            return
        with self._camera_lock:
            if self.camera:
                self.camera.stop()
                self.camera = None
                self._camera_index = None
                self._camera_url = None

    def _create_session(self, data, video_fps=30.0, live_timing=False):
        mode = data.get("mode", "CALIBRATION")
        exercise_name = data.get("exercise_name", "biceps_curl")
        target_score = get_target_score(data)
        target_reps = 3 if mode == "CALIBRATION" else get_target_reps(data)
        preparation_seconds = get_preparation_seconds(data)
        is_video = bool(data.get("video_path") or data.get("input_source") or data.get("output_mode") == "OFFLINE_VIDEO")
        record_video = bool(data.get("record_video", not is_video))
        filename_stem = data.get("filename_stem")
        if filename_stem is None and is_video and data.get("use_video_stem"):
            filename_stem = Path(data.get("video_path") or data.get("input_source")).stem

        manager = DataManager(
            player_id=data.get("patient_id", data.get("player_id", "0000")),
            patient_name=data.get("patient_name", "Unknown"),
            log_root=data.get("log_root"),
            patient_metadata=data.get("patient_metadata"),
            data_version=data.get("data_version", "v1.0.0(0)"),
            registration=data.get("registration", 1),
        )
        manager.begin_session(
            exercise_name=exercise_name,
            mode=mode,
            session=data.get("session"),
            round_number=data.get("round", 1),
            set_number=data.get("set", 1),
            record_video=record_video,
            video_fps=video_fps,
            target_score=target_score,
            play_session_id=data.get("play_session_id"),
            exercise_record_id=data.get("exercise_record_id"),
            slot=data.get("slot", 1),
            filename_stem=filename_stem,
        )

        self._active_log = manager.session_log
        manager.session_log.info.update(
            target_reps=target_reps,
            patient_name=manager.patient_name,
            auto_stop_on_target=bool(data.get("auto_stop_on_target", False)),
            preparation_seconds=preparation_seconds,
        )
        self._session_started = time.perf_counter()
        self._live_timing = live_timing
        self._last_capture_id = None
        self._measurement_ready_at = None
        exercise_config = config_for_mode(load_exercise_config(exercise_name), mode)

        custom_thresholds = None
        if mode == "MAIN":
            custom_thresholds = manager.get_custom_threshold(exercise_name)
            if custom_thresholds is None:
                if is_video:
                    custom_thresholds = exercise_config.get("default_thresholds", {})
                else:
                    raise ValueError(
                        "이 날짜·동작·라운드의 캘리브레이션 결과가 없습니다. 같은 라운드에서 캘리브레이션을 먼저 완료하세요."
                    )

        effective_reps = 3 if mode == "CALIBRATION" else (target_reps or target_score)
        motion_engine = MotionEngine(
            config=exercise_config,
            custom_thresholds=custom_thresholds,
            mode=mode,
            target_reps=effective_reps,
            target_score=target_score,
        )
        motion_engine.exercise_name = exercise_name
        manager.session_log.info["measurement"] = motion_engine.count_threshold_status()

        self.controller = SessionController(data_manager=manager, motion_engine=motion_engine)
        self.filter_engine = OneEuroKeypointFilter()
        return manager

    def init_live_session(self, data: dict):
        preparation_seconds = get_preparation_seconds(data)
        self._create_session(data, live_timing=True)
        max_w = int(data.get("camera_max_width", 1280))
        max_h = int(data.get("camera_max_height", 720))
        camera_url = (data.get("camera_url") or "").strip() or None
        self.start_camera(int(data.get("camera_index", 0)), max_width=max_w, max_height=max_h, camera_url=camera_url)
        if self.camera:
            self.camera.set_orientation_lock(True)
        self._preparation_seconds = preparation_seconds
        self._preparation_active = preparation_seconds > 0
        self._measurement_ready_at = None
        self._last_preparation_remaining = None
        self._session_started = time.perf_counter()

    async def _measurement_ready(self, websocket, joints_ok: bool, captured_at: float):
        """관절이 정상 인식된 이후부터 준비 시간(기본 3초)을 카운트다운한 뒤 측정을 개시한다."""
        if not self._preparation_active:
            return True

        if not joints_ok:
            # 고프로 대기 화면이거나 필수 관절이 아직 잡히지 않으면 카운트다운을 시작/유지하지 않는다
            self._measurement_ready_at = None
            if self._last_preparation_remaining != "WAITING_JOINTS":
                await websocket.send(
                    json.dumps({
                        "type": "SESSION_PREPARING",
                        "remaining_seconds": None,
                        "message": "카메라 및 관절 인식을 대기하고 있습니다. 화면에 자세를 잡아 주세요.",
                    })
                )
                self._last_preparation_remaining = "WAITING_JOINTS"
            return False

        now = time.perf_counter()
        if self._measurement_ready_at is None:
            self._measurement_ready_at = now + self._preparation_seconds

        remaining = self._measurement_ready_at - now
        if remaining > 0:
            seconds = int(np.ceil(remaining))
            if seconds != self._last_preparation_remaining:
                await websocket.send(
                    json.dumps({
                        "type": "SESSION_PREPARING",
                        "remaining_seconds": seconds,
                        "message": f"관절 인식 완료! ({seconds}초) 후 측정을 시작합니다.",
                    })
                )
                self._last_preparation_remaining = seconds
            return False

        self._preparation_active = False
        self._measurement_ready_at = None
        self._session_started = captured_at
        if self.filter_engine is not None:
            self.filter_engine.reset()
        await websocket.send(
            json.dumps({
                "type": "SESSION_MEASUREMENT_STARTED",
                "message": "관절 인식 및 준비가 끝나 측정을 시작합니다.",
            })
        )
        return True

    def _append_to_buffers(
        self,
        left_data,
        right_data,
        raw_keypoints,
        elapsed_sec=None,
        frame_idx=None,
        is_occluded=False,
        video_frame_idx=-1,
    ):
        log = self.controller.data_manager.session_log
        engine = self.controller.motion_engine
        if elapsed_sec is None:
            elapsed_sec = time.perf_counter() - self._session_started
        if frame_idx is None:
            frame_idx = len(log.timestamps)
        log.append_frame(
            elapsed_sec=elapsed_sec,
            frame_idx=frame_idx,
            raw_keypoints=raw_keypoints,
            left=left_data,
            right=right_data,
            metric_type=engine.metric_type,
            thresholds=engine.get_count_thresholds(),
            threshold_version=engine.threshold_version,
            is_occluded=is_occluded,
            calib_step=self.controller.calib_step if self.controller.mode == "CALIBRATION" else "",
            video_frame_idx=video_frame_idx,
            use_normalization=engine.use_normalization,
        )

    def _save_session_files(self, status="COMPLETED", end_elapsed_sec=None):
        if not self.controller or not self.controller.data_manager:
            return
        if end_elapsed_sec is None and self._live_timing:
            end_elapsed_sec = max(0.0, time.perf_counter() - self._session_started)
        rep_rows = []
        if self.controller.motion_engine:
            rep_rows.extend(self.controller.motion_engine.fsm_left.completed_reps_history)
            rep_rows.extend(self.controller.motion_engine.fsm_right.completed_reps_history)
        summary = self.controller.data_manager.finish_session(
            rep_rows,
            status=status,
            end_elapsed_sec=end_elapsed_sec,
        )
        self._active_log = None
        return summary

    async def _start_session(self, websocket, packet):
        if self._session_owner not in (None, websocket):
            await websocket.send(
                json.dumps({
                    "type": "ERROR",
                    "command": "CMD_SET_SESSION",
                    "request_id": packet.get("request_id"),
                    "code": "SERVER_BUSY",
                    "message": "Another client owns the active measurement session",
                })
            )
            return False
        if self.controller is not None:
            self._save_session_files(status="INTERRUPTED")
            self.stop_camera()
            self.controller = None

        self._session_owner = websocket
        self._last_client_finish = None
        try:
            self.output_mode = resolve_output_mode(packet)
            self.preview_separate = bool(packet.get("preview_separate"))
            if self.output_mode == "OFFLINE_VIDEO":
                self.stop_camera(force=True)
                await analyze_video(self, websocket, packet)
                self._session_owner = None
            else:
                self.init_live_session(packet)
                await websocket.send(
                    json.dumps({
                        "type": "SESSION_LOG_STARTED",
                        **self.controller.data_manager.session_log.info,
                    })
                )
            return True
        except Exception as exc:
            if self._active_log is not None:
                self._active_log.finish([], status="FAILED")
                self._active_log = None
            self.controller = None
            self.stop_camera()
            self._session_owner = None
            await websocket.send(
                json.dumps({
                    "type": "ERROR",
                    "command": "CMD_SET_SESSION",
                    "request_id": packet.get("request_id"),
                    "code": "SESSION_START_FAILED",
                    "message": str(exc),
                })
            )
            return False

    async def _finish_client_session(self, websocket, packet):
        request_id = packet.get("request_id")
        requested_session = packet.get("session_id")
        if self.controller is None:
            if self._last_client_finish and self._last_client_finish[0] is websocket:
                previous = dict(self._last_client_finish[1])
                if requested_session in (None, previous.get("session_id")):
                    previous["request_id"] = request_id
                    await websocket.send(json.dumps(previous))
                    return
            await websocket.send(
                json.dumps({
                    "type": "ERROR",
                    "command": "CMD_STOP_SESSION",
                    "request_id": request_id,
                    "code": "NO_ACTIVE_SESSION",
                    "message": "No active session to stop",
                })
            )
            return

        active_session = self.controller.data_manager.session_log.session_id
        if self._session_owner is not websocket or requested_session not in (None, active_session):
            await websocket.send(
                json.dumps({
                    "type": "ERROR",
                    "command": "CMD_STOP_SESSION",
                    "request_id": request_id,
                    "code": "SESSION_MISMATCH",
                    "message": "Cannot stop a session not owned by this client",
                })
            )
            return

        self._record_client_result(packet)
        reason = packet.get("reason", "USER_STOP")
        status = "COMPLETED" if reason == "TARGET_REACHED" else "STOPPED"
        summary = self._save_session_files(status=status)
        self.stop_camera()
        response = {
            "type": "SESSION_FINISHED",
            "session_id": active_session,
            "request_id": request_id,
            "status": status,
            "reason": reason,
            "summary": summary,
        }
        self.controller = None
        self._session_owner = None
        self._last_client_finish = (websocket, response)
        await websocket.send(json.dumps(response))

    def _record_client_result(self, packet):
        result = packet.get("client_result")
        if not isinstance(result, dict):
            return
        log = self.controller.data_manager.session_log
        log.info["client_result"] = result
        if "target_score" in result:
            log.info["target_score"] = result["target_score"]

    async def reset_session_connection(self, websocket, packet):
        owner = self._session_owner
        if owner is websocket:
            await websocket.send(
                json.dumps({
                    "type": "ERROR",
                    "code": "RESET_FROM_MONITOR_REQUIRED",
                    "message": "새 연결에서 복구하거나 현재 연결에서 CMD_STOP_SESSION을 사용하세요.",
                })
            )
            return
        done = self._connection_done.get(id(owner))
        if owner is not None:
            try:
                await asyncio.wait_for(owner.close(code=4001, reason="Session reset requested"), 5.0)
                if done is not None:
                    await asyncio.wait_for(done.wait(), 15.0)
            except asyncio.TimeoutError:
                await websocket.send(
                    json.dumps({
                        "type": "ERROR",
                        "code": "SESSION_RESET_TIMEOUT",
                        "message": "이전 세션 정리가 아직 끝나지 않았습니다.",
                    })
                )
                return
        await websocket.send(
            json.dumps({
                "type": "SERVER_SESSION_RESET",
                "reset": owner is not None,
                "request_id": packet.get("request_id"),
                "message": "이전 세션 연결 정리를 완료했습니다.",
            })
        )

    async def handle_client(self, websocket):
        connection_done = asyncio.Event()
        self._connection_done[id(websocket)] = connection_done
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
                    await websocket.send(
                        json.dumps({
                            "type": "ERROR",
                            "code": "MODEL_LOAD_FAILED",
                            "fatal": True,
                            "message": f"모델 준비 실패: {status.get('model_error')}",
                        })
                    )
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
                    recv_timeout = 0.001 if (self.controller and self._session_owner is websocket) else 0.01
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=recv_timeout)
                    packet = json.loads(raw_msg)
                    pkt_type = packet.get("type")

                    if pkt_type == "CMD_WATCH_CAMERA":
                        await watch_camera(self, websocket, packet)
                        return

                    if pkt_type == "CMD_SCAN_CAMERAS":
                        diag = await asyncio.to_thread(
                            scan_available_cameras, int(packet.get("max_index", 4))
                        )
                        await websocket.send(json.dumps({"type": "CAMERA_SCAN_RESULT", **diag}))
                        continue

                    if pkt_type == "CMD_DISCONNECT_CAMERA":
                        await asyncio.to_thread(self.stop_camera, True)
                        await websocket.send(
                            json.dumps({
                                "type": "CAMERA_DISCONNECTED",
                                "message": "카메라 연결을 해제했습니다.",
                            })
                        )
                        continue

                    if pkt_type == "CMD_SET_SESSION":
                        pending_session = packet
                        pending_commands.clear()
                        if not self.model_loader.snapshot()["ready"]:
                            await websocket.send(
                                json.dumps({
                                    "type": "SESSION_PENDING",
                                    "message": "모델 준비가 끝나면 요청한 세션을 시작합니다.",
                                })
                            )

                    elif pkt_type == "CMD_STOP_SESSION":
                        if pending_session and self._session_owner in (None, websocket):
                            pending_session = None
                            pending_commands.clear()
                            await websocket.send(
                                json.dumps({
                                    "type": "SESSION_FINISHED",
                                    "status": "STOPPED",
                                    "reason": "USER_STOP",
                                    "request_id": packet.get("request_id"),
                                    "summary": None,
                                })
                            )
                        else:
                            await self._finish_client_session(websocket, packet)

                    elif pkt_type == "CMD_RESET_SESSION":
                        await self.reset_session_connection(websocket, packet)

                    elif pkt_type == "CMD_GET_SERVER_STATUS":
                        await websocket.send(json.dumps(self.model_loader.snapshot()))

                    elif pkt_type == "CMD_STOP_CALIBRATION":
                        if self._session_owner is websocket and self.controller and self.controller.mode == "CALIBRATION":
                            self.controller.calib_step = "FINISHED"

                    elif pkt_type in COUNT_THRESHOLD_COMMANDS:
                        if pending_session:
                            pending_commands.append(packet)
                        elif self.controller is not None and self._session_owner is not websocket:
                            await websocket.send(
                                json.dumps({
                                    "type": "ERROR",
                                    "request_id": packet.get("request_id"),
                                    "command": pkt_type,
                                    "code": "SESSION_MISMATCH",
                                    "message": "Only the active session owner may access count thresholds",
                                })
                            )
                        else:
                            await websocket.send(json.dumps(handle_count_threshold_command(self.controller, packet)))
                except asyncio.TimeoutError:
                    pass

                if not self.controller or not self.camera or self._session_owner is not websocket:
                    await asyncio.sleep(0.05)
                    continue

                grabbed, frame, captured_at, capture_id = self.camera.read_sample()
                if not grabbed or frame is None or capture_id == self._last_capture_id:
                    await asyncio.sleep(0.005)
                    continue
                self._last_capture_id = capture_id

                raw_kpts = await asyncio.to_thread(self.skeleton_engine.extract_keypoints, frame)

                if self._preparation_active:
                    joints_ok = bool(raw_kpts) and not self.controller.check_occlusion(raw_kpts)
                    self._preview_keypoints = raw_kpts or []
                    self._preview_updated_at = time.perf_counter()
                    if not await self._measurement_ready(websocket, joints_ok, captured_at):
                        await asyncio.sleep(0)
                        continue

                elapsed_sec = max(0.0, captured_at - self._session_started)
                video_frame_idx = self.controller.data_manager.session_log.write_video_frame(frame, elapsed_sec)

                now_perf = time.perf_counter()
                self._fps_timestamps.append(now_perf)
                if len(self._fps_timestamps) >= 2:
                    span = self._fps_timestamps[-1] - self._fps_timestamps[0]
                    fps = round((len(self._fps_timestamps) - 1) / span, 1) if span > 0 else 0.0
                else:
                    fps = 0.0

                filtered_kpts = self.filter_engine.update(raw_kpts, timestamp=captured_at)
                is_occluded = self.controller.check_occlusion(filtered_kpts) if filtered_kpts else True

                if self.controller.mode == "CALIBRATION":
                    calib_res = self.controller.process_calibration_frame(
                        filtered_kpts, is_occluded, now=captured_at
                    )
                    self._append_to_buffers(
                        self.controller.last_left_data,
                        self.controller.last_right_data,
                        raw_kpts,
                        elapsed_sec=elapsed_sec,
                        is_occluded=is_occluded,
                        video_frame_idx=video_frame_idx,
                    )
                    pose_packet = {
                        "type": "POSE_UPDATE",
                        "timestamp": round(time.time(), 3),
                        "fps": fps,
                        "is_occluded": is_occluded,
                        "mode": "CALIBRATION",
                        "metric_type": self.controller.motion_engine.metric_type,
                        "metric_unit": self.controller.motion_engine.metric_unit,
                        "display_thresholds": self.controller.motion_engine.get_display_thresholds(),
                        "thresholds": self.controller.motion_engine.get_count_thresholds(),
                        "threshold_version": self.controller.motion_engine.threshold_version,
                        "calib_step": calib_res["calib_step"],
                        "calib_rep_count": calib_res["calib_rep_count"],
                        "calib_reps_left": calib_res["calib_reps_left"],
                        "calib_reps_right": calib_res["calib_reps_right"],
                        "target_calib_reps": calib_res["target_calib_reps"],
                        "calibration": calib_res["calibration"],
                        "left": self.controller.last_left_data,
                        "right": self.controller.last_right_data,
                        "keypoints": raw_kpts if (raw_kpts and not is_occluded) else [],
                    }
                    self._preview_pose = pose_packet
                    self._preview_keypoints = raw_kpts or []
                    self._preview_updated_at = time.perf_counter()
                    await websocket.send(json.dumps(pose_packet))
                    if not self.preview_separate:
                        await send_preview(self, websocket, frame, raw_kpts, pose_packet, elapsed_sec)

                    if calib_res["calib_step"] == "FINISHED":
                        thresholds = self.controller.compute_and_save_thresholds(
                            self.controller.motion_engine.exercise_name
                        )
                        summary_meta = self._save_session_files(
                            status="COMPLETED", end_elapsed_sec=elapsed_sec
                        )
                        await websocket.send(
                            json.dumps({
                                "type": "CALIBRATION_FINISHED",
                                "player_id": self.controller.data_manager.player_id,
                                "thresholds": thresholds,
                                "summary": summary_meta,
                            })
                        )
                        self.controller = None
                        self._session_owner = None
                        self.stop_camera()
                        continue

                elif self.controller.mode == "MAIN":
                    l_data, r_data, _ = self.controller.process_main_frame(
                        filtered_kpts, is_occluded, now=captured_at
                    )
                    self._append_to_buffers(
                        l_data,
                        r_data,
                        raw_kpts,
                        elapsed_sec=elapsed_sec,
                        is_occluded=is_occluded,
                        video_frame_idx=video_frame_idx,
                    )
                    pose_packet = {
                        "type": "POSE_UPDATE",
                        "timestamp": round(time.time(), 3),
                        "fps": fps,
                        "is_occluded": is_occluded,
                        "mode": self.controller.mode,
                        "metric_type": self.controller.motion_engine.metric_type,
                        "metric_unit": self.controller.motion_engine.metric_unit,
                        "display_thresholds": self.controller.motion_engine.get_display_thresholds(),
                        "thresholds": self.controller.motion_engine.get_count_thresholds(),
                        "threshold_version": self.controller.motion_engine.threshold_version,
                        "left": l_data,
                        "right": r_data,
                        "keypoints": raw_kpts if (raw_kpts and not is_occluded) else [],
                    }
                    self._preview_pose = pose_packet
                    self._preview_keypoints = raw_kpts or []
                    self._preview_updated_at = time.perf_counter()
                    await websocket.send(json.dumps(pose_packet))
                    if not self.preview_separate:
                        await send_preview(self, websocket, frame, raw_kpts, pose_packet, elapsed_sec)

                    for side_str, s_data in [("left", l_data), ("right", r_data)]:
                        if s_data.get("is_count_updated", False):
                            await websocket.send(
                                json.dumps({
                                    "type": "REP_COMPLETED",
                                    "mode": self.controller.mode,
                                    "metric_type": self.controller.motion_engine.metric_type,
                                    "side": side_str,
                                    "rep_id": s_data["rep_count"],
                                    "quality": s_data["quality"],
                                    "val": s_data["val"],
                                    "score": s_data["score"],
                                    "score_delta": s_data["score_delta"],
                                    "total_score": s_data["total_score"],
                                    "target_score": s_data["target_score"],
                                    "threshold_version": self.controller.motion_engine.threshold_version,
                                })
                            )
                await asyncio.sleep(0)

        except ConnectionClosed:
            end_status = "DISCONNECTED"
        except Exception as e:
            end_status = "ERROR"
            traceback.print_exc()
            try:
                await websocket.send(
                    json.dumps({
                        "type": "ERROR",
                        "code": "SESSION_RUNTIME_ERROR",
                        "message": str(e),
                    })
                )
            except Exception:
                pass
        finally:
            try:
                if self._session_owner is websocket:
                    if self.controller:
                        self._save_session_files(status=end_status)
                        self.controller = None
                    elif self._active_log is not None:
                        self._active_log.finish([], status=end_status)
                        self._active_log = None
                    self.stop_camera()
                    self._session_owner = None
            finally:
                connection_done.set()
                self._connection_done.pop(id(websocket), None)

    async def start_server(self):
        self.model_loader.start()
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            max_size=16 * 1024 * 1024,
            compression=None,
        ):
            print(f"[SERVER] Listening on ws://{self.host}:{self.port} (COCO-23 ONNX)", flush=True)
            try:
                await asyncio.Future()
            finally:
                self.stop_camera(force=True)
