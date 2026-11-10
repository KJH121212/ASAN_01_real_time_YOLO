"""실시간 수치 서버. 영상 실험 기능은 experimental_modes에서 필요할 때만 불러옵니다."""

import os
os.environ["MMCV_WITH_OPS"] = "0"

import asyncio
import json
import sys
import time
import threading
_BOOT_STARTED = time.perf_counter()
import traceback
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
from core.motion_engine import MotionEngine, config_for_mode
from core.model_loader import BackgroundModelLoader
from core.session_controller import SessionController
from utils.filters import OneEuroKeypointFilter
from network.camera import ThreadedWebcam, configure_local_and_gopro_no_proxy, scan_available_cameras
from network.session_commands import (
    COUNT_THRESHOLD_COMMANDS,
    handle_count_threshold_command,
    get_preparation_seconds,
    get_target_score,
    get_target_reps,
    resolve_output_mode,
)

configure_local_and_gopro_no_proxy()

class ExerciseSocketServer:
    def __init__(self, host="127.0.0.1", port=None, engine_factory=None):
        self.host = host
        self.port = int(os.environ.get("EXERCISE_SERVER_PORT", 8080)) if port is None else port

        # AI 모델 비동기 로딩 관리자 (서버 기동 속도 최적화)
        self.skeleton_engine = None
        self.model_loader = BackgroundModelLoader(engine_factory or self._create_skeleton_engine)
        self.bound_port = None

        # 세션 독점 관리 및 상태 제어
        self._session_owner = None
        self._connection_done = {}
        self._active_log = None
        self._live_timing = False
        self._last_capture_id = None
        self._measurement_ready_at = None
        self._last_preparation_remaining = None

        self.camera = None
        self.controller = None
        self.filter_engine = None
        self.output_mode = "LIGHTWEIGHT"
        self._camera_lock = threading.Lock()
        self._camera_index = None
        self._preview_watchers = 0
        self._preview_pose = {}
        self._preview_keypoints = []
        self._preview_updated_at = 0.
        self._separate_preview = False

    @staticmethod
    def _create_skeleton_engine(report_phase):
        """백그라운드 스레드에서 추론 엔진 로드"""
        from core.skeleton_engine import SkeletonEngine
        return SkeletonEngine(yolo_interval=5, on_progress=report_phase)

    async def _start_session(self, websocket, packet):
        """세션 시작 검증 및 비디오/웹캠 파이프라인 분기"""
        if packet.get('mode', 'CALIBRATION') not in {'CALIBRATION', 'MAIN'}:
            await websocket.send(json.dumps({
                'type': 'ERROR', 'code': 'INVALID_MODE',
                'message': '측정 모드는 CALIBRATION 또는 MAIN만 사용할 수 있습니다.',
            }))
            return False
        try:
            output_mode = resolve_output_mode(packet)
        except ValueError as exc:
            await websocket.send(json.dumps({
                "type": "ERROR",
                "code": "INVALID_OUTPUT_MODE",
                "message": str(exc),
            }))
            return False

        # 세션 중복 연결 방지
        if self._session_owner is not None and self._session_owner is not websocket:
            await websocket.send(json.dumps({
                "type": "ERROR",
                "code": "SERVER_BUSY",
                "message": "다른 연결에서 세션을 사용 중입니다."
            }))
            return False

        self._session_owner = websocket
        self.output_mode = output_mode
        self._separate_preview = bool(packet.get('preview_separate', False))
        self._preview_pose, self._preview_keypoints = {}, []
        self._last_preview_at = 0.0
        self._last_client_finish = None
        video_path = packet.get("video_path") or packet.get("input_source")

        try:
            # 기존 세션 정리
            if self.controller:
                self._save_session_files(status="STOPPED")
                self.stop_camera()
                self.controller = None

            # [모드 3] 비디오 배치 분석
            if video_path:
                if not os.path.isfile(str(video_path)):
                    raise ValueError(f"비디오 파일을 찾을 수 없습니다: {video_path}")
                from network.experimental_modes import analyze_video
                await analyze_video(self, websocket, packet)
            # [모드 1·2] 웹캠 실시간 세션
            else:
                await websocket.send(json.dumps({
                    "type": "SESSION_PREPARING",
                    "message": "카메라 연결 중입니다. 휴대폰 웹캠 앱의 연결 상태를 확인해 주세요."
                }))
                await asyncio.to_thread(self.init_live_session, packet)
                await websocket.send(json.dumps({
                    "type": "SESSION_LOG_STARTED",
                    **self.controller.data_manager.session_log.info
                }))
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
                "type": "ERROR",
                "code": "VIDEO_ANALYSIS_FAILED" if video_path else "SESSION_START_FAILED",
                "message": f"세션 시작/분석 실패: {exc}",
            }))

        return True

    def start_camera(self, cam_idx=0, max_width=1280, max_height=720):
        with self._camera_lock:
            if self.camera and self._camera_index == cam_idx:
                if max_width and max_height:
                    self.camera.max_width = int(max_width)
                    self.camera.max_height = int(max_height)
                    self.camera._scaled_frame_id = -1
                return
            if self.camera:
                self.camera.stop()
            self.camera = ThreadedWebcam(cam_idx, max_width=max_width, max_height=max_height).start()
            self._camera_index = cam_idx

    def stop_camera(self):
        with self._camera_lock:
            if self.camera and not self._preview_watchers:
                self.camera.stop()
                self.camera = None
                self._camera_index = None

    def _create_session(self, data, video_fps=30.0, live_timing=False):
        """실시간·파일 분석이 함께 사용하는 로그와 측정 엔진을 준비합니다."""
        mode = data.get("mode", "CALIBRATION")
        if "cal" in data and (not isinstance(data["cal"], bool) or data["cal"] != (mode == "CALIBRATION")):
            raise ValueError("cal must match mode == CALIBRATION")
        exercise_name = data.get("exercise_name", "biceps_curl")
        target_score = get_target_score(data)
        target_reps = get_target_reps(data)
        preparation_seconds = get_preparation_seconds(data)
        if mode == 'CALIBRATION':
            target_reps = 3

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
            record_video=True,
            video_fps=video_fps,
            target_score=target_score,
            play_session_id=data.get("play_session_id"),
            exercise_record_id=data.get("exercise_record_id"),
            slot=data.get("slot", 1),
        )

        self._active_log = manager.session_log
        manager.session_log.info.update(target_reps=target_reps, patient_name=manager.patient_name,
            auto_stop_on_target=bool(data.get('auto_stop_on_target', False)),
            preparation_seconds=preparation_seconds)
        self._session_started = time.perf_counter()
        self._live_timing = live_timing
        self._last_capture_id = None
        self._measurement_ready_at = None
        exercise_config = config_for_mode(load_exercise_config(exercise_name), mode)

        # 맞춤형 임계값 로드
        custom_thresholds = None
        is_video = bool(data.get("video_path") or data.get("input_source") or data.get("output_mode") == "OFFLINE_VIDEO")
        if mode == "MAIN":
            custom_thresholds = manager.get_custom_threshold(exercise_name)
            if custom_thresholds is None:
                if is_video:
                    custom_thresholds = exercise_config.get("default_thresholds", {})
                else:
                    raise ValueError('이 날짜·동작·라운드의 캘리브레이션 결과가 없습니다. 같은 라운드에서 캘리브레이션을 먼저 완료하세요.')
            if custom_thresholds and not is_video:
                if any(custom_thresholds.get(side, {}).get('metric_type') != exercise_config['eval_config']['metric_type']
                       for side in ('left', 'right') if side in custom_thresholds):
                    raise ValueError('캘리브레이션 이후 측정 방식이 바뀌었습니다. 같은 라운드에서 캘리브레이션을 다시 완료하세요.')
        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}

        effective_reps = 3 if mode == "CALIBRATION" else (target_reps or target_score)

        motion_engine = MotionEngine(
            config=exercise_config,
            custom_thresholds=custom_thresholds,
            mode=mode,
            target_reps=effective_reps,
            target_score=target_score,
        )
        motion_engine.exercise_name = exercise_name
        manager.session_log.info['measurement'] = motion_engine.count_threshold_status()

        self.controller = SessionController(data_manager=manager, motion_engine=motion_engine)
        self.filter_engine = OneEuroKeypointFilter()
        return manager

    def init_live_session(self, data: dict):
        """실시간 웹캠 측정 파이프라인 초기화"""
        preparation_seconds = get_preparation_seconds(data)
        self._create_session(data, live_timing=True)
        max_w = int(data.get("camera_max_width", 1280))
        max_h = int(data.get("camera_max_height", 720))
        self.start_camera(int(data.get("camera_index", 0)), max_width=max_w, max_height=max_h)

        # 준비 시간 설정
        self._session_started = time.perf_counter() + preparation_seconds
        self._measurement_ready_at = self._session_started if preparation_seconds else None
        self._last_preparation_remaining = None

    async def _measurement_ready(self, websocket):
        """준비 시간 카운트다운 게이트 제어"""
        if self._measurement_ready_at is None:
            return True

        remaining = self._measurement_ready_at - time.perf_counter()
        if remaining > 0:
            seconds = int(np.ceil(remaining))
            if seconds != self._last_preparation_remaining:
                await websocket.send(json.dumps({
                    "type": "SESSION_PREPARING",
                    "remaining_seconds": seconds,
                    "message": f"기본 자세를 취해 주세요. {seconds}초 후 측정을 시작합니다.",
                }))
                self._last_preparation_remaining = seconds
            return False

        self._measurement_ready_at = None
        await websocket.send(json.dumps({
            "type": "SESSION_MEASUREMENT_STARTED",
            "message": "준비 시간이 끝나 측정을 시작합니다.",
        }))
        return True

    def _append_to_buffers(
        self, left_data, right_data, raw_keypoints, elapsed_sec=None, frame_idx=None, is_occluded=False, video_frame_idx=-1
    ):
        """프레임별 원본 좌표 및 FSM 판정 로그 적재"""
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
            elapsed_sec,
            frame_idx,
            raw_keypoints,
            left_data,
            right_data,
            metric_type=engine.metric_type,
            use_normalization=engine.use_normalization,
            thresholds=engine.get_count_thresholds(),
            threshold_version=engine.threshold_version,
            is_occluded=is_occluded,
            calib_step=getattr(self.controller, "calib_step", ""),
            video_frame_idx=video_frame_idx,
        )

    def _save_session_files(self, status="COMPLETED"):
        """세션 데이터(NPZ/CSV/MP4) 저장 마무리"""
        if not self.controller or not self.controller.data_manager:
            return None

        engine = self.controller.motion_engine
        self._preview_pose = {**self._preview_pose, 'left': self.controller.last_left_data,
                              'right': self.controller.last_right_data}
        rows = engine.fsm_left.completed_reps_history + engine.fsm_right.completed_reps_history
        end_elapsed = max(0.0, time.perf_counter() - self._session_started) if self._live_timing else None
        summary = self.controller.data_manager.finish_session(rows, status=status, end_elapsed_sec=end_elapsed)
        self._active_log = None
        return summary

    async def _send_rep_events(self, websocket, left, right, timestamp=None):
        """반복(Repetition) 카운트 증가 시 단일 이벤트 패킷 발송"""
        log = self.controller.data_manager.session_log
        engine = self.controller.motion_engine

        for side, data in (("left", left), ("right", right)):
            if not data.get("is_count_updated"):
                continue

            await websocket.send(json.dumps({
                "type": "REP_COMPLETED",
                "session_id": log.session_id,
                "play_session_id": log.play_session_id,
                "exercise_record_id": log.exercise_record_id,
                "sensor_file_id": log.sensor_file_id,
                "exercise_name": engine.exercise_name,
                "mode": engine.mode,
                "metric_type": engine.metric_type,
                "side": side,
                "rep_id": data["rep_count"],
                "quality": data["quality"],
                "score": data.get('score', 0),
                "score_delta": data.get('score_delta', 0.),
                "total_score": data.get('total_score', 0.),
                "threshold_version": engine.threshold_version,
                "timestamp": timestamp,
                "val": data.get("val"),
            }))

    async def _finish_client_session(self, websocket, packet):
        """클라이언트 요청(CMD_STOP_SESSION)에 따른 세션 종료 처리"""
        request_id = packet.get("request_id")

        def error(code, message):
            return {
                "type": "ERROR",
                "command": "CMD_STOP_SESSION",
                "code": code,
                "request_id": request_id,
                "message": message,
            }

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

        response = {
            "type": "SESSION_FINISHED",
            "session_id": session_id,
            "request_id": request_id,
            "reason": packet.get("reason", "USER_STOP"),
            "status": status,
            "summary": summary,
        }

        self.stop_camera()
        self.controller = None
        self._session_owner = None
        self._last_client_finish = (websocket, response)
        await websocket.send(json.dumps(response))

    def _record_client_result(self, packet):
        """클라이언트 최종 결과(점수 등) 로그 메타데이터 추가"""
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


    # 메인 웹소켓 루프 (모드 1·2 실시간 처리)
    async def reset_session_connection(self, websocket, packet):
        """이전 점유 세션 강제 리셋 복구 명령 처리"""
        owner = self._session_owner
        if owner is websocket:
            await websocket.send(json.dumps({
                "type": "ERROR",
                "code": "RESET_FROM_MONITOR_REQUIRED",
                "message": "새 연결에서 복구하거나 현재 연결에서 CMD_STOP_SESSION을 사용하세요.",
            }))
            return

        done = self._connection_done.get(id(owner))
        if owner is not None:
            try:
                await asyncio.wait_for(owner.close(code=4001, reason="Session reset requested"), 5.0)
                if done is not None:
                    await asyncio.wait_for(done.wait(), 15.0)
                if self._session_owner is owner:
                    raise asyncio.TimeoutError()
            except asyncio.TimeoutError:
                await websocket.send(json.dumps({
                    "type": "ERROR",
                    "code": "SESSION_RESET_TIMEOUT",
                    "message": "이전 세션 정리가 아직 끝나지 않았습니다. 잠시 후 다시 복구하세요.",
                }))
                return

        await websocket.send(json.dumps({
            "type": "SERVER_SESSION_RESET",
            "reset": owner is not None,
            "request_id": packet.get("request_id"),
            "message": "이전 세션 연결 정리를 완료했습니다.",
        }))

    async def handle_client(self, websocket):
        """클라이언트별 WebSocket 제어 및 실시간 측정 루프"""
        connection_done = asyncio.Event()
        self._connection_done[id(websocket)] = connection_done
        client_address = websocket.remote_address
        print(f"[DEBUG][NETWORK] Client connected: {client_address}")

        prev_time = time.time()
        pending_session = None
        pending_commands = []
        last_model_status = None
        end_status = "STOPPED"
        self.model_loader.start()

        try:
            while True:
                # 1. 모델 준비 상태 브로드캐스트
                status = self.model_loader.snapshot()
                status_key = (status["model_state"], status["phase"], status.get("model_error"))
                if status_key != last_model_status:
                    await websocket.send(json.dumps(status))
                    last_model_status = status_key

                if pending_session and status["model_state"] == "FAILED":
                    await websocket.send(json.dumps({
                        "type": "ERROR",
                        "code": "MODEL_LOAD_FAILED",
                        "fatal": True,
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

                # 2. 클라이언트 명령 비차단 수신
                try:
                    raw_msg = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    packet = json.loads(raw_msg)
                    pkt_type = packet.get("type")

                    if pkt_type == 'CMD_WATCH_CAMERA':
                        from network.live_preview import watch_camera
                        await watch_camera(self, websocket, packet)
                        return

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
                            await websocket.send(json.dumps({
                                "type": "SESSION_FINISHED",
                                "status": "STOPPED",
                                "reason": "USER_STOP",
                                "request_id": packet.get("request_id"),
                                "summary": None,
                            }))
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
                            if len(pending_commands) < 16:
                                pending_commands.append(packet)
                        elif self._session_owner is not None and self._session_owner is not websocket:
                            await websocket.send(json.dumps({
                                "type": "ERROR",
                                "code": "SERVER_BUSY",
                                "command": pkt_type,
                                "request_id": packet.get("request_id"),
                                "message": "다른 연결에서 세션을 사용 중입니다.",
                            }))
                        else:
                            response = handle_count_threshold_command(self.controller, packet)
                            await websocket.send(json.dumps(response))

                except asyncio.TimeoutError:
                    pass

                # 3. 실시간 웹캠 비전 추론 (VISUAL / LIGHTWEIGHT)
                if self._session_owner is websocket and self.controller and self.camera:
                    if not await self._measurement_ready(websocket):
                        await asyncio.sleep(0.01)
                        continue

                    ret, frame, sample_time, capture_id = self.camera.read_sample()

                    if (
                        ret
                        and frame is not None
                        and capture_id != self._last_capture_id
                        and sample_time >= self._session_started
                    ):
                        self._last_capture_id = capture_id
                        log_elapsed = sample_time - self._session_started

                        # 로컬 비디오 프레임 백그라운드 기록
                        video_index = self.controller.data_manager.session_log.write_video_frame(frame, log_elapsed)

                        curr_time = time.time()
                        time_delta = curr_time - prev_time
                        fps = round(1.0 / time_delta, 1) if time_delta > 0 else 30.0
                        prev_time = curr_time

                        # 관절 추출 및 원유로(1€) 필터 적용
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

                        # --- CALIBRATION 진행 (양측 모두 3회 완수) ---
                        if current_mode == "CALIBRATION":
                            cal_res = self.controller.process_calibration_frame(filtered_kpts, is_occluded, now=sample_time)
                            calib_rep_count = cal_res["calib_rep_count"]
                            calib_reps_left = cal_res["calib_reps_left"]
                            calib_reps_right = cal_res["calib_reps_right"]

                            left_data, right_data = self.controller.last_left_data, self.controller.last_right_data
                            self._append_to_buffers(
                                left_data,
                                right_data,
                                raw_kpts,
                                elapsed_sec=log_elapsed,
                                is_occluded=is_occluded,
                                video_frame_idx=video_index,
                            )

                            if cal_res["calib_step"] == "FINISHED":
                                ex_name = getattr(self.controller.motion_engine, "exercise_name", "biceps_curl")
                                custom_th = self.controller.compute_and_save_thresholds(ex_name)
                                calib_summary = self._save_session_files()

                                await websocket.send(json.dumps({
                                    "type": "CALIBRATION_FINISHED",
                                    "player_id": self.controller.data_manager.player_id,
                                    "thresholds": custom_th,
                                    "summary": calib_summary,
                                }))
                                self.controller = None
                                self.stop_camera()
                                self._session_owner = None
                                continue

                        # --- 본 측정 진행 ---
                        elif current_mode == "MAIN":
                            left_data, right_data, is_finished = self.controller.process_main_frame(
                                filtered_kpts, is_occluded, now=sample_time
                            )
                            self._append_to_buffers(
                                left_data,
                                right_data,
                                raw_kpts,
                                elapsed_sec=log_elapsed,
                                is_occluded=is_occluded,
                                video_frame_idx=video_index,
                            )
                            await self._send_rep_events(websocket, left_data, right_data, log_elapsed)

                        # 두 실시간 모드에 공통 수치를 보내고 VISUAL에만 영상을 추가합니다.
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
                        self._preview_pose = pose_payload
                        # 화면의 관절은 원본 검출 좌표, 측정값·카운트는 필터 적용 결과를 사용합니다.
                        self._preview_keypoints = raw_kpts or []
                        self._preview_updated_at = time.perf_counter()
                        await websocket.send(json.dumps(pose_payload))
                        if self.output_mode == 'VISUAL' and not self._separate_preview:
                            from network.experimental_modes import send_preview
                            await send_preview(self, websocket, frame, raw_kpts, pose_payload, log_elapsed)

                await asyncio.sleep(0.001)

        except ConnectionClosed:
            print(f"[DEBUG][NETWORK] Client disconnected: {client_address}")
        except Exception as e:
            end_status = "FAILED"
            print(f"[ERROR][SERVER] Exception: {e}")
            traceback.print_exc()
            try:
                await websocket.send(json.dumps({
                    "type": "ERROR",
                    "code": "SESSION_PROCESSING_FAILED",
                    "message": str(e),
                }))
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

            self._connection_done.pop(id(websocket), None)
            connection_done.set()

    async def run(self):
        """소켓 서버 기동 및 백그라운드 리스너 유지"""
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
