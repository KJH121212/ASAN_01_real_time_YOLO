"""Persistent WebSocket transport for a Streamlit session (no Streamlit calls).

Widgets rerun the UI script. Keeping networking in a worker lets those reruns
change ROM through the original connection without restarting measurement.
"""

import asyncio
import copy
import json
import queue
import threading
import time
import uuid

import websockets

from network.client_progress import ClientProgress


FINISH_PACKETS = {"SESSION_FINISHED", "CALIBRATION_FINISHED", "VIDEO_ANALYSIS_COMPLETED"}


class StreamlitSessionClient:
    def __init__(self, uri, config, recorder=None):
        self.uri = uri
        self.config = copy.deepcopy(config)
        self.recorder = recorder
        self.session_id = uuid.uuid4().hex
        self._commands = queue.Queue()
        self._events = queue.Queue()
        self._stop = threading.Event()
        self._running = threading.Event()
        self._lock = threading.Lock()
        self._thread = None
        self._pose = None
        self._thresholds = None
        self._server_status = None
        self._log_info = None
        self._server_status_at = None
        self._status = "IDLE"
        self._packet_count = 0
        self._progress = ClientProgress(self.config)
        self._finish_request = None
        self._finish_deadline = None

    @property
    def running(self):
        return self._running.is_set()

    @property
    def can_edit_thresholds(self):
        return (self.running and not self._stop.is_set() and self._finish_request is None
                and (self._server_status is None or self._server_status.get("ready", False))
                and not self.config.get("video_path")
                and self.config.get("mode") in {"MAIN", "TEST"})

    def start(self):
        if self._thread is not None:
            raise RuntimeError("Create a new client for each session")
        self._running.set()
        self._status = "CONNECTING"
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self, timeout=6.0):
        if self.running and not self._stop.is_set():
            self.request_finish("USER_STOP")
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                with self._lock:
                    self._status = "ERROR"
                self._events.put({"type": "CLIENT_ERROR",
                    "message": "Session stop timed out; server log saving is unconfirmed"})
                self._stop.set()
                self._thread.join(timeout=1.0)

    def request_finish(self, reason="TARGET_REACHED"):
        with self._lock:
            if not self.running or self._finish_request is not None:
                return
            self._finish_request = uuid.uuid4().hex
            self._finish_deadline = time.monotonic() + 5.0
            self._status = "STOPPING"
            session_id = (self._log_info or {}).get("session_id")
        packet = {"type": "CMD_STOP_SESSION", "reason": reason,
                  "request_id": self._finish_request}
        if self._progress.enabled:
            with self._lock:
                packet["client_result"] = self._progress.snapshot()
        if session_id:
            packet["session_id"] = session_id
        self._commands.put(json.dumps(packet))

    def send(self, packet):
        if not self.running or self._stop.is_set():
            raise RuntimeError("No active connection")
        # Copy and validate before enqueueing so UI edits cannot change a packet.
        encoded = json.dumps(packet, allow_nan=False)
        self._commands.put(encoded)

    def threshold_command(self, thresholds=None):
        packet = {
            "type": "CMD_GET_COUNT_THRESHOLD" if thresholds is None else "CMD_UPDATE_COUNT_THRESHOLD",
            "exercise_name": self.config["exercise_name"],
            "request_id": uuid.uuid4().hex,
        }
        if thresholds is not None:
            if not self.can_edit_thresholds:
                raise RuntimeError("ROM updates require a live MAIN/TEST session")
            packet["thresholds"] = thresholds
        self.send(packet)
        return packet["request_id"]

    def snapshot(self):
        with self._lock:
            result = copy.deepcopy({
                "pose": self._pose,
                "threshold_status": self._thresholds,
                "server_status": self._server_status,
                "log_info": self._log_info,
                "status": self._status,
                "packet_count": self._packet_count,
                "progress": self._progress.snapshot(),
            })
            if result["server_status"] and result["server_status"].get("model_state") == "LOADING":
                result["server_status"]["elapsed_seconds"] += time.monotonic() - self._server_status_at
            if result["pose"] and self.config.get("mode") in {"MAIN", "TEST"}:
                for side in ("left", "right"):
                    result["pose"].setdefault(side, {})["total_score"] = result["progress"]["scores"][side]
            return result

    def drain_events(self):
        events = []
        while True:
            try:
                events.append(self._events.get_nowait())
            except queue.Empty:
                return events

    def _worker(self):
        try:
            asyncio.run(self._run())
        except Exception as exc:
            with self._lock:
                self._status = "ERROR"
            self._events.put({"type": "CLIENT_ERROR", "message": str(exc)})
        finally:
            if self.recorder:
                self.recorder.stop()
            with self._lock:
                if self._status not in {"FINISHED", "ERROR"}:
                    self._status = "STOPPED"
            self._running.clear()

    async def _run(self):
        async with websockets.connect(self.uri, open_timeout=5, close_timeout=1) as ws:
            if self._stop.is_set():
                return
            if self.recorder:
                self.recorder.start()
            # Goal/point rules belong to this client. Do not send them to the server.
            start_config = {key: value for key, value in self.config.items()
                            if key not in {"target_score", "target_reps", "finish_delay_seconds", "quality_points"}}
            if self.config.get("mode") == "CALIBRATION":
                start_config["target_reps"] = 3
            await ws.send(json.dumps({"type": "CMD_SET_SESSION", **start_config}))
            if not self.config.get("video_path"):
                await ws.send(json.dumps({
                    "type": "CMD_GET_COUNT_THRESHOLD",
                    "exercise_name": self.config["exercise_name"],
                    "request_id": uuid.uuid4().hex,
                }))
            with self._lock:
                self._status = "RUNNING"

            while not self._stop.is_set():
                if self._progress.finish_due(time.monotonic()) and self._finish_request is None:
                    self.request_finish("TARGET_REACHED")
                if self._finish_deadline is not None and time.monotonic() >= self._finish_deadline:
                    raise TimeoutError("Server did not acknowledge CMD_STOP_SESSION; log saving is unconfirmed")
                for _ in range(16):
                    try:
                        command = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    await ws.send(command)
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=0.05)
                except asyncio.TimeoutError:
                    continue
                packet = json.loads(message)
                packet_type = packet.get("type")
                with self._lock:
                    self._packet_count += 1
                    if packet_type == "REP_COMPLETED":
                        self._progress.accept(packet, time.monotonic())
                    if packet_type == "POSE_UPDATE":
                        self._pose = packet
                        self._status = "RUNNING"
                    elif packet_type == "SERVER_STATUS":
                        self._server_status = packet
                        self._server_status_at = time.monotonic()
                        if packet.get("model_state") == "LOADING":
                            self._status = "PREPARING"
                    elif packet_type == "SESSION_PENDING":
                        self._status = "PREPARING"
                    elif packet_type == "SESSION_PREPARING":
                        self._status = "PREPARING"
                    elif packet_type == "SESSION_MEASUREMENT_STARTED":
                        self._status = "RUNNING"
                    elif packet_type == "VIDEO_STATUS":
                        self._status = "RUNNING"
                    elif packet_type == "SESSION_LOG_STARTED":
                        self._log_info = packet
                        self._progress.bind_session(packet.get("session_id"))
                    elif packet_type in FINISH_PACKETS and (packet.get("summary") or {}).get("directory"):
                        self._log_info = packet["summary"]
                    if packet_type in {"POSE_UPDATE", "COUNT_THRESHOLD_STATUS", "COUNT_THRESHOLD_UPDATED"}:
                        if "thresholds" in packet:
                            self._thresholds = {
                                key: packet.get(key) for key in (
                                    "exercise_name", "metric_type", "motion_direction",
                                    "threshold_version", "thresholds",
                                )
                            }
                if self._finish_request is not None and packet_type not in FINISH_PACKETS:
                    with self._lock:
                        self._status = "STOPPING"
                if packet_type == "POSE_UPDATE":
                    if self.recorder:
                        self.recorder.record_pose(packet)
                else:
                    self._events.put(packet)
                if packet_type in FINISH_PACKETS:
                    with self._lock:
                        self._status = "FINISHED"
                    break
                if packet_type == "ERROR" and (packet.get("fatal") or not packet.get("command")):
                    with self._lock:
                        self._status = "ERROR"
                    break
