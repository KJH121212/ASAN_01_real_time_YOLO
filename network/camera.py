"""720p/1080p 웹캠 및 GoPro(가상 웹캠 / USB 이더넷 직접 스트림) 연결, 프록시 우회 및 고해상도 자동 다운스케일 처리."""

import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

# OpenCV C++ 백엔드 탐색 경고([WARN:... VIDEOIO(DSHOW/MSMF)]) 콘솔 출력 억제
os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")

import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:
    pass


GOPRO_NO_PROXY_HOSTS = (
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "::1",
    "172.16.0.0/12",
    "172.20.0.0/14",
    "172.24.0.0/14",
    "172.28.0.0/15",
    "*.local",
)

# 실시간 추론·전송 속도 저하 방지를 위한 기본 최대 캡처 해상도 (720p)
DEFAULT_MAX_CAMERA_WIDTH = 1280
DEFAULT_MAX_CAMERA_HEIGHT = 720

# 열기만 해도 콘솔에 'error = 0'을 출력하거나 검은 대기 화면만 내보내는 가상 카메라 이름 키워드
_DUMMY_VIRTUAL_CAM_KEYWORDS = ("omen cam", "obs virtual", "xsplit", "manycam", "snap camera")


def configure_local_and_gopro_no_proxy() -> None:
    """로컬 WebSocket(127.0.0.1)과 GoPro USB 이더넷(172.2X.1XX.51)이 시스템 프록시를 타지 않도록 설정."""
    for env_key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_key, "")
        items = [x.strip() for x in existing.split(",") if x.strip()]
        for host in GOPRO_NO_PROXY_HOSTS:
            if host not in items:
                items.append(host)
        os.environ[env_key] = ",".join(items)

    # OpenCV FFmpeg 백엔드가 환경변수 http_proxy를 참조해 로컬 UDP/HTTP 스트림을 차단하는 현상 방지
    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "overrun_nonfatal;1|fifo_size;50000000"


configure_local_and_gopro_no_proxy()


def _find_ffmpeg_exe() -> str | None:
    """시스템 PATH 또는 내장된 imageio_ffmpeg 바이너리 경로 반환."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    binaries = Path(__file__).resolve().parent.parent / "analysis/tools/video/imageio_ffmpeg/binaries"
    for path in sorted(binaries.glob("ffmpeg*")):
        if path.is_file() and (path.suffix.lower() == ".exe" if os.name == "nt" else os.access(path, os.X_OK)):
            return str(path)
    return None


def list_dshow_video_devices() -> list[dict]:
    """FFmpeg dshow 목록 조회를 통해 카메라를 직접 열지 않고도 0.05초 만에 정확한 DirectShow 장치 이름·순서를 가져옵니다."""
    ffmpeg_exe = _find_ffmpeg_exe()
    if not ffmpeg_exe:
        return []
    try:
        proc = subprocess.run(
            [ffmpeg_exe, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=2.5,
        )
        output = proc.stderr or ""
    except Exception:
        return []

    devices = []
    # 예: [dshow @ 0000...] "GoPro Webcam" (video) 또는 "HP True Vision FHD Camera" (none)
    pattern = re.compile(r'\[dshow\s+@\s+[0-9a-fA-F]+\]\s+"([^"]+)"\s+\((video|none)\)')
    for match in pattern.finditer(output):
        name, kind = match.group(1), match.group(2)
        idx = len(devices)
        lower_name = name.lower()
        is_gopro = "gopro" in lower_name
        is_dummy_virtual = any(kw in lower_name for kw in _DUMMY_VIRTUAL_CAM_KEYWORDS)
        devices.append({
            "index": idx,
            "name": name,
            "kind": kind,
            "is_gopro": is_gopro,
            "is_dummy_virtual": is_dummy_virtual,
        })
    return devices


def find_preferred_camera_index(default_idx: int = 2) -> int:
    """DirectShow 장치 목록에서 'GoPro Webcam'이 있으면 해당 번호를, 없으면 활성 가능한 실제 카메라 번호를 반환."""
    devices = list_dshow_video_devices()
    for dev in devices:
        if dev["is_gopro"] and dev["kind"] == "video":
            return dev["index"]
    for dev in devices:
        if dev["kind"] == "video" and not dev["is_dummy_virtual"]:
            return dev["index"]
    return default_idx


def compute_downscaled_size(
    width: int,
    height: int,
    max_width: int = DEFAULT_MAX_CAMERA_WIDTH,
    max_height: int = DEFAULT_MAX_CAMERA_HEIGHT,
) -> tuple[int, int]:
    """원본 종횡비를 유지하면서 (max_width, max_height) 이하의 짝수 해상도를 계산합니다."""
    if width <= 0 or height <= 0:
        return width, height
    if not max_width or not max_height or (width <= max_width and height <= max_height):
        return width, height
    scale = min(max_width / float(width), max_height / float(height))
    new_w = max(2, int(round(width * scale / 2.0)) * 2)
    new_h = max(2, int(round(height * scale / 2.0)) * 2)
    return new_w, new_h


def downscale_frame_if_needed(
    frame: np.ndarray,
    max_width: int = DEFAULT_MAX_CAMERA_WIDTH,
    max_height: int = DEFAULT_MAX_CAMERA_HEIGHT,
) -> np.ndarray:
    """고프로 1080p/4K 등 과도한 고해상도 프레임을 지정 해상도(기본 720p)로 고속 리사이즈합니다."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return frame
    h, w = frame.shape[:2]
    target_w, target_h = compute_downscaled_size(w, h, max_width, max_height)
    if (target_w, target_h) != (w, h):
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return frame


def detect_gopro_usb_ip() -> str | None:
    """PC의 네트워크 인터페이스 중 GoPro USB RNDIS 대역(172.20~29.X.50/52~59)을 찾아 GoPro IP(.51)를 반환."""
    candidates = []
    try:
        hostname = socket.gethostname()
        infos = socket.getaddrinfo(hostname, None, socket.AF_INET)
        for info in infos:
            ip = info[4][0]
            parts = ip.split(".")
            if len(parts) == 4 and parts[0] == "172":
                second = int(parts[1])
                if 20 <= second <= 29:
                    gopro_ip = f"{parts[0]}.{parts[1]}.{parts[2]}.51"
                    if gopro_ip not in candidates:
                        candidates.append(gopro_ip)
    except Exception:
        pass

    for gopro_ip in candidates:
        status = query_gopro_usb_status(gopro_ip, timeout=0.6)
        if status is not None:
            return gopro_ip
    return candidates[0] if candidates else None


def _proxy_free_http_get(url: str, timeout: float = 1.5) -> dict | None:
    """Windows 레지스트리 및 환경변수 프록시를 100% 무시하고 직접 HTTP GET 수행."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"User-Agent": "GoProWebcamClient/1.0"})
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw) if raw.strip().startswith("{") else {"raw": raw}
    except Exception:
        return None


def query_gopro_usb_status(gopro_ip: str, timeout: float = 1.0) -> dict | None:
    """프록시를 우회하여 GoPro USB 웹캠 상태(/gp/gpWebcam/STATUS) 조회."""
    return _proxy_free_http_get(f"http://{gopro_ip}:8080/gp/gpWebcam/STATUS", timeout=timeout)


def start_gopro_usb_webcam(gopro_ip: str, res: str = "720", fov: str = "4") -> bool:
    """프록시를 우회하여 GoPro USB 웹캠 스트리밍 시작 명령 전송 (기본 720p로 대역폭/발열/지연 최소화)."""
    res_code = "7" if str(res) in ("720", "7") else ("4" if str(res) in ("480", "360", "540", "4") else "12")
    for url in (
        f"http://{gopro_ip}:8080/gp/gpWebcam/START?res={res_code}&fov={fov}",
        f"http://{gopro_ip}:8080/gp/gpWebcam/START?res={res}&fov={fov}",
        f"http://{gopro_ip}:8080/gp/gpWebcam/START",
    ):
        resp = _proxy_free_http_get(url, timeout=2.0)
        if resp is not None:
            return True
    return False


def stop_gopro_usb_webcam(gopro_ip: str) -> bool:
    """프록시를 우회하여 GoPro USB 웹캠 스트리밍 중지."""
    resp = _proxy_free_http_get(f"http://{gopro_ip}:8080/gp/gpWebcam/STOP", timeout=1.5)
    return resp is not None


def _is_active_nonblack_frame(frame: np.ndarray) -> bool:
    """가상 카메라 대기 화면(예: OMEN Cam 검은색 배경 mean=0.5)과 실제 활성 카메라 영상을 구분합니다."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    sample = frame[::8, ::8]
    # OMEN Cam 등 가상 대기 화면은 배경이 검은색(mean < 2.0)이고 작은 로고만 있어 std만 약간 높으므로 mean > 5.0 기준 적용
    return bool(sample.mean() > 5.0)


def scan_available_cameras(max_index: int = 4) -> dict:
    """현재 PC의 DirectShow 장치 이름을 먼저 조회하여 비활성/가상 더미 장치를 건드리지 않고 고속 진단합니다."""
    configure_local_and_gopro_no_proxy()
    sys_proxies = urllib.request.getproxies()
    gopro_ip = detect_gopro_usb_ip()
    gopro_status = query_gopro_usb_status(gopro_ip) if gopro_ip else None
    dshow_devices = list_dshow_video_devices()
    dev_by_idx = {d["index"]: d for d in dshow_devices}

    cameras = []
    indices_to_check = [d["index"] for d in dshow_devices] if dshow_devices else list(range(max_index))

    for idx in indices_to_check:
        dev_info = dev_by_idx.get(idx)
        dev_name = dev_info["name"] if dev_info else f"Camera {idx}"

        # 1) 하드웨어적으로 비활성(none)인 장치(예: 닫힌 내장 카메라)는 열지 않고 즉시 표시 (OpenCV 경고 방지)
        if dev_info and dev_info["kind"] == "none":
            cameras.append({
                "index": idx,
                "name": dev_name,
                "backend": "DSHOW",
                "resolution": "비활성 장치",
                "active": False,
                "status_tag": "⚪ 비활성(none)",
            })
            continue

        # 2) OMEN Cam 등 열기만 해도 콘솔에 'error = 0'을 찍는 더미 가상 카메라는 직접 열지 않고 대기 상태로 표시
        if dev_info and dev_info["is_dummy_virtual"]:
            cameras.append({
                "index": idx,
                "name": dev_name,
                "backend": "DSHOW",
                "resolution": "가상 카메라",
                "active": False,
                "status_tag": "⚫ 가상 대기 장치",
            })
            continue

        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(8):
                    ok, frame = cap.read()
                    if ThreadedWebcam._valid_frame(ok, frame):
                        raw_h, raw_w = frame.shape[:2]
                        out_w, out_h = compute_downscaled_size(raw_w, raw_h)
                        active = _is_active_nonblack_frame(frame)
                        res_label = (
                            f"{raw_w}x{raw_h}"
                            if (raw_w, raw_h) == (out_w, out_h)
                            else f"{raw_w}x{raw_h} → {out_w}x{out_h} 자동축소"
                        )
                        status_tag = "🟢 활성 영상" if active else "⚫ 대기/검은 화면"
                        cameras.append({
                            "index": idx,
                            "name": dev_name,
                            "backend": "DSHOW",
                            "resolution": res_label,
                            "active": active,
                            "status_tag": status_tag,
                        })
                        break
                    time.sleep(0.04)
        except Exception:
            pass
        finally:
            cap.release()

    return {
        "cameras": cameras,
        "gopro_usb_ip": gopro_ip,
        "gopro_status": gopro_status,
        "system_proxies_detected": sys_proxies,
        "no_proxy_configured": os.environ.get("NO_PROXY", ""),
    }


class ThreadedWebcam:
    """실시간 웹캠 및 GoPro 프레임 캡처 백그라운드 스레드 (프록시 우회 + 고화질 프레임 자동 다운스케일링 내장)"""

    def __init__(self, cam_idx=2, max_width=DEFAULT_MAX_CAMERA_WIDTH, max_height=DEFAULT_MAX_CAMERA_HEIGHT):
        configure_local_and_gopro_no_proxy()
        self.cap = None
        self.active_cam_idx = cam_idx
        self.max_width = int(max_width) if max_width else DEFAULT_MAX_CAMERA_WIDTH
        self.max_height = int(max_height) if max_height else DEFAULT_MAX_CAMERA_HEIGHT
        self.raw_resolution = (0, 0)
        self.output_resolution = (0, 0)
        self.gopro_ip = detect_gopro_usb_ip()
        self._using_direct_udp = False

        if self.gopro_ip:
            status = query_gopro_usb_status(self.gopro_ip, timeout=0.8)
            print(f"[CAMERA] GoPro USB interface detected at {self.gopro_ip} (status={status})", flush=True)

        dshow_devices = list_dshow_video_devices()
        dev_by_idx = {d["index"]: d for d in dshow_devices}

        # 만약 사용자가 선택한 cam_idx가 (none) 비활성 장치이거나 더미 가상 카메라(OMEN Cam 등)이고,
        # 실제 GoPro Webcam이 다른 인덱스에 존재한다면 GoPro 인덱스를 최우선으로 시도하여 'error = 0' 스팸 방지
        preferred_gopro_idx = next((d["index"] for d in dshow_devices if d["is_gopro"] and d["kind"] == "video"), None)
        requested_dev = dev_by_idx.get(cam_idx)
        if preferred_gopro_idx is not None and requested_dev and (requested_dev["kind"] == "none" or requested_dev["is_dummy_virtual"]):
            print(
                f"[CAMERA] Index {cam_idx} ('{requested_dev['name']}') is inactive/virtual; "
                f"auto-selecting GoPro Webcam at index {preferred_gopro_idx}",
                flush=True,
            )
            candidate_indices = [preferred_gopro_idx, cam_idx]
        else:
            candidate_indices = [cam_idx]
            if preferred_gopro_idx is not None and preferred_gopro_idx not in candidate_indices:
                candidate_indices.append(preferred_gopro_idx)

        for fallback_idx in (2, 1, 0):
            if fallback_idx not in candidate_indices and cam_idx >= 0:
                fb_dev = dev_by_idx.get(fallback_idx)
                if fb_dev and (fb_dev["kind"] == "none" or fb_dev["is_dummy_virtual"]):
                    continue
                candidate_indices.append(fallback_idx)

        fallback_black_candidate = None

        # 1단계: DirectShow 기본 스트림으로 즉시 연결 (cap.set(FOURCC/WIDTH/HEIGHT)으로 인한 7초 블로킹 및 드라이버 오류 방지)
        for target_idx in candidate_indices:
            if target_idx < 0:
                continue
            dev_info = dev_by_idx.get(target_idx)
            if dev_info and dev_info["kind"] == "none":
                continue
            dev_name = dev_info["name"] if dev_info else f"Camera {target_idx}"

            for backend, backend_name in ((cv2.CAP_DSHOW, "DSHOW"), (cv2.CAP_MSMF, "MSMF")):
                cap = cv2.VideoCapture(target_idx, backend)
                keep_open = False
                try:
                    if not cap.isOpened():
                        continue
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    max_retries = 15 if target_idx == candidate_indices[0] else 8
                    got_valid = False
                    for _ in range(max_retries):
                        grabbed, raw_frame = cap.read()
                        if self._valid_frame(grabbed, raw_frame):
                            got_valid = True
                            if _is_active_nonblack_frame(raw_frame):
                                raw_h, raw_w = raw_frame.shape[:2]
                                frame = downscale_frame_if_needed(raw_frame, self.max_width, self.max_height)
                                out_h, out_w = frame.shape[:2]
                                self.raw_resolution = (raw_w, raw_h)
                                self.output_resolution = (out_w, out_h)
                                self.active_cam_idx = target_idx
                                scale_note = (
                                    f" -> auto-downscaled to {out_w}x{out_h}"
                                    if (raw_w, raw_h) != (out_w, out_h)
                                    else ""
                                )
                                print(
                                    f"[CAMERA] Connected to [{target_idx}] '{dev_name}' "
                                    f"(input={raw_w}x{raw_h}{scale_note}, backend={backend_name})",
                                    flush=True,
                                )
                                self.cap, self.grabbed, self.frame = cap, grabbed, frame
                                self._raw_frame = raw_frame
                                keep_open = True
                                break
                        time.sleep(0.06)

                    if self.cap is None and got_valid and fallback_black_candidate is None:
                        fallback_black_candidate = (target_idx, backend, backend_name, dev_name)

                    # DSHOW에서 이미 장치가 열렸다면 동일 인덱스에 대해 느린 MSMF를 중복 시도하지 않음
                    break
                except cv2.error as exc:
                    print(f"[WARN][CAMERA] index={target_idx} backend={backend_name} failed: {exc}", flush=True)
                finally:
                    if not keep_open:
                        cap.release()
                if self.cap is not None:
                    break
            if self.cap is not None:
                break

        # 다른 활성 카메라가 없고 검은 화면 카메라만 열렸다면 해당 카메라라도 연결 유지
        if self.cap is None and fallback_black_candidate is not None:
            fb_idx, fb_backend, fb_name, fb_dev_name = fallback_black_candidate
            cap = cv2.VideoCapture(fb_idx, fb_backend)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                grabbed, raw_frame = cap.read()
                if self._valid_frame(grabbed, raw_frame):
                    raw_h, raw_w = raw_frame.shape[:2]
                    frame = downscale_frame_if_needed(raw_frame, self.max_width, self.max_height)
                    out_h, out_w = frame.shape[:2]
                    self.raw_resolution = (raw_w, raw_h)
                    self.output_resolution = (out_w, out_h)
                    self.active_cam_idx = fb_idx
                    print(
                        f"[CAMERA] Opened [{fb_idx}] '{fb_dev_name}' ({raw_w}x{raw_h} -> {out_w}x{out_h}, standby/dark frame)",
                        flush=True,
                    )
                    self.cap, self.grabbed, self.frame = cap, grabbed, frame
                    self._raw_frame = raw_frame
                else:
                    cap.release()

        # 2단계: DirectShow 웹캠 열기가 실패했으나 GoPro USB(172.2X.1XX.51)가 감지된 경우 UDP 직접 스트림 시도
        if self.cap is None and self.gopro_ip:
            target_res = "480" if self.max_height < 720 else "720"
            print(
                f"[CAMERA] DirectShow open failed; attempting direct GoPro USB {target_res}p stream from {self.gopro_ip}...",
                flush=True,
            )
            if start_gopro_usb_webcam(self.gopro_ip, res=target_res, fov="4"):
                udp_url = "udp://@0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000"
                cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
                try:
                    if cap.isOpened():
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        for _ in range(30):
                            grabbed, raw_frame = cap.read()
                            if self._valid_frame(grabbed, raw_frame):
                                raw_h, raw_w = raw_frame.shape[:2]
                                frame = downscale_frame_if_needed(raw_frame, self.max_width, self.max_height)
                                out_h, out_w = frame.shape[:2]
                                self.raw_resolution = (raw_w, raw_h)
                                self.output_resolution = (out_w, out_h)
                                print(
                                    f"[CAMERA] GoPro Direct USB UDP stream initialized: input={raw_w}x{raw_h} -> output={out_w}x{out_h}",
                                    flush=True,
                                )
                                self.cap, self.grabbed, self.frame = cap, grabbed, frame
                                self._raw_frame = raw_frame
                                self._using_direct_udp = True
                                break
                            time.sleep(0.1)
                except cv2.error as exc:
                    print(f"[WARN][CAMERA] GoPro Direct UDP failed: {exc}", flush=True)
                finally:
                    if self.cap is not cap:
                        cap.release()

        if self.cap is None:
            gopro_hint = (
                f" (감지된 GoPro USB IP: {self.gopro_ip} — Windows 방화벽에서 UDP 8554 허용 여부 또는 GoPro Webcam 앱 점유를 확인하세요)"
                if self.gopro_ip
                else ""
            )
            raise RuntimeError(
                f"카메라 {cam_idx}에서 영상을 읽을 수 없습니다.{gopro_hint} "
                "GoPro Webcam 앱 트레이 아이콘에 파란색 불이 켜져 있는지, "
                "다른 앱(Zoom/OBS/미리보기)이 카메라를 점유하고 있지 않은지, "
                "카메라 인덱스 번호(cam_idx=0, 1, 2)를 확인해 주세요."
            )

        self.captured_at = time.perf_counter()
        self._last_valid_at = self.captured_at
        self._capture_error = None
        self.frame_id = 0
        self._scaled_frame_id = 0
        self.started = False
        self.read_lock = threading.Lock()

    @staticmethod
    def _valid_frame(grabbed, frame):
        """수신된 3채널 이미지 유효성 확인"""
        return (
            grabbed
            and isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
            and frame.shape[2] == 3
        )

    def start(self):
        """캡처 워커 스레드 시작"""
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    # camera.py의 update() 수정[cite: 1]
    def update(self):
        """최신 프레임 지속 갱신 루프 (밀림 방지: 캡처 스레드에서는 버퍼 비우기만 수행하고, 다운스케일은 소비 시점에 프레임당 1회만 수행)"""
        while self.started: #[cite: 1]
            try: #[cite: 1]
                grabbed, raw_frame = self.cap.read() #[cite: 1]
            except cv2.error as exc: #[cite: 1]
                with self.read_lock: #[cite: 1]
                    self._capture_error = str(exc) #[cite: 1]
                break #[cite: 1]

            if not grabbed: #[cite: 1]
                time.sleep(0.005) #[cite: 1]
                continue

            captured_at = time.perf_counter() #[cite: 1]

            with self.read_lock: #[cite: 1]
                self.grabbed = grabbed #[cite: 1]
                self._raw_frame = raw_frame #[cite: 1]
                self.captured_at = captured_at #[cite: 1]
                self.frame_id += 1 #[cite: 1]
                self._last_valid_at = captured_at #[cite: 1]

            # time.sleep을 줄이거나 없애서 OS 드라이버의 내부 캡처 버퍼를 비워냅니다.
            time.sleep(0.001)

    def read(self):
        grabbed, frame, _, _ = self.read_sample()
        return grabbed, frame

    def read_sample(self):
        """프레임 복사본(최대 해상도 이하로 자동 축소 캐싱됨), 타임스탬프, 프레임 ID를 원자적으로 추출"""
        with self.read_lock:
            if self._capture_error:
                raise RuntimeError(f"카메라 프레임 읽기 실패: {self._capture_error}")
            raw_frame = getattr(self, "_raw_frame", self.frame)
            if not self.grabbed or raw_frame is None:
                if time.perf_counter() - self._last_valid_at >= 5.0:
                    raise RuntimeError("카메라 영상이 5초 이상 끊겼습니다. GoPro USB 연결 상태를 확인해 주세요.")
                return False, None, self.captured_at, self.frame_id

            # 새 프레임이 소비될 때만 다운스케일을 1회 수행하고 캐싱 (60fps 캡처 중 버려지는 프레임에는 CPU 리사이즈 생략)
            if self._scaled_frame_id != self.frame_id or self.frame is None:
                self.frame = downscale_frame_if_needed(raw_frame, self.max_width, self.max_height)
                self._scaled_frame_id = self.frame_id

            return True, self.frame.copy(), self.captured_at, self.frame_id

    def stop(self):
        self.started = False
        if self.cap and self.cap.isOpened():
            self.cap.release()
        if getattr(self, "_using_direct_udp", False) and getattr(self, "gopro_ip", None):
            stop_gopro_usb_webcam(self.gopro_ip)
