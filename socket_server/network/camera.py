"""720p/1080p 웹캠·GoPro 연결, 프록시 우회 및 자동 해상도 축소(다운스케일) 모듈."""

from collections import deque
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

# OpenCV의 장치 탐색 경고 로그([WARN:... VIDEOIO(DSHOW/MSMF)]) 출력 억제
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

# 기본 최대 캡처 해상도 (720p)
DEFAULT_MAX_CAMERA_WIDTH = 1280
DEFAULT_MAX_CAMERA_HEIGHT = 720

# 실제 카메라가 아닌 가상 더미 카메라 판별 키워드
_DUMMY_VIRTUAL_CAM_KEYWORDS = ("omen cam", "obs virtual", "xsplit", "manycam", "snap camera")


def configure_local_and_gopro_no_proxy() -> None:
    """로컬 웹소켓과 GoPro USB 통신이 시스템 HTTP 프록시를 타지 않도록 NO_PROXY를 설정한다."""
    for env_key in ("NO_PROXY", "no_proxy"):
        existing = os.environ.get(env_key, "")
        items = [x.strip() for x in existing.split(",") if x.strip()]
        for host in GOPRO_NO_PROXY_HOSTS:
            if host not in items:
                items.append(host)
        os.environ[env_key] = ",".join(items)

    if "OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "overrun_nonfatal;1|fifo_size;50000000"


configure_local_and_gopro_no_proxy()


def _find_ffmpeg_exe() -> str | None:
    """사용 가능한 ffmpeg 실행 파일 경로를 찾는다."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    binaries = Path(__file__).resolve().parent.parent / "configs" / "bin"
    for path in sorted(binaries.glob("ffmpeg*")):
        if path.is_file() and (path.suffix.lower() == ".exe" if os.name == "nt" else os.access(path, os.X_OK)):
            return str(path)
    if os.name == "nt":
        local_app = Path(os.environ.get("LOCALAPPDATA", ""))
        winget_link = local_app / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
        if winget_link.is_file():
            return str(winget_link)
        winget_pkgs = local_app / "Microsoft" / "WinGet" / "Packages"
        if winget_pkgs.is_dir():
            for found in winget_pkgs.glob("Gyan.FFmpeg*/**/ffmpeg.exe"):
                if found.is_file():
                    return str(found)
    return None


def _find_scrcpy_exe() -> str | None:
    """사용 가능한 scrcpy 실행 파일 경로를 찾는다."""
    executable = shutil.which("scrcpy")
    if executable:
        return executable
    server_root = Path(__file__).resolve().parent.parent
    repo_root = server_root.parent
    candidates = [
        server_root / "configs" / "bin" / "scrcpy.exe",
        repo_root / "tmp" / "scrcpy" / "scrcpy.exe",
        repo_root / "scrcpy" / "scrcpy.exe",
    ]
    if os.name == "nt":
        local_app = Path(os.environ.get("LOCALAPPDATA", ""))
        candidates.insert(0, local_app / "Microsoft" / "WinGet" / "Links" / "scrcpy.exe")
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    if os.name == "nt":
        winget_pkgs = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
        if winget_pkgs.is_dir():
            for found in winget_pkgs.glob("Genymobile.scrcpy*/**/scrcpy.exe"):
                if found.is_file():
                    return str(found)
    for found in (repo_root / "tmp").glob("scrcpy-*/scrcpy.exe"):
        if found.is_file():
            return str(found)
    return None


def _resolve_adb_and_scrcpy_server(scrcpy_exe: str) -> tuple[str | None, Path | None]:
    """scrcpy 실행 파일 기준으로 adb 경로와 scrcpy-server 경로를 찾는다."""
    scrcpy_dir = Path(scrcpy_exe).resolve().parent
    adb_name = "adb.exe" if os.name == "nt" else "adb"
    adb_path = scrcpy_dir / adb_name
    adb_exe = str(adb_path) if adb_path.is_file() else shutil.which("adb")
    server_jar = scrcpy_dir / "scrcpy-server"
    return adb_exe, (server_jar if server_jar.is_file() else None)


def _get_scrcpy_version(scrcpy_exe: str) -> str:
    """설치된 scrcpy 버전을 조회한다."""
    try:
        out = subprocess.check_output(
            [scrcpy_exe, "--version"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=3.0,
        )
        match = re.search(r"scrcpy\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", out)
        if match:
            return match.group(1)
    except Exception:
        pass
    return "5.0.1"


def detect_android_usb_device() -> dict | None:
    """ADB USB로 연결된 Android 장치(serial, state)를 감지한다."""
    scrcpy_exe = _find_scrcpy_exe()
    adb_exe = None
    if scrcpy_exe:
        adb_exe, _ = _resolve_adb_and_scrcpy_server(scrcpy_exe)
    if not adb_exe:
        adb_exe = shutil.which("adb")
    if not adb_exe:
        return None
    try:
        startup_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        out = subprocess.check_output(
            [adb_exe, "devices"],
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=3.0,
            creationflags=startup_flags,
        )
    except Exception:
        return None

    unauthorized_serial = None
    for line in out.splitlines()[1:]:
        parts = line.strip().split()
        if len(parts) >= 2 and ":" not in parts[0]:
            serial, state = parts[0], parts[1]
            if state == "device":
                return {"serial": serial, "state": "device"}
            if state == "unauthorized" and unauthorized_serial is None:
                unauthorized_serial = serial
    if unauthorized_serial:
        return {"serial": unauthorized_serial, "state": "unauthorized"}
    return None


def list_dshow_video_devices() -> list[dict]:
    """ffmpeg로 Windows DirectShow 영상 장치 이름과 인덱스 목록을 가져온다."""
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


def find_preferred_camera_index(default_idx: int = 1) -> int:
    """GoPro Webcam이 있으면 그 번호를, 없으면 첫 번째 실제 카메라 번호를 반환한다."""
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
    """가로세로 비율을 유지하면서 max_width/max_height 이내(짝수 크기)로 축소 크기를 계산한다."""
    if width <= 0 or height <= 0:
        return width, height
    if not max_width or not max_height:
        return width, height
    # 세로(portrait) 영상은 720p 프리셋(1280x720)을 세로 기준(720x1280)으로 적용해 과도한 축소를 방지한다
    eff_max_w, eff_max_h = (max_height, max_width) if (height > width and max_width > max_height) else (max_width, max_height)
    if width <= eff_max_w and height <= eff_max_h:
        return width, height
    scale = min(eff_max_w / float(width), eff_max_h / float(height))
    new_w = max(2, int(round(width * scale / 2.0)) * 2)
    new_h = max(2, int(round(height * scale / 2.0)) * 2)
    return new_w, new_h


def downscale_frame_if_needed(
    frame: np.ndarray,
    max_width: int = DEFAULT_MAX_CAMERA_WIDTH,
    max_height: int = DEFAULT_MAX_CAMERA_HEIGHT,
) -> np.ndarray:
    """프레임이 지정 해상도보다 크면 빠르게 리사이즈한다."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return frame
    h, w = frame.shape[:2]
    target_w, target_h = compute_downscaled_size(w, h, max_width, max_height)
    if (target_w, target_h) != (w, h):
        return cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return frame


def normalize_rotation(rotation: int | None) -> int:
    """회전 각도를 0, 90, 180, 270 중 하나로 정규화한다."""
    try:
        deg = int(rotation or 0) % 360
    except (TypeError, ValueError):
        return 0
    return deg if deg in (0, 90, 180, 270) else 0


def rotate_frame_if_needed(frame: np.ndarray, rotation: int = 0) -> np.ndarray:
    """지정된 각도(90/180/270)로 프레임을 회전한다."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.ndim != 3:
        return frame
    deg = normalize_rotation(rotation)
    if deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def detect_gopro_usb_ip() -> str | None:
    """USB로 연결된 GoPro 네트워크 대역(172.2X.X.51)을 자동으로 찾는다."""
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
    """시스템 프록시를 거치지 않고 HTTP GET 요청을 보낸다."""
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(url, headers={"User-Agent": "GoProWebcamClient/1.0"})
    try:
        with opener.open(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
            return json.loads(raw) if raw.strip().startswith("{") else {"raw": raw}
    except Exception:
        return None


def query_gopro_usb_status(gopro_ip: str, timeout: float = 1.0) -> dict | None:
    """GoPro USB 웹캠 상태(/gp/gpWebcam/STATUS)를 조회한다."""
    return _proxy_free_http_get(f"http://{gopro_ip}:8080/gp/gpWebcam/STATUS", timeout=timeout)


def start_gopro_usb_webcam(gopro_ip: str, res: str = "720", fov: str = "0") -> bool:
    """GoPro USB 인터페이스에 웹캠 스트림 시작 명령을 보낸다. (fov=0: 광각 Wide, 4: Linear)"""
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


def set_gopro_wide_fov(gopro_ip: str) -> bool:
    """GoPro 웹캠 화각을 크롭 없는 광각(Wide, fov=0)으로 전환한다."""
    if not gopro_ip:
        return False
    resp = _proxy_free_http_get(f"http://{gopro_ip}:8080/gp/gpWebcam/SETTINGS?fov=0", timeout=0.8)
    return resp is not None


def stop_gopro_usb_webcam(gopro_ip: str) -> bool:
    """GoPro USB 인터페이스에 웹캠 스트림 중지 명령을 보낸다."""
    resp = _proxy_free_http_get(f"http://{gopro_ip}:8080/gp/gpWebcam/STOP", timeout=1.5)
    return resp is not None


def _is_active_nonblack_frame(frame: np.ndarray) -> bool:
    """검은 대기 화면이 아니라 실제 카메라 영상이 들어오는 프레임인지 확인한다."""
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return False
    sample = frame[::8, ::8]
    return bool(sample.mean() > 5.0)


def _read_exact(stream, num_bytes: int, stop_event: threading.Event) -> bytearray | None:
    """파이프에서 정확히 num_bytes 길이의 raw 프레임을 읽는다."""
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    pos = 0
    while pos < num_bytes and not stop_event.is_set():
        chunk_len = stream.readinto(view[pos:])
        if not chunk_len:
            return None
        pos += chunk_len
    if pos < num_bytes:
        return None
    return buf


class FFmpegDShowCapture:
    """FFmpeg DirectShow 파이프를 백그라운드 스레드로 비우는 캡처 래퍼.

    GoPro Webcam 가상 카메라는 스플래시(대기 로고)에서 실영상으로 전환될 때
    OpenCV CAP_DSHOW의 1초 타임아웃에 걸려 1 FPS 검은 화면으로 멈추는 문제가 있어,
    번들 FFmpeg subprocess로 직접 수신하며 별도 스레드가 파이프를 즉시 비운다.
    """

    def __init__(
        self,
        ffmpeg_exe: str,
        device_name: str,
        width: int = DEFAULT_MAX_CAMERA_WIDTH,
        height: int = DEFAULT_MAX_CAMERA_HEIGHT,
        pixel_format: str | None = "bgr0",
        startup_timeout: float = 5.0,
        capture_width: int | None = None,
        capture_height: int | None = None,
    ) -> None:
        self.ffmpeg_exe = ffmpeg_exe
        self.device_name = device_name
        self.width = int(width)
        self.height = int(height)
        self.capture_width = int(capture_width) if capture_width else self.width
        self.capture_height = int(capture_height) if capture_height else self.height
        self.pixel_format = pixel_format
        self.startup_timeout = float(startup_timeout)
        self.source_label = f"FFmpeg DirectShow ({device_name})"

        self._frame_bytes = self.width * self.height * 3
        self._proc: subprocess.Popen | None = None
        self._reader_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._cond = threading.Condition()

        self._latest_frame: np.ndarray | None = None
        self._frame_seq = 0
        self._last_read_seq = 0
        self._frame_times: deque[float] = deque(maxlen=60)
        self._stderr_lines: deque[str] = deque(maxlen=25)
        self._closed = False

    def _build_command(self) -> list[str]:
        """FFmpeg DirectShow 실행 커맨드를 구성한다."""
        cmd = [
            self.ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "error",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
            "-f",
            "dshow",
            "-rtbufsize",
            "32M",
        ]
        if self.pixel_format:
            # GoPro DirectShow는 1280x720 요청 시 1080p 영상을 크롭하는 경우가 있어 풀화각(capture_width x capture_height)으로 수신
            cmd.extend(["-pixel_format", self.pixel_format, "-video_size", f"{self.capture_width}x{self.capture_height}"])
        cmd.extend(["-i", f"video={self.device_name}"])
        if (not self.pixel_format) or (self.capture_width, self.capture_height) != (self.width, self.height):
            cmd.extend(["-vf", f"scale={self.width}:{self.height}"])
        cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
        return cmd

    def open(self) -> bool:
        """FFmpeg 프로세스를 시작하고 첫 프레임 수신(또는 초기 종료)까지 대기한다."""
        cmd = self._build_command()
        startup_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_bytes,
                creationflags=startup_flags,
            )
        except OSError as exc:
            self._stderr_lines.append(f"FFmpeg spawn failed: {exc}")
            return False

        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._stderr_thread = threading.Thread(target=self._stderr_loop, daemon=True)
        self._reader_thread.start()
        self._stderr_thread.start()

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._first_frame_event.wait(timeout=0.1):
                return True
            if self._proc.poll() is not None:
                break

        self.release()
        return False

    def _reader_loop(self) -> None:
        """stdout 파이프에서 raw BGR 프레임을 계속 읽어 최신 슬롯을 갱신한다."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            while not self._stop_event.is_set():
                raw = _read_exact(proc.stdout, self._frame_bytes, self._stop_event)
                if raw is None:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                now = time.monotonic()
                with self._cond:
                    self._latest_frame = frame
                    self._frame_seq += 1
                    self._frame_times.append(now)
                    self._first_frame_event.set()
                    self._cond.notify_all()
        except OSError:
            pass
        finally:
            with self._cond:
                self._cond.notify_all()

    def _stderr_loop(self) -> None:
        """stderr 파이프가 가득 차 FFmpeg가 멈추지 않도록 오류 메시지를 비운다."""
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for raw_line in proc.stderr:
                if self._stop_event.is_set():
                    break
                text = raw_line.decode("utf-8", errors="replace").strip()
                if text:
                    self._stderr_lines.append(text)
        except OSError:
            pass

    def _calc_capture_fps(self) -> float:
        """최근 수신 타임스탬프 기준 실제 캡처 FPS를 계산한다."""
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def isOpened(self) -> bool:
        """프로세스가 아직 살아 있고 닫히지 않았는지 반환한다."""
        return not self._closed and self._proc is not None and self._proc.poll() is None

    def read_latest(self, last_seq: int, timeout: float = 0.25) -> tuple[bool, np.ndarray | None, int, float]:
        """직전 last_seq 이후 새 프레임이 올 때까지 대기 후 반환한다."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while (
                self._frame_seq == last_seq
                and not self._stop_event.is_set()
                and self.isOpened()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None, last_seq, self._calc_capture_fps()
                self._cond.wait(timeout=remaining)

            if self._frame_seq > last_seq and self._latest_frame is not None:
                return True, self._latest_frame, self._frame_seq, self._calc_capture_fps()
            return False, None, last_seq, self._calc_capture_fps()

    def read(self) -> tuple[bool, np.ndarray | None]:
        """OpenCV VideoCapture 호환용 단일 프레임 읽기."""
        ok, frame, seq, _ = self.read_latest(self._last_read_seq, timeout=0.5)
        if ok:
            self._last_read_seq = seq
        return ok, frame

    def get(self, prop_id: int) -> float:
        """해상도/FPS 프로퍼티 조회."""
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return self._calc_capture_fps()
        return 0.0

    def get_stderr_summary(self) -> str:
        """최근 FFmpeg stderr 로그 요약을 반환한다."""
        return " | ".join(self._stderr_lines)

    def release(self) -> None:
        """DirectShow COM 필터가 꼬이지 않도록 FFmpeg에 'q'를 보내 정상 종료한다."""
        if self._closed:
            return
        self._closed = True

        proc = self._proc
        if proc is not None:
            # stdout 리더를 끄기 전에 'q'로 FFmpeg를 먼저 종료시켜 파이프 쓰기 대기로 인한 강제 종료(rc=1)를 방지한다.
            if proc.poll() is None and proc.stdin is not None:
                try:
                    proc.stdin.write(b"q\n")
                    proc.stdin.flush()
                    proc.stdin.close()
                except OSError:
                    pass
            try:
                proc.wait(timeout=2.5)
            except subprocess.TimeoutExpired:
                self._stop_event.set()
                with self._cond:
                    self._cond.notify_all()
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)

        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()

        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

        for thread in (self._reader_thread, self._stderr_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=1.0)


def open_ffmpeg_dshow_camera(
    device_name: str,
    width: int = DEFAULT_MAX_CAMERA_WIDTH,
    height: int = DEFAULT_MAX_CAMERA_HEIGHT,
    startup_timeout: float = 5.0,
    log_prefix: str | None = "[CAMERA]",
    capture_width: int | None = None,
    capture_height: int | None = None,
) -> FFmpegDShowCapture | None:
    """지정한 DirectShow 장치명으로 FFmpeg DirectShow 캡처를 연다(1080p 풀화각 수신 후 스케일 우선)."""
    ffmpeg_exe = _find_ffmpeg_exe()
    if not ffmpeg_exe:
        if log_prefix:
            print(f"{log_prefix} FFmpeg binary not found; cannot open DirectShow pipe for '{device_name}'.", flush=True)
        return None

    cap_w = int(capture_width) if capture_width else int(width)
    cap_h = int(capture_height) if capture_height else int(height)
    attempts = (
        (f"bgr0 ({cap_w}x{cap_h} -> {width}x{height})", "bgr0", cap_w, cap_h),
        ("auto-negotiated", None, width, height),
    )
    for label, pix_fmt, req_w, req_h in attempts:
        if log_prefix:
            print(f"{log_prefix} Attempting FFmpeg DirectShow '{device_name}' ({label})...", flush=True)
        cap = FFmpegDShowCapture(
            ffmpeg_exe=ffmpeg_exe,
            device_name=device_name,
            width=width,
            height=height,
            pixel_format=pix_fmt,
            startup_timeout=startup_timeout,
            capture_width=req_w,
            capture_height=req_h,
        )
        if cap.open():
            if log_prefix:
                print(f"{log_prefix} Connected '{device_name}' via FFmpeg DirectShow ({label}).", flush=True)
            return cap
        err = cap.get_stderr_summary()
        if err and log_prefix:
            print(f"{log_prefix} FFmpeg DirectShow failed ({label}): {err}", flush=True)
    return None


class ScrcpyUsbCapture:
    """Android USB(scrcpy-server raw H.264 -> FFmpeg -> raw BGR24) 초저지연 캡처 래퍼."""

    def __init__(
        self,
        scrcpy_exe: str,
        ffmpeg_exe: str,
        adb_exe: str,
        server_jar: Path,
        width: int = DEFAULT_MAX_CAMERA_WIDTH,
        height: int = DEFAULT_MAX_CAMERA_HEIGHT,
        fps: int = 30,
        facing: str = "back",
        camera_id: str | None = None,
        serial: str | None = None,
        port: int = 27183,
        startup_timeout: float = 8.0,
    ) -> None:
        self.scrcpy_exe = scrcpy_exe
        self.ffmpeg_exe = ffmpeg_exe
        self.adb_exe = adb_exe
        self.server_jar = server_jar
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.facing = facing
        self.camera_id = camera_id
        self.serial = serial
        self.port = int(port)
        self.startup_timeout = float(startup_timeout)

        self._frame_bytes = self.width * self.height * 3
        self._adb_proc: subprocess.Popen | None = None
        self._ffmpeg_proc: subprocess.Popen | None = None
        self._sock: socket.socket | None = None
        self._threads: list[threading.Thread] = []
        self._stop_event = threading.Event()
        self._first_frame_event = threading.Event()
        self._cond = threading.Condition()

        self._latest_frame: np.ndarray | None = None
        self._frame_seq = 0
        self._last_read_seq = 0
        self._frame_times: deque[float] = deque(maxlen=60)
        self._error_lines: deque[str] = deque(maxlen=25)
        self._closed = False

    def _adb_cmd(self, *args: str) -> list[str]:
        cmd = [self.adb_exe]
        if self.serial:
            cmd.extend(["-s", self.serial])
        cmd.extend(args)
        return cmd

    def open(self) -> bool:
        """scrcpy-server를 푸시·실행하고 FFmpeg 파이프를 통해 첫 프레임까지 대기한다."""
        startup_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        scrcpy_ver = _get_scrcpy_version(self.scrcpy_exe)

        push = subprocess.run(
            self._adb_cmd("push", str(self.server_jar), "/data/local/tmp/scrcpy-server.jar"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            creationflags=startup_flags,
        )
        if push.returncode != 0:
            self._error_lines.append(f"adb push failed: {(push.stderr or push.stdout).strip()}")
            return False

        subprocess.run(
            self._adb_cmd("forward", "--remove", f"tcp:{self.port}"),
            capture_output=True,
            creationflags=startup_flags,
        )
        fwd = subprocess.run(
            self._adb_cmd("forward", f"tcp:{self.port}", "localabstract:scrcpy"),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            creationflags=startup_flags,
        )
        if fwd.returncode != 0:
            self._error_lines.append(f"adb forward failed: {(fwd.stderr or fwd.stdout).strip()}")
            return False

        cam_arg = f"camera_id={self.camera_id}" if self.camera_id is not None else f"camera_facing={self.facing}"
        remote_cmd = (
            f"CLASSPATH=/data/local/tmp/scrcpy-server.jar "
            f"app_process / com.genymobile.scrcpy.Server {scrcpy_ver} "
            f"video_source=camera {cam_arg} camera_size={self.width}x{self.height} camera_fps={self.fps} "
            f"video_bit_rate=2000000 video_codec_options=i-frame-interval=1,max-bframes=0,priority=0 "
            f"audio=false control=false "
            f"tunnel_forward=true raw_stream=true send_dummy_byte=false log_level=info"
        )
        self._adb_proc = subprocess.Popen(
            self._adb_cmd("shell", remote_cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=startup_flags,
        )

        ffmpeg_cmd = [
            self.ffmpeg_exe,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-fflags",
            "nobuffer+discardcorrupt",
            "-flags",
            "low_delay",
            "-flags2",
            "fast",
            "-max_delay",
            "0",
            "-probesize",
            "32768",
            "-analyzeduration",
            "0",
            "-threads",
            "1",
            "-f",
            "h264",
            "-framerate",
            str(self.fps),
            "-i",
            "pipe:0",
            "-fps_mode",
            "passthrough",
            "-flush_packets",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ]
        self._ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
            creationflags=startup_flags,
        )

        self._threads = [
            threading.Thread(target=self._drain_stream, args=(self._adb_proc.stdout, "SCRCPY"), daemon=True),
            threading.Thread(target=self._drain_stream, args=(self._adb_proc.stderr, "SCRCPY-ERR"), daemon=True),
            threading.Thread(target=self._drain_stream, args=(self._ffmpeg_proc.stderr, "FFMPEG"), daemon=True),
            threading.Thread(target=self._tcp_to_ffmpeg_loop, daemon=True),
            threading.Thread(target=self._reader_loop, daemon=True),
        ]
        for t in self._threads:
            t.start()

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._first_frame_event.wait(timeout=0.1):
                return True
            if self._ffmpeg_proc.poll() is not None or self._stop_event.is_set():
                break

        self.release()
        return False

    def _drain_stream(self, stream, tag: str) -> None:
        if stream is None:
            return
        try:
            for raw_line in stream:
                if self._stop_event.is_set():
                    break
                text = raw_line.decode("utf-8", errors="replace").strip()
                if text:
                    self._error_lines.append(f"[{tag}] {text}")
        except OSError:
            pass

    def _tcp_to_ffmpeg_loop(self) -> None:
        sock = None
        deadline = time.monotonic() + min(6.0, self.startup_timeout)
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            try:
                s = socket.create_connection(("127.0.0.1", self.port), timeout=1.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(2.0)
                first = s.recv(65536)
                if first and self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                    sock = s
                    self._sock = s
                    self._ffmpeg_proc.stdin.write(first)
                    self._ffmpeg_proc.stdin.flush()
                    break
                s.close()
            except OSError:
                time.sleep(0.15)

        if sock is None:
            self._error_lines.append("scrcpy TCP socket connection failed")
            self._stop_event.set()
            with self._cond:
                self._cond.notify_all()
            return

        try:
            sock.settimeout(2.0)
            while not self._stop_event.is_set():
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                try:
                    if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                        self._ffmpeg_proc.stdin.write(chunk)
                        self._ffmpeg_proc.stdin.flush()
                    else:
                        break
                except OSError:
                    break
        finally:
            try:
                sock.close()
            except OSError:
                pass
            if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.close()
                except OSError:
                    pass

    def _reader_loop(self) -> None:
        proc = self._ffmpeg_proc
        if proc is None or proc.stdout is None:
            return
        try:
            while not self._stop_event.is_set():
                raw = _read_exact(proc.stdout, self._frame_bytes, self._stop_event)
                if raw is None:
                    break
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))
                now = time.monotonic()
                with self._cond:
                    self._latest_frame = frame
                    self._frame_seq += 1
                    self._frame_times.append(now)
                    self._first_frame_event.set()
                    self._cond.notify_all()
        except OSError:
            pass
        finally:
            with self._cond:
                self._cond.notify_all()

    def _calc_capture_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def isOpened(self) -> bool:
        return (
            not self._closed
            and not self._stop_event.is_set()
            and self._ffmpeg_proc is not None
            and self._ffmpeg_proc.poll() is None
        )

    def read(self) -> tuple[bool, np.ndarray | None]:
        deadline = time.monotonic() + 0.5
        with self._cond:
            while (
                self._frame_seq == self._last_read_seq
                and not self._stop_event.is_set()
                and self.isOpened()
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._cond.wait(timeout=remaining)

            if self._frame_seq > self._last_read_seq and self._latest_frame is not None:
                self._last_read_seq = self._frame_seq
                return True, self._latest_frame
            return False, None

    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop_id == cv2.CAP_PROP_FPS:
            return self._calc_capture_fps()
        return 0.0

    def get_error_summary(self) -> str:
        return " | ".join(self._error_lines)

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()

        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass

        for proc in (self._ffmpeg_proc, self._adb_proc):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=1.0)

        for proc in (self._ffmpeg_proc, self._adb_proc):
            if proc is not None:
                for stream in (proc.stdin, proc.stdout, proc.stderr):
                    if stream is not None:
                        try:
                            stream.close()
                        except OSError:
                            pass

        startup_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            subprocess.run(
                self._adb_cmd("forward", "--remove", f"tcp:{self.port}"),
                capture_output=True,
                timeout=2.0,
                creationflags=startup_flags,
            )
        except Exception:
            pass

        for t in self._threads:
            if t.is_alive() and threading.current_thread() is not t:
                t.join(timeout=1.0)


def open_scrcpy_usb_camera(
    width: int = DEFAULT_MAX_CAMERA_WIDTH,
    height: int = DEFAULT_MAX_CAMERA_HEIGHT,
    fps: int = 30,
    facing: str = "back",
    camera_id: str | None = None,
    serial: str | None = None,
    log_prefix: str | None = "[CAMERA]",
) -> ScrcpyUsbCapture | None:
    """Android USB(scrcpy) 초저지연 카메라 캡처를 연다."""
    scrcpy_exe = _find_scrcpy_exe()
    ffmpeg_exe = _find_ffmpeg_exe()
    if not scrcpy_exe or not ffmpeg_exe:
        if log_prefix:
            print(f"{log_prefix} scrcpy 또는 ffmpeg를 찾을 수 없습니다 (scrcpy={scrcpy_exe}, ffmpeg={ffmpeg_exe}).", flush=True)
        return None
    adb_exe, server_jar = _resolve_adb_and_scrcpy_server(scrcpy_exe)
    if not adb_exe or server_jar is None:
        if log_prefix:
            print(f"{log_prefix} adb 또는 scrcpy-server를 찾을 수 없습니다.", flush=True)
        return None

    if serial is None:
        dev = detect_android_usb_device()
        if not dev:
            if log_prefix:
                print(f"{log_prefix} USB로 연결된 Android 기기를 찾을 수 없습니다.", flush=True)
            return None
        if dev["state"] != "device":
            if log_prefix:
                print(f"{log_prefix} Android 기기({dev['serial']}) USB 디버깅 승인이 필요합니다 (state={dev['state']}).", flush=True)
            return None
        serial = dev["serial"]

    if log_prefix:
        print(f"{log_prefix} Attempting Android USB scrcpy ({serial}, {width}x{height}@{fps}, facing={facing})...", flush=True)
    cap = ScrcpyUsbCapture(
        scrcpy_exe=scrcpy_exe,
        ffmpeg_exe=ffmpeg_exe,
        adb_exe=adb_exe,
        server_jar=server_jar,
        width=width,
        height=height,
        fps=fps,
        facing=facing,
        camera_id=camera_id,
        serial=serial,
    )
    if cap.open():
        if log_prefix:
            print(f"{log_prefix} Connected Android USB scrcpy ({serial}, {width}x{height}@{fps}).", flush=True)
        return cap
    err = cap.get_error_summary()
    if err and log_prefix:
        print(f"{log_prefix} Android USB scrcpy failed: {err}", flush=True)
    return None


def scan_available_cameras(max_index: int = 4) -> dict:
    """사용 가능한 카메라 목록과 GoPro USB 상태를 점검해 진단 정보를 반환한다."""
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

        if dev_info and dev_info["kind"] == "none":
            cameras.append({
                "index": idx,
                "name": dev_name,
                "backend": "DSHOW",
                "resolution": "Inactive device",
                "active": False,
                "status_tag": "Inactive (none)",
            })
            continue

        if dev_info and dev_info["is_dummy_virtual"]:
            cameras.append({
                "index": idx,
                "name": dev_name,
                "backend": "DSHOW",
                "resolution": "Virtual camera",
                "active": False,
                "status_tag": "Virtual standby",
            })
            continue

        if dev_info and dev_info["is_gopro"] and _find_ffmpeg_exe():
            cap = open_ffmpeg_dshow_camera(
                dev_name,
                width=DEFAULT_MAX_CAMERA_WIDTH,
                height=DEFAULT_MAX_CAMERA_HEIGHT,
                startup_timeout=3.5,
                log_prefix=None,
            )
            if cap is not None:
                try:
                    ok, frame = cap.read()
                    if ThreadedWebcam._valid_frame(ok, frame):
                        raw_h, raw_w = frame.shape[:2]
                        out_w, out_h = compute_downscaled_size(raw_w, raw_h)
                        active = _is_active_nonblack_frame(frame)
                        res_label = (
                            f"{raw_w}x{raw_h}"
                            if (raw_w, raw_h) == (out_w, out_h)
                            else f"{raw_w}x{raw_h} -> {out_w}x{out_h}"
                        )
                        cameras.append({
                            "index": idx,
                            "name": dev_name,
                            "backend": "FFmpeg-DSHOW",
                            "resolution": res_label,
                            "active": active,
                            "status_tag": "Active stream" if active else "Standby frame",
                        })
                finally:
                    cap.release()
            continue

        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        try:
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                for _ in range(15):
                    ok, frame = cap.read()
                    if ThreadedWebcam._valid_frame(ok, frame):
                        raw_h, raw_w = frame.shape[:2]
                        out_w, out_h = compute_downscaled_size(raw_w, raw_h)
                        active = _is_active_nonblack_frame(frame)
                        res_label = (
                            f"{raw_w}x{raw_h}"
                            if (raw_w, raw_h) == (out_w, out_h)
                            else f"{raw_w}x{raw_h} -> {out_w}x{out_h}"
                        )
                        status_tag = "Active stream" if active else "Standby frame"
                        cameras.append({
                            "index": idx,
                            "name": dev_name,
                            "backend": "DSHOW",
                            "resolution": res_label,
                            "active": active,
                            "status_tag": status_tag,
                        })
                        break
                    time.sleep(0.06)
        except Exception:
            pass
        finally:
            cap.release()

    android_dev = detect_android_usb_device()
    return {
        "cameras": cameras,
        "gopro_usb_ip": gopro_ip,
        "gopro_status": gopro_status,
        "android_usb": android_dev,
        "scrcpy_available": bool(_find_scrcpy_exe() and _find_ffmpeg_exe()),
        "system_proxies_detected": sys_proxies,
        "no_proxy_configured": os.environ.get("NO_PROXY", ""),
    }


class ThreadedWebcam:
    """모바일 IP 스트림(HTTP/RTSP), 모바일 USB 웹캠(Iriun/DroidCam), GoPro DirectShow 및 일반 웹캠 통합 캡처 클래스."""

    def __init__(
        self,
        cam_idx=1,
        max_width=DEFAULT_MAX_CAMERA_WIDTH,
        max_height=DEFAULT_MAX_CAMERA_HEIGHT,
        camera_url: str | None = None,
        rotation: int = 0,
    ):
        configure_local_and_gopro_no_proxy()
        self.cap = None
        self.active_cam_idx = cam_idx
        self.camera_url = (camera_url or "").strip() or None
        self.max_width = int(max_width) if max_width else DEFAULT_MAX_CAMERA_WIDTH
        self.max_height = int(max_height) if max_height else DEFAULT_MAX_CAMERA_HEIGHT
        self.rotation = normalize_rotation(rotation)
        self.raw_resolution = (0, 0)
        self.output_resolution = (0, 0)
        self.gopro_ip = None if self.camera_url else detect_gopro_usb_ip()
        self._using_direct_udp = False
        self.stream_type = ""

        print("[CAMERA] Connecting video input...", flush=True)

        # ---------------------------------------------------------
        # 0차 시도: scrcpy:// Android USB 또는 모바일 IP 카메라 URL 직접 연결
        # ---------------------------------------------------------
        if self.camera_url:
            if self.camera_url.lower().startswith("scrcpy://"):
                target = self.camera_url[len("scrcpy://"):].strip().lower()
                facing = "front" if target == "front" else "back"
                cam_id = target.split("=", 1)[1].strip() if target.startswith("id=") else None
                out_w, out_h = compute_downscaled_size(
                    self.max_width, self.max_height, self.max_width, self.max_height
                )
                temp_cap = open_scrcpy_usb_camera(
                    width=out_w,
                    height=out_h,
                    fps=30,
                    facing=facing,
                    camera_id=cam_id,
                    log_prefix="[CAMERA]",
                )
                if temp_cap is not None:
                    grabbed, raw_frame = temp_cap.read()
                    if self._valid_frame(grabbed, raw_frame):
                        self._assign_initial_stream(
                            temp_cap,
                            cam_idx,
                            f"Android USB scrcpy ({self.camera_url})",
                            raw_frame,
                        )
                    else:
                        temp_cap.release()
                if self.cap is None:
                    raise RuntimeError(
                        f"Android USB(scrcpy) 카메라를 열 수 없습니다 ({self.camera_url}). "
                        "USB 디버깅 허용 및 scrcpy/ffmpeg 설치 상태를 확인하세요."
                    )
            else:
                print(f"[CAMERA] Attempting IP/Mobile stream: {self.camera_url}", flush=True)
                temp_cap = cv2.VideoCapture(self.camera_url, cv2.CAP_FFMPEG)
                if not temp_cap.isOpened():
                    temp_cap.release()
                    temp_cap = cv2.VideoCapture(self.camera_url)
                if temp_cap.isOpened():
                    temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    for _ in range(30):
                        grabbed, raw_frame = temp_cap.read()
                        if self._valid_frame(grabbed, raw_frame):
                            self._assign_initial_stream(
                                temp_cap,
                                cam_idx,
                                f"IP/Mobile Stream ({self.camera_url})",
                                raw_frame,
                            )
                            break
                        time.sleep(0.1)
                    if self.cap is None:
                        temp_cap.release()
                if self.cap is None:
                    raise RuntimeError(f"모바일/IP 카메라 스트림을 열 수 없습니다: {self.camera_url}")

        dshow_devices = [] if self.cap is not None else list_dshow_video_devices()
        dev_by_idx = {d["index"]: d for d in dshow_devices}
        gopro_dshow_idx = next((d["index"] for d in dshow_devices if d["is_gopro"] and d["kind"] == "video"), None)

        # ---------------------------------------------------------
        # 1차 시도: 사용자가 선택한 카메라 번호 우선 (모바일 USB 웹캠·일반 웹캠·GoPro 모두 지원)
        # ---------------------------------------------------------
        candidate_indices = []
        if cam_idx in dev_by_idx and dev_by_idx[cam_idx]["kind"] == "video" and not dev_by_idx[cam_idx]["is_dummy_virtual"]:
            candidate_indices.append(cam_idx)
        if gopro_dshow_idx is not None and gopro_dshow_idx not in candidate_indices:
            candidate_indices.append(gopro_dshow_idx)

        for target_dshow_idx in candidate_indices:
            if self.cap is not None:
                break
            dev_info = dev_by_idx[target_dshow_idx]
            dev_name = dev_info["name"]
            if dev_info["is_gopro"] and _find_ffmpeg_exe():
                if self.gopro_ip:
                    set_gopro_wide_fov(self.gopro_ip)
                # DirectShow에서 720p를 요청하면 GoPro 드라이버가 1080p 중앙을 크롭하는 현상을 막기 위해
                # 항상 1920x1080 풀화각으로 받은 뒤 FFmpeg 내부에서 목표 해상도로 축소(scale)한다
                out_w, out_h = compute_downscaled_size(1920, 1080, self.max_width, self.max_height)
                temp_cap = open_ffmpeg_dshow_camera(
                    dev_name,
                    width=out_w,
                    height=out_h,
                    capture_width=1920,
                    capture_height=1080,
                    startup_timeout=5.0,
                    log_prefix="[CAMERA]",
                )
                if temp_cap is not None:
                    grabbed, raw_frame = temp_cap.read()
                    if self._valid_frame(grabbed, raw_frame):
                        self._assign_initial_stream(
                            temp_cap,
                            target_dshow_idx,
                            f"FFmpeg DirectShow ({dev_name})",
                            raw_frame,
                        )
                    else:
                        temp_cap.release()
            else:
                print(f"[CAMERA] Attempting DirectShow '{dev_name}' (index {target_dshow_idx})...", flush=True)
                temp_cap = cv2.VideoCapture(target_dshow_idx, cv2.CAP_DSHOW)
                if temp_cap.isOpened():
                    temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    temp_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.max_width)
                    temp_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.max_height)
                    got_valid = False
                    for i in range(25):
                        grabbed, raw_frame = temp_cap.read()
                        if self._valid_frame(grabbed, raw_frame):
                            if _is_active_nonblack_frame(raw_frame) or i > 15:
                                got_valid = True
                                self._assign_initial_stream(
                                    temp_cap,
                                    target_dshow_idx,
                                    f"DirectShow ({dev_name})",
                                    raw_frame,
                                )
                                break
                        time.sleep(0.1)
                    if not got_valid:
                        temp_cap.release()

        # ---------------------------------------------------------
        # 2차 시도: DirectShow 연결이 안 되면 GoPro USB UDP 스트림을 직접 연다
        # ---------------------------------------------------------
        if self.cap is None and self.gopro_ip:
            target_res = "480" if self.max_height < 720 else "720"
            print(f"[CAMERA] Attempting direct GoPro USB {target_res}p stream from {self.gopro_ip}...", flush=True)
            if start_gopro_usb_webcam(self.gopro_ip, res=target_res, fov="0"):
                udp_url = "udp://@0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000"
                temp_cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
                if temp_cap.isOpened():
                    temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    for _ in range(30):
                        grabbed, raw_frame = temp_cap.read()
                        if self._valid_frame(grabbed, raw_frame):
                            self._assign_initial_stream(
                                temp_cap,
                                cam_idx,
                                f"Direct USB UDP ({self.gopro_ip})",
                                raw_frame,
                                using_direct_udp=True,
                            )
                            break
                        time.sleep(0.1)
                    if self.cap is None:
                        temp_cap.release()

        # ---------------------------------------------------------
        # 2.5차 시도: USB 연결된 Android 폰(scrcpy) 자동 감지 연결
        # ---------------------------------------------------------
        if self.cap is None:
            android_dev = detect_android_usb_device()
            if android_dev and android_dev["state"] == "device":
                out_w, out_h = compute_downscaled_size(
                    self.max_width, self.max_height, self.max_width, self.max_height
                )
                temp_cap = open_scrcpy_usb_camera(
                    width=out_w,
                    height=out_h,
                    fps=30,
                    facing="back",
                    serial=android_dev["serial"],
                    log_prefix="[CAMERA]",
                )
                if temp_cap is not None:
                    grabbed, raw_frame = temp_cap.read()
                    if self._valid_frame(grabbed, raw_frame):
                        self._assign_initial_stream(
                            temp_cap,
                            cam_idx,
                            f"Android USB scrcpy ({android_dev['serial']})",
                            raw_frame,
                        )
                    else:
                        temp_cap.release()

        # ---------------------------------------------------------
        # Strategy 3: Fallback to any active non-dummy webcam
        # ---------------------------------------------------------
        if self.cap is None:
            print("[CAMERA] Falling back to any available active video device...", flush=True)
            for dev in dshow_devices:
                if dev["kind"] == "video" and not dev["is_dummy_virtual"] and not dev["is_gopro"]:
                    idx = dev["index"]
                    temp_cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                    if temp_cap.isOpened():
                        temp_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        for _ in range(10):
                            grabbed, raw_frame = temp_cap.read()
                            if self._valid_frame(grabbed, raw_frame):
                                self._assign_initial_stream(
                                    temp_cap,
                                    idx,
                                    f"Webcam ({dev['name']})",
                                    raw_frame,
                                )
                                break
                            time.sleep(0.06)
                        if self.cap is not None:
                            break
                        temp_cap.release()

        if self.cap is None:
            raise RuntimeError(
                f"Failed to open video stream from camera (cam_idx={cam_idx}, gopro_ip={self.gopro_ip}). "
                "Please verify that GoPro USB is connected, GoPro Webcam utility is running, and no other app is using the camera."
            )

        self.captured_at = time.perf_counter()
        self._last_valid_at = self.captured_at
        self._capture_error = None
        self.frame_id = 0
        self._scaled_frame_id = 0
        self.started = False
        self.read_lock = threading.Lock()

    def _assign_initial_stream(
        self,
        cap,
        active_cam_idx: int,
        stream_type: str,
        raw_frame: np.ndarray,
        using_direct_udp: bool = False,
    ) -> None:
        """첫 유효 프레임 수신 시 해상도와 초기 스트림 상태를 설정한다."""
        self.cap = cap
        self.active_cam_idx = active_cam_idx
        self.stream_type = stream_type
        self._using_direct_udp = using_direct_udp
        raw_h, raw_w = raw_frame.shape[:2]
        frame = rotate_frame_if_needed(
            downscale_frame_if_needed(raw_frame, self.max_width, self.max_height),
            self.rotation,
        )
        out_h, out_w = frame.shape[:2]
        self.raw_resolution = (raw_w, raw_h)
        self.output_resolution = (out_w, out_h)
        self.grabbed = True
        self.frame = frame
        self._raw_frame = raw_frame
        rot_tag = f", rot={self.rotation}°" if self.rotation else ""
        print(f"[CAMERA] Connected via {stream_type}: {raw_w}x{raw_h} -> {out_w}x{out_h}{rot_tag}", flush=True)

    @staticmethod
    def _valid_frame(grabbed, frame):
        """3채널 컬러 영상 프레임이 정상적으로 읽혔는지 확인한다."""
        return (
            grabbed
            and isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
            and frame.shape[2] == 3
        )

    def start(self):
        """백그라운드 프레임 읽기 스레드를 시작한다."""
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()
        return self

    def update(self):
        """카메라에서 최신 프레임을 계속 읽어 내부 버퍼를 갱신한다."""
        while self.started:
            try:
                grabbed, raw_frame = self.cap.read()
            except cv2.error as exc:
                with self.read_lock:
                    self._capture_error = str(exc)
                break

            if not grabbed or raw_frame is None:
                if hasattr(self.cap, "isOpened") and not self.cap.isOpened():
                    with self.read_lock:
                        self._capture_error = "Camera stream closed unexpectedly."
                    break
                time.sleep(0.005)
                continue

            captured_at = time.perf_counter()

            with self.read_lock:
                self.grabbed = grabbed
                self._raw_frame = raw_frame
                self.captured_at = captured_at
                self.frame_id += 1
                self._last_valid_at = captured_at

            time.sleep(0.001)

    def read(self):
        """다운스케일된 최신 프레임과 성공 여부만 간단히 반환한다."""
        grabbed, frame, _, _ = self.read_sample()
        return grabbed, frame

    def read_sample(self):
        """다운스케일된 프레임 복사본, 캡처 시각, 프레임 번호를 스레드 안전하게 반환한다."""
        with self.read_lock:
            if self._capture_error:
                raise RuntimeError(f"Camera frame capture error: {self._capture_error}")
            raw_frame = getattr(self, "_raw_frame", self.frame)
            if not self.grabbed or raw_frame is None:
                if time.perf_counter() - self._last_valid_at >= 5.0:
                    raise RuntimeError("Camera feed interrupted for >5s. Check GoPro USB connection.")
                return False, None, self.captured_at, self.frame_id

            if self._scaled_frame_id != self.frame_id or self.frame is None:
                self.frame = rotate_frame_if_needed(
                    downscale_frame_if_needed(raw_frame, self.max_width, self.max_height),
                    self.rotation,
                )
                out_h, out_w = self.frame.shape[:2]
                self.output_resolution = (out_w, out_h)
                self._scaled_frame_id = self.frame_id

            return True, self.frame.copy(), self.captured_at, self.frame_id

    def set_rotation(self, rotation: int) -> None:
        """실행 중인 카메라를 재연결하지 않고 출력 회전 각도(0/90/180/270)를 즉시 변경한다."""
        new_rot = normalize_rotation(rotation)
        with self.read_lock:
            if self.rotation != new_rot:
                self.rotation = new_rot
                self._scaled_frame_id = -1
                print(f"[CAMERA] Output rotation updated -> {new_rot}°", flush=True)

    def stop(self):
        self.started = False
        thread = getattr(self, "thread", None)
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        if thread is not None and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=1.5)
        if getattr(self, "_using_direct_udp", False) and getattr(self, "gopro_ip", None):
            stop_gopro_usb_webcam(self.gopro_ip)
