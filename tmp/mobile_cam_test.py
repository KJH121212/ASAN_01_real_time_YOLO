"""모바일 카메라 연결 실험 스크립트 (1: 유선 scrcpy 카메라 / 2: Windows 휴대폰과 연결 안정화).

사용 예시:
    # 1) 유선 USB (scrcpy 카메라 직접 스트리밍 -> FFmpeg -> OpenCV)
    python tmp/mobile_cam_test.py --mode scrcpy --size 1280x720 --fps 30

    # 2) 안드로이드 폰 카메라 렌즈(광각/초광각 ID) 목록 확인
    python tmp/mobile_cam_test.py --mode scrcpy --list-cameras

    # 3) Windows "휴대폰과 연결 (Phone Link)" 가상 카메라 안정화 연결 (MSMF 우선 + 자동 재연결)
    python tmp/mobile_cam_test.py --mode phonelink --index 1

    # 4) 현재 PC의 모든 카메라(DirectShow + MSMF) 번호/이름 스캔
    python tmp/mobile_cam_test.py --scan
"""

import argparse
from collections import deque
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import threading
import time

os.environ.setdefault("OPENCV_LOG_LEVEL", "OFF")
import cv2
import numpy as np

try:
    cv2.setLogLevel(0)
except Exception:
    pass

TMP_DIR = Path(__file__).resolve().parent
ROOT_DIR = TMP_DIR.parent


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    for rel in ("socket_server/configs/bin/ffmpeg.exe", "streamlit_client/configs/bin/ffmpeg.exe"):
        p = ROOT_DIR / rel
        if p.is_file():
            return str(p)
    return None


def find_scrcpy() -> str | None:
    exe = shutil.which("scrcpy")
    if exe:
        return exe
    local_app = Path(os.environ.get("LOCALAPPDATA", ""))
    candidates = [
        local_app / "Microsoft" / "WinGet" / "Links" / "scrcpy.exe",
        TMP_DIR / "scrcpy" / "scrcpy.exe",
        ROOT_DIR / "scrcpy" / "scrcpy.exe",
        Path("C:/scrcpy/scrcpy.exe"),
    ]
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    winget_pkgs = local_app / "Microsoft" / "WinGet" / "Packages"
    if winget_pkgs.is_dir():
        for found in winget_pkgs.glob("Genymobile.scrcpy*/**/scrcpy.exe"):
            if found.is_file():
                return str(found)
    for found in TMP_DIR.glob("scrcpy-*/scrcpy.exe"):
        if found.is_file():
            return str(found)
    return None


def list_dshow_devices() -> list[dict]:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        return []
    try:
        proc = subprocess.run(
            [ffmpeg, "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
            capture_output=True, text=True, encoding="utf-8", errors="ignore", timeout=3.0,
        )
        output = proc.stderr or ""
    except Exception:
        return []
    devices = []
    for m in re.finditer(r'\[dshow\s+@\s+[0-9a-fA-F]+\]\s+"([^"]+)"\s+\((video|none)\)', output):
        devices.append({"index": len(devices), "name": m.group(1), "kind": m.group(2)})
    return devices


def scan_cameras(max_index: int = 6):
    """DirectShow 장치 목록과 MSMF / DSHOW 백엔드별 열림 여부를 점검한다."""
    print("==========================================================")
    print("  Windows 카메라 스캔 (Phone Link / 웹캠 / GoPro)")
    print("==========================================================")
    dshow = list_dshow_devices()
    if dshow:
        print("[DirectShow 등록 장치]")
        for d in dshow:
            print(f"  index {d['index']}: {d['name']} ({d['kind']})")
    else:
        print("[DirectShow 등록 장치] 없음")

    print("\n[OpenCV MSMF 백엔드 점검 (Windows 휴대폰과 연결은 MSMF 권장)]")
    for idx in range(max_index):
        cap = cv2.VideoCapture(idx, cv2.CAP_MSMF)
        if cap.isOpened():
            ok, frame = cap.read()
            if ok and frame is not None:
                h, w = frame.shape[:2]
                mean_val = float(frame[::8, ::8].mean())
                state = "활성 영상" if mean_val > 5.0 else "검은 대기 화면"
                print(f"  MSMF index {idx}: OK ({w}x{h}, {state})")
            else:
                print(f"  MSMF index {idx}: 열림 (초기 프레임 대기 필요)")
            cap.release()
    print("==========================================================")


def _read_exact(stream, num_bytes: int, stop_event: threading.Event) -> bytearray | None:
    buf = bytearray(num_bytes)
    view = memoryview(buf)
    received = 0
    while received < num_bytes and not stop_event.is_set():
        chunk = stream.read(num_bytes - received)
        if not chunk:
            return None
        n = len(chunk)
        view[received:received + n] = chunk
        received += n
    return buf if received == num_bytes else None


# ==============================================================================
# 모드 1: 유선 USB scrcpy-server 직접 스트리밍 (ADB raw_stream TCP -> FFmpeg -> OpenCV)
# ==============================================================================
def _get_scrcpy_version(scrcpy_exe: str) -> str:
    try:
        out = subprocess.check_output([scrcpy_exe, "--version"], text=True, encoding="utf-8", errors="ignore", timeout=3.0)
        m = re.search(r"scrcpy\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)", out)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "5.0.1"


def _adb_cmd(adb_exe: str, serial: str | None, *args: str) -> list[str]:
    cmd = [adb_exe]
    if serial:
        cmd.extend(["-s", serial])
    cmd.extend(args)
    return cmd


def _setup_scrcpy_wifi(adb_exe: str, wifi_ip: str | None = None) -> str | None:
    """USB 연결된 안드로이드 폰의 Wi-Fi IP를 자동 감지해 ADB TCP/IP(5555) 무선 모드로 전환한다."""
    out = subprocess.check_output([adb_exe, "devices"], text=True, encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in out.splitlines()[1:] if ln.strip() and "\tdevice" in ln]
    serials = [ln.split("\t")[0] for ln in lines]

    if wifi_ip:
        target = wifi_ip if ":" in wifi_ip else f"{wifi_ip}:5555"
        res = subprocess.run([adb_exe, "connect", target], capture_output=True, text=True)
        print(f"[WIFI] adb connect {target} -> {res.stdout.strip() or res.stderr.strip()}", flush=True)
        return target if "connected" in (res.stdout + res.stderr).lower() else None

    # 이미 무선 ADB(:5555 등 IP:PORT)로 붙어 있는 기기가 있으면 바로 재사용
    for s in serials:
        if ":" in s:
            print(f"[WIFI] 기존 무선 ADB 장치 사용: {s}", flush=True)
            return s

    usb_serials = [s for s in serials if ":" not in s]
    if not usb_serials:
        print("[ERROR] 무선 전환을 위해 최초 1회 USB 케이블로 안드로이드 폰을 연결하거나 --wifi-ip 를 입력하세요.")
        return None

    usb_serial = usb_serials[0]
    ip_out = subprocess.check_output(
        _adb_cmd(adb_exe, usb_serial, "shell", "ip", "-f", "inet", "addr", "show", "wlan0"),
        text=True, encoding="utf-8", errors="ignore",
    )
    m = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)/", ip_out)
    if not m:
        print("[ERROR] 폰의 Wi-Fi IP(wlan0)를 찾을 수 없습니다. 스마트폰이 PC와 같은 Wi-Fi에 연결되어 있는지 확인하세요.")
        return None

    phone_ip = m.group(1)
    target = f"{phone_ip}:5555"
    print(f"[WIFI] 폰 Wi-Fi IP 감지: {phone_ip} -> adb tcpip 5555 전환 중...", flush=True)
    subprocess.run(_adb_cmd(adb_exe, usb_serial, "tcpip", "5555"), capture_output=True)
    time.sleep(1.2)
    res = subprocess.run([adb_exe, "connect", target], capture_output=True, text=True)
    print(f"[WIFI] {res.stdout.strip() or res.stderr.strip()}", flush=True)
    if "connected" not in (res.stdout + res.stderr).lower():
        print("[ERROR] 무선 ADB 연결 실패. PC와 폰이 같은 Wi-Fi에 있는지 확인하세요.")
        return None
    print(f"[WIFI] 무선 연결 성공 ({target})! 이제 USB 케이블을 뽑아도 무선으로 동작합니다.", flush=True)
    return target


def run_scrcpy_camera(
    size: str = "1280x720",
    fps: int = 30,
    facing: str = "back",
    camera_id: str = None,
    list_only: bool = False,
    wireless: bool = False,
    wifi_ip: str = None,
):
    scrcpy = find_scrcpy()
    if not scrcpy:
        print("[ERROR] scrcpy.exe를 찾을 수 없습니다.")
        print("  해결 방법 1 (PowerShell): winget install Genymobile.scrcpy")
        print(f"  해결 방법 2: https://github.com/Genymobile/scrcpy/releases 에서 win64 zip을 받아")
        print(f"               {TMP_DIR / 'scrcpy'} 폴더에 압축 해제하세요.")
        return

    if list_only:
        print(f"[INFO] 연결된 안드로이드 카메라 목록 조회 중... ({scrcpy})")
        subprocess.run([scrcpy, "--list-cameras"])
        return

    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[ERROR] ffmpeg.exe를 찾을 수 없습니다.")
        return

    scrcpy_dir = Path(scrcpy).resolve().parent
    adb_exe = str(scrcpy_dir / "adb.exe") if (scrcpy_dir / "adb.exe").is_file() else (shutil.which("adb") or "adb")
    server_jar = scrcpy_dir / "scrcpy-server"
    if not server_jar.is_file():
        print(f"[ERROR] scrcpy-server 파일을 찾을 수 없습니다: {server_jar}")
        return

    serial = None
    if wireless:
        serial = _setup_scrcpy_wifi(adb_exe, wifi_ip=wifi_ip)
        if not serial:
            return
    else:
        out = subprocess.check_output([adb_exe, "devices"], text=True, encoding="utf-8", errors="ignore")
        usb_serials = [
            ln.split("\t")[0]
            for ln in out.splitlines()[1:]
            if "\tdevice" in ln and ":" not in ln.split("\t")[0]
        ]
        serial = usb_serials[0] if usb_serials else None

    scrcpy_ver = _get_scrcpy_version(scrcpy)
    w, h = [int(x) for x in size.lower().split("x")]
    port = 27183
    conn_label = f"Wi-Fi ({serial})" if wireless else "USB"

    # 1) scrcpy-server 푸시 및 ADB TCP 포트 포워딩
    print(f"[SCRCPY] v{scrcpy_ver} ({conn_label}) 서버 푸시 및 포트 포워딩(tcp:{port})...", flush=True)
    push = subprocess.run(
        _adb_cmd(adb_exe, serial, "push", str(server_jar), "/data/local/tmp/scrcpy-server.jar"),
        capture_output=True, text=True,
    )
    if push.returncode != 0:
        print(f"[ERROR] adb push 실패 (USB/Wi-Fi 디버깅 연결 확인): {push.stderr.strip()}")
        return
    subprocess.run(_adb_cmd(adb_exe, serial, "forward", "--remove", f"tcp:{port}"), capture_output=True)
    fwd = subprocess.run(
        _adb_cmd(adb_exe, serial, "forward", f"tcp:{port}", "localabstract:scrcpy"),
        capture_output=True, text=True,
    )
    if fwd.returncode != 0:
        print(f"[ERROR] adb forward 실패: {fwd.stderr.strip()}")
        return

    # 2) 폰에서 scrcpy-server를 초저지연(B-frame 제거, 2Mbps) 순수 H.264 스트림으로 실행
    cam_arg = f"camera_id={camera_id}" if camera_id is not None else f"camera_facing={facing}"
    remote_cmd = (
        f"CLASSPATH=/data/local/tmp/scrcpy-server.jar "
        f"app_process / com.genymobile.scrcpy.Server {scrcpy_ver} "
        f"video_source=camera {cam_arg} camera_size={w}x{h} camera_fps={fps} "
        f"video_bit_rate=2000000 video_codec_options=i-frame-interval=1,max-bframes=0,priority=0 "
        f"audio=false control=false "
        f"tunnel_forward=true raw_stream=true send_dummy_byte=false log_level=info"
    )
    startup_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    adb_proc = subprocess.Popen(
        _adb_cmd(adb_exe, serial, "shell", remote_cmd),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=startup_flags,
    )

    ffmpeg_cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "warning",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-flags2", "fast",
        "-max_delay", "0",
        "-probesize", "32768",
        "-analyzeduration", "0",
        "-threads", "1",
        "-f", "h264",
        "-framerate", str(fps),
        "-i", "pipe:0",
        "-fps_mode", "passthrough",
        "-flush_packets", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    ffmpeg_proc = subprocess.Popen(
        ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, creationflags=startup_flags,
    )

    stop_event = threading.Event()
    lock = threading.Lock()
    state = {"frame": None, "seq": 0, "times": deque(maxlen=40)}

    def drain_stream(stream, tag):
        for line in stream:
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                print(f"[{tag}] {text}", flush=True)

    def tcp_to_ffmpeg_loop():
        sock = None
        deadline = time.monotonic() + 6.0
        while not stop_event.is_set() and time.monotonic() < deadline:
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=1.0)
                s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                s.settimeout(2.0)
                # tunnel_forward=true 시 첫 바이트가 들어올 때까지 연결 확인
                first = s.recv(65536)
                if first:
                    sock = s
                    print(f"[SCRCPY] H.264 스트림 수신 시작! ({len(first)} bytes, head={first[:8].hex()})", flush=True)
                    ffmpeg_proc.stdin.write(first)
                    ffmpeg_proc.stdin.flush()
                    break
                s.close()
            except OSError:
                time.sleep(0.15)
        if sock is None:
            print("[ERROR] scrcpy USB 스트림 소켓 연결에 실패했습니다.", flush=True)
            stop_event.set()
            return
        try:
            sock.settimeout(2.0)
            while not stop_event.is_set():
                try:
                    chunk = sock.recv(65536)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                try:
                    ffmpeg_proc.stdin.write(chunk)
                    ffmpeg_proc.stdin.flush()
                except OSError:
                    break
        finally:
            try:
                sock.close()
            except OSError:
                pass
            try:
                ffmpeg_proc.stdin.close()
            except OSError:
                pass

    def reader_loop():
        frame_bytes = w * h * 3
        while not stop_event.is_set():
            raw = _read_exact(ffmpeg_proc.stdout, frame_bytes, stop_event)
            if raw is None:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            now = time.perf_counter()
            with lock:
                state["frame"] = frame
                state["seq"] += 1
                state["times"].append(now)

    threading.Thread(target=drain_stream, args=(adb_proc.stdout, "SCRCPY"), daemon=True).start()
    threading.Thread(target=drain_stream, args=(adb_proc.stderr, "SCRCPY"), daemon=True).start()
    threading.Thread(target=drain_stream, args=(ffmpeg_proc.stderr, "FFMPEG"), daemon=True).start()
    threading.Thread(target=tcp_to_ffmpeg_loop, daemon=True).start()
    threading.Thread(target=reader_loop, daemon=True).start()

    title = f"scrcpy USB Camera ({w}x{h}) - ESC to Quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)
    last_seq = -1

    try:
        while not stop_event.is_set():
            with lock:
                seq = state["seq"]
                frame = state["frame"].copy() if state["frame"] is not None else None
                times = state["times"]
                cur_fps = ((len(times) - 1) / (times[-1] - times[0])) if len(times) >= 2 and times[-1] > times[0] else 0.0

            if frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, "Connecting scrcpy USB camera...", (115, 185),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 220, 255), 2, cv2.LINE_AA)
                cv2.imshow(title, blank)
            elif seq != last_seq:
                cv2.rectangle(frame, (8, 8), (420, 42), (15, 15, 15), -1)
                cv2.putText(frame, f"scrcpy USB {w}x{h} | {cur_fps:.1f} FPS", (16, 31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2, cv2.LINE_AA)
                cv2.imshow(title, frame)
                last_seq = seq

            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop_event.set()
        for proc in (ffmpeg_proc, adb_proc):
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        subprocess.run(_adb_cmd(adb_exe, serial, "forward", "--remove", f"tcp:{port}"), capture_output=True)
        cv2.destroyAllWindows()


# ==============================================================================
# 모드 2: Windows "휴대폰과 연결 (Phone Link)" 가상 카메라 안정화 모드
#   - 핵심 1: MSMF 백엔드 우선 (Phone Link는 UWP 가상 카메라라 DSHOW보다 MSMF가 안정적)
#   - 핵심 2: 폰 카메라 깨우기(Wake-up) 시간 최대 8초 대기 (조기 포기 방지)
#   - 핵심 3: 3초 이상 프레임 정지 시 자동 재연결 워치독
# ==============================================================================
def _open_phonelink_cap(index: int, wake_timeout: float = 8.0):
    """MSMF 우선, 실패 시 DSHOW로 열어 폰 카메라가 깨어날 때까지 여유 있게 대기한다."""
    for backend_name, backend_flag in (("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW)):
        print(f"[PHONELINK] index {index} ({backend_name}) 연결 및 폰 카메라 깨우기 대기 중...", flush=True)
        cap = cv2.VideoCapture(index, backend_flag)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        deadline = time.monotonic() + wake_timeout
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if ok and frame is not None and frame.ndim == 3 and frame.size > 0:
                if float(frame[::8, ::8].mean()) > 5.0:
                    print(f"[PHONELINK] 연결 성공 ({backend_name}, {frame.shape[1]}x{frame.shape[0]})", flush=True)
                    return cap, backend_name, frame
            time.sleep(0.08)
        cap.release()
    return None, "", None


def run_phonelink_camera(index: int = 0, wake_timeout: float = 8.0, stall_timeout: float = 3.0):
    title = f"Windows Phone Link Camera (idx {index}) - ESC to Quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)

    cap = None
    backend_name = ""
    frame_times = deque(maxlen=40)
    last_good_at = time.monotonic()

    try:
        while True:
            if cap is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"Waking Phone Link camera (idx {index})...", (85, 175),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 220, 80), 2, cv2.LINE_AA)
                cv2.putText(blank, "Check 'Phone Link' notification on phone if needed", (70, 215),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.imshow(title, blank)
                cv2.waitKey(1)

                cap, backend_name, first_frame = _open_phonelink_cap(index, wake_timeout=wake_timeout)
                if cap is None:
                    print("[PHONELINK] 연결 재시도 대기 (1.5초)...", flush=True)
                    if (cv2.waitKey(1500) & 0xFF) in (27, ord("q")):
                        break
                    continue
                last_good_at = time.monotonic()
                frame_times.clear()
                frame = first_frame
                ok = True
            else:
                ok, frame = cap.read()

            now = time.monotonic()
            if ok and frame is not None and frame.size > 0:
                last_good_at = now
                frame_times.append(now)
                fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])) if len(frame_times) >= 2 and frame_times[-1] > frame_times[0] else 0.0
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (8, 8), (470, 42), (15, 15, 15), -1)
                cv2.putText(frame, f"PhoneLink ({backend_name}) {w}x{h} | {fps:.1f} FPS", (16, 31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (80, 255, 120), 2, cv2.LINE_AA)
                cv2.imshow(title, frame)
            elif now - last_good_at > stall_timeout:
                print(f"[PHONELINK] {stall_timeout:.1f}초간 프레임 멈춤 감지 -> 자동 재연결 수행", flush=True)
                cap.release()
                cap = None

            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


def _get_local_ipv4_list() -> list[str]:
    ips = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if not ip.startswith("127.") and ip not in ips:
                ips.append(ip)
    except Exception:
        pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            primary = s.getsockname()[0]
            if primary not in ips:
                ips.insert(0, primary)
    except Exception:
        pass
    return ips


# ==============================================================================
# 모드 3: 아이폰 / 안드로이드 공통 초저지연 SRT 리스너 (유선 USB 핫스팟 + 무선 Wi-Fi)
#   - 아이폰/안드로이드 무료 앱: Larix Broadcaster (또는 아이폰 Moblin)
#   - 앱에서 srt://<PC_IP>:9000 (Latency 50ms) 입력 후 스트리밍 시작
# ==============================================================================
def run_srt_listener(port: int = 9000, size: str = "1280x720", latency_ms: int = 50):
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        print("[ERROR] ffmpeg.exe를 찾을 수 없습니다.")
        return

    w, h = [int(x) for x in size.lower().split("x")]
    ips = _get_local_ipv4_list()
    print("==========================================================")
    print("  아이폰 / 안드로이드 공통 초저지연 SRT 수신기 (USB / Wi-Fi)")
    print("==========================================================")
    print("1) 폰에 무료 앱 'Larix Broadcaster' (또는 아이폰 'Moblin') 실행")
    print("2) 연결(Connection) URL에 아래 주소 중 하나를 입력:")
    for ip in ips:
        tag = " [아이폰 USB 핫스팟 대역]" if ip.startswith("172.20.10.") else ""
        print(f"     srt://{ip}:{port}{tag}")
    print(f"3) 앱 설정: Mode=Caller, Latency={latency_ms}ms, 해상도 720p, 30fps")
    print("4) 스트리밍(빨간 버튼) 시작 -> PC 창에 바로 영상 표시 (ESC 종료)")
    print("==========================================================", flush=True)

    srt_url = f"srt://0.0.0.0:{port}?mode=listener&latency={latency_ms * 1000}&transtype=live"
    ffmpeg_cmd = [
        ffmpeg,
        "-hide_banner", "-loglevel", "warning",
        "-fflags", "nobuffer+discardcorrupt",
        "-flags", "low_delay",
        "-flags2", "fast",
        "-max_delay", "0",
        "-probesize", "65536",
        "-analyzeduration", "100000",
        "-threads", "1",
        "-i", srt_url,
        "-fps_mode", "passthrough",
        "-vf", f"scale={w}:{h}",
        "-flush_packets", "1",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-",
    ]
    startup_flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=0, creationflags=startup_flags,
    )

    stop_event = threading.Event()
    lock = threading.Lock()
    state = {"frame": None, "seq": 0, "times": deque(maxlen=40)}

    def drain_stderr():
        for line in proc.stderr:
            text = line.decode("utf-8", errors="replace").strip()
            if text:
                print(f"[SRT/FFMPEG] {text}", flush=True)

    def reader_loop():
        frame_bytes = w * h * 3
        while not stop_event.is_set():
            raw = _read_exact(proc.stdout, frame_bytes, stop_event)
            if raw is None:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
            now = time.perf_counter()
            with lock:
                state["frame"] = frame
                state["seq"] += 1
                state["times"].append(now)

    threading.Thread(target=drain_stderr, daemon=True).start()
    threading.Thread(target=reader_loop, daemon=True).start()

    title = f"iPhone/Android SRT Camera (:{port}) - ESC to Quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)
    last_seq = -1

    try:
        while not stop_event.is_set():
            with lock:
                seq = state["seq"]
                frame = state["frame"].copy() if state["frame"] is not None else None
                times = state["times"]
                cur_fps = ((len(times) - 1) / (times[-1] - times[0])) if len(times) >= 2 and times[-1] > times[0] else 0.0

            if frame is None:
                blank = np.zeros((360, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"Waiting for iPhone/Android SRT on :{port}...", (55, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 220, 80), 2, cv2.LINE_AA)
                if ips:
                    cv2.putText(blank, f"srt://{ips[0]}:{port}", (180, 215),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 255, 120), 2, cv2.LINE_AA)
                cv2.imshow(title, blank)
            elif seq != last_seq:
                cv2.rectangle(frame, (8, 8), (460, 42), (15, 15, 15), -1)
                cv2.putText(frame, f"SRT Live {w}x{h} | {cur_fps:.1f} FPS", (16, 31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2, cv2.LINE_AA)
                cv2.imshow(title, frame)
                last_seq = seq

            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stop_event.set()
        try:
            proc.terminate()
            proc.wait(timeout=1.5)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        cv2.destroyAllWindows()


# ==============================================================================
# 모드 4: 아이폰 / 안드로이드 HTTP MJPEG 또는 RTSP URL 직접 수신
# ==============================================================================
def run_url_stream(url: str):
    if not url:
        print("[ERROR] 스트림 URL을 입력하세요. 예: http://172.20.10.1:8081/video 또는 rtsp://...")
        return
    print(f"[URL] 스트림 연결 중: {url}", flush=True)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print(f"[ERROR] 스트림을 열 수 없습니다: {url}")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    title = "iPhone/Android URL Stream - ESC to Quit"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 960, 540)
    frame_times = deque(maxlen=40)
    try:
        while True:
            ok, frame = cap.read()
            if ok and frame is not None:
                now = time.monotonic()
                frame_times.append(now)
                fps = ((len(frame_times) - 1) / (frame_times[-1] - frame_times[0])) if len(frame_times) >= 2 and frame_times[-1] > frame_times[0] else 0.0
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (8, 8), (430, 42), (15, 15, 15), -1)
                cv2.putText(frame, f"URL Stream {w}x{h} | {fps:.1f} FPS", (16, 31),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 255, 120), 2, cv2.LINE_AA)
                cv2.imshow(title, frame)
            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="모바일 카메라 실험 (scrcpy 유선/무선, 아이폰/안드로이드 SRT & URL)")
    parser.add_argument("--mode", choices=("scrcpy", "scrcpy_wifi", "srt", "url", "phonelink"), default="scrcpy")
    parser.add_argument("--scan", action="store_true", help="PC의 모든 카메라(DirectShow + MSMF) 스캔")
    parser.add_argument("--size", default="1280x720", help="카메라 해상도 (예: 1280x720)")
    parser.add_argument("--fps", type=int, default=30, help="목표 FPS")
    parser.add_argument("--facing", choices=("back", "front"), default="back", help="scrcpy 카메라 방향")
    parser.add_argument("--camera-id", default=None, help="scrcpy 특정 카메라 ID")
    parser.add_argument("--wifi-ip", default=None, help="무선 scrcpy 직접 연결할 폰 Wi-Fi IP")
    parser.add_argument("--port", type=int, default=9000, help="SRT 리스너 포트 (기본 9000)")
    parser.add_argument("--url", default="", help="HTTP MJPEG 또는 RTSP 스트림 URL")
    parser.add_argument("--list-cameras", action="store_true", help="scrcpy 안드로이드 카메라 목록 출력")
    parser.add_argument("--index", type=int, default=0, help="Phone Link 가상 카메라 인덱스")
    args = parser.parse_args()

    if args.scan:
        scan_cameras()
        return
    if args.mode == "scrcpy":
        run_scrcpy_camera(args.size, args.fps, args.facing, args.camera_id, args.list_cameras, wireless=False)
    elif args.mode == "scrcpy_wifi":
        run_scrcpy_camera(args.size, args.fps, args.facing, args.camera_id, False, wireless=True, wifi_ip=args.wifi_ip)
    elif args.mode == "srt":
        run_srt_listener(port=args.port, size=args.size)
    elif args.mode == "url":
        run_url_stream(args.url)
    else:
        run_phonelink_camera(args.index)


if __name__ == "__main__":
    main()
