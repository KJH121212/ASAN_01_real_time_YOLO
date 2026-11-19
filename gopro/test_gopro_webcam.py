"""GoPro 실시간 웹캠 영상 뷰어 및 진단 스크립트.

원인 및 해결 구조:
    Windows의 GoPro Webcam 가상 카메라는 초기 대기 로고(스플래시)에서 실제 카메라
    실시간 영상으로 전환될 때 짧은 포맷/신호 공백이 발생한다. 이때 OpenCV의 기본
    DirectShow 백엔드(cv2.CAP_DSHOW)는 내부 1초 타임아웃에 걸려 영구적으로
    검은 화면(1 FPS) 상태로 멈추는 문제가 있다.
    이를 해결하기 위해 본 스크립트는 번들 FFmpeg 프로세스로 DirectShow 스트림을
    직접 수신하고, 백그라운드 스레드로 파이프를 지연 없이 비워 실시간 영상(30~60 FPS)을
    안정적으로 표시한다.

실행:
    python gopro/test_gopro_webcam.py

종료:
    영상 창에서 'q' 또는 'ESC' 키 입력 (또는 콘솔에서 Ctrl+C)
"""

from collections import deque
import os
from pathlib import Path
import sys
import threading
import time

# 로컬 GoPro USB 통신 시 프록시 우회 및 OpenCV 내부 경고 로그 억제
os.environ["NO_PROXY"] = "localhost,127.0.0.1,172.16.0.0/12,*.local"
os.environ["OPENCV_LOG_LEVEL"] = "OFF"

import cv2
import numpy as np

# 프로젝트 루트 경로 등록
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from network.camera import (
    FFmpegDShowCapture,
    detect_gopro_usb_ip,
    list_dshow_video_devices,
    open_ffmpeg_dshow_camera,
    query_gopro_usb_status,
    start_gopro_usb_webcam,
    stop_gopro_usb_webcam,
)

WINDOW_NAME = "GoPro Live Viewer"
DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 720


class ThreadedOpenCVCapture:
    """일반 OpenCV VideoCapture(UDP/웹캠 폴백)용 백그라운드 스레드 래퍼."""

    def __init__(self, cap: cv2.VideoCapture, source_label: str, gopro_ip: str | None = None) -> None:
        self._cap = cap
        self.source_label = source_label
        self._gopro_ip = gopro_ip

        self._stop_event = threading.Event()
        self._cond = threading.Condition()
        self._latest_frame: np.ndarray | None = None
        self._frame_seq = 0
        self._last_read_seq = 0
        self._frame_times: deque[float] = deque(maxlen=60)
        self._closed = False
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        """OpenCV 내부 버퍼 지연을 없애기 위해 백그라운드에서 프레임을 비운다."""
        while not self._stop_event.is_set() and self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue
            now = time.monotonic()
            with self._cond:
                self._latest_frame = frame
                self._frame_seq += 1
                self._frame_times.append(now)
                self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()

    def _calc_capture_fps(self) -> float:
        if len(self._frame_times) < 2:
            return 0.0
        span = self._frame_times[-1] - self._frame_times[0]
        return (len(self._frame_times) - 1) / span if span > 0 else 0.0

    def isOpened(self) -> bool:
        return not self._closed and self._cap.isOpened()

    def read_latest(self, last_seq: int, timeout: float = 0.25) -> tuple[bool, np.ndarray | None, int, float]:
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._frame_seq == last_seq and not self._stop_event.is_set() and self.isOpened():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None, last_seq, self._calc_capture_fps()
                self._cond.wait(timeout=remaining)

            if self._frame_seq > last_seq and self._latest_frame is not None:
                return True, self._latest_frame.copy(), self._frame_seq, self._calc_capture_fps()
            return False, None, last_seq, self._calc_capture_fps()

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame, seq, _ = self.read_latest(self._last_read_seq, timeout=1.0)
        if ok:
            self._last_read_seq = seq
        return ok, frame

    def get(self, prop_id: int) -> float:
        return float(self._cap.get(prop_id))

    def release(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        self._cap.release()
        if self._gopro_ip:
            # 직접 USB UDP 세션을 연 경우 종료 시 카메라 웹캠 모드도 함께 정지
            stop_gopro_usb_webcam(self._gopro_ip)


def _open_ffmpeg_dshow_gopro(dshow_devices: list[dict]) -> FFmpegDShowCapture | None:
    """DirectShow 목록에서 GoPro Webcam 장치를 찾아 공용 FFmpeg 캡처로 연결한다."""
    gopro_devs = [
        dev for dev in dshow_devices
        if dev.get("is_gopro") and dev.get("kind") in ("video", "none", None)
    ]
    for dev in gopro_devs:
        cap = open_ffmpeg_dshow_camera(
            dev["name"],
            width=DEFAULT_WIDTH,
            height=DEFAULT_HEIGHT,
            startup_timeout=5.0,
            log_prefix="[Step 1]",
        )
        if cap is not None:
            return cap
    return None


def _open_direct_usb_udp() -> ThreadedOpenCVCapture | None:
    """GoPro Webcam 앱이 없거나 DirectShow 실패 시 USB HTTP/UDP 스트림 직접 연결을 시도한다."""
    gopro_ip = detect_gopro_usb_ip()
    if not gopro_ip:
        return None

    print(f"[Step 2] GoPro USB 네트워크 직접 연결 시도 ({gopro_ip})...")
    status = query_gopro_usb_status(gopro_ip, timeout=1.0)
    print(f"  GoPro 상태 응답: {status}")

    if not start_gopro_usb_webcam(gopro_ip, res="720", fov="4"):
        print("  -> GoPro 웹캠 시작 명령(HTTP) 실패")
        return None

    udp_url = "udp://@0.0.0.0:8554?overrun_nonfatal=1&fifo_size=50000000"
    cap = cv2.VideoCapture(udp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("  -> UDP 8554 수신 실패 (GoPro Webcam 트레이 앱이 포트를 점유 중일 수 있음)")
        stop_gopro_usb_webcam(gopro_ip)
        return None

    # 첫 프레임 도착 확인
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  -> GoPro USB UDP 연결 성공! ({frame.shape[1]}x{frame.shape[0]})")
            return ThreadedOpenCVCapture(cap, f"GoPro USB UDP ({gopro_ip})", gopro_ip=gopro_ip)
        time.sleep(0.05)

    cap.release()
    stop_gopro_usb_webcam(gopro_ip)
    print("  -> UDP 스트림 연결 타임아웃")
    return None


def _open_fallback_webcam(dshow_devices: list[dict]) -> ThreadedOpenCVCapture | None:
    """GoPro 연결이 모두 실패한 경우 일반 로컬 웹캠(예: 노트북 내장 카메라)으로 폴백한다."""
    print("[Step 3] 기타 로컬 웹캠 장치 탐색 중...")
    for dev in dshow_devices:
        if dev.get("is_gopro"):
            continue
        idx = int(dev["index"])
        name = dev["name"]
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"  -> '{name}' (index {idx}) 연결 성공!")
            return ThreadedOpenCVCapture(cap, f"Webcam ({name})")
        cap.release()
    return None


def connect_gopro_camera():
    """사용 가능한 최적의 경로로 GoPro(또는 폴백 웹캠)를 연결한다."""
    print("=== DirectShow Video Devices ===")
    dshow_devices = list_dshow_video_devices()
    if not dshow_devices:
        print("  (감지된 DirectShow 비디오 장치 없음)")
    for dev in dshow_devices:
        tag = " [GoPro]" if dev.get("is_gopro") else ""
        print(f"  - [{dev['index']}] {dev['name']} ({dev.get('kind', 'video')}){tag}")
    print()

    cap = _open_ffmpeg_dshow_gopro(dshow_devices)
    if cap is not None:
        return cap

    cap = _open_direct_usb_udp()
    if cap is not None:
        return cap

    cap = _open_fallback_webcam(dshow_devices)
    if cap is not None:
        return cap

    return None


def _classify_frame_state(frame: np.ndarray, motion_diff: float) -> tuple[str, tuple[int, int, int]]:
    """현재 프레임의 밝기/움직임으로 신호 상태(LIVE / BLACK / SPLASH)를 판별한다."""
    mean_val = float(frame.mean())
    if mean_val < 5.0:
        return "BLACK (TRANSITIONING)", (0, 165, 255)
    # GoPro 대기 로고는 밝기 약 185~225 사이의 정지 화면으로 나타남
    if 185.0 <= mean_val <= 225.0 and motion_diff < 0.25:
        return "SPLASH / STATIC", (0, 255, 255)
    return "LIVE", (0, 255, 0)


def _draw_hud(
    frame: np.ndarray,
    source_label: str,
    state_text: str,
    state_color: tuple[int, int, int],
    capture_fps: float,
    display_fps: float,
) -> np.ndarray:
    """OpenCV 창 상단에 상태 표시줄(ASCII 영문)을 오버레이한다."""
    vis = frame.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(vis, (0, 0), (w, 42), (0, 0, 0), -1)
    # OpenCV 기본 폰트는 한글을 렌더링하지 못하므로 창 HUD만 영문 ASCII 사용
    info = f"[{state_text}] {source_label} | {w}x{h} | Cap: {capture_fps:.1f} FPS | UI: {display_fps:.1f} FPS"
    cv2.putText(vis, info, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, state_color, 2, cv2.LINE_AA)
    cv2.putText(
        vis,
        "Press 'q' or 'ESC' to exit",
        (max(10, w - 210), 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    return vis


def main() -> None:
    print("=" * 56)
    print("  GoPro Live Webcam Viewer & Diagnostics (FFmpeg Pipe)")
    print("  영상 창에서 'q' 또는 'ESC' 키를 누르면 종료됩니다.")
    print("=" * 56)

    cap = connect_gopro_camera()
    if cap is None:
        print("\n[오류] 사용 가능한 GoPro 또는 웹캠 장치를 열지 못했습니다.")
        print("  - GoPro 전원이 켜져 있고 USB 케이블로 PC에 연결되어 있는지 확인하세요.")
        print("  - 트레이의 'GoPro Webcam' 앱이 실행 중인지 확인하세요.")
        return

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DEFAULT_WIDTH, DEFAULT_HEIGHT)

    last_seq = 0
    prev_small: np.ndarray | None = None
    last_frame: np.ndarray | None = None
    ui_times: deque[float] = deque(maxlen=60)

    start_time = time.monotonic()
    last_new_frame_time = start_time
    black_since: float | None = None
    live_announced = False
    black_warned = False

    print("\n영상 수신 시작... (초기 GoPro 로고 화면에서 실영상 전환 시 2~5초 걸릴 수 있습니다)")

    try:
        while True:
            ok, frame, seq, capture_fps = cap.read_latest(last_seq, timeout=0.25)
            now = time.monotonic()

            if ok and frame is not None:
                last_seq = seq
                last_new_frame_time = now
                last_frame = frame
                ui_times.append(now)

                # 프레임 간 변화량 계산(저해상도 다운샘플로 부하 최소화)
                small = cv2.resize(frame, (160, 90))
                motion_diff = 0.0 if prev_small is None else float(cv2.absdiff(small, prev_small).mean())
                prev_small = small

                state_text, state_color = _classify_frame_state(frame, motion_diff)
                if state_text == "LIVE":
                    black_since = None
                    if not live_announced:
                        live_announced = True
                        elapsed = now - start_time
                        print(f"[LIVE] GoPro 실시간 영상 전환 완료 ({elapsed:.1f}s, {capture_fps:.1f} FPS)")
                elif state_text.startswith("BLACK"):
                    if black_since is None:
                        black_since = now
                    elif (now - black_since) > 8.0 and not black_warned:
                        black_warned = True
                        print(
                            "\n[진단 경고] 8초 이상 검은 화면이 지속되고 있습니다.\n"
                            "  - GoPro LCD에 '웹캠 진행중'이 표시되는지 확인하세요.\n"
                            "  - GoPro Webcam 트레이 아이콘 -> 'Show Preview'가 켜져 있다면 닫아 주세요."
                        )
                else:
                    black_since = None

                display_fps = 0.0
                if len(ui_times) >= 2 and (ui_times[-1] - ui_times[0]) > 0:
                    display_fps = (len(ui_times) - 1) / (ui_times[-1] - ui_times[0])

                vis = _draw_hud(frame, cap.source_label, state_text, state_color, capture_fps, display_fps)
                cv2.imshow(WINDOW_NAME, vis)
            else:
                # 전환 대기 중에도 창 응답성 유지 및 마지막 프레임 표시
                if last_frame is not None:
                    vis = _draw_hud(
                        last_frame,
                        cap.source_label,
                        "WAITING STREAM...",
                        (0, 165, 255),
                        capture_fps,
                        0.0,
                    )
                    cv2.imshow(WINDOW_NAME, vis)

                if not cap.isOpened():
                    print("\n[오류] 카메라 스트림 프로세스가 종료되었습니다.")
                    if hasattr(cap, "get_stderr_summary"):
                        err = cap.get_stderr_summary()
                        if err:
                            print(f"  FFmpeg 상세 오류: {err}")
                    break

                if (now - last_new_frame_time) > 12.0:
                    print("\n[오류] 12초 이상 새 프레임이 수신되지 않아 연결을 종료합니다.")
                    break

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("\n사용자가 종료 키를 입력했습니다.")
                break

            # 창 닫기(X) 버튼 클릭 감지
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                print("\n영상 창이 닫혔습니다.")
                break
    except KeyboardInterrupt:
        print("\nCtrl+C 입력으로 종료합니다.")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("카메라 연결을 정상적으로 종료했습니다.")


if __name__ == "__main__":
    main()
