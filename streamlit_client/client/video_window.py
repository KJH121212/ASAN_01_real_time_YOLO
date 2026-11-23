"""Streamlit 클라이언트 PC 화면에 실시간 미리보기 영상을 띄우는 독립 OpenCV 창 모듈."""

from functools import lru_cache
import os
from pathlib import Path
import re
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=4)
def message_font(size):
    """안내 문구 출력에 쓸 한글 지원 폰트를 찾아 불러온다."""
    for path in (
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf",
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def video_image(state):
    """수신된 프레임 위에 경고·준비 안내 문구와 카운트다운 숫자를 그린다."""
    content = state.get("frame")
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) if content else None
    if image is None:
        image = np.zeros((360, 640, 3), np.uint8)
    message = state.get("preview_warning") or state.get("preview_notice") or ""
    received = state.get("frame_received_at")
    if received is not None and time.monotonic() - received > 2:
        message = state.get("preview_warning") or "영상 수신이 지연되고 있습니다. 마지막 수신 영상입니다."
    if not content:
        message = message or "카메라에 연결하고 있습니다."
    if message:
        height, width = image.shape[:2]
        font = message_font(max(12, round(width / 640 * 17)))
        canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        limit = round(width * 0.70) - 16
        lines, line = [], ""
        for character in message:
            if line and draw.textbbox((0, 0), line + character, font=font)[2] > limit:
                lines.append(line)
                line = ""
            line += character
        lines.append(line)
        line_height = font.size + 6 if hasattr(font, "size") else 22
        top = max(0, height - line_height * len(lines) - 12)
        draw.rectangle((0, top, round(width * 0.73), height), fill=(20, 20, 20))
        color = (255, 210, 80) if state.get("preview_warning") else (255, 255, 255)
        for index, line in enumerate(lines):
            draw.text((8, top + 4 + index * line_height), line, font=font, fill=color)

        m = re.search(r"\((\d+)초\)", message)
        if m:
            count_str = m.group(1)
            big_font = message_font(max(48, round(width / 640 * 72)))
            bb = draw.textbbox((0, 0), count_str, font=big_font)
            bw, bh = bb[2] - bb[0], bb[3] - bb[1]
            cx, cy = width // 2, height // 2 - 20
            r = max(bw, bh) // 2 + 24
            draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(0, 0, 0, 180), outline=(0, 255, 120), width=4)
            draw.text((cx - bw // 2, cy - bh // 2 - 6), count_str, font=big_font, fill=(0, 255, 120))

        image = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    return image


class VideoWindow:
    """백그라운드 스레드에서 OpenCV 창을 열어 최신 미리보기 영상을 표시한다."""

    def __init__(self):
        self.client = None
        self.thread = None
        self.lock = threading.Lock()

    def show(self, client):
        with self.lock:
            self.client = client
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._display, daemon=True)
                self.thread.start()

    def _display(self):
        title = "Motion Camera (COCO-23)"
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1120, 630)
            previous = None
            last_orientation = None
            while True:
                with self.lock:
                    client = self.client
                state = client.snapshot()
                received = state.get("frame_received_at")
                signature = (
                    id(state.get("frame")),
                    state.get("preview_warning"),
                    state.get("preview_notice"),
                    received is not None and time.monotonic() - received > 2,
                )
                if signature != previous:
                    img = video_image(state)
                    img_h, img_w = img.shape[:2]
                    orientation = "portrait" if img_h > img_w else "landscape"
                    if state.get("frame") is not None and orientation != last_orientation:
                        if orientation == "portrait":
                            win_h = 860
                            win_w = max(320, round(win_h * img_w / max(1, img_h)))
                        else:
                            win_w = 1120
                            win_h = max(360, round(win_w * img_h / max(1, img_w)))
                        cv2.resizeWindow(title, win_w, win_h)
                        last_orientation = orientation
                    cv2.imshow(title, img)
                    previous = signature
                key = cv2.waitKey(5) & 0xFF
                if key == 27 or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    client.preview_closed.set()
                    break
        except cv2.error as exc:
            with client.lock:
                client.state["preview_warning"] = f"영상 창을 표시할 수 없습니다: {exc}"
        finally:
            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass


video_window = VideoWindow()
