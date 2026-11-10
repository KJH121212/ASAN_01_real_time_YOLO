"""A native video-only window on the PC running Streamlit."""

from functools import lru_cache
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


@lru_cache(maxsize=4)
def message_font(size):
    for path in (Path(os.environ.get('WINDIR', 'C:/Windows'))/'Fonts/malgun.ttf',
                 Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')):
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def video_image(state):
    content = state.get('frame')
    image = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR) if content else None
    if image is None:
        image = np.zeros((360, 640, 3), np.uint8)
    message = state.get('preview_warning') or state.get('preview_notice') or ''
    received = state.get('frame_received_at')
    if received is not None and time.monotonic()-received > 2:
        message = state.get('preview_warning') or '영상 수신이 지연되고 있습니다. 마지막 수신 영상입니다.'
    if not content:
        message = message or '카메라에 연결하고 있습니다.'
    if message:
        height, width = image.shape[:2]
        font = message_font(max(12, round(width/640*17)))
        canvas = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(canvas)
        limit = round(width*.70)-16
        lines, line = [], ''
        for character in message:
            if line and draw.textbbox((0, 0), line+character, font=font)[2] > limit:
                lines.append(line)
                line = ''
            line += character
        lines.append(line)
        line_height = font.size+6 if hasattr(font, 'size') else 22
        top = max(0, height-line_height*len(lines)-12)
        draw.rectangle((0, top, round(width*.73), height), fill=(20, 20, 20))
        color = (255, 210, 80) if state.get('preview_warning') else (255, 255, 255)
        for index, line in enumerate(lines):
            draw.text((8, top+4+index*line_height), line, font=font, fill=color)
        image = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
    return image


class VideoWindow:
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
        title = 'Motion Camera'
        try:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 960, 540)
            previous = None
            while True:
                with self.lock:
                    client = self.client
                state = client.snapshot()
                received = state.get('frame_received_at')
                signature = (id(state.get('frame')), state.get('preview_warning'), state.get('preview_notice'),
                             received is not None and time.monotonic()-received > 2)
                if signature != previous:
                    cv2.imshow(title, video_image(state))
                    previous = signature
                if cv2.waitKey(30) & 0xff == 27:
                    break
                try:
                    if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
        except Exception as exc:
            with self.lock:
                client = self.client
            with client.lock:
                client.state['preview_warning'] = f'영상 창을 열 수 없습니다: {exc}'
        finally:
            try:
                cv2.destroyWindow(title)
            except cv2.error:
                pass


video_window = VideoWindow()
