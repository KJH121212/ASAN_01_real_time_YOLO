import argparse
import asyncio
import base64
import json
import os
import time
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import websockets

SKELETON_CONNECTIONS = [
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def put_korean_text(
    img: np.ndarray,
    text: str,
    pos: tuple,
    font_size: int = 20,
    color: tuple = (255, 255, 255),
) -> np.ndarray:
  img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  draw = ImageDraw.Draw(img_pil)
  font_path = "C:/Windows/Fonts/malgun.ttf"
  font = (
      ImageFont.truetype(font_path, font_size)
      if os.path.exists(font_path)
      else ImageFont.load_default()
  )
  draw.text(pos, text, font=font, fill=color)
  return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_quality_color(quality_str: str) -> tuple:
  if quality_str == "PERFECT":
    return (0, 255, 0)
  elif quality_str == "GOOD":
    return (0, 255, 255)
  elif quality_str == "BAD":
    return (0, 0, 255)
  elif quality_str == "CALIBRATING":
    return (255, 200, 0)
  elif quality_str == "FINISHED":
    return (180, 180, 180)
  return (200, 200, 200)


def draw_normalized_mini_box(
    canvas: np.ndarray,
    keypoints: list,
    yolo_detected: bool,
    bx: int,
    by: int,
    box_size: int = 160,
) -> np.ndarray:
  border_color = (0, 255, 0) if yolo_detected else (0, 0, 255)
  # 검정색 미니박스 배경 및 테두리 렌더링
  cv2.rectangle(
      canvas, (bx, by), (bx + box_size, by + box_size), (10, 10, 15), -1
  )
  cv2.rectangle(
      canvas, (bx, by), (bx + box_size, by + box_size), border_color, 2
  )

  if not yolo_detected or not keypoints:
    return put_korean_text(
        canvas, "YOLO 미감지", (bx + 25, by + 50), font_size=14, color=(0, 0, 255)
    )

  canvas = put_korean_text(
      canvas,
      "정규화 스켈레톤",
      (bx + 15, by + 5),
      font_size=11,
      color=(0, 255, 0),
  )

  # 골반 중심 (0, 0) 좌표를 미니박스 중심점으로 매핑
  center_x = bx + box_size // 2
  center_y = by + box_size // 2
  scale_factor = (box_size - 30) / 3.0  # 전신 관절이 미니박스 안에 들어오도록 조정

  kpt_dict = {}
  for kp in keypoints:
    kp_id = kp.get("id")
    kx = kp.get("x", 0.0)
    ky = kp.get("y", 0.0)
    score = kp.get("score", 0.0)

    # 원점 중심 좌표계 -> 검정색 미니박스 픽셀 좌표계 변환
    px = int(center_x + kx * scale_factor)
    py = int(center_y + ky * scale_factor)
    kpt_dict[kp_id] = (px, py, score)

  # 관절 연결선 렌더링
  for p1_id, p2_id in SKELETON_CONNECTIONS:
    if p1_id in kpt_dict and p2_id in kpt_dict:
      x1, y1, s1 = kpt_dict[p1_id]
      x2, y2, s2 = kpt_dict[p2_id]
      if s1 > 0.35 and s2 > 0.35:
        cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

  # 관절 포인트 렌더링
  for idx, (x, y, s) in kpt_dict.items():
    if idx >= 5 and s > 0.35:
      cv2.circle(canvas, (x, y), 4, (0, 165, 255), -1, cv2.LINE_AA)

  return canvas


def draw_center_overlay(
    canvas: np.ndarray,
    text: str,
    font_size: int = 36,
    color: tuple = (0, 255, 255),
) -> np.ndarray:
  h, w, _ = canvas.shape
  overlay = canvas.copy()
  box_h = 100
  cy = h // 2
  cv2.rectangle(overlay, (0, cy - box_h // 2), (w, cy + box_h // 2), (10, 10, 15), -1)
  cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)
  text_x = max(20, (w - (len(text) * font_size // 2)) // 2 - 20)
  return put_korean_text(
      canvas, text, (text_x, cy - font_size // 2 - 5), font_size=font_size, color=color
  )


async def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--player_id", type=str, default="patient_1")
  parser.add_argument("--patient_name", type=str, default="김지후")
  parser.add_argument("--exercise_name", type=str, default="biceps_curl")
  parser.add_argument("--mode", type=str, default="CALIBRATION")
  parser.add_argument("--target_reps", type=int, default=5)
  parser.add_argument("--camera_index", type=int, default=0)
  args = parser.parse_args()

  uri = "ws://127.0.0.1:8080"
  win_w, win_h = 800, 600

  PREP_SECONDS, START_MSG_SECONDS = 3.0, 1.5
  start_timer = time.time()

  latest_keypoints, fps, current_mode = [], 0.0, args.mode
  left_rep, right_rep = 0, 0
  left_progress, right_progress = 0.0, 0.0
  left_state, right_state = "READY", "READY"
  left_quality, right_quality = "CALIBRATING", "CALIBRATING"

  yolo_detected, latest_frame_b64 = False, None
  win_title = f"AI Motion Viewer - {args.patient_name}"
  cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
  cv2.resizeWindow(win_title, win_w, win_h)

  is_finished = False

  try:
    async with websockets.connect(uri) as websocket:
      await websocket.send(
          json.dumps({
              "type": "CMD_SET_SESSION",
              "player_id": args.player_id,
              "patient_name": args.patient_name,
              "exercise_name": args.exercise_name,
              "mode": args.mode,
              "camera_index": args.camera_index,
              "target_reps": args.target_reps,
          })
      )

      while True:
        try:
          response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
          data = json.loads(response)

          if data.get("type") == "POSE_UPDATE":
            fps = data.get("fps", 0.0)
            current_mode = data.get("mode", args.mode)
            latest_keypoints = data.get("keypoints", [])
            yolo_detected = data.get("yolo_detected", False)
            latest_frame_b64 = data.get("frame_b64", None)

            left_info, right_info = data.get("left", {}), data.get("right", {})
            left_rep, left_progress = left_info.get(
                "rep_count", 0
            ), left_info.get("progress_ratio", 0.0)
            left_state, left_quality = left_info.get(
                "state", "READY"
            ), left_info.get("quality", "CALIBRATING")

            right_rep, right_progress = right_info.get(
                "rep_count", 0
            ), right_info.get("progress_ratio", 0.0)
            right_state, right_quality = right_info.get(
                "state", "READY"
            ), right_info.get("quality", "CALIBRATING")

          elif data.get("type") in [
              "CALIBRATION_FINISHED",
              "SESSION_FINISHED",
          ]:
            is_finished = True

        except asyncio.TimeoutError:
          pass

        canvas = None
        if latest_frame_b64:
          try:
            img_bytes = base64.b64decode(latest_frame_b64)
            decoded_frame = cv2.imdecode(
                np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR
            )
            if decoded_frame is not None:
              canvas = cv2.resize(decoded_frame, (win_w, win_h))
          except Exception:
            pass

        if canvas is None:
          canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)

        target = 3 if current_mode == "CALIBRATION" else args.target_reps

        l_str = (
            "완료(대기중)"
            if left_state == "WAITING"
            else (
                "기록중"
                if current_mode == "CALIBRATION"
                else left_quality
            )
        )
        r_str = (
            "완료(대기중)"
            if right_state == "WAITING"
            else (
                "기록중"
                if current_mode == "CALIBRATION"
                else right_quality
            )
        )

        canvas = put_korean_text(
            canvas,
            f"환자: {args.patient_name} | {args.exercise_name}",
            (20, 10),
            font_size=16,
        )
        canvas = put_korean_text(
            canvas,
            f"L: {left_rep}/{target}회 [{l_str}]",
            (20, 40),
            font_size=20,
            color=get_quality_color(left_quality),
        )
        canvas = put_korean_text(
            canvas,
            f"R: {right_rep}/{target}회 [{r_str}]",
            (20, 70),
            font_size=20,
            color=get_quality_color(right_quality),
        )

        # 검정색 배경의 미니박스 영역에 스켈레톤 시각화
        canvas = draw_normalized_mini_box(
            canvas,
            latest_keypoints,
            yolo_detected,
            (win_w - 140) // 2,
            win_h - 210,
            box_size=140,
        )

        elapsed = time.time() - start_timer
        if elapsed < PREP_SECONDS:
          canvas = draw_center_overlay(
              canvas, f"준비... {int(np.ceil(PREP_SECONDS - elapsed))}"
          )
        elif elapsed < (PREP_SECONDS + START_MSG_SECONDS):
          canvas = draw_center_overlay(
              canvas, "동작을 시작하세요", color=(0, 255, 0)
          )

        cv2.imshow(win_title, canvas)

        if is_finished:
          cv2.waitKey(1000)
          break

        if (
            cv2.waitKey(1) & 0xFF == 27
            or cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1
        ):
          break

      cv2.destroyAllWindows()

  except Exception as e:
    print(f"[Error] {e}")


if __name__ == "__main__":
  asyncio.run(main())