import argparse
import asyncio
import websockets
import json
import cv2
import numpy as np
import time
import os
import base64
from PIL import ImageFont, ImageDraw, Image

# MMCV 패치
from unittest.mock import MagicMock
import sys
mock_ext = MagicMock()
mock_ext.__spec__ = MagicMock()
sys.modules['mmcv._ext'] = mock_ext

SKELETON_CONNECTIONS = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

def put_korean_text(img: np.ndarray, text: str, pos: tuple, font_size: int = 20, color: tuple = (255, 255, 255)) -> np.ndarray:
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    font_path = "C:/Windows/Fonts/malgun.ttf"
    font = ImageFont.truetype(font_path, font_size) if os.path.exists(font_path) else ImageFont.load_default()
    
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_normalized_mini_box(canvas: np.ndarray, keypoints: list, yolo_detected: bool, bx: int, by: int, box_size: int = 160) -> np.ndarray:
    """디버깅용 정보가 강화된 미니 박스 렌더링"""
    border_color = (0, 255, 0) if yolo_detected else (0, 0, 255)
    cv2.rectangle(canvas, (bx, by), (bx + box_size, by + box_size), (10, 10, 15), -1)
    cv2.rectangle(canvas, (bx, by), (bx + box_size, by + box_size), border_color, 2)
    
    kpt_len = len(keypoints) if keypoints else 0
    
    if not yolo_detected or kpt_len == 0:
        canvas = put_korean_text(canvas, "YOLO 미감지", (bx + 25, by + 50), font_size=14, color=(0, 0, 255))
        canvas = put_korean_text(canvas, f"KPT Count: {kpt_len}", (bx + 25, by + 75), font_size=12, color=(150, 150, 150))
        return canvas

    # 디버그 텍스트 표시
    sample_score = keypoints[5]["score"] if kpt_len > 5 else 0.0
    canvas = put_korean_text(canvas, f"KPT: {kpt_len}개 (Score: {sample_score:.2f})", (bx + 10, by + 5), font_size=11, color=(0, 255, 0))

    padding = 20
    inner_size = box_size - (padding * 2)

    kpt_dict = {}
    for kp in keypoints:
        kp_id = kp.get("id")
        kx = kp.get("x", 0.0)
        ky = kp.get("y", 0.0)
        score = kp.get("score", 0.0)

        # 정규화 좌표(0.0~1.0)를 박스 내부 공간으로 변환
        px = bx + padding + int(kx * inner_size)
        py = by + padding + int(ky * inner_size)
        kpt_dict[kp_id] = (px, py, score, kx, ky)

    # 1. 뼈대 연결선 그리기 (신뢰도 기준 0.5로 대폭 완화)
    for p1_id, p2_id in SKELETON_CONNECTIONS:
        if p1_id in kpt_dict and p2_id in kpt_dict:
            x1, y1, s1, _, _ = kpt_dict[p1_id]
            x2, y2, s2, _, _ = kpt_dict[p2_id]
            if s1 > 0.5 and s2 > 0.5:
                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)

    # 2. 관절 조인트 포인트 및 번호 그리기
    for idx, (x, y, s, kx, ky) in kpt_dict.items():
        if idx >= 5 and s > 0.5:
            # 점 표출
            cv2.circle(canvas, (x, y), 4, (0, 165, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)
            # 관절 번호 표출 (디버깅 핵심)
            cv2.putText(canvas, str(idx), (x + 3, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1, cv2.LINE_AA)

    return canvas

def draw_center_overlay(canvas: np.ndarray, text: str, font_size: int = 36, color: tuple = (0, 255, 255), bg_opacity: float = 0.6) -> np.ndarray:
    h, w, _ = canvas.shape
    overlay = canvas.copy()
    
    box_h = 100
    cy = h // 2
    cv2.rectangle(overlay, (0, cy - box_h//2), (w, cy + box_h//2), (10, 10, 15), -1)
    cv2.addWeighted(overlay, bg_opacity, canvas, 1 - bg_opacity, 0, canvas)
    
    text_x = max(20, (w - (len(text) * font_size // 2)) // 2 - 20)
    text_y = cy - font_size // 2 - 5
    
    return put_korean_text(canvas, text, (text_x, text_y), font_size=font_size, color=color)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player_id", type=str, default="patient_1")
    parser.add_argument("--patient_name", type=str, default="김지후")
    parser.add_argument("--exercise_name", type=str, default="biceps_curl")
    parser.add_argument("--mode", type=str, default="CALIBRATION")
    parser.add_argument("--target_reps", type=int, default=3)
    parser.add_argument("--camera_index", type=int, default=0)
    args = parser.parse_args()

    uri = "ws://127.0.0.1:8080"
    win_w, win_h = 800, 600

    PREP_SECONDS = 3.0
    START_MSG_SECONDS = 1.5
    start_timer = time.time()

    latest_keypoints = []
    fps = 0.0
    rep_count = 0
    current_mode = args.mode
    progress_ratio = 0.0
    yolo_detected = False
    latest_frame_b64 = None
    frame_counter = 0

    win_title = f"AI Motion Viewer - {args.patient_name}"
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_title, win_w, win_h)

    is_finished = False

    try:
        async with websockets.connect(uri) as websocket:
            session_cmd = {
                "type": "CMD_SET_SESSION",
                "player_id": args.player_id,
                "patient_name": args.patient_name,
                "exercise_name": args.exercise_name,
                "mode": args.mode,
                "camera_index": args.camera_index,
                "target_reps": args.target_reps
            }
            await websocket.send(json.dumps(session_cmd))

            while True:
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.01)
                    data = json.loads(response)

                    if data.get("type") == "POSE_UPDATE":
                        fps = data.get("fps", 0.0)
                        current_mode = data.get("mode", args.mode)
                        rep_count = data.get("rep_count", 0)
                        latest_keypoints = data.get("keypoints", [])
                        progress_ratio = data.get("progress_ratio", 0.0)
                        yolo_detected = data.get("yolo_detected", False)
                        latest_frame_b64 = data.get("frame_b64", None)

                        # 터미널 디버그 출력 (30프레임마다 1회)
                        frame_counter += 1
                        if frame_counter % 30 == 0:
                            kpt_cnt = len(latest_keypoints) if latest_keypoints else 0
                            sample_kp = latest_keypoints[5] if kpt_cnt > 5 else {}
                            print(f"[DEBUG] Detected: {yolo_detected} | KPT Count: {kpt_cnt} | Sample KPT[5]: {sample_kp}")

                    elif data.get("type") == "SESSION_FINISHED":
                        summary = data.get("summary", {})
                        session_id = summary.get("session_id", "UNKNOWN")
                        print(f"[Client] 세션 완료, 세션 ID: {session_id}")
                        is_finished = True

                except asyncio.TimeoutError:
                    pass

                # 메인 프레임 디코딩
                canvas = None
                if latest_frame_b64:
                    try:
                        img_bytes = base64.b64decode(latest_frame_b64)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        decoded_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                        if decoded_frame is not None:
                            canvas = cv2.resize(decoded_frame, (win_w, win_h))
                    except Exception:
                        pass

                if canvas is None:
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
                    canvas[:] = (20, 20, 25)

                # 헤더 오버레이
                header_overlay = canvas.copy()
                cv2.rectangle(header_overlay, (0, 0), (win_w, 110), (35, 35, 45), -1)
                cv2.addWeighted(header_overlay, 0.5, canvas, 0.3, 0, canvas)
                cv2.line(canvas, (0, 110), (win_w, 110), (0, 255, 255), 2)

                mode_str = "시범 측정 모드" if current_mode == "CALIBRATION" else "본 운동 모드"
                target = 3 if current_mode == "CALIBRATION" else args.target_reps
                status_str = f"YOLO 감지 성공 (KPT {len(latest_keypoints)}개)" if yolo_detected else "YOLO 감지 실패"
                status_color = (0, 255, 0) if yolo_detected else (0, 0, 255)

                canvas = put_korean_text(canvas, f"환자: {args.patient_name}  |  운동: {args.exercise_name} ({mode_str})", (20, 15), font_size=18, color=(255, 255, 255))
                canvas = put_korean_text(canvas, f"카운트: {rep_count} / {target}", (20, 50), font_size=28, color=(0, 255, 255))
                canvas = put_korean_text(canvas, f"상태: {status_str}", (300, 50), font_size=20, color=status_color)
                canvas = put_korean_text(canvas, f"FPS: {fps:.1f}", (win_w - 120, 15), font_size=16, color=(150, 150, 150))

                # 진행률 바
                bar_y = 560
                gauge_w = int((win_w - 40) * min(max(progress_ratio, 0.0), 1.0))
                cv2.rectangle(canvas, (20, bar_y), (win_w - 20, bar_y + 15), (50, 50, 60), -1)
                if gauge_w > 0:
                    cv2.rectangle(canvas, (20, bar_y), (20 + gauge_w, bar_y + 15), (0, 255, 0), -1)

                # 하단 미니 박스
                mini_size = 160
                mini_x = (win_w - mini_size) // 2
                mini_y = win_h - mini_size - 40
                canvas = draw_normalized_mini_box(canvas, latest_keypoints, yolo_detected, mini_x, mini_y, box_size=mini_size)

                # 타이머 안내 메시지
                elapsed = time.time() - start_timer
                if elapsed < PREP_SECONDS:
                    countdown_num = int(np.ceil(PREP_SECONDS - elapsed))
                    canvas = draw_center_overlay(
                        canvas, 
                        f"운동 준비... {countdown_num}", 
                        font_size=36, 
                        color=(255, 200, 0)
                    )
                elif elapsed < (PREP_SECONDS + START_MSG_SECONDS):
                    canvas = draw_center_overlay(
                        canvas, 
                        "동작을 시작하세요", 
                        font_size=34, 
                        color=(0, 255, 0)
                    )

                cv2.imshow(win_title, canvas)

                if rep_count >= target or is_finished:
                    cv2.waitKey(1500)
                    print("[Client] 목표 달성으로 세션을 종료합니다.")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key == 27 or cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:
                    break

            cv2.destroyAllWindows()

    except Exception as e:
        print(f"[Error] 접속 및 수신 에러: {e}")

if __name__ == "__main__":
    asyncio.run(main())