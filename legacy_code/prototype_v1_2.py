import cv2
import time
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

# 경로 설정
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils.counter_core import UniversalRepetitionCounter
from utils.normalization import normalize_realtime_12kpts
from utils.kalman import JointKalmanTracker

# 뼈대 연결 정보
BODY_EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # 상체
    (0, 6), (1, 7), (6, 7),                 # 몸통
    (6, 8), (8, 10), (7, 9), (9, 11)        # 하체
]

def draw_korean_text(img, text, position, font_size, color, bg_rect=False):
    font_path = "C:/Windows/Fonts/malgun.ttf"
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    if bg_rect:
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle([bbox[0]-5, bbox[1]-2, bbox[2]+5, bbox[3]+2], fill=(0, 0, 0, 150))

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_centered_skeleton(canvas, norm_kps, body_edges, scale=110):
    h, w = canvas.shape[:2]
    center_x = (norm_kps[6][0] + norm_kps[7][0]) / 2
    center_y = (norm_kps[6][1] + norm_kps[7][1]) / 2
    offset_x, offset_y = w // 2, h // 2

    for edge in body_edges:
        p1, p2 = edge
        if p1 < len(norm_kps) and p2 < len(norm_kps):
            x1, y1 = (norm_kps[p1][0] - center_x) * scale + offset_x, (norm_kps[p1][1] - center_y) * scale + offset_y
            x2, y2 = (norm_kps[p2][0] - center_x) * scale + offset_x, (norm_kps[p2][1] - center_y) * scale + offset_y
            cv2.line(canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.circle(canvas, (int(x1), int(y1)), 4, (0, 0, 255), -1)
    return canvas

def run_counting(ex_name, view_name, target_reps, cam_w, cam_h):
    model = YOLO("yolo11n-pose.pt")
    counter = UniversalRepetitionCounter(ex_name, view_name)
    cap = cv2.VideoCapture(0)
    
    check_joints = list(set(counter.sm_config.get('joints', []) + [6, 7]))
    trackers = {j: JointKalmanTracker() for j in check_joints}    
    
    prev_time = time.time()
    start_time = None
    final_counts = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        results = model.predict(frame, conf=0.1, verbose=False)
        left_display = frame.copy()
        right_board = np.zeros((cam_h, cam_w, 3), dtype=np.uint8)
        
        is_visible, missing_hip = False, False
        metrics = {}

        if len(results[0].keypoints) > 0:
            full_kps = results[0].keypoints.data[0].cpu().numpy()
            cropped_kps = full_kps[5:]
            smoothed_kps = cropped_kps.copy()
            is_visible = True

            for j_idx in check_joints:
                x, y, conf = cropped_kps[j_idx]
                if conf < 0.45 or x <= 15 or x >= (cam_w - 15) or y <= 15 or y >= (cam_h - 15):
                    is_visible = False
                    if j_idx in [6, 7]: missing_hip = True
                    break
                smoothed_kps[j_idx][0], smoothed_kps[j_idx][1] = trackers[j_idx].update(x, y)

            if is_visible:
                if start_time is None: start_time = datetime.now()
                calc_kps = normalize_realtime_12kpts(smoothed_kps)
                metrics, _ = counter.process_frame(calc_kps)
                final_counts = counter.counts

                for edge in BODY_EDGES:
                    p1, p2 = edge
                    c1, c2 = (int(smoothed_kps[p1][0]), int(smoothed_kps[p1][1])), (int(smoothed_kps[p2][0]), int(smoothed_kps[p2][1]))
                    cv2.line(left_display, c1, c2, (0, 255, 0), 2)

                norm_canvas = np.zeros((300, 300, 3), dtype=np.uint8)
                norm_canvas = draw_centered_skeleton(norm_canvas, calc_kps, BODY_EDGES)
                right_board[120:420, (cam_w//2-150):(cam_w//2+150)] = norm_canvas

        # 🌟 우측 보드 UI 구성
        cv2.putText(right_board, f"FPS: {fps:.1f}", (cam_w-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        y_count = 40
        for side in counter.sides:
            c = counter.counts.get(side, 0)
            color = (0, 215, 255) if c >= target_reps else (0, 255, 0)
            right_board = draw_korean_text(right_board, f"{side.upper()}: {c}/{target_reps}", (20, y_count), 30, color, bg_rect=True)
            y_count += 45

        # 🌟 수정된 게이지 바 (Flexion 수축 시 위로 향하는 로직)
        thresh = counter.sm_config.get('threshold', 0)
        max_val = 180.0 if counter.calc_method == 'angle' else 1.0
        
        for i, side in enumerate(counter.sides):
            val = metrics.get(side, 0)
            # 바 위치 설정 (우측 하단 배치)
            bx1 = cam_w - 70 - (i * 85)
            by1, bx2, by2 = cam_h - 280, bx1 + 40, cam_h - 60
            bar_height = by2 - by1

            # 1. 배경 사각형
            cv2.rectangle(right_board, (bx1, by1), (bx2, by2), (40, 40, 40), -1)
            
            # 2. 🚀 핵심 수정: 수축(값이 작아짐)할수록 바가 위로 차오르게 매핑 반전
            # [0, max_val] 입력 범위를 [bar_height, 0] 출력 범위로 뒤집음
            # 결과: 0도(완전수축) -> bar_height(꽉 참), 180도(이완) -> 0(바닥)
            fill_h = int(np.interp(val, [0, max_val], [bar_height, 0]))
            
            # 3. 색상 결정 (수축 임계값보다 작아지면 수축 성공이므로 초록색)
            # Biceps Curl은 operator가 "<" 이므로 val <= thresh일 때 목표 도달
            f_color = (0, 255, 0) if val <= thresh else (0, 165, 255)
            cv2.rectangle(right_board, (bx1, by2 - fill_h), (bx2, by2), f_color, -1)

            # 4. Threshold(임계값) 가이드라인도 반전된 위치에 표시
            ty = by2 - int(np.interp(thresh, [0, max_val], [bar_height, 0]))
            cv2.line(right_board, (bx1 - 10, ty), (bx2 + 10, ty), (255, 255, 255), 2)
            cv2.putText(right_board, "GOAL", (bx1 - 45, ty + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # 5. 현재 수치 텍스트 (바 상단)
            val_txt = f"{val:.1f}"
            right_board = draw_korean_text(right_board, val_txt, (bx1, by1 - 25), 15, (255, 255, 255))
            cv2.putText(right_board, side[0].upper(), (bx1 + 12, by2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        combined = np.hstack((left_display, right_board))
        
        if not is_visible:
            msg = "전신이 나오게 뒤로 물러나세요"
            combined = draw_korean_text(combined, msg, (cam_w - 200, cam_h - 30), 20, (255, 0, 0), bg_rect=True)

        finished = all(counter.counts.get(s, 0) >= target_reps for s in counter.sides) if counter.sides else False
        if finished:
            combined = draw_korean_text(combined, "MISSION COMPLETE!", (cam_w - 200, cam_h // 2), 40, (0, 255, 0), bg_rect=True)
            cv2.imshow("AI Workout Analysis", combined)
            cv2.waitKey(3000); break

        cv2.imshow("AI Workout Analysis", combined)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()
    return counter.counts, start_time or datetime.now(), datetime.now()