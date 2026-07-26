import cv2
import time
import numpy as np
import sys
import torch # 🌟 GPU 확인을 위해 torch 임포트 추가
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

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

def run_counting(ex_name, view_name, target_reps, cam_w, cam_h):
    # 🌟 GPU 자동 감지 로직 추가
    if torch.cuda.is_available():
        device_type = 0
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[INFO] GPU가 감지되었습니다! ({gpu_name}) 모델을 GPU에 할당합니다.\n")
    else:
        device_type = 'cpu'
        print("\n[INFO] 사용 가능한 GPU가 없습니다. CPU로 구동합니다.\n")

    model = YOLO("./configs/checkpoints/yolo11m-pose.pt")
    counter = UniversalRepetitionCounter(ex_name, view_name)
    cap = cv2.VideoCapture(0)
    
    check_joints = list(set(counter.sm_config.get('joints', []) + [6, 7]))
    trackers = {j: JointKalmanTracker() for j in check_joints}    
    
    prev_time = time.time()
    start_time = None
    final_counts = {}

    dash_w = max(cam_w // 3, 200) 

    window_name = "AI Workout Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (time.time() - prev_time)
        prev_time = time.time()

        # 🌟 판별한 device_type을 YOLO 추론 옵션에 적용
        results = model.predict(frame, conf=0.1, verbose=False, imgsz=480, device=device_type)
        left_display = frame.copy()
        
        right_board = np.full((cam_h, dash_w, 3), 40, dtype=np.uint8) 
        
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

        # ---------------------------------------------------------
        # 우측 보드 UI 구성
        # ---------------------------------------------------------
        cv2.putText(right_board, f"FPS: {fps:.1f}", (dash_w - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        y_count = 100
        for side in counter.sides:
            c = counter.counts.get(side, 0)
            color = (0, 215, 255) if c >= target_reps else (0, 255, 0)
            text = f"{side.upper()}: {c} / {target_reps}"
            
            max_w = dash_w - 40 
            font_scale = 1.5
            thickness = 3
            
            while True:
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                if tw <= max_w or font_scale <= 0.5:
                    break
                font_scale -= 0.1
                if font_scale < 1.0: 
                    thickness = 2
                    
            cv2.putText(right_board, text, (20, y_count), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
            y_count += th + 50 

        # ---------------------------------------------------------
        # 게이지 바 설정
        # ---------------------------------------------------------
        t_active_dict = counter.sm_config.get('trigger_active', {})
        t_start_dict = counter.sm_config.get('trigger_start', {})
        
        t_active = t_active_dict.get('threshold', 80.0)
        t_start = t_start_dict.get('threshold', 140.0)
        operator = t_active_dict.get('operator', '<')

        if counter.calc_method == 'angle':
            val_top, val_bottom = 0.0, 180.0
        else:
            val_top, val_bottom = 0.0, 1.2

        for i, side in enumerate(counter.sides):
            val = metrics.get(side, 0)
            
            bar_width = 30
            spacing = 40
            total_w = (len(counter.sides) * bar_width) + ((len(counter.sides) - 1) * spacing)
            start_x = (dash_w - total_w) // 2
            
            bx1 = start_x + i * (bar_width + spacing)
            bx2 = bx1 + bar_width
            by1 = cam_h - 200
            by2 = cam_h - 60
            bar_height = by2 - by1

            cv2.rectangle(right_board, (bx1, by1), (bx2, by2), (70, 70, 70), -1)
            cv2.rectangle(right_board, (bx1, by1), (bx2, by2), (120, 120, 120), 1)
            
            fill_h = int(np.interp(val, [val_top, val_bottom], [bar_height, 0]))
            fill_h = max(0, min(fill_h, bar_height)) 
            
            if operator == '<':
                f_color = (0, 255, 0) if val <= t_active else (0, 165, 255)
            else:
                f_color = (0, 255, 0) if val >= t_active else (0, 165, 255)
            
            cv2.rectangle(right_board, (bx1, by2 - fill_h), (bx2, by2), f_color, -1)

            ty_active = by2 - int(np.interp(t_active, [val_top, val_bottom], [bar_height, 0]))
            cv2.line(right_board, (bx1 - 10, ty_active), (bx2 + 10, ty_active), (0, 255, 0), 1)
            cv2.putText(right_board, "FLEX", (bx2 + 12, ty_active + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            ty_start = by2 - int(np.interp(t_start, [val_top, val_bottom], [bar_height, 0]))
            cv2.line(right_board, (bx1 - 10, ty_start), (bx2 + 10, ty_start), (0, 200, 255), 1)
            cv2.putText(right_board, "RELAX", (bx2 + 12, ty_start + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)

            cv2.putText(right_board, f"{val:.1f}", (bx1 - 5, by1 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(right_board, side[0].upper(), (bx1 + 8, by2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ---------------------------------------------------------
        combined = np.hstack((left_display, right_board))
        
        if not is_visible:
            cv2.rectangle(combined, (cam_w - 180, cam_h - 60), (cam_w - 20, cam_h - 20), (0, 0, 255), -1)
            cv2.putText(combined, "STEP BACK", (cam_w - 160, cam_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        finished = all(counter.counts.get(s, 0) >= target_reps for s in counter.sides) if counter.sides else False
        if finished:
            cv2.rectangle(combined, (cam_w // 2 - 200, cam_h // 2 - 50), (cam_w // 2 + 200, cam_h // 2 + 30), (0, 255, 0), -1)
            cv2.putText(combined, "MISSION COMPLETE!", (cam_w // 2 - 180, cam_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            cv2.imshow(window_name, combined)
            cv2.waitKey(3000); break

        cv2.imshow(window_name, combined)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release(); cv2.destroyAllWindows()
    return counter.counts, start_time or datetime.now(), datetime.now()