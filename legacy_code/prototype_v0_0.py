import cv2
import time
import numpy as np
import sys
from pathlib import Path
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image
from datetime import datetime

# 경로 설정
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils.counter_core import UniversalRepetitionCounter
from utils.normalization import normalize_realtime_12kpts

# 뼈대 연결 정보
BODY_EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # 상체
    (0, 6), (1, 7), (6, 7),                 # 몸통
    (6, 8), (8, 10), (7, 9), (9, 11)        # 하체
]

def draw_korean_text(img, text, position, font_size, color):
    """OpenCV 이미지에 한글을 깨짐 없이 그리는 함수"""
    font_path = "C:/Windows/Fonts/malgun.ttf"
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_visuals(frame, metrics, sides, sm_config, calc_method):
    h, w = frame.shape[:2]
    bar_w, bar_h = 30, 200
    pad_r, pad_b = 40, 50
    max_val = 180.0 if calc_method == 'angle' else 1.0

    for i, side in enumerate(sides):
        val = metrics.get(side)
        if val is None: continue
        x1 = w - pad_r - (i * 60) - bar_w
        y1 = h - pad_b - bar_h
        x2, y2 = x1 + bar_w, h - pad_b
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), -1)
        fill_h = int(np.interp(val, [0, max_val], [0, bar_h]))
        cv2.rectangle(frame, (x1, y2 - fill_h), (x2, y2), (0, 255, 0), -1)

# 🌟 1. 함수 선언부에 cam_w, cam_h 추가 (TypeError 해결)
def run_counting(ex_name, view_name, target_reps, cam_w, cam_h):
    model = YOLO("yolo11n-pose.pt")
    counter = UniversalRepetitionCounter(ex_name, view_name)
    cap = cv2.VideoCapture(0)
    
    required_joints = counter.sm_config.get('joints', [])
    final_counts = {}
    
    # 🌟 2. 추측 데이터 차단을 위한 임계값
    MIN_VISIBLE_CONF = 0.25   
    PX_MARGIN = 10  # 픽셀 단위 마진 (경계선 필터링)

    start_time = None
    end_time = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=0.1, verbose=False)
        annotated_frame = frame.copy()

        if len(results[0].keypoints) > 0:
            full_kps = results[0].keypoints.data[0].cpu().numpy()
            cropped_kps = full_kps[5:] 

            is_visible = True
            start_time = datetime.now()

            for j_idx in required_joints:
                x, y, conf = cropped_kps[j_idx]
                
                # 🌟 3. 전달받은 cam_w, cam_h를 사용하여 경계선 필터링
                # 신뢰도가 낮거나, 화면 끝 10px 이내에 관절이 붙어있으면 '추측'으로 간주
                if conf < MIN_VISIBLE_CONF or \
                   (x <= PX_MARGIN or x >= (cam_w - PX_MARGIN)) or \
                   (y <= PX_MARGIN or y >= (cam_h - PX_MARGIN)):
                    is_visible = False
                    break
            
            if is_visible:
                calc_kps = normalize_realtime_12kpts(cropped_kps)
                metrics, _ = counter.process_frame(calc_kps)
                final_counts = counter.counts
                
                # 시각화
                for edge in BODY_EDGES:
                    p1, p2 = edge
                    if cropped_kps[p1][2] > 0.3 and cropped_kps[p2][2] > 0.3:
                        c1 = (int(cropped_kps[p1][0]), int(cropped_kps[p1][1]))
                        c2 = (int(cropped_kps[p2][0]), int(cropped_kps[p2][1]))
                        cv2.line(annotated_frame, c1, c2, (255, 255, 255), 2)
                        cv2.circle(annotated_frame, c1, 5, (0, 255, 0), -1)

                for i, side in enumerate(counter.sides):
                    txt = f"{side.upper()}: {counter.counts.get(side, 0)} / {target_reps}"
                    cv2.putText(annotated_frame, txt, (20, 80 + (i*35)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                draw_visuals(annotated_frame, metrics, counter.sides, counter.sm_config, counter.calc_method)

                if any(count >= target_reps for count in final_counts.values()):
                    cv2.rectangle(annotated_frame, (cam_w//2-200, cam_h//2-40), (cam_w//2+200, cam_h//2+40), (0, 255, 0), -1)
                    annotated_frame = draw_korean_text(annotated_frame, "목표 달성 완료!", (cam_w//2-100, cam_h//2-20), 30, (255, 255, 255))
                    cv2.imshow("AI Workout Counter", annotated_frame)
                    cv2.waitKey(2000)
                    break
            else:
                # 🌟 관절 이탈 시 경고창 (전달받은 해상도 중앙에 배치)
                msg = "관절이 잘렸거나 화면 밖입니다.\n화면 중앙으로 물러나 주세요."
                cv2.rectangle(annotated_frame, (cam_w//2-220, cam_h//2-60), (cam_w//2+220, cam_h//2+60), (0, 0, 255), -1)
                annotated_frame = draw_korean_text(annotated_frame, msg, (cam_w//2-180, cam_h//2-30), 25, (255, 255, 255))

        cv2.imshow("AI Workout Counter", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    end_time = datetime.now()
    if start_time is None: start_time = end_time 


    cap.release()
    cv2.destroyAllWindows()
    return final_counts, start_time, end_time