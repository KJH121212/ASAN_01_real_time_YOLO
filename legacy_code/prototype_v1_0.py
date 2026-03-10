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

# 뼈대 연결 정보
BODY_EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # 상체
    (0, 6), (1, 7), (6, 7),                 # 몸통
    (6, 8), (8, 10), (7, 9), (9, 11)        # 하체
]

# ---------------------------------------------------------
# 유틸리티 및 시각화 함수
# ---------------------------------------------------------
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
    """우측 하단 메트릭(게이지 바) 시각화"""
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

def draw_pip_skeleton(pip_frame, norm_kps, body_edges):
    """우측 상단 미니맵에 정규화된 뼈대를 그리는 함수"""
    h, w = pip_frame.shape[:2]
    scale = 80 
    center_x, center_y = w // 2, h // 2

    for edge in body_edges:
        p1, p2 = edge
        if p1 < len(norm_kps) and p2 < len(norm_kps):
            x1, y1 = norm_kps[p1][:2]
            x2, y2 = norm_kps[p2][:2]

            cx1 = int(x1 * scale + center_x)
            cy1 = int(y1 * scale + center_y)
            cx2 = int(x2 * scale + center_x)
            cy2 = int(y2 * scale + center_y)

            cv2.line(pip_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            cv2.circle(pip_frame, (cx1, cy1), 3, (0, 0, 255), -1)
            cv2.circle(pip_frame, (cx2, cy2), 3, (0, 0, 255), -1)
            
    return pip_frame

# ---------------------------------------------------------
# 칼만 필터 클래스 (순간이동/떨림 방지)
# ---------------------------------------------------------
class JointKalmanTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                             [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, x, y, max_jump=150):
        meas = np.array([[np.float32(x)], [np.float32(y)]])
        if not self.initialized:
            self.kf.statePre = np.array([[meas[0,0]], [meas[1,0]], [0], [0]], np.float32)
            self.kf.statePost = np.array([[meas[0,0]], [meas[1,0]], [0], [0]], np.float32)
            self.initialized = True
            return x, y
        
        pred = self.kf.predict()
        dist = np.sqrt((x - pred[0,0])**2 + (y - pred[1,0])**2)
        
        if dist > max_jump:
            return None 
            
        estimated = self.kf.correct(meas)
        return estimated[0,0], estimated[1,0]

# ---------------------------------------------------------
# 메인 카운팅 루프
# ---------------------------------------------------------
def run_counting(ex_name, view_name, target_reps, cam_w, cam_h):
    model = YOLO("yolo11n-pose.pt")
    counter = UniversalRepetitionCounter(ex_name, view_name)
    cap = cv2.VideoCapture(0)
    
    required_joints = counter.sm_config.get('joints', [])
    # 🌟 어떤 운동이든 정규화를 위해 골반(6, 7)은 무조건 필수로 검사
    check_joints = list(set(required_joints + [6, 7]))
    
    final_counts = {}
    MIN_VISIBLE_CONF = 0.45   
    PX_MARGIN = 15
    
    start_time = None
    end_time = None

    # 모든 검사 대상 관절에 칼만 필터 할당
    trackers = {j_idx: JointKalmanTracker() for j_idx in check_joints}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model.predict(frame, conf=0.1, verbose=False)
        annotated_frame = frame.copy()
        
        is_visible = False 
        missing_hip = False

        if len(results[0].keypoints) > 0:
            full_kps = results[0].keypoints.data[0].cpu().numpy()
            cropped_kps = full_kps[5:]
            smoothed_kps = cropped_kps.copy() 

            is_visible = True 
            
            # 🌟 [Step 1] 관절 무결성 검사 (경계선, 신뢰도, 순간이동)
            for j_idx in check_joints:
                x, y, conf = cropped_kps[j_idx]
                
                # A. 신뢰도 및 경계선 필터
                if conf < MIN_VISIBLE_CONF or \
                   (x <= PX_MARGIN or x >= (cam_w - PX_MARGIN)) or \
                   (y <= PX_MARGIN or y >= (cam_h - PX_MARGIN)):
                    is_visible = False
                    if j_idx in [6, 7]: missing_hip = True
                    trackers[j_idx].initialized = False 
                    break
                
                # B. 칼만 필터 추적
                corrected_pos = trackers[j_idx].update(x, y, max_jump=150)
                if corrected_pos is None:
                    is_visible = False
                    if j_idx in [6, 7]: missing_hip = True
                    trackers[j_idx].initialized = False
                    break
                else:
                    smoothed_kps[j_idx][0] = corrected_pos[0]
                    smoothed_kps[j_idx][1] = corrected_pos[1]

            # 🌟 [Step 2] 데이터가 완벽할 때만 정규화 및 카운팅 진행
            if is_visible:
                if start_time is None:
                    start_time = datetime.now()

                calc_kps = normalize_realtime_12kpts(smoothed_kps)
                metrics, _ = counter.process_frame(calc_kps)
                final_counts = counter.counts
                
                # 우측 상단 미니맵 (PiP)
                pip_w, pip_h = 200, 250
                pip_frame = np.zeros((pip_h, pip_w, 3), dtype=np.uint8)
                pip_frame = draw_pip_skeleton(pip_frame, calc_kps, BODY_EDGES)
                cv2.rectangle(pip_frame, (0, 0), (pip_w-1, pip_h-1), (255, 255, 255), 1)
                cv2.putText(pip_frame, "Norm Output", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                x_offset, y_offset = cam_w - pip_w - 20, 20
                if x_offset > 0 and y_offset > 0:
                    annotated_frame[y_offset:y_offset+pip_h, x_offset:x_offset+pip_w] = pip_frame

                # 메인 뼈대 그리기
                for edge in BODY_EDGES:
                    p1, p2 = edge
                    if smoothed_kps[p1][2] > MIN_VISIBLE_CONF and smoothed_kps[p2][2] > MIN_VISIBLE_CONF:
                        c1 = (int(smoothed_kps[p1][0]), int(smoothed_kps[p1][1]))
                        c2 = (int(smoothed_kps[p2][0]), int(smoothed_kps[p2][1]))
                        cv2.line(annotated_frame, c1, c2, (0, 255, 0), 2)
                        cv2.circle(annotated_frame, c1, 5, (255, 0, 0), -1)

                draw_visuals(annotated_frame, metrics, counter.sides, counter.sm_config, counter.calc_method)

        # 🌟 [Step 3] 가시성 실패 시 원인별 독립 경고창 출력
        if not is_visible:
            if missing_hip:
                msg = "⚠️ 골반이 화면 밖에 있습니다!\n전신이 나오도록 뒤로 물러나세요."
            else:
                msg = "⚠️ 주요 관절이 보이지 않습니다.\n카메라 앵글을 조절하세요."
                
            cv2.rectangle(annotated_frame, (cam_w//2-250, cam_h//2-60), (cam_w//2+250, cam_h//2+60), (0, 0, 255), -1)
            annotated_frame = draw_korean_text(
                annotated_frame, msg, (cam_w//2-220, cam_h//2-30), 25, (255, 255, 255)
            )

        # 🌟 수정된 부분: 양쪽(Left, Right) 모두 목표 횟수를 채웠을 때만 종료
        if counter.sides and len(counter.sides) >= 2:
            # 양쪽 사이드가 있고, 모든 사이드의 횟수가 target_reps 이상인지 확인
            finished = all(counter.counts.get(side, 0) >= target_reps for side in counter.sides)
        else:
            # 사이드가 하나뿐이거나 없는 경우 (예: 스쿼트 등) 기존처럼 한쪽만 체크
            finished = any(count >= target_reps for count in final_counts.values()) if final_counts else False

        if finished:
            # 종료 전 달성 축하 메시지
            cv2.rectangle(annotated_frame, (cam_w//2-200, cam_h//2-40), (cam_w//2+250, cam_h//2+40), (0, 255, 0), -1)
            annotated_frame = draw_korean_text(
                annotated_frame, "모든 세트 목표 달성 완료!", (cam_w//2-180, cam_h//2-20), 30, (255, 255, 255)
            )
            cv2.imshow("AI Workout Counter", annotated_frame)
            cv2.waitKey(3000) # 3초간 메시지 표시 후 종료
            break

        cv2.imshow("AI Workout Counter", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    # 종료 시간 기록 및 반환
    end_time = datetime.now()
    if start_time is None: start_time = end_time 

    cap.release()
    cv2.destroyAllWindows()
    return final_counts, start_time, end_time