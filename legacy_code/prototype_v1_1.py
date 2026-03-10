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
def draw_korean_text(img, text, position, font_size, color, bg_rect=False):
    """한글 텍스트 렌더링 (배경 박스 옵션 추가)"""
    font_path = "C:/Windows/Fonts/malgun.ttf"
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
    
    if bg_rect:
        # 텍스트 영역 계산 후 반투명 박스 그리기
        bbox = draw.textbbox(position, text, font=font)
        draw.rectangle([bbox[0]-10, bbox[1]-5, bbox[2]+10, bbox[3]+5], fill=(0, 0, 0, 150))

    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def draw_enhanced_visuals(frame, metrics, sides, sm_config, calc_method):
    """임계값(Threshold)이 적용된 우측 하단 게이지 시각화"""
    h, w = frame.shape[:2]
    bar_w, bar_h = 35, 220
    pad_r, pad_b = 40, 50
    
    # 설정 파일에서 임계값 가져오기
    thresh = sm_config.get('threshold', 0)
    max_val = 180.0 if calc_method == 'angle' else 1.0

    for i, side in enumerate(sides):
        val = metrics.get(side, 0)
        x1 = w - pad_r - (i * 70) - bar_w
        y1 = h - pad_b - bar_h
        x2, y2 = x1 + bar_w, h - pad_b
        
        # 1. 바 배경 (어두운 회색)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 50), -1)
        
        # 2. 값에 따른 색상 결정 (임계값 넘으면 초록색, 아니면 빨간색)
        fill_color = (0, 255, 0) if val >= thresh else (0, 0, 255)
        fill_h = int(np.interp(val, [0, max_val], [0, bar_h]))
        cv2.rectangle(frame, (x1, y2 - fill_h), (x2, y2), fill_color, -1)
        
        # 3. 임계값(Threshold) 라인 표시
        thresh_y = y2 - int(np.interp(thresh, [0, max_val], [0, bar_h]))
        cv2.line(frame, (x1 - 5, thresh_y), (x2 + 5, thresh_y), (255, 255, 255), 2)
        
        # 4. 라벨 (L / R)
        cv2.putText(frame, side[0].upper(), (x1 + 10, y2 + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

def draw_pip_skeleton(pip_frame, norm_kps, body_edges):
    h, w = pip_frame.shape[:2]
    scale = 80 
    center_x, center_y = w // 2, h // 2
    for edge in body_edges:
        p1, p2 = edge
        if p1 < len(norm_kps) and p2 < len(norm_kps):
            x1, y1 = norm_kps[p1][:2]
            x2, y2 = norm_kps[p2][:2]
            cx1, cy1 = int(x1 * scale + center_x), int(y1 * scale + center_y)
            cx2, cy2 = int(x2 * scale + center_x), int(y2 * scale + center_y)
            cv2.line(pip_frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)
            cv2.circle(pip_frame, (cx1, cy1), 3, (0, 0, 255), -1)
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
    check_joints = list(set(required_joints + [6, 7]))
    trackers = {j_idx: JointKalmanTracker() for j_idx in check_joints}
    
    MIN_VISIBLE_CONF = 0.45 
    PX_MARGIN = 15
    start_time = datetime.now()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model.predict(frame, conf=0.1, verbose=False)
        annotated_frame = frame.copy()
        is_visible, missing_hip = False, False

        if len(results[0].keypoints) > 0:
            full_kps = results[0].keypoints.data[0].cpu().numpy()
            cropped_kps = full_kps[5:]
            smoothed_kps = cropped_kps.copy()
            is_visible = True

            for j_idx in check_joints:
                x, y, conf = cropped_kps[j_idx]
                if conf < MIN_VISIBLE_CONF or (x <= PX_MARGIN or x >= (cam_w - PX_MARGIN)) or (y <= PX_MARGIN or y >= (cam_h - PX_MARGIN)):
                    is_visible = False
                    if j_idx in [6, 7]: missing_hip = True
                    break
                
                pos = trackers[j_idx].update(x, y)
                if pos is None:
                    is_visible = False
                    break
                smoothed_kps[j_idx][0], smoothed_kps[j_idx][1] = pos

            if is_visible:
                if start_time is None: start_time = datetime.now()
                calc_kps = normalize_realtime_12kpts(smoothed_kps)
                metrics, _ = counter.process_frame(calc_kps)

                # 🌟 [시각화 1] 우측 상단 PiP 미니맵
                pip_frame = np.zeros((250, 200, 3), dtype=np.uint8)
                pip_frame = draw_pip_skeleton(pip_frame, calc_kps, BODY_EDGES)
                cv2.rectangle(pip_frame, (0, 0), (199, 249), (255, 255, 255), 1)
                annotated_frame[20:270, cam_w-220:cam_w-20] = pip_frame

                # 🌟 [시각화 2] 우측 하단 임계값 게이지 바
                draw_enhanced_visuals(annotated_frame, metrics, counter.sides, counter.sm_config, counter.calc_method)

                # 메인 뼈대 그리기
                for edge in BODY_EDGES:
                    p1, p2 = edge
                    c1 = (int(smoothed_kps[p1][0]), int(smoothed_kps[p1][1]))
                    c2 = (int(smoothed_kps[p2][0]), int(smoothed_kps[p2][1]))
                    cv2.line(annotated_frame, c1, c2, (0, 255, 0), 2)

        # 🌟 [시각화 3] 좌측 상단 카운트 정보 보드
        y_offset = 30
        for side in counter.sides:
            count = counter.counts.get(side, 0)
            status_text = f"{side.upper()}: {count} / {target_reps}"
            # 목표 달성 시 금색, 진행 중일 때 초록색
            color = (0, 215, 255) if count >= target_reps else (0, 255, 0)
            annotated_frame = draw_korean_text(annotated_frame, status_text, (20, y_offset), 30, color, bg_rect=True)
            y_offset += 50

        # 가시성 경고
        if not is_visible:
            msg = "⚠️ 골반이 잘렸습니다! 뒤로 물러나세요." if missing_hip else "⚠️ 전신이 나오게 조절하세요."
            annotated_frame = draw_korean_text(annotated_frame, msg, (cam_w//2-200, cam_h//2), 25, (255, 255, 255), bg_rect=True)

        # 종료 판정 로직
        finished = False
        if counter.sides and len(counter.sides) >= 2:
            finished = all(counter.counts.get(s, 0) >= target_reps for s in counter.sides)
        elif counter.counts:
            finished = any(c >= target_reps for c in counter.counts.values())

        if finished:
            annotated_frame = draw_korean_text(annotated_frame, "🎉 모든 세트 완료! 고생하셨습니다!", (cam_w//2-250, cam_h//2-20), 35, (0, 255, 0), bg_rect=True)
            cv2.imshow("AI Workout Counter", annotated_frame)
            cv2.waitKey(3000)
            break

        cv2.imshow("AI Workout Counter", annotated_frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    cap.release()
    cv2.destroyAllWindows()
    return counter.counts, start_time, datetime.now()