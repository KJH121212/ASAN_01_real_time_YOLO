import cv2
import time
import numpy as np
import sys
import torch
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
import sys
import pycocotools
sys.modules['xtcocotools'] = pycocotools

# 기존 코드 시작...
# from ultralytics import YOLO
# from mmpose.apis import init_model, inference_topdown
# MMPose 관련 라이브러리
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

# MMPose 커스텀 모듈 등록
register_all_modules()

# 경로 설정
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils.counter_core import UniversalRepetitionCounter
from utils.normalization import normalize_realtime_12kpts
from utils.kalman import JointKalmanTracker

# 12개 관절(어깨~발목) 기준 뼈대 연결 인덱스 (0~11)
# 0:L_Shoulder, 1:R_Shoulder, 2:L_Elbow, 3:R_Elbow, 4:L_Wrist, 5:R_Wrist,
# 6:L_Hip, 7:R_Hip, 8:L_Knee, 9:R_Knee, 10:L_Ankle, 11:R_Ankle
BODY_EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # 상체
    (0, 6), (1, 7), (6, 7),                 # 몸통
    (6, 8), (8, 10), (7, 9), (9, 11)        # 하체
]

def run_counting(ex_name, view_name, target_reps, cam_w, cam_h, cam_idx=0):
    # ---------------------------------------------------------
    # 1. Device 설정 (CUDA GPU 가속)
    # ---------------------------------------------------------
    if torch.cuda.is_available():
        device = 'cuda:0'
        gpu_name = torch.cuda.get_device_name(0)
        print(f"\n[INFO] 🚀 GPU 가속 활성화! ({gpu_name})\n")
    else:
        device = 'cpu'
        print("\n[INFO] ⚠️ GPU를 찾을 수 없어 CPU 모드로 동작합니다.\n")

    # ---------------------------------------------------------
    # 2. Detector(YOLO) 및 RTMPose 모델 초기화
    # ---------------------------------------------------------
    print("[INFO] Detector(YOLOv8n) 모델 로드 중...")
    detector = YOLO('yolov8n.pt')

    # Config 및 Checkpoint 경로 설정
    config_path = "./configs/checkpoints/RTMPose_config.py" # 제공해주신 Config 파일 저장 경로
    checkpoint_path = "./configs/checkpoints/RTMPose.pth"   # best.pth 모델 경로

    print("[INFO] 파인튜닝된 RTMPose 모델 초기화 중...")
    pose_model = init_model(config_path, checkpoint_path, device=device)
    print("[INFO] ✅ 모든 모델 로드 완료!")

    # ---------------------------------------------------------
    # 3. 카운터, 카메라 및 칼만 필터 초기화
    # ---------------------------------------------------------
    counter = UniversalRepetitionCounter(ex_name, view_name)
    cap = cv2.VideoCapture(cam_idx)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)

    check_joints = list(set(counter.sm_config.get('joints', []) + [6, 7]))
    trackers = {j: JointKalmanTracker() for j in check_joints}    

    prev_time = time.time()
    start_time = None
    final_counts = {}

    dash_w = max(cam_w // 3, 200)

    window_name = "AI Workout Analysis (RTMPose)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ---------------------------------------------------------
    # 🌟 [수정 완료] YOLO 캐싱 & 중앙 사람 선택용 변수 선언
    # ---------------------------------------------------------
    det_interval = 0.25  # YOLO 재검출 주기 (0.25초)
    last_det_time = 0.0  # 마지막 YOLO 실행 타임스탬프
    cached_bbox = None   # 가장 최근에 선택된 중앙 BBox 캐시

    # 화면 중심 좌표 (가운데 사람을 판별하는 기준)
    cam_center_x = cam_w / 2.0
    cam_center_y = cam_h / 2.0

    # ---------------------------------------------------------
    # 메인 루프 시작
    # ---------------------------------------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break
        
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        left_display = frame.copy()
        right_board = np.full((cam_h, dash_w, 3), 40, dtype=np.uint8) 
        
        is_visible, missing_hip = False, False
        metrics = {}

        # ---------------------------------------------------------
        # Step 1: 0.25초 주기 YOLO 검출 & 화면 중앙 BBox 1개 유지
        # ---------------------------------------------------------
        if (current_time - last_det_time) >= det_interval or cached_bbox is None:
            last_det_time = current_time
            det_results = detector(frame, classes=[0], verbose=False, device=device)[0]
            
            if len(det_results.boxes) > 0:
                boxes = det_results.boxes.xyxy.cpu().numpy()
                confidences = det_results.boxes.conf.cpu().numpy()
                
                min_dist = float('inf')
                selected_box = None
                
                for box, conf in zip(boxes, confidences):
                    if conf > 0.4:
                        x1, y1, x2, y2 = box
                        # BBox의 중심점 계산
                        box_center_x = (x1 + x2) / 2.0
                        box_center_y = (y1 + y2) / 2.0
                        
                        # 화면 중앙과의 유클리드 거리 계산
                        dist = np.sqrt((box_center_x - cam_center_x)**2 + (box_center_y - cam_center_y)**2)
                        
                        # 화면 중심에 가장 가까운 BBox 갱신
                        if dist < min_dist:
                            min_dist = dist
                            selected_box = box
                
                # 중앙에 가장 가까운 BBox를 캐시에 저장 (없으면 None)
                cached_bbox = selected_box

        # ---------------------------------------------------------
        # Step 2: 캐시된 BBox 1개로 RTMPose 추론 수행
        # ---------------------------------------------------------
        if cached_bbox is not None:
            # RTMPose 입력 형식에 맞게 2D 배열로 전달
            bboxes_np = np.array([cached_bbox], dtype=np.float32)
            pose_results = inference_topdown(pose_model, frame, bboxes_np)

            if len(pose_results) > 0:
                pred_instances = pose_results[0].pred_instances
                kpts = pred_instances.keypoints[0]         # (17, 2)
                scores = pred_instances.keypoint_scores[0] # (17,)

                full_kps = np.hstack([kpts, scores[:, None]])
                cropped_kps = full_kps[5:] # 상/하체 12개 관절 슬라이싱
                smoothed_kps = cropped_kps.copy()
                is_visible = True

                for j_idx in check_joints:
                    x, y, conf = cropped_kps[j_idx]
                    if conf < 0.45 or x <= 15 or x >= (cam_w - 15) or y <= 15 or y >= (cam_h - 15):
                        is_visible = False
                        if j_idx in [6, 7]: 
                            missing_hip = True
                        break
                    smoothed_kps[j_idx][0], smoothed_kps[j_idx][1] = trackers[j_idx].update(x, y)

                if is_visible:
                    if start_time is None: 
                        start_time = datetime.now()
                    calc_kps = normalize_realtime_12kpts(smoothed_kps)
                    metrics, _ = counter.process_frame(calc_kps)
                    final_counts = counter.counts

                    # 뼈대 및 관절 그리기
                    for edge in BODY_EDGES:
                        p1, p2 = edge
                        c1 = (int(smoothed_kps[p1][0]), int(smoothed_kps[p1][1]))
                        c2 = (int(smoothed_kps[p2][0]), int(smoothed_kps[p2][1]))
                        cv2.line(left_display, c1, c2, (0, 255, 0), 2)
                        
                    for kp in smoothed_kps:
                        cv2.circle(left_display, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)

                    # 선택된 사람의 BBox 영역 시각화
                    bx1, by1, bx2, by2 = map(int, cached_bbox)
                    cv2.rectangle(left_display, (bx1, by1), (bx2, by2), (255, 128, 0), 1)

        # ---------------------------------------------------------
        # Step 3: 대시보드 UI (카운트 & FPS)
        # ---------------------------------------------------------
        cv2.putText(right_board, f"FPS: {fps:.1f}", (dash_w - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

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
        # Step 4: 게이지 바 UI
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
        # Step 5: 화면 병합 및 완료 처리
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
            cv2.waitKey(3000)
            break

        cv2.imshow(window_name, combined)
        if cv2.waitKey(1) & 0xFF == 27: # ESC 키로 종료
            break

    cap.release()
    cv2.destroyAllWindows()
    return counter.counts, start_time or datetime.now(), datetime.now()