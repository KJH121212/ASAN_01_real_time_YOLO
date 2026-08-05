import os
import sys
import torch
import numpy as np
import cv2
from pathlib import Path
import pycocotools

sys.modules['xtcocotools'] = pycocotools

# CPU 쓰레드 제한
torch.set_num_threads(2)
cv2.setNumThreads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# OpenMMLab 라이브러리 임포트
from ultralytics import YOLO
from mmpose.apis import init_model, inference_topdown
from mmpose.utils import register_all_modules

register_all_modules()

# 경로 설정
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils.normalization import normalize_realtime_12kpts

BODY_EDGES = [
    (0, 1), (0, 2), (2, 4), (1, 3), (3, 5), # 상체
    (0, 6), (1, 7), (6, 7),                 # 몸통
    (6, 8), (8, 10), (7, 9), (9, 11)        # 하체
]

def draw_skeleton_on_image(image, keypoints_12, edges=BODY_EDGES):
    img_draw = image.copy()
    for p1, p2 in edges:
        pt1 = (int(keypoints_12[p1][0]), int(keypoints_12[p1][1]))
        pt2 = (int(keypoints_12[p2][0]), int(keypoints_12[p2][1]))
        cv2.line(img_draw, pt1, pt2, (0, 255, 0), 2)
    for kp in keypoints_12:
        cv2.circle(img_draw, (int(kp[0]), int(kp[1])), 4, (0, 0, 255), -1)
    return img_draw

def render_normalized_canvas(norm_kpts_12, canvas_size=256):
    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    coords = norm_kpts_12[:, :2]

    mapped_coords = (coords + 1.0) * 0.5 * (canvas_size - 60) + 30

    for p1, p2 in BODY_EDGES:
        pt1 = (int(mapped_coords[p1][0]), int(mapped_coords[p1][1]))
        pt2 = (int(mapped_coords[p2][0]), int(mapped_coords[p2][1]))
        cv2.line(canvas, pt1, pt2, (0, 255, 0), 2)

    for kp in mapped_coords:
        cv2.circle(canvas, (int(kp[0]), int(kp[1])), 4, (0, 255, 255), -1)

    return canvas

def run_pipeline_for_image(img_path="./data/000171.jpg", output_dir="./output_pngs"):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    yolo_path = str(current_dir / "configs" / "checkpoints" / "yolov8n.pt")
    config_path = str(current_dir / "configs" / "checkpoints" / "RTMPose_config.py")
    checkpoint_path = str(current_dir / "configs" / "checkpoints" / "RTMPose.pth")

    print(f"[INFO] 🚀 디바이스({device}) 기반 모델 로딩 중...")
    detector = YOLO(yolo_path)
    pose_model = init_model(config_path, checkpoint_path, device=device)

    # 1. Input Image
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"[ERROR] 이미지를 읽을 수 없습니다: {img_path}")
        return

    cam_h, cam_w = frame.shape[:2]
    cam_center = (cam_w / 2.0, cam_h / 2.0)

    cv2.imwrite(str(output_path / "1_input_frame.png"), frame)

    # 2. YOLO Detecting
    det_results = detector(frame, classes=[0], verbose=False, device=device)[0]
    if len(det_results.boxes) == 0:
        print("[ERROR] 사람이 감지되지 않았습니다.")
        return

    boxes = det_results.boxes.xyxy.cpu().numpy()
    confidences = det_results.boxes.conf.cpu().numpy()

    selected_box = None
    min_dist = float('inf')
    for box, conf in zip(boxes, confidences):
        if conf > 0.4:
            bx1, by1, bx2, by2 = box
            center_x, center_y = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
            dist = np.sqrt((center_x - cam_center[0])**2 + (center_y - cam_center[1])**2)
            if dist < min_dist:
                min_dist = dist
                selected_box = box

    if selected_box is None:
        print("[ERROR] BBox 조건 미충족")
        return

    img2_yolo = frame.copy()
    bx1, by1, bx2, by2 = map(int, selected_box)
    cv2.rectangle(img2_yolo, (bx1, by1), (bx2, by2), (255, 128, 0), 2)
    cv2.putText(img2_yolo, "Person", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 0), 2)
    cv2.imwrite(str(output_path / "2_yolo_detected.png"), img2_yolo)

    # 3. Crop & RTMPose Skeleton
    bboxes_np = np.array([selected_box], dtype=np.float32)
    pose_results = inference_topdown(pose_model, frame, bboxes_np)

    if len(pose_results) > 0:
        pred_instances = pose_results[0].pred_instances
        kpts = pred_instances.keypoints[0]
        scores = pred_instances.keypoint_scores[0]

        full_kps = np.hstack([kpts, scores[:, None]])
        cropped_kps = full_kps[5:]

        crop_area = frame[by1:by2, bx1:bx2].copy()
        crop_kpts = cropped_kps.copy()
        crop_kpts[:, 0] -= bx1
        crop_kpts[:, 1] -= by1
        img3_crop = draw_skeleton_on_image(crop_area, crop_kpts)
        cv2.imwrite(str(output_path / "3_cropped_rtmpose.png"), img3_crop)

        # 4. Normalized Skeleton (256x256)
        norm_kpts = normalize_realtime_12kpts(cropped_kps)
        img4_norm = render_normalized_canvas(norm_kpts, canvas_size=256)
        
        cv2.imwrite(str(output_path / "4_normalized_skeleton_256.png"), img4_norm)

        print(f"\n[SUCCESS] 4개의 PNG 저장 완료: {output_path.resolve()}\n")

if __name__ == "__main__":
    target_img = "./data/000171.jpg"
    run_pipeline_for_image(target_img)