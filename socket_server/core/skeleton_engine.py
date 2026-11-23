"""순수 ONNX Runtime + OpenCV + NumPy 기반 YOLOv8n + RTMPose-M (COCO-23) 고속 추론 엔진.

PyTorch · MMPose · MMCV · Ultralytics 없이 동작하며,
단일 프레임 실시간 추론과 배치 추론(MP4 파일 분석)을 모두 지원한다.
"""

import os
from pathlib import Path
import site
import sys
import cv2
import numpy as np
import onnxruntime as ort

_DLL_DIR_HANDLES = []


def _register_cuda_dll_dirs():
    """PyTorch 없이도 ONNX Runtime CUDA EP가 CUDA/cuDNN DLL을 찾을 수 있게 경로를 등록한다."""
    exe_dir = Path(sys.executable).resolve().parent
    site_dirs = [
        exe_dir / "Lib" / "site-packages",
        exe_dir.parent / "Lib" / "site-packages",
        *[Path(p) for p in site.getsitepackages()],
    ]
    candidates = []
    for base in site_dirs:
        nvidia_dir = base / "nvidia"
        if nvidia_dir.is_dir():
            candidates.extend(nvidia_dir.glob("*/bin"))
        candidates.append(base / "torch" / "lib")

    seen = set()
    cur_path = os.environ.get("PATH", "")
    added_paths = []
    for d in candidates:
        if not d.is_dir():
            continue
        d_str = str(d.resolve())
        key = d_str.lower()
        if key in seen:
            continue
        seen.add(key)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_DIR_HANDLES.append(os.add_dll_directory(d_str))
            except OSError:
                pass
        if key not in cur_path.lower():
            added_paths.append(d_str)
    if added_paths:
        os.environ["PATH"] = os.pathsep.join(added_paths) + os.pathsep + cur_path


_register_cuda_dll_dirs()

NUM_KEYPOINTS = 23
POSE_INPUT_W, POSE_INPUT_H = 192, 256
SIMCC_SPLIT_RATIO = 2.0
MEAN_RGB = np.array([123.675, 116.28, 103.53], dtype=np.float32).reshape(1, 1, 3)
STD_RGB = np.array([58.395, 57.12, 57.375], dtype=np.float32).reshape(1, 1, 3)


def _get_3rd_point(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    direction = a - b
    return b + np.array([-direction[1], direction[0]], dtype=np.float32)


def _get_warp_matrix(center: np.ndarray, scale: np.ndarray, out_size: tuple[int, int]) -> np.ndarray:
    """MMPose TopdownAffine과 동일한 2x3 Affine 변환 행렬을 계산한다."""
    dst_w, dst_h = out_size
    src_w = float(scale[0])
    src_dir = np.array([0.0, src_w * -0.5], dtype=np.float32)
    dst_dir = np.array([0.0, dst_w * -0.5], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center
    src[1, :] = center + src_dir
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    return cv2.getAffineTransform(src, dst)


def _bbox_xyxy_to_cs(bbox: np.ndarray, padding: float = 1.25, aspect_ratio: float = 192.0 / 256.0):
    """MMPose GetBBoxCenterScale과 동일하게 바운딩 박스를 (center, scale)로 변환한다."""
    x1, y1, x2, y2 = [float(v) for v in bbox]
    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    w = max(1.0, (x2 - x1) * padding)
    h = max(1.0, (y2 - y1) * padding)
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    return center, np.array([w, h], dtype=np.float32)


class SkeletonEngine:
    """중앙 대상자 1명을 검출해 23개 관절(COCO-23) 좌표를 추출하는 순수 ONNX Runtime 고속 엔진."""

    def __init__(self, device: str = None, yolo_interval: int = 10, on_progress=None):
        report = on_progress or (lambda phase: None)
        report("DEVICE_SETUP")

        available = ort.get_available_providers()
        use_gpu = (device is None or str(device).startswith("cuda")) and ("CUDAExecutionProvider" in available)
        if use_gpu:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device = "cuda:0 (ONNX)"
        else:
            providers = ["CPUExecutionProvider"]
            self.device = "cpu (ONNX)"

        ckpt_dir = Path(__file__).resolve().parent.parent / "configs" / "checkpoints"
        yolo_onnx = ckpt_dir / "yolov8n.onnx"
        pose_onnx = ckpt_dir / "rtmpose_coco23.onnx"
        if not yolo_onnx.is_file() or not pose_onnx.is_file():
            raise FileNotFoundError(
                f"ONNX 모델 파일을 찾을 수 없습니다: {yolo_onnx.name}, {pose_onnx.name} ({ckpt_dir})"
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        report("LOADING_YOLO")
        self.yolo_sess = ort.InferenceSession(str(yolo_onnx), sess_options=opts, providers=providers)
        report("LOADING_POSE")
        self.pose_sess = ort.InferenceSession(str(pose_onnx), sess_options=opts, providers=providers)

        self.active_providers = self.pose_sess.get_providers()
        self.yolo_input_name = self.yolo_sess.get_inputs()[0].name
        self.pose_input_name = self.pose_sess.get_inputs()[0].name
        self.yolo_interval = yolo_interval
        self.frame_count = 0
        self.cached_center_bbox = None

        print(f"[AI] ONNX Runtime 엔진 활성화 ({self.active_providers})", flush=True)

    def _detect_center_person(self, frame_bgr: np.ndarray, imgsz: int = 320, conf_thresh: float = 0.4):
        """YOLOv8n ONNX로 화면 중앙에 가장 가까운 사람(class 0) 바운딩 박스를 찾는다."""
        h, w = frame_bgr.shape[:2]
        scale = min(imgsz / w, imgsz / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        pad_x = (imgsz - nw) // 2
        pad_y = (imgsz - nh) // 2

        resized = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

        blob = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]

        preds = self.yolo_sess.run(None, {self.yolo_input_name: blob})[0][0]  # (5, 2100) or (84, 2100)
        mask = preds[4, :] > conf_thresh
        if not np.any(mask):
            return None

        cx, cy, bw, bh = preds[:4, mask]
        x1 = np.clip((cx - bw * 0.5 - pad_x) / scale, 0, w)
        y1 = np.clip((cy - bh * 0.5 - pad_y) / scale, 0, h)
        x2 = np.clip((cx + bw * 0.5 - pad_x) / scale, 0, w)
        y2 = np.clip((cy + bh * 0.5 - pad_y) / scale, 0, h)

        valid = (x2 > x1) & (y2 > y1)
        if not np.any(valid):
            return None

        x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
        dist = ((x1 + x2) * 0.5 - w * 0.5) ** 2 + ((y1 + y2) * 0.5 - h * 0.5) ** 2
        best = int(np.argmin(dist))
        return np.array([x1[best], y1[best], x2[best], y2[best]], dtype=np.float32)

    @staticmethod
    def _prepare_pose_input(frame_bgr: np.ndarray, bbox: np.ndarray):
        """바운딩 박스 영역을 192x256으로 Affine 크롭·정규화해 입력 텐서와 역변환 행렬을 만든다."""
        center, scale = _bbox_xyxy_to_cs(bbox)
        warp_mat = _get_warp_matrix(center, scale, (POSE_INPUT_W, POSE_INPUT_H))
        crop_bgr = cv2.warpAffine(frame_bgr, warp_mat, (POSE_INPUT_W, POSE_INPUT_H), flags=cv2.INTER_LINEAR)
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
        norm = (crop_rgb - MEAN_RGB) / STD_RGB
        return np.transpose(norm, (2, 0, 1)), cv2.invertAffineTransform(warp_mat)

    @staticmethod
    def _decode_and_format(simcc_x: np.ndarray, simcc_y: np.ndarray, inv_mat: np.ndarray, w: int, h: int) -> list:
        """SimCC 출력을 원본 영상 0~1 비율 좌표 딕셔너리 리스트(23개 관절)로 변환한다."""
        x_locs = np.argmax(simcc_x, axis=-1).astype(np.float32) / SIMCC_SPLIT_RATIO
        y_locs = np.argmax(simcc_y, axis=-1).astype(np.float32) / SIMCC_SPLIT_RATIO
        scores = np.minimum(np.max(simcc_x, axis=-1), np.max(simcc_y, axis=-1))

        pts_crop = np.concatenate([x_locs[:, None], y_locs[:, None], np.ones((NUM_KEYPOINTS, 1), dtype=np.float32)], axis=1)
        pts_img = pts_crop @ inv_mat.T
        aspect = w / h
        return [
            {
                "id": i,
                "x": round(float(pts_img[i, 0] / w), 4),
                "y": round(float(pts_img[i, 1] / h), 4),
                "score": round(float(scores[i]), 3),
                "image_width": w,
                "image_height": h,
                "image_aspect_ratio": aspect,
            }
            for i in range(NUM_KEYPOINTS)
        ]

    def extract_keypoints_batch(self, frames: list) -> list:
        """여러 영상 프레임을 배치로 묶어 23개 관절 좌표를 고속 추출한다."""
        if not frames:
            return []
        output = [None] * len(frames)
        tensors, metas = [], []

        for idx, frame in enumerate(frames):
            if frame is None:
                continue
            bbox = self._detect_center_person(frame)
            if bbox is None:
                continue
            tensor, inv_mat = self._prepare_pose_input(frame, bbox)
            tensors.append(tensor)
            h, w = frame.shape[:2]
            metas.append((idx, inv_mat, w, h))

        if tensors:
            simcc_x, simcc_y = self.pose_sess.run(None, {self.pose_input_name: np.stack(tensors, axis=0)})
            for b_idx, (orig_idx, inv_mat, w, h) in enumerate(metas):
                output[orig_idx] = self._decode_and_format(simcc_x[b_idx], simcc_y[b_idx], inv_mat, w, h)
        return output

    def extract_keypoints(self, frame: np.ndarray) -> list | None:
        """단일 프레임에서 중앙 대상자의 23개 관절 좌표를 고속 추출한다."""
        if frame is None:
            return None
        h, w = frame.shape[:2]
        self.frame_count += 1
        if self.cached_center_bbox is None or (self.frame_count % self.yolo_interval == 0):
            bbox = self._detect_center_person(frame)
            if bbox is not None:
                self.cached_center_bbox = bbox

        if self.cached_center_bbox is None:
            return None

        tensor, inv_mat = self._prepare_pose_input(frame, self.cached_center_bbox)
        simcc_x, simcc_y = self.pose_sess.run(None, {self.pose_input_name: tensor[None]})
        return self._decode_and_format(simcc_x[0], simcc_y[0], inv_mat, w, h)
