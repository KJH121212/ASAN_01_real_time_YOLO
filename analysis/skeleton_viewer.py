"""Render pure skeleton motion on a black canvas (Normalized or Unnormalized).

Provides frame-by-frame image rendering and fast H.264 MP4 / GIF animation generation.
"""

from pathlib import Path
import subprocess
import tempfile
import time

import cv2
import numpy as np

from analysis.video_overlay import find_ffmpeg
from utils.normalization import PoseNormalizer

COCO_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4),           # Head
    (5, 6), (5, 11), (6, 12), (11, 12),       # Torso
    (5, 7), (7, 9),                            # Left Arm
    (6, 8), (8, 10),                           # Right Arm
    (11, 13), (13, 15),                        # Left Leg
    (12, 14), (14, 16)                         # Right Leg
]

# Color palette: BGR format
LEFT_ARM_COLOR = (255, 200, 0)     # Cyan/Sky
RIGHT_ARM_COLOR = (0, 140, 255)    # Orange
LEFT_LEG_COLOR = (255, 255, 0)     # Yellow
RIGHT_LEG_COLOR = (0, 75, 255)     # Red
TORSO_COLOR = (0, 255, 128)        # Mint green
HEAD_COLOR = (200, 200, 200)       # Light gray


def get_edge_color(i, j):
    if i in (11, 13, 15) and j in (11, 13, 15):
        return LEFT_LEG_COLOR
    if i in (12, 14, 16) and j in (12, 14, 16):
        return RIGHT_LEG_COLOR
    if i in (5, 7, 9) and j in (5, 7, 9):
        return LEFT_ARM_COLOR
    if i in (6, 8, 10) and j in (6, 8, 10):
        return RIGHT_ARM_COLOR
    if i in (5, 6, 11, 12) and j in (5, 6, 11, 12):
        return TORSO_COLOR
    return HEAD_COLOR


def draw_skeleton_frame(
    raw_keypoints_17x3,
    is_norm: bool = True,
    width: int = 640,
    height: int = 640,
    confidence: float = 0.35,
    normalizer: PoseNormalizer = None,
    aspect_ratio: float = 1.0,
    info_text: str = "",
    hide_face: bool = True,
) -> np.ndarray:
    """Draw a single frame's skeleton on a black canvas."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    if raw_keypoints_17x3 is None or len(raw_keypoints_17x3) < 17:
        return canvas

    pts_dict = {}
    if is_norm:
        if normalizer is None:
            normalizer = PoseNormalizer(conf_threshold=confidence, invert_y=True)
        frame_list = [
            dict(
                id=i,
                x=float(raw_keypoints_17x3[i, 0]),
                y=float(raw_keypoints_17x3[i, 1]),
                score=float(raw_keypoints_17x3[i, 2]),
                image_aspect_ratio=float(aspect_ratio),
            )
            for i in range(17)
        ]
        norm_list = normalizer.normalize(frame_list)
        if norm_list:
            # Map normalized coords (hip center=(0,0), torso=1.0) into canvas
            scale = height / 2.8
            cx = width / 2.0
            cy = height / 2.0 - 10
            for p in norm_list:
                if p["score"] >= confidence:
                    px = int(round(cx + p["x"] * scale))
                    py = int(round(cy - p["y"] * scale))  # Invert Y back to canvas
                    pts_dict[p["id"]] = (px, py, p["score"])
    else:
        # Unnormalized [0, 1] coords mapped to width, height
        for i in range(17):
            x, y, score = raw_keypoints_17x3[i]
            if np.isfinite([x, y, score]).all() and score >= confidence:
                px = int(round(x * width))
                py = int(round(y * height))
                pts_dict[i] = (px, py, score)

    # Draw skeleton bones (lines)
    for i, j in COCO_EDGES:
        if hide_face and (i < 5 or j < 5):
            continue
        if i in pts_dict and j in pts_dict:
            p1 = pts_dict[i]
            p2 = pts_dict[j]
            color = get_edge_color(i, j)
            cv2.line(canvas, (p1[0], p1[1]), (p2[0], p2[1]), color, 3, cv2.LINE_AA)

    # Draw joint nodes (circles)
    for idx, (px, py, score) in pts_dict.items():
        if hide_face and idx < 5:
            continue
        if idx in (15, 16):       # Ankles
            radius = 7
            color = (0, 255, 255)
        elif idx in (13, 14):     # Knees
            radius = 6
            color = (0, 200, 255)
        elif idx in (11, 12):     # Hips
            radius = 6
            color = (0, 255, 120)
        else:
            radius = 4
            color = (220, 220, 220)
        cv2.circle(canvas, (px, py), radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

    # Overlay mode watermark
    if info_text:
        cv2.putText(canvas, info_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1, cv2.LINE_AA)

    mode_label = "NORMALIZED" if is_norm else "UNNORMALIZED"
    cv2.putText(canvas, f"[{mode_label}]", (width - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (140, 140, 140), 1, cv2.LINE_AA)
    return canvas


def render_skeleton_video(
    log: dict,
    output_path: Path,
    is_norm: bool = True,
    fps: float = 30.0,
    width: int = 640,
    height: int = 640,
    confidence: float = 0.35,
    hide_face: bool = True,
    on_progress = None,
) -> Path:
    """Render all frames of log into an MP4 video on black canvas using H.264."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kpts = log.get("keypoints")
    if kpts is None or len(kpts) == 0:
        raise ValueError("NPZ log does not contain 'keypoints'.")

    times = log.get("timestamps", np.arange(len(kpts)) / fps)
    aspects = log.get("image_aspect_ratio", np.ones(len(kpts)))
    if np.isscalar(aspects):
        aspects = np.full(len(kpts), float(aspects))

    total_frames = len(kpts)
    normalizer = PoseNormalizer(conf_threshold=confidence, invert_y=True) if is_norm else None

    # Write intermediate AVI
    with tempfile.NamedTemporaryFile(suffix=".avi", delete=False) as tmp_f:
        tmp_avi = Path(tmp_f.name)

    writer = cv2.VideoWriter(str(tmp_avi), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    try:
        for f in range(total_frames):
            asp = float(aspects[f]) if f < len(aspects) else 1.0
            t_sec = float(times[f]) if f < len(times) else f / fps
            txt = f"Frame {f+1}/{total_frames} | {t_sec:.2f}s"
            canvas = draw_skeleton_frame(
                kpts[f],
                is_norm=is_norm,
                width=width,
                height=height,
                confidence=confidence,
                normalizer=normalizer,
                aspect_ratio=asp,
                info_text=txt,
                hide_face=hide_face,
            )
            writer.write(canvas)
            if on_progress and (f % 15 == 0 or f == total_frames - 1):
                pct = (f + 1) / total_frames
                on_progress(pct, f"스켈레톤 렌더링 중... {f+1}/{total_frames} 프레임")
    finally:
        writer.release()

    # Convert to web-compatible H.264 MP4 using bundled FFmpeg
    ffmpeg = find_ffmpeg()
    if ffmpeg:
        cmd = [
            ffmpeg, "-y",
            "-i", str(tmp_avi),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-preset", "veryfast",
            "-crf", "22",
            str(output_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            tmp_avi.replace(output_path)
    else:
        tmp_avi.replace(output_path)

    tmp_avi.unlink(missing_ok=True)
    return output_path

