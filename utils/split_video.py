"""Video splitting utility based on split.txt timestamps.

Usage:
    python utils/split_video.py --target data/test_video/261109
    python utils/split_video.py --video data/test_video/261109/knee_extension/GX011110.MP4 --split-txt data/test_video/261109/knee_extension/split.txt
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import cv2


def parse_timestamp(ts_str: str) -> float:
    """Convert timestamp string (HH:MM:SS, MM:SS, or seconds) to float seconds."""
    ts_str = ts_str.strip()
    parts = ts_str.split(":")
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    elif len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    else:
        raise ValueError(f"Invalid timestamp format: {ts_str}")


def parse_split_file(split_path: Path):
    """Parse split.txt containing: start_time, end_time, label.

    Example line:
        00:11, 00:24, calibration
    """
    segments = []
    with open(split_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                print(f"[WARN] Line {line_num} in {split_path} has fewer than 3 columns: '{line}'. Skipped.")
                continue
            start_str, end_str, label = parts[0], parts[1], parts[2]
            try:
                start_sec = parse_timestamp(start_str)
                end_sec = parse_timestamp(end_str)
                segments.append((start_sec, end_sec, label))
            except Exception as e:
                print(f"[WARN] Failed to parse line {line_num}: '{line}' ({e}). Skipped.")
    return segments


def split_with_ffmpeg(video_path: Path, start_sec: float, end_sec: float, output_path: Path) -> bool:
    """Try splitting video with ffmpeg stream copy (lossless and instant)."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if not ffmpeg_bin:
        return False
    duration = end_sec - start_sec
    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss", f"{start_sec:.3f}",
        "-i", str(video_path),
        "-t", f"{duration:.3f}",
        "-c", "copy",
        str(output_path)
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return res.returncode == 0
    except Exception:
        return False


def split_with_opencv(video_path: Path, start_sec: float, end_sec: float, output_path: Path):
    """Split video using OpenCV VideoCapture & VideoWriter."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = max(0, int(round(start_sec * fps)))
    end_frame = min(total_frames, int(round(end_sec * fps)))
    frames_to_write = max(0, end_frame - start_frame)

    # Seek directly to start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    written = 0
    t0 = time.time()
    try:
        while written < frames_to_write:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            written += 1
            if written % 100 == 0 or written == frames_to_write:
                elapsed = time.time() - t0
                pct = (written / frames_to_write) * 100 if frames_to_write > 0 else 100
                speed = written / elapsed if elapsed > 0 else 0
                print(f"\r  Writing [{pct:5.1f}%] {written}/{frames_to_write} frames ({speed:.1f} fps)...", end="", flush=True)
    finally:
        cap.release()
        out.release()
    print()


def split_single_video(video_path: Path, split_path: Path, output_dir: Path = None):
    """Split a single video according to its split.txt."""
    if output_dir is None:
        output_dir = video_path.parent / "split"
    output_dir.mkdir(parents=True, exist_ok=True)

    segments = parse_split_file(split_path)
    if not segments:
        print(f"[WARN] No valid segments found in {split_path}")
        return []

    print(f"\n========================================================")
    print(f"Target Video : {video_path}")
    print(f"Split Config : {split_path}")
    print(f"Output Dir   : {output_dir}")
    print(f"Segments ({len(segments)}):")
    for s, e, lbl in segments:
        print(f"  - {lbl:15s}: {s:.2f}s ~ {e:.2f}s (duration: {e - s:.2f}s)")
    print(f"========================================================")

    created_files = []
    has_ffmpeg = shutil.which("ffmpeg") is not None

    for idx, (start_sec, end_sec, label) in enumerate(segments, 1):
        # Clean label for safe filename
        safe_label = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in label)
        out_filename = f"{safe_label}.mp4"
        out_path = output_dir / out_filename

        print(f"\n[{idx}/{len(segments)}] Splitting '{label}' ({start_sec:.2f}s ~ {end_sec:.2f}s)...")
        success = False
        if has_ffmpeg:
            print("  Using ffmpeg stream copy...")
            success = split_with_ffmpeg(video_path, start_sec, end_sec, out_path)
            if success:
                print(f"  [OK] Saved to: {out_path} (instant ffmpeg copy)")

        if not success:
            if has_ffmpeg:
                print("  ffmpeg copy failed or unavailable. Falling back to OpenCV...")
            else:
                print("  Encoding with OpenCV...")
            split_with_opencv(video_path, start_sec, end_sec, out_path)
            print(f"  [OK] Saved to: {out_path}")

        created_files.append(out_path)

    print(f"\n[DONE] Successfully split {len(created_files)} clips in {output_dir}\n")
    return created_files


def auto_discover_and_split(target_path: Path, output_dir: Path = None):
    """Auto discover video and split.txt files in target directory and process them."""
    if not target_path.exists():
        raise FileNotFoundError(f"Path does not exist: {target_path}")

    # Case 1: Pointed directly to a directory
    if target_path.is_dir():
        split_files = list(target_path.rglob("split.txt"))
        if not split_files:
            print(f"[ERROR] No 'split.txt' found under {target_path}")
            return

        for split_file in split_files:
            folder = split_file.parent
            # Find video file in the same folder
            video_candidates = [
                p for p in folder.glob("*")
                if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv") and not p.name.startswith("split_")
            ]
            if not video_candidates:
                print(f"[WARN] No video file found in {folder} for {split_file.name}. Skipped.")
                continue

            # Pick the primary video (e.g. GX*.MP4 or largest video)
            primary_video = max(video_candidates, key=lambda p: p.stat().st_size)
            target_out = output_dir if output_dir else folder / "split"
            split_single_video(primary_video, split_file, target_out)


def main():
    parser = argparse.ArgumentParser(description="Split video using split.txt timestamps.")
    parser.add_argument(
        "--target", "-t",
        type=str,
        default="data/test_video/261110",
        help="Target folder to search for split.txt and video (default: data/test_video/26111)"
    )
    parser.add_argument("--video", "-v", type=str, default=None, help="Explicit video file path")
    parser.add_argument("--split-txt", "-s", type=str, default=None, help="Explicit split.txt file path")
    parser.add_argument("--out-dir", "-o", type=str, default=None, help="Output directory for split clips")

    args = parser.parse_args()

    if args.video and args.split_txt:
        video_p = Path(args.video)
        split_p = Path(args.split_txt)
        out_p = Path(args.out_dir) if args.out_dir else None
        split_single_video(video_p, split_p, out_p)
    else:
        target_p = Path(args.target)
        out_p = Path(args.out_dir) if args.out_dir else None
        auto_discover_and_split(target_p, out_p)


if __name__ == "__main__":
    main()

