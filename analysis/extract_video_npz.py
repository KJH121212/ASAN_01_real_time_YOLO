"""녹화된 영상 파일에서 2D 관절 좌표와 운동 측정값을 추출해 NPZ/CSV로 저장하는 모듈.

반복 횟수 카운트 없이 전체 프레임의 관절 궤적과 측정값만 뽑아
표준 저장 구조(data/{환자ID}/{날짜}/{운동명}/)로 기록한다.

실행 예시:
    # 단일 영상 처리
    python -m analysis.extract_video_npz --video data/test_video/261109/knee_extension/calibration.mp4 --exercise knee_extension

    # 폴더 안의 모든 영상 일괄 처리
    python -m analysis.extract_video_npz --dir data/test_video/261109/knee_extension/ --exercise knee_extension
"""

import argparse
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.data_manager import DataManager, load_exercise_config
from core.motion_engine import MotionEngine
from utils.filters import OneEuroKeypointFilter
from utils.normalization import PoseNormalizer

KST = timezone(timedelta(hours=9))


def detect_mode_round_set(video_path: Path, mode=None, round_number=None, set_number=None):
    """파일명에 적힌 힌트(cali, round, set 등)로 모드·라운드·세트 번호를 자동 추론한다."""
    stem_lower = video_path.stem.lower()

    if mode is None:
        if any(kw in stem_lower for kw in ("cali", "calib", "calibration")):
            mode = "CALIBRATION"
        else:
            mode = "MAIN"

    if round_number is None:
        round_number = 1
        m = re.search(r"round(\d+)", stem_lower)
        if m:
            round_number = int(m.group(1))

    if set_number is None:
        set_number = 1
        m_set = re.search(r"set(\d+)", stem_lower)
        if m_set:
            set_number = int(m_set.group(1))
        else:
            m_test = re.search(r"test(\d+)", stem_lower)
            if m_test:
                set_number = int(m_test.group(1))

    return mode, round_number, set_number


def process_video_to_npz(
    video_path: Path,
    exercise_name: str = "knee_extension",
    patient_id: str = "0000",
    mode: str = None,
    round_number: int = None,
    set_number: int = None,
    batch_size: int = 16,
    engine = None,
    on_progress = None,
):
    """영상 1개를 읽어 포즈 배치 추론 후 표준 NPZ와 CSV 파일로 저장한다."""
    video_path = Path(video_path)
    if not video_path.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Infer mode, round, set from filename if not provided
    mode, round_number, set_number = detect_mode_round_set(video_path, mode, round_number, set_number)

    print(f"\n========================================================")
    print(f"[VIDEO TO NPZ] Processing: {video_path.name}")
    print(f"  Path        : {video_path}")
    print(f"  Exercise    : {exercise_name}")
    print(f"  Patient ID  : {patient_id}")
    print(f"  Session Mode: {mode} (Round {round_number}, Set {set_number})")
    print(f"  Batch size  : {batch_size}")
    print(f"========================================================")

    # 1. Video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if (np.isfinite(fps) and fps > 0) else 30.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_duration = 1.0 / fps

    print(f"  Video Info  : {width}x{height} @ {fps:.2f}fps ({frame_count} frames, ~{frame_count / fps:.2f}s)")

    # 2. SkeletonEngine initialization (lazy load if needed)
    if engine is None:
        from core.skeleton_engine import SkeletonEngine
        print("  Initializing SkeletonEngine (YOLO + RTMPose)...")
        engine = SkeletonEngine()

    # 3. MotionEngine & SessionController initialization (respecting exercise & mode!)
    config = load_exercise_config(exercise_name)
    from core.session_controller import SessionController
    motion_engine = MotionEngine(config=config, mode=mode)
    motion_engine.exercise_name = exercise_name
    controller = SessionController(motion_engine=motion_engine)
    filter_engine = OneEuroKeypointFilter()

    # 4. Standard DataManager initialization (exact same saving structure)
    data_manager = DataManager(player_id=patient_id, log_root=ROOT / "data")
    data_manager.begin_session(
        exercise_name=exercise_name,
        mode=mode,
        round_number=round_number,
        set_number=set_number,
        video_fps=fps,
        filename_stem=video_path.stem,
    )

    # 5. Extract all frames in batches
    all_raw_keypoints = []
    batch_frames = []
    decoded = 0
    t0 = time.time()

    from network.camera import downscale_frame_if_needed
    infer_w, infer_h = (width, height)
    if width > 1280 or height > 720:
        from network.camera import compute_downscaled_size
        infer_w, infer_h = compute_downscaled_size(width, height, 1280, 720)
        print(f"  Resolution  : High-res GoPro/video ({width}x{height}) -> auto-downscaling to {infer_w}x{infer_h} for fast inference")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if batch_frames:
                    kpts_batch = engine.extract_keypoints_batch(batch_frames)
                    all_raw_keypoints.extend(kpts_batch)
                break

            if (infer_w, infer_h) != (width, height):
                frame = downscale_frame_if_needed(frame, 1280, 720)
            decoded += 1
            batch_frames.append(frame)

            if len(batch_frames) >= batch_size:
                kpts_batch = engine.extract_keypoints_batch(batch_frames)
                all_raw_keypoints.extend(kpts_batch)
                batch_frames.clear()

                elapsed = time.time() - t0
                speed = decoded / elapsed if elapsed > 0 else 0
                pct = (decoded / frame_count) * 100 if frame_count > 0 else 0
                print(f"\r  Extracting poses [{pct:5.1f}%] {decoded}/{frame_count} frames ({speed:.1f} fps)...", end="", flush=True)
                if on_progress:
                    on_progress(decoded, frame_count, speed, f"포즈 추출 중... {decoded}/{frame_count} ({speed:.1f} FPS)")
    finally:
        cap.release()

    total_extracted = len(all_raw_keypoints)
    print(f"\r  Extracted {total_extracted} frames in {time.time() - t0:.2f}s! Writing standard session log...")

    # 6. Append frames to SessionLog without repetition counting
    thresholds = motion_engine.get_count_thresholds()

    for frame_idx, raw_kpts in enumerate(all_raw_keypoints):
        vtime = frame_idx * frame_duration

        # Filter keypoints
        filtered_kpts = filter_engine.update(raw_kpts, timestamp=vtime) if raw_kpts else None

        # Occlusion & coordinate space (normalized vs unnormalized per exercise/mode)
        is_occluded = controller.check_occlusion(filtered_kpts) if filtered_kpts else True
        measured_pts = controller.measurement_points(filtered_kpts) if (filtered_kpts and not is_occluded) else None
        points_map = {p['id']: p for p in measured_pts} if measured_pts else {}
        val_l = motion_engine._compute_side_value('left', points_map) if points_map else None
        val_r = motion_engine._compute_side_value('right', points_map) if points_map else None

        left_data = {
            "val": val_l,
            "rep_count": 0,
            "total_score": 0.0,
            "score_delta": 0.0,
            "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "",
            "is_count_updated": False,
        }
        right_data = {
            "val": val_r,
            "rep_count": 0,
            "total_score": 0.0,
            "score_delta": 0.0,
            "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "",
            "is_count_updated": False,
        }

        # Write frame using official ASAN SessionLog method
        data_manager.session_log.append_frame(
            elapsed_sec=vtime,
            frame_idx=frame_idx,
            raw_keypoints=raw_kpts,
            left=left_data,
            right=right_data,
            metric_type=motion_engine.metric_type,
            thresholds=thresholds,
            threshold_version=0,
            is_occluded=is_occluded,
            calib_step="EXTRACTED",
            video_frame_idx=frame_idx,
            use_normalization=motion_engine.use_normalization,
        )

        if on_progress and (frame_idx % 30 == 0 or frame_idx == total_extracted - 1):
            on_progress(frame_idx + 1, total_extracted, 0.0, f"로그 및 NPZ 작성 중... {frame_idx + 1}/{total_extracted}")

    # 7. Finalize session (standard compressed NPZ + CSV)
    data_manager.session_log.info["source_video"] = str(video_path.resolve())
    data_manager.session_log.info["video_mp4"] = str(video_path.resolve())
    summary = data_manager.finish_session(rep_rows=[], status="COMPLETED")
    npz_path = Path(summary["skeleton_npz"])
    csv_path = Path(summary["frame_csv"])

    print(f"\n  [OK] Saved NPZ: {npz_path}")
    print(f"  [OK] Saved CSV: {csv_path}")
    print(f"  Total {total_extracted} frames saved in original ASAN format (no counting).")
    return npz_path, csv_path


def main():
    """CLI 인자를 받아 단일 영상 또는 폴더 전체의 NPZ 추출 작업을 실행한다."""
    parser = argparse.ArgumentParser(description="Extract video keypoints to standard ASAN NPZ without counting.")
    parser.add_argument("--video", "-v", type=str, default=None, help="Path to single video file")
    parser.add_argument("--dir", "-d", type=str, default=None, help="Directory containing video files")
    parser.add_argument("--exercise", "-e", type=str, default="knee_extension", help="Exercise name (default: knee_extension)")
    parser.add_argument("--patient", "-p", type=str, default="0000", help="Patient ID (default: 0000)")
    parser.add_argument("--mode", "-m", type=str, default=None, help="Mode: CALIBRATION or MAIN (default: auto from filename)")
    parser.add_argument("--round", "-r", type=int, default=None, help="Round number (default: auto from filename)")
    parser.add_argument("--set", "-s", type=int, default=None, help="Set number (default: auto from filename)")
    parser.add_argument("--batch-size", "-b", type=int, default=16, help="Inference batch size (default: 16)")

    args = parser.parse_args()

    engine = None
    if args.video:
        process_video_to_npz(
            Path(args.video),
            exercise_name=args.exercise,
            patient_id=args.patient,
            mode=args.mode,
            round_number=args.round,
            set_number=args.set,
            batch_size=args.batch_size,
        )
    elif args.dir:
        folder = Path(args.dir)
        videos = sorted([p for p in folder.glob("*") if p.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv")])
        if not videos:
            print(f"[ERROR] No video files found in {folder}")
            return

        from core.skeleton_engine import SkeletonEngine
        print(f"Found {len(videos)} videos in {folder}. Initializing SkeletonEngine...")
        engine = SkeletonEngine()

        for idx, vid in enumerate(videos, 1):
            print(f"\n>>> [{idx}/{len(videos)}] Processing {vid.name}...")
            process_video_to_npz(
                vid,
                exercise_name=args.exercise,
                patient_id=args.patient,
                mode=args.mode,
                round_number=args.round,
                set_number=args.set,
                batch_size=args.batch_size,
                engine=engine,
            )
    else:
        print("[ERROR] Please provide --video or --dir. Run with --help for details.")


if __name__ == "__main__":
    main()
