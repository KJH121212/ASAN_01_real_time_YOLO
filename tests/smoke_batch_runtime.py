"""Manual smoke test with local YOLO/RTMPose weights and a real video."""

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', default=str(ROOT / 'data/test/biceps_curl/patient_1/calibration.mp4'))
    parser.add_argument('--device', default=None)
    args = parser.parse_args()

    print('Loading YOLO and RTMPose...', flush=True)
    import cv2
    from core.skeleton_engine import SkeletonEngine
    engine = SkeletonEngine(device=args.device)
    cap = cv2.VideoCapture(args.video)
    frames = []
    try:
        for _ in range(3):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError('Smoke test needs a readable video with at least 3 frames')
            frames.append(frame)
    finally:
        cap.release()

    print('Testing a two-frame batch, a missing frame, and a one-frame tail...', flush=True)
    started = time.perf_counter()
    results = engine.extract_keypoints_batch([None, frames[0], frames[1]])
    tail = engine.extract_keypoints_batch([frames[2]])
    assert len(results) == 3 and results[0] is None and len(tail) == 1
    found = [pose for pose in results + tail if pose is not None]
    assert found, 'No people detected in the first three frames; choose another video'
    assert all(len(pose) == 17 for pose in found)
    print(f'BATCH_SMOKE_OK device={engine.device} poses={len(found)} '
          f'seconds={time.perf_counter() - started:.3f}', flush=True)


if __name__ == '__main__':
    main()
