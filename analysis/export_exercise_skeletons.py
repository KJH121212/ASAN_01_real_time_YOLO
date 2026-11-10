"""Pick one characteristic pose per exercise and draw only its body skeleton.

Run from the project root: python -m analysis.export_exercise_skeletons
Reuse extracted poses:     python -m analysis.export_exercise_skeletons --reuse
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / 'analysis/skeleton_images'
CACHE = ROOT / 'analysis/outputs/exercise_skeletons/pose_candidates.json'
VIDEOS = {
    'biceps_curl': 'biceps_curl/AI_dataset__N01__N01_Treatment__frontal__biceps_curl.mp4',
    'clamshell': 'clamshell/AI_dataset__N01__N01_Ward__frontal__clamshell_supine.mp4',
    'knee_extension': 'leg_extension/AI_dataset__N04__N04_Treatment__frontal__knee_extension_chair.mp4',
    'hip_knee_flexion': 'leg_raise/AI_dataset__N01__N01_Ward__frontal__hip_knee_flexion.mp4',
    'shoulder_press': 'shoulder_press/game_test__norm_angle_cam__frontal__shoulder_press__stand1.mov',
    'slr': 'slr/AI_dataset__N01__N01_Treatment__frontal__slr_bobath_table.mp4',
}
EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11),
         (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def angle(points, a, b, c):
    u, v = points[a] - points[b], points[c] - points[b]
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / max(np.linalg.norm(u)*np.linalg.norm(v), 1e-8), -1, 1))))


def pose_score(exercise, xy):
    """Rank expressive poses in image coordinates, not patient count/quality scores."""
    torso = max(np.linalg.norm((xy[5]+xy[6]-xy[11]-xy[12])/2), 1.)
    if exercise == 'biceps_curl':
        return 180 - np.mean([angle(xy, 5, 7, 9), angle(xy, 6, 8, 10)])
    if exercise == 'shoulder_press':
        lift = ((xy[5, 1]-xy[9, 1])+(xy[6, 1]-xy[10, 1])) / (2*torso)
        return np.mean([angle(xy, 5, 7, 9), angle(xy, 6, 8, 10)]) / 180 + lift
    if exercise == 'clamshell':
        return np.mean([angle(xy, 12, 11, 13), angle(xy, 11, 12, 14)])
    if exercise == 'knee_extension':
        return -min((xy[15, 1]-xy[13, 1])/torso, (xy[16, 1]-xy[14, 1])/torso)
    if exercise == 'hip_knee_flexion':
        return 180 - min(angle(xy, 5, 11, 13), angle(xy, 6, 12, 14))
    # SLR: a raised, relatively straight leg is more characteristic than knee flexion.
    return max((180-angle(xy, shoulder, hip, knee))*np.clip(angle(xy, hip, knee, ankle)/160, 0, 1)
               for shoulder, hip, knee, ankle in ((5, 11, 13, 15), (6, 12, 14, 16)))


def extract(samples):
    # Import the same engine as the socket server only when inference is required.
    from core.skeleton_engine import SkeletonEngine
    engine = SkeletonEngine()
    output = {}
    for exercise, relative in VIDEOS.items():
        video = ROOT / 'data/test_video' / relative
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f'Cannot open {video}')
        try:
            fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if not np.isfinite(fps) or fps <= 0 or total < 1:
                raise RuntimeError(f'Invalid video timing: {video}')
            indices = np.unique(np.linspace(int(.03*total), int(.96*total), samples).astype(int))
            candidates = []
            for start in range(0, len(indices), 4):
                frames, frame_ids = [], []
                for index in indices[start:start+4]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                    ok, frame = cap.read()
                    if ok:
                        h, w = frame.shape[:2]
                        scale = min(1., 1280/max(h, w))
                        if scale < 1:
                            frame = cv2.resize(frame, (round(w*scale), round(h*scale)))
                        frames.append(frame)
                        frame_ids.append(int(index))
                if not frames:
                    continue
                for frame_id, frame, pose in zip(frame_ids, frames, engine.extract_keypoints_batch(frames)):
                    if not pose:
                        continue
                    points = np.full((17, 3), np.nan)
                    for p in pose:
                        points[p['id']] = (p['x']*frame.shape[1], p['y']*frame.shape[0], p['score'])
                    if not np.isfinite(points[5:]).all():
                        continue
                    candidates.append(dict(frame_idx=frame_id, timestamp=frame_id/fps,
                        width=frame.shape[1], height=frame.shape[0], points=points.tolist(),
                        pose_score=float(pose_score(exercise, points[:, :2])),
                        confidence=float(points[5:, 2].mean()), min_confidence=float(points[5:, 2].min())))
                print(f'{exercise}: sampled {min(start+4, len(indices))}/{len(indices)}', flush=True)
            output[exercise] = dict(video=str(video.relative_to(ROOT)), candidates=candidates)
        finally:
            cap.release()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(output, indent=2), encoding='utf-8')
    return output


def draw(points, size=1024):
    """Black background, no face joints, text, source image, bar or measurements."""
    points = np.asarray(points, dtype=float)
    valid = np.isfinite(points).all(axis=1) & (points[:, 2] >= .35)
    valid[:5] = False
    xy = points[:, :2].copy()
    lo, hi = xy[valid].min(axis=0), xy[valid].max(axis=0)
    scale = size*.76 / max(float((hi-lo).max()), 1.)
    xy = (xy-(lo+hi)/2)*scale + size/2
    canvas = np.zeros((size*2, size*2, 3), dtype=np.uint8)
    pixels = {i: tuple(np.round(xy[i]*2).astype(int)) for i in range(5, 17) if valid[i]}
    for a, b in EDGES:
        if a in pixels and b in pixels:
            cv2.line(canvas, pixels[a], pixels[b], (240, 240, 240), 10, cv2.LINE_AA)
    for point in pixels.values():
        cv2.circle(canvas, point, 15, (135, 235, 100), -1, cv2.LINE_AA)
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse', action='store_true', help='Render from the extracted pose cache')
    parser.add_argument('--samples', type=int, default=48)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error('--samples must be at least 2')
    data = json.loads(CACHE.read_text(encoding='utf-8')) if args.reuse else extract(args.samples)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = {}
    for exercise, record in data.items():
        candidates = [p for p in record['candidates'] if p['min_confidence'] >= .35]
        if not candidates:
            raise RuntimeError(f'No complete body skeleton for {exercise}; choose another video')
        # First find the expressive poses, then prefer the clearest of those.
        # Filtering by confidence first can accidentally pick a resting position.
        expressive = sorted(candidates, key=lambda p: p['pose_score'], reverse=True)
        expressive = expressive[:max(1, int(np.ceil(len(expressive)*.1)))]
        selected = max(expressive, key=lambda p: p['confidence'])
        target = OUTPUT / f'{exercise}_skeleton.png'
        if not cv2.imwrite(str(target), draw(selected['points'])):
            raise RuntimeError(f'Cannot save {target}')
        sources[exercise] = dict(video=record['video'], **selected)
        print(f'Saved {target.name}: frame {selected["frame_idx"]}, {selected["timestamp"]:.2f}s', flush=True)
    (OUTPUT/'sources.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
