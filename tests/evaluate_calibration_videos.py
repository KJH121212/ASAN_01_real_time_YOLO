"""Run real offline calibration on a video corpus, isolated from patient profiles.

python tests/evaluate_calibration_videos.py --inventory-only
python tests/evaluate_calibration_videos.py --output data/calibration_experiments/baseline
"""

import argparse
import asyncio
from collections import Counter, defaultdict
from contextlib import redirect_stdout, redirect_stderr
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXERCISES = {'biceps_curl': 'biceps_curl', 'shoulder_press': 'shoulder_press',
             'clamshell': 'clamshell', 'slr': 'slr',
             'leg_raise': 'hip_knee_flexion', 'leg_extension': 'knee_extension'}
EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}


def inventory(root):
    entries = []
    for path in sorted(root.rglob('*')):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        relative = path.relative_to(root)
        folder = relative.parts[0]
        cap = cv2.VideoCapture(str(path))
        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            entries.append(dict(video_id=hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12],
                source=str(path), relative_path=relative.as_posix(), folder=folder,
                exercise=EXERCISES.get(folder), opened=cap.isOpened(), fps=fps,
                frames=frames, duration_sec=frames / fps if fps > 0 else 0,
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                size_bytes=path.stat().st_size, modified_ns=path.stat().st_mtime_ns))
        finally:
            cap.release()
    return entries


def write_csv(path, records):
    if not records:
        return
    fields = list(dict.fromkeys(key for row in records for key in row))
    temporary = path.with_suffix('.csv.tmp')
    with temporary.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


class CaptureEngine:
    def __init__(self, engine):
        self.engine, self.raw = engine, []

    def extract_keypoints_batch(self, frames):
        result = self.engine.extract_keypoints_batch(frames)
        self.raw.extend(result)
        return result

    def save(self, path, fps):
        matrix = np.full((len(self.raw), 17, 3), np.nan, dtype=np.float32)
        matrix[:, :, 2] = 0
        for index, points in enumerate(self.raw):
            for point in points or []:
                matrix[index, point['id']] = [point['x'], point['y'], point['score']]
        np.savez_compressed(path, keypoints=matrix, fps=fps)


class ResultSink:
    def __init__(self):
        self.messages = []

    async def send(self, encoded):
        self.messages.append(json.loads(encoded))


def analyze_frames(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    count = len(rows)
    last = rows[-1] if rows else {}
    occluded = sum(row['is_occluded'] == 'True' for row in rows)
    result = dict(analyzed_frames=count, analyzed_sec=float(last.get('timestamp_ms', 0)) / 1000,
                  finished=last.get('calib_step') == 'FINISHED', final_step=last.get('calib_step', ''),
                  occluded_percent=round(100 * occluded / count, 2) if count else 100,
                  collection_frames=sum(row['calib_step'] == 'COLLECTING' for row in rows))
    reset_count = 0
    for previous, current in zip(rows, rows[1:]):
        if previous['calib_step'] != 'FULL_BODY_CHECK' and current['calib_step'] == 'FULL_BODY_CHECK':
            reset_count += 1
    result['occlusion_resets'] = reset_count
    for side in ('left', 'right'):
        values = [float(row[f'{side}_value']) for row in rows if row[f'{side}_value'] != '']
        result[f'{side}_final_reps'] = int(last.get(f'{side}_rep_count', 0))
        result[f'{side}_max_reps'] = max((int(row[f'{side}_rep_count']) for row in rows), default=0)
        result[f'{side}_returning_percent'] = round(100 * sum(row[f'{side}_state'] == 'RETURNING'
                                                               for row in rows) / count, 2) if count else 0
        result[f'{side}_p05'] = round(float(np.percentile(values, 5)), 4) if values else ''
        result[f'{side}_p95'] = round(float(np.percentile(values, 95)), 4) if values else ''
    return result


def report(output, results):
    groups = defaultdict(list)
    for row in results:
        groups[row['folder']].append(row)
    text = ['# Calibration baseline experiment', '',
            'Completion means FINISHED (both sides 3 reps plus cooldown), not verified counting accuracy.',
            'Original offline pipeline, all frames at source FPS, One Euro filter, existing default ROM.',
            'No live 5-second preparation delay. Patient profiles are isolated in this experiment.',
            'Leg-raise variants share hip_knee_flexion provisionally. No production algorithm changes.', '',
            '| Folder | Videos | Finished | EOF incomplete | Errors |', '|---|---:|---:|---:|---:|']
    for folder, rows in sorted(groups.items()):
        counts = Counter(row['result'] for row in rows)
        text.append(f"| {folder} | {len(rows)} | {counts['FINISHED']} | {counts['EOF_NOT_FINISHED']} | {counts['ERROR']} |")
    text += ['', '## Results', '',
             '- inventory.csv: source file metadata and folder-to-exercise mapping.',
             '- results.csv: completion, counts, occlusion, observed ranges and estimated thresholds.',
             '- raw_keypoints/: full-video model output for later replay without another GPU inference.',
             '- sessions/: production CSV/NPZ/summary logs and isolated calibration profiles.',
             '- diagnostics/: per-video runtime output and errors.', '',
             'The existing server labels EOF as COMPLETED and saves thresholds even without FINISHED.',
             'Use result/finished in results.csv to judge completion; incomplete thresholds are not validated.']
    (output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


async def run(args, entries, output):
    from core.skeleton_engine import SkeletonEngine
    from network.socket_server import ExerciseSocketServer
    import torch
    print('Loading real models...', flush=True)
    engine = SkeletonEngine(device=args.device)
    server = ExerciseSocketServer(engine_factory=lambda report: engine)
    (output / 'environment.json').write_text(json.dumps({
        'python': platform.python_version(), 'torch': torch.__version__, 'opencv': cv2.__version__,
        'device': engine.device, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'batch_size': args.batch_size, 'video_root': str(args.video_root),
        'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        for name in ('core/skeleton_engine.py', 'core/motion_engine.py',
                                     'core/session_controller.py', 'utils/filters.py',
                                     'utils/normalization.py', 'network/socket_server.py')},
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    config = ROOT / 'configs/exercise_kpt_config_coco.json'
    (output / 'exercise_config_snapshot.json').write_bytes(config.read_bytes())
    results = []
    for index, entry in enumerate(entries, 1):
        row = dict(entry)
        print(f"[{index}/{len(entries)}] START {entry['relative_path']} ({entry['duration_sec']:.1f}s)", flush=True)
        started = time.perf_counter()
        capture = CaptureEngine(engine)
        server.skeleton_engine = capture
        sink = ResultSink()
        diagnostic_path = output / 'diagnostics' / f"{entry['video_id']}.log"
        try:
            with diagnostic_path.open('w', encoding='utf-8') as diagnostic, redirect_stdout(diagnostic), redirect_stderr(diagnostic):
                await server.process_offline_video(sink, {
                    'video_path': entry['source'], 'exercise_name': entry['exercise'],
                    'patient_id': 'video_' + entry['video_id'], 'patient_name': 'Calibration experiment',
                    'mode': 'CALIBRATION', 'batch_size': args.batch_size, 'record_video': False,
                    'log_root': str(output / 'sessions'), 'session': 1, 'round': 0, 'set': 0})
            completed = next(message for message in sink.messages if message['type'] == 'VIDEO_ANALYSIS_COMPLETED')
            row.update(analyze_frames(completed['summary']['frame_csv']))
            row.update(result='FINISHED' if row['finished'] else 'EOF_NOT_FINISHED',
                       session_directory=completed['summary']['directory'])
            for side, limits in (completed.get('thresholds') or {}).items():
                row[f'{side}_estimated_start'] = limits['start_val']
                row[f'{side}_estimated_target'] = limits['target_val']
            capture.save(output / 'raw_keypoints' / f"{entry['video_id']}.npz", entry['fps'])
        except Exception as exc:
            row.update(result='ERROR', error=f'{type(exc).__name__}: {exc}')
            with diagnostic_path.open('a', encoding='utf-8') as diagnostic:
                traceback.print_exc(file=diagnostic)
            if server.controller is not None:
                try:
                    server._save_session_files(status='FAILED')
                except Exception:
                    pass
            server.controller = None
        row['runtime_sec'] = round(time.perf_counter() - started, 3)
        results.append(row)
        write_csv(output / 'results.csv', results)
        report(output, results)
        print(f"[{index}/{len(entries)}] {row['result']} L={row.get('left_final_reps', '?')} "
              f"R={row.get('right_final_reps', '?')} occluded={row.get('occluded_percent', '?')}% "
              f"runtime={row['runtime_sec']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video-root', type=Path, default=ROOT / 'test_video')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inventory-only', action='store_true')
    parser.add_argument('--exercise', choices=list(EXERCISES), action='append')
    parser.add_argument('--limit-per-exercise', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    if args.batch_size <= 0 or (args.limit_per_exercise is not None and args.limit_per_exercise <= 0):
        parser.error('Batch size and limit must be positive')
    args.video_root = args.video_root.resolve()
    if not args.video_root.is_dir():
        parser.error(f'Video folder not found: {args.video_root}')
    output = (args.output or ROOT / 'data/calibration_experiments' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.csv').exists():
        parser.error('Output already contains results; choose a new output directory')
    entries = inventory(args.video_root)
    entries = [entry for entry in entries if entry['exercise'] and
               (not args.exercise or entry['folder'] in args.exercise)]
    if args.limit_per_exercise:
        counts = Counter()
        selected = []
        for entry in entries:
            counts[entry['folder']] += 1
            if counts[entry['folder']] <= args.limit_per_exercise:
                selected.append(entry)
        entries = selected
    write_csv(output / 'inventory.csv', entries)
    for folder in sorted(set(entry['folder'] for entry in entries)):
        items = [entry for entry in entries if entry['folder'] == folder]
        print(f"{folder}: {len(items)} videos, {sum(item['frames'] for item in items)} frames, "
              f"{sum(item['duration_sec'] for item in items) / 60:.1f} minutes", flush=True)
    print(f'Output: {output}', flush=True)
    if args.inventory_only:
        return
    for directory in ('diagnostics', 'raw_keypoints', 'sessions'):
        (output / directory).mkdir(exist_ok=True)
    asyncio.run(run(args, entries, output))


if __name__ == '__main__':
    main()
