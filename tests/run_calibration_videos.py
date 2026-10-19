"""Send test_video clips sequentially to the real socket server; create ASS overlays.

Run: python tests/run_calibration_videos.py
The local server is started automatically if port 8080 isn't already listening.
This client never loads AI models. Video processing happens on the socket server.
ASS files are saved beside each source video with the same basename, ready to load.
Re-running replaces those generated ASS files. Logs are kept separately under data/.
"""

import argparse
import asyncio
from collections import Counter
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import websockets

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.calibration_subtitles import write_calibration_ass

EXERCISES = {'biceps_curl': 'biceps_curl', 'shoulder_press': 'shoulder_press',
             'clamshell': 'clamshell', 'slr': 'slr',
             'leg_extension': 'knee_extension', 'leg_raise': 'hip_knee_flexion'}


def find_videos(root, selected=None, limit=None):
    groups = {}
    for folder, exercise in EXERCISES.items():
        if selected and folder not in selected:
            continue
        paths = sorted(path for path in (root / folder).rglob('*')
                       if path.is_file() and path.suffix.lower() in {'.mp4', '.mov', '.avi', '.mkv', '.m4v'})
        groups[folder] = [(path, exercise) for path in (paths[:limit] if limit else paths)]
    # Round-robin makes all exercises visible early, rather than finishing one folder first.
    return [items[index] for index in range(max(map(len, groups.values()), default=0))
            for items in groups.values() if index < len(items)]


async def wait_for_result(ws, timeout):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Video analysis timeout')
        packet = json.loads(await asyncio.wait_for(ws.recv(), remaining))
        kind = packet.get('type')
        if kind == 'VIDEO_PROGRESS':
            print(f"  frames {packet['processed_frames']}/{packet['total_frames']}", flush=True)
        elif kind in {'VIDEO_STATUS', 'SESSION_PENDING'}:
            print('  ' + packet.get('message', kind), flush=True)
        elif kind == 'ERROR':
            raise RuntimeError(packet.get('message', 'Server error'))
        elif kind == 'VIDEO_ANALYSIS_COMPLETED':
            return packet


async def open_connection(args):
    return await websockets.connect(args.uri, open_timeout=2, ping_interval=None, max_size=16 * 1024 * 1024)


async def run(args, output, videos):
    process, log_file, ws = None, None, None
    try:
        try:
            ws = await open_connection(args)
        except OSError:
            if args.no_start_server or args.uri != 'ws://127.0.0.1:8080':
                raise RuntimeError('Start the socket server first or use the default local URI')
            print('Starting local socket server (models load once)...', flush=True)
            log_file = (output / 'server.log').open('w', encoding='utf-8')
            process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'network/socket_server.py')],
                cwd=str(ROOT), stdout=log_file, stderr=subprocess.STDOUT,
                env={**os.environ, 'PYTHONIOENCODING': 'utf-8'},
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic() + args.startup_timeout
            while ws is None:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError(f'Server startup failed; see {output / "server.log"}')
                try:
                    ws = await open_connection(args)
                except OSError:
                    await asyncio.sleep(0.25)
        print(f'Connected: {args.uri}', flush=True)
        # Wait for the existing server's model-ready notification before submitting videos.
        deadline = time.monotonic() + args.startup_timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('Model preparation timeout')
            packet = json.loads(await asyncio.wait_for(ws.recv(), remaining))
            if packet.get('type') == 'SERVER_STATUS':
                print('  model: ' + packet.get('phase', packet.get('model_state', '')), flush=True)
                if packet.get('model_state') == 'FAILED':
                    raise RuntimeError(packet.get('model_error', 'Model loading failed'))
                if packet.get('ready'):
                    break
        results = []
        fields = ['video', 'exercise', 'result', 'left_reps', 'right_reps', 'final_step',
                  'count_events', 'subtitle_path', 'frame_csv', 'runtime_sec', 'error']
        with (output / 'results.csv').open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            file.flush()
            for index, (path, exercise) in enumerate(videos, 1):
                print(f'\n[{index}/{len(videos)}] {path.relative_to(args.video_root)}', flush=True)
                row = dict(video=str(path), exercise=exercise)
                started = time.monotonic()
                try:
                    await ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'mode': 'CALIBRATION',
                        'video_path': str(path), 'exercise_name': exercise, 'patient_id': f'video_{index:04}',
                        'log_root': str(output / 'sessions'), 'session': 1, 'round': 0, 'set': 0,
                        'batch_size': args.batch_size, 'record_video': False, 'output_mode': 'LIGHTWEIGHT'}))
                    packet = await wait_for_result(ws, args.timeout)
                    summary = packet['summary']
                    frame_csv = Path(summary['frame_csv'])
                    # Read actual logged state as a fallback for an older server.
                    with frame_csv.open(encoding='utf-8-sig', newline='') as frames:
                        logged = list(csv.DictReader(frames))
                    last = logged[-1] if logged else {}
                    finished = last.get('calib_step') == 'FINISHED'
                    row.update(result='FINISHED' if finished else 'INCOMPLETE',
                        left_reps=last.get('left_rep_count', 0), right_reps=last.get('right_rep_count', 0),
                        final_step=last.get('calib_step', 'NO_FRAMES'), frame_csv=str(frame_csv))
                    ass = (args.subtitle_root / path.relative_to(args.video_root)).with_suffix('.ass')
                    duration = packet.get('source_duration_sec')
                    row.update(write_calibration_ass(frame_csv, ass, duration_sec=duration))
                    print(f"  {row['result']} | L={row['left_reps']} R={row['right_reps']} | {row['final_step']}", flush=True)
                    print(f'  ASS: {ass}', flush=True)
                except (TimeoutError, websockets.exceptions.ConnectionClosed):
                    # A timed-out job may still be processing: don't submit the next video on it.
                    raise
                except Exception as exc:
                    row.update(result='ERROR', error=str(exc))
                    print(f'  ERROR: {exc}', flush=True)
                row['runtime_sec'] = round(time.monotonic() - started, 2)
                results.append(row)
                writer.writerow(row)
                file.flush()
        counts = Counter(row['result'] for row in results)
        print(f"\nDONE: {len(results)} videos | FINISHED={counts['FINISHED']} "
              f"INCOMPLETE={counts['INCOMPLETE']} ERROR={counts['ERROR']}", flush=True)
        print(f'Results: {output / "results.csv"}', flush=True)
        print('ASS files are beside the source videos by default; open a video in an ASS-capable player.', flush=True)
        return 0 if counts['INCOMPLETE'] + counts['ERROR'] == 0 else 1
    finally:
        if ws is not None:
            await ws.close()
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if log_file is not None:
            log_file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uri', default='ws://127.0.0.1:8080')
    parser.add_argument('--video-root', type=Path, default=ROOT / 'test_video')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--subtitle-root', type=Path,
                        help='ASS destination root (default: beside source videos in --video-root)')
    parser.add_argument('--exercise', choices=list(EXERCISES), action='append')
    parser.add_argument('--limit-per-exercise', type=int)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--startup-timeout', type=float, default=180)
    parser.add_argument('--no-start-server', action='store_true')
    args = parser.parse_args()
    if (args.batch_size <= 0 or args.timeout <= 0 or args.startup_timeout <= 0
            or (args.limit_per_exercise is not None and args.limit_per_exercise <= 0)):
        parser.error('Batch size, timeouts and limits must be positive')
    args.video_root = args.video_root.resolve()
    args.subtitle_root = (args.subtitle_root or args.video_root).resolve()
    videos = find_videos(args.video_root, args.exercise, args.limit_per_exercise)
    if not videos:
        parser.error(f'No supported videos found in {args.video_root}')
    output = (args.output or ROOT / 'data/calibration_socket_tests' /
              datetime.now().strftime('%Y%m%d_%H%M%S_%f')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.csv').exists():
        parser.error('Choose a new output directory; existing results will not be overwritten')
    print(f'Videos: {len(videos)} | Output: {output}', flush=True)
    print(f'ASS destination: {args.subtitle_root} (existing same-name ASS files are replaced)', flush=True)
    try:
        return asyncio.run(run(args, output, videos))
    except KeyboardInterrupt:
        print('\nStopped by user.', flush=True)
        return 130
    except Exception as exc:
        print(f'FAILED: {exc}', file=sys.stderr, flush=True)
        return 2


if __name__ == '__main__':
    sys.exit(main())
