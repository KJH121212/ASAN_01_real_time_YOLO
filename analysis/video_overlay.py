"""이미 저장된 NPZ·CSV 측정 기록을 재추론 없이 원본 영상 위에 입혀 MP4로 내보내는 모듈."""

from pathlib import Path
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd

from analysis.log_reader import measurements, summary
from utils.pose_overlay import render_overlay


def find_ffmpeg():
    """PATH, 로컬 tools 폴더, imageio_ffmpeg 패키지 순으로 ffmpeg 실행 파일을 찾는다."""
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    binaries = Path(__file__).resolve().parent / 'tools/video/imageio_ffmpeg/binaries'
    for path in sorted(binaries.glob('ffmpeg*')):
        if path.is_file() and (path.suffix.lower() == '.exe' if os.name == 'nt' else os.access(path, os.X_OK)):
            return str(path)
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError):
        return None


def export_video(video_path, log, output_path, table=None, offset=0., progress=None):
    """원본 영상 프레임에 NPZ 관절 좌표·횟수·품질 오버레이를 그려 새 H.264 MP4로 저장한다."""
    video_path, output_path = Path(video_path).resolve(), Path(output_path).resolve()
    if output_path == video_path or output_path.exists():
        raise ValueError('원본 또는 기존 파일을 덮어쓸 수 없습니다. 새 출력 경로를 선택하세요.')
    times = np.asarray(log['timestamps'], dtype=float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('NPZ 시간이 비어 있거나 순서가 잘못되었습니다.')
    info = summary(log)
    values = measurements(log)
    if table is not None:
        if 'time_sec' not in table or len(table) != len(times) or not np.allclose(table.time_sec, times, atol=.001):
            raise ValueError('CSV와 NPZ의 시간이 일치하지 않습니다. 같은 세션의 파일을 선택하세요.')
        for column in table:
            if column != 'time_sec':
                values[column] = table[column].to_numpy()
    stored = info.get('measurement', {})
    method = stored.get('measurement', {}).get('count_on', '')
    # 예전 NPZ에는 점수 변화량만 있어 점수가 없는 캘리브레이션 반복 위치는 표시할 수 없다
    for side in ('left', 'right'):
        if f'{side}_rep_count' not in values and f'{side}_score_delta' in values:
            events = pd.to_numeric(values[f'{side}_score_delta'], errors='coerce').fillna(0).gt(0)
            if events.any() or info.get(f'{side}_reps') == 0:
                values[f'{side}_rep_count'] = events.cumsum()
    points = log.get('keypoints')
    if points is None and 'keypoints_px' in log and 'frame_sizes' in log:
        points = log['keypoints_px'].astype(float).copy()
        points[:, :, :2] /= log['frame_sizes'][:, None, :]
    if points is None or points.shape != (len(times), 17, 3):
        raise ValueError('NPZ에 프레임별 COCO 관절 좌표가 필요합니다.')
    capture = cv2.VideoCapture(str(video_path))
    writer = None
    intermediate = None
    try:
        if not capture.isOpened():
            raise ValueError('MP4 파일을 열 수 없습니다.')
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError('영상 FPS를 확인할 수 없습니다.')
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.asarray(log.get('video_frame_idx', []))
        if len(indices) == len(times) and np.all(indices >= 0) and np.all(np.diff(indices) >= 0):
            video_times = indices / fps + offset
        else:
            video_times = times - float(info.get('video_time_origin_sec') or 0.) + offset
        if video_times[-1] < 0 or total > 0 and video_times[0] >= total/fps:
            raise ValueError('영상과 NPZ의 시간 구간이 겹치지 않습니다. 시간 보정을 확인하세요.')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix='.overlay_render_', suffix='.mp4', dir=output_path.parent)
        os.close(handle)
        intermediate = Path(name)
        frame_index = 0
        rows = values.to_dict('records')
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[1] > 1920:
                scale_factor = 1920.0 / frame.shape[1]
                new_w = 1920
                new_h = int(round(frame.shape[0] * scale_factor))
                new_w = (new_w // 2) * 2
                new_h = (new_h // 2) * 2
                frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

            if writer is None:
                writer = cv2.VideoWriter(str(intermediate), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise ValueError('영상 저장 코덱을 열 수 없습니다.')
            t = frame_index/fps
            index = int(np.searchsorted(video_times, t, side='right')-1)
            valid = index >= 0 and t-video_times[index] <= 1.
            row = rows[max(index, 0)] if index >= 0 else {}
            pose = dict(mode=info.get('mode', ''), fps=fps, metric_type=row.get('metric_type', stored.get('metric_type', '')),
                        measurement=stored.get('measurement', {}),
                        metric_unit=row.get('metric_unit', stored.get('metric_unit', stored.get('measurement', {}).get('metric_unit', ''))),
                        is_occluded=not valid or str(row.get('is_occluded', '')).lower() == 'true', thresholds={},
                        display_thresholds=stored.get('display_thresholds', stored.get('measurement', {}).get('display_thresholds', {})))
            for side in ('left', 'right'):
                data = {k[len(side)+1:]: v for k, v in row.items() if k.startswith(side+'_') and pd.notna(v)}
                data['val'] = data.get('value') if valid else None
                data['counting_method'] = row.get('counting_method', method)
                pose[side] = data
                pose['thresholds'][side] = {k: data.get(k) for k in ('start_val', 'target_val')}
                if not any(v is not None for v in pose['thresholds'][side].values()):
                    pose['thresholds'][side] = stored.get('thresholds', {}).get(side, {})
            if info.get('mode') == 'MAIN':
                pose['display_thresholds'] = pose['thresholds']
            keypoints = [dict(id=i, x=float(x), y=float(y), score=float(score))
                         for i, (x, y, score) in enumerate(points[index])] if valid else []
            image = render_overlay(frame, keypoints, pose, info.get('motion_name', str(log.get('exercise_name', ''))),
                                   float(times[index]) if index >= 0 else t)
            writer.write(image)
            frame_index += 1
            if progress and (frame_index % max(1, round(fps)) == 0):
                progress(min(.95, frame_index/max(total, frame_index)*.95))
        if writer is None or frame_index == 0:
            raise ValueError('영상에서 읽을 수 있는 프레임이 없습니다.')
        writer.release()
        writer = None
        ffmpeg = find_ffmpeg()
        if ffmpeg:
            result = subprocess.run([ffmpeg, '-v', 'error', '-i', str(intermediate), '-c:v', 'libx264',
                                     '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p',
                                     '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart',
                                     '-an', str(output_path)], capture_output=True, text=True)
            if result.returncode:
                output_path.unlink(missing_ok=True)
                raise ValueError(f'H.264 변환 실패: {result.stderr[-500:]}')
        else:
            intermediate.replace(output_path)
        if progress:
            progress(1.)
        return dict(path=str(output_path), frames=frame_index, fps=fps, browser_compatible=bool(ffmpeg),
                    counts_available=all(f'{side}_rep_count' in values for side in ('left', 'right')))
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if intermediate is not None:
            intermediate.unlink(missing_ok=True)
