"""CSV·NPZ 로그 읽기와 관절 좌표 변환. 원본 파일을 수정하지 않는다."""

from io import BytesIO
import json

import numpy as np
import pandas as pd
from core.motion_engine import QUALITY_POINTS, QUALITY_CODES

JOINTS = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
          'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
          'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
          'left_knee', 'right_knee', 'left_ankle', 'right_ankle']


def count_events(table, time_column='time_sec'):
    """확정 플래그가 우선이다. 없는 과거 로그만 누적 횟수 증가 프레임을 사용한다."""
    rows = []
    for side in ('left', 'right'):
        flag, count = f'{side}_count_updated', f'{side}_rep_count'
        if flag in table:
            mask = table[flag].astype(str).str.strip().str.lower().isin(['true', '1', '1.0'])
            source = 'count_updated'
        elif count in table:
            mask = pd.to_numeric(table[count], errors='coerce').diff().gt(0)
            source = 'count_increase'
        else:
            continue
        for _, row in table.loc[mask].iterrows():
            rows.append({time_column: row[time_column], 'side': side,
                         'rep_count': row.get(count, np.nan), 'frame_idx': row.get('frame_idx', np.nan),
                         'value': row.get(f'{side}_value', np.nan), 'source': source,
                         'quality': row.get(f'{side}_quality', ''),
                         'score': row.get(f'{side}_score', QUALITY_CODES.get(str(row.get(f'{side}_quality', '')).upper(), np.nan)),
                         'score_delta': row.get(f'{side}_score_delta', np.nan),
                         'total_score': row.get(f'{side}_total_score', np.nan)})
    return pd.DataFrame(rows, columns=[time_column, 'side', 'rep_count', 'frame_idx', 'value', 'source',
                                      'quality', 'score', 'score_delta', 'total_score']).sort_values(time_column)


def read_csv(content):
    table = pd.read_csv(BytesIO(content), encoding='utf-8-sig')
    if table.empty:
        raise ValueError('CSV에 데이터가 없습니다.')
    for column, scale in (('time_sec', 1), ('time', 1), ('timestamp_ms', .001)):
        if column in table:
            times = pd.to_numeric(table[column], errors='coerce') * scale
            if not np.isfinite(times).all():
                raise ValueError(f'{column} 열에 유효하지 않은 시간이 있습니다.')
            table = table.copy()
            table['time_sec'] = times
            return table, 'time_sec'
    # 시간 열이 없는 일반 CSV도 행 번호 기준으로 확인할 수 있다.
    table['row_index'] = np.arange(len(table))
    return table, 'row_index'


def read_npz(content):
    with np.load(BytesIO(content), allow_pickle=False) as data:
        log = {key: data[key].copy() for key in data.files}
    times = np.asarray(log.get('timestamps', []), dtype=float)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or (np.diff(times) < 0).any():
        raise ValueError('NPZ의 timestamps가 비어 있거나 올바르지 않습니다.')
    log['timestamps'] = times
    for key in ('keypoints', 'keypoints_px'):
        if key in log and log[key].shape != (len(times), 17, 3):
            raise ValueError(f'{key}의 크기는 프레임 수 × 17 × 3이어야 합니다.')
    for key in ('values', 'total_scores', 'score_deltas', 'rep_counts', 'progress_ratios', 'states'):
        if key in log and log[key].shape != (len(times), 2):
            raise ValueError(f'{key}의 크기는 프레임 수 × 2이어야 합니다.')
    return log


def summary(log):
    value = log.get('summary_json')
    info = json.loads(str(value.item())) if value is not None else {}
    if not isinstance(info, dict):
        raise ValueError('summary_json은 JSON 객체여야 합니다.')
    return info


def measurements(log):
    table = pd.DataFrame({'time_sec': log['timestamps']})
    for key, suffix in (('values', 'value'), ('total_scores', 'total_score'), ('score_deltas', 'score_delta'),
                        ('rep_counts', 'rep_count'), ('progress_ratios', 'progress_ratio'), ('states', 'state')):
        if key in log:
            for i, side in enumerate(('left', 'right')):
                table[f'{side}_{suffix}'] = log[key][:, i]
    raw_history = log.get('reps_json')
    history = json.loads(str(raw_history.item())) if raw_history is not None else []
    if not isinstance(history, list):
        raise ValueError('reps_json must contain a list')
    for side in ('left', 'right'):
        grades = pd.Series(index=table.index, dtype=object)
        count = f'{side}_rep_count'
        if count in table:
            by_rep = {row.get('rep_num'): row.get('quality') for row in history
                      if isinstance(row, dict) and row.get('side') == side}
            grades = table[count].map(by_rep)
        delta = f'{side}_score_delta'
        if delta in table:
            mapped = table[delta].map({points: grade for grade, points in QUALITY_POINTS.items()})
            inferred = mapped.astype('string').ffill().astype(object)
            grades = grades.astype('string').fillna(inferred.astype('string')).astype(object)
        table[f'{side}_quality'] = grades
    return table


def coordinates(log, mode='영상 비율', confidence=.35):
    """영상 비율은 x/너비,y/높이. 신체 정규화는 서버 PoseNormalizer를 사용한다."""
    if 'keypoints' in log:
        points = log['keypoints'].astype(float).copy()
    elif 'keypoints_px' in log and 'frame_sizes' in log:
        points = log['keypoints_px'].astype(float).copy()
        points[:, :, :2] /= log['frame_sizes'][:, None, :]
    else:
        raise ValueError('관절 좌표 또는 영상 크기 정보가 없습니다.')
    if mode == '픽셀':
        if 'keypoints_px' in log:
            points = log['keypoints_px'].astype(float).copy()
        elif 'frame_sizes' in log:
            points[:, :, :2] *= log['frame_sizes'][:, None, :]
        elif 'video_width' in log and 'video_height' in log:
            points[:, :, :2] *= [float(log['video_width']), float(log['video_height'])]
        else:
            raise ValueError('픽셀 변환에 필요한 영상 크기가 없습니다.')
    elif mode == '신체 정규화':
        from utils.normalization import PoseNormalizer
        normalizer = PoseNormalizer(conf_threshold=confidence, invert_y=True)
        result = np.full_like(points, np.nan)
        result[:, :, 2] = points[:, :, 2]
        aspects = np.broadcast_to(log.get('image_aspect_ratio', 1.), (len(points),))
        for i, frame in enumerate(points):
            if not np.isfinite(aspects[i]) or aspects[i] <= 0:
                continue
            if not np.isfinite(frame[[11, 12]]).all() or (frame[[11, 12], 2] < confidence).any():
                continue
            keypoints = [dict(id=j, x=float(x), y=float(y), score=float(score) if np.isfinite([x, y, score]).all() else 0.,
                              image_aspect_ratio=float(aspects[i])) for j, (x, y, score) in enumerate(frame)]
            normalized = normalizer.normalize(keypoints)
            if normalized:
                result[i, :, :2] = [[p['x'], p['y']] for p in normalized]
        points = result
    points[(~np.isfinite(points[:, :, 2])) | (points[:, :, 2] < confidence), :2] = np.nan
    return points
