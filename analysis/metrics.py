"""로그의 측정 방식, 핵심 관절, 반복별 변화와 진단 수치를 정리한다."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.log_reader import summary
from core.motion_engine import QUALITY_THRESHOLDS

CONFIG_PATH = Path(__file__).resolve().parent.parent / 'configs/exercise_kpt_config_coco.json'
LABELS = {
    'ANGLE': ('관절 각도', '각도 (°)', '설정된 세 관절의 각도 변화를 봅니다.'),
    'COORDINATE': ('관절 위치', '정규화 좌표 (무차원)', '골반 중심을 원점으로 하고 몸통 길이를 1로 맞춘 좌표 또는 두 관절의 좌표 차이입니다. 실제 길이 단위가 아닙니다.'),
    'LENGTH': ('관절 사이 거리', '정규화 거리 (무차원)', '몸통 길이를 1로 맞춘 좌표에서 계산한 두 관절 사이 거리입니다. cm나 m 단위가 아닙니다.'),
}


def context(table, log=None):
    configs = json.loads(CONFIG_PATH.read_text(encoding='utf-8'))['exercises']
    info = summary(log) if log is not None else {}
    stored = info.get('measurement', {})
    name = info.get('motion_name', stored.get('exercise_name', ''))
    if not name and 'motion_name' in table:
        names = table['motion_name'].dropna()
        name = str(names.iloc[0]) if len(names) else ''
    if not name and log is not None:
        name = str(log.get('exercise_name', ''))
        if not name:
            source = str(log.get('source_video', ''))
            name = next((key for key in configs if key in source), '')
    if not name and 'video' in table:
        videos = table['video'].dropna()
        source = str(videos.iloc[0]) if len(videos) else ''
        name = next((key for key in configs if key in source), '')
    config = configs.get(name, {})
    evaluation = dict(config.get('eval_config', {}))
    metric = stored.get('metric_type', evaluation.get('metric_type', 'UNKNOWN'))
    if 'metric_type' in table and table['metric_type'].notna().any():
        metric = str(table['metric_type'].dropna().iloc[0])
    recorded = stored.get('measurement', {})
    if recorded:
        evaluation = dict(recorded)
    for key in ('axis', 'target_kpt', 'primary_kpts', 'reference_kpt', 'coordinate_unit'):
        if key in recorded:
            evaluation[key] = recorded[key]
    if 'movement_direction' in recorded:
        evaluation['motion_direction'] = recorded['movement_direction']
    by_side = {}
    for side in ('left', 'right'):
        fields = (('primary_kpts',) if metric in ('ANGLE', 'LENGTH') else
                  ('target_kpt', 'reference_kpt'))
        ids = []
        for field in fields:
            value = evaluation.get(field, {}).get(side, [])
            ids.extend(value if isinstance(value, list) else [value])
        by_side[side] = sorted({j for j in ids if type(j) is int and 0 <= j < 17})
    joints = sorted(set(by_side['left'] + by_side['right'])) or [11, 12, 13, 14, 15, 16]
    label, unit, description = LABELS.get(metric, ('저장된 측정값', '로그 값', '저장된 수치 그대로 표시합니다.'))
    normalized = recorded.get('use_normalization', evaluation.get('use_normalization', True))
    if 'use_normalization' in table and table.use_normalization.notna().any():
        normalized = str(table.use_normalization.dropna().iloc[0]).lower() == 'true'
    counting_method = recorded.get('count_on', 'RETURN')
    if 'counting_method' in table and table.counting_method.notna().any():
        counting_method = str(table.counting_method.dropna().iloc[0])
    if not normalized and metric in ('COORDINATE', 'LENGTH'):
        unit = '영상 높이 대비 좌표' if metric == 'COORDINATE' else '영상 높이 대비 거리'
        description = '몸통 정규화를 적용하지 않은 영상 좌표입니다. 위가 Y=0이며 아래쪽으로 증가합니다. 가로·세로 축은 영상 높이 기준으로 맞춥니다.'
    coordinate_unit = recorded.get('coordinate_unit', 'image' if recorded else evaluation.get('coordinate_unit', 'image'))
    if 'metric_unit' in table and table.metric_unit.notna().any():
        coordinate_unit = 'pixel' if str(table.metric_unit.dropna().iloc[0]) == 'px' else 'image'
    if coordinate_unit == 'pixel' and metric == 'COORDINATE':
        unit = '좌표 차이 (px)' if evaluation.get('reference_kpt') else '관절 위치 (px)'
        label = '관절 좌표 차이' if evaluation.get('reference_kpt') else '관절 위치'
        description = '몸통 정규화 없이 원본 영상 픽셀 좌표로 계산합니다. 기준 관절이 있으면 대상 좌표에서 기준 좌표를 뺍니다. 촬영 거리가 달라지면 재캘리브레이션이 필요합니다.'
    evaluation['coordinate_unit'] = coordinate_unit
    evaluation['use_normalization'] = normalized
    versions = pd.to_numeric(table.get('threshold_version', pd.Series(0, index=table.index)), errors='coerce').ffill().fillna(0)
    history = {}
    for version in versions.unique():
        bounds = {}
        for side in ('left', 'right'):
            bounds[side] = {}
            for key in ('start_val', 'target_val'):
                saved = stored.get('thresholds', {}).get(side, {}).get(key) if version == 0 else None
                values = pd.to_numeric(table.loc[versions.eq(version), f'{side}_{key}'], errors='coerce').dropna() if f'{side}_{key}' in table else pd.Series(dtype=float)
                bounds[side][key] = saved if saved is not None else float(values.iloc[0]) if len(values) else np.nan
        history[str(float(version))] = bounds
    return dict(exercise=name, name=config.get('display_name_kr', name or '동작 정보 없음'),
                metric=metric, label=label, unit=unit, description=description,
                evaluation=evaluation, joints=joints, by_side=by_side,
                factor=1,
                settings_source='로그에 저장된 설정' if recorded else '현재 운동 설정 참고',
                hold_seconds=recorded.get('up_hold_seconds'), counting_method=counting_method,
                threshold_history=history,
                threshold_source='로그에 저장된 초기 설정' if stored.get('thresholds') else '버전별 최초 기록값 (설정 원본 없음)')


def threshold_guides(table, ctx, time_column, sides):
    """Configured thresholds are constant within a version; peaks are separate diagnostics."""
    versions = pd.to_numeric(table.get('threshold_version', pd.Series(0, index=table.index)), errors='coerce').ffill().fillna(0)
    guides = pd.DataFrame(index=table.index)
    for side in sides:
        for key in ('start_val', 'target_val'):
            column = f'{side}_{key}'
            guides[column] = versions.map(lambda v: ctx['threshold_history'].get(str(float(v)), {}).get(side, {}).get(key, np.nan))
    guides.index = table[time_column]
    return guides.dropna(axis=1, how='all') * ctx['factor']


def quality_guides(table, ctx, time_column, sides):
    """Grade boundaries use the same fixed ROM and fractions as the server."""
    bounds = threshold_guides(table, ctx, time_column, sides)
    result = pd.DataFrame(index=bounds.index)
    if ctx.get('counting_method') == 'TARGET_POSITION':
        return result
    for side in sides:
        start, target = f'{side}_start_val', f'{side}_target_val'
        if start in bounds and target in bounds:
            for grade, fraction in QUALITY_THRESHOLDS.items():
                result[f'{side}_quality_{grade}'] = bounds[start] + (bounds[target]-bounds[start])*fraction
    return result


def longest_target_hold(table, side, time_column):
    value, target = f'{side}_value', f'{side}_target_val'
    if value not in table or target not in table or table.empty:
        return np.nan
    values = pd.to_numeric(table[value], errors='coerce')
    targets = pd.to_numeric(table[target], errors='coerce')
    mask = values.ge(targets) & values.notna() & targets.notna()
    times = table[time_column]
    runs = (mask.ne(mask.shift()) | times.diff().gt(.2) | targets.ne(targets.shift())).cumsum()
    spans = times.loc[mask].groupby(runs.loc[mask]).agg(['min', 'max'])
    return float((spans['max'] - spans['min']).max()) if len(spans) else 0.


def diagnostics(table, ctx, time_column='time_sec'):
    rows = []
    for side, label in (('left', '왼쪽'), ('right', '오른쪽')):
        value = f'{side}_value'
        if value not in table:
            continue
        values = pd.to_numeric(table[value], errors='coerce')
        row = dict(side=label, valid_pct=100 * values.notna().mean(), minimum=values.min(),
                   maximum=values.max(), span=values.max()-values.min(), std=values.std(),
                   p05=values.quantile(.05), p95=values.quantile(.95))
        if f'{side}_state' in table:
            states = table[f'{side}_state'].fillna('UNKNOWN')
            row['main_state'] = states.mode().iloc[0] if len(states) else ''
            row['baseline_wait_pct'] = 100 * states.eq('BASELINE').mean()
            scale = pd.to_numeric(table.get(f'{side}_scale_base', pd.Series(dtype=float)), errors='coerce')
            row['baseline_ready_pct'] = 100 * scale.gt(0).mean() if len(scale) else np.nan
            row['longest_target_hold_sec'] = (longest_target_hold(table, side, time_column)
                                            if time_column == 'time_sec' and ctx['counting_method'] != 'PEAK_REVERSAL' else np.nan)
            if f'{side}_state' in table and f'{side}_target_val' in table:
                active = values.ge(pd.to_numeric(table[f'{side}_target_val'], errors='coerce'))
                row['return_wait_above_target_pct'] = 100 * table.loc[active, f'{side}_state'].eq('RETURNING').mean()
        rows.append(row)
    return pd.DataFrame(rows)


def repetition_stats(table, events, time_column='time_sec'):
    """이전 카운트부터 이번까지의 변화. 실제 동작 시작 시각을 추정하지 않는다."""
    rows = []
    for side in ('left', 'right'):
        previous = None
        for _, event in events.loc[events.side == side].iterrows():
            now = event[time_column]
            start = previous if previous is not None else table[time_column].min()
            segment = table.loc[table[time_column].between(start, now)]
            values = pd.to_numeric(segment.get(f'{side}_value', pd.Series(dtype=float)), errors='coerce')
            rows.append(dict(side=side, rep_count=event.rep_count, **{time_column: now,
                             'count_interval_sec' if time_column == 'time_sec' else 'count_interval_rows': now-previous if previous is not None else np.nan},
                             minimum=values.min(), maximum=values.max(), span=values.max()-values.min()))
            previous = now
    return pd.DataFrame(rows)


def state_segments(table, time_column):
    rows = []
    times = table[time_column]
    delta = times.diff().loc[lambda s: s > 0].median()
    delta = float(delta) if np.isfinite(delta) else .001
    for side in ('left', 'right'):
        name = f'{side}_state'
        if name not in table:
            continue
        values = table[name].fillna('UNKNOWN')
        starts = np.flatnonzero((values.ne(values.shift()) | times.diff().gt(max(.2, delta*1.5))).to_numpy())
        for i, start in enumerate(starts):
            end = starts[i+1] if i+1 < len(starts) else len(table)
            rows.append(dict(side=side, state=str(values.iloc[start]), start=float(times.iloc[start]),
                             end=min(float(times.iloc[end]), float(times.iloc[end-1])+delta) if end < len(table) else float(times.iloc[-1])+delta))
    return pd.DataFrame(rows)
