"""Shared labels and colors for confirmed repetition results."""

from core.motion_engine import QUALITY_CODES, QUALITY_POINTS

QUALITY_COLORS = {'PERFECT': '#16a34a', 'GOOD': '#f59e0b', 'BAD': '#a855f7'}
QUALITY_LABELS = {'PERFECT': 'Perfect', 'GOOD': 'Good', 'BAD': 'Normal'}


def quality_result(data, mode='', metric_type=''):
    if mode == 'CALIBRATION':
        return None
    grade = str(data.get('quality', '')).upper()
    if grade == 'NORMAL':
        grade = 'BAD'
    if grade not in QUALITY_COLORS:
        return None
    return dict(grade=grade, label=QUALITY_LABELS[grade], color=QUALITY_COLORS[grade],
                code=QUALITY_CODES[grade], points=QUALITY_POINTS[grade])
