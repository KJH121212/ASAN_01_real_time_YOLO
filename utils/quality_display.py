"""확정된 운동 반복(rep)의 품질 등급별 표시 라벨과 색상을 정리한 모듈."""

from core.motion_engine import QUALITY_CODES, QUALITY_POINTS

# 품질 등급별 화면 표시 색상 및 라벨
QUALITY_COLORS = {'PERFECT': '#16a34a', 'GOOD': '#f59e0b', 'BAD': '#a855f7'}
QUALITY_LABELS = {'PERFECT': 'Perfect', 'GOOD': 'Good', 'BAD': 'Normal'}


def quality_result(data, mode='', metric_type=''):
    """측정 결과에서 표시용 등급·라벨·색상·코드·점수를 뽑아 반환한다."""
    if mode == 'CALIBRATION':
        return None
    grade = str(data.get('quality', '')).upper()
    if grade == 'NORMAL':
        grade = 'BAD'
    if grade not in QUALITY_COLORS:
        return None
    return dict(grade=grade, label=QUALITY_LABELS[grade], color=QUALITY_COLORS[grade],
                code=QUALITY_CODES[grade], points=QUALITY_POINTS[grade])
