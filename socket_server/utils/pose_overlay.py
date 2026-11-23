"""진행률 계산, 품질 표시 및 23개 관절(몸통+발) 실시간 미리보기 오버레이 렌더링 모듈."""

import math
import cv2
import numpy as np

# 캘리브레이션에서 움직임 시작 / 복귀로 보는 진행률 기준
CALIBRATION_ACTIVATION_FRACTION = 0.30
CALIBRATION_REARM_FRACTION = 0.15

# 품질 등급별 점수·코드·임계치 및 화면 표시 색상
QUALITY_POINTS = {'PERFECT': 1.0, 'GOOD': 0.5, 'BAD': 0.25}
QUALITY_THRESHOLDS = {'PERFECT': 0.70, 'GOOD': 0.50, 'BAD': 0.35}
QUALITY_CODES = {'PERFECT': 3, 'GOOD': 2, 'BAD': 1}
QUALITY_COLORS = {'PERFECT': '#16a34a', 'GOOD': '#f59e0b', 'BAD': '#a855f7'}
QUALITY_LABELS = {'PERFECT': 'Perfect', 'GOOD': 'Good', 'BAD': 'Normal'}

# COCO-23 관절 연결선 (몸통 5~16 + 발 17~22)
COCO23_BODY_EDGES = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
)
COCO23_FOOT_EDGES = (
    (15, 17), (15, 18), (15, 19),
    (16, 20), (16, 21), (16, 22),
)


def measurement_progress(value, metric_type, start=None, target=None):
    """현재 측정값(value)을 start~target 구간 기준 0~1 진행률로 변환한다."""
    if any(v is None or not math.isfinite(v) for v in (value, start, target)) or start == target:
        return None
    return max(0.0, min(1.0, (value - start) / (target - start)))


def quality_result(data, mode='', metric_type=''):
    """측정 결과에서 표시용 등급·라벨·색상·코드·점수를 반환한다."""
    if mode == 'CALIBRATION':
        return None
    grade = str(data.get('quality', '')).upper()
    if grade == 'NORMAL':
        grade = 'BAD'
    if grade not in QUALITY_COLORS:
        return None
    return dict(
        grade=grade,
        label=QUALITY_LABELS[grade],
        color=QUALITY_COLORS[grade],
        code=QUALITY_CODES[grade],
        points=QUALITY_POINTS[grade],
    )


def motion_progress(pose, side):
    """한쪽(left/right) 측정값과 기준값을 이용해 진행 막대용 0~1 비율을 계산한다."""
    data = pose.get(side, {})
    if pose.get('is_occluded') or data.get('val') is None:
        return None
    metric = pose.get('metric_type', '')
    display = pose.get('display_thresholds', {}).get(side)
    if display is not None:
        return measurement_progress(data['val'], 'COORDINATE', display.get('start_val'), display.get('target_val'))
    start = pose.get('thresholds', {}).get(side, {}).get('start_val')
    target = pose.get('thresholds', {}).get(side, {}).get('target_val')
    if start is not None and target is not None:
        return measurement_progress(data['val'], metric, start, target)
    return measurement_progress(data.get('progress_ratio'), 'COORDINATE', 0.0, 1.0)


def render_overlay(frame, keypoints, pose, exercise_name, elapsed_sec):
    """영상 프레임 위에 COCO-23 스켈레톤(몸통+발)과 우측 정보 패널을 그린다."""
    image = frame.copy()
    height, width = image.shape[:2]
    unit = min(width / 640, height / 360)
    thickness = max(1, round(2 * unit))
    points = {}
    for p in keypoints or []:
        x, y = p.get('x'), p.get('y')
        if (
            p['id'] >= 5
            and p.get('score', 0) >= 0.35
            and x is not None
            and y is not None
            and math.isfinite(x)
            and math.isfinite(y)
            and 0 <= x <= 1
            and 0 <= y <= 1
        ):
            points[p['id']] = (min(width - 1, round(x * width)), min(height - 1, round(y * height)))

    for first, second in COCO23_BODY_EDGES:
        if first in points and second in points:
            cv2.line(image, points[first], points[second], (70, 230, 100), thickness, cv2.LINE_AA)
    for first, second in COCO23_FOOT_EDGES:
        if first in points and second in points:
            cv2.line(image, points[first], points[second], (255, 140, 0), thickness, cv2.LINE_AA)
    for pid, point in points.items():
        pt_color = (255, 80, 255) if pid >= 17 else (0, 210, 255)
        cv2.circle(image, point, max(2, round(3 * unit)), pt_color, -1, cv2.LINE_AA)

    # 카메라 원본 화각(100%)을 가리지 않도록 정보 패널을 영상 오른쪽 바깥에 이어 붙인다
    panel_width = max(round(170 * unit), round(width * 0.27))
    edge = width
    total_width = width + panel_width
    margin = max(4, round(12 * unit))
    canvas = np.full((height, total_width, 3), 24, dtype=np.uint8)
    canvas[:, :width] = image
    image = canvas

    def text(value, x, y, space, size=0.5, color=(255, 255, 255), bold=False):
        scale = max(0.15, size * unit)
        weight = max(1, round((2 if bold else 1) * unit))
        text_width = cv2.getTextSize(str(value), cv2.FONT_HERSHEY_SIMPLEX, scale, weight)[0][0]
        scale *= min(1.0, max(1, space) / max(1, text_width))
        cv2.putText(image, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)

    header = max(1, round(58 * unit))
    image[:header, :edge] = (image[:header, :edge] * 0.3).astype('uint8')
    text(f'{exercise_name} | {pose.get("mode", "")}', margin, round(22 * unit), edge - 2 * margin)
    text(f'{elapsed_sec:.1f}s  {pose.get("fps", 0):.1f} FPS', margin, round(45 * unit), edge - 2 * margin, 0.45)

    for index, side in enumerate(('left', 'right')):
        data = pose.get(side, {})
        top = index * (height // 2)
        left, right = edge + margin, total_width - margin
        bar_width = max(8, round(18 * unit))
        bar_left = right - bar_width
        space = max(1, bar_left - left - margin)
        color = (255, 190, 80) if side == 'left' else (100, 220, 255)
        text(f'{side.upper()} REPS', left, top + round(25 * unit), space, 0.6, color, True)
        count = data.get('rep_count')
        count_text = '--' if count is None else str(int(count))
        if pose.get('mode') == 'CALIBRATION' and count is not None:
            count_text += ' / 3'
        text(count_text, left, top + round(61 * unit), space, 1.4, color, True)
        result = quality_result(data, pose.get('mode', ''), pose.get('metric_type', ''))
        if result:
            hex_color = result['color'].lstrip('#')
            grade_color = tuple(int(hex_color[i:i + 2], 16) for i in (4, 2, 0))
            text(f"{result['label'].upper()} [{result['code']}] +{result['points']:g}", left, top + round(87 * unit), space, 0.55, grade_color, True)
        elif pose.get('mode') == 'CALIBRATION':
            text('CALIBRATION', left, top + round(87 * unit), space, 0.4, color)
        else:
            text('RESULT --', left, top + round(87 * unit), space, 0.4)

        total = data.get('total_score')
        total_text = f'{total:g}' if total is not None and math.isfinite(total) else '--'
        text('TARGET 3' if pose.get('mode') == 'CALIBRATION' else f'TOTAL {total_text}', left, top + round(110 * unit), space, 0.55, color, True)

        value = data.get('val')
        suffix = ' px' if pose.get('metric_unit', pose.get('measurement', {}).get('metric_unit')) == 'px' else (' deg' if pose.get('metric_type') == 'ANGLE' else '')
        digits = 4 if pose.get('metric_type') in ('COORDINATE', 'RAW_COORDINATE') else 2
        value_text = '--' if value is None or not math.isfinite(value) else f'{value:.{digits}f}{suffix}'
        text(f'Value {value_text}', left, top + round(131 * unit), space, 0.42)

        maximum = pose.get('display_thresholds', {}).get(side, {}).get('target_val')
        if maximum is not None and math.isfinite(maximum):
            goal_label = 'GOAL' if pose.get('measurement', {}).get('count_on') == 'TARGET_POSITION' else 'MAX'
            text(f'{goal_label} {maximum:.{digits}f}{suffix}', left, top + round(150 * unit), space, 0.36)

        progress = motion_progress(pose, side)
        bar_top = top + margin
        bar_bottom = top + height // 2 - margin
        cv2.rectangle(image, (bar_left, bar_top), (right, bar_bottom), (85, 85, 85), 1)
        if progress is not None:
            fill = round((bar_bottom - bar_top - 2) * progress)
            cv2.rectangle(image, (bar_left + 1, bar_bottom - 1 - fill), (right - 1, bar_bottom - 1), color, -1)
        marks = (
            [(0.30, (90, 210, 120))]
            if pose.get('mode') == 'CALIBRATION'
            else []
            if pose.get('measurement', {}).get('count_on') == 'TARGET_POSITION'
            else [(fraction, tuple(int(QUALITY_COLORS[grade].lstrip('#')[i:i + 2], 16) for i in (4, 2, 0))) for grade, fraction in QUALITY_THRESHOLDS.items()]
        )
        for fraction, mark_color in marks:
            y = bar_bottom - 1 - round((bar_bottom - bar_top - 2) * fraction)
            cv2.line(image, (bar_left, y), (right, y), mark_color, max(1, round(unit)))

    cv2.line(image, (edge, 0), (edge, height), (90, 90, 90), 1)
    cv2.line(image, (edge, height // 2), (total_width, height // 2), (90, 90, 90), 1)
    return image
