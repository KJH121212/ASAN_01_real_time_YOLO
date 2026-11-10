"""Shared overlay for live previews and recorded log videos."""

import math
import cv2
from utils.measurement_progress import measurement_progress
from utils.quality_display import QUALITY_COLORS, quality_result
from core.motion_engine import QUALITY_THRESHOLDS

COCO_EDGES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12),
              (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def motion_progress(pose, side):
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
    # Older packets without metric metadata may only carry a display ratio.
    return measurement_progress(data.get('progress_ratio'), 'COORDINATE', 0., 1.)


def render_overlay(frame, keypoints, pose, exercise_name, elapsed_sec):
    image = frame.copy()
    height, width = image.shape[:2]
    unit = min(width / 640, height / 360)
    thickness = max(1, round(2 * unit))
    points = {}
    for p in keypoints or []:
        x, y = p.get('x'), p.get('y')
        if (p['id'] >= 5 and p.get('score', 0) >= .35 and x is not None and y is not None
                and math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            points[p['id']] = (min(width-1, round(x*width)), min(height-1, round(y*height)))
    for first, second in COCO_EDGES:
        if first in points and second in points:
            cv2.line(image, points[first], points[second], (70, 230, 100), thickness, cv2.LINE_AA)
    for point in points.values():
        cv2.circle(image, point, max(2, round(3*unit)), (0, 210, 255), -1, cv2.LINE_AA)

    panel_width = min(width-1, max(round(170*unit), round(width*.27)))
    edge = width-panel_width
    margin = max(4, round(12*unit))
    image[:, edge:] = (image[:, edge:]*.15).astype('uint8')

    def text(value, x, y, space, size=.5, color=(255, 255, 255), bold=False):
        scale = max(.15, size*unit)
        weight = max(1, round((2 if bold else 1)*unit))
        text_width = cv2.getTextSize(str(value), cv2.FONT_HERSHEY_SIMPLEX, scale, weight)[0][0]
        scale *= min(1., max(1, space)/max(1, text_width))
        cv2.putText(image, str(value), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, weight, cv2.LINE_AA)

    header = max(1, round(58*unit))
    image[:header, :edge] = (image[:header, :edge]*.3).astype('uint8')
    text(f'{exercise_name} | {pose.get("mode", "")}', margin, round(22*unit), edge-2*margin)
    text(f'{elapsed_sec:.1f}s  {pose.get("fps", 0):.1f} FPS', margin, round(45*unit), edge-2*margin, .45)
    for index, side in enumerate(('left', 'right')):
        data = pose.get(side, {})
        top = index*(height//2)
        left, right = edge+margin, width-margin
        bar_width = max(8, round(18*unit))
        bar_left = right-bar_width
        space = max(1, bar_left-left-margin)
        color = (255, 190, 80) if side == 'left' else (100, 220, 255)
        text(f'{side.upper()} REPS', left, top+round(25*unit), space, .6, color, True)
        count = data.get('rep_count')
        count_text = '--' if count is None else str(int(count))
        if pose.get('mode') == 'CALIBRATION' and count is not None:
            count_text += ' / 3'
        text(count_text, left, top+round(61*unit), space, 1.4, color, True)
        result = quality_result(data, pose.get('mode', ''), pose.get('metric_type', ''))
        if result:
            hex_color = result['color'].lstrip('#')
            grade_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
            text(f"{result['label'].upper()} [{result['code']}] +{result['points']:g}",
                 left, top+round(87*unit), space, .55, grade_color, True)
        elif pose.get('mode') == 'CALIBRATION':
            text('CALIBRATION', left, top+round(87*unit), space, .4, color)
        else:
            text('RESULT --', left, top+round(87*unit), space, .4)
        total = data.get('total_score')
        total_text = f'{total:g}' if total is not None and math.isfinite(total) else '--'
        text('TARGET 3' if pose.get('mode') == 'CALIBRATION' else f'TOTAL {total_text}',
             left, top+round(110*unit), space, .55, color, True)
        value = data.get('val')
        factor = 1
        suffix = ' px' if pose.get('metric_unit', pose.get('measurement', {}).get('metric_unit')) == 'px' else ' deg' if pose.get('metric_type') == 'ANGLE' else ''
        digits = 4 if pose.get('metric_type') == 'COORDINATE' else 2
        value_text = '--' if value is None or not math.isfinite(value) else f'{value*factor:.{digits}f}{suffix}'
        text(f'Value {value_text}', left, top+round(131*unit), space, .42)
        maximum = pose.get('display_thresholds', {}).get(side, {}).get('target_val')
        if maximum is not None and math.isfinite(maximum):
            goal_label = 'GOAL' if pose.get('measurement', {}).get('count_on') == 'TARGET_POSITION' else 'MAX'
            text(f'{goal_label} {maximum*factor:.{digits}f}{suffix}', left, top+round(150*unit), space, .36)
        progress = motion_progress(pose, side)
        bar_top, bar_bottom = top+round(34*unit), top+round(155*unit)
        cv2.rectangle(image, (bar_left, bar_top), (right, bar_bottom), (65, 65, 65), -1)
        if progress is not None and progress > 0:
            fill_top = bar_bottom-round((bar_bottom-bar_top)*progress)
            cv2.rectangle(image, (bar_left, fill_top), (right, bar_bottom), color, -1)
        if (pose.get('mode') == 'MAIN'
                and pose.get('measurement', {}).get('count_on') != 'TARGET_POSITION'):
            for grade, fraction in QUALITY_THRESHOLDS.items():
                hex_color = QUALITY_COLORS[grade].lstrip('#')
                grade_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))
                y = bar_bottom-round((bar_bottom-bar_top)*fraction)
                cv2.line(image, (bar_left-2, y), (right, y), grade_color, max(2, thickness))
                text('N' if grade == 'BAD' else grade[0], bar_left-round(10*unit), y+round(3*unit),
                     max(1, round(8*unit)), .3, grade_color, True)
        activation = pose.get('measurement', {}).get('calibration_activation_fraction')
        if pose.get('mode') == 'CALIBRATION' and activation is not None:
            count_y = bar_bottom-round((bar_bottom-bar_top)*activation)
            cv2.line(image, (bar_left-3, count_y), (right, count_y), (255, 255, 255), thickness)
        label = '--' if progress is None else f'{progress:.0%}'
        status = (f'COUNT {activation:.0%} | {label}' if pose.get('mode') == 'CALIBRATION' and activation is not None
                  else f'{label}  {data.get("state", "")}')
        text(status, left, top+round(173*unit), space, .36)
    if pose.get('is_occluded'):
        text('Tracking unavailable', margin, height-margin, edge-2*margin, .55, (60, 100, 255), True)
    return image
