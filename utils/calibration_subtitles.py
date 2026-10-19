"""Render calibration frame logs as ASS subtitles without re-encoding video."""

import csv
from pathlib import Path


HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 720
WrapStyle: 2

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: HUD,Arial,25,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,3,2,0,7,24,24,24,1
Style: Count,Arial,45,&H0000FFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,3,1,2,24,24,45,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def ass_time(centiseconds):
    seconds, fraction = divmod(max(0, centiseconds), 100)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours}:{minutes:02}:{seconds:02}.{fraction:02}'


def safe(value):
    return str(value).replace('\\', '/').replace('{', '(').replace('}', ')').replace('\n', ' ')


def write_calibration_ass(csv_path, output_path, duration_sec=None):
    """Use source-video timestamps from an OFFLINE_VIDEO calibration CSV."""
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    events, slots = [], {}
    last_counts = {'left': 0, 'right': 0}
    pulse_count = 0
    for row in rows:
        start = max(0, round(float(row['timestamp_ms']) / 10))
        lines = [safe(row['motion_name']) + ' | ' + safe(row.get('calib_step', ''))]
        for side, label in (('left', 'L'), ('right', 'R')):
            count = int(row.get(f'{side}_rep_count', 0))
            value = row.get(f'{side}_value') or 'MISSING'
            quality = row.get(f'{side}_quality', '')
            state = row.get(f'{side}_state', '')
            points = row.get(f'{side}_total_score', '0')
            lines.append(safe(f'{label}: count={count}  value={value}  state={state}  {quality}  points={points}'))
            if row.get(f'{side}_count_updated') == 'True' and count > last_counts[side] and row.get('is_occluded') != 'True':
                # Count is the authoritative event identifier; don't duplicate held flags.
                text = safe(f'{label} COUNT +1  -> {count}  {quality}')
                position = r'{\an2\pos(640,640)}' if side == 'left' else r'{\an2\pos(640,575)}'
                events.append((1, start, start + 100, 'Count', position + text))
                pulse_count += 1
            last_counts[side] = count
        if row.get('is_occluded') == 'True':
            lines.append('OCCLUDED / DETECTION LOST - calibration may reset')
        slots[start] = r'\N'.join(lines)
    starts = sorted(slots)
    end = round(duration_sec * 100) if duration_sec is not None else (starts[-1] + 100 if starts else 100)
    for index, start in enumerate(starts):
        stop = starts[index + 1] if index + 1 < len(starts) else max(start + 1, end)
        events.append((0, start, stop, 'HUD', slots[start]))
    if rows and rows[-1].get('calib_step') == 'FINISHED':
        finish = starts[-1]
        events.append((1, finish, max(finish + 1, end), 'Count', 'CALIBRATION FINISHED - L/R 3 reps'))
    elif rows:
        # Show the actual unfinished state at the end; no accuracy claim is made.
        start = max(0, end - 150)
        events.append((1, start, max(start + 1, end), 'Count', 'CALIBRATION INCOMPLETE AT VIDEO END'))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='\n') as file:
        file.write(HEADER)
        for layer, start, stop, style, text in sorted(events, key=lambda item: (item[1], item[0])):
            file.write(f'Dialogue: {layer},{ass_time(start)},{ass_time(stop)},{style},,0,0,0,,{text}\n')
    return {'subtitle_path': str(path.resolve()), 'count_events': pulse_count}
