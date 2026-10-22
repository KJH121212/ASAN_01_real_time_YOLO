# 한국어 코드 해설: 캘리브레이션 이벤트의 영상 시각을 ASS 자막 시각으로 바꾸고 횟수 변화와 상태를 자막으로 표시합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Render calibration frame logs as ASS subtitles without re-encoding video."""

# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path


# `HEADER` 값에 '[Script Info]\nScriptType: v4.00+\nPlayResX: 1280\nPlayResY: 720\nWrapStyle: 2\n\n[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\nStyle: HUD,Arial,25,&H00FFFFFF,&H000000FF,&H00000000,&H90000000,0,0,0,0,100,100,0,0,3,2,0,7,24,24,24,1\nStyle: Count,Arial,45,&H0000FFFF,&H000000FF,&H00000000,&H90000000,-1,0,0,0,100,100,0,0,1,3,1,2,24,24,45,1\n\n[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n'을 저장합니다. 대상: `HEADER`.
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


# ass 시각 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def ass_time(centiseconds):
    # `(seconds, fraction)` 값에 `divmod(max(0, centiseconds), 100)` 호출 결과를 저장합니다. 대상: `(seconds, fraction)`.
    seconds, fraction = divmod(max(0, centiseconds), 100)
    # `(minutes, seconds)` 값에 `divmod(seconds, 60)` 호출 결과를 저장합니다. 대상: `(minutes, seconds)`.
    minutes, seconds = divmod(seconds, 60)
    # `(hours, minutes)` 값에 `divmod(minutes, 60)` 호출 결과를 저장합니다. 대상: `(hours, minutes)`.
    hours, minutes = divmod(minutes, 60)
    # 현재 함수 처리를 끝내고 계산식 `f'{hours}:{minutes:02}:{seconds:02}.{fraction:02}'`의 결과를 호출한 쪽으로 반환합니다.
    return f'{hours}:{minutes:02}:{seconds:02}.{fraction:02}'


# `safe` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def safe(value):
    # 현재 함수 처리를 끝내고 `str(value).replace('\\', '/').replace('{', '(').replace('}', ')').replace('\n', ' ')` 호출 결과를 호출한 쪽으로 반환합니다.
    return str(value).replace('\\', '/').replace('{', '(').replace('}', ')').replace('\n', ' ')


# 기록 캘리브레이션 ass 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def write_calibration_ass(csv_path, output_path, duration_sec=None):
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """Use source-video timestamps from an OFFLINE_VIDEO calibration CSV."""
    # `Path(csv_path).open(encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as file:
        # `rows` 값에 `list(csv.DictReader(file))` 호출 결과를 저장합니다. 대상: `rows`.
        rows = list(csv.DictReader(file))
    # `(events, slots)` 값에 순서·구성을 지정한 `([], {})`을 저장합니다. 대상: `(events, slots)`.
    events, slots = [], {}
    # `last_counts` 값에 키 'left', 'right'로 구성한 사전을 저장합니다. 대상: `last_counts`.
    last_counts = {'left': 0, 'right': 0}
    # pulse 횟수에 0을 저장합니다. 대상: `pulse_count`.
    pulse_count = 0
    # `rows`에서 항목을 차례로 꺼내 `row`에 넣고 아래 처리를 반복합니다.
    for row in rows:
        # 시작에 `max(0, round(float(row['timestamp_ms']) / 10))` 호출 결과를 저장합니다. 대상: `start`.
        start = max(0, round(float(row['timestamp_ms']) / 10))
        # `lines` 값에 순서·구성을 지정한 `[safe(row['motion_name']) + ' | ' + safe(row.get('calib_step', ''))]`을 저장합니다. 대상: `lines`.
        lines = [safe(row['motion_name']) + ' | ' + safe(row.get('calib_step', ''))]
        # `(('left', 'L'), ('right', 'R'))`에서 항목을 차례로 꺼내 `(side, label)`에 넣고 아래 처리를 반복합니다.
        for side, label in (('left', 'L'), ('right', 'R')):
            # 횟수에 `int(row.get(f'{side}_rep_count', 0))` 호출 결과를 저장합니다. 대상: `count`.
            count = int(row.get(f'{side}_rep_count', 0))
            # 측정값에 계산식 `row.get(f'{side}_value') or 'MISSING'`의 결과를 저장합니다. 대상: `value`.
            value = row.get(f'{side}_value') or 'MISSING'
            # 동작 품질 등급에 `row.get(f'{side}_quality', '')` 호출 결과를 저장합니다. 대상: `quality`.
            quality = row.get(f'{side}_quality', '')
            # 처리 단계에 `row.get(f'{side}_state', '')` 호출 결과를 저장합니다. 대상: `state`.
            state = row.get(f'{side}_state', '')
            # 계산에 사용할 관절 또는 좌표 묶음에 `row.get(f'{side}_total_score', '0')` 호출 결과를 저장합니다. 대상: `points`.
            points = row.get(f'{side}_total_score', '0')
            # `lines.append(safe(f'{label}: count={count} value={value} state={state} {quality} points={points}'))`을 호출하여 항목 하나를 끝에 추가합니다.
            lines.append(safe(f'{label}: count={count}  value={value}  state={state}  {quality}  points={points}'))
            # 조건 `row.get(f'{side}_count_updated') == 'True' and count > last_counts[side] and (row.get('is_occluded') != 'True')`을 만족하는 경우에만 아래 분기를 실행합니다.
            if row.get(f'{side}_count_updated') == 'True' and count > last_counts[side] and row.get('is_occluded') != 'True':
                # 횟수 증가를 이벤트 식별 기준으로 삼아 유지 중인 플래그를 중복 반영하지 않습니다.
                # `text` 값에 `safe(f'{label} COUNT +1 -> {count} {quality}')` 호출 결과를 저장합니다. 대상: `text`.
                text = safe(f'{label} COUNT +1  -> {count}  {quality}')
                # `position` 값에 `side == 'left'`이면 `'{\\an2\\pos(640,640)}'`, 아니면 `'{\\an2\\pos(640,575)}'`을 저장합니다. 대상: `position`.
                position = r'{\an2\pos(640,640)}' if side == 'left' else r'{\an2\pos(640,575)}'
                # `events.append((1, start, start + 100, 'Count', position + text))`을 호출하여 항목 하나를 끝에 추가합니다.
                events.append((1, start, start + 100, 'Count', position + text))
                # pulse 횟수에 `1`을 더해 누적합니다. 대상: `pulse_count`.
                pulse_count += 1
            # `last_counts[side]` 값에 횟수(`count`)을 저장합니다. 대상: `last_counts[side]`.
            last_counts[side] = count
        # 조건 `row.get('is_occluded') == 'True'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if row.get('is_occluded') == 'True':
            # `lines.append('OCCLUDED / DETECTION LOST - calibration may reset')`을 호출하여 항목 하나를 끝에 추가합니다.
            lines.append('OCCLUDED / DETECTION LOST - calibration may reset')
        # `slots[start]` 값에 `'\\N'.join(lines)` 호출 결과를 저장합니다. 대상: `slots[start]`.
        slots[start] = r'\N'.join(lines)
    # `starts` 값에 `sorted(slots)` 호출 결과를 저장합니다. 대상: `starts`.
    starts = sorted(slots)
    # `end` 값에 `duration_sec is not None`이면 `round(duration_sec * 100)`, 아니면 `starts[-1] + 100 if starts else 100`을 저장합니다. 대상: `end`.
    end = round(duration_sec * 100) if duration_sec is not None else (starts[-1] + 100 if starts else 100)
    # `enumerate(starts)`에서 항목을 차례로 꺼내 `(index, start)`에 넣고 아래 처리를 반복합니다.
    for index, start in enumerate(starts):
        # 중지에 `index + 1 < len(starts)`이면 `starts[index + 1]`, 아니면 `max(start + 1, end)`을 저장합니다. 대상: `stop`.
        stop = starts[index + 1] if index + 1 < len(starts) else max(start + 1, end)
        # `events.append((0, start, stop, 'HUD', slots[start]))`을 호출하여 항목 하나를 끝에 추가합니다.
        events.append((0, start, stop, 'HUD', slots[start]))
    # 조건 `rows and rows[-1].get('calib_step') == 'FINISHED'`을 만족하는 경우에만 아래 분기를 실행합니다.
    if rows and rows[-1].get('calib_step') == 'FINISHED':
        # 종료에 계산식 `starts[-1]`의 결과를 저장합니다. 대상: `finish`.
        finish = starts[-1]
        # `events.append((1, finish, max(finish + 1, end), 'Count', 'CALIBRATION FINISHED - L/R 3 reps'))`을 호출하여 항목 하나를 끝에 추가합니다.
        events.append((1, finish, max(finish + 1, end), 'Count', 'CALIBRATION FINISHED - L/R 3 reps'))
    # 조건 `rows`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif rows:
        # 종료 시 실제 미완료 상태도 표시합니다. 정답 횟수가 없으므로 정확도를 단정하지 않습니다.
        # 시작에 `max(0, end - 150)` 호출 결과를 저장합니다. 대상: `start`.
        start = max(0, end - 150)
        # `events.append((1, start, max(start + 1, end), 'Count', 'CALIBRATION INCOMPLETE AT VIDEO END'))`을 호출하여 항목 하나를 끝에 추가합니다.
        events.append((1, start, max(start + 1, end), 'Count', 'CALIBRATION INCOMPLETE AT VIDEO END'))
    # 경로에 `Path(output_path)` 호출 결과를 저장합니다. 대상: `path`.
    path = Path(output_path)
    # `path.parent.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
    path.parent.mkdir(parents=True, exist_ok=True)
    # `path.open('w', encoding='utf-8-sig', newline='\n')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with path.open('w', encoding='utf-8-sig', newline='\n') as file:
        # `file.write(HEADER)`을 호출하여 파일 또는 영상에 데이터를 기록합니다.
        file.write(HEADER)
        # `sorted(events, key=lambda item: (item[1], item[0]))`에서 항목을 차례로 꺼내 `(layer, start, stop, style, text)`에 넣고 아래 처리를 반복합니다.
        for layer, start, stop, style, text in sorted(events, key=lambda item: (item[1], item[0])):
            # `file.write(f'Dialogue: {layer},{ass_time(start)},{ass_time(stop)},{style},,0,0,0,,{text}\n')`을 호출하여 파일 또는 영상에 데이터를 기록합니다.
            file.write(f'Dialogue: {layer},{ass_time(start)},{ass_time(stop)},{style},,0,0,0,,{text}\n')
    # 현재 함수 처리를 끝내고 키 'subtitle_path', 'count_events'로 구성한 사전을 호출한 쪽으로 반환합니다.
    return {'subtitle_path': str(path.resolve()), 'count_events': pulse_count}
