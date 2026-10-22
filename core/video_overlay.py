# 한국어 코드 해설: 저장 영상을 찾아 분석 로그와 맞추고, 로그가 없으면 소켓 서버로 분석한 뒤 스켈레톤·가동 범위 막대를 그립니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Background rendering of saved measurements onto videos; no AI models in the UI."""

# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# `datetime`에서 `datetime` 기능을 가져옵니다.
from datetime import datetime
# 이 파일에서 사용할 `hashlib` 모듈을 불러옵니다.
import hashlib
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `threading` 모듈을 불러옵니다.
import threading
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time

# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np
# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets

# `utils.image_geometry`에서 `resize_within` 기능을 가져옵니다.
from utils.image_geometry import resize_within
# `utils.overlay_renderer`에서 `OverlayRenderer` 기능을 가져옵니다.
from utils.overlay_renderer import OverlayRenderer

# 영상 extensions에 순서·구성을 지정한 `{'.mp4', '.mov', '.avi', '.mkv', '.m4v'}`을 저장합니다. 대상: `VIDEO_EXTENSIONS`.
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}
# `EXERCISES` 값에 키 'biceps_curl', 'shoulder_press', 'clamshell', 'slr', 'hip_knee_flexion', 'knee_extension', 'leg_raise', 'leg_…로 구성한 사전을 저장합니다. 대상: `EXERCISES`.
EXERCISES = {'biceps_curl': 'biceps_curl', 'shoulder_press': 'shoulder_press',
             # 구성 항목: `clamshell` 값(`'clamshell'`)에 'clamshell'; `slr` 값(`'slr'`)에 'slr'; hip 무릎 flexion(`'hip_knee_flexion'`)에 'hip_knee_flexion'을 지정합니다.
             'clamshell': 'clamshell', 'slr': 'slr', 'hip_knee_flexion': 'hip_knee_flexion',
             # 구성 항목: 무릎 신전(`'knee_extension'`)에 'knee_extension'; `leg_raise` 값(`'leg_raise'`)에 'hip_knee_flexion'을 지정합니다.
             'knee_extension': 'knee_extension', 'leg_raise': 'hip_knee_flexion',
             # 구성 항목: leg 신전(`'leg_extension'`)에 'knee_extension'을 지정합니다.
             'leg_extension': 'knee_extension'}


# 지정 폴더 아래의 원본 영상만 찾고 이미 생성한 오버레이 결과는 다시 처리하지 않습니다.
def find_videos(root):
    # 최상위 경로에 `Path(root).resolve()` 호출 결과를 저장합니다. 대상: `root`.
    root = Path(root).resolve()
    # 현재 함수 처리를 끝내고 `sorted((path for path in root.rglob('*') if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS and ('_overlays' not in path.relative_to(root).parts) and (not pat…` 호출 결과를 호출한 쪽으로 반환합니다.
    return sorted(path for path in root.rglob('*') if path.is_file() and
                  # 앞줄에서 시작한 식·인수·자료 구조를 `path.suffix.lower() in VIDEO_EXTENSIONS and` 내용으로 이어서 구성합니다.
                  path.suffix.lower() in VIDEO_EXTENSIONS and
                  # 앞줄에서 시작한 식·인수·자료 구조를 `'_overlays' not in path.relative_to(root).parts and` 내용으로 이어서 구성합니다.
                  '_overlays' not in path.relative_to(root).parts and
                  # 앞줄에서 시작한 식·인수·자료 구조를 `not path.stem.lower().endswith(('_overlay', '.overlay')))` 내용으로 이어서 구성합니다.
                  not path.stem.lower().endswith(('_overlay', '.overlay')))


# `infer_exercise` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def infer_exercise(path):
    # `reversed(path.parts)`에서 항목을 차례로 꺼내 `part`에 넣고 아래 처리를 반복합니다.
    for part in reversed(path.parts):
        # 조건 `part in EXERCISES`을 만족하는 경우에만 아래 분기를 실행합니다.
        if part in EXERCISES:
            # 현재 함수 처리를 끝내고 계산식 `EXERCISES[part]`의 결과를 호출한 쪽으로 반환합니다.
            return EXERCISES[part]
    # 현재 함수 처리를 끝내고 `next((exercise for (name, exercise) in EXERCISES.items() if name in path.stem.lower()), None)` 호출 결과를 호출한 쪽으로 반환합니다.
    return next((exercise for name, exercise in EXERCISES.items() if name in path.stem.lower()), None)


# usable 로그 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def usable_log(path):
    # 조건 `not Path(path).is_file()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not Path(path).is_file():
        # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
        return False
    # `Path(path).open(encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        # `row` 값에 `next(csv.DictReader(file), None)` 호출 결과를 저장합니다. 대상: `row`.
        row = next(csv.DictReader(file), None)
    # 현재 함수 처리를 끝내고 `bool(row and 'frame_idx' in row and (number(row.get('video_frame_idx'), -1) >= 0))` 호출 결과를 호출한 쪽으로 반환합니다.
    return bool(row and 'frame_idx' in row and number(row.get('video_frame_idx'), -1) >= 0)


# `number` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def number(value, default=0.):
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # 함수 또는 처리 단계의 결과에 `float(value)` 호출 결과를 저장합니다. 대상: `result`.
        result = float(value)
        # 현재 함수 처리를 끝내고 `np.isfinite(result)`이면 `result`, 아니면 `default`을 호출한 쪽으로 반환합니다.
        return result if np.isfinite(result) else default
    # `(TypeError, ValueError)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
    except (TypeError, ValueError):
        # 현재 함수 처리를 끝내고 기본(`default`)을 호출한 쪽으로 반환합니다.
        return default


# `draw_measurements` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def draw_measurements(frame, row, stale=False, flashes=()):
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """Draw two ROM bars, measured values, count, quality and calibration state."""
    # `(height, width)` 값에 계산식 `frame.shape[:2]`의 결과를 저장합니다. 대상: `(height, width)`.
    height, width = frame.shape[:2]
    # `panel_height` 값에 `min(height, max(90, round(height * 0.24)))` 호출 결과를 저장합니다. 대상: `panel_height`.
    panel_height = min(height, max(90, round(height * .24)))
    # `shade` 값에 `frame.copy()` 호출 결과를 저장합니다. 대상: `shade`.
    shade = frame.copy()
    # `cv2.rectangle(shade, (0, 0), (width, panel_height), (15, 15, 15), -1)`을 호출하여 해당 단계의 작업을 수행합니다.
    cv2.rectangle(shade, (0, 0), (width, panel_height), (15, 15, 15), -1)
    # `cv2.addWeighted(shade, 0.8, frame, 0.2, 0, frame)`을 호출하여 해당 단계의 작업을 수행합니다.
    cv2.addWeighted(shade, .8, frame, .2, 0, frame)
    # `scale` 값에 `max(0.3, min(0.7, width / 1000))` 호출 결과를 저장합니다. 대상: `scale`.
    scale = max(.3, min(.7, width / 1000))
    # `line_height` 값에 `max(14, round(26 * scale / 0.7))` 호출 결과를 저장합니다. 대상: `line_height`.
    line_height = max(14, round(26 * scale / .7))
    # `title` 값에 계산식 `f"{row.get('motion_name', '')} | {row.get('calib_step', '')}"`의 결과를 저장합니다. 대상: `title`.
    title = f"{row.get('motion_name', '')} | {row.get('calib_step', '')}"
    # 조건 `stale`을 만족하는 경우에만 아래 분기를 실행합니다.
    if stale:
        # `title` 값에 `' | ANALYSIS ENDED'`을 더해 누적합니다. 대상: `title`.
        title += ' | ANALYSIS ENDED'
    # 조건 `str(row.get('is_occluded')) == 'True'`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif str(row.get('is_occluded')) == 'True':
        # `title` 값에 `' | OCCLUDED'`을 더해 누적합니다. 대상: `title`.
        title += ' | OCCLUDED'
    # `cv2.putText(frame, title, (10, line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
    cv2.putText(frame, title, (10, line_height), cv2.FONT_HERSHEY_SIMPLEX, scale,
                # 앞줄에서 시작한 식·인수·자료 구조를 `(255, 255, 255), 1, cv2.LINE_AA)` 내용으로 이어서 구성합니다.
                (255, 255, 255), 1, cv2.LINE_AA)
    # `enumerate((('left', (255, 180, 40)), ('right', (40, 210, 255))))`에서 항목을 차례로 꺼내 `(index, (side, color))`에 넣고 아래 처리를 반복합니다.
    for index, (side, color) in enumerate((('left', (255, 180, 40)), ('right', (40, 210, 255)))):
        # `x` 값에 계산식 `10 + index * (width // 2)`의 결과를 저장합니다. 대상: `x`.
        x = 10 + index * (width // 2)
        # `y` 값에 계산식 `line_height * 2`의 결과를 저장합니다. 대상: `y`.
        y = line_height * 2
        # 측정값에 `number(row.get(f'{side}_value'), None)` 호출 결과를 저장합니다. 대상: `value`.
        value = number(row.get(f'{side}_value'), None)
        # 시작에 `number(row.get(f'{side}_start_val'))` 호출 결과를 저장합니다. 대상: `start`.
        start = number(row.get(f'{side}_start_val'))
        # 목표에 `number(row.get(f'{side}_target_val'))` 호출 결과를 저장합니다. 대상: `target`.
        target = number(row.get(f'{side}_target_val'))
        # 비율에 `value is not None and target != start and (not stale)`이면 `min(1.0, abs(value - start) / abs(target - start))`, 아니면 `0.0`을 저장합니다. 대상: `ratio`.
        ratio = min(1., abs(value - start) / abs(target - start)) if value is not None and target != start and not stale else 0.
        # `label` 값에 계산식 `f"{side.upper()} reps={row.get(f'{side}_rep_count', 0)} points={row.get(f'{side}_total_score', 0)}"`의 결과를 저장합니다. 대상: `label`.
        label = f"{side.upper()}  reps={row.get(f'{side}_rep_count', 0)}  points={row.get(f'{side}_total_score', 0)}"
        # `cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        # 측정값 text에 `value is None or stale`이면 `'-'`, 아니면 `f'{value:.2f}'`을 저장합니다. 대상: `value_text`.
        value_text = '-' if value is None or stale else f'{value:.2f}'
        # `cv2.putText(frame, f"value={value_text} {row.get(f'{side}_quality', '')}", (x, y + line_height), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
        cv2.putText(frame, f"value={value_text}  {row.get(f'{side}_quality', '')}", (x, y + line_height),
                    # 앞줄에서 시작한 식·인수·자료 구조를 `cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)` 내용으로 이어서 구성합니다.
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), 1, cv2.LINE_AA)
        # `(bar_y, bar_width)` 값에 순서·구성을 지정한 `(y + line_height + 6, max(1, width // 2 - 25))`을 저장합니다. 대상: `(bar_y, bar_width)`.
        bar_y, bar_width = y + line_height + 6, max(1, width // 2 - 25)
        # `cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + 9), (65, 65, 65), -1)`을 호출하여 해당 단계의 작업을 수행합니다.
        cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + 9), (65, 65, 65), -1)
        # 조건 `ratio > 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if ratio > 0:
            # `cv2.rectangle(frame, (x, bar_y), (x + round(bar_width * ratio), bar_y + 9), color, -1)`을 호출하여 해당 단계의 작업을 수행합니다.
            cv2.rectangle(frame, (x, bar_y), (x + round(bar_width * ratio), bar_y + 9), color, -1)
        # 처리 단계에 계산식 `row.get(f'{side}_calibration_status') or row.get(f'{side}_state', '')`의 결과를 저장합니다. 대상: `state`.
        state = row.get(f'{side}_calibration_status') or row.get(f'{side}_state', '')
        # `cv2.putText(frame, f'ROM {ratio * 100:.0f}% | {state}', (x, bar_y + line_height + 8), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
        cv2.putText(frame, f"ROM {ratio * 100:.0f}% | {state}", (x, bar_y + line_height + 8),
                    # 앞줄에서 시작한 식·인수·자료 구조를 `cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)` 내용으로 이어서 구성합니다.
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        # 조건 `side in flashes`을 만족하는 경우에만 아래 분기를 실행합니다.
        if side in flashes:
            # `cv2.putText(frame, f'{side.upper()} COUNT +1', (x, max(panel_height + 25, height - 35)), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.3, color, 2, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
            cv2.putText(frame, f'{side.upper()} COUNT +1', (x, max(panel_height + 25, height - 35)),
                        # 앞줄에서 시작한 식·인수·자료 구조를 `cv2.FONT_HERSHEY_SIMPLEX, scale * 1.3, color, 2, cv2.LINE_AA)` 내용으로 이어서 구성합니다.
                        cv2.FONT_HERSHEY_SIMPLEX, scale * 1.3, color, 2, cv2.LINE_AA)
    # 현재 함수 처리를 끝내고 OpenCV BGR 영상 프레임(`frame`)을 호출한 쪽으로 반환합니다.
    return frame


# 원본 영상과 프레임 로그를 동기화하여 스켈레톤·측정 막대·횟수 변화가 보이는 새 영상을 저장합니다.
def render_video(source, output, csv_path, npz_path=None, cancel=None, on_progress=None):
    # `(source, output)` 값에 순서·구성을 지정한 `(Path(source), Path(output))`을 저장합니다. 대상: `(source, output)`.
    source, output = Path(source), Path(output)
    # 조건 `output.exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if output.exists():
        # 이 조건에서는 처리를 계속할 수 없으므로 `FileExistsError(f'Output already exists: {output}')` 예외를 발생시킵니다.
        raise FileExistsError(f'Output already exists: {output}')
    # `Path(csv_path).open(encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with Path(csv_path).open(encoding='utf-8-sig', newline='') as file:
        # `rows` 값에 `list(csv.DictReader(file))` 호출 결과를 저장합니다. 대상: `rows`.
        rows = list(csv.DictReader(file))
    # 조건 `not rows`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not rows:
        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('No analyzed frames in CSV')` 예외를 발생시킵니다.
        raise ValueError('No analyzed frames in CSV')
    # `timeline` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `timeline`.
    timeline = []
    # `rows`에서 항목을 차례로 꺼내 `row`에 넣고 아래 처리를 반복합니다.
    for row in rows:
        # `index` 값에 `int(number(row.get('video_frame_idx'), -1))` 호출 결과를 저장합니다. 대상: `index`.
        index = int(number(row.get('video_frame_idx'), -1))
        # 조건 `index < 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if index < 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('CSV has no video_frame_idx; frame alignment cannot be verified')` 예외를 발생시킵니다.
            raise ValueError('CSV has no video_frame_idx; frame alignment cannot be verified')
        # `timeline.append((index, row))`을 호출하여 항목 하나를 끝에 추가합니다.
        timeline.append((index, row))
    # 조건 `any((a[0] > b[0] for (a, b) in zip(timeline, timeline[1:])))`을 만족하는 경우에만 아래 분기를 실행합니다.
    if any(a[0] > b[0] for a, b in zip(timeline, timeline[1:])):
        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('CSV video frame indices are not ordered')` 예외를 발생시킵니다.
        raise ValueError('CSV video frame indices are not ordered')
    # `poses` 값에 빈 사전을 저장합니다. 대상: `poses`.
    poses = {}
    # 조건 `npz_path and Path(npz_path).is_file()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if npz_path and Path(npz_path).is_file():
        # `np.load(npz_path, allow_pickle=False)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with np.load(npz_path, allow_pickle=False) as data:
            # `indices` 값에 `'frame_idx' in data`이면 `data['frame_idx']`, 아니면 `range(len(data['keypoints']))`을 저장합니다. 대상: `indices`.
            indices = data['frame_idx'] if 'frame_idx' in data else range(len(data['keypoints']))
            # `poses` 값에 반복·조건식 `{int(index): matrix for (index, matrix) in zip(indices, data['keypoints'])}`으로 구성한 값을 저장합니다. 대상: `poses`.
            poses = {int(index): matrix for index, matrix in zip(indices, data['keypoints'])}
    # OpenCV 영상 입력 장치에 `cv2.VideoCapture(str(source))` 호출 결과를 저장합니다. 대상: `cap`.
    cap = cv2.VideoCapture(str(source))
    # CSV 또는 영상 출력 객체에 None을 저장합니다. 대상: `writer`.
    writer = None
    # `success` 값에 False을 저장합니다. 대상: `success`.
    success = False
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # 조건 `not cap.isOpened()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not cap.isOpened():
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(f'Cannot open video: {source}')` 예외를 발생시킵니다.
            raise RuntimeError(f'Cannot open video: {source}')
        # 영상의 초당 프레임 수에 `number(cap.get(cv2.CAP_PROP_FPS), 0)` 호출 결과를 저장합니다. 대상: `fps`.
        fps = number(cap.get(cv2.CAP_PROP_FPS), 0)
        # 조건 `fps <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if fps <= 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Source video has invalid FPS')` 예외를 발생시킵니다.
            raise ValueError('Source video has invalid FPS')
        # `total` 값에 `int(cap.get(cv2.CAP_PROP_FRAME_COUNT))` 호출 결과를 저장합니다. 대상: `total`.
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # `renderer` 값에 `OverlayRenderer()` 호출 결과를 저장합니다. 대상: `renderer`.
        renderer = OverlayRenderer()
        # `(cursor, index, current, points)` 값에 순서·구성을 지정한 `(0, 0, None, [])`을 저장합니다. 대상: `(cursor, index, current, points)`.
        cursor, index, current, points = 0, 0, None, []
        # `flashes` 값에 키 'left', 'right'로 구성한 사전을 저장합니다. 대상: `flashes`.
        flashes = {'left': -1, 'right': -1}
        # `output.parent.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
        output.parent.mkdir(parents=True, exist_ok=True)
        # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
        while True:
            # 조건 `cancel and cancel.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
            if cancel and cancel.is_set():
                # 이 조건에서는 처리를 계속할 수 없으므로 `InterruptedError('Overlay cancelled')` 예외를 발생시킵니다.
                raise InterruptedError('Overlay cancelled')
            # `(ok, frame)` 값에 `cap.read()` 호출 결과를 저장합니다. 대상: `(ok, frame)`.
            ok, frame = cap.read()
            # 조건 `not ok`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not ok:
                # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                break
            # OpenCV BGR 영상 프레임에 `resize_within(frame, 1280, 1280)` 호출 결과를 저장합니다. 대상: `frame`.
            frame = resize_within(frame, 1280, 1280)
            # mp4v 인코딩을 위해 출력 가로·세로 길이를 짝수로 맞춥니다.
            # `(height, width)` 값에 계산식 `frame.shape[:2]`의 결과를 저장합니다. 대상: `(height, width)`.
            height, width = frame.shape[:2]
            # OpenCV BGR 영상 프레임에 계산식 `frame[:height - height % 2, :width - width % 2]`의 결과를 저장합니다. 대상: `frame`.
            frame = frame[:height - height % 2, :width - width % 2]
            # 조건 `writer is None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if writer is None:
                # CSV 또는 영상 출력 객체에 `cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))` 호출 결과를 저장합니다. 대상: `writer`.
                writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                         # 앞줄에서 시작한 식·인수·자료 구조를 `(frame.shape[1], frame.shape[0]))` 내용으로 이어서 구성합니다.
                                         (frame.shape[1], frame.shape[0]))
                # 조건 `not writer.isOpened()`을 만족하는 경우에만 아래 분기를 실행합니다.
                if not writer.isOpened():
                    # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(f'Cannot create overlay video: {output}')` 예외를 발생시킵니다.
                    raise RuntimeError(f'Cannot create overlay video: {output}')
            # 조건 `cursor < len(timeline) and timeline[cursor][0] <= index`이 유지되는 동안 아래 처리를 반복합니다.
            while cursor < len(timeline) and timeline[cursor][0] <= index:
                # `current` 값에 계산식 `timeline[cursor][1]`의 결과를 저장합니다. 대상: `current`.
                current = timeline[cursor][1]
                # `matrix` 값에 `poses.get(int(current['frame_idx']))` 호출 결과를 저장합니다. 대상: `matrix`.
                matrix = poses.get(int(current['frame_idx']))
                # 계산에 사용할 관절 또는 좌표 묶음에 `matrix is None`이면 `[]`, 아니면 `[dict(id=i, x=float(p[0]), y=float(p[1]), score=float(p[2])) for (i, p) in enumerate(matrix) if np.isfinite(p).all()]`을 저장합니다. 대상: `points`.
                points = [] if matrix is None else [dict(id=i, x=float(p[0]), y=float(p[1]), score=float(p[2]))
                    # 앞줄에서 시작한 식·인수·자료 구조를 `for i, p in enumerate(matrix) if np.isfinite(p).all()]` 내용으로 이어서 구성합니다.
                    for i, p in enumerate(matrix) if np.isfinite(p).all()]
                # `flashes`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
                for side in flashes:
                    # 조건 `current.get(f'{side}_count_updated') == 'True' and current.get('is_occluded') != 'True'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if current.get(f'{side}_count_updated') == 'True' and current.get('is_occluded') != 'True':
                        # `flashes[side]` 값에 계산식 `timeline[cursor][0] + round(fps * 0.8)`의 결과를 저장합니다. 대상: `flashes[side]`.
                        flashes[side] = timeline[cursor][0] + round(fps * .8)
                # `cursor` 값에 `1`을 더해 누적합니다. 대상: `cursor`.
                cursor += 1
            # 오래된에 계산식 `index > timeline[-1][0]`의 결과를 저장합니다. 대상: `stale`.
            stale = index > timeline[-1][0]
            # 조건 `current`을 만족하는 경우에만 아래 분기를 실행합니다.
            if current:
                # 조건 `not stale`을 만족하는 경우에만 아래 분기를 실행합니다.
                if not stale:
                    # OpenCV BGR 영상 프레임에 `renderer.draw_skeleton(frame, points, current.get('is_occluded') == 'True')` 호출 결과를 저장합니다. 대상: `frame`.
                    frame = renderer.draw_skeleton(frame, points, current.get('is_occluded') == 'True')
                # OpenCV BGR 영상 프레임에 `draw_measurements(frame, current, stale, [side for (side, end) in flashes.items() if index <= end])` 호출 결과를 저장합니다. 대상: `frame`.
                frame = draw_measurements(frame, current, stale,
                                          # 앞줄에서 시작한 식·인수·자료 구조를 `[side for side, end in flashes.items() if index <= end])` 내용으로 이어서 구성합니다.
                                          [side for side, end in flashes.items() if index <= end])
            # `writer.write(frame)`을 호출하여 파일 또는 영상에 데이터를 기록합니다.
            writer.write(frame)
            # `index` 값에 `1`을 더해 누적합니다. 대상: `index`.
            index += 1
            # 조건 `on_progress and index % max(1, round(fps)) == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
            if on_progress and index % max(1, round(fps)) == 0:
                # `on_progress(index, total)`을 호출하여 해당 단계의 작업을 수행합니다.
                on_progress(index, total)
        # 조건 `index == 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if index == 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('No readable source frames')` 예외를 발생시킵니다.
            raise ValueError('No readable source frames')
        # `success` 값에 True을 저장합니다. 대상: `success`.
        success = True
        # 현재 함수 처리를 끝내고 `dict(frames=index, fps=fps, skeleton=bool(poses))` 호출 결과를 호출한 쪽으로 반환합니다.
        return dict(frames=index, fps=fps, skeleton=bool(poses))
    # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
    finally:
        # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
        cap.release()
        # 조건 `writer is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if writer is not None:
            # `writer.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            writer.release()
        # 조건 `not success`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not success:
            # `output.unlink(missing_ok=True)`을 호출하여 지정한 파일을 제거합니다.
            output.unlink(missing_ok=True)


# `VideoOverlayJob` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class VideoOverlayJob:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, root, uri, analyze_missing=True):
        # `(self.root, self.uri)` 값에 순서·구성을 지정한 `(Path(root).resolve(), uri)`을 저장합니다. 대상: `(self.root, self.uri)`.
        self.root, self.uri = Path(root).resolve(), uri
        # analyze 결측에 analyze 결측(`analyze_missing`)을 저장합니다. 대상: `self.analyze_missing`.
        self.analyze_missing = analyze_missing
        # 생성할 결과 데이터 또는 경로에 계산식 `self.root / '_overlays' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')`의 결과를 저장합니다. 대상: `self.output`.
        self.output = self.root / '_overlays' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        # 취소에 스레드 사이의 상태 전달 신호(`threading.Event()`)을 저장합니다. 대상: `self.cancel`.
        self.cancel = threading.Event()
        # 스레드 사이의 공유 상태를 보호하는 잠금에 스레드 공유 데이터용 잠금(`threading.Lock()`)을 저장합니다. 대상: `self._lock`.
        self._lock = threading.Lock()
        # 단계에 `dict(running=False, total=0, completed=0, current='', phase='', frame=0, frames=0, results=[], output=str(self.output))` 호출 결과를 저장합니다. 대상: `self._state`.
        self._state = dict(running=False, total=0, completed=0, current='', phase='',
                           # 구성 항목: OpenCV BGR 영상 프레임(`frame`)에 0; 영상 프레임 묶음(`frames`)에 0; `results` 값(`results`)에 순서·구성을 지정한 `[]`; 생성할 결과 데이터 또는 경로(`output`)에 `str(self.output)` 호출 결과를 지정합니다.
                           frame=0, frames=0, results=[], output=str(self.output))

    # `snapshot` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def snapshot(self):
        # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self._lock:
            # 현재 함수 처리를 끝내고 키 'results'로 구성한 사전을 호출한 쪽으로 반환합니다.
            return {**self._state, 'results': [dict(row) for row in self._state['results']]}

    # 갱신 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _update(self, **values):
        # `self._lock`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self._lock:
            # `self._state.update(values)`을 호출하여 해당 단계의 작업을 수행합니다.
            self._state.update(values)

    # 시작 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def start(self):
        # 조건 `not self.root.is_dir()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.root.is_dir():
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'Video folder does not exist: {self.root}')` 예외를 발생시킵니다.
            raise ValueError(f'Video folder does not exist: {self.root}')
        # 조건 `hasattr(self, '_thread')`을 만족하는 경우에만 아래 분기를 실행합니다.
        if hasattr(self, '_thread'):
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Create a new job to run again')` 예외를 발생시킵니다.
            raise RuntimeError('Create a new job to run again')
        # `self._update(running=True)`을 호출하여 해당 단계의 작업을 수행합니다.
        self._update(running=True)
        # 백그라운드 작업 스레드에 백그라운드 실행 스레드(`threading.Thread(target=self._worker, daemon=True)`)을 저장합니다. 대상: `self._thread`.
        self._thread = threading.Thread(target=self._worker, daemon=True)
        # `self._thread.start()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._thread.start()

    # `_analyze` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def _analyze(self, source, exercise):
        # `websockets.connect(self.uri, open_timeout=5, close_timeout=2, ping_interval=None)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(self.uri, open_timeout=5, close_timeout=2,
                                      # 구성 항목: `ping_interval` 값(`ping_interval`)에 None을 지정합니다.
                                      ping_interval=None) as ws:
            # `(deadline, sent, stopped)` 값에 순서·구성을 지정한 `(time.monotonic() + 900, False, False)`을 저장합니다. 대상: `(deadline, sent, stopped)`.
            deadline, sent, stopped = time.monotonic() + 900, False, False
            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:
                # 조건 `self.cancel.is_set() and (not sent)`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self.cancel.is_set() and not sent:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `InterruptedError('Overlay cancelled')` 예외를 발생시킵니다.
                    raise InterruptedError('Overlay cancelled')
                # 조건 `self.cancel.is_set() and sent and (not stopped)`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self.cancel.is_set() and sent and not stopped:
                    # `ws.send(json.dumps({'type': 'CMD_STOP_SESSION', 'reason': 'USER_STOP'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await ws.send(json.dumps({'type': 'CMD_STOP_SESSION', 'reason': 'USER_STOP'}))
                    # `stopped` 값에 True을 저장합니다. 대상: `stopped`.
                    stopped = True
                # 조건 `time.monotonic() > deadline`을 만족하는 경우에만 아래 분기를 실행합니다.
                if time.monotonic() > deadline:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `TimeoutError('Socket analysis timeout')` 예외를 발생시킵니다.
                    raise TimeoutError('Socket analysis timeout')
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # 수신 또는 전송할 JSON 메시지 사전에 JSON 문자열을 해석한 Python 값(`json.loads(await asyncio.wait_for(ws.recv(), 0.2))`)을 저장합니다. 대상: `packet`.
                    packet = json.loads(await asyncio.wait_for(ws.recv(), .2))
                # `asyncio.TimeoutError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except asyncio.TimeoutError:
                    # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                    continue
                # `kind` 값에 `packet.get('type')` 호출 결과를 저장합니다. 대상: `kind`.
                kind = packet.get('type')
                # 조건 `kind == 'SERVER_STATUS'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if kind == 'SERVER_STATUS':
                    # 조건 `packet.get('model_state') == 'FAILED'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet.get('model_state') == 'FAILED':
                        # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(packet.get('model_error', 'Model loading failed'))` 예외를 발생시킵니다.
                        raise RuntimeError(packet.get('model_error', 'Model loading failed'))
                    # 조건 `packet.get('ready') and (not sent)`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if packet.get('ready') and not sent:
                        # 조건 `self.cancel.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if self.cancel.is_set():
                            # 이 조건에서는 처리를 계속할 수 없으므로 `InterruptedError('Overlay cancelled')` 예외를 발생시킵니다.
                            raise InterruptedError('Overlay cancelled')
                        # `identity` 값에 계산식 `hashlib.sha256(str(source).encode()).hexdigest()[:12]`의 결과를 저장합니다. 대상: `identity`.
                        identity = hashlib.sha256(str(source).encode()).hexdigest()[:12]
                        # `ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'patient_id': f'overlay_{identity}', 'exercise_name': exercise, 'mode': 'TEST', 'output_mode': 'LIGHTWEIGHT', 'video_path'…`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                        await ws.send(json.dumps({'type': 'CMD_SET_SESSION', 'patient_id': f'overlay_{identity}',
                            # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 `exercise` 값(`exercise`); 실행 모드(`'mode'`)에 'TEST'; 출력 mode(`'output_mode'`)에 'LIGHTWEIGHT'을 지정합니다.
                            'exercise_name': exercise, 'mode': 'TEST', 'output_mode': 'LIGHTWEIGHT',
                            # 구성 항목: 서버 PC에서 읽을 원본 영상 경로(`'video_path'`)에 `str(source)` 호출 결과; 영상 녹화 여부(`'record_video'`)에 True; 한 번에 추론 또는 학습할 표본 수(`'batch_size'`)에 8을 지정합니다.
                            'video_path': str(source), 'record_video': True, 'batch_size': 8,
                            # 구성 항목: 세션 로그를 저장하는 최상위 폴더(`'log_root'`)에 `str(self.output / 'analysis')` 호출 결과; 측정 세션(`'session'`)에 1을 지정합니다.
                            'log_root': str(self.output / 'analysis'), 'session': 1}))
                        # `sent` 값에 True을 저장합니다. 대상: `sent`.
                        sent = True
                # 조건 `kind == 'VIDEO_PROGRESS'`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif kind == 'VIDEO_PROGRESS':
                    # `self._update(frame=packet['processed_frames'], frames=packet['total_frames'])`을 호출하여 해당 단계의 작업을 수행합니다.
                    self._update(frame=packet['processed_frames'], frames=packet['total_frames'])
                # 조건 `kind == 'ERROR'`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif kind == 'ERROR':
                    # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(packet.get('message', 'Server error'))` 예외를 발생시킵니다.
                    raise RuntimeError(packet.get('message', 'Server error'))
                # 조건 `kind in ('VIDEO_ANALYSIS_COMPLETED', 'SESSION_FINISHED')`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif kind in ('VIDEO_ANALYSIS_COMPLETED', 'SESSION_FINISHED'):
                    # 조건 `self.cancel.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if self.cancel.is_set():
                        # 이 조건에서는 처리를 계속할 수 없으므로 `InterruptedError('Overlay cancelled')` 예외를 발생시킵니다.
                        raise InterruptedError('Overlay cancelled')
                    # 현재 함수 처리를 끝내고 계산식 `packet['summary']`의 결과를 호출한 쪽으로 반환합니다.
                    return packet['summary']

    # 작업 스레드 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _worker(self):
        # `results` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `results`.
        results = []
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `videos` 값에 `find_videos(self.root)` 호출 결과를 저장합니다. 대상: `videos`.
            videos = find_videos(self.root)  # Snapshot once, before creating any output videos.
            # 조건 `not videos`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not videos:
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('No videos found in the selected folder')` 예외를 발생시킵니다.
                raise ValueError('No videos found in the selected folder')
            # `self.output.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
            self.output.mkdir(parents=True, exist_ok=True)
            # `self._update(total=len(videos))`을 호출하여 해당 단계의 작업을 수행합니다.
            self._update(total=len(videos))
            # `videos`에서 항목을 차례로 꺼내 `source`에 넣고 아래 처리를 반복합니다.
            for source in videos:
                # 조건 `self.cancel.is_set()`을 만족하는 경우에만 아래 분기를 실행합니다.
                if self.cancel.is_set():
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break
                # `row` 값에 `dict(source=str(source), output='', status='', message='', analysis='SAVED_LOG')` 호출 결과를 저장합니다. 대상: `row`.
                row = dict(source=str(source), output='', status='', message='', analysis='SAVED_LOG')
                # `self._update(current=str(source), phase='PREPARING', frame=0, frames=0)`을 호출하여 해당 단계의 작업을 수행합니다.
                self._update(current=str(source), phase='PREPARING', frame=0, frames=0)
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `(csv_path, npz_path)` 값에 순서·구성을 지정한 `(source.with_suffix('.csv'), source.with_suffix('.npz'))`을 저장합니다. 대상: `(csv_path, npz_path)`.
                    csv_path, npz_path = source.with_suffix('.csv'), source.with_suffix('.npz')
                    # manifest 경로에 `source.with_suffix('.json')` 호출 결과를 저장합니다. 대상: `manifest_path`.
                    manifest_path = source.with_suffix('.json')
                    # 조건 `manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')).get('status') == 'RECORDING'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')).get('status') == 'RECORDING':
                        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Session is still recording; finish it before rendering')` 예외를 발생시킵니다.
                        raise ValueError('Session is still recording; finish it before rendering')
                    # 조건 `not usable_log(csv_path)`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if not usable_log(csv_path):
                        # `row['analysis']` 값에 'SERVER_DEFAULT_ROM'을 저장합니다. 대상: `row['analysis']`.
                        row['analysis'] = 'SERVER_DEFAULT_ROM'
                        # `exercise` 값에 `infer_exercise(source)` 호출 결과를 저장합니다. 대상: `exercise`.
                        exercise = infer_exercise(source)
                        # 조건 `not self.analyze_missing`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if not self.analyze_missing:
                            # `row.update(status='SKIPPED', message='No matching CSV')`을 호출하여 해당 단계의 작업을 수행합니다.
                            row.update(status='SKIPPED', message='No matching CSV')
                        # 조건 `not exercise`을 만족하는 경우에만 아래 분기를 실행합니다.
                        elif not exercise:
                            # `row.update(status='SKIPPED', message='Cannot identify exercise from folder/file name')`을 호출하여 해당 단계의 작업을 수행합니다.
                            row.update(status='SKIPPED', message='Cannot identify exercise from folder/file name')
                        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                        else:
                            # `self._update(phase='SOCKET_ANALYSIS')`을 호출하여 해당 단계의 작업을 수행합니다.
                            self._update(phase='SOCKET_ANALYSIS')
                            # 저장·종료 결과 요약에 `asyncio.run(self._analyze(source, exercise))` 호출 결과를 저장합니다. 대상: `summary`.
                            summary = asyncio.run(self._analyze(source, exercise))
                            # `(csv_path, npz_path)` 값에 순서·구성을 지정한 `(Path(summary['frame_csv']), Path(summary['skeleton_npz']))`을 저장합니다. 대상: `(csv_path, npz_path)`.
                            csv_path, npz_path = Path(summary['frame_csv']), Path(summary['skeleton_npz'])
                    # 조건 `not row['status']`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if not row['status']:
                        # `self._update(phase='RENDERING', frame=0, frames=0)`을 호출하여 해당 단계의 작업을 수행합니다.
                        self._update(phase='RENDERING', frame=0, frames=0)
                        # 생성할 결과 데이터 또는 경로에 계산식 `self.output / 'videos' / source.relative_to(self.root).parent / (source.name + '.overlay.mp4')`의 결과를 저장합니다. 대상: `output`.
                        output = self.output / 'videos' / source.relative_to(self.root).parent / (source.name + '.overlay.mp4')
                        # `render_video(source, output, csv_path, npz_path, cancel=self.cancel, on_progress=lambda frame, total: self._update(frame=frame, frames=total))`을 호출하여 해당 단계의 작업을 수행합니다.
                        render_video(source, output, csv_path, npz_path, cancel=self.cancel,
                                     # 이 줄의 입력 인수: `frame` = OpenCV BGR 영상 프레임; `total` = `total` 값. 기본값·형식은 아래 선언을 따릅니다.
                                     on_progress=lambda frame, total: self._update(frame=frame, frames=total))
                        # `row.update(status='COMPLETED', output=str(output))`을 호출하여 해당 단계의 작업을 수행합니다.
                        row.update(status='COMPLETED', output=str(output))
                # `InterruptedError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except InterruptedError:
                    # `row.update(status='CANCELLED', message='Cancelled by user')`을 호출하여 해당 단계의 작업을 수행합니다.
                    row.update(status='CANCELLED', message='Cancelled by user')
                # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except Exception as exc:
                    # `row.update(status='ERROR', message=str(exc))`을 호출하여 해당 단계의 작업을 수행합니다.
                    row.update(status='ERROR', message=str(exc))
                # `results.append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
                results.append(row)
                # `self._update(completed=len(results), results=list(results))`을 호출하여 해당 단계의 작업을 수행합니다.
                self._update(completed=len(results), results=list(results))
                # `(self.output / 'results.csv').open('w', encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with (self.output / 'results.csv').open('w', encoding='utf-8-sig', newline='') as file:
                    # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=['source', 'output', 'status', 'analysis', 'message'])` 호출 결과를 저장합니다. 대상: `writer`.
                    writer = csv.DictWriter(file, fieldnames=['source', 'output', 'status', 'analysis', 'message'])
                    # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
                    writer.writeheader()
                    # `writer.writerows(results)`을 호출하여 CSV에 여러 행을 기록합니다.
                    writer.writerows(results)
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception as exc:
            # `self._update(error=str(exc))`을 호출하여 해당 단계의 작업을 수행합니다.
            self._update(error=str(exc))
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `self._update(running=False, phase='CANCELLED' if self.cancel.is_set() else 'FINISHED')`을 호출하여 해당 단계의 작업을 수행합니다.
            self._update(running=False, phase='CANCELLED' if self.cancel.is_set() else 'FINISHED')
