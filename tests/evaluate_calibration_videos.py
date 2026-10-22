# 한국어 코드 해설: `evaluate_calibration_videos`에서 사용하는 실행·분석 보조 기능을 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Run real offline calibration on a video corpus, isolated from patient profiles.

python tests/evaluate_calibration_videos.py --inventory-only
python tests/evaluate_calibration_videos.py --output data/calibration_experiments/baseline
"""

# 이 파일에서 사용할 `argparse` 모듈을 불러옵니다.
import argparse
# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio
# `collections`에서 `Counter`, `defaultdict` 기능을 가져옵니다.
from collections import Counter, defaultdict
# `contextlib`에서 `redirect_stdout`, `redirect_stderr` 기능을 가져옵니다.
from contextlib import redirect_stdout, redirect_stderr
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
# 이 파일에서 사용할 `platform` 모듈을 불러옵니다.
import platform
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time
# 이 파일에서 사용할 `traceback` 모듈을 불러옵니다.
import traceback

# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# 최상위 경로에 계산식 `Path(__file__).resolve().parents[1]`의 결과를 저장합니다. 대상: `ROOT`.
ROOT = Path(__file__).resolve().parents[1]
# `sys.path.insert(0, str(ROOT))`을 호출하여 해당 단계의 작업을 수행합니다.
sys.path.insert(0, str(ROOT))
# `EXERCISES` 값에 키 'biceps_curl', 'shoulder_press', 'clamshell', 'slr', 'leg_raise', 'leg_extension'로 구성한 사전을 저장합니다. 대상: `EXERCISES`.
EXERCISES = {'biceps_curl': 'biceps_curl', 'shoulder_press': 'shoulder_press',
             # 구성 항목: `clamshell` 값(`'clamshell'`)에 'clamshell'; `slr` 값(`'slr'`)에 'slr'을 지정합니다.
             'clamshell': 'clamshell', 'slr': 'slr',
             # 구성 항목: `leg_raise` 값(`'leg_raise'`)에 'hip_knee_flexion'; leg 신전(`'leg_extension'`)에 'knee_extension'을 지정합니다.
             'leg_raise': 'hip_knee_flexion', 'leg_extension': 'knee_extension'}
# `EXTENSIONS` 값에 순서·구성을 지정한 `{'.mp4', '.mov', '.avi', '.mkv', '.m4v'}`을 저장합니다. 대상: `EXTENSIONS`.
EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v'}


# `inventory` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def inventory(root):
    # `entries` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `entries`.
    entries = []
    # `sorted(root.rglob('*'))`에서 항목을 차례로 꺼내 `path`에 넣고 아래 처리를 반복합니다.
    for path in sorted(root.rglob('*')):
        # 조건 `path.suffix.lower() not in EXTENSIONS or not path.is_file()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
            continue
        # `relative` 값에 `path.relative_to(root)` 호출 결과를 저장합니다. 대상: `relative`.
        relative = path.relative_to(root)
        # `folder` 값에 계산식 `relative.parts[0]`의 결과를 저장합니다. 대상: `folder`.
        folder = relative.parts[0]
        # OpenCV 영상 입력 장치에 `cv2.VideoCapture(str(path))` 호출 결과를 저장합니다. 대상: `cap`.
        cap = cv2.VideoCapture(str(path))
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 영상의 초당 프레임 수에 `cap.get(cv2.CAP_PROP_FPS)` 호출 결과를 저장합니다. 대상: `fps`.
            fps = cap.get(cv2.CAP_PROP_FPS)
            # 영상 프레임 묶음에 `cap.isOpened()`이면 `int(cap.get(cv2.CAP_PROP_FRAME_COUNT))`, 아니면 `0`을 저장합니다. 대상: `frames`.
            frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
            # `entries.append(dict(video_id=hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12], source=str(path), relative_path=relative.as_posix(), folder=folder, exercise=…`을 호출하여 항목 하나를 끝에 추가합니다.
            entries.append(dict(video_id=hashlib.sha256(relative.as_posix().encode()).hexdigest()[:12],
                # 구성 항목: 읽을 원본 경로 또는 데이터(`source`)에 `str(path)` 호출 결과; relative 경로(`relative_path`)에 `relative.as_posix()` 호출 결과; `folder` 값(`folder`)에 `folder` 값(`folder`)을 지정합니다.
                source=str(path), relative_path=relative.as_posix(), folder=folder,
                # 구성 항목: `exercise` 값(`exercise`)에 `EXERCISES.get(folder)` 호출 결과; `opened` 값(`opened`)에 `cap.isOpened()` 호출 결과; 영상의 초당 프레임 수(`fps`)에 영상의 초당 프레임 수(`fps`)을 지정합니다.
                exercise=EXERCISES.get(folder), opened=cap.isOpened(), fps=fps,
                # 구성 항목: 영상 프레임 묶음(`frames`)에 영상 프레임 묶음(`frames`); 길이 sec(`duration_sec`)에 `fps > 0`이면 `frames / fps`, 아니면 `0`을 지정합니다.
                frames=frames, duration_sec=frames / fps if fps > 0 else 0,
                # 구성 항목: `width` 값(`width`)에 `int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))` 호출 결과; `height` 값(`height`)에 `int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))` 호출 결과를 지정합니다.
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                # 구성 항목: 크기 bytes(`size_bytes`)에 st 크기(`path.stat().st_size`); `modified_ns` 값(`modified_ns`)에 `path.stat().st_mtime_ns` 값(`path.stat().st_mtime_ns`)을 지정합니다.
                size_bytes=path.stat().st_size, modified_ns=path.stat().st_mtime_ns))
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()
    # 현재 함수 처리를 끝내고 `entries` 값(`entries`)을 호출한 쪽으로 반환합니다.
    return entries


# 기록 csv 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def write_csv(path, records):
    # 조건 `not records`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not records:
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return
    # `fields` 값에 `list(dict.fromkeys((key for row in records for key in row)))` 호출 결과를 저장합니다. 대상: `fields`.
    fields = list(dict.fromkeys(key for row in records for key in row))
    # 원본을 안전하게 교체하기 위한 임시 파일 경로에 `path.with_suffix('.csv.tmp')` 호출 결과를 저장합니다. 대상: `temporary`.
    temporary = path.with_suffix('.csv.tmp')
    # `temporary.open('w', encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with temporary.open('w', encoding='utf-8-sig', newline='') as file:
        # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=fields)` 호출 결과를 저장합니다. 대상: `writer`.
        writer = csv.DictWriter(file, fieldnames=fields)
        # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
        writer.writeheader()
        # `writer.writerows(records)`을 호출하여 CSV에 여러 행을 기록합니다.
        writer.writerows(records)
    # `temporary.replace(path)`을 호출하여 임시 파일을 최종 경로로 교체합니다.
    temporary.replace(path)


# `CaptureEngine` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class CaptureEngine:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, engine):
        # `(self.engine, self.raw)` 값에 순서·구성을 지정한 `(engine, [])`을 저장합니다. 대상: `(self.engine, self.raw)`.
        self.engine, self.raw = engine, []

    # 영상 프레임 묶음의 사람 검출과 자세 추론을 실행하고 입력 순서에 대응하는 관절 목록을 반환합니다.
    def extract_keypoints_batch(self, frames):
        # 함수 또는 처리 단계의 결과에 `self.engine.extract_keypoints_batch(frames)` 호출 결과를 저장합니다. 대상: `result`.
        result = self.engine.extract_keypoints_batch(frames)
        # `self.raw.extend(result)`을 호출하여 여러 항목을 순서대로 추가합니다.
        self.raw.extend(result)
        # 현재 함수 처리를 끝내고 함수 또는 처리 단계의 결과(`result`)을 호출한 쪽으로 반환합니다.
        return result

    # 저장 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def save(self, path, fps):
        # `matrix` 값에 `np.full((len(self.raw), 17, 3), np.nan, dtype=np.float32)` 호출 결과를 저장합니다. 대상: `matrix`.
        matrix = np.full((len(self.raw), 17, 3), np.nan, dtype=np.float32)
        # `matrix[:, :, 2]` 값에 0을 저장합니다. 대상: `matrix[:, :, 2]`.
        matrix[:, :, 2] = 0
        # `enumerate(self.raw)`에서 항목을 차례로 꺼내 `(index, points)`에 넣고 아래 처리를 반복합니다.
        for index, points in enumerate(self.raw):
            # `points or []`에서 항목을 차례로 꺼내 `point`에 넣고 아래 처리를 반복합니다.
            for point in points or []:
                # `matrix[index, point['id']]` 값에 순서·구성을 지정한 `[point['x'], point['y'], point['score']]`을 저장합니다. 대상: `matrix[index, point['id']]`.
                matrix[index, point['id']] = [point['x'], point['y'], point['score']]
        # `aspects` 값에 반복·조건식 `[points[0].get('image_aspect_ratio', 1.0) if points else np.nan for points in self.raw]`으로 구성한 값을 저장합니다. 대상: `aspects`.
        aspects = [points[0].get('image_aspect_ratio', 1.0) if points else np.nan for points in self.raw]
        # `np.savez_compressed(path, keypoints=matrix, fps=fps, image_aspect_ratio=aspects)`을 호출하여 해당 단계의 작업을 수행합니다.
        np.savez_compressed(path, keypoints=matrix, fps=fps, image_aspect_ratio=aspects)


# `ResultSink` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ResultSink:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self):
        # `self.messages` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.messages`.
        self.messages = []

    # 전송 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    async def send(self, encoded):
        # `self.messages.append(json.loads(encoded))`을 호출하여 항목 하나를 끝에 추가합니다.
        self.messages.append(json.loads(encoded))


# analyze 프레임 묶음 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def analyze_frames(path):
    # `Path(path).open(encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with Path(path).open(encoding='utf-8-sig', newline='') as file:
        # `rows` 값에 `list(csv.DictReader(file))` 호출 결과를 저장합니다. 대상: `rows`.
        rows = list(csv.DictReader(file))
    # 횟수에 `len(rows)` 호출 결과를 저장합니다. 대상: `count`.
    count = len(rows)
    # `last` 값에 `rows`이면 `rows[-1]`, 아니면 `{}`을 저장합니다. 대상: `last`.
    last = rows[-1] if rows else {}
    # `occluded` 값에 `sum((row['is_occluded'] == 'True' for row in rows))` 호출 결과를 저장합니다. 대상: `occluded`.
    occluded = sum(row['is_occluded'] == 'True' for row in rows)
    # 함수 또는 처리 단계의 결과에 `dict(analyzed_frames=count, analyzed_sec=float(last.get('timestamp_ms', 0)) / 1000, finished=last.get('calib_step') == 'FINISHED', final_step=last.get('calib_step', ''),…` 호출 결과를 저장합니다. 대상: `result`.
    result = dict(analyzed_frames=count, analyzed_sec=float(last.get('timestamp_ms', 0)) / 1000,
                  # 구성 항목: `finished` 값(`finished`)에 계산식 `last.get('calib_step') == 'FINISHED'`의 결과; `final_step` 값(`final_step`)에 `last.get('calib_step', '')` 호출 결과를 지정합니다.
                  finished=last.get('calib_step') == 'FINISHED', final_step=last.get('calib_step', ''),
                  # 구성 항목: `occluded_percent` 값(`occluded_percent`)에 `count`이면 `round(100 * occluded / count, 2)`, 아니면 `100`을 지정합니다.
                  occluded_percent=round(100 * occluded / count, 2) if count else 100,
                  # 구성 항목: collection 프레임 묶음(`collection_frames`)에 `sum((row['calib_step'] == 'COLLECTING' for row in rows))` 호출 결과를 지정합니다.
                  collection_frames=sum(row['calib_step'] == 'COLLECTING' for row in rows))
    # 초기화 횟수에 0을 저장합니다. 대상: `reset_count`.
    reset_count = 0
    # `zip(rows, rows[1:])`에서 항목을 차례로 꺼내 `(previous, current)`에 넣고 아래 처리를 반복합니다.
    for previous, current in zip(rows, rows[1:]):
        # 조건 `previous['calib_step'] != 'FULL_BODY_CHECK' and current['calib_step'] == 'FULL_BODY_CHECK'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if previous['calib_step'] != 'FULL_BODY_CHECK' and current['calib_step'] == 'FULL_BODY_CHECK':
            # 초기화 횟수에 `1`을 더해 누적합니다. 대상: `reset_count`.
            reset_count += 1
    # `result['occlusion_resets']` 값에 초기화 횟수(`reset_count`)을 저장합니다. 대상: `result['occlusion_resets']`.
    result['occlusion_resets'] = reset_count
    # `('left', 'right')`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
    for side in ('left', 'right'):
        # 측정값에 반복·조건식 `[float(row[f'{side}_value']) for row in rows if row[f'{side}_value'] != '']`으로 구성한 값을 저장합니다. 대상: `values`.
        values = [float(row[f'{side}_value']) for row in rows if row[f'{side}_value'] != '']
        # `result[f'{side}_final_reps']` 값에 `int(last.get(f'{side}_rep_count', 0))` 호출 결과를 저장합니다. 대상: `result[f'{side}_final_reps']`.
        result[f'{side}_final_reps'] = int(last.get(f'{side}_rep_count', 0))
        # `result[f'{side}_max_reps']` 값에 `max((int(row[f'{side}_rep_count']) for row in rows), default=0)` 호출 결과를 저장합니다. 대상: `result[f'{side}_max_reps']`.
        result[f'{side}_max_reps'] = max((int(row[f'{side}_rep_count']) for row in rows), default=0)
        # `result[f'{side}_returning_percent']` 값에 `count`이면 `round(100 * sum((row[f'{side}_state'] == 'RETURNING' for row in rows)) / count, 2)`, 아니면 `0`을 저장합니다. 대상: `result[f'{side}_returning_percent']`.
        result[f'{side}_returning_percent'] = round(100 * sum(row[f'{side}_state'] == 'RETURNING'
                                                               # 앞줄에서 시작한 식·인수·자료 구조를 `for row in rows) / count, 2) if count else 0` 내용으로 이어서 구성합니다.
                                                               for row in rows) / count, 2) if count else 0
        # `result[f'{side}_p05']` 값에 `values`이면 `round(float(np.percentile(values, 5)), 4)`, 아니면 `''`을 저장합니다. 대상: `result[f'{side}_p05']`.
        result[f'{side}_p05'] = round(float(np.percentile(values, 5)), 4) if values else ''
        # `result[f'{side}_p95']` 값에 `values`이면 `round(float(np.percentile(values, 95)), 4)`, 아니면 `''`을 저장합니다. 대상: `result[f'{side}_p95']`.
        result[f'{side}_p95'] = round(float(np.percentile(values, 95)), 4) if values else ''
    # 현재 함수 처리를 끝내고 함수 또는 처리 단계의 결과(`result`)을 호출한 쪽으로 반환합니다.
    return result


# `report` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def report(output, results):
    # `groups` 값에 `defaultdict(list)` 호출 결과를 저장합니다. 대상: `groups`.
    groups = defaultdict(list)
    # `results`에서 항목을 차례로 꺼내 `row`에 넣고 아래 처리를 반복합니다.
    for row in results:
        # `groups[row['folder']].append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
        groups[row['folder']].append(row)
    # `text` 값에 순서·구성을 지정한 `['# Calibration baseline experiment', '', 'Completion means FINISHED (both sides 3 reps plus cooldown), not verified counting accuracy.', 'Original offline pipeline, all…`을 저장합니다. 대상: `text`.
    text = ['# Calibration baseline experiment', '',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'Completion means FINISHED (both sides 3 reps plus cooldown), not verified counting accuracy.',` 내용으로 이어서 구성합니다.
            'Completion means FINISHED (both sides 3 reps plus cooldown), not verified counting accuracy.',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'Original offline pipeline, all frames at source FPS, One Euro filter, existing default ROM.',` 내용으로 이어서 구성합니다.
            'Original offline pipeline, all frames at source FPS, One Euro filter, existing default ROM.',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'No live 5-second preparation delay. Patient profiles are isolated in this experiment.',` 내용으로 이어서 구성합니다.
            'No live 5-second preparation delay. Patient profiles are isolated in this experiment.',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'Leg-raise variants share hip_knee_flexion provisionally. No production algorithm changes.', '',` 내용으로 이어서 구성합니다.
            'Leg-raise variants share hip_knee_flexion provisionally. No production algorithm changes.', '',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'| Folder | Videos | Finished | EOF incomplete | Errors |', '|---|---:|---:|---:|---:|']` 내용으로 이어서 구성합니다.
            '| Folder | Videos | Finished | EOF incomplete | Errors |', '|---|---:|---:|---:|---:|']
    # `sorted(groups.items())`에서 항목을 차례로 꺼내 `(folder, rows)`에 넣고 아래 처리를 반복합니다.
    for folder, rows in sorted(groups.items()):
        # `counts` 값에 `Counter((row['result'] for row in rows))` 호출 결과를 저장합니다. 대상: `counts`.
        counts = Counter(row['result'] for row in rows)
        # `text.append(f"| {folder} | {len(rows)} | {counts['FINISHED']} | {counts['EOF_NOT_FINISHED']} | {counts['ERROR']} |")`을 호출하여 항목 하나를 끝에 추가합니다.
        text.append(f"| {folder} | {len(rows)} | {counts['FINISHED']} | {counts['EOF_NOT_FINISHED']} | {counts['ERROR']} |")
    # `text` 값에 `['', '## Results', '', '- inventory.csv: source file metadata and folder-to-exercise mapping.', '- results.csv: completion, counts, occlusion, observed ranges and estima…`을 더해 누적합니다. 대상: `text`.
    text += ['', '## Results', '',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- inventory.csv: source file metadata and folder-to-exercise mapping.',` 내용으로 이어서 구성합니다.
             '- inventory.csv: source file metadata and folder-to-exercise mapping.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- results.csv: completion, counts, occlusion, observed ranges and estimated thresholds.',` 내용으로 이어서 구성합니다.
             '- results.csv: completion, counts, occlusion, observed ranges and estimated thresholds.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- raw_keypoints/: full-video model output for later replay without another GPU inference.',` 내용으로 이어서 구성합니다.
             '- raw_keypoints/: full-video model output for later replay without another GPU inference.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- sessions/: production CSV/NPZ/summary logs and isolated calibration profiles.',` 내용으로 이어서 구성합니다.
             '- sessions/: production CSV/NPZ/summary logs and isolated calibration profiles.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- diagnostics/: per-video runtime output and errors.', '',` 내용으로 이어서 구성합니다.
             '- diagnostics/: per-video runtime output and errors.', '',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'The existing server labels EOF as COMPLETED and saves thresholds even without FINISHED.',` 내용으로 이어서 구성합니다.
             'The existing server labels EOF as COMPLETED and saves thresholds even without FINISHED.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'Use result/finished in results.csv to judge completion; incomplete thresholds are not validated.']` 내용으로 이어서 구성합니다.
             'Use result/finished in results.csv to judge completion; incomplete thresholds are not validated.']
    # `(output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')`을 호출하여 해당 단계의 작업을 수행합니다.
    (output / 'report.md').write_text('\n'.join(text) + '\n', encoding='utf-8')


# 실행 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
async def run(args, entries, output):
    # `core.skeleton_engine`에서 `SkeletonEngine` 기능을 가져옵니다.
    from core.skeleton_engine import SkeletonEngine
    # `network.socket_server`에서 `ExerciseSocketServer` 기능을 가져옵니다.
    from network.socket_server import ExerciseSocketServer
    # 이 파일에서 사용할 `torch` 모듈을 불러옵니다.
    import torch
    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print('Loading real models...', flush=True)
    # 생성 또는 검증할 분석 엔진에 `SkeletonEngine(device=args.device)` 호출 결과를 저장합니다. 대상: `engine`.
    engine = SkeletonEngine(device=args.device)
    # 서버에 `ExerciseSocketServer(engine_factory=lambda report: engine)` 호출 결과를 저장합니다. 대상: `server`.
    server = ExerciseSocketServer(engine_factory=lambda report: engine)
    # `(output / 'environment.json').write_text(json.dumps({'python': platform.python_version(), 'torch': torch.__version__, 'opencv': cv2.__version__, 'device': engine.device,…`을 호출하여 해당 단계의 작업을 수행합니다.
    (output / 'environment.json').write_text(json.dumps({
        # 구성 항목: `python` 값(`'python'`)에 `platform.python_version()` 호출 결과; `torch` 값(`'torch'`)에 `torch.__version__` 값(`torch.__version__`); `opencv` 값(`'opencv'`)에 `cv2.__version__` 값(`cv2.__version__`)을 지정합니다.
        'python': platform.python_version(), 'torch': torch.__version__, 'opencv': cv2.__version__,
        # 구성 항목: `device` 값(`'device'`)에 `engine.device` 값(`engine.device`); `gpu` 값(`'gpu'`)에 `torch.cuda.is_available()`이면 `torch.cuda.get_device_name(0)`, 아니면 `None`을 지정합니다.
        'device': engine.device, 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        # 구성 항목: 한 번에 추론 또는 학습할 표본 수(`'batch_size'`)에 한 번에 추론 또는 학습할 표본 수(`args.batch_size`); 영상 최상위 경로(`'video_root'`)에 `str(args.video_root)` 호출 결과를 지정합니다.
        'batch_size': args.batch_size, 'video_root': str(args.video_root),
        # 구성 항목: `code_sha256` 값(`'code_sha256'`)에 반복·조건식 `{name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in ('core/skeleton_engine.py', 'core/motion_engine.py', 'core/session_controller.py', 'utils/filte…`으로 구성한 값을 지정합니다.
        'code_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                        # 앞줄에서 시작한 식·인수·자료 구조를 `for name in ('core/skeleton_engine.py', 'core/motion_engine.py',` 내용으로 이어서 구성합니다.
                        for name in ('core/skeleton_engine.py', 'core/motion_engine.py',
                                     # 앞줄에서 시작한 식·인수·자료 구조를 `'core/session_controller.py', 'utils/filters.py',` 내용으로 이어서 구성합니다.
                                     'core/session_controller.py', 'utils/filters.py',
                                     # 앞줄에서 시작한 식·인수·자료 구조를 `'utils/normalization.py', 'network/socket_server.py')},` 내용으로 이어서 구성합니다.
                                     'utils/normalization.py', 'network/socket_server.py')},
    # 구성 항목: `encoding` 값(`encoding`)에 'utf-8'; `ensure_ascii` 값(`ensure_ascii`)에 False; `indent` 값(`indent`)에 2을 지정합니다.
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    # 현재 작업 설정에 계산식 `ROOT / 'configs/exercise_kpt_config_coco.json'`의 결과를 저장합니다. 대상: `config`.
    config = ROOT / 'configs/exercise_kpt_config_coco.json'
    # `(output / 'exercise_config_snapshot.json').write_bytes(config.read_bytes())`을 호출하여 해당 단계의 작업을 수행합니다.
    (output / 'exercise_config_snapshot.json').write_bytes(config.read_bytes())
    # `results` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `results`.
    results = []
    # `enumerate(entries, 1)`에서 항목을 차례로 꺼내 `(index, entry)`에 넣고 아래 처리를 반복합니다.
    for index, entry in enumerate(entries, 1):
        # `row` 값에 `dict(entry)` 호출 결과를 저장합니다. 대상: `row`.
        row = dict(entry)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[{index}/{len(entries)}] START {entry['relative_path']} ({entry['duration_sec']:.1f}s)", flush=True)
        # 동작 시작 시각 또는 실행 여부에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `started`.
        started = time.perf_counter()
        # `capture` 값에 `CaptureEngine(engine)` 호출 결과를 저장합니다. 대상: `capture`.
        capture = CaptureEngine(engine)
        # 사람 검출·관절 추출 AI 엔진에 `capture` 값(`capture`)을 저장합니다. 대상: `server.skeleton_engine`.
        server.skeleton_engine = capture
        # `sink` 값에 `ResultSink()` 호출 결과를 저장합니다. 대상: `sink`.
        sink = ResultSink()
        # diagnostic 경로에 계산식 `output / 'diagnostics' / f"{entry['video_id']}.log"`의 결과를 저장합니다. 대상: `diagnostic_path`.
        diagnostic_path = output / 'diagnostics' / f"{entry['video_id']}.log"
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `diagnostic_path.open('w', encoding='utf-8'), redirect_stdout(diagnostic), redirect_stderr(diagnostic)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with diagnostic_path.open('w', encoding='utf-8') as diagnostic, redirect_stdout(diagnostic), redirect_stderr(diagnostic):
                # `server.process_offline_video(sink, {'video_path': entry['source'], 'exercise_name': entry['exercise'], 'patient_id': 'video_' + entry['video_id'], 'patient_name': 'Calib…`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
                await server.process_offline_video(sink, {
                    # 구성 항목: 서버 PC에서 읽을 원본 영상 경로(`'video_path'`)에 계산식 `entry['source']`의 결과; 운동 설정의 식별 이름(`'exercise_name'`)에 계산식 `entry['exercise']`의 결과를 지정합니다.
                    'video_path': entry['source'], 'exercise_name': entry['exercise'],
                    # 구성 항목: 환자 식별 번호(`'patient_id'`)에 계산식 `'video_' + entry['video_id']`의 결과; `patient_name` 값(`'patient_name'`)에 'Calibration experiment'을 지정합니다.
                    'patient_id': 'video_' + entry['video_id'], 'patient_name': 'Calibration experiment',
                    # 구성 항목: 실행 모드(`'mode'`)에 'CALIBRATION'; 한 번에 추론 또는 학습할 표본 수(`'batch_size'`)에 한 번에 추론 또는 학습할 표본 수(`args.batch_size`); 영상 녹화 여부(`'record_video'`)에 True을 지정합니다.
                    'mode': 'CALIBRATION', 'batch_size': args.batch_size, 'record_video': True,
                    # 구성 항목: 세션 로그를 저장하는 최상위 폴더(`'log_root'`)에 `str(output / 'sessions')` 호출 결과; 측정 세션(`'session'`)에 1; `round` 값(`'round'`)에 0; 설정(`'set'`)에 0을 지정합니다.
                    'log_root': str(output / 'sessions'), 'session': 1, 'round': 0, 'set': 0})
            # `completed` 값에 `next((message for message in sink.messages if message['type'] == 'VIDEO_ANALYSIS_COMPLETED'))` 호출 결과를 저장합니다. 대상: `completed`.
            completed = next(message for message in sink.messages if message['type'] == 'VIDEO_ANALYSIS_COMPLETED')
            # `row.update(analyze_frames(completed['summary']['frame_csv']))`을 호출하여 해당 단계의 작업을 수행합니다.
            row.update(analyze_frames(completed['summary']['frame_csv']))
            # `row.update(result='FINISHED' if row['finished'] else 'EOF_NOT_FINISHED', session_directory=completed['summary']['directory'])`을 호출하여 해당 단계의 작업을 수행합니다.
            row.update(result='FINISHED' if row['finished'] else 'EOF_NOT_FINISHED',
                       # 구성 항목: 측정 세션 폴더(`session_directory`)에 계산식 `completed['summary']['directory']`의 결과를 지정합니다.
                       session_directory=completed['summary']['directory'])
            # `(completed.get('thresholds') or {}).items()`에서 항목을 차례로 꺼내 `(side, limits)`에 넣고 아래 처리를 반복합니다.
            for side, limits in (completed.get('thresholds') or {}).items():
                # `row[f'{side}_estimated_start']` 값에 계산식 `limits['start_val']`의 결과를 저장합니다. 대상: `row[f'{side}_estimated_start']`.
                row[f'{side}_estimated_start'] = limits['start_val']
                # `row[f'{side}_estimated_target']` 값에 계산식 `limits['target_val']`의 결과를 저장합니다. 대상: `row[f'{side}_estimated_target']`.
                row[f'{side}_estimated_target'] = limits['target_val']
            # `capture.save(output / 'raw_keypoints' / f"{entry['video_id']}.npz", entry['fps'])`을 호출하여 해당 단계의 작업을 수행합니다.
            capture.save(output / 'raw_keypoints' / f"{entry['video_id']}.npz", entry['fps'])
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception as exc:
            # `row.update(result='ERROR', error=f'{type(exc).__name__}: {exc}')`을 호출하여 해당 단계의 작업을 수행합니다.
            row.update(result='ERROR', error=f'{type(exc).__name__}: {exc}')
            # `diagnostic_path.open('a', encoding='utf-8')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with diagnostic_path.open('a', encoding='utf-8') as diagnostic:
                # `traceback.print_exc(file=diagnostic)`을 호출하여 현재 예외의 상세 호출 경로를 출력합니다.
                traceback.print_exc(file=diagnostic)
            # 조건 `server.controller is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if server.controller is not None:
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `server._save_session_files(status='FAILED')`을 호출하여 해당 단계의 작업을 수행합니다.
                    server._save_session_files(status='FAILED')
                # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except Exception:
                    # 이 분기에는 추가 동작이 없으며 문법적으로 필요한 빈 블록만 유지합니다.
                    pass
            # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `server.controller`.
            server.controller = None
        # `row['runtime_sec']` 값에 `round(time.perf_counter() - started, 3)` 호출 결과를 저장합니다. 대상: `row['runtime_sec']`.
        row['runtime_sec'] = round(time.perf_counter() - started, 3)
        # `results.append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
        results.append(row)
        # `write_csv(output / 'results.csv', results)`을 호출하여 해당 단계의 작업을 수행합니다.
        write_csv(output / 'results.csv', results)
        # `report(output, results)`을 호출하여 해당 단계의 작업을 수행합니다.
        report(output, results)
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[{index}/{len(entries)}] {row['result']} L={row.get('left_final_reps', '?')} "
              # 앞줄에서 시작한 식·인수·자료 구조를 `f"R={row.get('right_final_reps', '?')} occluded={row.get('occluded_percent', '?')}% "` 내용으로 이어서 구성합니다.
              f"R={row.get('right_final_reps', '?')} occluded={row.get('occluded_percent', '?')}% "
              # 구성 항목: `flush` 값(`flush`)에 True을 지정합니다.
              f"runtime={row['runtime_sec']:.1f}s", flush=True)


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def main():
    # `parser` 값에 `argparse.ArgumentParser(description=__doc__)` 호출 결과를 저장합니다. 대상: `parser`.
    parser = argparse.ArgumentParser(description=__doc__)
    # `parser.add_argument('--video-root', type=Path, default=ROOT / 'test_video')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--video-root', type=Path, default=ROOT / 'test_video')
    # `parser.add_argument('--output', type=Path)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--output', type=Path)
    # `parser.add_argument('--inventory-only', action='store_true')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--inventory-only', action='store_true')
    # `parser.add_argument('--exercise', choices=list(EXERCISES), action='append')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--exercise', choices=list(EXERCISES), action='append')
    # `parser.add_argument('--limit-per-exercise', type=int)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--limit-per-exercise', type=int)
    # `parser.add_argument('--batch-size', type=int, default=8)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--batch-size', type=int, default=8)
    # `parser.add_argument('--device', default=None)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--device', default=None)
    # `args` 값에 `parser.parse_args()` 호출 결과를 저장합니다. 대상: `args`.
    args = parser.parse_args()
    # 조건 `args.batch_size <= 0 or (args.limit_per_exercise is not None and args.limit_per_exercise <= 0)`을 만족하는 경우에만 아래 분기를 실행합니다.
    if args.batch_size <= 0 or (args.limit_per_exercise is not None and args.limit_per_exercise <= 0):
        # `parser.error('Batch size and limit must be positive')`을 호출하여 해당 단계의 작업을 수행합니다.
        parser.error('Batch size and limit must be positive')
    # 영상 최상위 경로에 `args.video_root.resolve()` 호출 결과를 저장합니다. 대상: `args.video_root`.
    args.video_root = args.video_root.resolve()
    # 조건 `not args.video_root.is_dir()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not args.video_root.is_dir():
        # `parser.error(f'Video folder not found: {args.video_root}')`을 호출하여 해당 단계의 작업을 수행합니다.
        parser.error(f'Video folder not found: {args.video_root}')
    # 생성할 결과 데이터 또는 경로에 `(args.output or ROOT / 'data/calibration_experiments' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()` 호출 결과를 저장합니다. 대상: `output`.
    output = (args.output or ROOT / 'data/calibration_experiments' / datetime.now().strftime('%Y%m%d_%H%M%S')).resolve()
    # `output.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
    output.mkdir(parents=True, exist_ok=True)
    # 조건 `(output / 'results.csv').exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if (output / 'results.csv').exists():
        # `parser.error('Output already contains results; choose a new output directory')`을 호출하여 해당 단계의 작업을 수행합니다.
        parser.error('Output already contains results; choose a new output directory')
    # `entries` 값에 `inventory(args.video_root)` 호출 결과를 저장합니다. 대상: `entries`.
    entries = inventory(args.video_root)
    # `entries` 값에 반복·조건식 `[entry for entry in entries if entry['exercise'] and (not args.exercise or entry['folder'] in args.exercise)]`으로 구성한 값을 저장합니다. 대상: `entries`.
    entries = [entry for entry in entries if entry['exercise'] and
               # 앞줄에서 시작한 식·인수·자료 구조를 `(not args.exercise or entry['folder'] in args.exercise)]` 내용으로 이어서 구성합니다.
               (not args.exercise or entry['folder'] in args.exercise)]
    # 조건 `args.limit_per_exercise`을 만족하는 경우에만 아래 분기를 실행합니다.
    if args.limit_per_exercise:
        # `counts` 값에 `Counter()` 호출 결과를 저장합니다. 대상: `counts`.
        counts = Counter()
        # `selected` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `selected`.
        selected = []
        # `entries`에서 항목을 차례로 꺼내 `entry`에 넣고 아래 처리를 반복합니다.
        for entry in entries:
            # `counts[entry['folder']]` 값에 `1`을 더해 누적합니다. 대상: `counts[entry['folder']]`.
            counts[entry['folder']] += 1
            # 조건 `counts[entry['folder']] <= args.limit_per_exercise`을 만족하는 경우에만 아래 분기를 실행합니다.
            if counts[entry['folder']] <= args.limit_per_exercise:
                # `selected.append(entry)`을 호출하여 항목 하나를 끝에 추가합니다.
                selected.append(entry)
        # `entries` 값에 `selected` 값(`selected`)을 저장합니다. 대상: `entries`.
        entries = selected
    # `write_csv(output / 'inventory.csv', entries)`을 호출하여 해당 단계의 작업을 수행합니다.
    write_csv(output / 'inventory.csv', entries)
    # `sorted(set((entry['folder'] for entry in entries)))`에서 항목을 차례로 꺼내 `folder`에 넣고 아래 처리를 반복합니다.
    for folder in sorted(set(entry['folder'] for entry in entries)):
        # `items` 값에 반복·조건식 `[entry for entry in entries if entry['folder'] == folder]`으로 구성한 값을 저장합니다. 대상: `items`.
        items = [entry for entry in entries if entry['folder'] == folder]
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"{folder}: {len(items)} videos, {sum(item['frames'] for item in items)} frames, "
              # 구성 항목: `flush` 값(`flush`)에 True을 지정합니다.
              f"{sum(item['duration_sec'] for item in items) / 60:.1f} minutes", flush=True)
    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print(f'Output: {output}', flush=True)
    # 조건 `args.inventory_only`을 만족하는 경우에만 아래 분기를 실행합니다.
    if args.inventory_only:
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return
    # `('diagnostics', 'raw_keypoints', 'sessions')`에서 항목을 차례로 꺼내 `directory`에 넣고 아래 처리를 반복합니다.
    for directory in ('diagnostics', 'raw_keypoints', 'sessions'):
        # `(output / directory).mkdir(exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
        (output / directory).mkdir(exist_ok=True)
    # `asyncio.run(run(args, entries, output))`을 호출하여 해당 단계의 작업을 수행합니다.
    asyncio.run(run(args, entries, output))


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `main()`을 호출하여 해당 단계의 작업을 수행합니다.
    main()
