# 한국어 코드 해설: `analyze_calibration_experiment`에서 사용하는 실행·분석 보조 기능을 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Summarize a completed/ongoing calibration experiment and plot representative traces."""

# 이 파일에서 사용할 `argparse` 모듈을 불러옵니다.
import argparse
# `collections`에서 `Counter`, `defaultdict` 기능을 가져옵니다.
from collections import Counter, defaultdict
# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys

# 이 파일에서 사용할 `matplotlib` 모듈을 불러옵니다.
import matplotlib
# `matplotlib.use('Agg')`을 호출하여 해당 단계의 작업을 수행합니다.
matplotlib.use('Agg')
# 이 파일에서 사용할 `matplotlib.pyplot`(`plt`) 모듈을 불러옵니다.
import matplotlib.pyplot as plt
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# 최상위 경로에 계산식 `Path(__file__).resolve().parents[1]`의 결과를 저장합니다. 대상: `ROOT`.
ROOT = Path(__file__).resolve().parents[1]
# `sys.path.insert(0, str(ROOT))`을 호출하여 해당 단계의 작업을 수행합니다.
sys.path.insert(0, str(ROOT))
# `tests.evaluate_calibration_videos`에서 `write_csv` 기능을 가져옵니다.
from tests.evaluate_calibration_videos import write_csv


# 읽기 csv 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def read_csv(path):
    # `path.open(encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with path.open(encoding='utf-8-sig', newline='') as file:
        # 현재 함수 처리를 끝내고 `list(csv.DictReader(file))` 호출 결과를 호출한 쪽으로 반환합니다.
        return list(csv.DictReader(file))


# `examine` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def examine(row):
    # `row` 값에 `dict(row)` 호출 결과를 저장합니다. 대상: `row`.
    row = dict(row)
    # `row['source_group']` 값에 계산식 `Path(row['relative_path']).name.split('__')[0]`의 결과를 저장합니다. 대상: `row['source_group']`.
    row['source_group'] = Path(row['relative_path']).name.split('__')[0]
    # 조건 `row['result'] == 'ERROR'`을 만족하는 경우에만 아래 분기를 실행합니다.
    if row['result'] == 'ERROR':
        # `row['candidate_reason']` 값에 'runtime_error'을 저장합니다. 대상: `row['candidate_reason']`.
        row['candidate_reason'] = 'runtime_error'
        # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(row, [])`을 호출한 쪽으로 반환합니다.
        return row, []
    # `paths` 값에 `list(Path(row['session_directory']).glob('session1_cali.csv'))` 호출 결과를 저장합니다. 대상: `paths`.
    # 새 카메라 파일명과 기존 실험 결과 모두 읽되 반복별 요약 CSV는 제외합니다.
    directory = Path(row['session_directory'])
    paths = list(directory.glob('*_Calibration_Camera_*.csv')) or list(directory.glob('session1_cali.csv'))
    paths = [path for path in paths if not path.stem.endswith('_reps')]
    # 영상 프레임 묶음에 `paths`이면 `read_csv(paths[0])`, 아니면 `[]`을 저장합니다. 대상: `frames`.
    frames = read_csv(paths[0]) if paths else []
    # `collecting` 값에 반복·조건식 `[frame for frame in frames if frame['calib_step'] == 'COLLECTING']`으로 구성한 값을 저장합니다. 대상: `collecting`.
    collecting = [frame for frame in frames if frame['calib_step'] == 'COLLECTING']
    # 이전에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `before`.
    before = []
    # `frames`에서 항목을 차례로 꺼내 `frame`에 넣고 아래 처리를 반복합니다.
    for frame in frames:
        # 조건 `frame['calib_step'] in ('COLLECTING', 'COOLDOWN', 'FINISHED')`을 만족하는 경우에만 아래 분기를 실행합니다.
        if frame['calib_step'] in ('COLLECTING', 'COOLDOWN', 'FINISHED'):
            # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
            break
        # `before.append(frame)`을 호출하여 항목 하나를 끝에 추가합니다.
        before.append(frame)
    # row['precollection 횟수 events']에 `sum((frame.get(f'{side}_count_updated') == 'True' for frame in before for side in ('left', 'right')))` 호출 결과를 저장합니다. 대상: `row['precollection_count_events']`.
    row['precollection_count_events'] = sum(frame.get(f'{side}_count_updated') == 'True'
                                           # 앞줄에서 시작한 식·인수·자료 구조를 `for frame in before for side in ('left', 'right'))` 내용으로 이어서 구성합니다.
                                           for frame in before for side in ('left', 'right'))
    # `('left', 'right')`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
    for side in ('left', 'right'):
        # `row[f'{side}_collection_entry_reps']` 값에 `collecting`이면 `collecting[0][f'{side}_rep_count']`, 아니면 `''`을 저장합니다. 대상: `row[f'{side}_collection_entry_reps']`.
        row[f'{side}_collection_entry_reps'] = collecting[0][f'{side}_rep_count'] if collecting else ''
        # `valid` 값에 반복·조건식 `[frame for frame in frames if frame[f'{side}_value'] != '']`으로 구성한 값을 저장합니다. 대상: `valid`.
        valid = [frame for frame in frames if frame[f'{side}_value'] != '']
        # `row[f'{side}_valid_returning_percent']` 값에 `valid`이면 `round(100 * sum((frame[f'{side}_state'] == 'RETURNING' for frame in valid)) / len(valid), 2)`, 아니면 `0`을 저장합니다. 대상: `row[f'{side}_valid_returning_percent']`.
        row[f'{side}_valid_returning_percent'] = round(100 * sum(frame[f'{side}_state'] == 'RETURNING'
                                                               # 앞줄에서 시작한 식·인수·자료 구조를 `for frame in valid) / len(valid), 2) if valid else 0` 내용으로 이어서 구성합니다.
                                                               for frame in valid) / len(valid), 2) if valid else 0
    # 조건 `row['result'] == 'FINISHED'`을 만족하는 경우에만 아래 분기를 실행합니다.
    if row['result'] == 'FINISHED':
        # `reason` 값에 `row['precollection_count_events']`이면 `'finished_but_precollection_counts'`, 아니면 `'finished'`을 저장합니다. 대상: `reason`.
        reason = 'finished_but_precollection_counts' if row['precollection_count_events'] else 'finished'
    # 조건 `row['final_step'] == 'COOLDOWN'`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif row['final_step'] == 'COOLDOWN':
        # `reason` 값에 'eof_during_final_cooldown'을 저장합니다. 대상: `reason`.
        reason = 'eof_during_final_cooldown'
    # 조건 `not collecting`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif not collecting:
        # `reason` 값에 'never_reached_collection'을 저장합니다. 대상: `reason`.
        reason = 'never_reached_collection'
    # 조건 `float(row.get('occluded_percent', 0)) >= 20`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif float(row.get('occluded_percent', 0)) >= 20:
        # `reason` 값에 'frequent_occlusion_or_detection_loss'을 저장합니다. 대상: `reason`.
        reason = 'frequent_occlusion_or_detection_loss'
    # 조건 `max((float(row[f'{side}_valid_returning_percent']) for side in ('left', 'right'))) >= 50`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif max(float(row[f'{side}_valid_returning_percent']) for side in ('left', 'right')) >= 50:
        # `reason` 값에 'possible_return_threshold_lock'을 저장합니다. 대상: `reason`.
        reason = 'possible_return_threshold_lock'
    # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
    else:
        # `reason` 값에 'insufficient_counts_check_metric_rom_and_video'을 저장합니다. 대상: `reason`.
        reason = 'insufficient_counts_check_metric_rom_and_video'
    # `row['candidate_reason']` 값에 `reason` 값(`reason`)을 저장합니다. 대상: `row['candidate_reason']`.
    row['candidate_reason'] = reason
    # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(row, frames)`을 호출한 쪽으로 반환합니다.
    return row, frames


# `plot_trace` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def plot_trace(row, frames, path):
    # 조건 `not frames`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not frames:
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return
    # `times` 값에 `np.asarray([float(frame['timestamp_ms']) / 1000 for frame in frames])` 호출 결과를 저장합니다. 대상: `times`.
    times = np.asarray([float(frame['timestamp_ms']) / 1000 for frame in frames])
    # `(figure, axes)` 값에 `plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)` 호출 결과를 저장합니다. 대상: `(figure, axes)`.
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    # `(('left', 'tab:blue'), ('right', 'tab:orange'))`에서 항목을 차례로 꺼내 `(side, color)`에 넣고 아래 처리를 반복합니다.
    for side, color in (('left', 'tab:blue'), ('right', 'tab:orange')):
        # 측정값에 반복·조건식 `[float(frame[f'{side}_value']) if frame[f'{side}_value'] != '' else np.nan for frame in frames]`으로 구성한 값을 저장합니다. 대상: `values`.
        values = [float(frame[f'{side}_value']) if frame[f'{side}_value'] != '' else np.nan for frame in frames]
        # `axes[0].plot(times, values, label=side, color=color, linewidth=1)`을 호출하여 해당 단계의 작업을 수행합니다.
        axes[0].plot(times, values, label=side, color=color, linewidth=1)
        # `axes[0].axhline(float(frames[0][f'{side}_start_val']), color=color, linestyle='--', alpha=0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
        axes[0].axhline(float(frames[0][f'{side}_start_val']), color=color, linestyle='--', alpha=0.6)
        # `axes[0].axhline(float(frames[0][f'{side}_target_val']), color=color, linestyle=':', alpha=0.6)`을 호출하여 해당 단계의 작업을 수행합니다.
        axes[0].axhline(float(frames[0][f'{side}_target_val']), color=color, linestyle=':', alpha=0.6)
        # `axes[1].step(times, [int(frame[f'{side}_rep_count']) for frame in frames], where='post', color=color, label=side)`을 호출하여 해당 단계의 작업을 수행합니다.
        axes[1].step(times, [int(frame[f'{side}_rep_count']) for frame in frames],
                     # 구성 항목: `where` 값(`where`)에 'post'; `color` 값(`color`)에 `color` 값(`color`); `label` 값(`label`)에 좌우 구분(`side`)을 지정합니다.
                     where='post', color=color, label=side)
    # `axes[0].set_ylabel(frames[0]['metric_type'])`을 호출하여 해당 단계의 작업을 수행합니다.
    axes[0].set_ylabel(frames[0]['metric_type'])
    # `axes[1].set_ylabel('Repetitions')`을 호출하여 해당 단계의 작업을 수행합니다.
    axes[1].set_ylabel('Repetitions')
    # `axes[1].set_xlabel('Source video time (seconds)')`을 호출하여 해당 단계의 작업을 수행합니다.
    axes[1].set_xlabel('Source video time (seconds)')
    # `axes[1].axhline(3, color='gray', linestyle='--', linewidth=0.8)`을 호출하여 해당 단계의 작업을 수행합니다.
    axes[1].axhline(3, color='gray', linestyle='--', linewidth=0.8)
    # `axes`에서 항목을 차례로 꺼내 `axis`에 넣고 아래 처리를 반복합니다.
    for axis in axes:
        # `axis.grid(alpha=0.2)`을 호출하여 해당 단계의 작업을 수행합니다.
        axis.grid(alpha=0.2)
        # `axis.legend(loc='upper left')`을 호출하여 해당 단계의 작업을 수행합니다.
        axis.legend(loc='upper left')
    # `figure.suptitle(f"{row['folder']} / {row['result']}\n{Path(row['relative_path']).name}", fontsize=10)`을 호출하여 해당 단계의 작업을 수행합니다.
    figure.suptitle(f"{row['folder']} / {row['result']}\n{Path(row['relative_path']).name}", fontsize=10)
    # `figure.savefig(path, dpi=140)`을 호출하여 해당 단계의 작업을 수행합니다.
    figure.savefig(path, dpi=140)
    # `plt.close(figure)`을 호출하여 해당 단계의 작업을 수행합니다.
    plt.close(figure)


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def main():
    # `parser` 값에 `argparse.ArgumentParser(description=__doc__)` 호출 결과를 저장합니다. 대상: `parser`.
    parser = argparse.ArgumentParser(description=__doc__)
    # `parser.add_argument('output', type=Path)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('output', type=Path)
    # `args` 값에 `parser.parse_args()` 호출 결과를 저장합니다. 대상: `args`.
    args = parser.parse_args()
    # 생성할 결과 데이터 또는 경로에 `args.output.resolve()` 호출 결과를 저장합니다. 대상: `output`.
    output = args.output.resolve()
    # `results` 값에 `read_csv(output / 'results.csv')` 호출 결과를 저장합니다. 대상: `results`.
    results = read_csv(output / 'results.csv')
    # `(assessed, frame_sets)` 값에 순서·구성을 지정한 `([], {})`을 저장합니다. 대상: `(assessed, frame_sets)`.
    assessed, frame_sets = [], {}
    # `results`에서 항목을 차례로 꺼내 `result`에 넣고 아래 처리를 반복합니다.
    for result in results:
        # `(row, frames)` 값에 `examine(result)` 호출 결과를 저장합니다. 대상: `(row, frames)`.
        row, frames = examine(result)
        # `assessed.append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
        assessed.append(row)
        # 프레임 sets[row['video id']]에 영상 프레임 묶음(`frames`)을 저장합니다. 대상: `frame_sets[row['video_id']]`.
        frame_sets[row['video_id']] = frames
    # `write_csv(output / 'diagnostics_summary.csv', assessed)`을 호출하여 해당 단계의 작업을 수행합니다.
    write_csv(output / 'diagnostics_summary.csv', assessed)
    # `groups` 값에 `defaultdict(list)` 호출 결과를 저장합니다. 대상: `groups`.
    groups = defaultdict(list)
    # `assessed`에서 항목을 차례로 꺼내 `row`에 넣고 아래 처리를 반복합니다.
    for row in assessed:
        # `groups[row['folder']].append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
        groups[row['folder']].append(row)
    # `inventory` 값에 `read_csv(output / 'inventory.csv')` 호출 결과를 저장합니다. 대상: `inventory`.
    inventory = read_csv(output / 'inventory.csv')
    # `text` 값에 순서·구성을 지정한 `['# 캘리브레이션 실영상 실험', '', f'대상 {len(inventory)}개 중 {len(assessed)}개 처리. 아래 완료는 정확도가 아니라 FINISHED 도달 여부입니다.', '', '원본 소켓 서버 오프라인 경로와 실제 YOLO/RTMPose 모델을 사용했습니다. 프레임 샘플링 없이 …`을 저장합니다. 대상: `text`.
    text = ['# 캘리브레이션 실영상 실험', '',
            # 앞줄에서 시작한 식·인수·자료 구조를 `f'대상 {len(inventory)}개 중 {len(assessed)}개 처리. 아래 완료는 정확도가 아니라 FINISHED 도달 여부입니다.', '',` 내용으로 이어서 구성합니다.
            f'대상 {len(inventory)}개 중 {len(assessed)}개 처리. 아래 완료는 정확도가 아니라 FINISHED 도달 여부입니다.', '',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'원본 소켓 서버 오프라인 경로와 실제 YOLO/RTMPose 모델을 사용했습니다. 프레임 샘플링 없이 소스 FPS를 사용합니다.',` 내용으로 이어서 구성합니다.
            '원본 소켓 서버 오프라인 경로와 실제 YOLO/RTMPose 모델을 사용했습니다. 프레임 샘플링 없이 소스 FPS를 사용합니다.',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'개인 프로필과 로그는 실험 폴더 안에 격리했습니다. 실시간 전용 5초 준비 시간은 적용하지 않습니다.', '',` 내용으로 이어서 구성합니다.
            '개인 프로필과 로그는 실험 폴더 안에 격리했습니다. 실시간 전용 5초 준비 시간은 적용하지 않습니다.', '',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'| 운동 폴더 | 처리 | 완료 | 미완료 | 오류 | 수집 시작 전 카운트 발생 영상 |',` 내용으로 이어서 구성합니다.
            '| 운동 폴더 | 처리 | 완료 | 미완료 | 오류 | 수집 시작 전 카운트 발생 영상 |',
            # 앞줄에서 시작한 식·인수·자료 구조를 `'|---|---:|---:|---:|---:|---:|']` 내용으로 이어서 구성합니다.
            '|---|---:|---:|---:|---:|---:|']
    # `sorted(groups.items())`에서 항목을 차례로 꺼내 `(folder, rows)`에 넣고 아래 처리를 반복합니다.
    for folder, rows in sorted(groups.items()):
        # `counts` 값에 `Counter((row['result'] for row in rows))` 호출 결과를 저장합니다. 대상: `counts`.
        counts = Counter(row['result'] for row in rows)
        # `early` 값에 `sum((int(row.get('precollection_count_events', 0)) > 0 for row in rows))` 호출 결과를 저장합니다. 대상: `early`.
        early = sum(int(row.get('precollection_count_events', 0)) > 0 for row in rows)
        # `text.append(f"| {folder} | {len(rows)} | {counts['FINISHED']} | {counts['EOF_NOT_FINISHED']} | {counts['ERROR']} | {early} |")`을 호출하여 항목 하나를 끝에 추가합니다.
        text.append(f"| {folder} | {len(rows)} | {counts['FINISHED']} | {counts['EOF_NOT_FINISHED']} | {counts['ERROR']} | {early} |")
    # `text` 값에 `['', '## 미완료 원인 후보', '', '아래 분류는 로그에 근거한 조사 우선순위이며 수동 영상 판독으로 확정한 원인이 아닙니다.', '']`을 더해 누적합니다. 대상: `text`.
    text += ['', '## 미완료 원인 후보', '',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'아래 분류는 로그에 근거한 조사 우선순위이며 수동 영상 판독으로 확정한 원인이 아닙니다.', '']` 내용으로 이어서 구성합니다.
             '아래 분류는 로그에 근거한 조사 우선순위이며 수동 영상 판독으로 확정한 원인이 아닙니다.', '']
    # `Counter((row['candidate_reason'] for row in assessed)).most_common()`에서 항목을 차례로 꺼내 `(reason, count)`에 넣고 아래 처리를 반복합니다.
    for reason, count in Counter(row['candidate_reason'] for row in assessed).most_common():
        # `text.append(f'- `{reason}`: {count}개')`을 호출하여 항목 하나를 끝에 추가합니다.
        text.append(f'- `{reason}`: {count}개')
    # `text` 값에 `['', '## 촬영 그룹별 완료', '', '| 운동 | 촬영 그룹 | 완료 / 처리 |', '|---|---|---:|']`을 더해 누적합니다. 대상: `text`.
    text += ['', '## 촬영 그룹별 완료', '', '| 운동 | 촬영 그룹 | 완료 / 처리 |', '|---|---|---:|']
    # 원본 groups에 `defaultdict(list)` 호출 결과를 저장합니다. 대상: `source_groups`.
    source_groups = defaultdict(list)
    # `assessed`에서 항목을 차례로 꺼내 `row`에 넣고 아래 처리를 반복합니다.
    for row in assessed:
        # `source_groups[row['folder'], row['source_group']].append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
        source_groups[(row['folder'], row['source_group'])].append(row)
    # `sorted(source_groups.items())`에서 항목을 차례로 꺼내 `((folder, source), rows)`에 넣고 아래 처리를 반복합니다.
    for (folder, source), rows in sorted(source_groups.items()):
        # `text.append(f"| {folder} | {source} | {sum((row['result'] == 'FINISHED' for row in rows))} / {len(rows)} |")`을 호출하여 항목 하나를 끝에 추가합니다.
        text.append(f"| {folder} | {source} | {sum(row['result'] == 'FINISHED' for row in rows)} / {len(rows)} |")
    # `plot_dir` 값에 계산식 `output / 'plots'`의 결과를 저장합니다. 대상: `plot_dir`.
    plot_dir = output / 'plots'
    # `plot_dir.mkdir(exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
    plot_dir.mkdir(exist_ok=True)
    # `text` 값에 `['', '## 대표 측정값과 횟수 추이', '', '측정값 그래프의 파선은 시작값, 점선은 목표값입니다. 검출 누락 구간은 선이 끊어집니다.', '']`을 더해 누적합니다. 대상: `text`.
    text += ['', '## 대표 측정값과 횟수 추이', '',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'측정값 그래프의 파선은 시작값, 점선은 목표값입니다. 검출 누락 구간은 선이 끊어집니다.', '']` 내용으로 이어서 구성합니다.
             '측정값 그래프의 파선은 시작값, 점선은 목표값입니다. 검출 누락 구간은 선이 끊어집니다.', '']
    # `sorted(groups.items())`에서 항목을 차례로 꺼내 `(folder, rows)`에 넣고 아래 처리를 반복합니다.
    for folder, rows in sorted(groups.items()):
        # `('FINISHED', 'EOF_NOT_FINISHED')`에서 항목을 차례로 꺼내 `status`에 넣고 아래 처리를 반복합니다.
        for status in ('FINISHED', 'EOF_NOT_FINISHED'):
            # `candidates` 값에 반복·조건식 `[row for row in rows if row['result'] == status]`으로 구성한 값을 저장합니다. 대상: `candidates`.
            candidates = [row for row in rows if row['result'] == status]
            # 조건 `not candidates`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not candidates:
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # `example` 값에 `min(candidates, key=lambda row: float(row.get('occluded_percent', 100)))` 호출 결과를 저장합니다. 대상: `example`.
            example = min(candidates, key=lambda row: float(row.get('occluded_percent', 100)))
            # 경로에 계산식 `plot_dir / f'{folder}_{status.lower()}.png'`의 결과를 저장합니다. 대상: `path`.
            path = plot_dir / f'{folder}_{status.lower()}.png'
            # `plot_trace(example, frame_sets[example['video_id']], path)`을 호출하여 해당 단계의 작업을 수행합니다.
            plot_trace(example, frame_sets[example['video_id']], path)
            # `text.append(f'![{folder} {status}](plots/{path.name})')`을 호출하여 항목 하나를 끝에 추가합니다.
            text.append(f'![{folder} {status}](plots/{path.name})')
    # `text` 값에 `['', '## 판독 시 주의점', '', '- 기존 서버의 COMPLETED와 프로필 저장 여부는 실제 캘리브레이션 완료를 보장하지 않습니다. FINISHED를 별도로 확인했습니다.', '- 기존 코드가 전신 확인 후 3초 COUNTDOWN 중에도 FSM을 갱신하므로, 수집 시작 전 반복이 포함되는지…`을 더해 누적합니다. 대상: `text`.
    text += ['', '## 판독 시 주의점', '',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- 기존 서버의 COMPLETED와 프로필 저장 여부는 실제 캘리브레이션 완료를 보장하지 않습니다. FINISHED를 별도로 확인했습니다.',` 내용으로 이어서 구성합니다.
             '- 기존 서버의 COMPLETED와 프로필 저장 여부는 실제 캘리브레이션 완료를 보장하지 않습니다. FINISHED를 별도로 확인했습니다.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- 기존 코드가 전신 확인 후 3초 COUNTDOWN 중에도 FSM을 갱신하므로, 수집 시작 전 반복이 포함되는지 별도 집계했습니다.',` 내용으로 이어서 구성합니다.
             '- 기존 코드가 전신 확인 후 3초 COUNTDOWN 중에도 FSM을 갱신하므로, 수집 시작 전 반복이 포함되는지 별도 집계했습니다.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- leg_extension은 knee_extension, leg_raise는 hip_knee_flexion에 연결했습니다. 후자는 서로 다른 자세 변형이 섞여 있습니다.',` 내용으로 이어서 구성합니다.
             '- leg_extension은 knee_extension, leg_raise는 hip_knee_flexion에 연결했습니다. 후자는 서로 다른 자세 변형이 섞여 있습니다.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- 실제 횟수와 대상자 정답 라벨이 없어 과대·과소 카운트 정확도는 산출하지 않았습니다.',` 내용으로 이어서 구성합니다.
             '- 실제 횟수와 대상자 정답 라벨이 없어 과대·과소 카운트 정확도는 산출하지 않았습니다.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'- 결과는 현재 2D 측정값과 기본 ROM에 대한 실험이며 카메라 방향·자세·환자별 조건에 영향을 받습니다.', '',` 내용으로 이어서 구성합니다.
             '- 결과는 현재 2D 측정값과 기본 ROM에 대한 실험이며 카메라 방향·자세·환자별 조건에 영향을 받습니다.', '',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'`results.csv`: 영상별 기본 결과. `diagnostics_summary.csv`: 수집 전 카운트 등 추가 진단.',` 내용으로 이어서 구성합니다.
             '`results.csv`: 영상별 기본 결과. `diagnostics_summary.csv`: 수집 전 카운트 등 추가 진단.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'`raw_keypoints/`: 전체 입력 영상의 추론 좌표 캐시. `sessions/`: 프레임 로그와 격리된 프로필.',` 내용으로 이어서 구성합니다.
             '`raw_keypoints/`: 전체 입력 영상의 추론 좌표 캐시. `sessions/`: 프레임 로그와 격리된 프로필.',
             # 앞줄에서 시작한 식·인수·자료 구조를 `'`exercise_config_snapshot.json`, `environment.json`: 설정 및 실행 환경.']` 내용으로 이어서 구성합니다.
             '`exercise_config_snapshot.json`, `environment.json`: 설정 및 실행 환경.']
    # `(output / 'analysis_ko.md').write_text('\n'.join(text) + '\n', encoding='utf-8')`을 호출하여 해당 단계의 작업을 수행합니다.
    (output / 'analysis_ko.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print(f'Analyzed {len(assessed)}/{len(inventory)} videos: {output / "analysis_ko.md"}')


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `main()`을 호출하여 해당 단계의 작업을 수행합니다.
    main()
