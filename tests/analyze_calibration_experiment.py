"""Summarize a completed/ongoing calibration experiment and plot representative traces."""

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.evaluate_calibration_videos import write_csv


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))


def examine(row):
    row = dict(row)
    row['source_group'] = Path(row['relative_path']).name.split('__')[0]
    if row['result'] == 'ERROR':
        row['candidate_reason'] = 'runtime_error'
        return row, []
    paths = list(Path(row['session_directory']).glob('session1_cali.csv'))
    frames = read_csv(paths[0]) if paths else []
    collecting = [frame for frame in frames if frame['calib_step'] == 'COLLECTING']
    before = []
    for frame in frames:
        if frame['calib_step'] in ('COLLECTING', 'COOLDOWN', 'FINISHED'):
            break
        before.append(frame)
    row['precollection_count_events'] = sum(frame.get(f'{side}_count_updated') == 'True'
                                           for frame in before for side in ('left', 'right'))
    for side in ('left', 'right'):
        row[f'{side}_collection_entry_reps'] = collecting[0][f'{side}_rep_count'] if collecting else ''
        valid = [frame for frame in frames if frame[f'{side}_value'] != '']
        row[f'{side}_valid_returning_percent'] = round(100 * sum(frame[f'{side}_state'] == 'RETURNING'
                                                               for frame in valid) / len(valid), 2) if valid else 0
    if row['result'] == 'FINISHED':
        reason = 'finished_but_precollection_counts' if row['precollection_count_events'] else 'finished'
    elif row['final_step'] == 'COOLDOWN':
        reason = 'eof_during_final_cooldown'
    elif not collecting:
        reason = 'never_reached_collection'
    elif float(row.get('occluded_percent', 0)) >= 20:
        reason = 'frequent_occlusion_or_detection_loss'
    elif max(float(row[f'{side}_valid_returning_percent']) for side in ('left', 'right')) >= 50:
        reason = 'possible_return_threshold_lock'
    else:
        reason = 'insufficient_counts_check_metric_rom_and_video'
    row['candidate_reason'] = reason
    return row, frames


def plot_trace(row, frames, path):
    if not frames:
        return
    times = np.asarray([float(frame['timestamp_ms']) / 1000 for frame in frames])
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True, constrained_layout=True)
    for side, color in (('left', 'tab:blue'), ('right', 'tab:orange')):
        values = [float(frame[f'{side}_value']) if frame[f'{side}_value'] != '' else np.nan for frame in frames]
        axes[0].plot(times, values, label=side, color=color, linewidth=1)
        axes[0].axhline(float(frames[0][f'{side}_start_val']), color=color, linestyle='--', alpha=0.6)
        axes[0].axhline(float(frames[0][f'{side}_target_val']), color=color, linestyle=':', alpha=0.6)
        axes[1].step(times, [int(frame[f'{side}_rep_count']) for frame in frames],
                     where='post', color=color, label=side)
    axes[0].set_ylabel(frames[0]['metric_type'])
    axes[1].set_ylabel('Repetitions')
    axes[1].set_xlabel('Source video time (seconds)')
    axes[1].axhline(3, color='gray', linestyle='--', linewidth=0.8)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(loc='upper left')
    figure.suptitle(f"{row['folder']} / {row['result']}\n{Path(row['relative_path']).name}", fontsize=10)
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    results = read_csv(output / 'results.csv')
    assessed, frame_sets = [], {}
    for result in results:
        row, frames = examine(result)
        assessed.append(row)
        frame_sets[row['video_id']] = frames
    write_csv(output / 'diagnostics_summary.csv', assessed)
    groups = defaultdict(list)
    for row in assessed:
        groups[row['folder']].append(row)
    inventory = read_csv(output / 'inventory.csv')
    text = ['# 캘리브레이션 실영상 실험', '',
            f'대상 {len(inventory)}개 중 {len(assessed)}개 처리. 아래 완료는 정확도가 아니라 FINISHED 도달 여부입니다.', '',
            '원본 소켓 서버 오프라인 경로와 실제 YOLO/RTMPose 모델을 사용했습니다. 프레임 샘플링 없이 소스 FPS를 사용합니다.',
            '개인 프로필과 로그는 실험 폴더 안에 격리했습니다. 실시간 전용 5초 준비 시간은 적용하지 않습니다.', '',
            '| 운동 폴더 | 처리 | 완료 | 미완료 | 오류 | 수집 시작 전 카운트 발생 영상 |',
            '|---|---:|---:|---:|---:|---:|']
    for folder, rows in sorted(groups.items()):
        counts = Counter(row['result'] for row in rows)
        early = sum(int(row.get('precollection_count_events', 0)) > 0 for row in rows)
        text.append(f"| {folder} | {len(rows)} | {counts['FINISHED']} | {counts['EOF_NOT_FINISHED']} | {counts['ERROR']} | {early} |")
    text += ['', '## 미완료 원인 후보', '',
             '아래 분류는 로그에 근거한 조사 우선순위이며 수동 영상 판독으로 확정한 원인이 아닙니다.', '']
    for reason, count in Counter(row['candidate_reason'] for row in assessed).most_common():
        text.append(f'- `{reason}`: {count}개')
    text += ['', '## 촬영 그룹별 완료', '', '| 운동 | 촬영 그룹 | 완료 / 처리 |', '|---|---|---:|']
    source_groups = defaultdict(list)
    for row in assessed:
        source_groups[(row['folder'], row['source_group'])].append(row)
    for (folder, source), rows in sorted(source_groups.items()):
        text.append(f"| {folder} | {source} | {sum(row['result'] == 'FINISHED' for row in rows)} / {len(rows)} |")
    plot_dir = output / 'plots'
    plot_dir.mkdir(exist_ok=True)
    text += ['', '## 대표 측정값과 횟수 추이', '',
             '측정값 그래프의 파선은 시작값, 점선은 목표값입니다. 검출 누락 구간은 선이 끊어집니다.', '']
    for folder, rows in sorted(groups.items()):
        for status in ('FINISHED', 'EOF_NOT_FINISHED'):
            candidates = [row for row in rows if row['result'] == status]
            if not candidates:
                continue
            example = min(candidates, key=lambda row: float(row.get('occluded_percent', 100)))
            path = plot_dir / f'{folder}_{status.lower()}.png'
            plot_trace(example, frame_sets[example['video_id']], path)
            text.append(f'![{folder} {status}](plots/{path.name})')
    text += ['', '## 판독 시 주의점', '',
             '- 기존 서버의 COMPLETED와 프로필 저장 여부는 실제 캘리브레이션 완료를 보장하지 않습니다. FINISHED를 별도로 확인했습니다.',
             '- 기존 코드가 전신 확인 후 3초 COUNTDOWN 중에도 FSM을 갱신하므로, 수집 시작 전 반복이 포함되는지 별도 집계했습니다.',
             '- leg_extension은 knee_extension, leg_raise는 hip_knee_flexion에 연결했습니다. 후자는 서로 다른 자세 변형이 섞여 있습니다.',
             '- 실제 횟수와 대상자 정답 라벨이 없어 과대·과소 카운트 정확도는 산출하지 않았습니다.',
             '- 결과는 현재 2D 측정값과 기본 ROM에 대한 실험이며 카메라 방향·자세·환자별 조건에 영향을 받습니다.', '',
             '`results.csv`: 영상별 기본 결과. `diagnostics_summary.csv`: 수집 전 카운트 등 추가 진단.',
             '`raw_keypoints/`: 전체 입력 영상의 추론 좌표 캐시. `sessions/`: 프레임 로그와 격리된 프로필.',
             '`exercise_config_snapshot.json`, `environment.json`: 설정 및 실행 환경.']
    (output / 'analysis_ko.md').write_text('\n'.join(text) + '\n', encoding='utf-8')
    print(f'Analyzed {len(assessed)}/{len(inventory)} videos: {output / "analysis_ko.md"}')


if __name__ == '__main__':
    main()
