import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


def load_patient_data(data_dir: Path, player_id: str):
  """JSON, CSV, NPZ 파일 일괄 로드"""
  patient_dir = data_dir / player_id
  json_path = patient_dir / f'{player_id}.json'
  csv_path = patient_dir / 'rep_details.csv'
  skeleton_dir = patient_dir / 'skeleton memory'

  # 1. JSON 로드
  json_data = {}
  if json_path.exists():
    with open(json_path, 'r', encoding='utf-8') as f:
      json_data = json.load(f)

  # 2. CSV 로드
  df_reps = pd.DataFrame()
  if csv_path.exists():
    df_reps = pd.read_csv(csv_path)

  # 3. 최신 NPZ 파일 로드
  npz_data = None
  npz_files = list(skeleton_dir.glob('*.npz'))
  if npz_files:
    latest_npz = max(npz_files, key=lambda x: x.stat().st_mtime)
    npz_data = np.load(latest_npz)
    print(f'불러온 NPZ 파일: {latest_npz.name}')

  return json_data, df_reps, npz_data, patient_dir


def analyze_json_calibration(json_data: dict):
  """1. JSON 맞춤 Threshold 분석"""
  print('=== 1. 환자 기본 정보 및 맞춤 Threshold ===')
  print(f"환자 ID: {json_data.get('patient_id')}")
  print(f"환자 성함: {json_data.get('patient_name')}")

  custom_th = json_data.get('custom_thresholds', {})
  for exercise, th in custom_th.items():
    print(f'\n[운동 종목: {exercise}]')
    print(f"  - 왼쪽: 시작 {th['left']['start_val']}도 / 목표 {th['left']['target_val']}도")
    print(f"  - 오른쪽: 시작 {th['right']['start_val']}도 / 목표 {th['right']['target_val']}도")


def analyze_csv_performance(df_reps: pd.DataFrame, save_dir: Path):
  """2. CSV 회차별 성과 요약 및 PNG 저장"""
  if df_reps.empty:
    print('\nCSV 데이터가 없습니다.')
    return

  print('\n=== 2. 회차별 요약 통계 ===')
  summary = (
      df_reps.groupby(['side', 'quality'])
      .agg(
          count=('rep_num', 'count'),
          avg_duration=('duration_sec', 'mean'),
          avg_rom=('achieved_rom', 'mean'),
      )
      .reset_index()
  )
  print(summary)

  # 시각화: 좌/우 및 회차별 achieved_rom 변화
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))

  sns.barplot(
      data=df_reps,
      x='rep_num',
      y='achieved_rom',
      hue='side',
      ax=axes[0],
      palette='Set2',
  )
  axes[0].set_title('회차별 가동 범위 (ROM)')
  axes[0].set_xlabel('반복 회차')
  axes[0].set_ylabel('가동 범위 (도)')

  sns.countplot(
      data=df_reps, x='quality', hue='side', ax=axes[1], palette='Pastel1'
  )
  axes[1].set_title('동작 품질 평가 분포')
  axes[1].set_xlabel('품질 등급')
  axes[1].set_ylabel('횟수')

  plt.tight_layout()

  # PNG 저장
  save_path = save_dir / 'eda_rep_performance.png'
  fig.savefig(save_path, dpi=300, bbox_inches='tight')
  print(f'그래프 저장 완료: {save_path}')
  plt.close(fig)


def analyze_npz_trajectories(npz_data, save_dir: Path):
  """3. NPZ 실시간 시계열 관절 궤적 및 속도 분석 (PNG 저장)"""
  if npz_data is None:
    print('\nNPZ 데이터가 없습니다.')
    return

  timestamps = npz_data['timestamps']  # (N,)
  values = npz_data['values']  # (N, 2)
  keypoints = npz_data['keypoints']  # (N, 17, 3)

  print('\n=== 3. NPZ 원천 시계열 데이터 분석 ===')
  print(f'총 프레임 수: {len(timestamps)}')
  print(f'측정 시간: {timestamps[-1]:.2f} 초')

  # 각속도 산출 (dt 미분)
  dt = np.diff(timestamps, prepend=timestamps[0])
  dt[dt == 0] = 1e-5

  vel_left = np.gradient(values[:, 0], timestamps)
  vel_right = np.gradient(values[:, 1], timestamps)

  # 시각화 1: 시계열 각도 및 각속도
  fig1, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

  axes[0].plot(
      timestamps, values[:, 0], label='Left Side', color='blue', alpha=0.8
  )
  axes[0].plot(
      timestamps, values[:, 1], label='Right Side', color='red', alpha=0.8
  )
  axes[0].set_ylabel('관절 각도 (도)')
  axes[0].set_title('실시간 관절 각도 변화')
  axes[0].legend()
  axes[0].grid(True, linestyle='--', alpha=0.5)

  axes[1].plot(
      timestamps,
      vel_left,
      label='Left Velocity',
      color='blue',
      linestyle=':',
      alpha=0.7,
  )
  axes[1].plot(
      timestamps,
      vel_right,
      label='Right Velocity',
      color='red',
      linestyle=':',
      alpha=0.7,
  )
  axes[1].set_xlabel('시간 (초)')
  axes[1].set_ylabel('각속도 (도/초)')
  axes[1].set_title('관절 각속도 (동작 부드러움 및 떨림 측정)')
  axes[1].legend()
  axes[1].grid(True, linestyle='--', alpha=0.5)

  plt.tight_layout()
  save_path1 = save_dir / 'eda_joint_angles_velocity.png'
  fig1.savefig(save_path1, dpi=300, bbox_inches='tight')
  print(f'그래프 저장 완료: {save_path1}')
  plt.close(fig1)

  # 시각화 2: 특정 관절 2D 궤적 (손목 ID 9, 10)
  fig2 = plt.figure(figsize=(6, 6))
  plt.plot(
      keypoints[:, 9, 0],
      keypoints[:, 9, 1],
      label='Left Wrist Trajectory',
      color='blue',
      alpha=0.6,
  )
  plt.plot(
      keypoints[:, 10, 0],
      keypoints[:, 10, 1],
      label='Right Wrist Trajectory',
      color='red',
      alpha=0.6,
  )
  plt.scatter(0, 0, color='black', marker='x', s=100, label='Origin (Pelvis)')
  plt.gca().invert_yaxis()
  plt.title('골반 중심 기준 손목 관절 2D 이동 궤적')
  plt.xlabel('정규화 X')
  plt.ylabel('정규화 Y')
  plt.legend()
  plt.grid(True, linestyle='--', alpha=0.5)

  save_path2 = save_dir / 'eda_2d_wrist_trajectory.png'
  fig2.savefig(save_path2, dpi=300, bbox_inches='tight')
  print(f'그래프 저장 완료: {save_path2}')
  plt.close(fig2)


# 메인 실행부
if __name__ == '__main__':
  PROJECT_ROOT = Path(__file__).resolve().parent
  DATA_DIR = PROJECT_ROOT / 'data'
  TARGET_PATIENT = 'patient_1'

  json_data, df_reps, npz_data, save_dir = load_patient_data(
      DATA_DIR, TARGET_PATIENT
  )

  print(save_dir)

  if json_data:
    analyze_json_calibration(json_data)
  if not df_reps.empty:
    analyze_csv_performance(df_reps, save_dir)
  if npz_data is not None:
    analyze_npz_trajectories(npz_data, save_dir)