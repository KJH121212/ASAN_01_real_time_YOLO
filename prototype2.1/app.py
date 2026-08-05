import json
import os
import subprocess
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title="AI 운동 재활 시스템 제어 센터", layout="wide")

st.title("AI 운동 재활 시스템 제어 센터")
st.markdown("---")

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent

CONFIG_PATH = CURRENT_DIR / "configs" / "exercise_kpt_config_coco.json"
if not CONFIG_PATH.exists():
  CONFIG_PATH = ROOT_DIR / "configs" / "exercise_kpt_config_coco.json"

EXERCISE_LIST = []
if CONFIG_PATH.exists():
  with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = json.load(f)
    EXERCISE_LIST = list(cfg.get("exercises", {}).keys())

col1, col2 = st.columns([1, 1])

with col1:
  st.header("측정 세션 설정")

  col_p1, col_p2 = st.columns(2)
  player_id = col_p1.text_input("환자 ID (파일명용)", value="patient_1")
  patient_name = col_p2.text_input("환자 성함 (한글)", value="김지후")

  exercise_name = st.selectbox(
      f"운동 종목 선택 (총 {len(EXERCISE_LIST)}종)",
      EXERCISE_LIST if EXERCISE_LIST else ["biceps_curl"],
  )
  camera_index = st.selectbox(
      "카메라 장치 선택",
      [0, 1, 2],
      format_func=lambda x: f"카메라 {x}번 (기본웹캠: 0)",
  )
  target_reps = st.number_input(
      "실제 운동 목표 횟수", min_value=1, max_value=50, value=10
  )

  st.markdown("---")

  col_btn1, col_btn2 = st.columns(2)

  if col_btn1.button(
      "시범 동작 측정 (3회)", type="primary", use_container_width=True
  ):
    cmd = [
        sys.executable,
        str(CURRENT_DIR / "test_client.py"),
        "--player_id",
        player_id,
        "--patient_name",
        patient_name,
        "--exercise_name",
        exercise_name,
        "--mode",
        "CALIBRATION",
        "--target_reps",
        "3",
        "--camera_index",
        str(camera_index),
    ]
    subprocess.Popen(cmd)
    st.info(f"시범 동작 창이 열렸습니다. (카메라 {camera_index}번)")

  if col_btn2.button("실제 운동 측정 시작", use_container_width=True):
    cmd = [
        sys.executable,
        str(CURRENT_DIR / "test_client.py"),
        "--player_id",
        player_id,
        "--patient_name",
        patient_name,
        "--exercise_name",
        exercise_name,
        "--mode",
        "MAIN",
        "--target_reps",
        str(target_reps),
        "--camera_index",
        str(camera_index),
    ]
    subprocess.Popen(cmd)
    st.success(
        f"실제 운동 창이 열렸습니다. {target_reps}회 측정을 시작합니다."
    )

with col2:
  st.header("환자 개인 맞춤 데이터 확인")
  patient_dir = ROOT_DIR / "data" / player_id
  patient_json = patient_dir / f"{player_id}.json"
  patient_csv = patient_dir / "rep_details.csv"

  if patient_json.exists():
    with open(patient_json, "r", encoding="utf-8") as f:
      p_data = json.load(f)
    st.subheader(
        f"환자 정보: {p_data.get('patient_name', '미지정')} ({player_id})"
    )

    st.markdown(f"**현재 선택 운동 ({exercise_name}) 맞춤 Threshold**")
    custom_th = p_data.get("custom_thresholds", {}).get(exercise_name)
    if custom_th:
      st.dataframe(pd.DataFrame(custom_th).T, use_container_width=True)
    else:
      st.info("해당 운동의 설정된 임계값이 없습니다. (시범 동작 필요)")

    st.markdown("**과거 시범 동작 측정 히스토리**")
    calib_history = p_data.get("calibration_history", [])
    if calib_history:
      st.dataframe(pd.DataFrame(calib_history), use_container_width=True)
    else:
      st.write("기록된 시범 동작 히스토리가 없습니다.")
  else:
    st.subheader(f"환자 정보: {patient_name} ({player_id})")
    st.warning("기록된 환자 프로필 파일이 없습니다.")

  st.markdown("---")

  st.markdown("**회차별 상세 운동 기록 (rep_details.csv)**")
  if patient_csv.exists():
    try:
      df = pd.read_csv(patient_csv)
      if not df.empty:
        st.dataframe(df, use_container_width=True)
      else:
        st.write("기록된 상세 운동 데이터가 없습니다.")
    except Exception as e:
      st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
  else:
    st.warning("운동 기록 CSV 파일이 없습니다.")