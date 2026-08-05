import streamlit as st
import subprocess
import sys
import json
import os
from pathlib import Path

st.set_page_config(page_title="AI 운동 재활 시스템 제어 센터", layout="wide")

st.title("🏋️‍♂️ AI 운동 재활 시스템 제어 센터")
st.markdown("---")

# 🌟 app.py 위치 기반 절대 경로로 JSON 위치 고정
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
CONFIG_PATH = ROOT_DIR / "prototype2.1" / "configs" / "exercise_kpt_config_coco.json"

EXERCISE_LIST = []

if CONFIG_PATH.exists():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
        EXERCISE_LIST = list(cfg.get("exercises", {}).keys())

col1, col2 = st.columns([1, 1])

with col1:
    st.header("⚙️ 측정 세션 설정")
    
    col_p1, col_p2 = st.columns(2)
    player_id = col_p1.text_input("환자 ID (파일명용)", value="patient_1")
    patient_name = col_p2.text_input("환자 성함 (한글)", value="김지후")
    
    # 15개 전체 운동 목록 바인딩
    exercise_name = st.selectbox(f"운동 종목 선택 (총 {len(EXERCISE_LIST)}종)", EXERCISE_LIST)
    camera_index = st.selectbox("카메라 장치 선택", [0, 1, 2], format_func=lambda x: f"카메라 {x}번 (기본웹캠: 0)")
    target_reps = st.number_input("실제 운동 목표 횟수", min_value=1, max_value=50, value=10)

    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns(2)
    
    if col_btn1.button("🎯 시범 동작 측정 (3회)", type="primary", use_container_width=True):
        cmd = [
            sys.executable, str(CURRENT_DIR / "test_client.py"),
            "--player_id", player_id,
            "--patient_name", patient_name,
            "--exercise_name", exercise_name,
            "--mode", "CALIBRATION",
            "--target_reps", "3",
            "--camera_index", str(camera_index)
        ]
        subprocess.Popen(cmd)
        st.info(f"🎯 시범 동작 창이 열렸습니다. (카메라 {camera_index}번)")

    if col_btn2.button("🏋️ 실제 운동 측정 시작", use_container_width=True):
        cmd = [
            sys.executable, str(CURRENT_DIR / "test_client.py"),
            "--player_id", player_id,
            "--patient_name", patient_name,
            "--exercise_name", exercise_name,
            "--mode", "MAIN",
            "--target_reps", str(target_reps),
            "--camera_index", str(camera_index)
        ]
        subprocess.Popen(cmd)
        st.success(f"🏋️ 실제 운동 창이 열렸습니다. {target_reps}회 측정을 시작합니다.")

with col2:
    st.header("📊 환자 개인 맞춤 데이터 확인")
    patient_file = ROOT_DIR / "data" / f"{player_id}.json"
    
    if patient_file.exists():
        with open(patient_file, 'r', encoding='utf-8') as f:
            p_data = json.load(f)
            
        st.subheader(f"📌 환자 정보: {p_data.get('patient_name', '미지정')} ({player_id})")
        
        st.markdown("**현재 설정된 맞춤 Threshold (최신 덮어쓰기)**")
        custom_th = p_data.get("custom_thresholds", {}).get(exercise_name, "설정값 없음 (시범 동작 필요)")
        st.json(custom_th)

        st.markdown("**과거 시범 동작 히스토리**")
        calib_history = p_data.get("calibration_history", [])
        if calib_history:
            st.dataframe(calib_history, use_container_width=True)
        else:
            st.write("기록된 시범 동작 히스토리가 없습니다.")
    else:
        st.warning(f"'{player_id}' 환자의 기존 기록 데이터가 없습니다.")