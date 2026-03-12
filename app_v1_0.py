import streamlit as st
import pandas as pd  # 🌟 누락되었던 pandas 추가
import sys
from pathlib import Path
from datetime import datetime
import os
import time

# 경로 설정
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from utils.config_loader import load_exercise_configs
from prototype_v1_3 import run_counting
from utils.camera import get_camera_resolution

st.set_page_config(page_title="AI Exercise Counter", layout="wide")

# --- 세션 상태 초기화 ---
if 'exercise_logs' not in st.session_state:
    st.session_state.exercise_logs = pd.DataFrame(columns=[
        '운동명', '촬영각도', '왼손 횟수', '오른손 횟수', '시작 시간', '종료 시간', '소요 시간(초)'
    ])

st.title("Repetition Counter Prototype v1.3")

# YAML 설정 로드
all_configs = load_exercise_configs()
exercise_options = list(all_configs.keys())

if not exercise_options:
    st.error("❌ configs/exercises 폴더에 YAML 파일이 없습니다.")
    st.stop()

# --- 사이드바 UI 구성 ---
st.sidebar.header("⚙️ 운동 설정")
selected_ex = st.sidebar.selectbox("운동 종목 선택", exercise_options)
available_views = list(all_configs[selected_ex].keys())
selected_view = st.sidebar.radio("촬영 각도 선택", available_views)

st.sidebar.header("🎯 목표 설정")
target_reps = st.sidebar.number_input("목표 횟수를 입력하세요", min_value=1, value=10, step=1)
# app.py "운동 시작" 버튼 클릭 시
if st.sidebar.button("🚀 운동 시작", use_container_width=True):
    width, height = get_camera_resolution(0)
    
    if width:
        # 🌟 3개의 반환값을 모두 받습니다.
        final_counts, start_time, end_time = run_counting(selected_ex, selected_view, target_reps, width, height)
        
        # 소요 시간 계산
        duration = (end_time - start_time).total_seconds()
        
        # 새로운 로그 생성
        new_log = {
            '운동명': selected_ex,
            '촬영각도': selected_view,
            '왼손 횟수': final_counts.get('left', 0),
            '오른손 횟수': final_counts.get('right', 0),
            '시작 시간': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            '종료 시간': end_time.strftime('%Y-%m-%d %H:%M:%S'),
            '소요 시간(초)': round(duration, 1)
        }

        # 데이터프레임에 업데이트
        st.session_state.exercise_logs = pd.concat([
            st.session_state.exercise_logs, 
            pd.DataFrame([new_log])
        ], ignore_index=True)
        
        st.success("운동이 기록되었습니다!")

# 프로그램 완전 종료 버튼
# --- 사이드바 최하단 ---

# 프로그램 완전 종료 버튼
if st.sidebar.button("🛑 전체 프로그램 종료", use_container_width=True):
    # 1. 브라우저 창을 닫거나 안내 메시지를 띄우는 JavaScript
    st.components.v1.html(
        """
        <script>
            window.parent.window.close(); // 일반적인 창 닫기 시도
            alert("운동 프로그램이 종료되었습니다. 이 탭을 닫으셔도 됩니다.");
        </script>
        """,
        height=0,
    )
    
    # 2. 약간의 지연 시간을 주어 JS가 실행될 기회를 준 뒤 프로세스 종료
    st.warning("서버를 종료합니다...")
    time.sleep(2) 
    os._exit(0)

# --- 메인 화면: 로그 표 표시 ---
st.write("---") # 구분선
st.subheader("📊 최근 운동 기록")

if not st.session_state.exercise_logs.empty:
    # 표 출력 (최근 기록이 위로 오게 역순 출력)
    st.dataframe(st.session_state.exercise_logs.iloc[::-1], use_container_width=True)

    # --- CSV 내보내기 기능 ---
    csv = st.session_state.exercise_logs.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 전체 운동 기록 CSV로 저장하기",
        data=csv,
        file_name=f"workout_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
else:
    st.info("아직 운동 기록이 없습니다. 왼쪽 '운동 시작' 버튼을 눌러보세요!")