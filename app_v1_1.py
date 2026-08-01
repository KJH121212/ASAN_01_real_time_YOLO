import sys
import torch
import cv2
import os
from unittest.mock import MagicMock
from importlib.machinery import ModuleSpec

# 🌟 [CPU 폭주 방지] CPU 쓰레드 수를 2개로 제한
torch.set_num_threads(2)
cv2.setNumThreads(2)
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# 🌟 [CRITICAL FIX 1] Device Guard 및 mmcv-lite 환경용 C++ 모듈 스펙 모킹
mock_ext = MagicMock()
mock_ext.__spec__ = ModuleSpec("mmcv._ext", None)
sys.modules['mmcv._ext'] = mock_ext

# 🌟 [CRITICAL FIX 2] PyTorch 2.6+ weights_only=True 기본값 변경 우회
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import streamlit as st
import pandas as pd
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
# 🌟 [수정] legacy_code 패키지 제거 및 로컬 prototype_v1_3 연동
from prototype_v2 import run_counting
from utils.camera import get_camera_resolution

st.set_page_config(page_title="AI Exercise Counter", layout="wide")

# --- 세션 상태 초기화 ---
if 'exercise_logs' not in st.session_state:
    st.session_state.exercise_logs = pd.DataFrame(columns=[
        '운동명', '촬영각도', '왼손 횟수', '오른손 횟수', '시작 시간', '종료 시간', '소요 시간(초)'
    ])

st.title("Repetition Counter Prototype v1.3 (RTMPose)")

# YAML 설정 로드
all_configs = load_exercise_configs()
exercise_options = list(all_configs.keys())

if not exercise_options:
    st.error("❌ configs/exercises 폴더에 YAML 파일이 없습니다.")
    st.stop()

# --- 사이드바 UI 구성 ---
st.sidebar.header("⚙️ 운동 및 카메라 설정")

# 📷 [추가] 카메라 선택 드롭다운 / 인덱스 지정
camera_option = st.sidebar.selectbox(
    "📷 카메라 선택",
    options=[0, 1, 2, "직접 입력"],
    format_func=lambda x: f"카메라 {x}" if isinstance(x, int) else x,
    help="기본 내장 웹캠은 0번, 외장 USB 웹캠은 보통 1번 이상입니다."
)

if camera_option == "직접 입력":
    cam_idx = st.sidebar.number_input("카메라 인덱스 번호 (0, 1, 2...)", min_value=0, value=0, step=1)
else:
    cam_idx = camera_option

selected_ex = st.sidebar.selectbox("운동 종목 선택", exercise_options)
available_views = list(all_configs[selected_ex].keys())
selected_view = st.sidebar.radio("촬영 각도 선택", available_views)

st.sidebar.header("🎯 목표 설정")
target_reps = st.sidebar.number_input("목표 횟수를 입력하세요", min_value=1, value=10, step=1)

# app.py "운동 시작" 버튼 클릭 시
if st.sidebar.button("🚀 운동 시작", use_container_width=True):
    # 선택한 cam_idx의 해상도 가져오기
    width, height = get_camera_resolution(cam_idx)
    
    if width:
        # 🌟 [수정] RTMPose run_counting에 선택한 카메라 인덱스(cam_idx) 전달
        final_counts, start_time, end_time = run_counting(
            selected_ex, 
            selected_view, 
            target_reps, 
            width, 
            height, 
            cam_idx
        )
        
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
    else:
        st.error(f"❌ {cam_idx}번 카메라 연결 실패! 카메라 연결 및 장치 번호를 확인해 주세요.")

# --- 사이드바 최하단: 프로그램 종료 버튼 ---
st.sidebar.markdown("---")
if st.sidebar.button("🛑 전체 프로그램 종료", use_container_width=True):
    st.components.v1.html(
        """
        <script>
            window.parent.window.close();
            alert("운동 프로그램이 종료되었습니다. 이 탭을 닫으셔도 됩니다.");
        </script>
        """,
        height=0,
    )
    
    st.warning("서버를 종료합니다...")
    time.sleep(2) 
    os._exit(0)

# --- 메인 화면: 로그 표 표시 ---
st.write("---") # 구분선
st.subheader("📊 최근 운동 기록")

if not st.session_state.exercise_logs.empty:
    # 최근 기록이 상단에 오도록 역순 출력
    st.dataframe(st.session_state.exercise_logs.iloc[::-1], use_container_width=True)

    # CSV 저장 기능
    csv = st.session_state.exercise_logs.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        label="📥 전체 운동 기록 CSV로 저장하기",
        data=csv,
        file_name=f"workout_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )
else:
    st.info("아직 운동 기록이 없습니다. 왼쪽 '운동 시작' 버튼을 눌러보세요!")