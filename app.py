# 한국어 코드 해설: Streamlit 화면에서 세션 시작·종료, 가동 범위 편집, 저장 영상 오버레이 작업을 제어합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [모듈 정보]
# 파일: app.py
# 설명: AI 재활 운동 측정용 Streamlit 클라이언트입니다.
# 실시간 수치 측정과 저장 영상 파일 분석을 지원합니다.
# 2) 경량 모드: 웹캠 분석 결과를 수치로 확인합니다.
# 3) 파일 모드: 녹화 영상을 배치 분석합니다.
# ==============================================================================

# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 새 화면에서 이전 소켓 연결을 정리하는 비동기 요청을 실행합니다.
import asyncio
# 이 파일에서 사용할 `logging` 모듈을 불러옵니다.
import logging
# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time
# 이 파일에서 사용할 `streamlit`(`st`) 모듈을 불러옵니다.
import streamlit as st

# current 파일 경로에 `os.path.abspath(__file__)` 호출 결과를 저장합니다. 대상: `current_file_path`.
current_file_path = os.path.abspath(__file__)
# 최상위 경로 dir에 `os.path.dirname(current_file_path)` 호출 결과를 저장합니다. 대상: `root_dir`.
root_dir = os.path.dirname(current_file_path)
# 조건 `root_dir not in sys.path`을 만족하는 경우에만 아래 분기를 실행합니다.
if root_dir not in sys.path:
    # `sys.path.insert(0, root_dir)`을 호출하여 해당 단계의 작업을 수행합니다.
    sys.path.insert(0, root_dir)

# `core.video_overlay`에서 `VideoOverlayJob` 기능을 가져옵니다.
from core.video_overlay import VideoOverlayJob
# `network.streamlit_client`에서 `FINISH_PACKETS`, `StreamlitSessionClient` 기능을 가져옵니다.
from network.streamlit_client import FINISH_PACKETS, StreamlitSessionClient
# 모델을 다시 로딩하지 않고 서버의 이전 세션만 정리합니다.
from network.server_admin import reset_server_session
# `core.session_log`에서 `default_log_root` 기능을 가져옵니다.
from core.session_log import default_log_root
# `network.session_commands`에서 `DEFAULT_PREPARATION_SECONDS` 기능을 가져옵니다.
from network.session_commands import DEFAULT_PREPARATION_SECONDS


# 조회 최상위 경로 dir 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def get_root_dir() -> str:
    # 현재 함수 처리를 끝내고 `os.path.dirname(os.path.abspath(__file__))` 호출 결과를 호출한 쪽으로 반환합니다.
    return os.path.dirname(os.path.abspath(__file__))


# 데코레이터 `st.cache_data`를 바로 아래 함수·클래스에 적용합니다.
@st.cache_data
# 불러오기 exercise 설정 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def load_exercise_config() -> tuple[list[str], dict[str, str]]:
    # `search_paths` 값에 순서·구성을 지정한 `[os.path.join(get_root_dir(), 'exercise_kpt_config_coco.json'), os.path.join(get_root_dir(), 'configs', 'exercise_kpt_config_coco.json'), os.path.join(get_root_dir(), 'd…`을 저장합니다. 대상: `search_paths`.
    search_paths = [
        # 앞줄에서 시작한 식·인수·자료 구조를 `os.path.join(get_root_dir(), "exercise_kpt_config_coco.json"),` 내용으로 이어서 구성합니다.
        os.path.join(get_root_dir(), "exercise_kpt_config_coco.json"),
        # 앞줄에서 시작한 식·인수·자료 구조를 `os.path.join(get_root_dir(), "configs", "exercise_kpt_config_coco.json"),` 내용으로 이어서 구성합니다.
        os.path.join(get_root_dir(), "configs", "exercise_kpt_config_coco.json"),
        # 앞줄에서 시작한 식·인수·자료 구조를 `os.path.join(get_root_dir(), "data", "exercise_kpt_config_coco.json"),` 내용으로 이어서 구성합니다.
        os.path.join(get_root_dir(), "data", "exercise_kpt_config_coco.json"),
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    ]

    # `search_paths`에서 항목을 차례로 꺼내 `p`에 넣고 아래 처리를 반복합니다.
    for p in search_paths:
        # 조건 `os.path.exists(p)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if os.path.exists(p):
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `open(p, 'r', encoding='utf-8')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with open(p, "r", encoding="utf-8") as f:
                    # cfg 데이터에 `json.load(f)` 호출 결과를 저장합니다. 대상: `cfg_data`.
                    cfg_data = json.load(f)
                # `exercises` 값에 `cfg_data.get('exercises', {})` 호출 결과를 저장합니다. 대상: `exercises`.
                exercises = cfg_data.get("exercises", {})
                # `ex_keys` 값에 `list(exercises.keys())` 호출 결과를 저장합니다. 대상: `ex_keys`.
                ex_keys = list(exercises.keys())
                # `display_map` 값에 반복·조건식 `{k: exercises[k].get('display_name_kr', k) for k in ex_keys}`으로 구성한 값을 저장합니다. 대상: `display_map`.
                display_map = {
                    # 앞줄에서 시작한 식·인수·자료 구조를 `k: exercises[k].get("display_name_kr", k) for k in ex_keys` 내용으로 이어서 구성합니다.
                    k: exercises[k].get("display_name_kr", k) for k in ex_keys
                # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
                }
                # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(ex_keys, display_map)`을 호출한 쪽으로 반환합니다.
                return ex_keys, display_map
            # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except Exception as e:
                # `st.warning(f'설정 파일 파싱 중 오류 발생: {e}')`을 호출하여 해당 단계의 작업을 수행합니다.
                st.warning(f"설정 파일 파싱 중 오류 발생: {e}")

    # `fallback_keys` 값에 순서·구성을 지정한 `['biceps_curl', 'shoulder_press', 'clamshell', 'slr', 'hip_knee_flexion', 'knee_extension']`을 저장합니다. 대상: `fallback_keys`.
    fallback_keys = ["biceps_curl", "shoulder_press", "clamshell", "slr", "hip_knee_flexion", "knee_extension"]
    # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(fallback_keys, {k: k for k in fallback_keys})`을 호출한 쪽으로 반환합니다.
    return fallback_keys, {k: k for k in fallback_keys}




# 화면 표시 pose 갱신 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def render_pose_update(data, config, placeholders, packet_count=0, is_recording=False):
    # `current_mode` 값에 `data.get('mode', config['mode'])` 호출 결과를 저장합니다. 대상: `current_mode`.
    current_mode = data.get("mode", config["mode"])
    # 왼쪽 info에 계산식 `data.get('left', {}) or {}`의 결과를 저장합니다. 대상: `left_info`.
    left_info = data.get("left", {}) or {}
    # 오른쪽 info에 계산식 `data.get('right', {}) or {}`의 결과를 저장합니다. 대상: `right_info`.
    right_info = data.get("right", {}) or {}
    # 필수 관절이 가려졌거나 유효하지 않은지 여부에 `data.get('is_occluded', False)` 호출 결과를 저장합니다. 대상: `is_occluded`.
    is_occluded = data.get("is_occluded", False)
    # 캘리브레이션 전체 진행 단계에 `data.get('calib_step', '')` 호출 결과를 저장합니다. 대상: `calib_step`.
    calib_step = data.get("calib_step", "")

    # calib 반복 횟수 왼쪽에 `data.get('calib_reps_left', left_info.get('rep_count', 0))` 호출 결과를 저장합니다. 대상: `calib_reps_left`.
    calib_reps_left = data.get("calib_reps_left", left_info.get("rep_count", 0))
    # calib 반복 횟수 오른쪽에 `data.get('calib_reps_right', right_info.get('rep_count', 0))` 호출 결과를 저장합니다. 대상: `calib_reps_right`.
    calib_reps_right = data.get("calib_reps_right", right_info.get("rep_count", 0))
    # 목표 반복 횟수에 `current_mode == 'CALIBRATION'`이면 `3`, 아니면 `config.get('target_score', config.get('target_reps', 10))`을 저장합니다. 대상: `target_reps`.
    target_reps = 3 if current_mode == "CALIBRATION" else config.get('target_score', config.get('target_reps', 10))
    # 초당 프레임 수 val에 `data.get('fps', 0.0)` 호출 결과를 저장합니다. 대상: `fps_val`.
    fps_val = data.get("fps", 0.0)

    # Mode 2 shows measurements only; videos are overlaid in a separate batch job.
    # `l_cnt` 값에 `current_mode == 'CALIBRATION'`이면 `calib_reps_left`, 아니면 `left_info.get('total_score', 0)`을 저장합니다. 대상: `l_cnt`.
    l_cnt = calib_reps_left if current_mode == "CALIBRATION" else left_info.get("total_score", 0)
    # `r_cnt` 값에 `current_mode == 'CALIBRATION'`이면 `calib_reps_right`, 아니면 `right_info.get('total_score', 0)`을 저장합니다. 대상: `r_cnt`.
    r_cnt = calib_reps_right if current_mode == "CALIBRATION" else right_info.get("total_score", 0)
    # l 비율에 `int((left_info.get('progress_ratio', 0.0) or 0.0) * 100)` 호출 결과를 저장합니다. 대상: `l_ratio`.
    l_ratio = int((left_info.get("progress_ratio", 0.0) or 0.0) * 100)
    # r 비율에 `int((right_info.get('progress_ratio', 0.0) or 0.0) * 100)` 호출 결과를 저장합니다. 대상: `r_ratio`.
    r_ratio = int((right_info.get("progress_ratio", 0.0) or 0.0) * 100)

    # `placeholders['metrics']`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with placeholders["metrics"]:
        # `(c1, c2, c3, c4)` 값에 `st.columns(4)` 호출 결과를 저장합니다. 대상: `(c1, c2, c3, c4)`.
        c1, c2, c3, c4 = st.columns(4)
        # `c1.metric('왼쪽 (Left)', f'{l_cnt:g} / {target_reps:g}' + ('점' if current_mode != 'CALIBRATION' else '회'), f"{left_info.get('quality', 'READY')} · {left_info.get('rep_coun…`을 호출하여 해당 단계의 작업을 수행합니다.
        c1.metric("왼쪽 (Left)", f"{l_cnt:g} / {target_reps:g}" + ('점' if current_mode != 'CALIBRATION' else '회'),
                  # 앞줄에서 시작한 식·인수·자료 구조를 `f"{left_info.get('quality', 'READY')} · {left_info.get('rep_count', 0)}회")` 내용으로 이어서 구성합니다.
                  f"{left_info.get('quality', 'READY')} · {left_info.get('rep_count', 0)}회")
        # `c2.metric('Left ROM', f'{l_ratio}%', f"{left_info.get('state', 'READY')}")`을 호출하여 해당 단계의 작업을 수행합니다.
        c2.metric("Left ROM", f"{l_ratio}%", f"{left_info.get('state', 'READY')}")
        # `c3.metric('오른쪽 (Right)', f'{r_cnt:g} / {target_reps:g}' + ('점' if current_mode != 'CALIBRATION' else '회'), f"{right_info.get('quality', 'READY')} · {right_info.get('rep_…`을 호출하여 해당 단계의 작업을 수행합니다.
        c3.metric("오른쪽 (Right)", f"{r_cnt:g} / {target_reps:g}" + ('점' if current_mode != 'CALIBRATION' else '회'),
                  # 앞줄에서 시작한 식·인수·자료 구조를 `f"{right_info.get('quality', 'READY')} · {right_info.get('rep_count', 0)}회")` 내용으로 이어서 구성합니다.
                  f"{right_info.get('quality', 'READY')} · {right_info.get('rep_count', 0)}회")
        # `c4.metric('Right ROM', f'{r_ratio}%', f"{right_info.get('state', 'READY')}")`을 호출하여 해당 단계의 작업을 수행합니다.
        c4.metric("Right ROM", f"{r_ratio}%", f"{right_info.get('state', 'READY')}")

    # 상단 알림 배너 갱신
    # 조건 `current_mode == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
    if current_mode == "CALIBRATION":
        # 조건 `is_occluded`을 만족하는 경우에만 아래 분기를 실행합니다.
        if is_occluded:
            # `placeholders['alert'].warning('검출 일시 중단: 시작 자세가 보이도록 해 주세요. 짧은 누락은 완료 횟수를 유지하며, 지속되면 초기 자세부터 다시 측정합니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders["alert"].warning("검출 일시 중단: 시작 자세가 보이도록 해 주세요. 짧은 누락은 완료 횟수를 유지하며, 지속되면 초기 자세부터 다시 측정합니다.")
        # 조건 `calib_step in ('COUNTDOWN', 'FULL_BODY_CHECK')`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif calib_step in ('COUNTDOWN', 'FULL_BODY_CHECK'):
            # `placeholders['alert'].info('초기 자세 측정 중: 양쪽을 시작 자세로 유지하고 잠시 멈춰 주세요.')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders['alert'].info('초기 자세 측정 중: 양쪽을 시작 자세로 유지하고 잠시 멈춰 주세요.')
        # 조건 `calib_step == 'CHECK_CONSISTENCY'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif calib_step == 'CHECK_CONSISTENCY':
            # `placeholders['alert'].warning('반복 크기가 일정하지 않습니다. 비슷한 범위로 추가 반복하고 시작 자세로 복귀해 주세요.')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders['alert'].warning('반복 크기가 일정하지 않습니다. 비슷한 범위로 추가 반복하고 시작 자세로 복귀해 주세요.')
        # 조건 `calib_step == 'COLLECTING'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif calib_step == "COLLECTING":
            # `placeholders['alert'].info(f'측정 진행 중 (양측 각각 3회) ➔ 왼쪽: {calib_reps_left}/{target_reps} | 오른쪽: {calib_reps_right}/{target_reps}. 시작 자세로 복귀한 뒤 잠시 멈춰 주세요.')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders["alert"].info(
                # 앞줄에서 시작한 식·인수·자료 구조를 `f"측정 진행 중 (양측 각각 3회) ➔ 왼쪽: {calib_reps_left}/{target_reps} | 오른쪽: {calib_reps_right}/{target_reps}. 시작 자세로 복귀한 뒤 잠시 멈춰 주세요."` 내용으로 이어서 구성합니다.
                f"측정 진행 중 (양측 각각 3회) ➔ 왼쪽: {calib_reps_left}/{target_reps} | 오른쪽: {calib_reps_right}/{target_reps}. 시작 자세로 복귀한 뒤 잠시 멈춰 주세요."
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            )
        # 조건 `calib_step == 'COOLDOWN'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif calib_step == "COOLDOWN":
            # `placeholders['alert'].success('양측 3회 달성 완수! 임계값 저장 중...')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders["alert"].success("양측 3회 달성 완수! 임계값 저장 중...")
    # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
    else:
        # `l_cnt` 값에 `left_info.get('total_score', 0)` 호출 결과를 저장합니다. 대상: `l_cnt`.
        l_cnt = left_info.get("total_score", 0)
        # `r_cnt` 값에 `right_info.get('total_score', 0)` 호출 결과를 저장합니다. 대상: `r_cnt`.
        r_cnt = right_info.get("total_score", 0)
        # 조건 `is_occluded`을 만족하는 경우에만 아래 분기를 실행합니다.
        if is_occluded:
            # `placeholders['alert'].error('신체 가림 발생: 동작 측정이 일시 중단되었습니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders["alert"].error("신체 가림 발생: 동작 측정이 일시 중단되었습니다.")
        # 조건 `l_cnt > target_reps or r_cnt > target_reps`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif l_cnt > target_reps or r_cnt > target_reps:
            # `placeholders['alert'].success(f'목표 점수 달성! (L: {l_cnt:g}, R: {r_cnt:g})')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders["alert"].success(f"목표 점수 달성! (L: {l_cnt:g}, R: {r_cnt:g})")
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # `placeholders['alert'].empty()`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders["alert"].empty()

    # rec 상태에 `is_recording`이면 `' | REC: ON'`, 아니면 `' | REC: OFF'`을 저장합니다. 대상: `rec_status`.
    rec_status = " | REC: ON" if is_recording else " | REC: OFF"
    # 캘리브레이션 상태에 ''을 저장합니다. 대상: `calibration_status`.
    calibration_status = ''
    # 조건 `current_mode == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
    if current_mode == 'CALIBRATION':
        # `(('L', left_info), ('R', right_info))`에서 항목을 차례로 꺼내 `(label, side)`에 넣고 아래 처리를 반복합니다.
        for label, side in (('L', left_info), ('R', right_info)):
            # `detail` 값에 `side.get('calibration', {})` 호출 결과를 저장합니다. 대상: `detail`.
            detail = side.get('calibration', {})
            # 조건 `detail`을 만족하는 경우에만 아래 분기를 실행합니다.
            if detail:
                # 캘리브레이션 상태에 `f" | {label}: {detail.get('state', '')} · 준비 샘플 {detail.get('baseline_samples', 0)}/{detail.get('min_baseline_samples', 3)} · 복귀 허용 {detail.get('return_tolerance', 0):.3…`을 더해 누적합니다. 대상: `calibration_status`.
                calibration_status += (f" | {label}: {detail.get('state', '')}"
                    # 앞줄에서 시작한 식·인수·자료 구조를 `f" · 준비 샘플 {detail.get('baseline_samples', 0)}/{detail.get('min_baseline_samples', 3)}"` 내용으로 이어서 구성합니다.
                    f" · 준비 샘플 {detail.get('baseline_samples', 0)}/{detail.get('min_baseline_samples', 3)}"
                    # 앞줄에서 시작한 식·인수·자료 구조를 `f" · 복귀 허용 {detail.get('return_tolerance', 0):.3f}")` 내용으로 이어서 구성합니다.
                    f" · 복귀 허용 {detail.get('return_tolerance', 0):.3f}")
    # `placeholders['fps'].caption(f"FPS: {fps_val:.1f} | Mode: {current_mode} ({config['output_mode']}){rec_status} | Packets: {packet_count}{calibration_status}")`을 호출하여 해당 단계의 작업을 수행합니다.
    placeholders["fps"].caption(
        # 앞줄에서 시작한 식·인수·자료 구조를 `f"FPS: {fps_val:.1f} | Mode: {current_mode} ({config['output_mode']}){rec_status} | Packets: {packet_count}{calibration_status}"` 내용으로 이어서 구성합니다.
        f"FPS: {fps_val:.1f} | Mode: {current_mode} ({config['output_mode']}){rec_status} | Packets: {packet_count}{calibration_status}"
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    )



# 화면 표시 가동 범위 editor 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def render_rom_editor(client, status):
    # `editable` 값에 can edit 가동 범위 기준(`client.can_edit_thresholds`)을 저장합니다. 대상: `editable`.
    editable = client.can_edit_thresholds
    # 조건 `not status or not status.get('thresholds')`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not status or not status.get("thresholds"):
        # `st.info('세션을 시작하면 서버의 가동 범위를 불러옵니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
        st.info("세션을 시작하면 서버의 가동 범위를 불러옵니다.")
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return
    # `prefix` 값에 계산식 `f'rom_{client.session_id}'`의 결과를 저장합니다. 대상: `prefix`.
    prefix = f"rom_{client.session_id}"
    # 조건 `st.session_state.get('rom_editor_session') != client.session_id`을 만족하는 경우에만 아래 분기를 실행합니다.
    if st.session_state.get("rom_editor_session") != client.session_id:
        # `status['thresholds'].items()`에서 항목을 차례로 꺼내 `(side, values)`에 넣고 아래 처리를 반복합니다.
        for side, values in status["thresholds"].items():
            # `values.items()`에서 항목을 차례로 꺼내 `(field, value)`에 넣고 아래 처리를 반복합니다.
            for field, value in values.items():
                # 측정 세션 state[f'{prefix} {side} {field}']에 `float(value)` 호출 결과를 저장합니다. 대상: `st.session_state[f'{prefix}_{side}_{field}']`.
                st.session_state[f"{prefix}_{side}_{field}"] = float(value)
        # 가동 범위 editor 측정 세션에 현재 측정 세션 식별자(`client.session_id`)을 저장합니다. 대상: `st.session_state.rom_editor_session`.
        st.session_state.rom_editor_session = client.session_id

    # `metric` 값에 `status.get('metric_type')` 호출 결과를 저장합니다. 대상: `metric`.
    metric = status.get("metric_type")
    # `unit` 값에 `{'ANGLE': '각도 (°)', 'COORDINATE': '몸통 길이 기준 상대 높이', 'LENGTH': '거리 지표', 'RATIO': '길이 비율 (Length Ratio)'}.get(metric, metric)` 호출 결과를 저장합니다. 대상: `unit`.
    unit = {"ANGLE": "각도 (°)", "COORDINATE": "몸통 길이 기준 상대 높이",
            # 구성 항목: 길이(`'LENGTH'`)에 '거리 지표'; 비율(`'RATIO'`)에 '길이 비율 (Length Ratio)'을 지정합니다.
            "LENGTH": "거리 지표", "RATIO": "길이 비율 (Length Ratio)"}.get(metric, metric)
    # `st.caption(f'단위: {unit} · 현재 세션에 적용')`을 호출하여 해당 단계의 작업을 수행합니다.
    st.caption(f"단위: {unit} · 현재 세션에 적용")
    # `st.caption('변경 후 시작 자세로 돌아오면 다음 반복부터 새 범위로 판정합니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
    st.caption("변경 후 시작 자세로 돌아오면 다음 반복부터 새 범위로 판정합니다.")
    # 조건 `not editable`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not editable:
        # `st.info('가동 범위 수정은 실시간 MAIN / TEST 측정 중에 사용할 수 있습니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
        st.info("가동 범위 수정은 실시간 MAIN / TEST 측정 중에 사용할 수 있습니다.")

    # `selected` 값에 `st.radio('수정 대상', ['both', 'left', 'right'], format_func=lambda side: {'both': '양쪽', 'left': '왼쪽', 'right': '오른쪽'}[side], horizontal=True, key=f'{prefix}_side', disabled…` 호출 결과를 저장합니다. 대상: `selected`.
    selected = st.radio("수정 대상", ["both", "left", "right"],
                        # 이 줄의 입력 인수: `side` = 좌우 구분. 기본값·형식은 아래 선언을 따릅니다.
                        format_func=lambda side: {"both": "양쪽", "left": "왼쪽", "right": "오른쪽"}[side],
                        # 구성 항목: `horizontal` 값(`horizontal`)에 True; `key` 값(`key`)에 계산식 `f'{prefix}_side'`의 결과; `disabled` 값(`disabled`)에 계산식 `not editable`의 결과를 지정합니다.
                        horizontal=True, key=f"{prefix}_side", disabled=not editable)
    # `sides` 값에 `selected == 'both'`이면 `('left', 'right')`, 아니면 `(selected,)`을 저장합니다. 대상: `sides`.
    sides = ("left", "right") if selected == "both" else (selected,)
    # `(sync_col, query_col)` 값에 `st.columns(2)` 호출 결과를 저장합니다. 대상: `(sync_col, query_col)`.
    sync_col, query_col = st.columns(2)
    # 조건 `sync_col.button('현재 적용값 불러오기', key=f'{prefix}_load')`을 만족하는 경우에만 아래 분기를 실행합니다.
    if sync_col.button("현재 적용값 불러오기", key=f"{prefix}_load"):
        # `status['thresholds'].items()`에서 항목을 차례로 꺼내 `(side, values)`에 넣고 아래 처리를 반복합니다.
        for side, values in status["thresholds"].items():
            # `values.items()`에서 항목을 차례로 꺼내 `(field, value)`에 넣고 아래 처리를 반복합니다.
            for field, value in values.items():
                # 측정 세션 state[f'{prefix} {side} {field}']에 `float(value)` 호출 결과를 저장합니다. 대상: `st.session_state[f'{prefix}_{side}_{field}']`.
                st.session_state[f"{prefix}_{side}_{field}"] = float(value)
    # 조건 `query_col.button('서버 적용값 조회', key=f'{prefix}_query', disabled=not client.running or bool(client.config.get('video_path')))`을 만족하는 경우에만 아래 분기를 실행합니다.
    if query_col.button("서버 적용값 조회", key=f"{prefix}_query",
                        # 구성 항목: `disabled` 값(`disabled`)에 계산식 `not client.running or bool(client.config.get('video_path'))`의 결과를 지정합니다.
                        disabled=not client.running or bool(client.config.get("video_path"))):
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `client.threshold_command()`을 호출하여 해당 단계의 작업을 수행합니다.
            client.threshold_command()
            # 가동 범위 notice에 순서·구성을 지정한 `('info', '서버에 현재 가동 범위를 조회했습니다.')`을 저장합니다. 대상: `st.session_state.rom_notice`.
            st.session_state.rom_notice = ("info", "서버에 현재 가동 범위를 조회했습니다.")
        # `(RuntimeError, ValueError)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except (RuntimeError, ValueError) as exc:
            # 가동 범위 notice에 순서·구성을 지정한 `('error', str(exc))`을 저장합니다. 대상: `st.session_state.rom_notice`.
            st.session_state.rom_notice = ("error", str(exc))

    # `st.form(f'{prefix}_form')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with st.form(f"{prefix}_form"):
        # `columns` 값에 `st.columns(2)` 호출 결과를 저장합니다. 대상: `columns`.
        columns = st.columns(2)
        # `draft` 값에 빈 사전을 저장합니다. 대상: `draft`.
        draft = {}
        # `zip(columns, ('left', 'right'), ('왼쪽', '오른쪽'))`에서 항목을 차례로 꺼내 `(column, side, label)`에 넣고 아래 처리를 반복합니다.
        for column, side, label in zip(columns, ("left", "right"), ("왼쪽", "오른쪽")):
            # `column`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with column:
                # `st.markdown(f'**{label}**')`을 호출하여 해당 단계의 작업을 수행합니다.
                st.markdown(f"**{label}**")
                # `kwargs` 값에 키 'step', 'format', 'disabled'로 구성한 사전을 저장합니다. 대상: `kwargs`.
                kwargs = {"step": 1.0 if metric == "ANGLE" else 0.01, "format": "%.3f",
                          # 구성 항목: `disabled` 값(`'disabled'`)에 계산식 `not editable or side not in sides`의 결과를 지정합니다.
                          "disabled": not editable or side not in sides}
                # 조건 `metric in {'ANGLE', 'LENGTH', 'RATIO'}`을 만족하는 경우에만 아래 분기를 실행합니다.
                if metric in {"ANGLE", "LENGTH", "RATIO"}:
                    # `kwargs['min_value']` 값에 0.0을 저장합니다. 대상: `kwargs['min_value']`.
                    kwargs["min_value"] = 0.0
                # 조건 `metric == 'ANGLE'`을 만족하는 경우에만 아래 분기를 실행합니다.
                if metric == "ANGLE":
                    # `kwargs['max_value']` 값에 180.0을 저장합니다. 대상: `kwargs['max_value']`.
                    kwargs["max_value"] = 180.0
                # 시작에 `st.number_input(f'{label} 시작값', key=f'{prefix}_{side}_start_val', **kwargs)` 호출 결과를 저장합니다. 대상: `start`.
                start = st.number_input(f"{label} 시작값", key=f"{prefix}_{side}_start_val", **kwargs)
                # 목표에 `st.number_input(f'{label} 목표값', key=f'{prefix}_{side}_target_val', **kwargs)` 호출 결과를 저장합니다. 대상: `target`.
                target = st.number_input(f"{label} 목표값", key=f"{prefix}_{side}_target_val", **kwargs)
                # `draft[side]` 값에 키 'start_val', 'target_val'로 구성한 사전을 저장합니다. 대상: `draft[side]`.
                draft[side] = {"start_val": start, "target_val": target}
        # `submitted` 값에 `st.form_submit_button('가동 범위 적용', type='primary', disabled=not editable or bool(st.session_state.get('pending_rom_request')))` 호출 결과를 저장합니다. 대상: `submitted`.
        submitted = st.form_submit_button(
            # 구성 항목: `type` 값(`type`)에 'primary'을 지정합니다.
            "가동 범위 적용", type="primary",
            # 구성 항목: `disabled` 값(`disabled`)에 계산식 `not editable or bool(st.session_state.get('pending_rom_request'))`의 결과를 지정합니다.
            disabled=not editable or bool(st.session_state.get("pending_rom_request")))
    # 조건 `submitted`을 만족하는 경우에만 아래 분기를 실행합니다.
    if submitted:
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `patch` 값에 반복·조건식 `{side: {key: float(value) for (key, value) in draft[side].items()} for side in sides}`으로 구성한 값을 저장합니다. 대상: `patch`.
            patch = {side: {key: float(value) for key, value in draft[side].items()} for side in sides}
            # 명령과 응답을 연결하는 요청 식별자에 `client.threshold_command(patch)` 호출 결과를 저장합니다. 대상: `request_id`.
            request_id = client.threshold_command(patch)
            # 대기 가동 범위 request에 명령과 응답을 연결하는 요청 식별자(`request_id`)을 저장합니다. 대상: `st.session_state.pending_rom_request`.
            st.session_state.pending_rom_request = request_id
            # 대기 가동 범위 at에 시스템 시계 변경에 영향받지 않는 시각(초)(`time.monotonic()`)을 저장합니다. 대상: `st.session_state.pending_rom_at`.
            st.session_state.pending_rom_at = time.monotonic()
            # 가동 범위 notice에 순서·구성을 지정한 `('info', '가동 범위 수정 요청을 보냈습니다. 서버 응답을 기다립니다.')`을 저장합니다. 대상: `st.session_state.rom_notice`.
            st.session_state.rom_notice = ("info", "가동 범위 수정 요청을 보냈습니다. 서버 응답을 기다립니다.")
        # `(RuntimeError, TypeError, ValueError)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except (RuntimeError, TypeError, ValueError) as exc:
            # 가동 범위 notice에 순서·구성을 지정한 `('error', f'입력값을 확인하세요: {exc}')`을 저장합니다. 대상: `st.session_state.rom_notice`.
            st.session_state.rom_notice = ("error", f"입력값을 확인하세요: {exc}")


# 데코레이터 `st.fragment(run_every=0.2)`를 바로 아래 함수·클래스에 적용합니다.
@st.fragment(run_every=0.2)
# 표시 갱신 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def refresh_session(client, placeholders):
    # `client.touch_ui()`을 호출하여 해당 단계의 작업을 수행합니다.
    client.touch_ui()
    # 대기 이전에 `st.session_state.get('pending_rom_request')` 호출 결과를 저장합니다. 대상: `pending_before`.
    pending_before = st.session_state.get("pending_rom_request")
    # `snapshot` 값에 `client.snapshot()` 호출 결과를 저장합니다. 대상: `snapshot`.
    snapshot = client.snapshot()
    # `client.drain_events()`에서 항목을 차례로 꺼내 `packet`에 넣고 아래 처리를 반복합니다.
    for packet in client.drain_events():
        # 메시지 type에 `packet.get('type')` 호출 결과를 저장합니다. 대상: `packet_type`.
        packet_type = packet.get("type")
        # 조건 `packet_type in {'COUNT_THRESHOLD_STATUS', 'COUNT_THRESHOLD_UPDATED'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        if packet_type in {"COUNT_THRESHOLD_STATUS", "COUNT_THRESHOLD_UPDATED"}:
            # 조건 `packet_type == 'COUNT_THRESHOLD_UPDATED'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if packet_type == "COUNT_THRESHOLD_UPDATED":
                # 조건 `packet.get('request_id') == st.session_state.get('pending_rom_request')`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet.get("request_id") == st.session_state.get("pending_rom_request"):
                    # 대기 가동 범위 request에 None을 저장합니다. 대상: `st.session_state.pending_rom_request`.
                    st.session_state.pending_rom_request = None
                # 메시지에 `packet.get('changed_sides')`이면 `f"가동 범위 적용 완료 · 버전 {packet.get('threshold_version')}. 시작 자세로 복귀한 뒤 다시 움직여 주세요."`, 아니면 `'기존 값과 같아 변경하지 않았습니다.'`을 저장합니다. 대상: `message`.
                message = (f"가동 범위 적용 완료 · 버전 {packet.get('threshold_version')}. "
                           # 앞줄에서 시작한 식·인수·자료 구조를 `"시작 자세로 복귀한 뒤 다시 움직여 주세요."` 내용으로 이어서 구성합니다.
                           "시작 자세로 복귀한 뒤 다시 움직여 주세요."
                           # 앞줄에서 시작한 식·인수·자료 구조를 `if packet.get("changed_sides") else "기존 값과 같아 변경하지 않았습니다.")` 내용으로 이어서 구성합니다.
                           if packet.get("changed_sides") else "기존 값과 같아 변경하지 않았습니다.")
                # 가동 범위 notice에 순서·구성을 지정한 `('success', message)`을 저장합니다. 대상: `st.session_state.rom_notice`.
                st.session_state.rom_notice = ("success", message)
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # 가동 범위 notice에 순서·구성을 지정한 `('info', '서버의 현재 가동 범위를 확인했습니다.')`을 저장합니다. 대상: `st.session_state.rom_notice`.
                st.session_state.rom_notice = ("info", "서버의 현재 가동 범위를 확인했습니다.")
        # 조건 `packet_type == 'SESSION_PREPARING'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif packet_type == "SESSION_PREPARING":
            # 측정 세션 notice에 순서·구성을 지정한 `('info', packet['message'])`을 저장합니다. 대상: `st.session_state.session_notice`.
            st.session_state.session_notice = ("info", packet["message"])
        # 조건 `packet_type == 'SESSION_MEASUREMENT_STARTED'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif packet_type == "SESSION_MEASUREMENT_STARTED":
            # 측정 세션 notice에 None을 저장합니다. 대상: `st.session_state.session_notice`.
            st.session_state.session_notice = None
        # 조건 `packet_type == 'VIDEO_STATUS'`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif packet_type == "VIDEO_STATUS":
            # 측정 세션 notice에 순서·구성을 지정한 `('info', packet.get('message', '영상 분석 중입니다.'))`을 저장합니다. 대상: `st.session_state.session_notice`.
            st.session_state.session_notice = ("info", packet.get("message", "영상 분석 중입니다."))
        # 조건 `packet_type in FINISH_PACKETS`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif packet_type in FINISH_PACKETS:
            # 저장·종료 결과 요약에 계산식 `packet.get('summary') or {}`의 결과를 저장합니다. 대상: `summary`.
            summary = packet.get("summary") or {}
            # 관절 신뢰도 배열 또는 좌우 게임 점수(사용 위치에 따름)에 계산식 `snapshot['progress']['scores']`의 결과를 저장합니다. 대상: `scores`.
            scores = snapshot["progress"]["scores"]
            # 조건 `packet.get('status') == 'STOPPED'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if packet.get("status") == "STOPPED":
                # 측정 세션 notice에 순서·구성을 지정한 `('info', '세션을 중지하고 기록을 저장했습니다.')`을 저장합니다. 대상: `st.session_state.session_notice`.
                st.session_state.session_notice = ("info", "세션을 중지하고 기록을 저장했습니다.")
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # 조건 `packet.get('calibration_completed') is False`을 만족하는 경우에만 아래 분기를 실행합니다.
            if packet.get('calibration_completed') is False:
                # 측정 세션 notice에 순서·구성을 지정한 `('warning', '영상이 끝났지만 캘리브레이션은 미완료입니다. 개인 가동 범위는 저장하지 않았습니다.')`을 저장합니다. 대상: `st.session_state.session_notice`.
                st.session_state.session_notice = ('warning', '영상이 끝났지만 캘리브레이션은 미완료입니다. 개인 가동 범위는 저장하지 않았습니다.')
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # 조건 `packet.get('thresholds')`을 만족하는 경우에만 아래 분기를 실행합니다.
            if packet.get("thresholds"):
                # last calib 가동 범위 기준에 계산식 `packet['thresholds']`의 결과를 저장합니다. 대상: `st.session_state.last_calib_thresholds`.
                st.session_state.last_calib_thresholds = packet["thresholds"]
            # 메시지에 `'CALIBRATION' in packet_type or client.config['mode'] == 'CALIBRATION'`이면 `'캘리브레이션 완료. 맞춤 가동 범위를 저장했습니다.'`, 아니면 `f"분석 완료 · 왼쪽 {scores['left']:g}점 / 오른쪽 {scores['right']:g}점 · 반복 기록 {summary.get('saved_reps_count', 0)}개"`을 저장합니다. 대상: `message`.
            message = ("캘리브레이션 완료. 맞춤 가동 범위를 저장했습니다."
                       # 앞줄에서 시작한 식·인수·자료 구조를 `if "CALIBRATION" in packet_type or client.config["mode"] == "CALIBRATION"` 내용으로 이어서 구성합니다.
                       if "CALIBRATION" in packet_type or client.config["mode"] == "CALIBRATION"
                       # 앞줄에서 시작한 식·인수·자료 구조를 `else f"분석 완료 · 왼쪽 {scores['left']:g}점 / "` 내용으로 이어서 구성합니다.
                       else f"분석 완료 · 왼쪽 {scores['left']:g}점 / "
                            # 앞줄에서 시작한 식·인수·자료 구조를 `f"오른쪽 {scores['right']:g}점 · "` 내용으로 이어서 구성합니다.
                            f"오른쪽 {scores['right']:g}점 · "
                            # 앞줄에서 시작한 식·인수·자료 구조를 `f"반복 기록 {summary.get('saved_reps_count', 0)}개")` 내용으로 이어서 구성합니다.
                            f"반복 기록 {summary.get('saved_reps_count', 0)}개")
            # 측정 세션 notice에 순서·구성을 지정한 `('success', message)`을 저장합니다. 대상: `st.session_state.session_notice`.
            st.session_state.session_notice = ("success", message)
        # 조건 `packet_type in {'ERROR', 'CLIENT_ERROR'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif packet_type in {"ERROR", "CLIENT_ERROR"}:
            # 메시지에 `packet.get('message', '서버 통신 오류')` 호출 결과를 저장합니다. 대상: `message`.
            message = packet.get("message", "서버 통신 오류")
            # 조건 `packet.get('command') in {'CMD_GET_COUNT_THRESHOLD', 'CMD_UPDATE_COUNT_THRESHOLD'}`을 만족하는 경우에만 아래 분기를 실행합니다.
            if packet.get("command") in {"CMD_GET_COUNT_THRESHOLD", "CMD_UPDATE_COUNT_THRESHOLD"}:
                # 조건 `packet.get('request_id') == st.session_state.get('pending_rom_request')`을 만족하는 경우에만 아래 분기를 실행합니다.
                if packet.get("request_id") == st.session_state.get("pending_rom_request"):
                    # 대기 가동 범위 request에 None을 저장합니다. 대상: `st.session_state.pending_rom_request`.
                    st.session_state.pending_rom_request = None
                # 가동 범위 notice에 순서·구성을 지정한 `('error', message)`을 저장합니다. 대상: `st.session_state.rom_notice`.
                st.session_state.rom_notice = ("error", message)
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # 측정 세션 notice에 순서·구성을 지정한 `('error', message)`을 저장합니다. 대상: `st.session_state.session_notice`.
                st.session_state.session_notice = ("error", message)

    # 조건 `st.session_state.get('pending_rom_request') and time.monotonic() - st.session_state.get('pending_rom_at', 0) > 10`을 만족하는 경우에만 아래 분기를 실행합니다.
    if (st.session_state.get("pending_rom_request")
            # 앞줄에서 시작한 식·인수·자료 구조를 `and time.monotonic() - st.session_state.get("pending_rom_at", 0) > 10):` 내용으로 이어서 구성합니다.
            and time.monotonic() - st.session_state.get("pending_rom_at", 0) > 10):
        # 대기 가동 범위 request에 None을 저장합니다. 대상: `st.session_state.pending_rom_request`.
        st.session_state.pending_rom_request = None
        # 가동 범위 notice에 순서·구성을 지정한 `('warning', '수정 응답이 지연되고 있습니다. 서버 적용값을 조회해 확인하세요.')`을 저장합니다. 대상: `st.session_state.rom_notice`.
        st.session_state.rom_notice = ("warning", "수정 응답이 지연되고 있습니다. 서버 적용값을 조회해 확인하세요.")

    # `pose` 값에 계산식 `snapshot['pose']`의 결과를 저장합니다. 대상: `pose`.
    pose = snapshot["pose"]
    # 조건 `pose`을 만족하는 경우에만 아래 분기를 실행합니다.
    if pose:
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `render_pose_update(pose, client.config, placeholders, snapshot['packet_count'], client.config.get('record_video', False))`을 호출하여 해당 단계의 작업을 수행합니다.
            render_pose_update(pose, client.config, placeholders,
                               # 앞줄에서 시작한 식·인수·자료 구조를 `snapshot["packet_count"], client.config.get("record_video", False))` 내용으로 이어서 구성합니다.
                               snapshot["packet_count"], client.config.get("record_video", False))
        # `(ValueError, TypeError, KeyError, OverflowError)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            # `logging.getLogger(__name__).exception('Measurement rendering failed')`을 호출하여 해당 단계의 작업을 수행합니다.
            logging.getLogger(__name__).exception('Measurement rendering failed')
            # `placeholders['video'].error(f'측정 결과 표시 오류: {exc}')`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders['video'].error(f'측정 결과 표시 오류: {exc}')
    # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
    else:
        # `placeholders['fps'].caption(f"연결 상태: {snapshot['status']}")`을 호출하여 해당 단계의 작업을 수행합니다.
        placeholders["fps"].caption(f"연결 상태: {snapshot['status']}")
        # 서버 상태에 `snapshot.get('server_status')` 호출 결과를 저장합니다. 대상: `server_status`.
        server_status = snapshot.get("server_status")
        # 조건 `server_status and server_status.get('model_state') == 'LOADING'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if server_status and server_status.get("model_state") == "LOADING":
            # `elapsed` 값에 `server_status.get('elapsed_seconds', 0)` 호출 결과를 저장합니다. 대상: `elapsed`.
            elapsed = server_status.get("elapsed_seconds", 0)
            # `placeholders['alert'].info(f"서버 연결됨 · {server_status.get('message')} ({elapsed:.1f}초) 준비가 끝나면 세션을 자동으로 시작합니다.")`을 호출하여 해당 단계의 작업을 수행합니다.
            placeholders["alert"].info(
                # 앞줄에서 시작한 식·인수·자료 구조를 `f"서버 연결됨 · {server_status.get('message')} ({elapsed:.1f}초) "` 내용으로 이어서 구성합니다.
                f"서버 연결됨 · {server_status.get('message')} ({elapsed:.1f}초) "
                # 앞줄에서 시작한 식·인수·자료 구조를 `"준비가 끝나면 세션을 자동으로 시작합니다.")` 내용으로 이어서 구성합니다.
                "준비가 끝나면 세션을 자동으로 시작합니다.")
    # `notice` 값에 `st.session_state.get('session_notice')` 호출 결과를 저장합니다. 대상: `notice`.
    notice = st.session_state.get("session_notice")
    # 조건 `notice`을 만족하는 경우에만 아래 분기를 실행합니다.
    if notice:
        # `getattr(placeholders['alert'], notice[0])(notice[1])`을 호출하여 해당 단계의 작업을 수행합니다.
        getattr(placeholders["alert"], notice[0])(notice[1])
    # 가동 범위 notice에 `st.session_state.get('rom_notice')` 호출 결과를 저장합니다. 대상: `rom_notice`.
    rom_notice = st.session_state.get("rom_notice")
    # 조건 `rom_notice`을 만족하는 경우에만 아래 분기를 실행합니다.
    if rom_notice:
        # `getattr(placeholders['rom_feedback'], rom_notice[0])(rom_notice[1])`을 호출하여 해당 단계의 작업을 수행합니다.
        getattr(placeholders["rom_feedback"], rom_notice[0])(rom_notice[1])
    # 로그 info에 `snapshot.get('log_info')` 호출 결과를 저장합니다. 대상: `log_info`.
    log_info = snapshot.get("log_info")
    # 조건 `log_info`을 만족하는 경우에만 아래 분기를 실행합니다.
    if log_info:
        # `placeholders['log_paths'].container()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with placeholders["log_paths"].container():
            # `st.caption('서버의 실제 로그 저장 위치')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption("서버의 실제 로그 저장 위치")
            # `st.code(log_info['directory'], language=None)`을 호출하여 해당 단계의 작업을 수행합니다.
            st.code(log_info["directory"], language=None)
            # `st.caption(f"CSV: {Path(log_info['frame_csv']).name} · NPZ: {Path(log_info['skeleton_npz']).name}")`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption(f"CSV: {Path(log_info['frame_csv']).name} · NPZ: {Path(log_info['skeleton_npz']).name}")
            # 조건 `log_info.get('video_mp4')`을 만족하는 경우에만 아래 분기를 실행합니다.
            if log_info.get("video_mp4"):
                # `st.caption(f"MP4: {Path(log_info['video_mp4']).name}")`을 호출하여 해당 단계의 작업을 수행합니다.
                st.caption(f"MP4: {Path(log_info['video_mp4']).name}")

    # 상태에 계산식 `snapshot['threshold_status']`의 결과를 저장합니다. 대상: `status`.
    status = snapshot["threshold_status"]
    # `placeholders['applied_rom'].container()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with placeholders["applied_rom"].container():
        # 조건 `status and status.get('thresholds')`을 만족하는 경우에만 아래 분기를 실행합니다.
        if status and status.get("thresholds"):
            # `st.caption(f"서버 적용값 · 버전 {status.get('threshold_version')} · {status.get('metric_type')}")`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption(f"서버 적용값 · 버전 {status.get('threshold_version')} · {status.get('metric_type')}")
            # `st.table([{'측면': '왼쪽' if side == 'left' else '오른쪽', '시작값': values['start_val'], '목표값': values['target_val'], '가동 범위': abs(values['target_val'] - values['start_val'])} fo…`을 호출하여 해당 단계의 작업을 수행합니다.
            st.table([{"측면": "왼쪽" if side == "left" else "오른쪽",
                       # 구성 항목: 시작값(`'시작값'`)에 계산식 `values['start_val']`의 결과; 목표값(`'목표값'`)에 계산식 `values['target_val']`의 결과를 지정합니다.
                       "시작값": values["start_val"], "목표값": values["target_val"],
                       # 구성 항목: 가동 범위(`'가동 범위'`)에 `abs(values['target_val'] - values['start_val'])` 호출 결과를 지정합니다.
                       "가동 범위": abs(values["target_val"] - values["start_val"])}
                      # 앞줄에서 시작한 식·인수·자료 구조를 `for side, values in status["thresholds"].items()])` 내용으로 이어서 구성합니다.
                      for side, values in status["thresholds"].items()])
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # `st.caption('서버 적용값 대기 중')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption("서버 적용값 대기 중")

    # Seed the editor once. Later polling doesn't rerun widgets or networking.
    # 조건 `status and st.session_state.get('rom_editor_session') != client.session_id`을 만족하는 경우에만 아래 분기를 실행합니다.
    if status and st.session_state.get("rom_editor_session") != client.session_id:
        # `st.rerun()`을 호출하여 해당 단계의 작업을 수행합니다.
        st.rerun()
    # 조건 `pending_before and (not st.session_state.get('pending_rom_request'))`을 만족하는 경우에만 아래 분기를 실행합니다.
    if pending_before and not st.session_state.get("pending_rom_request"):
        # `st.rerun()`을 호출하여 해당 단계의 작업을 수행합니다.
        st.rerun()
    # 조건 `st.session_state.get('is_running') != client.running`을 만족하는 경우에만 아래 분기를 실행합니다.
    if st.session_state.get("is_running") != client.running:
        # 여부 running에 `client.running` 값(`client.running`)을 저장합니다. 대상: `st.session_state.is_running`.
        st.session_state.is_running = client.running
        # 대기 가동 범위 request에 None을 저장합니다. 대상: `st.session_state.pending_rom_request`.
        st.session_state.pending_rom_request = None
        # `st.rerun()`을 호출하여 해당 단계의 작업을 수행합니다.
        st.rerun()


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def main():
    # `st.set_page_config(page_title='AI Rehabilitation Motion Studio', layout='wide')`을 호출하여 해당 단계의 작업을 수행합니다.
    st.set_page_config(page_title="AI Rehabilitation Motion Studio", layout="wide")
    # `st.title('AI Rehabilitation Motion System')`을 호출하여 해당 단계의 작업을 수행합니다.
    st.title("AI Rehabilitation Motion System")
    # `st.caption('세션을 시작하고, 측정 중 환자에게 맞게 좌우 가동 범위를 조정합니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
    st.caption("세션을 시작하고, 측정 중 환자에게 맞게 좌우 가동 범위를 조정합니다.")
    # `{'is_running': False, 'last_calib_thresholds': None, 'pending_rom_request': None, 'session_notice': None, 'rom_notice': None}.items()`에서 항목을 차례로 꺼내 `(key, value)`에 넣고 아래 처리를 반복합니다.
    for key, value in {"is_running": False, "last_calib_thresholds": None,
                       # 구성 항목: 대기 가동 범위 request(`'pending_rom_request'`)에 None; 측정 세션 notice(`'session_notice'`)에 None을 지정합니다.
                       "pending_rom_request": None, "session_notice": None,
                       # 구성 항목: 가동 범위 notice(`'rom_notice'`)에 None을 지정합니다.
                       "rom_notice": None}.items():
        # 조건 `key not in st.session_state`을 만족하는 경우에만 아래 분기를 실행합니다.
        if key not in st.session_state:
            # 측정 세션 state[key]에 측정값(`value`)을 저장합니다. 대상: `st.session_state[key]`.
            st.session_state[key] = value
    # 클라이언트에 `st.session_state.get('session_client')` 호출 결과를 저장합니다. 대상: `client`.
    client = st.session_state.get("session_client")
    # `running` 값에 `client`이면 `client.running`, 아니면 `False`을 저장합니다. 대상: `running`.
    running = client.running if client else False
    # 오버레이 job에 `st.session_state.get('overlay_job')` 호출 결과를 저장합니다. 대상: `overlay_job`.
    overlay_job = st.session_state.get('overlay_job')
    # 오버레이 running에 `bool(overlay_job and overlay_job.snapshot()['running'])` 호출 결과를 저장합니다. 대상: `overlay_running`.
    overlay_running = bool(overlay_job and overlay_job.snapshot()['running'])
    # 오버레이 ui running에 오버레이 running(`overlay_running`)을 저장합니다. 대상: `st.session_state.overlay_ui_running`.
    st.session_state.overlay_ui_running = overlay_running
    # 여부 running에 `running` 값(`running`)을 저장합니다. 대상: `st.session_state.is_running`.
    st.session_state.is_running = running
    # `(exercise_keys, display_map)` 값에 `load_exercise_config()` 호출 결과를 저장합니다. 대상: `(exercise_keys, display_map)`.
    exercise_keys, display_map = load_exercise_config()
    # 기본 영상 경로에 `os.path.join(get_root_dir(), 'data', 'test', 'biceps_curl', '0000', 'calibration.mp4')` 호출 결과를 저장합니다. 대상: `default_video_path`.
    default_video_path = os.path.join(get_root_dir(), "data", "test", "biceps_curl", "0000", "calibration.mp4")

    # `st.sidebar`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with st.sidebar:
        # `st.header('세션 설정')`을 호출하여 해당 단계의 작업을 수행합니다.
        st.header("세션 설정")
        # WebSocket 접속 주소에 `st.text_input('Socket Server', value='ws://127.0.0.1:8080', disabled=running)` 호출 결과를 저장합니다. 대상: `uri`.
        uri = st.text_input("Socket Server", value="ws://127.0.0.1:8080", disabled=running)
        # 화면 재시작 뒤 남은 연결은 사용자가 복구 버튼으로 명시적으로 종료합니다.
        if st.button("이전 서버 연결 정리", disabled=running or overlay_running):
            # 복구는 다른 창의 측정도 종료할 수 있으므로 버튼 옆에 용도를 안내합니다.
            try:
                # 응답을 받기 전까지 성공으로 표시하지 않습니다.
                response = asyncio.run(reset_server_session(uri))
                # AI 모델과 서버 프로세스는 그대로 유지됩니다.
                st.success(response["message"])
            # 서버 미실행·응답 지연·정리 실패를 화면에서 확인합니다.
            except Exception as exc:
                # 잘못된 주소·소켓 단절도 화면 전체 오류 대신 진단 로그와 안내로 처리합니다.
                logging.exception("Server session reset failed")
                st.error(f"서버 연결 정리 실패: {exc}")
        # 정상 측정 중인 다른 창이 있을 때는 복구하지 않도록 동작 범위를 표시합니다.
        st.caption("SERVER_BUSY가 계속될 때 사용하세요. 이전 측정 연결을 종료하고 로그를 저장합니다.")
        # 회귀 검증 시나리오 test_mode_choice에 `st.radio('검증 방식 선택', ['모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)', '모드 3: 오프라인 비디오 분석 (Video Batch 검증)'], index=0, disabled=running)` 호출 결과를 저장합니다. 대상: `test_mode_choice`.
        test_mode_choice = st.radio(
            # 앞줄에서 시작한 식·인수·자료 구조를 `"검증 방식 선택",` 내용으로 이어서 구성합니다.
            "검증 방식 선택",
            # 앞줄에서 시작한 식·인수·자료 구조를 `["모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)",` 내용으로 이어서 구성합니다.
            ["모드 2: 초경량 수치 전송 (Webcam + Unity 사전 검증)",
             # 앞줄에서 시작한 식·인수·자료 구조를 `"모드 3: 오프라인 비디오 분석 (Video Batch 검증)"],` 내용으로 이어서 구성합니다.
             "모드 3: 오프라인 비디오 분석 (Video Batch 검증)"],
            # 구성 항목: `index` 값(`index`)에 0; `disabled` 값(`disabled`)에 `running` 값(`running`)을 지정합니다.
            index=0, disabled=running)
        # 환자 식별 번호의 기존 필드명에 `st.text_input('Patient ID', value='0000', disabled=running)` 호출 결과를 저장합니다. 대상: `player_id`.
        player_id = st.text_input("Patient ID", value="0000", disabled=running)
        # Unity와 동일한 버전·등록 차수·플레이 ID를 입력하면 같은 환자 폴더로 연결됩니다.
        data_version = st.text_input("Data Version", value="v1.0.0(0)", disabled=running)
        registration = st.number_input("Registration", min_value=1, max_value=99, value=1, step=1, disabled=running)
        play_session_id = st.text_input("Unity Play Session ID", value="", disabled=running)
        exercise_record_id = st.text_input("Unity Exercise Record ID", value="", disabled=running)
        slot_number = st.number_input("Exercise Slot", min_value=0, value=1, step=1, disabled=running)
        st.caption("등록 차수는 Unity가 발급한 값입니다. 공통 ID를 비우면 단독 테스트용 ID를 생성합니다.")
        # `patient_name` 값에 `st.text_input('Patient Name', value='김지후', disabled=running)` 호출 결과를 저장합니다. 대상: `patient_name`.
        patient_name = st.text_input("Patient Name", value="김지후", disabled=running)
        # 세션 로그를 저장하는 최상위 폴더에 `st.text_input('로그 저장 폴더 (서버 기준)', value=str(default_log_root()), disabled=running, help='분석 결과를 저장할 폴더입니다. 영상 파일 경로는 Video File Path에 입력하세요.')` 호출 결과를 저장합니다. 대상: `log_root`.
        log_root = st.text_input("로그 저장 폴더 (서버 기준)", value=str(default_log_root()), disabled=running,
                                 # 구성 항목: `help` 값(`help`)에 '분석 결과를 저장할 폴더입니다. 영상 파일 경로는 Video File Path에 입력하세요.'을 지정합니다.
                                 help="분석 결과를 저장할 폴더입니다. 영상 파일 경로는 Video File Path에 입력하세요.")
        # `st.caption('환자 / 날짜 / 운동 폴더로 저장합니다. 기존 data 폴더는 이동하지 않습니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
        st.caption("환자 / 날짜 / 운동 폴더로 저장합니다. 기존 data 폴더는 이동하지 않습니다.")
        # `(session_column, round_column, set_column)` 값에 `st.columns(3)` 호출 결과를 저장합니다. 대상: `(session_column, round_column, set_column)`.
        session_column, round_column, set_column = st.columns(3)
        # 측정 세션 number에 `session_column.number_input('Session', min_value=0, value=1, step=1, disabled=running)` 호출 결과를 저장합니다. 대상: `session_number`.
        session_number = session_column.number_input("Session", min_value=0, value=1, step=1, disabled=running)
        # `round_number` 값에 `round_column.number_input('Round', min_value=0, value=1, step=1, disabled=running)` 호출 결과를 저장합니다. 대상: `round_number`.
        round_number = round_column.number_input("Round", min_value=0, value=1, step=1, disabled=running)
        # 설정 number에 `set_column.number_input('Set', min_value=0, value=1, step=1, disabled=running)` 호출 결과를 저장합니다. 대상: `set_number`.
        set_number = set_column.number_input("Set", min_value=0, value=1, step=1, disabled=running)
        # 운동 설정의 식별 이름에 `st.selectbox('Exercise Name', exercise_keys, format_func=lambda key: display_map.get(key, key), disabled=running)` 호출 결과를 저장합니다. 대상: `exercise_name`.
        exercise_name = st.selectbox("Exercise Name", exercise_keys,
                                    # 이 줄의 입력 인수: `key` = `key` 값. 기본값·형식은 아래 선언을 따릅니다.
                                    format_func=lambda key: display_map.get(key, key), disabled=running)
        # 실행 모드에 `st.radio('Execution Mode', ['CALIBRATION', 'TEST', 'MAIN'], index=1, disabled=running)` 호출 결과를 저장합니다. 대상: `mode`.
        mode = st.radio("Execution Mode", ["CALIBRATION", "TEST", "MAIN"], index=1, disabled=running)
        # 조건 `mode == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if mode == "CALIBRATION":
            # 조건 `exercise_name == 'knee_extension'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if exercise_name == 'knee_extension':
                # `st.info('무릎 펴기: 양발을 내린 채 2초 이상 정지한 뒤, 좌우 각각 3회 천천히 펴고 완전히 내려주세요. 움직임이 불안정하면 준비 자세를 계속 측정합니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
                st.info("무릎 펴기: 양발을 내린 채 2초 이상 정지한 뒤, 좌우 각각 3회 천천히 펴고 완전히 내려주세요. 움직임이 불안정하면 준비 자세를 계속 측정합니다.")
            # `st.info('양측 각각 3회로 개인 맞춤 가동 범위를 측정합니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.info("양측 각각 3회로 개인 맞춤 가동 범위를 측정합니다.")
            # 목표 반복 횟수에 3을 저장합니다. 대상: `target_reps`.
            target_reps = 3
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # 목표 반복 횟수에 `st.number_input('목표 점수 (Target Score)', min_value=0.25, max_value=1000.0, value=10.0, step=0.25, disabled=running)` 호출 결과를 저장합니다. 대상: `target_reps`.
            target_reps = st.number_input("목표 점수 (Target Score)", min_value=0.25, max_value=1000.0,
                                         # 구성 항목: 측정값(`value`)에 10.0; `step` 값(`step`)에 0.25; `disabled` 값(`disabled`)에 `running` 값(`running`)을 지정합니다.
                                         value=10.0, step=0.25, disabled=running)
            # `st.caption('Perfect +1점 · Good +0.5점 · Bad +0.25점 · 한쪽이 목표 초과 시 1.5초 후 종료')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption("Perfect +1점 · Good +0.5점 · Bad +0.25점 · 한쪽이 목표 초과 시 1.5초 후 종료")
        # chosen 원본에 None을 저장합니다. 대상: `chosen_source`.
        chosen_source = None
        # 한 번에 추론 또는 학습할 표본 수에 16을 저장합니다. 대상: `batch_size`.
        batch_size = 16
        # `camera_index` 값에 0을 저장합니다. 대상: `camera_index`.
        camera_index = 0
        # preparation 초에 기본 preparation 초(`DEFAULT_PREPARATION_SECONDS`)을 저장합니다. 대상: `preparation_seconds`.
        preparation_seconds = DEFAULT_PREPARATION_SECONDS
        # 출력 mode에 'LIGHTWEIGHT'을 저장합니다. 대상: `output_mode`.
        output_mode = "LIGHTWEIGHT"
        # 조건 `'모드 3' in test_mode_choice`을 만족하는 경우에만 아래 분기를 실행합니다.
        if "모드 3" in test_mode_choice:
            # 서버 PC에서 읽을 원본 영상 경로에 `st.text_input('Video File Path', value=default_video_path, disabled=running)` 호출 결과를 저장합니다. 대상: `video_path`.
            video_path = st.text_input("Video File Path", value=default_video_path, disabled=running)
            # chosen 원본에 `video_path`이면 `str(Path(video_path).resolve())`, 아니면 `None`을 저장합니다. 대상: `chosen_source`.
            chosen_source = str(Path(video_path).resolve()) if video_path else None
            # 한 번에 추론 또는 학습할 표본 수에 `st.select_slider('Batch Size', [8, 16, 32], value=16, disabled=running)` 호출 결과를 저장합니다. 대상: `batch_size`.
            batch_size = st.select_slider("Batch Size", [8, 16, 32], value=16, disabled=running)
            # `st.caption('오프라인 분석 중에는 가동 범위를 수정할 수 없습니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption("오프라인 분석 중에는 가동 범위를 수정할 수 없습니다.")
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # Camera ownership stays on the socket server across UI reruns.
            # `camera_choice` 값에 `st.selectbox('카메라 선택 (Camera)', options=[0, 1, 2, 3, 4, 5, 'manual'], key='camera_choice', disabled=running, format_func=lambda value: '직접 입력 (Custom Index)' if value ==…` 호출 결과를 저장합니다. 대상: `camera_choice`.
            camera_choice = st.selectbox(
                # 구성 항목: `options` 값(`options`)에 순서·구성을 지정한 `[0, 1, 2, 3, 4, 5, 'manual']`을 지정합니다.
                "카메라 선택 (Camera)", options=[0, 1, 2, 3, 4, 5, "manual"],
                # 구성 항목: `key` 값(`key`)에 'camera_choice'; `disabled` 값(`disabled`)에 `running` 값(`running`)을 지정합니다.
                key="camera_choice", disabled=running,
                # 이 줄의 입력 인수: `value` = 측정값. 기본값·형식은 아래 선언을 따릅니다.
                format_func=lambda value: (
                    # 앞줄에서 시작한 식·인수·자료 구조를 `"직접 입력 (Custom Index)" if value == "manual"` 내용으로 이어서 구성합니다.
                    "직접 입력 (Custom Index)" if value == "manual"
                    # 앞줄에서 시작한 식·인수·자료 구조를 `else f"카메라 {value} (Camera {value})"),` 내용으로 이어서 구성합니다.
                    else f"카메라 {value} (Camera {value})"),
                # 구성 항목: `help` 값(`help`)에 '서버 PC의 카메라 장치 번호를 선택하세요. 기본 번호는 0입니다.'을 지정합니다.
                help="서버 PC의 카메라 장치 번호를 선택하세요. 기본 번호는 0입니다.")
            # 조건 `camera_choice == 'manual'`을 만족하는 경우에만 아래 분기를 실행합니다.
            if camera_choice == "manual":
                # `camera_index` 값에 `st.number_input('카메라 번호 (Camera Index)', min_value=0, value=0, step=1, key='custom_camera_index', disabled=running)` 호출 결과를 저장합니다. 대상: `camera_index`.
                camera_index = st.number_input(
                    # 구성 항목: min 측정값(`min_value`)에 0; 측정값(`value`)에 0; `step` 값(`step`)에 1을 지정합니다.
                    "카메라 번호 (Camera Index)", min_value=0, value=0, step=1,
                    # 구성 항목: `key` 값(`key`)에 'custom_camera_index'; `disabled` 값(`disabled`)에 `running` 값(`running`)을 지정합니다.
                    key="custom_camera_index", disabled=running)
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # `camera_index` 값에 `camera_choice` 값(`camera_choice`)을 저장합니다. 대상: `camera_index`.
                camera_index = camera_choice
            # `st.caption('서버 PC의 카메라를 사용합니다. 연결된 장치에 따라 번호가 달라집니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption("서버 PC의 카메라를 사용합니다. 연결된 장치에 따라 번호가 달라집니다.")
            # preparation 초에 `st.number_input('측정 전 준비 시간 (초)', min_value=0.0, value=DEFAULT_PREPARATION_SECONDS, step=1.0, disabled=running, help='0으로 설정하면 준비 시간 없이 바로 측정합니다.')` 호출 결과를 저장합니다. 대상: `preparation_seconds`.
            preparation_seconds = st.number_input(
                # 구성 항목: min 측정값(`min_value`)에 0.0; 측정값(`value`)에 기본 preparation 초(`DEFAULT_PREPARATION_SECONDS`)을 지정합니다.
                "측정 전 준비 시간 (초)", min_value=0.0, value=DEFAULT_PREPARATION_SECONDS,
                # 구성 항목: `step` 값(`step`)에 1.0; `disabled` 값(`disabled`)에 `running` 값(`running`); `help` 값(`help`)에 '0으로 설정하면 준비 시간 없이 바로 측정합니다.'을 지정합니다.
                step=1.0, disabled=running, help="0으로 설정하면 준비 시간 없이 바로 측정합니다.")
        # `st.caption('영상은 소켓 서버에서 항상 MP4로 저장합니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
        st.caption("영상은 소켓 서버에서 항상 MP4로 저장합니다.")
        # 조건 `not running`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not running:
            # 조건 `st.button('세션 시작 (Start Session)', type='primary', disabled=overlay_running)`을 만족하는 경우에만 아래 분기를 실행합니다.
            if st.button("세션 시작 (Start Session)", type="primary", disabled=overlay_running):
                # 조건 `'모드 3' in test_mode_choice and (not chosen_source or not os.path.isfile(chosen_source))`을 만족하는 경우에만 아래 분기를 실행합니다.
                if "모드 3" in test_mode_choice and (not chosen_source or not os.path.isfile(chosen_source)):
                    # `st.error('분석할 영상 파일 경로를 확인하세요.')`을 호출하여 해당 단계의 작업을 수행합니다.
                    st.error("분석할 영상 파일 경로를 확인하세요.")
                # 조건 `log_root and Path(log_root).expanduser().is_file()`을 만족하는 경우에만 아래 분기를 실행합니다.
                elif log_root and Path(log_root).expanduser().is_file():
                    # `st.error('로그 저장 폴더에는 영상 파일이 아닌 폴더를 입력하세요. 영상 경로는 Video File Path에 입력하세요.')`을 호출하여 해당 단계의 작업을 수행합니다.
                    st.error("로그 저장 폴더에는 영상 파일이 아닌 폴더를 입력하세요. 영상 경로는 Video File Path에 입력하세요.")
                # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                else:
                    # `cfg` 값에 `dict(player_id=player_id, patient_name=patient_name, exercise_name=exercise_name, mode=mode, target_reps=target_reps, target_score=float(target_reps), finish_delay_secon…` 호출 결과를 저장합니다. 대상: `cfg`.
                    cfg = dict(player_id=player_id, patient_name=patient_name,
                               # 공통 플레이 식별 정보이며 기존 숫자 Session과는 별개입니다.
                               data_version=data_version, registration=int(registration), slot=int(slot_number),
                               play_session_id=play_session_id or None, exercise_record_id=exercise_record_id or None,
                               # 구성 항목: 운동 설정의 식별 이름(`exercise_name`)에 운동 설정의 식별 이름(`exercise_name`); 실행 모드(`mode`)에 실행 모드(`mode`); 목표 반복 횟수(`target_reps`)에 목표 반복 횟수(`target_reps`)을 지정합니다.
                               exercise_name=exercise_name, mode=mode, target_reps=target_reps,
                               # 구성 항목: 게임 목표 점수(`target_score`)에 `float(target_reps)` 호출 결과; 종료 delay 초(`finish_delay_seconds`)에 1.5을 지정합니다.
                               target_score=float(target_reps), finish_delay_seconds=1.5,
                               # 구성 항목: 품질 등급 points(`quality_points`)에 키 'PERFECT', 'GOOD', 'BAD'로 구성한 사전; `PERFECT` 값(`'PERFECT'`)에 1.0; `GOOD` 값(`'GOOD'`)에 0.5; `BAD` 값(`'BAD'`)에 0.25을 지정합니다.
                               quality_points={"PERFECT": 1.0, "GOOD": 0.5, "BAD": 0.25},
                               # 구성 항목: 출력 mode(`output_mode`)에 출력 mode(`output_mode`); `camera_index` 값(`camera_index`)에 `int(camera_index)` 호출 결과를 지정합니다.
                               output_mode=output_mode, camera_index=int(camera_index),
                               # 구성 항목: 서버 PC에서 읽을 원본 영상 경로(`video_path`)에 chosen 원본(`chosen_source`); 한 번에 추론 또는 학습할 표본 수(`batch_size`)에 한 번에 추론 또는 학습할 표본 수(`batch_size`); 세션 로그를 저장하는 최상위 폴더(`log_root`)에 세션 로그를 저장하는 최상위 폴더(`log_root`)을 지정합니다.
                               video_path=chosen_source, batch_size=batch_size, log_root=log_root,
                               # 구성 항목: 측정 세션(`session`)에 `int(session_number)` 호출 결과; `round` 값(`round`)에 `int(round_number)` 호출 결과; 설정(`set`)에 `int(set_number)` 호출 결과를 지정합니다.
                               session=int(session_number), round=int(round_number), set=int(set_number),
                               # 구성 항목: `cal` 값(`cal`)에 계산식 `mode == 'CALIBRATION'`의 결과; 영상 녹화 여부(`record_video`)에 True을 지정합니다.
                               cal=mode == "CALIBRATION", record_video=True,
                               # 구성 항목: preparation 초(`preparation_seconds`)에 `float(preparation_seconds)` 호출 결과를 지정합니다.
                               preparation_seconds=float(preparation_seconds))
                    # 클라이언트에 `StreamlitSessionClient(uri, cfg, ui_heartbeat_timeout=10.0)` 호출 결과를 저장합니다. 대상: `client`.
                    client = StreamlitSessionClient(uri, cfg, ui_heartbeat_timeout=10.0)
                    # 측정 세션 클라이언트에 클라이언트(`client`)을 저장합니다. 대상: `st.session_state.session_client`.
                    st.session_state.session_client = client
                    # 측정 세션 notice에 None을 저장합니다. 대상: `st.session_state.session_notice`.
                    st.session_state.session_notice = None
                    # 가동 범위 notice에 None을 저장합니다. 대상: `st.session_state.rom_notice`.
                    st.session_state.rom_notice = None
                    # 대기 가동 범위 request에 None을 저장합니다. 대상: `st.session_state.pending_rom_request`.
                    st.session_state.pending_rom_request = None
                    # `client.start()`을 호출하여 해당 단계의 작업을 수행합니다.
                    client.start()
                    # `st.rerun()`을 호출하여 해당 단계의 작업을 수행합니다.
                    st.rerun()
        # 조건 `st.button('세션 중지 (Stop Session)')`을 만족하는 경우에만 아래 분기를 실행합니다.
        elif st.button("세션 중지 (Stop Session)"):
            # `client.stop()`을 호출하여 해당 단계의 작업을 수행합니다.
            client.stop()
            # 여부 running에 `client.running` 값(`client.running`)을 저장합니다. 대상: `st.session_state.is_running`.
            st.session_state.is_running = client.running
            # 대기 가동 범위 request에 None을 저장합니다. 대상: `st.session_state.pending_rom_request`.
            st.session_state.pending_rom_request = None
            # 측정 세션 notice에 `client.snapshot()['status'] == 'ERROR'`이면 `('error', '종료 응답을 확인하지 못했습니다. 서버의 기록 저장 상태를 확인하세요.')`, 아니면 `('info', '세션을 중지하고 기록 저장을 완료했습니다.')`을 저장합니다. 대상: `st.session_state.session_notice`.
            st.session_state.session_notice = (
                # 앞줄에서 시작한 식·인수·자료 구조를 `("error", "종료 응답을 확인하지 못했습니다. 서버의 기록 저장 상태를 확인하세요.")` 내용으로 이어서 구성합니다.
                ("error", "종료 응답을 확인하지 못했습니다. 서버의 기록 저장 상태를 확인하세요.")
                # 앞줄에서 시작한 식·인수·자료 구조를 `if client.snapshot()["status"] == "ERROR"` 내용으로 이어서 구성합니다.
                if client.snapshot()["status"] == "ERROR"
                # 앞줄에서 시작한 식·인수·자료 구조를 `else ("info", "세션을 중지하고 기록 저장을 완료했습니다."))` 내용으로 이어서 구성합니다.
                else ("info", "세션을 중지하고 기록 저장을 완료했습니다."))
            # `st.rerun()`을 호출하여 해당 단계의 작업을 수행합니다.
            st.rerun()
        # 조건 `st.session_state.last_calib_thresholds`을 만족하는 경우에만 아래 분기를 실행합니다.
        if st.session_state.last_calib_thresholds:
            # `st.caption('최근 캘리브레이션 결과')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.caption("최근 캘리브레이션 결과")
            # `st.json(st.session_state.last_calib_thresholds)`을 호출하여 해당 단계의 작업을 수행합니다.
            st.json(st.session_state.last_calib_thresholds)

    # `(live_column, controls_column)` 값에 `st.columns([3, 2])` 호출 결과를 저장합니다. 대상: `(live_column, controls_column)`.
    live_column, controls_column = st.columns([3, 2])
    # `live_column`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with live_column:
        # `alert_box` 값에 `st.empty()` 호출 결과를 저장합니다. 대상: `alert_box`.
        alert_box = st.empty()
        # 영상 box에 `st.empty()` 호출 결과를 저장합니다. 대상: `video_box`.
        video_box = st.empty()
        # `metrics_box` 값에 `st.empty()` 호출 결과를 저장합니다. 대상: `metrics_box`.
        metrics_box = st.empty()
        # 초당 프레임 수 box에 `st.empty()` 호출 결과를 저장합니다. 대상: `fps_box`.
        fps_box = st.empty()
        # 로그 paths box에 `st.empty()` 호출 결과를 저장합니다. 대상: `log_paths_box`.
        log_paths_box = st.empty()
        # 조건 `not client`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not client:
            # `alert_box.info('소켓 서버 실행 후 세션을 시작하세요.')`을 호출하여 해당 단계의 작업을 수행합니다.
            alert_box.info("소켓 서버 실행 후 세션을 시작하세요.")
    # `controls_column`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with controls_column:
        # `st.subheader('가동 범위 조정')`을 호출하여 해당 단계의 작업을 수행합니다.
        st.subheader("가동 범위 조정")
        # `applied_box` 값에 `st.empty()` 호출 결과를 저장합니다. 대상: `applied_box`.
        applied_box = st.empty()
        # `feedback_box` 값에 `st.empty()` 호출 결과를 저장합니다. 대상: `feedback_box`.
        feedback_box = st.empty()
        # 조건 `client`을 만족하는 경우에만 아래 분기를 실행합니다.
        if client:
            # `render_rom_editor(client, client.snapshot()['threshold_status'])`을 호출하여 해당 단계의 작업을 수행합니다.
            render_rom_editor(client, client.snapshot()["threshold_status"])
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # `st.info('실시간 TEST 또는 MAIN 세션에서 시작값과 목표값을 수정할 수 있습니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.info("실시간 TEST 또는 MAIN 세션에서 시작값과 목표값을 수정할 수 있습니다.")
    # 조건 `client`을 만족하는 경우에만 아래 분기를 실행합니다.
    if client:
        # `refresh_session(client, {'alert': alert_box, 'video': video_box, 'metrics': metrics_box, 'fps': fps_box, 'applied_rom': applied_box, 'rom_feedback': feedback_box, 'log_p…`을 호출하여 해당 단계의 작업을 수행합니다.
        refresh_session(client, {
            # 구성 항목: `alert` 값(`'alert'`)에 `alert_box` 값(`alert_box`); 영상(`'video'`)에 영상 box(`video_box`); `metrics` 값(`'metrics'`)에 `metrics_box` 값(`metrics_box`); 영상의 초당 프레임 수(`'fps'`)에 초당 프레임 수 box(`fps_box`)을 지정합니다.
            "alert": alert_box, "video": video_box, "metrics": metrics_box, "fps": fps_box,
            # 구성 항목: applied 가동 범위(`'applied_rom'`)에 `applied_box` 값(`applied_box`); 가동 범위 feedback(`'rom_feedback'`)에 `feedback_box` 값(`feedback_box`)을 지정합니다.
            "applied_rom": applied_box, "rom_feedback": feedback_box,
            # 구성 항목: 로그 paths(`'log_paths'`)에 로그 paths box(`log_paths_box`)을 지정합니다.
            "log_paths": log_paths_box,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        })

    # `st.divider()`을 호출하여 해당 단계의 작업을 수행합니다.
    st.divider()
    # `st.subheader('저장 영상 일괄 오버레이')`을 호출하여 해당 단계의 작업을 수행합니다.
    st.subheader('저장 영상 일괄 오버레이')
    # 오버레이 최상위 경로에 `st.text_input('오버레이 영상 폴더', value=str(Path(get_root_dir()) / 'data'), disabled=overlay_running)` 호출 결과를 저장합니다. 대상: `overlay_root`.
    overlay_root = st.text_input('오버레이 영상 폴더', value=str(default_log_root()),
                                 # 구성 항목: `disabled` 값(`disabled`)에 오버레이 running(`overlay_running`)을 지정합니다.
                                 disabled=overlay_running)
    # analyze 결측에 `st.checkbox('로그 없는 영상도 소켓 서버에서 분석', value=True, disabled=overlay_running)` 호출 결과를 저장합니다. 대상: `analyze_missing`.
    analyze_missing = st.checkbox('로그 없는 영상도 소켓 서버에서 분석', value=True, disabled=overlay_running)
    # `st.caption('하위 폴더의 영상을 처리합니다. CSV/NPZ가 있으면 재사용하고, 없으면 운동 이름을 폴더·파일명에서 찾아 TEST 모드의 기본 가동 범위로 분석합니다. 원본은 보존하며 결과는 _overlays에 저장합니다.')`을 호출하여 해당 단계의 작업을 수행합니다.
    st.caption('하위 폴더의 영상을 처리합니다. CSV/NPZ가 있으면 재사용하고, 없으면 운동 이름을 폴더·파일명에서 찾아 TEST 모드의 기본 가동 범위로 분석합니다. 원본은 보존하며 결과는 _overlays에 저장합니다.')
    # 조건 `st.button('data 영상 전체 오버레이 생성', disabled=running or overlay_running)`을 만족하는 경우에만 아래 분기를 실행합니다.
    if st.button('data 영상 전체 오버레이 생성', disabled=running or overlay_running):
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 오버레이 job에 `VideoOverlayJob(overlay_root, uri, analyze_missing=analyze_missing)` 호출 결과를 저장합니다. 대상: `overlay_job`.
            overlay_job = VideoOverlayJob(overlay_root, uri, analyze_missing=analyze_missing)
            # `overlay_job.start()`을 호출하여 해당 단계의 작업을 수행합니다.
            overlay_job.start()
            # 오버레이 job에 오버레이 job(`overlay_job`)을 저장합니다. 대상: `st.session_state.overlay_job`.
            st.session_state.overlay_job = overlay_job
            # `st.rerun()`을 호출하여 해당 단계의 작업을 수행합니다.
            st.rerun()
        # `(ValueError, RuntimeError)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except (ValueError, RuntimeError) as exc:
            # `st.error(str(exc))`을 호출하여 해당 단계의 작업을 수행합니다.
            st.error(str(exc))
    # 조건 `overlay_job`을 만족하는 경우에만 아래 분기를 실행합니다.
    if overlay_job:
        # `refresh_overlay_job(overlay_job)`을 호출하여 해당 단계의 작업을 수행합니다.
        refresh_overlay_job(overlay_job)


# 데코레이터 `st.fragment(run_every=0.5)`를 바로 아래 함수·클래스에 적용합니다.
@st.fragment(run_every=0.5)
# 표시 갱신 오버레이 job 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def refresh_overlay_job(job):
    # 처리 단계에 `job.snapshot()` 호출 결과를 저장합니다. 대상: `state`.
    state = job.snapshot()
    # 조건 `st.session_state.get('overlay_ui_running') and (not state['running'])`을 만족하는 경우에만 아래 분기를 실행합니다.
    if st.session_state.get('overlay_ui_running') and not state['running']:
        # 오버레이 ui running에 False을 저장합니다. 대상: `st.session_state.overlay_ui_running`.
        st.session_state.overlay_ui_running = False
        # `st.rerun()`을 호출하여 해당 단계의 작업을 수행합니다.
        st.rerun()
    # `st.progress(state['completed'] / max(1, state['total']), text=f"영상 {state['completed']}/{state['total']} · {state['phase']}")`을 호출하여 해당 단계의 작업을 수행합니다.
    st.progress(state['completed'] / max(1, state['total']),
                # 구성 항목: `text` 값(`text`)에 계산식 `f"영상 {state['completed']}/{state['total']} · {state['phase']}"`의 결과를 지정합니다.
                text=f"영상 {state['completed']}/{state['total']} · {state['phase']}")
    # 조건 `state.get('current')`을 만족하는 경우에만 아래 분기를 실행합니다.
    if state.get('current'):
        # `st.caption(state['current'])`을 호출하여 해당 단계의 작업을 수행합니다.
        st.caption(state['current'])
    # 조건 `state.get('frames')`을 만족하는 경우에만 아래 분기를 실행합니다.
    if state.get('frames'):
        # `st.caption(f"현재 영상 프레임 {state['frame']}/{state['frames']}")`을 호출하여 해당 단계의 작업을 수행합니다.
        st.caption(f"현재 영상 프레임 {state['frame']}/{state['frames']}")
    # 조건 `state['running']`을 만족하는 경우에만 아래 분기를 실행합니다.
    if state['running']:
        # 조건 `st.button('오버레이 작업 중지')`을 만족하는 경우에만 아래 분기를 실행합니다.
        if st.button('오버레이 작업 중지'):
            # `job.cancel.set()`을 호출하여 스레드 신호를 활성화합니다.
            job.cancel.set()
    # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
    else:
        # 조건 `state.get('error')`을 만족하는 경우에만 아래 분기를 실행합니다.
        if state.get('error'):
            # `st.error(state['error'])`을 호출하여 해당 단계의 작업을 수행합니다.
            st.error(state['error'])
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # `st.info('오버레이 작업이 종료되었습니다. 영상별 성공·건너뜀·오류를 확인하세요.')`을 호출하여 해당 단계의 작업을 수행합니다.
            st.info('오버레이 작업이 종료되었습니다. 영상별 성공·건너뜀·오류를 확인하세요.')
        # `st.code(state['output'], language=None)`을 호출하여 해당 단계의 작업을 수행합니다.
        st.code(state['output'], language=None)
    # 조건 `state['results']`을 만족하는 경우에만 아래 분기를 실행합니다.
    if state['results']:
        # `st.dataframe(state['results'], use_container_width=True, hide_index=True)`을 호출하여 해당 단계의 작업을 수행합니다.
        st.dataframe(state['results'], use_container_width=True, hide_index=True)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == "__main__":
    # `main()`을 호출하여 해당 단계의 작업을 수행합니다.
    main()
