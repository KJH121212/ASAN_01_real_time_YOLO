# ==============================================================================
# [파일 정보]
# 파일명: app.py (포트 충돌 방지 패치 버전)
# 작성자: 개발자 (Developer)
# 설명: 8080 포트 활성화 여부를 감지하여 소켓 서버 중복 실행을 원천 차단하는 Streamlit 앱
# ==============================================================================

# ------------------------------------------------------------------------------
# [코드 설명]
# socket.connect_ex()를 사용하여 8080 포트가 이미 열려있는지 먼저 검사합니다.
# 포트가 이미 활성화되어 있다면 백그라운드 subprocess 실행을 건너뛰고 기존 서버에 접속하며,
# 닫혀있을 때만 socket_server.py를 새롭게 가동하여 Errno 10048 오류를 방지합니다.
# ------------------------------------------------------------------------------

import asyncio  # 비동기 통신 코루틴 제어를 위한 내장 모듈 로드
import base64  # Base64 비디오 프레임 디코딩 모듈 로드
import json  # JSON 패킷 데이터 파싱 모듈 로드
import os  # 시스템 파일 경로 탐색 모듈 로드
import socket  # 포트 활성화 여부를 점검하기 위한 저수준 소켓 모듈 로드
import subprocess  # 백엔드 서버 프로세스 제어를 위한 서브프로세스 모듈 로드
import sys  # 파이썬 인터프리터 경로 획득 모듈 로드
import time  # 시간 지연 및 타임스탬프 계산 모듈 로드
import cv2  # 영상 바이트를 이미지 행렬로 변환하기 위한 OpenCV 로드
import numpy as np  # 배열 연산 처리를 위한 NumPy 라이브러리 로드
import streamlit as st  # 웹 UI 구축을 위한 Streamlit 라이브러리 로드
import websockets  # 비동기 WebSocket 클라이언트 통신 모듈 로드

from utils.overlay_renderer import OverlayRenderer  # 뼈대 렌더러 로드


def get_root_dir() -> str:  # 프로젝트 루트 디렉터리 경로 반환 함수 정의
    return os.path.dirname(os.path.abspath(__file__))  # 현재 파일 기준 부모 디렉터리 절대 경로 반환


def is_port_in_use(port: int = 8080, host: str = "127.0.0.1") -> bool:  # 특정 포트가 이미 사용 중인지 검사하는 함수 정의
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:  # TCP 소켓 객체 생성
        return s.connect_ex((host, port)) == 0  # 0을 반환하면 포트가 이미 열려있음(True), 아니면 닫혀있음(False) 반환


def ensure_socket_server(root_dir: str):  # 소켓 서버 중복 실행 방지 구동 함수 정의
    if is_port_in_use(8080):  # 이미 8080 포트에서 서버가 돌고 있는 경우
        return  # 새 프로세스를 띄우지 않고 기존 구동 중인 서버를 그대로 활용

    if "server_process" not in st.session_state or st.session_state.server_process is None:  # 세션 내 프로세스가 없는 경우
        server_script = os.path.join(root_dir, "network", "socket_server.py")  # 소켓 서버 스크립트 경로 조립
        process = subprocess.Popen([sys.executable, server_script])  # 독립 서브프로세스로 서버 비동기 실행
        st.session_state.server_process = process  # 세션에 프로세스 핸들 저장
        time.sleep(2.5)  # AI 모델 로딩 및 소켓 포트 바인딩 완료까지 안전 대기


def toggle_session():  # 세션 시작/중지 상태를 반전시키는 콜백 함수 정의
    st.session_state.is_running = not st.session_state.is_running  # 실행 상태 플래그 반전


async def run_unity_client_session(uri: str, config: dict, placeholders: dict, renderer: OverlayRenderer):  # Unity 통신 루프 함수 정의
    try:  # 소켓 연결 예외 처리 블록
        async with websockets.connect(uri) as ws:  # 소켓 서버 연결 수립
            cmd_packet = {  # 초기화 제어 패킷 생성
                "type": "CMD_SET_SESSION",  # 패킷 타입 지정
                "player_id": config["player_id"],  # 환자 ID 탑재
                "patient_name": config["patient_name"],  # 환자 이름 탑재
                "exercise_name": config["exercise_name"],  # 운동명 탑재
                "mode": config["mode"],  # 모드 탑재
                "target_reps": config["target_reps"],  # 목표 횟수 탑재
                "cal_time": config["cal_time"],  # 캘리브레이션 시간 탑재
                "camera_index": config["camera_index"],  # 웹캠 인덱스 번호 탑재
            }  # 제어 패킷 조립 완료
            await ws.send(json.dumps(cmd_packet))  # 서버에 초기화 제어 패킷 전송

            while st.session_state.get("is_running", False):  # 활성 플래그가 True인 동안 루프 가동
                try:  # 패킷 수신 대기 블록
                    raw_res = await asyncio.wait_for(ws.recv(), timeout=0.03)  # 30ms 제한시간으로 패킷 수신
                    data = json.loads(raw_res)  # 수신 JSON 역직렬화
                    
                    if data.get("type") in ["SESSION_FINISHED", "CALIBRATION_FINISHED"]:  # 세션 완료 신호 수신 시
                        placeholders["alert"].success(f"세션이 정상적으로 완료되었습니다: {data.get('type')}")  # 완료 메시지 표출
                        st.session_state.is_running = False  # 실행 플래그 종료
                        break  # 루프 탈출

                    if data.get("type") == "POSE_UPDATE":  # 실시간 포즈 데이터 처리
                        img_b64 = data.get("frame_b64")  # Base64 영상 데이터 추출
                        canvas = np.zeros((480, 640, 3), dtype=np.uint8)  # 기본 캔버스 도화지 생성
                        
                        if img_b64:  # 영상 바이트가 있는 경우
                            img_bytes = base64.b64decode(img_b64)  # 디코딩
                            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)  # 이미지 복원
                            if frame is not None:  # 복원 성공 시
                                canvas = cv2.resize(frame, (640, 480))  # 캔버스 해상도 맞춤 리사이징

                        keypoints = data.get("keypoints", [])  # 관절 좌표 리스트 추출
                        is_occluded = data.get("is_occluded", False)  # 가림 여부 플래그 추출
                        current_mode = data.get("mode", config["mode"])  # 현재 동작 모드 추출

                        canvas = renderer.draw_skeleton(  # 해상도 자동 동기화 뼈대 렌더링 함수 호출
                            canvas=canvas,  # 작업 도화지 전달
                            keypoints=keypoints,  # 관절 좌표 전달
                            is_occluded=is_occluded  # 가림 상태 전달
                        )  # 뼈대 오버레이 완료

                        rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # RGB 변환
                        placeholders["video"].image(rgb_canvas, channels="RGB")  # 화면 출력 갱신

                        if current_mode == "CALIBRATION":  # 캘리브레이션 세션 UI 처리
                            calib_step = data.get("calib_step", "")  # 진행 세부 단계 추출
                            rem_sec = data.get("remaining_sec", 0.0)  # 남은 시간 추출
                            
                            if is_occluded:  # 전신 가림 시
                                placeholders["alert"].error("경고: 전신 가림 감지. 화면 중앙으로 들어오면 다시 시작합니다.")  # 경고 출력
                            elif calib_step == "FULL_BODY_CHECK":  # 전신 대기 상태
                                placeholders["alert"].info("전신 확인 중: 카메라를 정면으로 바라보고 서주세요.")  # 대기 안내 출력
                            elif calib_step == "COUNTDOWN":  # 카운트다운 상태
                                placeholders["alert"].warning(f"측정 준비: {int(np.ceil(rem_sec))}초 후 수집이 시작됩니다.")  # 카운트다운 출력
                            elif calib_step == "COLLECTING":  # 데이터 수집 상태
                                placeholders["alert"].success(f"동작 수집 중: {rem_sec:.1f}초 남음 (자유롭게 움직여주세요)")  # 수집 상태 출력
                        else:  # 메인 운동 세션 UI 처리
                            if is_occluded:  # 가림 발생 시
                                placeholders["alert"].error("신체 일부 가림 감지: 정확한 측정을 위해 전신을 보여주세요. (카운트 동결)")  # 경고 출력
                            else:  # 정상 시
                                placeholders["alert"].empty()  # 경고 제거

                            left_info = data.get("left", {})  # 좌측 FSM 데이터 추출
                            right_info = data.get("right", {})  # 우측 FSM 데이터 추출

                            placeholders["left_metric"].metric(  # 좌측 회차 및 품질 갱신
                                label="좌측 운동 회차 (Left Reps)",  # 레이블
                                value=f"{left_info.get('rep_count', 0)} / {config['target_reps']} 회",  # 회차 수치
                                delta=f"품질: {left_info.get('quality', '-')}"  # 품질 등급
                            )  # 좌측 카드 갱신

                            placeholders["right_metric"].metric(  # 우측 회차 및 품질 갱신
                                label="우측 운동 회차 (Right Reps)",  # 레이블
                                value=f"{right_info.get('rep_count', 0)} / {config['target_reps']} 회",  # 회차 수치
                                delta=f"품질: {right_info.get('quality', '-')}"  # 품질 등급
                            )  # 우측 카드 갱신

                            l_ratio = min(max(float(left_info.get("progress_ratio", 0.0)), 0.0), 1.0)  # 좌측 비율 클램핑
                            r_ratio = min(max(float(right_info.get("progress_ratio", 0.0)), 0.0), 1.0)  # 우측 비율 클램핑

                            placeholders["left_progress"].progress(l_ratio, text=f"좌측 ROM 가동률: {int(l_ratio * 100)}%")  # 좌측 바 갱신
                            placeholders["right_progress"].progress(r_ratio, text=f"우측 ROM 가동률: {int(r_ratio * 100)}%")  # 우측 바 갱신

                        placeholders["fps"].caption(f"통신 상태: 정상 | FPS: {data.get('fps', 0.0):.1f} | 모드: {current_mode}")  # 통신 캡션 갱신

                except asyncio.TimeoutError:  # 수신 타임아웃 시
                    pass  # 다음 루프로 통과

    except Exception as e:  # 네트워크 오류 발생 시
        placeholders["alert"].error(f"소켓 서버 연결 실패: {e}")  # 화면 에러 표시
        st.session_state.is_running = False  # 실행 플래그 해제


def main():  # Streamlit 메인 뷰포트 레이아웃 함수 선언
    st.set_page_config(page_title="AI Rehab Unity Simulator", layout="wide")  # 페이지 기본 설정
    st.title("AI 재활 운동 유니티(Unity) 프론트엔드 시뮬레이터")  # 메인 타이틀
    st.markdown("---")  # 구분선 출력

    root_dir = get_root_dir()  # 루트 디렉터리 경로 획득
    renderer = OverlayRenderer(conf_threshold=0.35)  # 렌더러 인스턴스 생성

    if "is_running" not in st.session_state:  # 세션 실행 상태 초기화 검사
        st.session_state.is_running = False  # 기본값 False 지정

    with st.sidebar:  # 사이드바 설정 영역
        st.header("Unity 제어 콘솔")  # 사이드바 제목
        
        player_id = st.text_input("환자 고유 ID", value="patient_1")  # 환자 ID
        patient_name = st.text_input("환자 이름", value="김지후")  # 환자 이름
        exercise_name = st.selectbox("운동 종목 선택", ["biceps_curl", "shoulder_press", "knee_extension"], index=0)  # 운동 종목
        mode = st.radio("실행 모드 (Mode)", ["CALIBRATION", "MAIN"], index=0)  # 모드 라디오 버튼
        target_reps = st.number_input("목표 반복 횟수 (MAIN 전용)", min_value=1, max_value=50, value=5, step=1)  # 목표 횟수
        cal_time = st.slider("캘리브레이션 수집 시간(초)", min_value=3.0, max_value=15.0, value=5.0, step=1.0)  # 캘리브레이션 시간
        camera_index = st.number_input("카메라 인덱스 번호", min_value=0, max_value=5, value=0, step=1)  # 웹캠 인덱스 번호
        
        st.markdown("---")  # 구분선
        
        if not st.session_state.is_running:  # 세션 정지 상태일 때
            st.button("세션 시작 (Start Session)", type="primary", on_click=toggle_session)  # 단일 클릭 시작 버튼
        else:  # 세션 실행 상태일 때
            st.button("세션 중지 (Stop Session)", on_click=toggle_session)  # 단일 클릭 중지 버튼

    col_view, col_stats = st.columns([3, 2])  # 3:2 분할 레이아웃 생성

    with col_view:  # 좌측 뷰포트
        st.subheader("Unity 아바타 / 스켈레톤 뷰포트")  # 서브헤더
        alert_box = st.empty()  # 알림 컨테이너
        video_box = st.empty()  # 영상 컨테이너
        fps_box = st.empty()  # FPS 컨테이너

    with col_stats:  # 우측 HUD
        st.subheader("실시간 재활 운동 HUD")  # 서브헤더
        col_m1, col_m2 = st.columns(2)  # 2단 컬럼
        with col_m1:  # 좌측 메트릭
            left_metric_box = st.empty()  # 좌측 컨테이너
        with col_m2:  # 우측 메트릭
            right_metric_box = st.empty()  # 우측 컨테이너
            
        st.markdown("##### 실시간 관절 가동 범위 (ROM)")  # 진행률 헤딩
        left_prog_box = st.empty()  # 좌측 바
        right_prog_box = st.empty()  # 우측 바

    placeholders = {  # 컨테이너 딕셔너리 구성
        "video": video_box,  # 영상
        "alert": alert_box,  # 알림
        "fps": fps_box,  # FPS
        "left_metric": left_metric_box,  # 좌측 수치
        "right_metric": right_metric_box,  # 우측 수치
        "left_progress": left_prog_box,  # 좌측 프로그레스
        "right_progress": right_prog_box,  # 우측 프로그레스
    }  # 딕셔너리 조립 완료

    if st.session_state.is_running:  # 실행 상태일 때
        ensure_socket_server(root_dir)  # 포트 사용 여부 점검 후 안전하게 서버 확인
        
        cfg = {  # 설정 페이로드 번들
            "player_id": player_id,  # 환자 ID
            "patient_name": patient_name,  # 환자명
            "exercise_name": exercise_name,  # 종목명
            "mode": mode,  # 모드
            "target_reps": int(target_reps),  # 회차
            "cal_time": float(cal_time),  # 시간
            "camera_index": int(camera_index),  # 카메라 인덱스
        }  # 번들 완료
        
        asyncio.run(run_unity_client_session("ws://127.0.0.1:8080", cfg, placeholders, renderer))  # 비동기 루프 가동

if __name__ == "__main__":  # 스크립트 직접 실행 판별
    main()  # 메인 함수 실행