# 한국어 코드 해설: 소켓 서버에 연결하는 수동 테스트 화면 또는 명령을 제공합니다. 파일명에 test가 있지만 운영 연동을 확인하는 클라이언트입니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [파일 정보]
# 파일명: network/test_client.py
# 작성자: 개발자 (Developer)
# 설명: WebSocket 기반 실시간 영상 수신 및 모드별 맞춤 UI 렌더링 클라이언트 (파라미터 보완)
# ==============================================================================

# 이 파일에서 사용할 `argparse` 모듈을 불러옵니다.
import argparse  # 커맨드 라인 인자 파싱을 위한 내장 모듈 로드
# 이 파일에서 사용할 `asyncio` 모듈을 불러옵니다.
import asyncio  # 비동기 통신 코루틴 제어를 위한 내장 모듈 로드
# 이 파일에서 사용할 `base64` 모듈을 불러옵니다.
import base64  # 서버로부터 수신된 압축 이미지 텍스트 디코딩을 위한 모듈 로드
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json  # 웹소켓 패킷 데이터 파싱을 위한 JSON 모듈 로드
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys  # 시스템 경로 제어를 위한 내장 모듈 로드
# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os  # 운영체제 환경 및 파일 존재 여부 검사를 위한 모듈 로드
# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2  # 수신된 이미지 복원 및 화면 UI 출력을 위한 OpenCV 라이브러리 로드
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np  # 바이너리 버퍼 배열 조작을 위한 NumPy 라이브러리 로드
# 이 파일에서 사용할 `websockets` 모듈을 불러옵니다.
import websockets  # 비동기 소켓 클라이언트 연결을 위한 외부 라이브러리 로드

# `current_dir` 값에 `os.path.dirname(os.path.abspath(__file__))` 호출 결과를 저장합니다. 대상: `current_dir`.
current_dir = os.path.dirname(os.path.abspath(__file__))  # 현재 스크립트가 위치한 디렉터리 경로 추출
# 최상위 경로 dir에 `os.path.dirname(current_dir)` 호출 결과를 저장합니다. 대상: `root_dir`.
root_dir = os.path.dirname(current_dir)  # 부모 디렉터리인 최상위 프로젝트 경로 추출
# 조건 `root_dir not in sys.path`을 만족하는 경우에만 아래 분기를 실행합니다.
if root_dir not in sys.path:  # 파이썬 모듈 탐색 경로에 루트 폴더가 없다면
    # `sys.path.insert(0, root_dir)`을 호출하여 해당 단계의 작업을 수행합니다.
    sys.path.insert(0, root_dir)  # 타 계층 파일 임포트를 위해 경로 최우선 삽입 처리

# `utils.overlay_renderer`에서 `OverlayRenderer` 기능을 가져옵니다.
from utils.overlay_renderer import OverlayRenderer  # 뼈대 시각화 유틸리티 클래스 수입


# 조회 품질 등급 color 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def get_quality_color(quality_str: str) -> tuple:  # 운동 수행 품질 문자열에 대응하는 BGR 색상을 반환하는 함수
    # 조건 `quality_str == 'PERFECT'`을 만족하는 경우에만 아래 분기를 실행합니다.
    if quality_str == "PERFECT": return (0, 255, 0)  # 완벽할 땐 초록색 반환
    # 조건 `quality_str == 'GOOD'`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif quality_str == "GOOD": return (0, 255, 255)  # 양호할 땐 노란색 반환
    # 조건 `quality_str == 'BAD'`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif quality_str == "BAD": return (0, 0, 255)  # 미흡할 땐 빨간색 반환
    # 조건 `quality_str == 'CALIBRATING'`을 만족하는 경우에만 아래 분기를 실행합니다.
    elif quality_str == "CALIBRATING": return (255, 200, 0)  # 측정 중일 땐 주황색 반환
    # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(180, 180, 180)`을 호출한 쪽으로 반환합니다.
    return (180, 180, 180)  # 무효이거나 기본값일 땐 회색 반환


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
async def main():  # 클라이언트 메인 비동기 루프 함수
    # `parser` 값에 `argparse.ArgumentParser()` 호출 결과를 저장합니다. 대상: `parser`.
    parser = argparse.ArgumentParser()  # 인자 파서 객체 생성
    # `parser.add_argument('--player_id', type=str, default='patient_1')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--player_id", type=str, default="patient_1")  # 테스트용 환자 고유 번호 인자 등록
    # `parser.add_argument('--patient_name', type=str, default='김지후')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--patient_name", type=str, default="김지후")  # 화면에 표시할 환자 이름 인자 등록
    # `parser.add_argument('--exercise_name', type=str, default='biceps_curl')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--exercise_name", type=str, default="biceps_curl")  # 수행할 운동 종목 인자 등록
    # `parser.add_argument('--mode', type=str, default='CALIBRATION')`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--mode", type=str, default="CALIBRATION")  # 실행 목적(캘리브레이션/메인) 인자 등록
    # `parser.add_argument('--target_reps', type=int, default=5)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--target_reps", type=int, default=5)  # 목표 반복 횟수 인자 등록
    # `parser.add_argument('--cal_time', type=float, default=5.0)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument("--cal_time", type=float, default=5.0)  # 캘리브레이션 수집 시간 인자 추가 등록
    # `args` 값에 `parser.parse_args()` 호출 결과를 저장합니다. 대상: `args`.
    args = parser.parse_args()  # 커맨드 라인 인자 파싱 수행

    # WebSocket 접속 주소에 'ws://127.0.0.1:8080'을 저장합니다. 대상: `uri`.
    uri = "ws://127.0.0.1:8080"  # 접속할 로컬 웹소켓 서버 주소 및 포트 정의
    # `(win_w, win_h)` 값에 순서·구성을 지정한 `(800, 600)`을 저장합니다. 대상: `(win_w, win_h)`.
    win_w, win_h = 800, 600  # 디스플레이 창의 해상도 고정 규격 정의
    # `renderer` 값에 `OverlayRenderer(conf_threshold=0.35)` 호출 결과를 저장합니다. 대상: `renderer`.
    renderer = OverlayRenderer(conf_threshold=0.35)  # 시각화를 전담할 렌더러 인스턴스 할당

    # `win_title` 값에 계산식 `f'AI Motion Viewer - {args.patient_name}'`의 결과를 저장합니다. 대상: `win_title`.
    win_title = f"AI Motion Viewer - {args.patient_name}"  # 창 상단에 표시될 타이틀 텍스트 포맷팅
    # `cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)`을 호출하여 해당 단계의 작업을 수행합니다.
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)  # 크기 조절이 가능한 OpenCV 윈도우 생성
    # `cv2.resizeWindow(win_title, win_w, win_h)`을 호출하여 해당 단계의 작업을 수행합니다.
    cv2.resizeWindow(win_title, win_w, win_h)  # 윈도우 크기를 설정한 해상도로 강제 조정

    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:  # 소켓 연결 중 발생할 수 있는 에러를 대비한 보호 블록
        # `websockets.connect(uri)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        async with websockets.connect(uri) as websocket:  # 서버 주소로 비동기 소켓 연결 수립
            # `init_payload` 값에 키 'type', 'player_id', 'patient_name', 'exercise_name', 'mode', 'target_reps', 'cal_time'로 구성한 사전을 저장합니다. 대상: `init_payload`.
            init_payload = {  # 서버 측 세션 초기화를 위한 명령어 패킷 딕셔너리 구축
                # 구성 항목: `type` 값(`'type'`)에 'CMD_SET_SESSION'을 지정합니다.
                "type": "CMD_SET_SESSION",              # 세션 설정 타입 식별자
                # 구성 항목: 환자 식별 번호의 기존 필드명(`'player_id'`)에 환자 식별 번호의 기존 필드명(`args.player_id`)을 지정합니다.
                "player_id": args.player_id,            # 환자 아이디 파라미터 탑재
                # 구성 항목: `patient_name` 값(`'patient_name'`)에 `args.patient_name` 값(`args.patient_name`)을 지정합니다.
                "patient_name": args.patient_name,      # 환자 이름 파라미터 탑재
                # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 운동 설정의 식별 이름(`args.exercise_name`)을 지정합니다.
                "exercise_name": args.exercise_name,    # 운동명 파라미터 탑재
                # 구성 항목: 실행 모드(`'mode'`)에 실행 모드(`args.mode`)을 지정합니다.
                "mode": args.mode,                      # 모드 파라미터 탑재
                # 구성 항목: 목표 반복 횟수(`'target_reps'`)에 목표 반복 횟수(`args.target_reps`)을 지정합니다.
                "target_reps": args.target_reps,        # 횟수 파라미터 탑재
                # 구성 항목: cal 시각(`'cal_time'`)에 cal 시각(`args.cal_time`)을 지정합니다.
                "cal_time": args.cal_time,              # 터미널에서 입력받은 캘리브레이션 시간 탑재
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            }
            # `websocket.send(json.dumps(init_payload))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
            await websocket.send(json.dumps(init_payload))  # 서버로 페이로드 JSON 직렬화 전송

            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:  # 렌더링 무한 루프 진입
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:  # 패킷 논블로킹 수신 대기 블록
                    # `response` 값에 계산식 `await asyncio.wait_for(websocket.recv(), timeout=0.05)`의 결과를 저장합니다. 대상: `response`.
                    response = await asyncio.wait_for(websocket.recv(), timeout=0.05)  # 0.05초 동안 짧게 서버 응답 대기
                    # 데이터에 JSON 문자열을 해석한 Python 값(`json.loads(response)`)을 저장합니다. 대상: `data`.
                    data = json.loads(response)  # 수신된 JSON 텍스트 역직렬화
                    
                    # 조건 `data.get('type') in ['SESSION_FINISHED', 'CALIBRATION_FINISHED']`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if data.get("type") in ["SESSION_FINISHED", "CALIBRATION_FINISHED"]:  # 세션이 성공적으로 완수된 경우
                        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                        print("\n[Client] 시스템에서 세션 완료를 확인했습니다.")  # 콘솔에 완료 메시지 출력
                        # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                        break  # 무한 루프를 파기하고 프로그램 종료 절차 돌입

                    # 조건 `data.get('type') == 'POSE_UPDATE'`을 만족하는 경우에만 아래 분기를 실행합니다.
                    if data.get("type") == "POSE_UPDATE":  # 프레임 상태 업데이트 패킷을 수신한 경우
                        # `img_b64` 값에 `data.get('frame_b64')` 호출 결과를 저장합니다. 대상: `img_b64`.
                        img_b64 = data.get("frame_b64")  # Base64 이미지 문자열 추출
                        # `canvas` 값에 `np.zeros((win_h, win_w, 3), dtype=np.uint8)` 호출 결과를 저장합니다. 대상: `canvas`.
                        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)  # 기본 빈 검은색 도화지 생성
                        
                        # 조건 `img_b64`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if img_b64:  # 이미지가 수신되었다면
                            # `img_bytes` 값에 `base64.b64decode(img_b64)` 호출 결과를 저장합니다. 대상: `img_bytes`.
                            img_bytes = base64.b64decode(img_b64)  # Base64 문자열을 바이트 배열로 해독
                            # OpenCV BGR 영상 프레임에 `cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)` 호출 결과를 저장합니다. 대상: `frame`.
                            frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)  # 바이트를 BGR 이미지 행렬로 복원
                            # 조건 `frame is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
                            if frame is not None:  # 이미지 복원에 성공했다면
                                # `canvas` 값에 `cv2.resize(frame, (win_w, win_h))` 호출 결과를 저장합니다. 대상: `canvas`.
                                canvas = cv2.resize(frame, (win_w, win_h))  # 지정된 창 크기에 맞게 리사이징하여 도화지 덮어쓰기

                        # COCO 관절 좌표 목록에 `data.get('keypoints', [])` 호출 결과를 저장합니다. 대상: `keypoints`.
                        keypoints = data.get("keypoints", [])           # 현재 프레임의 관절 데이터 추출
                        # 필수 관절이 가려졌거나 유효하지 않은지 여부에 `data.get('is_occluded', False)` 호출 결과를 저장합니다. 대상: `is_occluded`.
                        is_occluded = data.get("is_occluded", False)    # 가림 발생 여부 플래그 추출
                        # `current_mode` 값에 `data.get('mode', args.mode)` 호출 결과를 저장합니다. 대상: `current_mode`.
                        current_mode = data.get("mode", args.mode)      # 현재 서버 구동 모드 추출
                        
                        # `canvas` 값에 `renderer.draw_skeleton(canvas=canvas, keypoints=keypoints, scale=win_w, is_occluded=is_occluded)` 호출 결과를 저장합니다. 대상: `canvas`.
                        canvas = renderer.draw_skeleton(  # 렌더러를 통해 뼈대 시각화 함수 호출
                            # 구성 항목: `canvas` 값(`canvas`)에 `canvas` 값(`canvas`)을 지정합니다.
                            canvas=canvas,              # 작업할 도화지 전달
                            # 구성 항목: COCO 관절 좌표 목록(`keypoints`)에 COCO 관절 좌표 목록(`keypoints`)을 지정합니다.
                            keypoints=keypoints,        # 그릴 관절 데이터 전달
                            # 구성 항목: `scale` 값(`scale`)에 `win_w` 값(`win_w`)을 지정합니다.
                            scale=win_w,                # 0.0~1.0 비율이므로 가로 해상도를 곱해 픽셀 단위로 환산
                            # 구성 항목: 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)을 지정합니다.
                            is_occluded=is_occluded     # 가림 상태 전달
                        # 앞줄에서 시작한 식·인수·자료 구조를 `) # 렌더링 완료된 캔버스 회수` 내용으로 이어서 구성합니다.
                        )  # 렌더링 완료된 캔버스 회수

                        # 조건 `current_mode == 'MAIN'`을 만족하는 경우에만 아래 분기를 실행합니다.
                        if current_mode == "MAIN":  # 본 운동(Test) 모드일 경우 동적 UI 표시
                            # 왼쪽 info에 `data.get('left', {})` 호출 결과를 저장합니다. 대상: `left_info`.
                            left_info = data.get("left", {})  # 좌측 상태 딕셔너리 추출
                            # 오른쪽 info에 `data.get('right', {})` 호출 결과를 저장합니다. 대상: `right_info`.
                            right_info = data.get("right", {})  # 우측 상태 딕셔너리 추출
                            
                            # `cv2.putText(canvas, f"L: {left_info.get('rep_count', 0)}/{args.target_reps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_quality_color(left_info.get('quality', '')), 2)`을 호출하여 해당 단계의 작업을 수행합니다.
                            cv2.putText(canvas, f"L: {left_info.get('rep_count', 0)}/{args.target_reps}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_quality_color(left_info.get("quality", "")), 2)  # 좌측 횟수 텍스트 렌더링
                            # `cv2.putText(canvas, f"R: {right_info.get('rep_count', 0)}/{args.target_reps}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_quality_color(right_info.get('quality', '')),…`을 호출하여 해당 단계의 작업을 수행합니다.
                            cv2.putText(canvas, f"R: {right_info.get('rep_count', 0)}/{args.target_reps}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, get_quality_color(right_info.get("quality", "")), 2)  # 우측 횟수 텍스트 렌더링
                            
                            # 조건 `is_occluded`을 만족하는 경우에만 아래 분기를 실행합니다.
                            if is_occluded:  # 가려진 상태일 때
                                # `cv2.putText(canvas, 'Please show full body to resume', (win_w // 2 - 240, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
                                cv2.putText(canvas, "Please show full body to resume", (win_w // 2 - 240, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)  # 화면 중앙 경고 문구 렌더링

                        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                        else:  # 캘리브레이션 모드일 경우 중앙 안내 메시지 표시
                            # 캘리브레이션 전체 진행 단계에 `data.get('calib_step', '')` 호출 결과를 저장합니다. 대상: `calib_step`.
                            calib_step = data.get("calib_step", "")  # 현재 진행 단계 추출
                            # `rem_sec` 값에 `data.get('remaining_sec', 0.0)` 호출 결과를 저장합니다. 대상: `rem_sec`.
                            rem_sec = data.get("remaining_sec", 0.0)  # 남은 타이머 시간 추출
                            
                            # 조건 `calib_step == 'FULL_BODY_CHECK'`을 만족하는 경우에만 아래 분기를 실행합니다.
                            if calib_step == "FULL_BODY_CHECK":  # 전신 대기 중일 때
                                # `cv2.putText(canvas, 'PLEASE SHOW FULL BODY', (win_w // 2 - 220, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
                                cv2.putText(canvas, "PLEASE SHOW FULL BODY", (win_w // 2 - 220, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)  # 전신 노출 요구 렌더링
                            # 조건 `calib_step == 'COUNTDOWN'`을 만족하는 경우에만 아래 분기를 실행합니다.
                            elif calib_step == "COUNTDOWN":  # 카운트다운 중일 때
                                # `cv2.putText(canvas, f'READY... {int(np.ceil(rem_sec))}', (win_w // 2 - 110, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
                                cv2.putText(canvas, f"READY... {int(np.ceil(rem_sec))}", (win_w // 2 - 110, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)  # 3초 타이머 렌더링
                            # 조건 `calib_step == 'COLLECTING'`을 만족하는 경우에만 아래 분기를 실행합니다.
                            elif calib_step == "COLLECTING":  # 데이터 수집 중일 때
                                # `cv2.putText(canvas, f'TRACKING... {rem_sec:.1f}s', (win_w // 2 - 140, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)`을 호출하여 해당 단계의 작업을 수행합니다.
                                cv2.putText(canvas, f"TRACKING... {rem_sec:.1f}s", (win_w // 2 - 140, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)  # 타이머 렌더링

                        # `cv2.imshow(win_title, canvas)`을 호출하여 해당 단계의 작업을 수행합니다.
                        cv2.imshow(win_title, canvas)  # 조합이 완료된 프레임 캔버스를 창에 띄우기

                # `asyncio.TimeoutError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
                except asyncio.TimeoutError:  # 송신된 패킷이 없는 경우
                    # 이 분기에는 추가 동작이 없으며 문법적으로 필요한 빈 블록만 유지합니다.
                    pass  # 지연 없이 반복문 통과

                # `key` 값에 계산식 `cv2.waitKey(1) & 255`의 결과를 저장합니다. 대상: `key`.
                key = cv2.waitKey(1) & 0xFF  # 사용자 키보드 입력 1ms 대기
                # 조건 `key in [27, ord('q'), ord('Q')]`을 만족하는 경우에만 아래 분기를 실행합니다.
                if key in [27, ord('q'), ord('Q')]:  # ESC 또는 Q 키를 누른 경우
                    # `websocket.send(json.dumps({'type': 'CMD_STOP_CALIBRATION'}))`을 호출하여 상대 WebSocket 연결로 메시지를 전송합니다. 완료될 때까지 비동기로 기다립니다.
                    await websocket.send(json.dumps({"type": "CMD_STOP_CALIBRATION"}))  # 서버에 안전 종료 요청 패킷 발송
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break  # 프로그램 루프 이탈
                
                # 조건 `cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1`을 만족하는 경우에만 아래 분기를 실행합니다.
                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:  # 윈도우 창 닫기 버튼을 누른 경우
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break  # 루프 이탈

    # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
    except Exception as e:  # 네트워크 접근 실패 등 에러 발생 시
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[Client Error] 서버 연결 실패: {e}")  # 오류 원인 출력
    # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
    finally:  # 루프 이탈 시 반드시 실행
        # `cv2.destroyAllWindows()`을 호출하여 해당 단계의 작업을 수행합니다.
        cv2.destroyAllWindows()  # 모든 창 파괴 및 메모리 회수

# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == "__main__":  # 모듈 직접 실행 판별
    # `asyncio.run(main())`을 호출하여 해당 단계의 작업을 수행합니다.
    asyncio.run(main())  # 메인 비동기 루프 호출