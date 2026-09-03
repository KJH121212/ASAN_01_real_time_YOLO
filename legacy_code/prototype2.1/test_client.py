# ==============================================================================
# [파일 정보]
# 파일명: test_client.py
# 설명: WebSocket 기반 실시간 AI 모션 시각화 및 모드 분리형 클라이언트 UI
# ==============================================================================

import argparse  # 명령줄 인자 파싱 모듈
import asyncio   # 비동기 통신 모듈
import base64    # 이미지 디코딩 모듈
import json      # JSON 파싱 모듈
import os        # 파일 검사 모듈
import time      # 지연 모듈
import warnings  # 경고 메세지 차단 모듈
from PIL import Image, ImageDraw, ImageFont  # 한글 렌더링용 PIL 라이브러리
import cv2       # 이미지 처리 라이브러리
import numpy as np  # 배열 처리 라이브러리
import websockets   # 소켓 통신 모듈

warnings.filterwarnings("ignore", category=FutureWarning)  # 경고 메세지 통제

SKELETON_CONNECTIONS = [  # 뼈대 렌더링 연결 리스트
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)
]

FONT_PATH = "C:/Windows/Fonts/malgun.ttf"  # 한글 폰트 주소
HAS_FONT = os.path.exists(FONT_PATH)  # 존재 여부 검사


def create_static_ui_layer(win_w: int, win_h: int, patient_name: str, exercise_name: str) -> np.ndarray:
    """캘리브레이션과 테스트 모드 공통으로 쓰이는 뒷배경(패널) 정적 레이어를 생성합니다."""
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)  # 블랙 캔버스 할당

    # 정보 패널 뒷배경 (바, 텍스트가 얹어질 공간)
    panel_x1, panel_y1, panel_x2, panel_y2 = win_w - 210, 55, win_w - 10, 100 + 260 + 40  # 좌표 연산
    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (15, 15, 20), -1)  # 진한 배경
    cv2.rectangle(canvas, (panel_x1, panel_y1), (panel_x2, panel_y2), (80, 80, 80), 1)  # 테두리 선

    # 스켈레톤 미니맵 박스 뒷배경
    mini_bx = (win_w - 140) // 2  # 중앙 하단 정렬 연산
    mini_by = win_h - 210  # 높이 연산
    cv2.rectangle(canvas, (mini_bx, mini_by), (mini_bx + 140, mini_by + 140), (10, 10, 15), -1)  # 상자 생성

    img_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))  # PIL 색상 공간 전환
    draw = ImageDraw.Draw(img_pil)  # 그리기 객체 확보

    font_title = ImageFont.truetype(FONT_PATH, 15) if HAS_FONT else ImageFont.load_default()  # 폰트 로드
    font_sub = ImageFont.truetype(FONT_PATH, 13) if HAS_FONT else ImageFont.load_default()  # 폰트 로드

    draw.text((20, 10), f"{patient_name} | {exercise_name}", font=font_title, fill=(255, 255, 255))  # 타이틀 이름 그리기
    draw.text((panel_x1 + 35, panel_y1 + 8), "가동범위 (ROM)", font=font_sub, fill=(255, 255, 255))  # 서브 타이틀 그리기

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)  # 원본 OpenCV 포맷 복귀 반환


def get_quality_color(quality_str: str) -> tuple:
    """수행 품질에 따른 BGR 색상을 반환합니다."""
    if quality_str == "PERFECT": return (0, 255, 0)  # 초록색
    elif quality_str == "GOOD": return (0, 255, 255)  # 노란색
    elif quality_str == "BAD": return (0, 0, 255)  # 빨간색
    elif quality_str == "CALIBRATING": return (255, 200, 0)  # 주황색
    return (180, 180, 180)  # 기본 회색


async def main():
    parser = argparse.ArgumentParser()  # 인자 파서 생성
    parser.add_argument("--player_id", type=str, default="patient_1")  # 옵션 등록
    parser.add_argument("--patient_name", type=str, default="김지후")  # 옵션 등록
    parser.add_argument("--exercise_name", type=str, default="biceps_curl")  # 옵션 등록
    parser.add_argument("--mode", type=str, default="CALIBRATION")  # 옵션 등록
    parser.add_argument("--target_reps", type=int, default=5)  # 목표 횟수 등록
    parser.add_argument("--camera_index", type=int, default=0)  # 캠 등록
    parser.add_argument("--cal_time", type=float, default=5.0)  # 캘리 시간 등록
    args = parser.parse_args()  # 파싱 수행

    uri = "ws://127.0.0.1:8080"  # 연결 소켓 주소
    win_w, win_h = 800, 600  # 윈도우 크기 고정

    static_ui_layer = create_static_ui_layer(win_w, win_h, args.patient_name, args.exercise_name)  # 배경 패널 단 1회 생성
    ui_mask = static_ui_layer != 0  # 렌더링 덮어쓰기를 위한 불리언 마스크

    latest_keypoints, fps, current_mode = [], 0.0, args.mode  # 변수 초기화
    left_rep, right_rep = 0, 0  # 횟수 그릇
    left_progress, right_progress = 0.0, 0.0  # 게이지 그릇
    left_quality, right_quality = "CALIBRATING", "CALIBRATING"  # 퀄리티 그릇

    yolo_detected, latest_frame_b64 = False, None  # 영상 데이터 변수
    calib_step, remaining_sec = "FULL_BODY_CHECK", 0.0  # 캘리 제어 변수
    is_occluded = False  # 서버에서 판단한 가림 여부 변수

    win_title = f"AI Motion Viewer - {args.patient_name}"  # 창 이름
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)  # 창 할당
    cv2.resizeWindow(win_title, win_w, win_h)  # 비율 고정

    is_finished = False  # 종료 플래그
    finish_message = ""  # 안내 문구 저장소

    try:
        async with websockets.connect(uri) as websocket:  # 소켓 연결
            await websocket.send(  # 초기 세팅 커맨드 발사
                json.dumps({
                    "type": "CMD_SET_SESSION",  # 커맨드명
                    "player_id": args.player_id,  # 데이터
                    "patient_name": args.patient_name,  # 데이터
                    "exercise_name": args.exercise_name,  # 데이터
                    "mode": args.mode,  # 데이터
                    "camera_index": args.camera_index,  # 데이터
                    "target_reps": args.target_reps,  # 목표 횟수
                    "cal_time": args.cal_time,  # 설정 시간 연동
                })
            )

            while True:  # 렌더링 코어 루프
                while True:  # 소켓 풀링 블록
                    try:
                        response = await asyncio.wait_for(websocket.recv(), timeout=0.0001)  # 논블로킹 수신
                        data = json.loads(response)  # 언패킹

                        if data.get("type") == "POSE_UPDATE":  # 정보 업데이트 시
                            fps = data.get("fps", 0.0)  # FPS 추출
                            current_mode = data.get("mode", args.mode)  # 모드 추출
                            latest_keypoints = data.get("keypoints", [])  # 위치 추출
                            yolo_detected = data.get("yolo_detected", False)  # 검출 여부 추출
                            is_occluded = data.get("is_occluded", False)  # 화면 가림 여부 추출
                            calib_step = data.get("calib_step", "FULL_BODY_CHECK")  # 캘리 스텝 추출
                            remaining_sec = data.get("remaining_sec", 0.0)  # 타임 타이머 추출

                            if data.get("frame_b64"):  # 영상 수신 시
                                latest_frame_b64 = data.get("frame_b64")  # 보관

                            left_info = data.get("left", {})  # 왼쪽 기록
                            right_info = data.get("right", {})  # 오른쪽 기록

                            left_rep, left_progress = left_info.get("rep_count", 0), left_info.get("progress_ratio", 0.0)  # 배정
                            left_quality = left_info.get("quality", "CALIBRATING")  # 배정

                            right_rep, right_progress = right_info.get("rep_count", 0), right_info.get("progress_ratio", 0.0)  # 배정
                            right_quality = right_info.get("quality", "CALIBRATING")  # 배정

                        elif data.get("type") == "CALIBRATION_FINISHED":  # 캘리 완수 시
                            is_finished = True  # 플래그 활성화
                            finish_message = "CALIBRATION FINISHED!"  # 문구 할당

                        elif data.get("type") == "SESSION_FINISHED":  # 운동 완수 시
                            is_finished = True  # 플래그 활성화
                            finish_message = "EXERCISE COMPLETED!"  # 문구 할당

                    except asyncio.TimeoutError:  # 큐가 비었으면
                        break  # 통과

                if latest_frame_b64:  # 이미지가 존재하면
                    try:
                        img_bytes = base64.b64decode(latest_frame_b64)  # 텍스트 풀기
                        decoded_frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)  # 이미지로 해석
                        canvas = cv2.resize(decoded_frame, (win_w, win_h)) if decoded_frame is not None else np.zeros((win_h, win_w, 3), dtype=np.uint8)  # 해상도 크기 조정
                    except Exception:  # 깨짐 방지
                        canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)  # 검은 도화지 유지
                else:  # 부재 시
                    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)  # 검은 도화지 유지

                canvas[ui_mask] = static_ui_layer[ui_mask]  # 뒷배경 UI 정적 마스크 덮어쓰기
                
                is_calib_mode = (current_mode == "CALIBRATION")  # 현재 모드 판별 플래그
                draw_calibration_guide(canvas, win_w, is_calib_mode)  # 상단 안내 바 렌더링

                target = args.target_reps  # 표시용 목표 변수 할당

                # --------------------------------------------------------------
                # [동적 UI 렌더링] TEST(MAIN) 모드일 때만 카운트 텍스트 및 가동범위 게이지(Bar)를 그립니다.
                # --------------------------------------------------------------
                if not is_calib_mode:  # 캘리브레이션 모드가 아닌 본 운동일 경우에만 표시
                    # Rep 텍스트 표시
                    cv2.putText(canvas, f"L (REP): {left_rep}/{target}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.55, get_quality_color(left_quality), 2, cv2.LINE_AA)  # 왼쪽 카운트
                    cv2.putText(canvas, f"R (REP): {right_rep}/{target}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, get_quality_color(right_quality), 2, cv2.LINE_AA)  # 오른쪽 카운트

                    # ROM 게이지 바 배경 및 선 그리기
                    bar_h, bar_w, by = 260, 26, 100  # 높이 및 너비 규격 설정
                    bx_l, bx_r = win_w - 110, win_w - 50  # 좌우 게이지 위치 연산
                    panel_x1 = win_w - 210  # 텍스트 레이블 X 좌표 기준선

                    cv2.rectangle(canvas, (bx_l, by), (bx_l + 26, by + bar_h), (40, 40, 40), -1)  # 왼쪽 빈 게이지 그리기
                    cv2.rectangle(canvas, (bx_r, by), (bx_r + 26, by + bar_h), (40, 40, 40), -1)  # 오른쪽 빈 게이지 그리기

                    thresholds = [(0.80, "80%", (0, 255, 0)), (0.60, "60%", (0, 255, 255)), (0.40, "40%", (0, 0, 255)), (0.20, "20%", (200, 200, 200))]  # 기준선 정보
                    for ratio, label, color in thresholds:  # 기준선 배열 순회
                        line_y = by + int(bar_h * (1.0 - ratio))  # Y축 픽셀 위치 연산
                        cv2.line(canvas, (bx_l - 4, line_y), (bx_r + 26 + 4, line_y), color, 1, cv2.LINE_AA)  # 가로 선 긋기
                        cv2.putText(canvas, label, (panel_x1 + 8, line_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)  # 레이블 표기

                    cv2.putText(canvas, "L", (bx_l + 6, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 100), 2, cv2.LINE_AA)  # L 표기
                    cv2.putText(canvas, "R", (bx_r + 6, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2, cv2.LINE_AA)  # R 표기

                    # 실시간 진행률 채우기 함수
                    def draw_fill(bx, p):  # 내부 도색 래퍼 함수
                        p_clamped = min(max(p, 0.0), 1.0)  # 0~1 퍼센트 한정 클램핑
                        fill_h = int(bar_h * p_clamped)  # 높이 환산
                        if fill_h > 0:  # 채울 영역이 있다면
                            color = (0, 255, 0) if p_clamped >= 0.8 else ((0, 255, 255) if p_clamped >= 0.6 else ((0, 100, 255) if p_clamped >= 0.4 else (200, 200, 200)))  # 색상 할당
                            cv2.rectangle(canvas, (bx + 2, by + bar_h - fill_h), (bx + bar_w - 2, by + bar_h), color, -1)  # 위로 솟아오르는 사각형 칠하기

                    draw_fill(bx_l, left_progress)  # 왼쪽 게이지 칠하기
                    draw_fill(bx_r, right_progress)  # 오른쪽 게이지 칠하기

                    cv2.putText(canvas, f"{int(left_progress * 100)}%", (bx_l - 2, by + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_quality_color(left_quality), 1, cv2.LINE_AA)  # 하단 왼쪽 퍼센트 문자열
                    cv2.putText(canvas, f"{int(right_progress * 100)}%", (bx_r - 2, by + bar_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, get_quality_color(right_quality), 1, cv2.LINE_AA)  # 하단 오른쪽 퍼센트 문자열

                    # [핵심] TEST 모드 가려짐(Occlusion) 발생 시 화면 중앙에 경고 메시지 출력
                    if is_occluded:  # 가려진 상태일 때
                        cv2.putText(canvas, "Please show full body to resume", (win_w // 2 - 240, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)  # 이어서 하라는 멘트 발동

                # --------------------------------------------------------------
                # 캘리브레이션 전용 안내 모드 렌더링
                # --------------------------------------------------------------
                else:  # 캘리브레이션 모드일 경우 (게이지 제거, 상태 텍스트 중앙 강조)
                    cv2.putText(canvas, "CALIBRATION", (win_w - 180, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # 우측 패널 안내 문자열
                    cv2.putText(canvas, "IN PROGRESS", (win_w - 180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)  # 안내 문자열 하단부

                    if calib_step == "FULL_BODY_CHECK":  # 가려짐 등으로 전신 대기 중일 때
                        cv2.putText(canvas, "PLEASE SHOW FULL BODY", (win_w // 2 - 220, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)  # 진입 권유 텍스트
                    elif calib_step == "COUNTDOWN":  # 3초 카운트 다운 대기 중일 때 (정확히 3초만 출력됨)
                        cv2.putText(canvas, f"READY... {int(np.ceil(remaining_sec))}", (win_w // 2 - 110, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3, cv2.LINE_AA)  # 카운트 텍스트 렌더링
                    elif calib_step == "COLLECTING":  # 본 수집(5초) 돌입 상태일 때
                        cv2.putText(canvas, f"TRACKING... {remaining_sec:.1f}s", (win_w // 2 - 140, win_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)  # 소수점 타이머 렌더링
                        cv2.putText(canvas, "(Press 'Q' or 'ESC' to Stop Early)", (win_w // 2 - 160, win_h // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)  # 조기 종료 멘트

                # 공통 렌더링 영역 (FPS 및 미니맵 박스)
                cv2.putText(canvas, f"{fps:.0f} FPS", (win_w - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)  # FPS 그리기

                # 미니 박스 스켈레톤 라인업 그리기
                mini_bx, mini_by, box_size = (win_w - 140) // 2, win_h - 210, 140  # 미니 박스 범위 산출
                border_color = (0, 255, 0) if yolo_detected else (0, 0, 255)  # 테두리 색상 분기
                cv2.rectangle(canvas, (mini_bx, mini_by), (mini_bx + box_size, mini_by + box_size), border_color, 2)  # 외곽선

                if yolo_detected and latest_keypoints:  # 스켈레톤 투영
                    center_x, center_y = mini_bx + box_size // 2, mini_by + box_size // 2  # 중앙 점
                    scale_factor = (box_size - 30) / 3.0  # 상대 비율
                    kpt_dict = {}  # 매핑 사전 할당

                    for kp in latest_keypoints:  # 키포인트 반복
                        px = int(center_x + (kp.get("x", 0.0) - 0.5) * scale_factor * 3.0)  # 상대 좌표 평탄화
                        py = int(center_y + (kp.get("y", 0.0) - 0.5) * scale_factor * 3.0)  # 상대 좌표 평탄화
                        kpt_dict[kp.get("id")] = (px, py, kp.get("score", 0.0))  # 투영 좌표 할당

                    for p1_id, p2_id in SKELETON_CONNECTIONS:  # 연결선 그리기
                        if p1_id in kpt_dict and p2_id in kpt_dict:  # 양쪽 뼈대 존재 시
                            x1, y1, s1 = kpt_dict[p1_id]  # 첫 점
                            x2, y2, s2 = kpt_dict[p2_id]  # 두번째 점
                            if s1 > 0.35 and s2 > 0.35:  # 신뢰도 평가
                                cv2.line(canvas, (x1, y1), (x2, y2), (0, 255, 255), 2, cv2.LINE_AA)  # 선긋기

                    for idx, (x, y, s) in kpt_dict.items():  # 관절점 그리기
                        if idx >= 5 and s > 0.35:  # 머리 제외 관절일 경우
                            cv2.circle(canvas, (x, y), 3, (0, 165, 255), -1, cv2.LINE_AA)  # 점 표기

                if is_finished:  # 종료 상태라면
                    overlay = canvas.copy()  # 화면 블러 처리를 위한 사본
                    cv2.rectangle(overlay, (win_w // 2 - 220, win_h // 2 - 40), (win_w // 2 + 220, win_h // 2 + 40), (0, 0, 0), -1)  # 반투명 박스
                    cv2.addWeighted(overlay, 0.8, canvas, 0.2, 0, canvas)  # 병합
                    cv2.putText(canvas, finish_message, (win_w // 2 - 190, win_h // 2 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)  # 문구 강조
                    cv2.imshow(win_title, canvas)  # 마지막 화면 출력
                    cv2.waitKey(1500)  # 유지 후
                    break  # 루프 이탈

                cv2.imshow(win_title, canvas)  # 지속적인 화면 출력 수행

                key = cv2.waitKey(1) & 0xFF  # 키보드 이벤트 응답 대기
                if key in [27, ord('q'), ord('Q')]:  # 종료 인터럽트
                    if is_calib_mode and calib_step == "COLLECTING":  # 수집 중이라면
                        await websocket.send(json.dumps({"type": "CMD_STOP_CALIBRATION"}))  # 안전 조기종료 패킷 발송
                    else:  # 일반 상태라면
                        break  # 창 파괴

                if cv2.getWindowProperty(win_title, cv2.WND_PROP_VISIBLE) < 1:  # 창의 X버튼을 클릭했다면
                    break  # 루프 이탈

                await asyncio.sleep(0.001)  # 비동기 시스템 양보

            cv2.destroyAllWindows()  # GUI 해제

    except Exception as e:  # 예외 관리
        print(f"[Error] {e}")  # 표출


def draw_calibration_guide(canvas: np.ndarray, win_w: int, is_calib_mode: bool):
    """모드별 상단 헤더 안내선을 표시합니다."""
    if is_calib_mode:  # 캘리브레이션용 상단 바
        cv2.rectangle(canvas, (0, 0), (win_w, 40), (20, 20, 30), -1)  # 배경 띠
        cv2.rectangle(canvas, (0, 0), (win_w, 40), (0, 140, 255), 1)  # 테두리
        guide_text = "[CALIBRATION] Stand still with full body in frame for countdown."  # 권장 텍스트
        cv2.putText(canvas, guide_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)  # 안내


if __name__ == "__main__":
    asyncio.run(main())  # 메인 루프 시동