# ==============================================================================
# [파일 정보]
# 파일명: socket_server.py
# 설명: 캘리브레이션 임계값 자동 연산 및 data_manager.py 연동 JSON 저장 기능 탑재 소켓 서버
# 주요 기능:
#    1. 캘리브레이션 수집 간 실시간 모션 수치(Angle/Relative Y) 추출 및 배열 누적
#    2. 상/하위 5% 백분위수를 활용한 노이즈 제거 및 85% 가동 범위 맞춤 임계값 산출
#    3. DataManager를 통한 환자 개인 JSON 프로필에 custom_thresholds 영구 저장
# ==============================================================================

import asyncio                  # 비동기 I/O 이벤트 루프 제어 모듈 로드
import base64                   # 이미지 프레임 문자열 변환용 인코딩 모듈 로드
from datetime import datetime   # 타임스탬프 기반 파일명 생성을 위한 모듈 로드
import json                     # 웹소켓 패킷 데이터 파싱을 위한 JSON 모듈 로드
import os                       # 시스템 디렉토리 경로 탐색용 모듈 로드
import sys                      # 파이썬 런타임 환경 경로 조작용 모듈 로드
import threading                # 카메라 백그라운드 프레임 수집용 멀티스레드 모듈 로드
import time                     # 카운트다운 및 타임아웃 측정을 위한 시간 모듈 로드
import traceback                # 에러 발생 시 상세 원인 출력을 위한 모듈 로드
from unittest.mock import MagicMock  # C++ 확장 패키지 로드 에러 방지용 모킹 객체 로드
import warnings                 # 터미널 경고 메시지 숨김 처리를 위한 모듈 로드
import cv2                      # 카메라 제어 및 이미지 리사이징용 OpenCV 라이브러리 로드
import numpy as np              # 백분위수(Percentile) 통계 연산용 NumPy 라이브러리 로드
import torch                    # 포즈 추론 딥러닝 구동을 위한 PyTorch 프레임워크 로드
import websockets               # 비동기 양방향 통신 서버 구축용 웹소켓 라이브러리 로드

warnings.filterwarnings("ignore", category=FutureWarning)  # 불필요한 미래 버전 경고 메시지 출력 차단

mock_ext = MagicMock()  # MMCV 임포트 오류를 우회하기 위한 빈 모킹 객체 생성
mock_ext.__spec__ = MagicMock()  # 스펙 속성 모킹 처리
sys.modules['mmcv._ext'] = mock_ext  # 시스템 모듈 딕셔너리에 가짜 확장 모듈 강제 등록

import pycocotools  # xtcocotools 패키지 종속성 해결을 위한 코코툴스 로드
sys.modules['xtcocotools'] = pycocotools  # 패키지명 맵핑으로 임포트 에러 방지

current_file_path = os.path.abspath(__file__)  # 현재 실행 중인 파일의 절대 경로 추출
server_dir = os.path.dirname(current_file_path)  # 상위 서버 폴더 경로 추출
prototype_dir = os.path.dirname(server_dir)  # 프로토타입 메인 폴더 경로 추출
root_dir = os.path.dirname(prototype_dir)  # 프로젝트 최상위 루트 경로 추출

if prototype_dir not in sys.path:  # 모듈 탐색 경로에 메인 폴더가 없다면
    sys.path.insert(0, prototype_dir)  # 최우선 탐색 경로로 삽입
if root_dir not in sys.path:  # 모듈 탐색 경로에 루트 폴더가 없다면
    sys.path.insert(1, root_dir)  # 2순위 탐색 경로로 삽입

_original_torch_load = torch.load  # 기존 PyTorch 모델 로드 함수 원본 백업

def _patched_torch_load(*args, **kwargs):  # 체크포인트 로드 보안 경고를 끄기 위한 래퍼 함수 정의
    kwargs['weights_only'] = False  # 안전 경고 해제 파라미터 강제 삽입
    return _original_torch_load(*args, **kwargs)  # 패치된 설정으로 원본 함수 실행

torch.load = _patched_torch_load  # 시스템 전역의 torch.load 함수를 패치된 함수로 교체

from core.data_manager import DataManager, load_exercise_config  # 로컬 JSON 저장 제어기 및 설정 로더 수입
from core.motion_engine import MotionEngine  # FSM 기반 운동 상태 분석 엔진 수입
from core.skeleton_engine import SkeletonEngine  # YOLO 및 RTMPose 기반 관절 좌표 추출기 수입


class ThreadedCamera:

    def __init__(self, camera_index=0):
        self.camera_index = camera_index  # 사용할 웹캠 디바이스 인덱스 할당
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # 윈도우 환경에 최적화된 DirectShow 방식으로 오픈
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 영상 출력 지연을 막기 위해 프레임 버퍼를 1로 제한
        self.grabbed, self.frame = self.cap.read()  # 센서로부터 첫 번째 프레임 초기 수집
        self.started = False  # 백그라운드 스레드 동작 상태 플래그
        self.read_lock = threading.Lock()  # 데이터 동시 접근 충돌을 막기 위한 동기화 락 객체

    def start(self):
        if self.started:  # 이미 스레드가 돌고 있는 상태라면
            return self  # 추가 실행 없이 자신 객체 반환
        self.started = True  # 동작 상태 활성화
        self.thread = threading.Thread(target=self.update, daemon=True)  # 메인 종료 시 함께 꺼지는 데몬 스레드 생성
        self.thread.start()  # 스레드 무한 루프 가동
        return self  # 메서드 체이닝을 위한 자신 반환

    def update(self):
        while self.started:  # 스레드가 활성화된 동안 지속 반복
            grabbed, frame = self.cap.read()  # 센서 버퍼에서 최신 이미지 획득
            with self.read_lock:  # 외부 읽기 로직과 겹치지 않게 락 잠금
                self.grabbed = grabbed  # 프레임 수집 성공 여부 갱신
                self.frame = frame  # 최신 프레임 덮어쓰기

    def read(self):
        with self.read_lock:  # 안전하게 데이터를 내어주기 위해 락 잠금
            if not self.grabbed or self.frame is None:  # 프레임이 불량하거나 끊겼다면
                return False, None  # 실패 플래그 및 None 반환
            return True, self.frame.copy()  # 데이터 오염 방지를 위해 프레임 사본을 전달

    def stop(self):
        self.started = False  # 업데이트 무한 루프 탈출 조건 부여
        if self.cap and self.cap.isOpened():  # 카메라 장치가 연결되어 있다면
            self.cap.release()  # 시스템 리소스 반환


class ExerciseSocketServer:

    def __init__(self, host="127.0.0.1", port=8080):
        self.host = host  # 허용할 접속 IP 주소 매핑
        self.port = port  # 통신 포트 번호 맵핑

        print("[SocketServer] SkeletonEngine AI 모델 로드 중...")  # 모델 초기화 진입 텍스트 출력
        self.skeleton_engine = SkeletonEngine(yolo_interval=5)  # 사람 인식은 5프레임당 1번만 하여 연산 절약

        self.camera = None  # 비동기 카메라 인스턴스 그릇
        self.active_session = False  # 현재 세션 진행 상태 플래그

        self.data_manager = None  # JSON/CSV 파일 입출력 매니저 그릇
        self.motion_engine = None  # 각도 연산 및 FSM 카운팅 엔진 그릇
        self.current_exercise = "biceps_curl"  # 기본 운동 종목 명칭
        self.player_id = "patient_1"  # 환자 고유 식별 아이디
        self.patient_name = "Harry"  # 환자 표기 이름
        self.target_reps = 10  # 운동 목표 횟수 기본값
        self.session_start_time = None  # 세션 구동 타임스탬프

        self.calib_step = "FULL_BODY_CHECK"  # 캘리브레이션 진입 1단계 명칭
        self.calib_start_time = None  # 카운트다운 타이머용 기준 시간
        self.calib_duration = 5.0  # (수정) 캘리브레이션 정보 수집 지속 시간 5초

        # [저장 버퍼] 원본 데이터 시퀀스 저장용 리스트
        self.buf_timestamps = []  # 경과 시간 기록 리스트
        self.buf_values = []  # 단순 포맷용 데이터 리스트
        self.buf_keypoints = []  # 17개 관절 전체 위치 리스트

        # [추가 버퍼] 캘리브레이션 임계값 계산용 10초간의 모션 수치(Angle) 저장 리스트
        self.calib_values_left = []  # 좌측 관절 수치 누적 리스트
        self.calib_values_right = []  # 우측 관절 수치 누적 리스트

        # [UI 유지용 캐시] 가려짐 발생 시 화면 수치가 0으로 떨어지는 현상 방지용 캐시
        self.last_left_data = {"val": None, "rep_count": 0, "progress_ratio": 0.0, "state": "READY", "quality": "CALIBRATING"}  # 좌측 상태 복사본
        self.last_right_data = {"val": None, "rep_count": 0, "progress_ratio": 0.0, "state": "READY", "quality": "CALIBRATING"}  # 우측 상태 복사본

    def start_camera(self, camera_index: int = 0):
        if self.camera:  # 이전에 가동 중이던 카메라가 있다면
            self.stop_camera()  # 리소스 해제 후 정리
        self.camera = ThreadedCamera(camera_index).start()  # 전달받은 번호로 렌즈 오픈

    def stop_camera(self):
        if self.camera:  # 카메라 객체가 메모리에 존재하면
            self.camera.stop()  # 백그라운드 구동 스레드 소멸
            self.camera = None  # 변수 초기화

    def init_session(
        self,
        player_id: str,
        patient_name: str,
        exercise_name: str,
        mode: str = "CALIBRATION",
        target_reps: int = 10,
        camera_index: int = 0,
        cal_time: float = 5.0,
    ):
        self.player_id = player_id  # 세션 환자 번호 적용
        self.patient_name = patient_name  # 세션 환자 이름 적용
        self.current_exercise = exercise_name  # 수행할 운동명 적용
        self.target_reps = target_reps  # 반복 횟수 세팅
        self.calib_duration = float(cal_time)  # 타이머 지속 시간 세팅
        self.session_start_time = time.time()  # 기준 타이머 온

        self.calib_step = "FULL_BODY_CHECK"  # 캘리브레이션 스테이지 리셋
        self.calib_start_time = None  # 카운터 리셋

        self.buf_timestamps.clear()  # NPZ 백업용 시간 배열 클리어
        self.buf_values.clear()  # NPZ 백업용 밸류 배열 클리어
        self.buf_keypoints.clear()  # NPZ 백업용 스켈레톤 배열 클리어

        self.calib_values_left.clear()  # 임계값 계산용 좌측 배열 클리어
        self.calib_values_right.clear()  # 임계값 계산용 우측 배열 클리어

        self.last_left_data = {"val": None, "rep_count": 0, "progress_ratio": 0.0, "state": "READY", "quality": "CALIBRATING"}  # 캐시 안전 리셋
        self.last_right_data = {"val": None, "rep_count": 0, "progress_ratio": 0.0, "state": "READY", "quality": "CALIBRATING"}  # 캐시 안전 리셋

        self.start_camera(camera_index)  # 렌즈 개방 명령어 하달
        self.data_manager = DataManager(player_id=player_id, patient_name=patient_name)  # 환자 프로필용 파일 제어기 생성[cite: 2]

        exercise_config = load_exercise_config(exercise_name)  # 타겟 관절 등이 명시된 기본 JSON 로드[cite: 2]
        custom_thresholds = self.data_manager.get_custom_threshold(exercise_name)  # 이전에 계산된 환자 전용 임계값 탐색[cite: 2]

        if custom_thresholds:  # 커스텀 임계값이 발견되었는데
            if "left" not in custom_thresholds and "start_val" in custom_thresholds:  # 뎁스 구조가 깨진 구버전 포맷이라면
                custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}  # 좌우 양방향 데이터로 강제 감싸서 에러 방지

        self.motion_engine = MotionEngine(
            exercise_config,
            custom_thresholds,
            mode=mode,
            target_reps=target_reps,
        )  # 관절 수치 연산 및 FSM 카운팅을 책임질 엔진 인스턴스화
        self.active_session = True  # 세션 온 플래그 활성화

    def check_full_body(self, keypoints: list) -> bool:
        if not keypoints or len(keypoints) < 17:  # 검출된 관절 수가 모자라면
            return False  # 가림(Occlusion) 처리
        
        kpt_map = {kp["id"]: kp.get("score", 0.0) for kp in keypoints}  # 번호별 신뢰도 추출
        required_ids = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]  # 전신 12개 주요 파트 지정
        return all(kpt_map.get(i, 0.0) >= 0.35 for i in required_ids)  # 모두 신뢰성 있게 식별되면 True 리턴

    def _calculate_and_save_thresholds(self) -> dict:
        """수집된 캘리브레이션 배열을 바탕으로 임계값을 산출하고 JSON 프로필에 저장합니다."""
        def _calc(values: list) -> dict:  # 편측 배열 연산용 내부 함수
            if not values or len(values) < 10:  # 데이터 풀이 너무 작으면
                return {"start_val": 160.0, "target_val": 60.0}  # 하드코딩된 기본값 배출
            
            q_min = float(np.percentile(values, 5))  # 노이즈를 쳐낸 하위 5% 최소 가동범위 추출
            q_max = float(np.percentile(values, 95))  # 노이즈를 쳐낸 상위 5% 최대 가동범위 추출

            if self.motion_engine and self.motion_engine.motion_direction == "DECREASING":  # 수치 감소형 운동일 경우
                return {
                    "start_val": round(q_max, 1),  # 편 상태를 시작점으로
                    "target_val": round(q_max - (q_max - q_min) * 0.85, 1)  # 85% 굽힌 지점을 타겟으로 산출
                }  # 생성된 딕셔너리 반환
            else:  # 수치 증가형 운동일 경우
                return {
                    "start_val": round(q_min, 1),  # 굽힌 상태를 시작점으로
                    "target_val": round(q_min + (q_max - q_min) * 0.85, 1)  # 85% 편 지점을 타겟으로 산출
                }  # 생성된 딕셔너리 반환

        custom_th = {
            "left": _calc(self.calib_values_left),  # 좌측 배열 85% 임계 연산
            "right": _calc(self.calib_values_right)  # 우측 배열 85% 임계 연산
        }  # 양측 결과 병합

        if self.data_manager:  # 파일 매니저가 정상적이라면
            self.data_manager.save_custom_threshold(
                self.current_exercise, custom_th, is_confirmed=True
            )  # 산출된 임계값을 data_manager를 통해 환자 JSON 파일에 영구 기록[cite: 2]

        return custom_th  # 통신 송신을 위한 딕셔너리 리턴

    def _save_current_session_data(self):
        if not self.data_manager or len(self.buf_timestamps) == 0:  # 버퍼에 내용이 없으면
            return None  # 종료 회피

        session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 세션 타임 아이디 할당
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 표기용 포맷팅

        left_reps = self.motion_engine.fsm_left.completed_reps_history  # 좌측 회차 상세 로그 가져오기
        right_reps = self.motion_engine.fsm_right.completed_reps_history  # 우측 회차 상세 로그 가져오기

        all_rep_rows = [
            {**r, "session_id": session_id, "timestamp": timestamp_str}
            for r in left_reps + right_reps
        ]  # 세션 메타 정보 병합 합치기

        self.data_manager.save_rep_details_csv(all_rep_rows)  # 운동 완료 상세 정보를 CSV로 기록[cite: 2]

        self.data_manager.save_trajectory_npz(
            session_id=session_id,  # 고유 키
            exercise_name=self.current_exercise,  # 종목
            timestamps=self.buf_timestamps,  # 시계열
            values=self.buf_values,  # 모의 밸류
            keypoints=self.buf_keypoints,  # 추출 좌표계
        )  # 궤적 원본을 압축 파일로 디스크 드랍[cite: 2]

        return {
            "session_id": session_id,
            "patient_id": self.player_id,
            "saved_reps_count": len(all_rep_rows),
        }  # 요약 정보 배출

    async def handle_client(self, websocket):
        print(f"[SocketServer] 클라이언트 접속: {websocket.remote_address}")  # 연결 환영 콘솔
        prev_time = time.time()  # FPS 체크용 시간
        frame_count = 0  # 드롭 카운터
        last_frame_b64 = None  # 압축 프레임 보관소

        try:
            while True:  # 렌더링 무한 루프
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.001)  # 비동기 리시버 찰나 블로킹
                    data = json.loads(message)  # 데이터 언패킹
                    
                    if data.get("type") == "CMD_SET_SESSION":  # 모드 지시 패킷일 경우
                        self.init_session(
                            player_id=data.get("player_id", "patient_1"),
                            patient_name=data.get("patient_name", "미지정"),
                            exercise_name=data.get("exercise_name", "biceps_curl"),
                            mode=data.get("mode", "CALIBRATION"),
                            target_reps=data.get("target_reps", 10),
                            camera_index=data.get("camera_index", 0),
                            cal_time=data.get("cal_time", 5.0),  # 시간 옵션 지시
                        )  # 세션 엔진 세팅 완료
                    elif data.get("type") == "CMD_STOP_CALIBRATION":  # 유저 강제 이탈 버튼 입력 시
                        if self.active_session and self.motion_engine and self.motion_engine.mode == "CALIBRATION":  # 캘리브레이션 중이라면
                            self.calib_step = "FINISHED"  # 바로 수집 종료 단계로 점프
                except asyncio.TimeoutError:  # 송신 패킷이 없다면
                    pass  # 유유히 통과

                if self.active_session and self.camera:  # 카메라 구동 중
                    ret, frame = self.camera.read()  # 이미지 수확

                    if ret and frame is not None:  # 정상 이미지
                        curr_time = time.time()  # 현 시점 마커
                        fps = round(1.0 / (curr_time - prev_time), 1) if (curr_time - prev_time) > 0 else 30.0  # 속도 변환
                        prev_time = curr_time  # 마커 갱신
                        frame_count += 1  # 루프 상승

                        if frame_count % 2 == 0:  # 소켓 과부하 방지 압축 스킵
                            def encode_jpg(img):  # 인코더
                                small = cv2.resize(img, (400, 300))  # 소형 스케일 다운
                                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])  # 품질 최하
                                return base64.b64encode(buf).decode("utf-8")  # 문자열 리턴

                            last_frame_b64 = await asyncio.to_thread(encode_jpg, frame)  # 쓰레딩 인코딩 하달

                        keypoints = await asyncio.to_thread(
                            self.skeleton_engine.extract_keypoints, frame
                        )  # 엔진에 프레임 밀어넣고 좌표 확보
                        
                        yolo_detected = keypoints is not None and len(keypoints) > 0  # 화면에 사람 여부
                        is_occluded = not self.check_full_body(keypoints) if yolo_detected else True  # 12관절 가림 상태 도출

                        current_mode = self.motion_engine.mode if self.motion_engine else "CALIBRATION"  # 모드 식별
                        
                        # [UI 캐시 적용] 가림 발생 시 화면의 횟수/진행률이 0으로 초기화되지 않도록 캐시본 사용
                        left_data = self.last_left_data.copy()  # 직전 상태 카피
                        right_data = self.last_right_data.copy()  # 직전 상태 카피

                        # ------------------------------------------------------
                        # [캘리브레이션 모드 FSM] 전신확인 -> 3초 카운트다운 -> 데이터 수집
                        # ------------------------------------------------------
                        remaining_sec = 0.0  # 잔여 시간 용기
                        if self.active_session and current_mode == "CALIBRATION":  # 캘리브레이션 로직
                            
                            if is_occluded:  # 가려짐이 발생하면 즉결 초기화
                                self.calib_step = "FULL_BODY_CHECK"  # 대기 상태로 강제 복귀
                                self.calib_start_time = None  # 시계 초기화
                                self.buf_timestamps.clear()  # NPZ 백업 쓰레기 데이터 폐기
                                self.buf_values.clear()  # NPZ 백업 쓰레기 데이터 폐기
                                self.buf_keypoints.clear()  # NPZ 백업 쓰레기 데이터 폐기
                                self.calib_values_left.clear()  # 오염된 좌측 계산 수치 폐기
                                self.calib_values_right.clear()  # 오염된 우측 계산 수치 폐기
                            else:  # 가려짐이 없는 양호한 상태라면
                                if self.calib_step == "FULL_BODY_CHECK":  # 대기 상태에서
                                    self.calib_step = "COUNTDOWN"  # 카운팅 전입
                                    self.calib_start_time = time.time()  # 시간 기록 시작

                                elif self.calib_step == "COUNTDOWN":  # 3초 카운트다운 페이즈
                                    elapsed = time.time() - self.calib_start_time  # 지나간 시간 산출
                                    remaining_sec = max(0.0, 3.0 - elapsed)  # 타이머 UI 표시용
                                    if elapsed >= 3.0:  # 3초가 지나면
                                        self.calib_step = "COLLECTING"  # 진짜 수집 페이즈 전환
                                        self.calib_start_time = time.time()  # 5초 측정을 위해 타이머 리셋

                                elif self.calib_step == "COLLECTING":  # 수집 페이즈
                                    elapsed = time.time() - self.calib_start_time  # 수집 개시 후 흐른 시간
                                    remaining_sec = max(0.0, self.calib_duration - elapsed)  # 설정된 타임(5초) 차감

                                    kpt_matrix = [
                                        [kp["x"], kp["y"], kp.get("score", 0.0)]
                                        for kp in keypoints
                                    ] if yolo_detected else []  # 행렬 변환
                                    
                                    # [핵심] 임계값 연산을 위해 순수 좌표뿐 아니라 관절의 각도/위치 수치도 추출합니다.
                                    kpt_map = {kp["id"]: kp for kp in keypoints} if yolo_detected else {}  # ID 매핑
                                    val_left = self.motion_engine._compute_side_value("left", kpt_map) if self.motion_engine else None  # 모션엔진 내장 함수로 수치 산출
                                    val_right = self.motion_engine._compute_side_value("right", kpt_map) if self.motion_engine else None  # 우측도 산출

                                    if val_left is not None:  # 에러값이 아니라면
                                        self.calib_values_left.append(val_left)  # 임계 계산용 리스트에 보존
                                    if val_right is not None:  # 에러값이 아니라면
                                        self.calib_values_right.append(val_right)  # 임계 계산용 리스트에 보존

                                    self.buf_timestamps.append(round(time.time() - self.session_start_time, 3))  # 타이밍 누적
                                    self.buf_values.append([val_left or 0.0, val_right or 0.0])  # 모션수치 함께 저장
                                    self.buf_keypoints.append(kpt_matrix)  # 17개 관절 매트릭스 삽입

                                    if elapsed >= self.calib_duration:  # 5초 제한을 채웠다면
                                        self.calib_step = "FINISHED"  # 캘리브레이션 성료 판정

                            if self.calib_step == "FINISHED":  # 성료 시 동작
                                custom_th = self._calculate_and_save_thresholds()  # [추가] 모인 데이터로 임계값 계산 후 JSON 저장 로직 호출
                                summary = self._save_current_session_data()  # 좌표를 NPZ로 디스크 적재
                                await websocket.send(
                                    json.dumps({
                                        "type": "CALIBRATION_FINISHED",  # 클라이언트로 종료 패킷 전송
                                        "player_id": self.player_id,  # 소유주 아이디
                                        "thresholds": custom_th,  # 산출된 임계값 데이터 함께 송달
                                        "summary": summary,  # 요약 정보
                                    })
                                )  # 발송
                                self.active_session = False  # 구동 회로 폐쇄

                        # ------------------------------------------------------
                        # [실제 운동 모드 (MAIN)] FSM 분석 및 카운팅
                        # ------------------------------------------------------
                        if self.active_session and current_mode == "MAIN":  # 운동 본 궤도 진행 시
                            if not yolo_detected or is_occluded:  # 가려짐 발생 시
                                pass  # 상단에서 복사한 기존 캐시 데이터가 클라이언트로 날아가므로 화면의 진행 게이지가 얼어붙은 채 유지됨
                            else:  # 정상 시퀀스 시
                                result = self.motion_engine.process_keypoints(keypoints)  # 분석 깊게 진입
                                if result:  # 결과 창출 시
                                    current_mode = result["mode"]  # 모드 갱신
                                    left_data, right_data = result["left"], result["right"]  # 최신 상태 업데이트
                                    
                                    self.last_left_data = left_data.copy()  # 화면 정지 대비 캐시 덮어쓰기
                                    self.last_right_data = right_data.copy()  # 화면 정지 대비 캐시 덮어쓰기

                                    kpt_matrix = [
                                        [kp["x"], kp["y"], kp.get("score", 0.0)]
                                        for kp in keypoints
                                    ]  # 행렬 인코딩
                                    self.buf_timestamps.append(round(time.time() - self.session_start_time, 3))  # 시퀀스 트래킹
                                    self.buf_values.append([left_data["val"] or 0.0, right_data["val"] or 0.0])  # 모션 각도 기록
                                    self.buf_keypoints.append(kpt_matrix)  # 위치 기록

                                    if (
                                        (left_data["rep_count"] >= self.target_reps and self.target_reps > 0)
                                        or (right_data["rep_count"] >= self.target_reps and self.target_reps > 0)
                                    ):  # 어느 한쪽이라도 채우면 통과 (편측운동 대응)
                                        summary = self._save_current_session_data()  # 유효 기록 저장 명령
                                        if summary:  # 응답 시
                                            await websocket.send(
                                                json.dumps({"type": "SESSION_FINISHED", "summary": summary})
                                            )  # 성공 패킷 타격
                                        self.active_session = False  # 세션 완전 차단

                        # 3. 실시간 UI 동기화용 상태 패킷 전송
                        payload = {
                            "type": "POSE_UPDATE",  # 소켓 토픽
                            "mode": current_mode,  # 소속 세션
                            "left": left_data,  # 좌측 패킷
                            "right": right_data,  # 우측 패킷
                            "keypoints": keypoints if yolo_detected else [],  # 라인 시각화 좌표
                            "fps": fps,  # 프레임
                            "yolo_detected": yolo_detected,  # 감지 스펙
                            "is_occluded": is_occluded,  # 클라이언트 UI 렌더링에 사용할 전신 가림 여부 플래그
                            "frame_b64": last_frame_b64,  # 영상 인코딩본
                            "calib_step": self.calib_step,  # 대기 상태 모드
                            "remaining_sec": round(remaining_sec, 1),  # 남은 타임
                        }  # 최종 정리
                        await websocket.send(json.dumps(payload))  # 페이로드 분출

                await asyncio.sleep(0.001)  # 비동기 시스템 락 해제 타임

        except websockets.exceptions.ConnectionClosed:  # 클라이언트 브라우저 닫힘
            print("[SocketServer] 클라이언트 연결 종료")  # 종료 안내
            if self.active_session and len(self.buf_timestamps) > 0:  # 미처 못 다한 데이터 저장
                self._save_current_session_data()  # 디스크 백업 실시
        except Exception as e:  # 예외 사항 발생
            print(f"[SocketServer Error] {e}")  # 에러 표출
            traceback.print_exc()  # 상세 콜스택 출력
        finally:  # 안전보장 처리
            self.stop_camera()  # 렌즈 반납 완료

    async def run(self):
        async with websockets.serve(self.handle_client, self.host, self.port):  # 소켓 주소 바인딩
            print(f"[SocketServer] AI Headless 서버 구동 중 (ws://{self.host}:{self.port})")  # 실행 시작 문구
            await asyncio.Future()  # 무한 엔진 대기


if __name__ == "__main__":
    server = ExerciseSocketServer()  # 서버 객체 할당
    try:
        asyncio.run(server.run())  # 구동
    except KeyboardInterrupt:  # 유저 강제 종료 (Ctrl+C)
        print("\n[SocketServer] 서버 종료")  # 종료 노티