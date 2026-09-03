# ==============================================================================
# [파일 정보]
# 파일명: network/socket_server.py
# 작성자: 개발자 (Developer)
# 설명: 웹캠 영상 실시간 처리, AI 모듈 연동 및 WebSocket 클라이언트 통신을 담당하는 서버
# ==============================================================================

import os  # 운영체제 환경 변수 및 경로 제어를 위한 내장 모듈 로드
# [오류 해결 1] MMCV C++ 확장 모듈(DLL) 비활성화를 통한 ImportError 원천 차단
os.environ["MMCV_WITH_OPS"] = "0"  # mmcv 모듈이 임포트되기 전에 환경 변수 강제 설정

import asyncio  # 비동기 I/O 이벤트 루프 제어 모듈 로드
import base64  # 비디오 프레임 Base64 인코딩을 위한 모듈 로드
from datetime import datetime  # 세션 타임스탬프 생성을 위한 모듈 로드
import json  # WebSocket JSON 패킷 데이터 파싱을 위한 모듈 로드
import sys  # 파이썬 인터프리터 경로 설정용 모듈 로드
import threading  # 카메라 프레임 비동기 수집용 멀티스레딩 모듈 로드
import time  # 시간 측정 및 FPS 계산용 타임 모듈 로드
import traceback  # 예외 스택 트레이스 출력용 모듈 로드
import cv2  # 카메라 프레임 획득 및 이미지 처리 라이브러리 로드
import numpy as np  # 배열 처리 및 PyTorch 전역 허용 객체 전달을 위한 라이브러리 로드
import torch  # 모델 로드 환경 설정을 위한 PyTorch 프레임워크 로드
import websockets  # 비동기 WebSocket 서버 구축 모듈 로드

# [오류 해결 2] PyTorch 2.6+ weights_only=True 정책으로 인한 UnpicklingError 해결
try:  # 버전별 호환성을 위한 예외 처리 블록
    from torch.serialization import add_safe_globals  # PyTorch 직렬화 모듈에서 전역 허용 함수 로드
    add_safe_globals([np.core.multiarray._reconstruct])  # numpy 배열 복원 객체를 안전 목록에 명시적 추가
    add_safe_globals([np.ndarray])  # 기본 numpy 다차원 배열 객체를 안전 목록에 명시적 추가
except ImportError:  # add_safe_globals 함수가 없는 하위 버전 PyTorch일 경우
    pass  # 추가 작업 없이 통과

_original_torch_load = torch.load  # 혹시 모를 내부 로드를 대비하여 기존 PyTorch 로드 함수 원본 백업

def _patched_torch_load(*args, **kwargs):  # 체크포인트 로드 보안 경고를 끄기 위한 래퍼 함수 정의
    kwargs['weights_only'] = False  # 안전 경고 해제 파라미터 강제 삽입
    return _original_torch_load(*args, **kwargs)  # 패치된 설정으로 원본 함수 실행

torch.load = _patched_torch_load  # 시스템 전역의 torch.load 함수를 패치된 함수로 교체
try:  # 내부 serialization 모듈 확인
    import torch.serialization  # 직렬화 전용 모듈 로드
    torch.serialization.load = _patched_torch_load  # 내부 load 함수도 동일하게 패치 교체
except Exception:  # 접근 불가 예외 발생 시
    pass  # 무시하고 통과

import warnings  # 시스템 경고 메세지 제어용 모듈 로드
warnings.filterwarnings("ignore", category=FutureWarning)  # 미래 버전 호환성 경고 메시지 출력 차단
warnings.filterwarnings("ignore", category=UserWarning)  # mmcv 모듈 탐색 실패 등 불필요한 유저 경고 메시지 출력 차단

import pycocotools  # mmpose 내부의 xtcocotools 종속성 에러를 방지하기 위한 코코툴스 로드
sys.modules['xtcocotools'] = pycocotools  # 이름을 매핑하여 정상 로드되도록 강제

current_file_path = os.path.abspath(__file__)  # 현재 파일의 절대 경로 산출
network_dir = os.path.dirname(current_file_path)  # 네트워크 폴더 경로 산출
root_dir = os.path.dirname(network_dir)  # 최상위 프로젝트 루트 경로 산출

if root_dir not in sys.path:  # 모듈 경로에 루트 디렉토리가 없다면
    sys.path.insert(0, root_dir)  # 루트 경로를 시스템 패스에 최우선 등록

from core.data_manager import DataManager, load_exercise_config  # 데이터 매니저 및 설정 로더 수입
from core.motion_engine import MotionEngine  # FSM 기반 운동 상태 분석 엔진 수입
from core.skeleton_engine import SkeletonEngine  # 관절 좌표 추출기 수입
from core.session_controller import SessionController  # 세션 및 가림 제어 컨트롤러 수입
from utils.filters import RealtimeEMAFilter  # 실시간 노이즈 제거 필터 수입


class ThreadedCamera:  # 카메라 캡처 지연 방지용 백그라운드 스레드 클래스

    def __init__(self, camera_index=0):  # 카메라 초기화 생성자
        self.camera_index = camera_index  # 사용할 카메라 인덱스 할당
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # Windows DirectShow 기반 카메라 연결
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 프레임 밀림 방지를 위해 버퍼 크기 최소화
        self.grabbed, self.frame = self.cap.read()  # 첫 번째 프레임 추출
        self.started = False  # 스레드 동작 상태 플래그 초기화
        self.read_lock = threading.Lock()  # 데이터 동시 접근 방지용 락 객체 할당

    def start(self):  # 프레임 수집 데몬 스레드 가동 메서드
        if self.started:  # 중복 실행 방지
            return self  # 자신 객체 반환
        self.started = True  # 동작 플래그 활성화
        self.thread = threading.Thread(target=self.update, daemon=True)  # 메인 스레드 종료 시 동시 종료되는 데몬 스레드 생성
        self.thread.start()  # 스레드 실행
        return self  # 체이닝 지원

    def update(self):  # 백그라운드 무한 루프 갱신 메서드
        while self.started:  # 구동 중일 때 무한 반복
            grabbed, frame = self.cap.read()  # 카메라 센서 버퍼에서 최신 프레임 획득
            with self.read_lock:  # 외부 읽기 로직과 충돌하지 않도록 락 획득
                self.grabbed = grabbed  # 프레임 획득 성공 여부 갱신
                self.frame = frame  # 최신 프레임 갱신

    def read(self):  # 최신 프레임 복사본 반환 메서드
        with self.read_lock:  # 락 획득 후 안전하게 반환
            if not self.grabbed or self.frame is None:  # 프레임이 유효하지 않은 경우
                return False, None  # 에러 반환
            return True, self.frame.copy()  # 데이터 오염 방지를 위해 사본 반환

    def stop(self):  # 카메라 스레드 정지 메서드
        self.started = False  # 무한 루프 탈출 조건 부여
        if self.cap and self.cap.isOpened():  # 장치가 켜져 있다면
            self.cap.release()  # 시스템 리소스 반납


class ExerciseSocketServer:  # AI 운동 제어 및 클라이언트 통신을 총괄하는 웹소켓 서버 클래스

    def __init__(self, host="127.0.0.1", port=8080):  # 서버 설정 및 엔진 생성자
        self.host = host  # 접속을 허용할 IP 할당
        self.port = port  # 통신 포트 번호 할당
        
        print("[SocketServer] SkeletonEngine AI 모델 로드 중...")  # 로딩 안내
        self.skeleton_engine = SkeletonEngine(yolo_interval=5)  # 5프레임 주기 YOLO 검출이 적용된 AI 엔진 메모리 로드
        self.camera = None  # 카메라 스레드 객체 공간 확보
        self.controller = None  # 세션 통합 컨트롤러 공간 확보
        
        self.last_frame_b64 = None  # 클라이언트로 보낼 인코딩 이미지 캐시 초기화
        self.filter_engine = None  # 노이즈 제거 필터 객체 공간 확보

    def start_camera(self, camera_index: int = 0):  # 지정된 인덱스로 카메라 재가동
        if self.camera:  # 기존 카메라 존재 시
            self.camera.stop()  # 먼저 해제
        self.camera = ThreadedCamera(camera_index).start()  # 새 스레드로 렌즈 개방

    def stop_camera(self):  # 동작 중인 카메라 안전 종료
        if self.camera:  # 카메라 객체 유효 시
            self.camera.stop()  # 스레드 종료
            self.camera = None  # 참조 삭제

    def init_session(self, data: dict):  # 클라이언트 명령에 따른 시스템 초기화 메서드
        player_id = data.get("player_id", "patient_1")  # 세션 환자 번호 할당
        patient_name = data.get("patient_name", "미지정")  # 세션 환자 이름 할당
        exercise_name = data.get("exercise_name", "biceps_curl")  # 수행할 운동명 할당
        mode = data.get("mode", "CALIBRATION")  # 구동 목적(캘리브레이션/메인) 할당
        target_reps = data.get("target_reps", 10)  # 목표 반복 횟수 할당
        cal_time = data.get("cal_time", 10.0)  # 캘리브레이션 지속 시간 할당
        
        self.start_camera(data.get("camera_index", 0))  # 카메라 초기화 가동
        
        data_manager = DataManager(player_id=player_id, patient_name=patient_name)  # 환자 프로필용 파일 제어기 생성
        exercise_config = load_exercise_config(exercise_name)  # 타겟 관절 등이 명시된 기본 JSON 로드
        custom_thresholds = data_manager.get_custom_threshold(exercise_name)  # 이전에 계산된 환자 전용 임계값 탐색

        if custom_thresholds and "left" not in custom_thresholds and "start_val" in custom_thresholds:  # 뎁스 구조가 깨진 포맷 방어
            custom_thresholds = {"left": custom_thresholds, "right": custom_thresholds}  # 좌우 양방향 데이터로 강제 래핑하여 에러 방지

        motion_engine = MotionEngine(exercise_config, custom_thresholds, mode=mode, target_reps=target_reps)  # 관절 수치 연산 및 FSM 카운팅 엔진 인스턴스화
        self.controller = SessionController(data_manager=data_manager, motion_engine=motion_engine, calib_duration=cal_time)  # 상태 제어 및 데이터 수집을 총괄할 세션 컨트롤러 구축
        self.filter_engine = RealtimeEMAFilter(max_jump=0.15, alpha=0.6)  # 프레임 간 튀는 값을 잡아줄 실시간 스무딩 필터 생성

    async def handle_client(self, websocket):  # 클라이언트 연결 시 실행되는 메인 비동기 소켓 루프
        prev_time = time.time()  # 실시간 FPS 계산용 타이머
        frame_count = 0  # 영상 압축 생략 주기를 위한 카운터

        try:  # 오류 발생 시 자원 반환을 위한 try 블록
            while True:  # 클라이언트와 연결된 동안 무한 루프 가동
                try:  # 명령 수신 논블로킹 대기 블록
                    message = await asyncio.wait_for(websocket.recv(), timeout=0.001)  # 1ms 대기하며 패킷 수신 시도
                    data = json.loads(message)  # JSON 문자열 역직렬화
                    
                    if data.get("type") == "CMD_SET_SESSION":  # 모드 설정 패킷 수신 시
                        self.init_session(data)  # 추출된 딕셔너리로 세션 엔진 세팅 완료
                    elif data.get("type") == "CMD_STOP_CALIBRATION":  # 유저 강제 캘리브레이션 종료 요청 시
                        if self.controller and self.controller.mode == "CALIBRATION":  # 현재 모드가 캘리브레이션이라면
                            self.controller.calib_step = "FINISHED"  # 컨트롤러 상태를 즉시 수집 종료 단계로 점프
                except asyncio.TimeoutError:  # 송신 패킷이 없다면
                    pass  # 루프 지연 없이 통과

                if self.controller and self.camera:  # 세션이 활성화되고 카메라가 구동 중일 때
                    ret, frame = self.camera.read()  # 백그라운드 스레드에서 최신 프레임 획득

                    if ret and frame is not None:  # 프레임 획득 성공 시
                        curr_time = time.time()  # 현재 시각 갱신
                        fps = round(1.0 / (curr_time - prev_time), 1) if (curr_time - prev_time) > 0 else 30.0  # 타임 차분을 이용한 FPS 연산
                        prev_time = curr_time  # 다음 계산을 위해 타이머 최신화
                        frame_count += 1  # 렌더링 프레임 횟수 증가

                        if frame_count % 2 == 0:  # 네트워크 과부하 방지를 위한 2프레임 당 1회 압축 스킵
                            def encode_jpg(img):  # 내부 JPEG 압축 함수
                                small = cv2.resize(img, (400, 300))  # 저화질 스케일 다운
                                _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 30])  # 품질 30 압축
                                return base64.b64encode(buf).decode("utf-8")  # Base64화 하여 반환

                            self.last_frame_b64 = await asyncio.to_thread(encode_jpg, frame)  # 논블로킹 스레드 위임 실행

                        raw_keypoints = await asyncio.to_thread(self.skeleton_engine.extract_keypoints, frame)  # 딥러닝 추출기 가동 및 원본 좌표 획득
                        keypoints = self.filter_engine.update(raw_keypoints)  # 필터 엔진을 통과시켜 이상치(Jitter)가 제거된 좌표 확보
                        
                        yolo_detected = keypoints is not None and len(keypoints) > 0  # 화면 내 인체 추론 유무 판별
                        is_occluded = self.controller.check_occlusion(keypoints) if yolo_detected else True  # 12관절 신뢰도 기반 화면 가림 상태 도출
                        current_mode = self.controller.motion_engine.mode  # 모션 엔진 구동 목적 추출

                        left_data, right_data = self.controller.last_left_data, self.controller.last_right_data  # 캐시 붕괴 방지를 위해 이전 프레임 데이터 로드
                        remaining_sec = 0.0  # 타이머용 잔여 시간 초기화

                        # network/socket_server.py 의 handle_client 내부 (CALIBRATION 분기)
                        if current_mode == "CALIBRATION":
                            cal_result = self.controller.process_calibration_frame(keypoints, is_occluded)
                            remaining_sec = cal_result["remaining_sec"]
                            
                            # 1. 캘리브레이션 데이터 수집 중일 때 NPZ용 버퍼 누적 추가[cite: 5, 6]
                            if cal_result["calib_step"] == "COLLECTING" and not is_occluded and keypoints:
                                vl = self.controller.calib_vals_left[-1] if self.controller.calib_vals_left else 0.0
                                vr = self.controller.calib_vals_right[-1] if self.controller.calib_vals_right else 0.0
                                self._append_to_buffers({"val": vl}, {"val": vr}, keypoints)

                            # 2. 수집 완료 시 파일 저장 및 패킷 발송[cite: 6]
                            if cal_result["calib_step"] == "FINISHED":
                                ex_name = getattr(self.controller.motion_engine, "exercise_name", "biceps_curl")
                                custom_th = self.controller.compute_and_save_thresholds(ex_name)
                                summary = self._save_session_files()
                                
                                await websocket.send(json.dumps({
                                    "type": "CALIBRATION_FINISHED", 
                                    "player_id": self.controller.data_manager.player_id, 
                                    "thresholds": custom_th, 
                                    "summary": summary
                                }))
                                self.controller = None
                                continue  # 하단의 중복 POSE_UPDATE 전송 방지[cite: 6]
                            
                        elif current_mode == "MAIN":  # 현재 동작 모드가 실전 운동(Test)인 경우
                            left_data, right_data, is_finished = self.controller.process_main_frame(keypoints, is_occluded)  # FSM 엔진 처리 (가려진 경우 이전 캐시 유지됨)
                            
                            if is_occluded == False and keypoints:  # 전신이 정상 노출된 경우 데이터 보존
                                self._append_to_buffers(left_data, right_data, keypoints)  # NPZ 저장을 위해 시계열 버퍼에 좌표 누적

                            if is_finished:  # 어느 한쪽이라도 목표 횟수를 100% 충족한 경우
                                summary = self._save_session_files()  # 현재까지 수행한 모든 궤적과 회차 기록을 CSV/NPZ 파일로 영구 보존
                                await websocket.send(json.dumps({"type": "SESSION_FINISHED", "summary": summary}))  # 목표 달성 종료 알림 패킷 타격
                                self.controller = None  # 엔진 회로 닫기

                        payload = {  # 실시간 클라이언트 화면 동기화를 위한 통합 데이터 패킷 구성
                            "type": "POSE_UPDATE",  # 소켓 라우팅 명칭
                            "mode": current_mode,  # 소속 세션
                            "left": left_data,  # 좌측 상태
                            "right": right_data,  # 우측 상태
                            "keypoints": keypoints if yolo_detected else [],  # 렌더링용 스무딩 좌표
                            "fps": fps,  # 송출 프레임
                            "yolo_detected": yolo_detected,  # 감지 스펙
                            "is_occluded": is_occluded,  # 클라이언트 UI 렌더링에 사용할 가림 여부 플래그
                            "frame_b64": self.last_frame_b64,  # 압축 인코딩본
                            "calib_step": self.controller.calib_step if self.controller else "FINISHED",  # 대기 상태 모드
                            "remaining_sec": remaining_sec,  # 남은 타임
                        }  # 딕셔너리 구조 생성 완료
                        await websocket.send(json.dumps(payload))  # 웹소켓 송신 실행

                await asyncio.sleep(0.001)  # 비동기 시스템 락 해제 타임

        except websockets.exceptions.ConnectionClosed:  # 웹 브라우저 이탈
            if self.controller and len(self.controller.data_manager.csv_path.name) > 0:  # 미처 못 다한 데이터가 있다면
                self._save_session_files()  # 디스크에 안전히 보존
        except Exception:  # 그 외 파이프라인 에러 상황
            traceback.print_exc()  # 상세 추적 출력
        finally:  # 안전장치
            self.stop_camera()  # 렌즈 반납 확인 사살

    def _append_to_buffers(self, left_data: dict, right_data: dict, keypoints: list):  # 타임스탬프와 데이터를 메모리에 추가하는 메서드
        kpt_matrix = [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in keypoints]  # 17x3 행렬 압축
        elapsed_time = round(time.time() - self.controller.timer_start, 3)  # 기준 시간 대비 초 환산
        
        self.controller.data_manager.skeleton_dir.parent.mkdir(parents=True, exist_ok=True)  # 저장 폴더 안전성 재검사
        self.controller.data_manager.buf_timestamps = getattr(self.controller.data_manager, "buf_timestamps", []) + [elapsed_time]  # 타임스탬프 누적
        self.controller.data_manager.buf_values = getattr(self.controller.data_manager, "buf_values", []) + [[left_data.get("val") or 0.0, right_data.get("val") or 0.0]]  # 모션 수치 배열 누적
        self.controller.data_manager.buf_keypoints = getattr(self.controller.data_manager, "buf_keypoints", []) + [kpt_matrix]  # 공간 좌표 배열 누적

    def _save_session_files(self) -> dict:  # 수집된 버퍼 데이터를 디스크에 기록하는 메서드
        dm = self.controller.data_manager  # 코드 단축을 위한 데이터 매니저 맵핑
        if not dm or not getattr(dm, "buf_timestamps", []):  # 저장할 이력이 없으면
            return None  # 종료 회피

        session_id = f"SESSION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"  # 세션 타임 아이디 할당
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 기록 시간 포맷팅

        left_reps = self.controller.motion_engine.fsm_left.completed_reps_history  # 좌측 회차 상세 로그 가져오기
        right_reps = self.controller.motion_engine.fsm_right.completed_reps_history  # 우측 회차 상세 로그 가져오기

        all_rep_rows = [{**r, "session_id": session_id, "timestamp": timestamp_str} for r in left_reps + right_reps]  # 세션 메타 정보 병합 합치기
        dm.save_rep_details_csv(all_rep_rows)  # 운동 완료 상세 정보를 CSV로 기록

        dm.save_trajectory_npz(
            session_id=session_id,  # 고유 키
            exercise_name=self.controller.motion_engine.config.get("exercise_name", "unknown") if self.controller.motion_engine else "unknown",  # 종목
            timestamps=dm.buf_timestamps,  # 시계열
            values=dm.buf_values,  # 모의 밸류
            keypoints=dm.buf_keypoints,  # 추출 좌표계
        )  # 궤적 원본을 압축 파일로 디스크 드랍

        return {"session_id": session_id, "saved_reps_count": len(all_rep_rows)}  # 요약 데이터 리턴

    async def run(self):  # 서버 개시 메서드
        async with websockets.serve(self.handle_client, self.host, self.port):  # 소켓 주소 점유
            print(f"[SocketServer] AI Headless 서버 구동 중 (ws://{self.host}:{self.port})")  # 실행 시작 안내
            await asyncio.Future()  # 무한 대기


if __name__ == "__main__":  # 모듈 직접 실행 판별
    server = ExerciseSocketServer()  # 서버 구축
    try:  # 키보드 인터럽트 대기
        asyncio.run(server.run())  # 비동기 구동
    except KeyboardInterrupt:  # 유저 강제 종료 시
        print("\n[SocketServer] 서버 종료")  # 종료 멘트