# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `types`에서 `SimpleNamespace` 기능을 가져옵니다.
from types import SimpleNamespace
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# `unittest.mock`에서 `AsyncMock`, `Mock`, `patch` 기능을 가져옵니다.
from unittest.mock import AsyncMock, Mock, patch

# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np
# `websockets.exceptions`에서 `ConnectionClosedOK` 기능을 가져옵니다.
from websockets.exceptions import ConnectionClosedOK

# `tests.test_skeleton_batch`에서 `load_model_free_module` 기능을 가져옵니다.
from tests.test_skeleton_batch import load_model_free_module
# `core.data_manager`에서 `DataManager` 기능을 가져옵니다.
from core.data_manager import DataManager


# COCO 관절 좌표 목록 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def keypoints():
    # 현재 함수 처리를 끝내고 반복·조건식 `[{'id': i, 'x': 0.5, 'y': 0.5, 'score': 0.9} for i in range(17)]`으로 구성한 값을 호출한 쪽으로 반환합니다.
    return [{'id': i, 'x': 0.5, 'y': 0.5, 'score': 0.9} for i in range(17)]


# `OfflineVideoTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class OfflineVideoTests(unittest.IsolatedAsyncioTestCase):
    # 데코레이터 `classmethod`를 바로 아래 함수·클래스에 적용합니다.
    @classmethod
    # 설정 up class 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def setUpClass(cls):
        # `skeleton` 값에 `load_model_free_module('core/skeleton_engine.py')` 호출 결과를 저장합니다. 대상: `skeleton`.
        skeleton = load_model_free_module('core/skeleton_engine.py')
        # `cls.module` 값에 `load_model_free_module('network/socket_server.py', {'core.skeleton_engine': skeleton})` 호출 결과를 저장합니다. 대상: `cls.module`.
        cls.module = load_model_free_module('network/socket_server.py', {'core.skeleton_engine': skeleton})

    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # OpenCV 영상 입력 장치에 `Mock()` 호출 결과를 저장합니다. 대상: `self.cap`.
        self.cap = Mock()
        # return 측정값에 True을 저장합니다. 대상: `self.cap.isOpened.return_value`.
        self.cap.isOpened.return_value = True
        # `self.cap.get.side_effect` 값에 계산식 `lambda prop: 30.0 if prop == 5 else 3`의 결과를 저장합니다. 대상: `self.cap.get.side_effect`.
        self.cap.get.side_effect = lambda prop: 30.0 if prop == 5 else 3
        # OpenCV BGR 영상 프레임에 `np.zeros((100, 200, 3), dtype=np.uint8)` 호출 결과를 저장합니다. 대상: `frame`.
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        # `self.cap.read.side_effect` 값에 계산식 `[(True, frame)] * 3 + [(False, None)]`의 결과를 저장합니다. 대상: `self.cap.read.side_effect`.
        self.cap.read.side_effect = [(True, frame)] * 3 + [(False, None)]
        # `self.module.cv2` 값에 `SimpleNamespace(VideoCapture=Mock(return_value=self.cap), CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)` 호출 결과를 저장합니다. 대상: `self.module.cv2`.
        self.module.cv2 = SimpleNamespace(VideoCapture=Mock(return_value=self.cap),
                                           # 구성 항목: cap prop 초당 프레임 수(`CAP_PROP_FPS`)에 5; cap prop 프레임 횟수(`CAP_PROP_FRAME_COUNT`)에 7을 지정합니다.
                                           CAP_PROP_FPS=5, CAP_PROP_FRAME_COUNT=7)
        # 원본을 안전하게 교체하기 위한 임시 파일 경로에 `tempfile.TemporaryDirectory()` 호출 결과를 저장합니다. 대상: `self.temporary`.
        self.temporary = tempfile.TemporaryDirectory()
        # `self.manager` 값에 None을 저장합니다. 대상: `self.manager`.
        self.manager = None

        # 생성 manager 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def create_manager(**kwargs):
            # `kwargs['log_root']` 값에 `self.temporary.name` 값(`self.temporary.name`)을 저장합니다. 대상: `kwargs['log_root']`.
            kwargs['log_root'] = self.temporary.name
            # `self.manager` 값에 `DataManager(**kwargs)` 호출 결과를 저장합니다. 대상: `self.manager`.
            self.manager = DataManager(**kwargs)
            # 현재 함수 처리를 끝내고 `self.manager` 값(`self.manager`)을 호출한 쪽으로 반환합니다.
            return self.manager

        # 데이터 manager에 `Mock(side_effect=create_manager)` 호출 결과를 저장합니다. 대상: `self.module.DataManager`.
        self.module.DataManager = Mock(side_effect=create_manager)
        # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.controller`.
        self.controller = None

        # 생성 제어기 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def create_controller(data_manager, motion_engine):
            # 캘리브레이션·본 운동 진행 제어기에 `SimpleNamespace(data_manager=data_manager, motion_engine=motion_engine, timer_start=0.0, check_occlusion=lambda kpts: False, last_left_data={'val': 0.1}, last_right_data…` 호출 결과를 저장합니다. 대상: `self.controller`.
            self.controller = SimpleNamespace(
                # 구성 항목: 환자 정보 및 세션 로그 관리자(`data_manager`)에 환자 정보 및 세션 로그 관리자(`data_manager`); 관절 측정값 계산 및 반복 판정 엔진(`motion_engine`)에 관절 측정값 계산 및 반복 판정 엔진(`motion_engine`)을 지정합니다.
                data_manager=data_manager, motion_engine=motion_engine,
                # 이 줄의 입력 인수: `kpts` = `kpts` 값. 기본값·형식은 아래 선언을 따릅니다.
                timer_start=0.0, check_occlusion=lambda kpts: False,
                # 구성 항목: last 왼쪽 데이터(`last_left_data`)에 키 'val'로 구성한 사전; last 오른쪽 데이터(`last_right_data`)에 키 'val'로 구성한 사전; `val` 값(`'val'`)에 0.1; `val` 값(`'val'`)에 0.2을 지정합니다.
                last_left_data={'val': 0.1}, last_right_data={'val': 0.2},
                # 구성 항목: 처리 진입점 프레임(`process_main_frame`)에 `Mock(return_value=({'val': 0.1}, {'val': 0.2}, False))` 호출 결과; return 측정값(`return_value`)에 순서·구성을 지정한 `({'val': 0.1}, {'val': 0.2}, False)`; `val` 값(`'val'`)에 0.1; `val` 값(`'val'`)에 0.2을 지정합니다.
                process_main_frame=Mock(return_value=({'val': 0.1}, {'val': 0.2}, False)))
            # 현재 함수 처리를 끝내고 캘리브레이션·본 운동 진행 제어기(`self.controller`)을 호출한 쪽으로 반환합니다.
            return self.controller

        # 측정 세션 제어기에 생성 제어기(`create_controller`)을 저장합니다. 대상: `self.module.SessionController`.
        self.module.SessionController = create_controller
        # 서버에 `self.module.ExerciseSocketServer.__new__(self.module.ExerciseSocketServer)` 호출 결과를 저장합니다. 대상: `self.server`.
        self.server = self.module.ExerciseSocketServer.__new__(self.module.ExerciseSocketServer)
        # 생성자를 생략한 모의 서버에도 연결 정리 완료 신호 저장소를 준비합니다.
        self.server._connection_done = {}
        # 캘리브레이션·본 운동 진행 제어기에 None을 저장합니다. 대상: `self.server.controller`.
        self.server.controller = None
        # `self.server.camera` 값에 None을 저장합니다. 대상: `self.server.camera`.
        self.server.camera = None
        # 현재 기록 중인 세션 로그에 None을 저장합니다. 대상: `self.server._active_log`.
        self.server._active_log = None
        # 중지 camera에 `Mock()` 호출 결과를 저장합니다. 대상: `self.server.stop_camera`.
        self.server.stop_camera = Mock()
        # 일괄 처리 sizes에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.batch_sizes`.
        self.batch_sizes = []

        # extract 일괄 처리 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def extract_batch(frames):
            # `self.batch_sizes.append(len(frames))`을 호출하여 항목 하나를 끝에 추가합니다.
            self.batch_sizes.append(len(frames))
            # 현재 함수 처리를 끝내고 반복·조건식 `[keypoints() for _ in frames]`으로 구성한 값을 호출한 쪽으로 반환합니다.
            return [keypoints() for _ in frames]

        # 사람 검출·관절 추출 AI 엔진에 `SimpleNamespace(extract_keypoints_batch=Mock(side_effect=extract_batch))` 호출 결과를 저장합니다. 대상: `self.server.skeleton_engine`.
        self.server.skeleton_engine = SimpleNamespace(extract_keypoints_batch=Mock(side_effect=extract_batch))
        # 현재 서버의 측정 자원을 점유한 연결에 None을 저장합니다. 대상: `self.server._session_owner`.
        self.server._session_owner = None
        # 백그라운드 AI 모델 준비 관리자에 `SimpleNamespace(start=lambda : None, engine=self.server.skeleton_engine, snapshot=lambda : {'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': T…` 호출 결과를 저장합니다. 대상: `self.server.model_loader`.
        self.server.model_loader = SimpleNamespace(
            # 구성 항목: 시작(`start`)에 계산식 `lambda : None`의 결과; 생성 또는 검증할 분석 엔진(`engine`)에 사람 검출·관절 추출 AI 엔진(`self.server.skeleton_engine`)을 지정합니다.
            start=lambda: None, engine=self.server.skeleton_engine,
            # 구성 항목: `snapshot` 값(`snapshot`)에 계산식 `lambda : {'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True}`의 결과; `type` 값(`'type'`)에 'SERVER_STATUS'; 모델 단계(`'model_state'`)에 'READY'; `phase` 값(`'phase'`)에 'READY'; 준비 상태(`'ready'`)에 True을 지정합니다.
            snapshot=lambda: {'type': 'SERVER_STATUS', 'model_state': 'READY', 'phase': 'READY', 'ready': True})
        # 저장 측정 세션 파일에 `Mock(return_value={'saved_frames': 3, 'saved_reps_count': 0})` 호출 결과를 저장합니다. 대상: `self.server._save_session_files`.
        self.server._save_session_files = Mock(return_value={'saved_frames': 3, 'saved_reps_count': 0})
        # WebSocket 연결에 `SimpleNamespace(send=AsyncMock(), remote_address=('127.0.0.1', 1234))` 호출 결과를 저장합니다. 대상: `self.ws`.
        self.ws = SimpleNamespace(send=AsyncMock(), remote_address=('127.0.0.1', 1234))
        # 수신 또는 전송할 JSON 메시지 사전에 `dict(type='CMD_SET_SESSION', video_path='example.mp4', exercise_name='biceps_curl', mode='MAIN', target_reps=10, batch_size=2)` 호출 결과를 저장합니다. 대상: `self.packet`.
        self.packet = dict(type='CMD_SET_SESSION', video_path='example.mp4',
                           # 구성 항목: 운동 설정의 식별 이름(`exercise_name`)에 'biceps_curl'; 실행 모드(`mode`)에 'MAIN'; 목표 반복 횟수(`target_reps`)에 10; 한 번에 추론 또는 학습할 표본 수(`batch_size`)에 2을 지정합니다.
                           exercise_name='biceps_curl', mode='MAIN', target_reps=10, batch_size=2)

    # 각 테스트가 끝나면 패치·임시 자원을 해제하여 다음 테스트에 영향을 남기지 않습니다.
    def tearDown(self):
        # 조건 `self.manager and self.manager.session_log`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.manager and self.manager.session_log:
            # `self.manager.finish_session([], status='TEST')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.manager.finish_session([], status='TEST')
        # `self.temporary.cleanup()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.temporary.cleanup()

    # 회귀 테스트: `test_full_and_partial_batches_are_both_analyzed_and_completed` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_full_and_partial_batches_are_both_analyzed_and_completed(self):
        # `self.server.process_offline_video(self.ws, self.packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, self.packet)
        # 검증문 `self.assertEqual(self.batch_sizes, [2, 1])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.batch_sizes, [2, 1])
        # 검증문 `self.cap.release.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.cap.release.assert_called_once()
        # 검증문 `self.assertEqual(self.controller.process_main_frame.call_count, 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.controller.process_main_frame.call_count, 3)
        # 검증문 `self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])
        # 함수 또는 처리 단계의 결과에 JSON 문자열을 해석한 Python 값(`json.loads(self.ws.send.call_args_list[-1].args[0])`)을 저장합니다. 대상: `result`.
        result = json.loads(self.ws.send.call_args_list[-1].args[0])
        # 검증문 `self.assertEqual(result['type'], 'VIDEO_ANALYSIS_COMPLETED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result['type'], 'VIDEO_ANALYSIS_COMPLETED')
        # 검증문 `self.assertEqual(result['total_frames_analyzed'], 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result['total_frames_analyzed'], 3)
        # 검증문 `self.assertIsNone(self.server.controller)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertIsNone(self.server.controller)

    # 회귀 테스트: `test_client_stop_interrupts_offline_extraction_and_acknowledges_save` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_client_stop_interrupts_offline_extraction_and_acknowledges_save(self):
        # `self.ws.recv` 값에 `AsyncMock(return_value=json.dumps({'type': 'CMD_STOP_SESSION', 'reason': 'USER_STOP', 'request_id': 'stop-video'}))` 호출 결과를 저장합니다. 대상: `self.ws.recv`.
        self.ws.recv = AsyncMock(return_value=json.dumps({
            # 구성 항목: `type` 값(`'type'`)에 'CMD_STOP_SESSION'; `reason` 값(`'reason'`)에 'USER_STOP'; 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 'stop-video'을 지정합니다.
            'type': 'CMD_STOP_SESSION', 'reason': 'USER_STOP', 'request_id': 'stop-video'}))
        # `self.server.process_offline_video(self.ws, self.packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, self.packet)
        # 함수 또는 처리 단계의 결과에 JSON 문자열을 해석한 Python 값(`json.loads(self.ws.send.call_args_list[-1].args[0])`)을 저장합니다. 대상: `result`.
        result = json.loads(self.ws.send.call_args_list[-1].args[0])
        # 검증문 `self.assertEqual(result['type'], 'SESSION_FINISHED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result['type'], 'SESSION_FINISHED')
        # 검증문 `self.assertEqual(result['request_id'], 'stop-video')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result['request_id'], 'stop-video')
        # 검증문 `self.assertEqual(result['status'], 'STOPPED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(result['status'], 'STOPPED')
        # 검증문 `self.assertEqual(self.batch_sizes, [2])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.batch_sizes, [2])
        # 검증문 `self.controller.process_main_frame.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.controller.process_main_frame.assert_not_called()
        # 검증문 `self.cap.release.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.cap.release.assert_called_once()

    # 회귀 테스트: `test_offline_main_ignores_legacy_controller_finish_flag` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_offline_main_ignores_legacy_controller_finish_flag(self):
        # `original` 값에 측정 세션 제어기(`self.module.SessionController`)을 저장합니다. 대상: `original`.
        original = self.module.SessionController
        # 제어기 포함 기존 호환 flag 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
        def controller_with_legacy_flag(**kwargs):
            # 함수 또는 처리 단계의 결과에 `original(**kwargs)` 호출 결과를 저장합니다. 대상: `result`.
            result = original(**kwargs)
            # return 측정값에 순서·구성을 지정한 `({'val': 0.1}, {'val': 0.2}, True)`을 저장합니다. 대상: `result.process_main_frame.return_value`.
            result.process_main_frame.return_value = ({'val': .1}, {'val': .2}, True)
            # 현재 함수 처리를 끝내고 함수 또는 처리 단계의 결과(`result`)을 호출한 쪽으로 반환합니다.
            return result
        # 측정 세션 제어기에 제어기 포함 기존 호환 flag(`controller_with_legacy_flag`)을 저장합니다. 대상: `self.module.SessionController`.
        self.module.SessionController = controller_with_legacy_flag
        # `self.server.process_offline_video(self.ws, self.packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, self.packet)
        # 검증문 `self.assertEqual(self.controller.process_main_frame.call_count, 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.controller.process_main_frame.call_count, 3)

    # 회귀 테스트: `test_missing_person_does_not_shift_following_frame_timestamp` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_missing_person_does_not_shift_following_frame_timestamp(self):
        # `self.server.skeleton_engine.extract_keypoints_batch.side_effect` 값에 순서·구성을 지정한 `[[keypoints(), None], [keypoints()]]`을 저장합니다. 대상: `self.server.skeleton_engine.extract_keypoints_batch.side_effect`.
        self.server.skeleton_engine.extract_keypoints_batch.side_effect = [[keypoints(), None], [keypoints()]]
        # `self.server.process_offline_video(self.ws, self.packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, self.packet)
        # 검증문 `self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])
        # 검증문 `self.assertTrue(np.isnan(self.manager.buf_raw_keypoints[1][:, :2]).all())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(np.isnan(self.manager.buf_raw_keypoints[1][:, :2]).all())

    # 회귀 테스트: `test_capture_released_even_when_batch_inference_fails` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_capture_released_even_when_batch_inference_fails(self):
        # `self.server.skeleton_engine.extract_keypoints_batch.side_effect` 값에 `RuntimeError('inference failed')` 호출 결과를 저장합니다. 대상: `self.server.skeleton_engine.extract_keypoints_batch.side_effect`.
        self.server.skeleton_engine.extract_keypoints_batch.side_effect = RuntimeError('inference failed')
        # `self.assertRaisesRegex(RuntimeError, 'inference failed')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            # `self.server.process_offline_video(self.ws, self.packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.server.process_offline_video(self.ws, self.packet)
        # 검증문 `self.cap.release.assert_called_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.cap.release.assert_called_once()
        # 검증문 `self.server._save_session_files.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server._save_session_files.assert_not_called()

    # 회귀 테스트: `test_invalid_batch_size_is_reported_before_opening_video` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_invalid_batch_size_is_reported_before_opening_video(self):
        # `self.server.process_offline_video(self.ws, {**self.packet, 'batch_size': 0})`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, {**self.packet, 'batch_size': 0})
        # 검증문 `self.module.cv2.VideoCapture.assert_not_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.module.cv2.VideoCapture.assert_not_called()
        # 검증문 `self.assertEqual(json.loads(self.ws.send.call_args.args[0])['code'], 'INVALID_BATCH_SIZE')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(json.loads(self.ws.send.call_args.args[0])['code'], 'INVALID_BATCH_SIZE')

    # 회귀 테스트: `test_unusable_fps_falls_back_to_30` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_unusable_fps_falls_back_to_30(self):
        # `self.cap.get.side_effect` 값에 계산식 `lambda prop: float('nan') if prop == 5 else 3`의 결과를 저장합니다. 대상: `self.cap.get.side_effect`.
        self.cap.get.side_effect = lambda prop: float('nan') if prop == 5 else 3
        # `self.server.process_offline_video(self.ws, self.packet)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, self.packet)
        # 검증문 `self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 30, 2 / 30])

    # 회귀 테스트: `test_actual_csv_npz_mp4_and_summary_use_requested_session_name` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_actual_csv_npz_mp4_and_summary_use_requested_session_name(self):
        # `pathlib`에서 `Path` 기능을 가져옵니다.
        from pathlib import Path
        # 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
        import csv
        # 저장 측정 세션 파일에 `self.module.ExerciseSocketServer._save_session_files.__get__(self.server)` 호출 결과를 저장합니다. 대상: `self.server._save_session_files`.
        self.server._save_session_files = self.module.ExerciseSocketServer._save_session_files.__get__(self.server)
        # `self.server.process_offline_video(self.ws, {**self.packet, 'session': 2, 'round': 3, 'set': 4, 'record_video': False})`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, {**self.packet,
            # 구성 항목: 측정 세션(`'session'`)에 2; `round` 값(`'round'`)에 3; 설정(`'set'`)에 4; 영상 녹화 여부(`'record_video'`)에 False을 지정합니다.
            'session': 2, 'round': 3, 'set': 4, 'record_video': False,
            # 서버 입력이 실제 DataManager와 프레임 파일까지 전달되는지 확인합니다.
            'data_version': 'v2.3.4(27)', 'registration': '02', 'play_session_id': 'PLAY_OFFLINE',
            'exercise_record_id': 'EX_OFFLINE', 'slot': 5})
        # 함수 또는 처리 단계의 결과에 JSON 문자열을 해석한 Python 값(`json.loads(self.ws.send.call_args_list[-1].args[0])`)을 저장합니다. 대상: `result`.
        result = json.loads(self.ws.send.call_args_list[-1].args[0])
        # 저장·종료 결과 요약에 계산식 `result['summary']`의 결과를 저장합니다. 대상: `summary`.
        summary = result['summary']
        # 검증문 `self.assertTrue(summary['record_video'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(summary['record_video'])
        # 검증문 `self.assertEqual(Path(summary['frame_csv']).stem, 'session2_round3_set4')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(Path(summary['frame_csv']).stem, f"R03_S05_Set04_Play_Camera_{summary['sensor_file_id']}")
        # 공통 ID가 입력대로 유지되고 서버의 측정 ID와 혼동되지 않는지 확인합니다.
        self.assertEqual(summary['play_session_id'], 'PLAY_OFFLINE')
        self.assertEqual(summary['exercise_record_id'], 'EX_OFFLINE')
        self.assertEqual(summary['data_version'], 'v2.3.4(27)')
        self.assertEqual(summary['registration'], '02')
        # `('frame_csv', 'skeleton_npz', 'video_mp4', 'gamedata_csv')`에서 항목을 차례로 꺼내 `key`에 넣고 아래 처리를 반복합니다.
        for key in ('frame_csv', 'skeleton_npz', 'video_mp4', 'gamedata_csv'):
            # 검증문 `self.assertTrue(Path(summary[key]).is_relative_to(Path(self.temporary.name)))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(Path(summary[key]).is_relative_to(Path(self.temporary.name)))
            # 검증문 `self.assertTrue(Path(summary[key]).is_file())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(Path(summary[key]).is_file())
        # `Path(summary['frame_csv']).open(newline='', encoding='utf-8-sig')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with Path(summary['frame_csv']).open(newline='', encoding='utf-8-sig') as file:
            # 검증문 `self.assertEqual(len(list(csv.DictReader(file))), 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(len(list(csv.DictReader(file))), 3)
        # `np.load(summary['skeleton_npz'])`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with np.load(summary['skeleton_npz']) as data:
            # 검증문 `self.assertEqual(data['keypoints'].shape, (3, 17, 3))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(data['keypoints'].shape, (3, 17, 3))
            # 검증문 `np.testing.assert_array_equal(data['video_frame_idx'], [0, 1, 2])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            np.testing.assert_array_equal(data['video_frame_idx'], [0, 1, 2])
        # 검증문 `self.assertEqual(summary['video_frames'], 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(summary['video_frames'], 3)

    # 회귀 테스트: `test_dispatch_reports_offline_failure_instead_of_silent_disconnect` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_dispatch_reports_offline_failure_instead_of_silent_disconnect(self):
        # 처리 파일 분석 영상에 `AsyncMock(side_effect=RuntimeError('batch failed'))` 호출 결과를 저장합니다. 대상: `self.server.process_offline_video`.
        self.server.process_offline_video = AsyncMock(side_effect=RuntimeError('batch failed'))
        # `self.ws.recv` 값에 `AsyncMock(side_effect=[json.dumps(self.packet), ConnectionClosedOK(None, None)])` 호출 결과를 저장합니다. 대상: `self.ws.recv`.
        self.ws.recv = AsyncMock(side_effect=[json.dumps(self.packet),
                                             # 앞줄에서 시작한 식·인수·자료 구조를 `ConnectionClosedOK(None, None)])` 내용으로 이어서 구성합니다.
                                             ConnectionClosedOK(None, None)])
        # `patch.object(self.module.os.path, 'isfile', return_value=True), patch.object(self.module.traceback, 'print_exc')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch.object(self.module.os.path, 'isfile', return_value=True), \
                patch.object(self.module.traceback, 'print_exc'):
            # `self.server.handle_client(self.ws)`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
            await self.server.handle_client(self.ws)
        # 검증문 `self.server.process_offline_video.assert_awaited_once()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server.process_offline_video.assert_awaited_once()
        # `response` 값에 JSON 문자열을 해석한 Python 값(`json.loads(self.ws.send.call_args.args[0])`)을 저장합니다. 대상: `response`.
        response = json.loads(self.ws.send.call_args.args[0])
        # 검증문 `self.assertEqual(response['type'], 'ERROR')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['type'], 'ERROR')
        # 검증문 `self.assertEqual(response['code'], 'VIDEO_ANALYSIS_FAILED')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(response['code'], 'VIDEO_ANALYSIS_FAILED')
        # 검증문 `self.server.stop_camera.assert_called()`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.server.stop_camera.assert_called()

    # 회귀 테스트: `test_offline_recording_keeps_source_fps_and_not_batch_wall_time` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    async def test_offline_recording_keeps_source_fps_and_not_batch_wall_time(self):
        # 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
        import cv2
        # `self.cap.get.side_effect` 값에 계산식 `lambda prop: 6.0 if prop == 5 else 3`의 결과를 저장합니다. 대상: `self.cap.get.side_effect`.
        self.cap.get.side_effect = lambda prop: 6.0 if prop == 5 else 3
        # 저장 측정 세션 파일에 `self.module.ExerciseSocketServer._save_session_files.__get__(self.server)` 호출 결과를 저장합니다. 대상: `self.server._save_session_files`.
        self.server._save_session_files = self.module.ExerciseSocketServer._save_session_files.__get__(self.server)
        # `self.server.process_offline_video(self.ws, {**self.packet, 'record_video': True})`을 호출하여 해당 단계의 작업을 수행합니다. 완료될 때까지 비동기로 기다립니다.
        await self.server.process_offline_video(self.ws, {**self.packet, 'record_video': True})
        # 저장·종료 결과 요약에 계산식 `json.loads(self.ws.send.call_args_list[-1].args[0])['summary']`의 결과를 저장합니다. 대상: `summary`.
        summary = json.loads(self.ws.send.call_args_list[-1].args[0])['summary']
        # 검증문 `self.assertEqual(self.manager.buf_timestamps, [0, 1 / 6, 2 / 6])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.manager.buf_timestamps, [0, 1 / 6, 2 / 6])
        # OpenCV 영상 입력 장치에 `cv2.VideoCapture(summary['video_mp4'])` 호출 결과를 저장합니다. 대상: `cap`.
        cap = cv2.VideoCapture(summary['video_mp4'])
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 검증문 `self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(cap.get(cv2.CAP_PROP_FRAME_COUNT), 3)
            # 검증문 `self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 6)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 6)
            # 검증문 `self.assertAlmostEqual(summary['video_duration_sec'], 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertAlmostEqual(summary['video_duration_sec'], 0.5)
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
