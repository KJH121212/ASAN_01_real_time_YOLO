# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# `datetime`에서 `datetime`, `timezone` 기능을 가져옵니다.
from datetime import datetime, timezone
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest
# `unittest.mock`에서 `patch` 기능을 가져옵니다.
from unittest.mock import patch

# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# `core.data_manager`에서 `DataManager` 기능을 가져옵니다.
from core.data_manager import DataManager
# `core.session_log`에서 `KST`, `default_log_root` 기능을 가져옵니다.
from core.session_log import KST, default_log_root


# 가동 범위 기준에 반복·조건식 `{side: {'start_val': 0.1, 'target_val': 0.9} for side in ('left', 'right')}`으로 구성한 값을 저장합니다. 대상: `THRESHOLDS`.
THRESHOLDS = {side: {'start_val': 0.1, 'target_val': 0.9} for side in ('left', 'right')}


# `rows` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def rows(path):
    # `Path(path).open(newline='', encoding='utf-8-sig')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
    with Path(path).open(newline='', encoding='utf-8-sig') as file:
        # 현재 함수 처리를 끝내고 `list(csv.DictReader(file))` 호출 결과를 호출한 쪽으로 반환합니다.
        return list(csv.DictReader(file))


# `SessionLogTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class SessionLogTests(unittest.TestCase):
    # 각 테스트 전에 독립된 입력·객체를 준비하여 앞 테스트의 상태가 결과에 섞이지 않게 합니다.
    def setUp(self):
        # 원본을 안전하게 교체하기 위한 임시 파일 경로에 `tempfile.TemporaryDirectory()` 호출 결과를 저장합니다. 대상: `self.temporary`.
        self.temporary = tempfile.TemporaryDirectory()
        # 최상위 경로에 `Path(self.temporary.name)` 호출 결과를 저장합니다. 대상: `self.root`.
        self.root = Path(self.temporary.name)
        # `self.manager` 값에 `DataManager('0001', 'Test Patient', log_root=self.root, patient_metadata={'height_cm': 170, 'weight_kg': 65})` 호출 결과를 저장합니다. 대상: `self.manager`.
        self.manager = DataManager('0001', 'Test Patient', log_root=self.root,
                                   # 구성 항목: 환자의 키·몸무게 등 부가 정보(`patient_metadata`)에 키 'height_cm', 'weight_kg'로 구성한 사전; 키(cm)(`'height_cm'`)에 170; 몸무게(kg)(`'weight_kg'`)에 65을 지정합니다.
                                   patient_metadata={'height_cm': 170, 'weight_kg': 65})
        # 시작에 `datetime(2026, 1, 1, 12, 0, tzinfo=KST)` 호출 결과를 저장합니다. 대상: `self.start`.
        self.start = datetime(2026, 1, 1, 12, 0, tzinfo=KST)

    # 각 테스트가 끝나면 패치·임시 자원을 해제하여 다음 테스트에 영향을 남기지 않습니다.
    def tearDown(self):
        # 조건 `self.manager.session_log`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.manager.session_log:
            # `self.manager.finish_session([], status='TEST')`을 호출하여 해당 단계의 작업을 수행합니다.
            self.manager.finish_session([], status='TEST')
        # `self.temporary.cleanup()`을 호출하여 해당 단계의 작업을 수행합니다.
        self.temporary.cleanup()

    # `begin` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def begin(self, **kwargs):
        # 현재 함수 처리를 끝내고 `self.manager.begin_session('biceps_curl', 'MAIN', session=1, started_at=self.start, **kwargs)` 호출 결과를 호출한 쪽으로 반환합니다.
        return self.manager.begin_session('biceps_curl', 'MAIN', session=1, started_at=self.start, **kwargs)

    # 회귀 테스트: `test_video_file_cannot_be_used_as_log_directory` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_video_file_cannot_be_used_as_log_directory(self):
        # 영상에 계산식 `self.root / 'source.mp4'`의 결과를 저장합니다. 대상: `video`.
        video = self.root / 'source.mp4'
        # `video.write_bytes(b'original video')`을 호출하여 해당 단계의 작업을 수행합니다.
        video.write_bytes(b'original video')
        # `(video, video / 'logs')`에서 항목을 차례로 꺼내 `destination`에 넣고 아래 처리를 반복합니다.
        for destination in (video, video / 'logs'):
            # `self.assertRaisesRegex(NotADirectoryError, 'Video File Path')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.assertRaisesRegex(NotADirectoryError, 'Video File Path'):
                # `DataManager('0001', log_root=destination)`을 호출하여 해당 단계의 작업을 수행합니다.
                DataManager('0001', log_root=destination)
        # 검증문 `self.assertEqual(video.read_bytes(), b'original video')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(video.read_bytes(), b'original video')

    # `append` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def append(self, frame=0, missing=False, video_index=-1):
        # 계산에 사용할 관절 또는 좌표 묶음에 반복·조건식 `[{'id': i, 'x': 0.25, 'y': 0.75, 'score': 0.9} for i in range(17)]`으로 구성한 값을 저장합니다. 대상: `points`.
        points = [{'id': i, 'x': 0.25, 'y': 0.75, 'score': 0.9} for i in range(17)]
        # `self.manager.session_log.append_frame(frame / 30, frame, None if missing else points, {'val': None if missing else 0.5, 'rep_count': 2}, {'val': 0.7}, 'RELATIVE_Y', THRE…`을 호출하여 해당 단계의 작업을 수행합니다.
        self.manager.session_log.append_frame(frame / 30, frame, None if missing else points,
            # 구성 항목: `val` 값(`'val'`)에 `missing`이면 `None`, 아니면 `0.5`; 인정된 누적 반복 횟수(`'rep_count'`)에 2; `val` 값(`'val'`)에 0.7을 지정합니다.
            {'val': None if missing else 0.5, 'rep_count': 2}, {'val': 0.7},
            # 구성 항목: 현재 가동 범위 설정의 변경 번호(`threshold_version`)에 3을 지정합니다.
            'RELATIVE_Y', THRESHOLDS, threshold_version=3,
            # 구성 항목: 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`)에 결측(`missing`); 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)에 영상 index(`video_index`)을 지정합니다.
            is_occluded=missing, video_frame_idx=video_index)

    # 회귀 테스트: `test_csv_npz_and_metadata_follow_patient_date_exercise_structure` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_csv_npz_and_metadata_follow_patient_date_exercise_structure(self):
        # `info` 값에 `self.begin(round_number=2, set_number=3)` 호출 결과를 저장합니다. 대상: `info`.
        info = self.begin(round_number=2, set_number=3)
        # 테스트에서 기대하는 결과에 계산식 `self.root / '0001' / '260101' / 'biceps_curl' / 'session1_round2_set3.csv'`의 결과를 저장합니다. 대상: `expected`.
        expected = self.root / '0001' / '260101' / 'biceps_curl' / 'session1_round2_set3.csv'
        # 검증문 `self.assertEqual(Path(info['frame_csv']), expected)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(Path(info['frame_csv']), expected)
        # `self.append()`을 호출하여 항목 하나를 끝에 추가합니다.
        self.append()
        # `self.append(1, missing=True)`을 호출하여 항목 하나를 끝에 추가합니다.
        self.append(1, missing=True)
        # 저장·종료 결과 요약에 `self.manager.finish_session([])` 호출 결과를 저장합니다. 대상: `summary`.
        summary = self.manager.finish_session([])
        # 검증문 `self.assertEqual(summary['saved_frames'], 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(summary['saved_frames'], 2)
        # 영상 프레임 묶음에 `rows(expected)` 호출 결과를 저장합니다. 대상: `frames`.
        frames = rows(expected)
        # 검증문 `self.assertEqual(frames[0]['left_value'], '0.5')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(frames[0]['left_value'], '0.5')
        # 검증문 `self.assertEqual(frames[0]['threshold_version'], '3')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(frames[0]['threshold_version'], '3')
        # 검증문 `self.assertEqual(frames[1]['left_value'], '')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(frames[1]['left_value'], '')
        # 검증문 `self.assertEqual(frames[1]['is_occluded'], 'True')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(frames[1]['is_occluded'], 'True')
        # `np.load(info['skeleton_npz'], allow_pickle=False)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with np.load(info['skeleton_npz'], allow_pickle=False) as data:
            # 검증문 `self.assertEqual(data['keypoints'].shape, (2, 17, 3))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(data['keypoints'].shape, (2, 17, 3))
            # 검증문 `np.testing.assert_array_equal(data['frame_idx'], [0, 1])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            np.testing.assert_array_equal(data['frame_idx'], [0, 1])
            # 검증문 `self.assertTrue(np.isnan(data['keypoints'][1, :, :2]).all())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(np.isnan(data['keypoints'][1, :, :2]).all())
            # 검증문 `self.assertEqual(data['timestamps'].dtype, np.float64)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(data['timestamps'].dtype, np.float64)
            # 검증문 `self.assertAlmostEqual(data['timestamp_ms'][1], 1000 / 30)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertAlmostEqual(data['timestamp_ms'][1], 1000 / 30)
        # 환자 또는 세션 부가 정보에 계산식 `rows(self.manager.metadata_path)[0]`의 결과를 저장합니다. 대상: `metadata`.
        metadata = rows(self.manager.metadata_path)[0]
        # 검증문 `self.assertEqual(metadata['patient_id'], '0001')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(metadata['patient_id'], '0001')
        # 검증문 `self.assertEqual(metadata['height_cm'], '170.0')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(metadata['height_cm'], '170.0')
        # 검증문 `self.assertEqual(rows(summary['gamedata_csv'])[0]['calories_kcal'], '')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(rows(summary['gamedata_csv'])[0]['calories_kcal'], '')

    # 회귀 테스트: `test_actual_mp4_uses_same_basename_and_frame_indices` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_actual_mp4_uses_same_basename_and_frame_indices(self):
        # `info` 값에 `self.begin(record_video=True)` 호출 결과를 저장합니다. 대상: `info`.
        info = self.begin(record_video=True)
        # `range(3)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
        for i in range(3):
            # `index` 값에 `self.manager.session_log.write_video_frame(np.full((24, 32, 3), i * 40, np.uint8), elapsed_sec=i / 30)` 호출 결과를 저장합니다. 대상: `index`.
            index = self.manager.session_log.write_video_frame(
                # 구성 항목: `elapsed_sec` 값(`elapsed_sec`)에 계산식 `i / 30`의 결과를 지정합니다.
                np.full((24, 32, 3), i * 40, np.uint8), elapsed_sec=i / 30)
            # 검증문 `self.assertEqual(index, i)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(index, i)
            # `self.append(i, video_index=index)`을 호출하여 항목 하나를 끝에 추가합니다.
            self.append(i, video_index=index)
        # `self.manager.finish_session([])`을 호출하여 해당 단계의 작업을 수행합니다.
        self.manager.finish_session([])
        # `('frame_csv', 'skeleton_npz', 'video_mp4')`에서 항목을 차례로 꺼내 `field`에 넣고 아래 처리를 반복합니다.
        for field in ('frame_csv', 'skeleton_npz', 'video_mp4'):
            # 검증문 `self.assertTrue(Path(info[field]).is_file())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(Path(info[field]).is_file())
            # 검증문 `self.assertEqual(Path(info[field]).stem, 'session1_round1_set1')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(Path(info[field]).stem, 'session1_round1_set1')
        # OpenCV 영상 입력 장치에 `cv2.VideoCapture(info['video_mp4'])` 호출 결과를 저장합니다. 대상: `cap`.
        cap = cv2.VideoCapture(info['video_mp4'])
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 검증문 `self.assertTrue(cap.isOpened())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertTrue(cap.isOpened())
            # 검증문 `self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 3)
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()
        # `np.load(info['skeleton_npz'])`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with np.load(info['skeleton_npz']) as data:
            # 검증문 `np.testing.assert_array_equal(data['video_frame_idx'], [0, 1, 2])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            np.testing.assert_array_equal(data['video_frame_idx'], [0, 1, 2])

    # 회귀 테스트: `test_repeat_does_not_overwrite_and_finish_is_idempotent` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_repeat_does_not_overwrite_and_finish_is_idempotent(self):
        # `first` 값에 `self.begin()` 호출 결과를 저장합니다. 대상: `first`.
        first = self.begin()
        # `self.append()`을 호출하여 항목 하나를 끝에 추가합니다.
        self.append()
        # `self.manager.finish_session([])`을 호출하여 해당 단계의 작업을 수행합니다.
        self.manager.finish_session([])
        # `original` 값에 `Path(first['frame_csv']).read_bytes()` 호출 결과를 저장합니다. 대상: `original`.
        original = Path(first['frame_csv']).read_bytes()
        # `self.manager.finish_session([])`을 호출하여 해당 단계의 작업을 수행합니다.
        self.manager.finish_session([])
        # `second` 값에 `self.begin()` 호출 결과를 저장합니다. 대상: `second`.
        second = self.begin()
        # 검증문 `self.assertTrue(second['frame_csv'].endswith('session1_round1_set1_take2.csv'))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(second['frame_csv'].endswith('session1_round1_set1_take2.csv'))
        # `self.manager.finish_session([], status='STOPPED')`을 호출하여 해당 단계의 작업을 수행합니다.
        self.manager.finish_session([], status='STOPPED')
        # 검증문 `self.assertEqual(Path(first['frame_csv']).read_bytes(), original)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(Path(first['frame_csv']).read_bytes(), original)
        # 검증문 `self.assertEqual(len(rows(first['gamedata_csv'])), 2)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(rows(first['gamedata_csv'])), 2)

    # 회귀 테스트: `test_live_video_preserves_elapsed_time_between_analysis_frames` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_live_video_preserves_elapsed_time_between_analysis_frames(self):
        # `info` 값에 `self.begin(record_video=True)` 호출 결과를 저장합니다. 대상: `info`.
        info = self.begin(record_video=True)
        # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.manager.session_log`)을 저장합니다. 대상: `log`.
        log = self.manager.session_log
        # OpenCV BGR 영상 프레임에 `np.zeros((24, 32, 3), dtype=np.uint8)` 호출 결과를 저장합니다. 대상: `frame`.
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        # `indices` 값에 반복·조건식 `[log.write_video_frame(frame, elapsed_sec=time) for time in (0.5, 0.6, 0.7)]`으로 구성한 값을 저장합니다. 대상: `indices`.
        indices = [log.write_video_frame(frame, elapsed_sec=time) for time in (0.5, 0.6, 0.7)]
        # 검증문 `self.assertEqual(indices, [0, 3, 6])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(indices, [0, 3, 6])
        # 저장·종료 결과 요약에 `self.manager.finish_session([])` 호출 결과를 저장합니다. 대상: `summary`.
        summary = self.manager.finish_session([])
        # 검증문 `self.assertEqual(summary['video_time_origin_sec'], 0.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(summary['video_time_origin_sec'], 0.5)
        # OpenCV 영상 입력 장치에 `cv2.VideoCapture(info['video_mp4'])` 호출 결과를 저장합니다. 대상: `cap`.
        cap = cv2.VideoCapture(info['video_mp4'])
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 검증문 `self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 7)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 7)
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()

    # 회귀 테스트: `test_recorded_duration_matches_real_time_at_different_input_rates` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_recorded_duration_matches_real_time_at_different_input_rates(self):
        # 입력이 6 FPS 또는 60 FPS여도 4초 촬영 영상은 약 4초 동안 재생되어야 합니다.
        # `(6, 13, 30, 60)`에서 항목을 차례로 꺼내 `input_fps`에 넣고 아래 처리를 반복합니다.
        for input_fps in (6, 13, 30, 60):
            # `self.subTest(input_fps=input_fps)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.subTest(input_fps=input_fps):
                # `info` 값에 `self.begin(record_video=True)` 호출 결과를 저장합니다. 대상: `info`.
                info = self.begin(record_video=True)
                # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.manager.session_log`)을 저장합니다. 대상: `log`.
                log = self.manager.session_log
                # OpenCV BGR 영상 프레임에 `np.zeros((24, 32, 3), dtype=np.uint8)` 호출 결과를 저장합니다. 대상: `frame`.
                frame = np.zeros((24, 32, 3), dtype=np.uint8)
                # `range(input_fps * 4)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
                for i in range(input_fps * 4):
                    # `log.write_video_frame(frame, elapsed_sec=0.25 + i / input_fps)`을 호출하여 해당 단계의 작업을 수행합니다.
                    log.write_video_frame(frame, elapsed_sec=0.25 + i / input_fps)
                # 저장·종료 결과 요약에 `self.manager.finish_session([], end_elapsed_sec=4.25)` 호출 결과를 저장합니다. 대상: `summary`.
                summary = self.manager.finish_session([], end_elapsed_sec=4.25)
                # OpenCV 영상 입력 장치에 `cv2.VideoCapture(info['video_mp4'])` 호출 결과를 저장합니다. 대상: `cap`.
                cap = cv2.VideoCapture(info['video_mp4'])
                # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
                try:
                    # `playback` 값에 계산식 `cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)`의 결과를 저장합니다. 대상: `playback`.
                    playback = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    # 검증문 `self.assertAlmostEqual(playback, 4.0, delta=1 / 30)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertAlmostEqual(playback, 4.0, delta=1 / 30)
                    # 검증문 `self.assertEqual(summary['video_duration_sec'], playback)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertEqual(summary['video_duration_sec'], playback)
                # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
                finally:
                    # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
                    cap.release()

    # 회귀 테스트: `test_irregular_capture_holds_previous_image_and_includes_final_wait` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_irregular_capture_holds_previous_image_and_includes_final_wait(self):
        # `info` 값에 `self.begin(record_video=True)` 호출 결과를 저장합니다. 대상: `info`.
        info = self.begin(record_video=True)
        # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.manager.session_log`)을 저장합니다. 대상: `log`.
        log = self.manager.session_log
        # 느린 입력 구간·빠른 연속 입력·중지 직전 대기를 포함해 시간 보존을 확인합니다.
        # `times` 값에 순서·구성을 지정한 `(0.5, 0.7, 0.92, 1.4, 1.41, 2.3, 3.0)`을 저장합니다. 대상: `times`.
        times = (0.5, 0.7, 0.92, 1.4, 1.41, 2.3, 3.0)
        # `enumerate(times)`에서 항목을 차례로 꺼내 `(i, timestamp)`에 넣고 아래 처리를 반복합니다.
        for i, timestamp in enumerate(times):
            # `log.write_video_frame(np.full((24, 32, 3), i * 32, np.uint8), timestamp)`을 호출하여 해당 단계의 작업을 수행합니다.
            log.write_video_frame(np.full((24, 32, 3), i * 32, np.uint8), timestamp)
        # 저장·종료 결과 요약에 `self.manager.finish_session([], status='STOPPED', end_elapsed_sec=5.5)` 호출 결과를 저장합니다. 대상: `summary`.
        summary = self.manager.finish_session([], status='STOPPED', end_elapsed_sec=5.5)
        # 검증문 `self.assertEqual(summary['video_frames'], 150)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(summary['video_frames'], 150)
        # 검증문 `self.assertEqual(summary['duration_sec'], 5.5)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(summary['duration_sec'], 5.5)
        # OpenCV 영상 입력 장치에 `cv2.VideoCapture(info['video_mp4'])` 호출 결과를 저장합니다. 대상: `cap`.
        cap = cv2.VideoCapture(info['video_mp4'])
        # 영상 프레임 묶음에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `frames`.
        frames = []
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
            while True:
                # `(ok, frame)` 값에 `cap.read()` 호출 결과를 저장합니다. 대상: `(ok, frame)`.
                ok, frame = cap.read()
                # 조건 `not ok`을 만족하는 경우에만 아래 분기를 실행합니다.
                if not ok:
                    # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                    break
                # `frames.append(frame)`을 호출하여 항목 하나를 끝에 추가합니다.
                frames.append(frame)
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
            cap.release()
        # 검증문 `self.assertEqual(len(frames), 150)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(len(frames), 150)
        # MP4 압축으로 픽셀값이 조금 달라질 수 있으므로 허용 범위 안에서 영상 내용과 길이를 함께 확인합니다.
        # 검증문 `self.assertAlmostEqual(float(frames[5].mean()), 0, delta=6)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(float(frames[5].mean()), 0, delta=6)
        # 검증문 `self.assertAlmostEqual(float(frames[20].mean()), 64, delta=6)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(float(frames[20].mean()), 64, delta=6)
        # 검증문 `self.assertAlmostEqual(float(frames[35].mean()), 128, delta=6)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(float(frames[35].mean()), 128, delta=6)
        # 검증문 `self.assertAlmostEqual(float(frames[-1].mean()), 192, delta=6)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(float(frames[-1].mean()), 192, delta=6)

    # 회귀 테스트: `test_video_rejects_invalid_or_backward_timestamps` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_video_rejects_invalid_or_backward_timestamps(self):
        # `self.begin(record_video=True)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.begin(record_video=True)
        # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`self.manager.session_log`)을 저장합니다. 대상: `log`.
        log = self.manager.session_log
        # OpenCV BGR 영상 프레임에 `np.zeros((24, 32, 3), dtype=np.uint8)` 호출 결과를 저장합니다. 대상: `frame`.
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        # `(-1, float('nan'), float('inf'))`에서 항목을 차례로 꺼내 `invalid`에 넣고 아래 처리를 반복합니다.
        for invalid in (-1, float('nan'), float('inf')):
            # `self.subTest(timestamp=invalid), self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.subTest(timestamp=invalid), self.assertRaises(ValueError):
                # `log.write_video_frame(frame, invalid)`을 호출하여 해당 단계의 작업을 수행합니다.
                log.write_video_frame(frame, invalid)
        # `log.write_video_frame(frame, 1.0)`을 호출하여 해당 단계의 작업을 수행합니다.
        log.write_video_frame(frame, 1.0)
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `log.write_video_frame(frame, 0.9)`을 호출하여 해당 단계의 작업을 수행합니다.
            log.write_video_frame(frame, 0.9)
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `self.manager.finish_session([], end_elapsed_sec=0.9)`을 호출하여 해당 단계의 작업을 수행합니다.
            self.manager.finish_session([], end_elapsed_sec=0.9)

    # 회귀 테스트: `test_calibration_uses_cali_name_and_zero_indices_are_allowed` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_calibration_uses_cali_name_and_zero_indices_are_allowed(self):
        # `info` 값에 `self.manager.begin_session('biceps_curl', 'CALIBRATION', session=0, round_number=0, set_number=0, started_at=self.start)` 호출 결과를 저장합니다. 대상: `info`.
        info = self.manager.begin_session('biceps_curl', 'CALIBRATION', session=0,
                                         # 구성 항목: `round_number` 값(`round_number`)에 0; 설정 number(`set_number`)에 0; `started_at` 값(`started_at`)에 시작(`self.start`)을 지정합니다.
                                         round_number=0, set_number=0, started_at=self.start)
        # 검증문 `self.assertEqual(Path(info['frame_csv']).name, 'session0_cali.csv')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(Path(info['frame_csv']).name, 'session0_cali.csv')
        # `self.manager.finish_session([])`을 호출하여 해당 단계의 작업을 수행합니다.
        self.manager.finish_session([])
        # `info` 값에 `self.manager.begin_session('biceps_curl', 'TEST', started_at=self.start)` 호출 결과를 저장합니다. 대상: `info`.
        info = self.manager.begin_session('biceps_curl', 'TEST', started_at=self.start)
        # 검증문 `self.assertEqual(Path(info['frame_csv']).name, 'session1_round1_set1.csv')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(Path(info['frame_csv']).name, 'session1_round1_set1.csv')

    # 회귀 테스트: `test_date_uses_korea_timezone` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_date_uses_korea_timezone(self):
        # 시작 utc에 `datetime(2025, 12, 31, 16, 0, tzinfo=timezone.utc)` 호출 결과를 저장합니다. 대상: `start_utc`.
        start_utc = datetime(2025, 12, 31, 16, 0, tzinfo=timezone.utc)
        # `info` 값에 `self.manager.begin_session('biceps_curl', 'MAIN', started_at=start_utc)` 호출 결과를 저장합니다. 대상: `info`.
        info = self.manager.begin_session('biceps_curl', 'MAIN', started_at=start_utc)
        # 검증문 `self.assertEqual(Path(info['directory']).parent.name, '260101')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(Path(info['directory']).parent.name, '260101')

    # 회귀 테스트: `test_legacy_profile_is_read_without_moving_or_overwriting_it` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_legacy_profile_is_read_without_moving_or_overwriting_it(self):
        # 기존 호환에 계산식 `self.root / 'legacy.json'`의 결과를 저장합니다. 대상: `legacy`.
        legacy = self.root / 'legacy.json'
        # `legacy.write_text(json.dumps({'custom_thresholds': {'biceps_curl': THRESHOLDS}}), encoding='utf-8')`을 호출하여 해당 단계의 작업을 수행합니다.
        legacy.write_text(json.dumps({'custom_thresholds': {'biceps_curl': THRESHOLDS}}), encoding='utf-8')
        # 이전에 `legacy.read_bytes()` 호출 결과를 저장합니다. 대상: `before`.
        before = legacy.read_bytes()
        # 기존 호환 json 경로에 기존 호환(`legacy`)을 저장합니다. 대상: `self.manager.legacy_json_path`.
        self.manager.legacy_json_path = legacy
        # 검증문 `self.assertEqual(self.manager.get_custom_threshold('biceps_curl'), THRESHOLDS)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.manager.get_custom_threshold('biceps_curl'), THRESHOLDS)
        # `changed` 값에 반복·조건식 `{side: {'start_val': 0.1, 'target_val': 0.5} for side in ('left', 'right')}`으로 구성한 값을 저장합니다. 대상: `changed`.
        changed = {side: {'start_val': 0.1, 'target_val': 0.5} for side in ('left', 'right')}
        # `self.manager.save_custom_threshold('biceps_curl', changed)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.manager.save_custom_threshold('biceps_curl', changed)
        # 검증문 `self.assertEqual(self.manager.get_custom_threshold('biceps_curl'), changed)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(self.manager.get_custom_threshold('biceps_curl'), changed)
        # 검증문 `self.assertEqual(legacy.read_bytes(), before)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(legacy.read_bytes(), before)
        # 검증문 `self.assertTrue(self.manager.json_path.is_relative_to(self.root))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertTrue(self.manager.json_path.is_relative_to(self.root))

    # 회귀 테스트: `test_default_root_is_project_data_and_environment_can_override_it` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_default_root_is_project_data_and_environment_can_override_it(self):
        # `patch.dict(os.environ, {'ASAN_LOG_ROOT': str(self.root / 'custom')})`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch.dict(os.environ, {'ASAN_LOG_ROOT': str(self.root / 'custom')}):
            # 검증문 `self.assertEqual(default_log_root(), self.root / 'custom')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(default_log_root(), self.root / 'custom')
        # `patch.dict(os.environ, {'LOCALAPPDATA': str(self.root)}, clear=True)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with patch.dict(os.environ, {'LOCALAPPDATA': str(self.root)}, clear=True):
            # 검증문 `self.assertEqual(default_log_root(), Path(__file__).resolve().parent.parent / 'data')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(default_log_root(), Path(__file__).resolve().parent.parent / 'data')

    # 회귀 테스트: `test_invalid_ids_indices_and_metadata_are_rejected` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_invalid_ids_indices_and_metadata_are_rejected(self):
        # `('../escape', 'a/b', 'a\\b', '.', 'C:', '')`에서 항목을 차례로 꺼내 `invalid`에 넣고 아래 처리를 반복합니다.
        for invalid in ('../escape', 'a/b', 'a\\b', '.', 'C:', ''):
            # `self.subTest(invalid=invalid), self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                # `DataManager(invalid, log_root=self.root)`을 호출하여 해당 단계의 작업을 수행합니다.
                DataManager(invalid, log_root=self.root)
        # `(-1, True, 1.5)`에서 항목을 차례로 꺼내 `invalid`에 넣고 아래 처리를 반복합니다.
        for invalid in (-1, True, 1.5):
            # `self.subTest(invalid=invalid), self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                # `self.begin(round_number=invalid)`을 호출하여 해당 단계의 작업을 수행합니다.
                self.begin(round_number=invalid)
        # `self.manager.patient_metadata['height_cm']` 값에 `float('nan')` 호출 결과를 저장합니다. 대상: `self.manager.patient_metadata['height_cm']`.
        self.manager.patient_metadata['height_cm'] = float('nan')
        # `self.assertRaises(ValueError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(ValueError):
            # `self.begin()`을 호출하여 해당 단계의 작업을 수행합니다.
            self.begin()

    # 회귀 테스트: `test_storage_failure_is_not_reported_as_success` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_storage_failure_is_not_reported_as_success(self):
        # 경로에 계산식 `self.root / 'not_a_directory'`의 결과를 저장합니다. 대상: `path`.
        path = self.root / 'not_a_directory'
        # `path.write_text('occupied', encoding='utf-8')`을 호출하여 해당 단계의 작업을 수행합니다.
        path.write_text('occupied', encoding='utf-8')
        # `self.assertRaises(OSError)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with self.assertRaises(OSError):
            # `DataManager('0002', log_root=path)`을 호출하여 해당 단계의 작업을 수행합니다.
            DataManager('0002', log_root=path)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
