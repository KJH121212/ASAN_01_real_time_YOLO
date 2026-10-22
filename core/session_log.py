# 한국어 코드 해설: 동일한 세션 이름으로 CSV·NPZ·MP4·요약 JSON을 저장하고 촬영 시간에 맞춰 녹화 프레임을 배치합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""PatientData 공통 경로에서 카메라 CSV·NPZ·MP4와 독립적인 관리 기록을 저장합니다."""

# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# `datetime`에서 `datetime`, `timedelta`, `timezone` 기능을 가져옵니다.
from datetime import datetime, timedelta, timezone
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math
# 이 파일에서 사용할 `os` 모듈을 불러옵니다.
import os
# exe 배포 시 임시 내부 폴더가 아닌 실행 파일 옆에 데이터를 저장하기 위해 사용합니다.
import sys
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `re` 모듈을 불러옵니다.
import re
# 이 파일에서 사용할 `uuid` 모듈을 불러옵니다.
import uuid

# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np
# 공유 플레이·운동 ID와 카메라 파일 ID를 구분합니다.
from core.patient_data import DEFAULT_DATA_VERSION, shared_identifier


# `KST` 값에 `timezone(timedelta(hours=9))` 호출 결과를 저장합니다. 대상: `KST`.
KST = timezone(timedelta(hours=9))
# 프레임 fields에 순서·구성을 지정한 `['timestamp_ms', 'frame_idx', 'video_frame_idx', 'motion_name', 'metric_type', 'left_value', 'right_value', 'left_rep_count', 'right_rep_count', 'left_total_score', 'rig…`을 저장합니다. 대상: `FRAME_FIELDS`.
FRAME_FIELDS = [
    # 같은 플레이의 IMU와 카메라 데이터를 연결하기 위한 식별 열입니다.
    'time', 'play_session_id', 'exercise_record_id', 'sensor_file_id', 'source', 'collection_phase', 'metric_unit',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'timestamp_ms', 'frame_idx', 'video_frame_idx', 'motion_name', 'metric_type',` 내용으로 이어서 구성합니다.
    'timestamp_ms', 'frame_idx', 'video_frame_idx', 'motion_name', 'metric_type',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'left_value', 'right_value', 'left_rep_count', 'right_rep_count',` 내용으로 이어서 구성합니다.
    'left_value', 'right_value', 'left_rep_count', 'right_rep_count',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'left_total_score', 'right_total_score', 'left_score_delta', 'right_score_delta',` 내용으로 이어서 구성합니다.
    'left_total_score', 'right_total_score', 'left_score_delta', 'right_score_delta',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'left_state', 'right_state', 'left_quality', 'right_quality',` 내용으로 이어서 구성합니다.
    'left_state', 'right_state', 'left_quality', 'right_quality',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'left_calibration_status', 'right_calibration_status',` 내용으로 이어서 구성합니다.
    'left_calibration_status', 'right_calibration_status',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'left_calibration_baseline', 'right_calibration_baseline',` 내용으로 이어서 구성합니다.
    'left_calibration_baseline', 'right_calibration_baseline',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'left_calibration_return_tolerance', 'right_calibration_return_tolerance',` 내용으로 이어서 구성합니다.
    'left_calibration_return_tolerance', 'right_calibration_return_tolerance',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'left_count_updated', 'right_count_updated', 'is_occluded', 'calib_step',` 내용으로 이어서 구성합니다.
    'left_count_updated', 'right_count_updated', 'is_occluded', 'calib_step',
    # 앞줄에서 시작한 식·인수·자료 구조를 `'threshold_version', 'left_start_val', 'left_target_val', 'right_start_val', 'right_target_val',` 내용으로 이어서 구성합니다.
    'threshold_version', 'left_start_val', 'left_target_val', 'right_start_val', 'right_target_val',
# 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
]
# 반복 fields에 순서·구성을 지정한 `['session_id', 'side', 'rep_num', 'duration_sec', 'min_val', 'max_val', 'achieved_rom', 'quality', 'score_delta', 'total_score', 'start_val', 'target_val', 'threshold_ve…`을 저장합니다. 대상: `REP_FIELDS`.
REP_FIELDS = ['session_id', 'side', 'rep_num', 'duration_sec', 'min_val', 'max_val',
              'play_session_id', 'exercise_record_id', 'sensor_file_id',
              # 앞줄에서 시작한 식·인수·자료 구조를 `'achieved_rom', 'quality', 'score_delta', 'total_score', 'start_val', 'target_val', 'threshold_version']` 내용으로 이어서 구성합니다.
              'achieved_rom', 'quality', 'score_delta', 'total_score', 'start_val', 'target_val', 'threshold_version']
# `GAME_FIELDS` 값에 순서·구성을 지정한 `['session_id', 'started_at', 'patient_id', 'motion_name', 'mode', 'session', 'round', 'set', 'status', 'saved_frames', 'left_reps', 'right_reps', 'left_total_score', 'ri…`을 저장합니다. 대상: `GAME_FIELDS`.
GAME_FIELDS = ['session_id', 'started_at', 'patient_id', 'motion_name', 'mode', 'session', 'round', 'set',
               'play_session_id', 'exercise_record_id', 'sensor_file_id', 'data_version', 'registration', 'slot',
               'ended_at', 'collection_phase', 'source',
               'frame_csv_relative', 'skeleton_npz_relative', 'video_mp4_relative',
               # 앞줄에서 시작한 식·인수·자료 구조를 `'status', 'saved_frames', 'left_reps', 'right_reps', 'left_total_score', 'right_total_score',` 내용으로 이어서 구성합니다.
               'status', 'saved_frames', 'left_reps', 'right_reps', 'left_total_score', 'right_total_score',
               # 앞줄에서 시작한 식·인수·자료 구조를 `'target_score', 'duration_sec',` 내용으로 이어서 구성합니다.
               'target_score', 'duration_sec',
               # 앞줄에서 시작한 식·인수·자료 구조를 `'calories_kcal', 'max_speed', 'frame_csv', 'skeleton_npz', 'video_mp4']` 내용으로 이어서 구성합니다.
               'calories_kcal', 'max_speed', 'frame_csv', 'skeleton_npz', 'video_mp4']


# ASAN_LOG_ROOT 환경 변수가 있으면 사용하고, 없으면 프로젝트 data 폴더를 반환합니다.
def default_log_root():
    # `configured` 값에 `os.environ.get('ASAN_LOG_ROOT')` 호출 결과를 저장합니다. 대상: `configured`.
    configured = os.environ.get('ASAN_LOG_ROOT')
    # 조건 `configured`을 만족하는 경우에만 아래 분기를 실행합니다.
    if configured:
        # 현재 함수 처리를 끝내고 `Path(configured).expanduser().resolve()` 호출 결과를 호출한 쪽으로 반환합니다.
        return Path(configured).expanduser().resolve()
    # PyInstaller 실행 파일에서는 기본 data 경로를 exe가 있는 폴더 기준으로 고정합니다.
    if getattr(sys, "frozen", False):
        return (Path(sys.executable).resolve().parent / 'PatientData').resolve()
    # Python 실행의 기본 로그 위치는 프로젝트의 PatientData입니다.
    return (Path(__file__).resolve().parent.parent / 'PatientData').resolve()


# 경로 component 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def path_component(value, name):
    # 측정값에 `str(value)` 호출 결과를 저장합니다. 대상: `value`.
    value = str(value)
    # 조건 `not re.fullmatch('[A-Za-z0-9_-]+', value) or Path(value).is_reserved()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if not re.fullmatch(r'[A-Za-z0-9_-]+', value) or Path(value).is_reserved():
        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{name} must contain only letters, numbers, underscores or hyphens')` 예외를 발생시킵니다.
        raise ValueError(f'{name} must contain only letters, numbers, underscores or hyphens')
    # 현재 함수 처리를 끝내고 측정값(`value`)을 호출한 쪽으로 반환합니다.
    return value


# `sequence_number` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def sequence_number(value, name):
    # 조건 `isinstance(value, bool) or not isinstance(value, int) or value < 0`을 만족하는 경우에만 아래 분기를 실행합니다.
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{name} must be a nonnegative integer')` 예외를 발생시킵니다.
        raise ValueError(f'{name} must be a nonnegative integer')
    # 현재 함수 처리를 끝내고 측정값(`value`)을 호출한 쪽으로 반환합니다.
    return value


# 임시 JSON 파일을 먼저 완성한 뒤 원래 경로로 교체하여 중간 상태 노출을 줄입니다.
def write_json_atomic(path, data):
    # 원본을 안전하게 교체하기 위한 임시 파일 경로에 `path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')` 호출 결과를 저장합니다. 대상: `temporary`.
    temporary = path.with_name(path.name + f'.{uuid.uuid4().hex}.tmp')
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # `temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')`을 호출하여 해당 단계의 작업을 수행합니다.
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        # `temporary.replace(path)`을 호출하여 임시 파일을 최종 경로로 교체합니다.
        temporary.replace(path)
    # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
    finally:
        # `temporary.unlink(missing_ok=True)`을 호출하여 지정한 파일을 제거합니다.
        temporary.unlink(missing_ok=True)


# `SessionLog` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class SessionLog:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, patient_dir, exercise_name, mode, session=None, round_number=1,
                 # 이 줄의 입력 인수: `set_number` = 설정 number; `record_video` = 영상 녹화 여부; `video_fps` = 저장 영상의 초당 프레임 수; `started_at` = `started_at` 값; `target_score` = 게임 목표 점수. 기본값·형식은 아래 선언을 따릅니다.
                 set_number=1, record_video=False, video_fps=30.0, started_at=None, target_score=None,
                 *, play_session_id=None, exercise_record_id=None, slot=1, patient_id=None,
                 data_version=DEFAULT_DATA_VERSION, registration='01'):
        # 환자별 로그 폴더에 `Path(patient_dir)` 호출 결과를 저장합니다. 대상: `self.patient_dir`.
        self.patient_dir = Path(patient_dir)
        # 운동 설정의 식별 이름에 `path_component(exercise_name, 'exercise_name')` 호출 결과를 저장합니다. 대상: `self.exercise_name`.
        self.exercise_name = path_component(exercise_name, 'exercise_name')
        # 조건 `mode not in {'CALIBRATION', 'MAIN', 'TEST'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        if mode not in {'CALIBRATION', 'MAIN', 'TEST'}:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('mode must be CALIBRATION, MAIN or TEST')` 예외를 발생시킵니다.
            raise ValueError('mode must be CALIBRATION, MAIN or TEST')
        # 조건 `not isinstance(record_video, bool)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not isinstance(record_video, bool):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('record_video must be a boolean')` 예외를 발생시킵니다.
            raise ValueError('record_video must be a boolean')
        # 조건 `not np.isfinite(video_fps) or video_fps <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not np.isfinite(video_fps) or video_fps <= 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('video_fps must be positive')` 예외를 발생시킵니다.
            raise ValueError('video_fps must be positive')
        # `self.started_at` 값에 계산식 `started_at or datetime.now(KST)`의 결과를 저장합니다. 대상: `self.started_at`.
        self.started_at = started_at or datetime.now(KST)
        # 조건 `self.started_at.tzinfo is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.started_at.tzinfo is None:
            # `self.started_at` 값에 `self.started_at.replace(tzinfo=KST)` 호출 결과를 저장합니다. 대상: `self.started_at`.
            self.started_at = self.started_at.replace(tzinfo=KST)
        # `self.started_at` 값에 `self.started_at.astimezone(KST)` 호출 결과를 저장합니다. 대상: `self.started_at`.
        self.started_at = self.started_at.astimezone(KST)
        # 실행 모드에 실행 모드(`mode`)을 저장합니다. 대상: `self.mode`.
        self.mode = mode
        # `self.round` 값에 `sequence_number(round_number, 'round')` 호출 결과를 저장합니다. 대상: `self.round`.
        self.round = sequence_number(round_number, 'round')
        # 설정에 `sequence_number(set_number, 'set')` 호출 결과를 저장합니다. 대상: `self.set`.
        self.set = sequence_number(set_number, 'set')
        # 서버의 세트 단위 측정 ID와 Unity 전체 플레이 ID는 서로 다른 식별자입니다.
        self.play_session_id = shared_identifier(play_session_id, 'play_session_id')
        # 여러 세트를 같은 운동 기록으로 연결하려면 Unity가 같은 운동 기록 ID를 전달합니다.
        self.exercise_record_id = shared_identifier(exercise_record_id, 'exercise_record_id')
        # 같은 구간의 CSV·NPZ·MP4를 하나의 파일 ID로 묶습니다.
        self.sensor_file_id = uuid.uuid4().hex
        self.slot = sequence_number(slot, 'slot')
        self.collection_phase = 'Calibration' if mode == 'CALIBRATION' else 'Play'
        self.patient_id = patient_id or self.patient_dir.name
        # 센서 날짜는 수집 시작일(KST)이며 종료 시 자정이 지나도 원시 파일을 옮기지 않습니다.
        self.date_dir = self.patient_dir / 'Sensors' / self.started_at.strftime('%Y-%m-%d')
        # Unity 플레이 ID 아래 Camera 영역에서 IMU Left·Right 파일과 구분합니다.
        self.directory = self.date_dir / self.play_session_id / 'Camera'
        # `self.directory.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
        self.directory.mkdir(parents=True, exist_ok=True)
        # 조건 `session is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if session is None:
            # 숫자 Session은 기존 클라이언트용 부가 정보입니다. 파일 구분은 고유 ID를 사용합니다.
            session = 1
        # 측정 세션에 `sequence_number(session, 'session')` 호출 결과를 저장합니다. 대상: `self.session`.
        self.session = sequence_number(session, 'session')
        # `take` 값에 1을 저장합니다. 대상: `take`.
        take = 1
        # 좌우 IMU 파일과 충돌하지 않도록 수집원을 Camera로 명시합니다.
        stem = f'R{self.round:02d}_S{self.slot:02d}_Set{self.set:02d}_{self.collection_phase}_Camera_{self.sensor_file_id}'
        # 조건 `True`이 유지되는 동안 아래 처리를 반복합니다.
        while True:
            # `self.basename` 값에 `take == 1`이면 `stem`, 아니면 `f'{stem}_take{take}'`을 저장합니다. 대상: `self.basename`.
            self.basename = stem if take == 1 else f'{stem}_take{take}'
            # `self.base` 값에 계산식 `self.directory / self.basename`의 결과를 저장합니다. 대상: `self.base`.
            self.base = self.directory / self.basename
            # 조건 `any((self.base.with_suffix(suffix).exists() for suffix in ('.csv', '.npz', '.mp4', '.json')))`을 만족하는 경우에만 아래 분기를 실행합니다.
            if any(self.base.with_suffix(suffix).exists() for suffix in ('.csv', '.npz', '.mp4', '.json')):
                # `take` 값에 `1`을 더해 누적합니다. 대상: `take`.
                take += 1
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `self._csv` 값에 `self.base.with_suffix('.csv').open('x', newline='', encoding='utf-8-sig')` 호출 결과를 저장합니다. 대상: `self._csv`.
                self._csv = self.base.with_suffix('.csv').open('x', newline='', encoding='utf-8-sig')
                # 현재 반복문을 즉시 끝내고 반복문 다음 처리로 이동합니다.
                break
            # `FileExistsError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except FileExistsError:
                # `take` 값에 `1`을 더해 누적합니다. 대상: `take`.
                take += 1
        # 현재 측정 세션 식별자에 `uuid.uuid4().hex` 값(`uuid.uuid4().hex`)을 저장합니다. 대상: `self.session_id`.
        self.session_id = self.sensor_file_id
        # `self._writer` 값에 `csv.DictWriter(self._csv, fieldnames=FRAME_FIELDS)` 호출 결과를 저장합니다. 대상: `self._writer`.
        self._writer = csv.DictWriter(self._csv, fieldnames=FRAME_FIELDS)
        # `self._writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
        self._writer.writeheader()
        # `self._csv.flush()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._csv.flush()
        # `(self.timestamps, self.values, self.keypoints)` 값에 순서·구성을 지정한 `([], [], [])`을 저장합니다. 대상: `(self.timestamps, self.values, self.keypoints)`.
        self.timestamps, self.values, self.keypoints = [], [], []
        # 영상 indices)에 순서·구성을 지정한 `([], [])`을 저장합니다. 대상: `(self.frame_indices, self.video_indices)`.
        self.frame_indices, self.video_indices = [], []
        # 이미지 aspect ratios에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `self.image_aspect_ratios`.
        self.image_aspect_ratios = []
        # 점수 deltas)에 순서·구성을 지정한 `([], [])`을 저장합니다. 대상: `(self.total_scores, self.score_deltas)`.
        self.total_scores, self.score_deltas = [], []
        # 영상 녹화 여부에 영상 녹화 여부(`record_video`)을 저장합니다. 대상: `self.record_video`.
        self.record_video = record_video
        # 저장 영상의 초당 프레임 수에 `float(video_fps)` 호출 결과를 저장합니다. 대상: `self.video_fps`.
        self.video_fps = float(video_fps)
        # 영상 writer에 None을 저장합니다. 대상: `self._video_writer`.
        self._video_writer = None
        # 영상 크기에 None을 저장합니다. 대상: `self._video_size`.
        self._video_size = None
        # 영상 프레임 묶음에 0을 저장합니다. 대상: `self._video_frames`.
        self._video_frames = 0
        # 영상 origin에 None을 저장합니다. 대상: `self._video_origin`.
        self._video_origin = None
        # last 영상 elapsed에 None을 저장합니다. 대상: `self._last_video_elapsed`.
        self._last_video_elapsed = None
        # previous 영상 프레임에 None을 저장합니다. 대상: `self._previous_video_frame`.
        self._previous_video_frame = None
        # 요약에 None을 저장합니다. 대상: `self._summary`.
        self._summary = None
        # `self.info` 값에 키 'session_id', 'started_at', 'patient_id', 'motion_name', 'mode', 'session', 'round', 'set', 'log_root', 'dire…로 구성한 사전을 저장합니다. 대상: `self.info`.
        self.info = {
            # Unity의 내부 JSON 스키마와 구분되는 카메라 전용 명세 버전입니다.
            'storage_schema': 'PatientData.Camera/1',
            # 구성 항목: 현재 측정 세션 식별자(`'session_id'`)에 현재 측정 세션 식별자(`self.session_id`); `started_at` 값(`'started_at'`)에 `self.started_at.isoformat()` 호출 결과를 지정합니다.
            'session_id': self.session_id, 'started_at': self.started_at.isoformat(),
            # 구성 항목: 환자 식별 번호(`'patient_id'`)에 `self.patient_dir.name` 값(`self.patient_dir.name`); 운동 name(`'motion_name'`)에 운동 설정의 식별 이름(`self.exercise_name`); 실행 모드(`'mode'`)에 실행 모드(`mode`)을 지정합니다.
            'patient_id': self.patient_id, 'motion_name': self.exercise_name, 'mode': mode,
            # 기존 session_id는 클라이언트 이벤트 중복 방지용 측정 ID로 유지합니다.
            'play_session_id': self.play_session_id, 'exercise_record_id': self.exercise_record_id,
            'sensor_file_id': self.sensor_file_id, 'source': 'Camera', 'collection_phase': self.collection_phase,
            'data_version': data_version, 'registration': registration, 'slot': self.slot,
            # 구성 항목: 측정 세션(`'session'`)에 측정 세션(`self.session`); `round` 값(`'round'`)에 `self.round` 값(`self.round`); 설정(`'set'`)에 설정(`self.set`)을 지정합니다.
            'session': self.session, 'round': self.round, 'set': self.set,
            # 구성 항목: 세션 로그를 저장하는 최상위 폴더(`'log_root'`)에 `str(self.patient_dir.parent)` 호출 결과; 폴더(`'directory'`)에 `str(self.directory)` 호출 결과를 지정합니다.
            'log_root': str(self.patient_dir.parents[2]), 'directory': str(self.directory),
            # 구성 항목: 프레임 csv(`'frame_csv'`)에 `str(self.base.with_suffix('.csv'))` 호출 결과를 지정합니다.
            'frame_csv': str(self.base.with_suffix('.csv')),
            # 구성 항목: `skeleton_npz` 값(`'skeleton_npz'`)에 `str(self.base.with_suffix('.npz'))` 호출 결과를 지정합니다.
            'skeleton_npz': str(self.base.with_suffix('.npz')),
            # 구성 항목: 영상 mp4(`'video_mp4'`)에 `record_video`이면 `str(self.base.with_suffix('.mp4'))`, 아니면 `None`을 지정합니다.
            'video_mp4': str(self.base.with_suffix('.mp4')) if record_video else None,
            # 구성 항목: 반복 횟수 csv(`'reps_csv'`)에 `str(self.directory / f'{self.basename}_reps.csv')` 호출 결과를 지정합니다.
            'reps_csv': str(self.directory / f'{self.basename}_reps.csv'),
            # 구성 항목: `manifest_json` 값(`'manifest_json'`)에 `str(self.base.with_suffix('.json'))` 호출 결과를 지정합니다.
            'manifest_json': str(self.base.with_suffix('.json')),
            # 구성 항목: `gamedata_csv` 값(`'gamedata_csv'`)에 `str(self.date_dir / 'gamedata.csv')` 호출 결과를 지정합니다.
            # Unity의 일별 xlsx·Records JSON과 별개로 카메라 요약을 기록합니다.
            'gamedata_csv': str(self.patient_dir / 'Management' / 'Camera' / 'Daily' / f'{self.started_at:%Y-%m-%d}.csv'),
            'camera_record_json': str(self.patient_dir / 'Management' / 'Camera' / 'Records' / f'{self.sensor_file_id}.json'),
            # 구성 항목: 영상 녹화 여부(`'record_video'`)에 영상 녹화 여부(`record_video`); 저장 영상의 초당 프레임 수(`'video_fps'`)에 저장 영상의 초당 프레임 수(`self.video_fps`)을 지정합니다.
            'record_video': record_video, 'video_fps': self.video_fps,
            # 구성 항목: 게임 목표 점수(`'target_score'`)에 게임 목표 점수(`target_score`)을 지정합니다.
            'target_score': target_score,
            # 구성 항목: 점수 판정 mode(`'scoring_mode'`)에 `mode == 'CALIBRATION'`이면 `'REPETITIONS'`, 아니면 `'POINTS'`을 지정합니다.
            'scoring_mode': 'REPETITIONS' if mode == 'CALIBRATION' else 'POINTS',
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        }
        # 등록 차수 폴더 전체를 복사해도 파일을 찾을 수 있도록 상대 경로도 함께 보관합니다.
        for field in ('frame_csv', 'skeleton_npz', 'video_mp4', 'reps_csv', 'manifest_json', 'camera_record_json'):
            # 녹화를 사용하지 않는 내부 테스트에서는 video_mp4가 None일 수 있습니다.
            path = self.info.get(field)
            self.info[f'{field}_relative'] = Path(path).relative_to(self.patient_dir).as_posix() if path else None
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `write_json_atomic(self.base.with_suffix('.json'), {**self.info, 'status': 'RECORDING'})`을 호출하여 해당 단계의 작업을 수행합니다.
            write_json_atomic(self.base.with_suffix('.json'), {**self.info, 'status': 'RECORDING'})
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception:
            # `self._csv.close()`을 호출하여 해당 단계의 작업을 수행합니다.
            self._csv.close()
            # 이 조건에서는 처리를 계속할 수 없으므로 `None` 예외를 발생시킵니다.
            raise

    # 기록 영상 프레임 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def write_video_frame(self, frame, elapsed_sec):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """Resample capture timestamps onto the MP4 clock, independent of inference FPS."""
        # 조건 `not self.record_video`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.record_video:
            # 현재 함수 처리를 끝내고 계산식 `-1`의 결과를 호출한 쪽으로 반환합니다.
            return -1
        # 조건 `self._summary is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._summary is not None:
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Session log is already closed')` 예외를 발생시킵니다.
            raise RuntimeError('Session log is already closed')
        # `elapsed_sec` 값에 `float(elapsed_sec)` 호출 결과를 저장합니다. 대상: `elapsed_sec`.
        elapsed_sec = float(elapsed_sec)
        # 조건 `not math.isfinite(elapsed_sec) or elapsed_sec < 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(elapsed_sec) or elapsed_sec < 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Video timestamp must be finite and nonnegative')` 예외를 발생시킵니다.
            raise ValueError('Video timestamp must be finite and nonnegative')
        # 조건 `self._last_video_elapsed is not None and elapsed_sec < self._last_video_elapsed`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._last_video_elapsed is not None and elapsed_sec < self._last_video_elapsed:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Video timestamps must be nondecreasing')` 예외를 발생시킵니다.
            raise ValueError('Video timestamps must be nondecreasing')
        # 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
        import cv2
        # 크기에 순서·구성을 지정한 `(frame.shape[1], frame.shape[0])`을 저장합니다. 대상: `size`.
        size = (frame.shape[1], frame.shape[0])
        # 조건 `self._video_writer is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._video_writer is None:
            # 영상 크기에 크기(`size`)을 저장합니다. 대상: `self._video_size`.
            self._video_size = size
            # 영상 writer에 `cv2.VideoWriter(str(self.base.with_suffix('.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, size)` 호출 결과를 저장합니다. 대상: `self._video_writer`.
            self._video_writer = cv2.VideoWriter(
                # 앞줄에서 시작한 식·인수·자료 구조를 `str(self.base.with_suffix('.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, size)` 내용으로 이어서 구성합니다.
                str(self.base.with_suffix('.mp4')), cv2.VideoWriter_fourcc(*'mp4v'), self.video_fps, size)
            # 조건 `not self._video_writer.isOpened()`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not self._video_writer.isOpened():
                # `self._video_writer.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
                self._video_writer.release()
                # 영상 writer에 None을 저장합니다. 대상: `self._video_writer`.
                self._video_writer = None
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError(f"Could not open MP4 recorder: {self.base.with_suffix('.mp4')}")` 예외를 발생시킵니다.
                raise RuntimeError(f'Could not open MP4 recorder: {self.base.with_suffix(".mp4")}')
        # 조건 `size != self._video_size`을 만족하는 경우에만 아래 분기를 실행합니다.
        if size != self._video_size:
            # OpenCV BGR 영상 프레임에 `cv2.resize(frame, self._video_size)` 호출 결과를 저장합니다. 대상: `frame`.
            frame = cv2.resize(frame, self._video_size)
        # 조건 `self._video_origin is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._video_origin is None:
            # 영상 origin에 `elapsed_sec` 값(`elapsed_sec`)을 저장합니다. 대상: `self._video_origin`.
            self._video_origin = elapsed_sec
        # 목표 index에 `math.floor((elapsed_sec - self._video_origin) * self.video_fps + 1e-06)` 호출 결과를 저장합니다. 대상: `target_index`.
        target_index = math.floor((elapsed_sec - self._video_origin) * self.video_fps + 1e-6)
        # 입력 간격이 느리거나 불규칙하면 빈 출력 프레임 구간을 채웁니다.
        # 같은 출력 시간 구간에 여러 입력이 들어와도 영상을 불필요하게 늘리지 않습니다.
        # 조건 `self._video_frames < target_index`이 유지되는 동안 아래 처리를 반복합니다.
        while self._video_frames < target_index:
            # `self._video_writer.write(self._previous_video_frame)`을 호출하여 파일 또는 영상에 데이터를 기록합니다.
            self._video_writer.write(self._previous_video_frame)
            # 영상 프레임 묶음에 `1`을 더해 누적합니다. 대상: `self._video_frames`.
            self._video_frames += 1
        # 조건 `self._video_frames == target_index`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._video_frames == target_index:
            # `self._video_writer.write(frame)`을 호출하여 파일 또는 영상에 데이터를 기록합니다.
            self._video_writer.write(frame)
            # 영상 프레임 묶음에 `1`을 더해 누적합니다. 대상: `self._video_frames`.
            self._video_frames += 1
        # previous 영상 프레임에 `frame.copy()` 호출 결과를 저장합니다. 대상: `self._previous_video_frame`.
        self._previous_video_frame = frame.copy()
        # last 영상 elapsed에 `elapsed_sec` 값(`elapsed_sec`)을 저장합니다. 대상: `self._last_video_elapsed`.
        self._last_video_elapsed = elapsed_sec
        # 현재 함수 처리를 끝내고 목표 index(`target_index`)을 호출한 쪽으로 반환합니다.
        return target_index

    # append 프레임 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def append_frame(self, elapsed_sec, frame_idx, raw_keypoints, left, right, metric_type,
                     # 이 줄의 입력 인수: `thresholds` = 좌우 가동 범위 시작값·목표값; `threshold_version` = 현재 가동 범위 설정의 변경 번호; `is_occluded` = 필수 관절이 가려졌거나 유효하지 않은지 여부; `calib_step` = 캘리브레이션 전체 진행 단계; `video_frame_idx` = 저장 영상에서 대응하는 프레임 번호. 기본값·형식은 아래 선언을 따릅니다.
                     thresholds, threshold_version=0, is_occluded=False, calib_step='', video_frame_idx=-1):
        # 조건 `self._summary is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._summary is not None:
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Session log is already closed')` 예외를 발생시킵니다.
            raise RuntimeError('Session log is already closed')
        # `matrix` 값에 `np.full((17, 3), np.nan, dtype=np.float32)` 호출 결과를 저장합니다. 대상: `matrix`.
        matrix = np.full((17, 3), np.nan, dtype=np.float32)
        # `matrix[:, 2]` 값에 0.0을 저장합니다. 대상: `matrix[:, 2]`.
        matrix[:, 2] = 0.0
        # `raw_keypoints or []`에서 항목을 차례로 꺼내 `kp`에 넣고 아래 처리를 반복합니다.
        for kp in raw_keypoints or []:
            # `index` 값에 계산식 `kp['id']`의 결과를 저장합니다. 대상: `index`.
            index = kp['id']
            # 조건 `0 <= index < 17`을 만족하는 경우에만 아래 분기를 실행합니다.
            if 0 <= index < 17:
                # `matrix[index]` 값에 순서·구성을 지정한 `[kp['x'], kp['y'], kp.get('score', 0.0)]`을 저장합니다. 대상: `matrix[index]`.
                matrix[index] = [kp['x'], kp['y'], kp.get('score', 0.0)]
        # 측정값에 반복·조건식 `[np.nan if side.get('val') is None else float(side['val']) for side in (left, right)]`으로 구성한 값을 저장합니다. 대상: `values`.
        values = [np.nan if side.get('val') is None else float(side['val']) for side in (left, right)]
        # `row` 값에 키 'timestamp_ms', 'frame_idx', 'video_frame_idx', 'motion_name', 'metric_type', 'is_occluded', 'calib_step', 't…로 구성한 사전을 저장합니다. 대상: `row`.
        row = {'timestamp_ms': round(elapsed_sec * 1000, 3), 'frame_idx': frame_idx,
               # time은 문서와 동일하게 수집 구간 시작 이후 경과 시간(초)입니다.
               'time': float(elapsed_sec), 'play_session_id': self.play_session_id,
               'exercise_record_id': self.exercise_record_id, 'sensor_file_id': self.sensor_file_id,
               'source': 'Camera', 'collection_phase': self.collection_phase,
               'metric_unit': 'deg' if metric_type == 'ANGLE' else 'ratio' if metric_type == 'RATIO' else 'body_length',
               # 구성 항목: 저장 영상에서 대응하는 프레임 번호(`'video_frame_idx'`)에 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`); 운동 name(`'motion_name'`)에 운동 설정의 식별 이름(`self.exercise_name`)을 지정합니다.
               'video_frame_idx': video_frame_idx, 'motion_name': self.exercise_name,
               # 구성 항목: 측정 방식(각도·좌표·길이·비율)(`'metric_type'`)에 측정 방식(각도·좌표·길이·비율)(`metric_type`); 필수 관절이 가려졌거나 유효하지 않은지 여부(`'is_occluded'`)에 필수 관절이 가려졌거나 유효하지 않은지 여부(`is_occluded`); 캘리브레이션 전체 진행 단계(`'calib_step'`)에 캘리브레이션 전체 진행 단계(`calib_step`)을 지정합니다.
               'metric_type': metric_type, 'is_occluded': is_occluded, 'calib_step': calib_step,
               # 구성 항목: 현재 가동 범위 설정의 변경 번호(`'threshold_version'`)에 현재 가동 범위 설정의 변경 번호(`threshold_version`)을 지정합니다.
               'threshold_version': threshold_version}
        # `zip(('left', 'right'), (left, right), values)`에서 항목을 차례로 꺼내 `(side, data, value)`에 넣고 아래 처리를 반복합니다.
        for side, data, value in zip(('left', 'right'), (left, right), values):
            # 캘리브레이션에 `data.get('calibration', {})` 호출 결과를 저장합니다. 대상: `calibration`.
            calibration = data.get('calibration', {})
            # `row.update({f'{side}_value': '' if np.isnan(value) else value, f'{side}_calibration_status': calibration.get('state', ''), f'{side}_calibration_baseline': calibration.ge…`을 호출하여 해당 단계의 작업을 수행합니다.
            row.update({f'{side}_value': '' if np.isnan(value) else value,
                        # 구성 항목: f'{side} 캘리브레이션 상태(`f'{side}_calibration_status'`)에 `calibration.get('state', '')` 호출 결과를 지정합니다.
                        f'{side}_calibration_status': calibration.get('state', ''),
                        # 구성 항목: f'{side} 캘리브레이션 baseline(`f'{side}_calibration_baseline'`)에 `calibration.get('baseline', '')` 호출 결과를 지정합니다.
                        f'{side}_calibration_baseline': calibration.get('baseline', ''),
                        # 구성 항목: f'{side} 캘리브레이션 return tolerance(`f'{side}_calibration_return_tolerance'`)에 `calibration.get('return_tolerance', '')` 호출 결과를 지정합니다.
                        f'{side}_calibration_return_tolerance': calibration.get('return_tolerance', ''),
                        # 구성 항목: f'{side} 반복 횟수(`f'{side}_rep_count'`)에 `data.get('rep_count', 0)` 호출 결과; f'{side} 단계(`f'{side}_state'`)에 `data.get('state', '')` 호출 결과를 지정합니다.
                        f'{side}_rep_count': data.get('rep_count', 0), f'{side}_state': data.get('state', ''),
                        # 구성 항목: f'{side} total 점수(`f'{side}_total_score'`)에 `data.get('total_score', 0.0)` 호출 결과를 지정합니다.
                        f'{side}_total_score': data.get('total_score', 0.0),
                        # 구성 항목: f'{side} 점수 delta(`f'{side}_score_delta'`)에 `is_occluded`이면 `0.0`, 아니면 `data.get('score_delta', 0.0)`을 지정합니다.
                        f'{side}_score_delta': 0.0 if is_occluded else data.get('score_delta', 0.0),
                        # 구성 항목: f'{side} 품질 등급(`f'{side}_quality'`)에 `data.get('quality', '')` 호출 결과를 지정합니다.
                        f'{side}_quality': data.get('quality', ''),
                        # 구성 항목: f'{side} 횟수 updated(`f'{side}_count_updated'`)에 `data.get('is_count_updated', False)` 호출 결과를 지정합니다.
                        f'{side}_count_updated': data.get('is_count_updated', False),
                        # 구성 항목: f'{side} 시작 val(`f'{side}_start_val'`)에 계산식 `thresholds[side]['start_val']`의 결과를 지정합니다.
                        f'{side}_start_val': thresholds[side]['start_val'],
                        # 구성 항목: f'{side} 목표 val(`f'{side}_target_val'`)에 계산식 `thresholds[side]['target_val']`의 결과를 지정합니다.
                        f'{side}_target_val': thresholds[side]['target_val']})
        # `self._writer.writerow(row)`을 호출하여 CSV에 행 하나를 기록합니다.
        self._writer.writerow(row)
        # `self._csv.flush()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._csv.flush()
        # `self.timestamps.append(float(elapsed_sec))`을 호출하여 항목 하나를 끝에 추가합니다.
        self.timestamps.append(float(elapsed_sec))
        # `self.values.append(values)`을 호출하여 항목 하나를 끝에 추가합니다.
        self.values.append(values)
        # `self.keypoints.append(matrix)`을 호출하여 항목 하나를 끝에 추가합니다.
        self.keypoints.append(matrix)
        # `self.image_aspect_ratios.append(float(raw_keypoints[0].get('image_aspect_ratio', 1.0)) if raw_keypoints else np.nan)`을 호출하여 항목 하나를 끝에 추가합니다.
        self.image_aspect_ratios.append(
            # 앞줄에서 시작한 식·인수·자료 구조를 `float(raw_keypoints[0].get('image_aspect_ratio', 1.0)) if raw_keypoints else np.nan)` 내용으로 이어서 구성합니다.
            float(raw_keypoints[0].get('image_aspect_ratio', 1.0)) if raw_keypoints else np.nan)
        # `self.frame_indices.append(frame_idx)`을 호출하여 항목 하나를 끝에 추가합니다.
        self.frame_indices.append(frame_idx)
        # `self.video_indices.append(video_frame_idx)`을 호출하여 항목 하나를 끝에 추가합니다.
        self.video_indices.append(video_frame_idx)
        # `self.total_scores.append([data.get('total_score', 0.0) for data in (left, right)])`을 호출하여 항목 하나를 끝에 추가합니다.
        self.total_scores.append([data.get('total_score', 0.0) for data in (left, right)])
        # `self.score_deltas.append([0.0 if is_occluded else data.get('score_delta', 0.0) for data in (left, right)])`을 호출하여 항목 하나를 끝에 추가합니다.
        self.score_deltas.append([0.0 if is_occluded else data.get('score_delta', 0.0) for data in (left, right)])

    # 종료 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def finish(self, rep_rows, status='COMPLETED', end_elapsed_sec=None):
        # 조건 `self._summary is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._summary is not None:
            # 현재 함수 처리를 끝내고 `dict(self._summary)` 호출 결과를 호출한 쪽으로 반환합니다.
            return dict(self._summary)
        # 조건 `end_elapsed_sec is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if end_elapsed_sec is not None:
            # `end_elapsed_sec` 값에 `float(end_elapsed_sec)` 호출 결과를 저장합니다. 대상: `end_elapsed_sec`.
            end_elapsed_sec = float(end_elapsed_sec)
            # 조건 `not math.isfinite(end_elapsed_sec) or end_elapsed_sec < 0`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not math.isfinite(end_elapsed_sec) or end_elapsed_sec < 0:
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Session end timestamp must be finite and nonnegative')` 예외를 발생시킵니다.
                raise ValueError('Session end timestamp must be finite and nonnegative')
            # 조건 `self._last_video_elapsed is not None and end_elapsed_sec < self._last_video_elapsed`을 만족하는 경우에만 아래 분기를 실행합니다.
            if self._last_video_elapsed is not None and end_elapsed_sec < self._last_video_elapsed:
                # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Session cannot end before the last video timestamp')` 예외를 발생시킵니다.
                raise ValueError('Session cannot end before the last video timestamp')
        # `self._csv.close()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._csv.close()
        # 조건 `self._video_writer is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self._video_writer is not None:
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # 조건 `end_elapsed_sec is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if end_elapsed_sec is not None:
                    # end 프레임 묶음에 `math.ceil((end_elapsed_sec - self._video_origin) * self.video_fps - 1e-06)` 호출 결과를 저장합니다. 대상: `end_frames`.
                    end_frames = math.ceil((end_elapsed_sec - self._video_origin) * self.video_fps - 1e-6)
                    # 마지막 추론 뒤 중지·연결 종료까지의 대기 시간도 녹화 길이에 반영합니다.
                    # 조건 `self._video_frames < end_frames`이 유지되는 동안 아래 처리를 반복합니다.
                    while self._video_frames < end_frames:
                        # `self._video_writer.write(self._previous_video_frame)`을 호출하여 파일 또는 영상에 데이터를 기록합니다.
                        self._video_writer.write(self._previous_video_frame)
                        # 영상 프레임 묶음에 `1`을 더해 누적합니다. 대상: `self._video_frames`.
                        self._video_frames += 1
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # `self._video_writer.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
                self._video_writer.release()
                # 영상 writer에 None을 저장합니다. 대상: `self._video_writer`.
                self._video_writer = None
        # npz 경로에 `self.base.with_suffix('.npz')` 호출 결과를 저장합니다. 대상: `npz_path`.
        npz_path = self.base.with_suffix('.npz')
        # 원본을 안전하게 교체하기 위한 임시 파일 경로에 `self.base.with_suffix('.npz.tmp')` 호출 결과를 저장합니다. 대상: `temporary`.
        temporary = self.base.with_suffix('.npz.tmp')
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `temporary.open('wb')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with temporary.open('wb') as file:
                # `np.savez_compressed(file, timestamps=np.asarray(self.timestamps, dtype=np.float64), timestamp_ms=np.asarray(self.timestamps, dtype=np.float64) * 1000, frame_idx=np.asarr…`을 호출하여 해당 단계의 작업을 수행합니다.
                np.savez_compressed(file, timestamps=np.asarray(self.timestamps, dtype=np.float64),
                    # 구성 항목: 세션 시작 기준 경과 시각(밀리초)(`timestamp_ms`)에 계산식 `np.asarray(self.timestamps, dtype=np.float64) * 1000`의 결과; `dtype` 값(`dtype`)에 `np.float64` 값(`np.float64`)을 지정합니다.
                    timestamp_ms=np.asarray(self.timestamps, dtype=np.float64) * 1000,
                    # 구성 항목: 분석 프레임 번호(`frame_idx`)에 `np.asarray(self.frame_indices, dtype=np.int64)` 호출 결과; `dtype` 값(`dtype`)에 `np.int64` 값(`np.int64`)을 지정합니다.
                    frame_idx=np.asarray(self.frame_indices, dtype=np.int64),
                    # 구성 항목: 저장 영상에서 대응하는 프레임 번호(`video_frame_idx`)에 `np.asarray(self.video_indices, dtype=np.int64)` 호출 결과; `dtype` 값(`dtype`)에 `np.int64` 값(`np.int64`)을 지정합니다.
                    video_frame_idx=np.asarray(self.video_indices, dtype=np.int64),
                    # 구성 항목: 측정값(`values`)에 `np.asarray(self.values, dtype=np.float32).reshape(-1, 2)` 호출 결과; `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
                    values=np.asarray(self.values, dtype=np.float32).reshape(-1, 2),
                    # 구성 항목: `total_scores` 값(`total_scores`)에 `np.asarray(self.total_scores, dtype=np.float32).reshape(-1, 2)` 호출 결과; `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
                    total_scores=np.asarray(self.total_scores, dtype=np.float32).reshape(-1, 2),
                    # 구성 항목: 점수 deltas(`score_deltas`)에 `np.asarray(self.score_deltas, dtype=np.float32).reshape(-1, 2)` 호출 결과; `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
                    score_deltas=np.asarray(self.score_deltas, dtype=np.float32).reshape(-1, 2),
                    # 구성 항목: COCO 관절 좌표 목록(`keypoints`)에 `np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 17, 3)` 호출 결과; `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
                    keypoints=np.asarray(self.keypoints, dtype=np.float32).reshape(-1, 17, 3),
                    # 구성 항목: 입력 영상의 가로/세로 비율(`image_aspect_ratio`)에 `np.asarray(self.image_aspect_ratios, dtype=np.float32)` 호출 결과; `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
                    image_aspect_ratio=np.asarray(self.image_aspect_ratios, dtype=np.float32),
                    # 구성 항목: 환자 식별 번호의 기존 필드명(`player_id`)에 `self.patient_dir.name` 값(`self.patient_dir.name`); 운동 설정의 식별 이름(`exercise_name`)에 운동 설정의 식별 이름(`self.exercise_name`)을 지정합니다.
                    player_id=self.patient_id, exercise_name=self.exercise_name,
                    play_session_id=self.play_session_id, exercise_record_id=self.exercise_record_id,
                    sensor_file_id=self.sensor_file_id, data_version=self.info['data_version'],
                    registration=self.info['registration'], slot=self.slot,
                    # 구성 항목: 현재 측정 세션 식별자(`session_id`)에 현재 측정 세션 식별자(`self.session_id`); 측정 세션 type(`session_type`)에 실행 모드(`self.mode`)을 지정합니다.
                    session_id=self.session_id, session_type=self.mode)
            # `temporary.replace(npz_path)`을 호출하여 임시 파일을 최종 경로로 교체합니다.
            temporary.replace(npz_path)
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `temporary.unlink(missing_ok=True)`을 호출하여 지정한 파일을 제거합니다.
            temporary.unlink(missing_ok=True)
        # `Path(self.info['reps_csv']).open('w', newline='', encoding='utf-8-sig')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with Path(self.info['reps_csv']).open('w', newline='', encoding='utf-8-sig') as file:
            # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=REP_FIELDS, extrasaction='ignore')` 호출 결과를 저장합니다. 대상: `writer`.
            writer = csv.DictWriter(file, fieldnames=REP_FIELDS, extrasaction='ignore')
            # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
            writer.writeheader()
            # `writer.writerows(({**row, 'session_id': self.session_id} for row in rep_rows))`을 호출하여 CSV에 여러 행을 기록합니다.
            # 반복 파일 하나만 분리해 읽어도 Unity 플레이·운동 기록에 연결할 수 있습니다.
            writer.writerows({**row, 'session_id': self.session_id, 'play_session_id': self.play_session_id,
                              'exercise_record_id': self.exercise_record_id, 'sensor_file_id': self.sensor_file_id}
                             for row in rep_rows)
        # 저장·종료 결과 요약에 키 'status', 'saved_frames', 'saved_reps_count', 'left_reps', 'right_reps', 'left_total_score', 'right_total_sco…로 구성한 사전을 저장합니다. 대상: `summary`.
        summary = {**self.info, 'status': status, 'saved_frames': len(self.timestamps),
                   # 구성 항목: saved 반복 횟수 횟수(`'saved_reps_count'`)에 `len(rep_rows)` 호출 결과; 왼쪽 반복 횟수(`'left_reps'`)에 `sum((row['side'] == 'left' for row in rep_rows))` 호출 결과를 지정합니다.
                   'saved_reps_count': len(rep_rows), 'left_reps': sum(row['side'] == 'left' for row in rep_rows),
                   # 구성 항목: 오른쪽 반복 횟수(`'right_reps'`)에 `sum((row['side'] == 'right' for row in rep_rows))` 호출 결과를 지정합니다.
                   'right_reps': sum(row['side'] == 'right' for row in rep_rows),
                   # 구성 항목: 왼쪽 total 점수(`'left_total_score'`)에 `sum((row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'left'))` 호출 결과를 지정합니다.
                   'left_total_score': sum(row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'left'),
                   # 구성 항목: 오른쪽 total 점수(`'right_total_score'`)에 `sum((row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'right'))` 호출 결과를 지정합니다.
                   'right_total_score': sum(row.get('score_delta', 0.0) for row in rep_rows if row['side'] == 'right'),
                   # 구성 항목: 길이 sec(`'duration_sec'`)에 `end_elapsed_sec is not None`이면 `end_elapsed_sec`, 아니면 `self.timestamps[-1] if self.timestamps else 0.0`을 지정합니다.
                   'duration_sec': (end_elapsed_sec if end_elapsed_sec is not None
                                    # 앞줄에서 시작한 식·인수·자료 구조를 `else self.timestamps[-1] if self.timestamps else 0.0),` 내용으로 이어서 구성합니다.
                                    else self.timestamps[-1] if self.timestamps else 0.0),
                   # 구성 항목: 영상 프레임 묶음(`'video_frames'`)에 영상 프레임 묶음(`self._video_frames`); 영상 시각 origin sec(`'video_time_origin_sec'`)에 영상 origin(`self._video_origin`)을 지정합니다.
                   'video_frames': self._video_frames, 'video_time_origin_sec': self._video_origin,
                   # 구성 항목: 영상 길이 sec(`'video_duration_sec'`)에 계산식 `self._video_frames / self.video_fps`의 결과를 지정합니다.
                   'video_duration_sec': self._video_frames / self.video_fps}
        # 칼로리·속도는 계산 구현이 없으므로 임의로 채우지 않고 빈 값으로 둡니다.
        # `daily` 값에 `Path(self.info['gamedata_csv'])` 호출 결과를 저장합니다. 대상: `daily`.
        # 원시 파일은 시작일에 유지하고 일별 요약만 종료일(KST)에 기록합니다.
        ended_at = self.started_at + timedelta(seconds=summary['duration_sec'])
        summary['ended_at'] = ended_at.isoformat()
        daily = self.patient_dir / 'Management' / 'Camera' / 'Daily' / f'{ended_at:%Y-%m-%d}.csv'
        daily.parent.mkdir(parents=True, exist_ok=True)
        summary['gamedata_csv'] = str(daily)
        self.info['gamedata_csv'] = str(daily)
        # `existing_rows` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `existing_rows`.
        existing_rows = []
        # 조건 `daily.exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if daily.exists():
            # `daily.open(newline='', encoding='utf-8-sig')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with daily.open(newline='', encoding='utf-8-sig') as file:
                # `existing_rows` 값에 반복·조건식 `[row for row in csv.DictReader(file) if row['session_id'] != self.session_id]`으로 구성한 값을 저장합니다. 대상: `existing_rows`.
                existing_rows = [row for row in csv.DictReader(file) if row['session_id'] != self.session_id]
        # `daily_temporary` 값에 `daily.with_suffix('.csv.tmp')` 호출 결과를 저장합니다. 대상: `daily_temporary`.
        daily_temporary = daily.with_suffix('.csv.tmp')
        # `daily_temporary.open('w', newline='', encoding='utf-8-sig')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with daily_temporary.open('w', newline='', encoding='utf-8-sig') as file:
            # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=GAME_FIELDS, extrasaction='ignore')` 호출 결과를 저장합니다. 대상: `writer`.
            writer = csv.DictWriter(file, fieldnames=GAME_FIELDS, extrasaction='ignore')
            # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
            writer.writeheader()
            # `writer.writerows(existing_rows)`을 호출하여 CSV에 여러 행을 기록합니다.
            writer.writerows(existing_rows)
            # `writer.writerow(summary)`을 호출하여 CSV에 행 하나를 기록합니다.
            writer.writerow(summary)
        # `daily_temporary.replace(daily)`을 호출하여 임시 파일을 최종 경로로 교체합니다.
        daily_temporary.replace(daily)
        # `write_json_atomic(self.base.with_suffix('.json'), summary)`을 호출하여 해당 단계의 작업을 수행합니다.
        write_json_atomic(self.base.with_suffix('.json'), summary)
        # 플레이 결과 원본은 건드리지 않고 카메라 관리 원본을 별도로 저장합니다.
        camera_record = Path(self.info['camera_record_json'])
        camera_record.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(camera_record, summary)
        # 요약에 저장·종료 결과 요약(`summary`)을 저장합니다. 대상: `self._summary`.
        self._summary = summary
        # 현재 함수 처리를 끝내고 `dict(summary)` 호출 결과를 호출한 쪽으로 반환합니다.
        return dict(summary)
