# 한국어 코드 해설: 환자별 메타데이터·개인 가동 범위와 세션 로그의 저장·조회 진입점을 제공합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [모듈 정보]
# 파일: core/data_manager.py
# 설명: 환자 JSON 프로필과 반복별 CSV 로그를 통합 관리합니다.
# 신체 정규화 전 스켈레톤 궤적은 NPZ로 보관합니다.
# ==============================================================================

# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# 이 파일에서 사용할 `json` 모듈을 불러옵니다.
import json
# `datetime`에서 `datetime` 기능을 가져옵니다.
from datetime import datetime
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np
# `core.session_log`에서 `KST`, `SessionLog`, `default_log_root`, `path_component`, `write_json_atomic` 기능을 가져옵니다.
from core.session_log import KST, SessionLog, default_log_root, path_component, write_json_atomic
# 공통 버전 폴더와 등록 차수 검증 규칙입니다.
from core.patient_data import DEFAULT_DATA_VERSION, registration_number, version_folder


# 불러오기 exercise 설정 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def load_exercise_config(exercise_name: str) -> dict:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """JSON 설정 파일로부터 특정 운동의 평가 규칙 및 기본 임계값을 로드합니다."""
    # project 최상위 경로에 `Path(__file__).resolve().parent.parent` 값(`Path(__file__).resolve().parent.parent`)을 저장합니다. 대상: `project_root`.
    project_root = Path(__file__).resolve().parent.parent
    # 설정 경로에 계산식 `project_root / 'configs' / 'exercise_kpt_config_coco.json'`의 결과를 저장합니다. 대상: `config_path`.
    config_path = project_root / "configs" / "exercise_kpt_config_coco.json"

    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print(f"[DEBUG][CONFIG] Reading exercise configuration file: {config_path}")

    # 조건 `config_path.exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
    if config_path.exists():
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `open(config_path, 'r', encoding='utf-8')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with open(config_path, "r", encoding="utf-8") as f:
                # 데이터에 `json.load(f)` 호출 결과를 저장합니다. 대상: `data`.
                data = json.load(f)
                # `exercise_info` 값에 `data.get('exercises', {}).get(exercise_name)` 호출 결과를 저장합니다. 대상: `exercise_info`.
                exercise_info = data.get("exercises", {}).get(exercise_name)
                # 조건 `exercise_info`을 만족하는 경우에만 아래 분기를 실행합니다.
                if exercise_info:
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(f"[DEBUG][CONFIG] Exercise '{exercise_name}' configuration loaded.")
                    # 현재 함수 처리를 끝내고 `exercise_info` 값(`exercise_info`)을 호출한 쪽으로 반환합니다.
                    return exercise_info
                # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
                else:
                    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                    print(f"[WARN][CONFIG] Exercise '{exercise_name}' not defined. Falling back to default.")
        # `Exception` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except Exception as e:
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"[ERROR][CONFIG] Failed reading JSON config: {e}")
            # `traceback.print_exc()`을 호출하여 현재 예외의 상세 호출 경로를 출력합니다.
            traceback.print_exc()

    # 파싱 실패 또는 파일 미존재 시 시스템 다운을 방지하기 위한 폴백 기본 구조체
    # `fallback` 값에 키 'exercise_name', 'eval_config', 'default_thresholds'로 구성한 사전을 저장합니다. 대상: `fallback`.
    fallback = {
        # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 운동 설정의 식별 이름(`exercise_name`)을 지정합니다.
        "exercise_name": exercise_name,
        # 구성 항목: eval 설정(`'eval_config'`)에 키 'metric_type', 'motion_direction'로 구성한 사전을 지정합니다.
        "eval_config": {
            # 구성 항목: 측정 방식(각도·좌표·길이·비율)(`'metric_type'`)에 'COORDINATE'을 지정합니다.
            "metric_type": "COORDINATE",
            # 구성 항목: 측정값 증가·감소 방향(`'motion_direction'`)에 'DECREASING'을 지정합니다.
            "motion_direction": "DECREASING",
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        },
        # 구성 항목: 기본 가동 범위 기준(`'default_thresholds'`)에 키 'left', 'right'로 구성한 사전을 지정합니다.
        "default_thresholds": {
            # 구성 항목: 왼쪽(`'left'`)에 키 'start_val', 'target_val'로 구성한 사전; 개인 가동 범위의 시작값(`'start_val'`)에 0.8; 개인 가동 범위의 목표값(`'target_val'`)에 계산식 `-0.3`의 결과를 지정합니다.
            "left": {"start_val": 0.8, "target_val": -0.3},
            # 구성 항목: 오른쪽(`'right'`)에 키 'start_val', 'target_val'로 구성한 사전; 개인 가동 범위의 시작값(`'start_val'`)에 0.8; 개인 가동 범위의 목표값(`'target_val'`)에 계산식 `-0.3`의 결과를 지정합니다.
            "right": {"start_val": 0.8, "target_val": -0.3},
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        },
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    }
    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print(f"[DEBUG][CONFIG] Returning fallback configuration for: {exercise_name}")
    # 현재 함수 처리를 끝내고 `fallback` 값(`fallback`)을 호출한 쪽으로 반환합니다.
    return fallback


# `DataManager` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class DataManager:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """Patient profiles and session logs under a configurable external root."""

    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, player_id="0001", patient_name="Unknown", log_root=None, patient_metadata=None,
                 data_version=DEFAULT_DATA_VERSION, registration=1):
        # 환자 식별 번호의 기존 필드명에 `path_component(player_id, 'patient_id')` 호출 결과를 저장합니다. 대상: `self.player_id`.
        self.player_id = path_component(player_id, "patient_id")
        # 숫자 환자번호는 앞의 0을 포함한 네 자리로 보관합니다. 비숫자 ID는 기존 테스트 호환용입니다.
        if self.player_id.isdigit():
            if len(self.player_id) > 4:
                raise ValueError('patient_id must have at most four digits')
            self.player_id = self.player_id.zfill(4)
        # 환자번호가 같아도 등록 차수·버전이 다르면 새 폴더를 사용합니다.
        self.data_version = version_folder(data_version)
        self.registration = registration_number(registration)
        # `self.patient_name` 값에 `patient_name` 값(`patient_name`)을 저장합니다. 대상: `self.patient_name`.
        self.patient_name = patient_name
        # 세션 로그를 저장하는 최상위 폴더에 `log_root`이면 `Path(log_root).expanduser().resolve()`, 아니면 `default_log_root()`을 저장합니다. 대상: `self.log_root`.
        self.log_root = Path(log_root).expanduser().resolve() if log_root else default_log_root()
        # `(self.log_root, *self.log_root.parents)`에서 항목을 차례로 꺼내 `directory`에 넣고 아래 처리를 반복합니다.
        for directory in (self.log_root, *self.log_root.parents):
            # 조건 `directory.exists() and (not directory.is_dir())`을 만족하는 경우에만 아래 분기를 실행합니다.
            if directory.exists() and not directory.is_dir():
                # 이 조건에서는 처리를 계속할 수 없으므로 `NotADirectoryError(f'로그 저장 폴더는 영상 파일이 아닌 폴더로 지정하세요: {directory}. 영상 경로는 Video File Path에 입력하세요.')` 예외를 발생시킵니다.
                raise NotADirectoryError(
                    # 앞줄에서 시작한 식·인수·자료 구조를 `f"로그 저장 폴더는 영상 파일이 아닌 폴더로 지정하세요: {directory}. "` 내용으로 이어서 구성합니다.
                    f"로그 저장 폴더는 영상 파일이 아닌 폴더로 지정하세요: {directory}. "
                    # 앞줄에서 시작한 식·인수·자료 구조를 `"영상 경로는 Video File Path에 입력하세요.")` 내용으로 이어서 구성합니다.
                    "영상 경로는 Video File Path에 입력하세요.")
        # 버전·환자번호·등록 차수가 모두 같은 경우에만 같은 저장 영역을 사용합니다.
        self.patient_dir = self.log_root / self.data_version / self.player_id / self.registration
        # `self.patient_dir.mkdir(parents=True, exist_ok=True)`을 호출하여 필요한 폴더를 생성합니다.
        self.patient_dir.mkdir(parents=True, exist_ok=True)
        # Unity의 Records·Recovery 폴더를 건드리지 않는 카메라 전용 관리 영역입니다.
        self.camera_management = self.patient_dir / 'Management' / 'Camera'
        self.camera_management.mkdir(parents=True, exist_ok=True)
        self.json_path = self.camera_management / 'profile.json'
        # 카메라 측에 제공된 환자 정보는 Unity 소유 파일과 분리해 관리합니다.
        self.metadata_path = self.camera_management / "patient_metadata.csv"
        # 기존 호환 json 경로에 계산식 `Path(__file__).resolve().parent.parent / 'data' / self.player_id / f'{self.player_id}.json'`의 결과를 저장합니다. 대상: `self.legacy_json_path`.
        self.legacy_json_path = Path(__file__).resolve().parent.parent / "data" / self.player_id / f"{self.player_id}.json"
        # 환자의 키·몸무게 등 부가 정보에 계산식 `patient_metadata or {}`의 결과를 저장합니다. 대상: `self.patient_metadata`.
        self.patient_metadata = patient_metadata or {}
        # 조건 `not isinstance(self.patient_metadata, dict)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not isinstance(self.patient_metadata, dict):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('patient_metadata must be an object')` 예외를 발생시킵니다.
            raise ValueError("patient_metadata must be an object")
        # 현재 세션의 CSV·NPZ·MP4 기록 객체에 None을 저장합니다. 대상: `self.session_log`.
        self.session_log = None
        # `(self.buf_timestamps, self.buf_values, self.buf_raw_keypoints)` 값에 순서·구성을 지정한 `([], [], [])`을 저장합니다. 대상: `(self.buf_timestamps, self.buf_values, self.buf_raw_keypoints)`.
        self.buf_timestamps, self.buf_values, self.buf_raw_keypoints = [], [], []
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[LOG] Patient directory: {self.patient_dir}", flush=True)

    # 저장 부가 정보 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def _save_metadata(self):
        # `fields` 값에 순서·구성을 지정한 `['patient_id', 'patient_name', 'height_cm', 'weight_kg']`을 저장합니다. 대상: `fields`.
        fields = ["patient_id", "patient_name", "height_cm", "weight_kg"]
        # 환자 또는 세션 부가 정보에 빈 사전을 저장합니다. 대상: `metadata`.
        metadata = {}
        # 조건 `self.metadata_path.exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.metadata_path.exists():
            # `self.metadata_path.open(newline='', encoding='utf-8-sig')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with self.metadata_path.open(newline="", encoding="utf-8-sig") as file:
                # 환자 또는 세션 부가 정보에 `next(csv.DictReader(file), {})` 호출 결과를 저장합니다. 대상: `metadata`.
                metadata = next(csv.DictReader(file), {})
        # `metadata.update(patient_id=self.player_id, patient_name=self.patient_name)`을 호출하여 해당 단계의 작업을 수행합니다.
        metadata.update(patient_id=self.player_id, patient_name=self.patient_name)
        # `('height_cm', 'weight_kg')`에서 항목을 차례로 꺼내 `key`에 넣고 아래 처리를 반복합니다.
        for key in ("height_cm", "weight_kg"):
            # 조건 `key in self.patient_metadata and self.patient_metadata[key] is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if key in self.patient_metadata and self.patient_metadata[key] is not None:
                # 측정값에 `float(self.patient_metadata[key])` 호출 결과를 저장합니다. 대상: `value`.
                value = float(self.patient_metadata[key])
                # 조건 `not np.isfinite(value) or value <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
                if not np.isfinite(value) or value <= 0:
                    # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError(f'{key} must be positive')` 예외를 발생시킵니다.
                    raise ValueError(f"{key} must be positive")
                # `metadata[key]` 값에 측정값(`value`)을 저장합니다. 대상: `metadata[key]`.
                metadata[key] = value
        # 원본을 안전하게 교체하기 위한 임시 파일 경로에 `self.metadata_path.with_suffix('.csv.tmp')` 호출 결과를 저장합니다. 대상: `temporary`.
        temporary = self.metadata_path.with_suffix(".csv.tmp")
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # `temporary.open('w', newline='', encoding='utf-8-sig')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with temporary.open("w", newline="", encoding="utf-8-sig") as file:
                # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')` 호출 결과를 저장합니다. 대상: `writer`.
                writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
                # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
                writer.writeheader()
                # `writer.writerow(metadata)`을 호출하여 CSV에 행 하나를 기록합니다.
                writer.writerow(metadata)
            # `temporary.replace(self.metadata_path)`을 호출하여 임시 파일을 최종 경로로 교체합니다.
            temporary.replace(self.metadata_path)
        # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
        finally:
            # `temporary.unlink(missing_ok=True)`을 호출하여 지정한 파일을 제거합니다.
            temporary.unlink(missing_ok=True)

    # begin 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def begin_session(self, exercise_name, mode, session=None, round_number=1, set_number=1,
                      # 이 줄의 입력 인수: `record_video` = 영상 녹화 여부; `video_fps` = 저장 영상의 초당 프레임 수; `started_at` = `started_at` 값; `target_score` = 게임 목표 점수. 기본값·형식은 아래 선언을 따릅니다.
                      record_video=False, video_fps=30.0, started_at=None, target_score=None,
                      play_session_id=None, exercise_record_id=None, slot=1):
        # 조건 `self.session_log is not None and self.session_log._summary is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.session_log is not None and self.session_log._summary is None:
            # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Finish the active log before starting another')` 예외를 발생시킵니다.
            raise RuntimeError("Finish the active log before starting another")
        # `self._save_metadata()`을 호출하여 해당 단계의 작업을 수행합니다.
        self._save_metadata()
        # 현재 세션의 CSV·NPZ·MP4 기록 객체에 `SessionLog(self.patient_dir, exercise_name, mode, session, round_number, set_number, record_video, video_fps, started_at, target_score)` 호출 결과를 저장합니다. 대상: `self.session_log`.
        self.session_log = SessionLog(self.patient_dir, exercise_name, mode, session,
                                      # 앞줄에서 시작한 식·인수·자료 구조를 `round_number, set_number, record_video, video_fps, started_at, target_score)` 내용으로 이어서 구성합니다.
                                      round_number, set_number, record_video, video_fps, started_at, target_score,
                                      play_session_id=play_session_id, exercise_record_id=exercise_record_id,
                                      slot=slot, patient_id=self.player_id, data_version=self.data_version,
                                      registration=self.registration)
        # `self.buf_timestamps` 값에 프레임별 경과 시각 목록(`self.session_log.timestamps`)을 저장합니다. 대상: `self.buf_timestamps`.
        self.buf_timestamps = self.session_log.timestamps
        # buf 측정값에 측정값(`self.session_log.values`)을 저장합니다. 대상: `self.buf_values`.
        self.buf_values = self.session_log.values
        # buf raw 관절에 COCO 관절 좌표 목록(`self.session_log.keypoints`)을 저장합니다. 대상: `self.buf_raw_keypoints`.
        self.buf_raw_keypoints = self.session_log.keypoints
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[LOG] Session files: {self.session_log.base}.*", flush=True)
        # 현재 함수 처리를 끝내고 `dict(self.session_log.info)` 호출 결과를 호출한 쪽으로 반환합니다.
        return dict(self.session_log.info)

    # 저장 custom 가동 범위 기준 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def save_custom_threshold(self, exercise_name, threshold_data, is_confirmed=True):
        # 데이터에 빈 사전을 저장합니다. 대상: `data`.
        data = {}
        # 새 캘리브레이션을 처음 저장할 때도 기존 프로필의 다른 내용은 유지합니다.
        # 읽을 원본 경로 또는 데이터에 `self.json_path.exists()`이면 `self.json_path`, 아니면 `self.legacy_json_path`을 저장합니다. 대상: `source`.
        # 재등록한 환자에게 과거 등록의 개인 가동 범위를 복사하지 않습니다.
        source = self.json_path
        # 이전 위치의 프로필 보조 읽기는 최초 버전·최초 등록에만 적용합니다.
        if not source.exists() and self.registration == '01' and self.data_version == DEFAULT_DATA_VERSION:
            source = next((path for path in (self.log_root / self.player_id / f'{self.player_id}.json', self.legacy_json_path) if path.exists()), self.json_path)
        # 조건 `source.exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
        if source.exists():
            # 데이터에 JSON 문자열을 해석한 Python 값(`json.loads(source.read_text(encoding='utf-8'))`)을 저장합니다. 대상: `data`.
            data = json.loads(source.read_text(encoding="utf-8"))
        # `data.update(patient_id=self.player_id, patient_name=self.patient_name)`을 호출하여 해당 단계의 작업을 수행합니다.
        data.update(patient_id=self.player_id, patient_name=self.patient_name)
        # 등록·버전 정보를 프로필에도 기록하여 폴더를 옮겼을 때 출처를 확인합니다.
        data.update(data_version=self.data_version, registration=self.registration)
        # `data.setdefault('custom_thresholds', {})[exercise_name]` 값에 가동 범위 기준 데이터(`threshold_data`)을 저장합니다. 대상: `data.setdefault('custom_thresholds', {})[exercise_name]`.
        data.setdefault("custom_thresholds", {})[exercise_name] = threshold_data
        # `data.setdefault('calibration_history', []).append({'exercise_name': exercise_name, 'calibration_date': datetime.now(KST).isoformat(), 'is_confirmed': is_confirmed, 'thre…`을 호출하여 항목 하나를 끝에 추가합니다.
        data.setdefault("calibration_history", []).append({
            # 구성 항목: 운동 설정의 식별 이름(`'exercise_name'`)에 운동 설정의 식별 이름(`exercise_name`)을 지정합니다.
            "exercise_name": exercise_name,
            # 구성 항목: 캘리브레이션 date(`'calibration_date'`)에 `datetime.now(KST).isoformat()` 호출 결과를 지정합니다.
            "calibration_date": datetime.now(KST).isoformat(),
            # 구성 항목: 여부 confirmed(`'is_confirmed'`)에 여부 confirmed(`is_confirmed`); 좌우 가동 범위 시작값·목표값(`'thresholds'`)에 가동 범위 기준 데이터(`threshold_data`)을 지정합니다.
            "is_confirmed": is_confirmed, "thresholds": threshold_data,
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        })
        # `write_json_atomic(self.json_path, data)`을 호출하여 해당 단계의 작업을 수행합니다.
        write_json_atomic(self.json_path, data)

    # 조회 custom 가동 범위 기준 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def get_custom_threshold(self, exercise_name):
        # `(self.json_path, self.legacy_json_path)`에서 항목을 차례로 꺼내 `source`에 넣고 아래 처리를 반복합니다.
        # 구형 프로필은 첫 등록에 한해서 보조로 읽고 원래 파일은 유지합니다.
        sources = (self.json_path, self.log_root / self.player_id / f'{self.player_id}.json', self.legacy_json_path) if self.registration == '01' and self.data_version == DEFAULT_DATA_VERSION else (self.json_path,)
        for source in sources:
            # 조건 `not source.exists()`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not source.exists():
                # 현재 항목의 남은 처리를 건너뛰고 반복문의 다음 항목으로 이동합니다.
                continue
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `profile` 값에 JSON 문자열을 해석한 Python 값(`json.loads(source.read_text(encoding='utf-8'))`)을 저장합니다. 대상: `profile`.
                profile = json.loads(source.read_text(encoding="utf-8"))
                # 잡음과 최소 변화량을 반영한 동작 시작 기준에 `profile.get('custom_thresholds', {}).get(exercise_name)` 호출 결과를 저장합니다. 대상: `threshold`.
                threshold = profile.get("custom_thresholds", {}).get(exercise_name)
                # 조건 `threshold is not None`을 만족하는 경우에만 아래 분기를 실행합니다.
                if threshold is not None:
                    # 현재 함수 처리를 끝내고 잡음과 최소 변화량을 반영한 동작 시작 기준(`threshold`)을 호출한 쪽으로 반환합니다.
                    return threshold
            # `(OSError, ValueError, AttributeError)` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
            except (OSError, ValueError, AttributeError) as exc:
                # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
                print(f"[WARN][LOG] Could not read profile {source}: {exc}")
        # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
        return None

    # 종료 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def finish_session(self, rep_rows, status="COMPLETED", end_elapsed_sec=None):
        # 조건 `self.session_log is None`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.session_log is None:
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None
        # 현재 함수 처리를 끝내고 `self.session_log.finish(rep_rows, status, end_elapsed_sec=end_elapsed_sec)` 호출 결과를 호출한 쪽으로 반환합니다.
        return self.session_log.finish(rep_rows, status, end_elapsed_sec=end_elapsed_sec)
