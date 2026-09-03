# ==============================================================================
# [파일 정보]
# 파일명: core/data_manager.py
# 작성자: 개발자 (Developer)
# 설명: 환자별 맞춤 임계값(JSON), 세션 기록(CSV), 시계열 관절 궤적(NPZ) 통합 입출력 제어 모듈
# ==============================================================================

import csv  # CSV 파일 입출력 처리를 위한 내장 모듈 로드
import json  # JSON 파일 파싱 및 포맷팅을 위한 내장 모듈 로드
import os  # 운영체제 레벨의 경로 조작 및 디렉토리 확인을 위한 모듈 로드
from datetime import datetime  # 타임스탬프 및 날짜 포맷팅을 위한 모듈 로드
from pathlib import Path  # 객체 지향적인 파일 경로 제어를 위한 pathlib 모듈 로드
import numpy as np  # 대용량 바이너리 배열 압축 및 저장을 위한 NumPy 라이브러리 로드


def load_exercise_config(exercise_name: str) -> dict:  # 운동 종목별 타겟 관절 및 기본 설정 로드 함수
    project_root = Path(__file__).resolve().parent.parent  # 현재 파일 기준 프로젝트 최상위 루트 경로 탐색
    config_path = project_root / "configs" / "exercise_kpt_config_coco.json"  # 환경 설정 JSON 파일 절대 경로 조립

    if config_path.exists():  # 해당 경로에 파일이 실제로 존재하는 경우
        try:  # 파일 읽기 중 발생할 수 있는 에러 대비 블록
            with open(config_path, "r", encoding="utf-8") as f:  # 한글 깨짐 방지를 위해 UTF-8 인코딩으로 파일 오픈
                data = json.load(f)  # JSON 문자열을 파이썬 딕셔너리로 역직렬화
                exercise_info = data.get("exercises", {}).get(exercise_name)  # 타겟 운동의 설정값만 안전하게 추출
                if exercise_info:  # 추출된 설정값이 유효한 경우
                    return exercise_info  # 설정값 딕셔너리 반환
        except Exception as e:  # 파싱 실패 등 예외 발생 시
            print(f"[DataManager Warning] 설정 로드 실패: {e}")  # 콘솔에 경고 메시지 및 에러 내용 출력

    return {  # JSON 로드 실패 시 서버 다운 방지를 위한 기본(Fallback) 반환값 제공
        "exercise_name": exercise_name,  # 요청받은 운동명 할당
        "eval_config": {  # 기본 평가 방식 설정 딕셔너리
            "metric_type": "RELATIVE_Y",  # 기본 측정 방식을 상대 Y좌표로 지정
            "motion_direction": "DECREASING",  # 기본 동작 방향을 수치 감소형으로 지정
        },  # 평가 설정 종료
        "default_thresholds": {  # 기본 시작/목표 가동 범위 임계값
            "left": {"start_val": 0.8, "target_val": -0.3},  # 좌측 기본값 세팅
            "right": {"start_val": 0.8, "target_val": -0.3},  # 우측 기본값 세팅
        },  # 임계값 설정 종료
    }  # 기본 딕셔너리 반환 완료


class DataManager:  # 환자 데이터 기록 및 탐색을 전담하는 클래스 선언

    def __init__(self, player_id: str = "patient_1", patient_name: str = "미지정"):  # 매니저 객체 초기화 생성자
        self.player_id = player_id  # 환자 고유 식별 ID 저장
        self.patient_name = patient_name  # 환자 표시 이름 저장

        project_root = Path(__file__).resolve().parent.parent  # 프로젝트 루트 경로 재확인
        self.patient_dir = (project_root / "data" / player_id).resolve()  # 개별 환자 전용 데이터 폴더 경로 지정
        self.skeleton_dir = (self.patient_dir / "skeleton_memory").resolve()  # 스켈레톤 궤적 전용 하위 폴더 경로 지정

        self.patient_dir.mkdir(parents=True, exist_ok=True)  # 환자 폴더가 없으면 상위 폴더 포함 자동 생성
        self.skeleton_dir.mkdir(parents=True, exist_ok=True)  # 스켈레톤 폴더가 없으면 자동 생성

        self.json_path = self.patient_dir / f"{player_id}.json"  # 환자 프로필 및 임계값 저장용 JSON 파일 경로
        self.csv_path = self.patient_dir / "rep_details.csv"  # 운동 회차별 세부 성과 기록용 CSV 파일 경로

    def save_custom_threshold(self, exercise_name: str, threshold_data: dict, is_confirmed: bool = True):  # 산출된 환자 맞춤 임계값 저장 메서드
        data = {}  # 파일에 덮어쓸 베이스 딕셔너리 할당
        if self.json_path.exists():  # 기존 환자 프로필 JSON이 존재하는지 확인
            try:  # 파일 파싱 에러 방지 블록
                with open(self.json_path, "r", encoding="utf-8") as f:  # 읽기 모드로 오픈
                    data = json.load(f)  # 기존 프로필 데이터 로드
            except Exception:  # 파싱 에러 발생 시
                data = {}  # 빈 딕셔너리로 초기화하여 포맷 붕괴 방지

        data["patient_id"] = self.player_id  # 환자 ID 메타데이터 갱신
        data["patient_name"] = self.patient_name  # 환자 이름 메타데이터 갱신

        if "custom_thresholds" not in data or not isinstance(data["custom_thresholds"], dict):  # 맞춤 임계값 키가 없거나 타입이 불량할 때
            data["custom_thresholds"] = {}  # 빈 딕셔너리로 구조 초기화
        if "calibration_history" not in data or not isinstance(data["calibration_history"], list):  # 캘리브레이션 이력 키가 불량할 때
            data["calibration_history"] = []  # 빈 리스트로 구조 초기화

        data["custom_thresholds"][exercise_name] = threshold_data  # 해당 운동 종목의 최신 임계값 덮어쓰기 등록

        history_entry = {  # 캘리브레이션 누적 이력용 엔트리 딕셔너리 구성
            "exercise_name": exercise_name,  # 캘리브레이션 수행 종목
            "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 수행 시각 포맷팅 저장
            "is_confirmed": is_confirmed,  # 확정 여부 상태값 저장
            "thresholds": threshold_data,  # 산출된 임계값 데이터 저장
        }  # 이력 구성 완료
        data["calibration_history"].append(history_entry)  # 전체 이력 리스트에 신규 이력 추가

        with open(self.json_path, "w", encoding="utf-8") as f:  # 쓰기 모드로 오픈
            json.dump(data, f, ensure_ascii=False, indent=4)  # 한글 깨짐 방지 및 가독성 높은 들여쓰기(indent) 적용하여 저장

    def get_custom_threshold(self, exercise_name: str) -> dict:  # 디스크에서 특정 운동의 맞춤 임계값 조회 메서드
        if self.json_path.exists():  # 프로필 JSON 파일이 존재할 경우
            try:  # 안전한 파싱을 위한 예외 블록
                with open(self.json_path, "r", encoding="utf-8") as f:  # 읽기 모드로 오픈
                    data = json.load(f)  # 전체 데이터 역직렬화
                    custom_th = data.get("custom_thresholds", {})  # 임계값 모음 딕셔너리 안전 추출
                    if exercise_name in custom_th:  # 찾고자 하는 운동 종목이 존재하는 경우
                        return custom_th[exercise_name]  # 해당 종목의 좌/우 임계값 반환
            except Exception:  # 읽기 실패 시
                pass  # 로깅 생략 후 스킵
        return None  # 조회 실패 또는 파일 부재 시 None 반환 (엔진에서 기본값 사용하도록 유도)

    def save_rep_details_csv(self, rep_rows: list):  # 세션 완료 후 회차별 가동 범위 기록을 CSV에 누적 저장하는 메서드
        if not rep_rows:  # 저장할 회차 데이터가 빈 리스트라면
            return  # 불필요한 I/O 방지 후 종료

        fieldnames = [  # CSV 헤더 컬럼명 순서 정의
            "session_id", "timestamp", "side", "rep_num", "duration_sec",  # 메타 및 시간 정보
            "min_angle", "max_angle", "achieved_rom", "quality"  # 측정 결과 및 품질 지표 (불필요한 2D Bbox 제외)
        ]  # 컬럼명 리스트 완료

        file_exists = self.csv_path.exists()  # 파일 존재 여부 확인 (헤더 추가 분기를 위해)
        with open(self.csv_path, "a", newline="", encoding="utf-8-sig") as f:  # 엑셀 한글 깨짐 방지용 utf-8-sig 인코딩 및 Append 모드 오픈
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")  # 정의되지 않은 추가 키는 무시하는 DictWriter 생성
            if not file_exists:  # 파일이 최초 생성되는 상황이라면
                writer.writeheader()  # 첫 줄에 헤더 컬럼명 작성
            for row in rep_rows:  # 입력받은 회차 결과 리스트 순회
                writer.writerow(row)  # 개별 회차 결과를 CSV 행으로 누적 작성

    def save_trajectory_npz(self, session_id: str, exercise_name: str, timestamps: list, values: list, keypoints: list):  # 17개 관절 시퀀스를 고압축 바이너리로 저장하는 메서드
        if not timestamps:  # 저장할 타임스탬프 데이터가 없다면
            return  # 불필요한 저장 프로세스 종료

        npz_path = self.skeleton_dir / f"{session_id}_skeleton.npz"  # 세션 ID 기반의 고유한 파일 경로 조립
        np.savez_compressed(  # 내부 데이터를 zlib으로 고압축하여 용량을 최소화하는 함수 호출
            npz_path,  # 타겟 파일 경로
            timestamps=np.array(timestamps, dtype=np.float32),  # 시간 배열을 가벼운 float32 NumPy 배열로 형변환하여 할당
            values=np.array(values, dtype=np.float32),  # 모션 연산 수치 배열을 형변환하여 할당
            keypoints=np.array(keypoints, dtype=np.float32),  # (N, 17, 3) 스켈레톤 행렬을 형변환하여 할당
            player_id=self.player_id,  # 환자 식별 아이디 메타데이터 추가
            exercise_name=exercise_name,  # 운동명 메타데이터 추가
        )  # 파일 기록 완료