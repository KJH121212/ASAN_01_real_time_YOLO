"""공통 PatientData 식별자 연결·등록 분리·Unity 파일 보존을 검증합니다."""

# 새 CSV의 실제 열과 값을 읽습니다.
import csv
# 자정을 넘는 날짜 규칙을 실제 로그로 확인합니다.
from datetime import datetime
# 실제 파일 명세를 확인합니다.
import json
# 임시 경로에서만 합성 데이터를 기록합니다.
from pathlib import Path
import tempfile
# 테스트 실행기와 수치 배열 도구입니다.
import unittest
import numpy as np
# 운영 코드의 저장 관리자·시간대·오버레이 추론을 그대로 사용합니다.
from core.data_manager import DataManager
from core.session_log import KST
from core.video_overlay import infer_exercise


# 카메라 로그에 IMU 값이나 Unity 결과를 임의 생성하지 않는지 확인합니다.
class PatientDataLayoutTests(unittest.TestCase):
    # 실제 환자나 영상을 사용하지 않는 소규모 프레임 입력입니다.
    def append(self, manager):
        # 좌우 다른 값으로 저장 시 순서가 유지되는지도 확인합니다.
        sides = {'left': {'start_val': 0., 'target_val': 1.}, 'right': {'start_val': 0., 'target_val': 1.}}
        manager.session_log.append_frame(.125, 0, [], {'val': .3}, {'val': .6}, 'LENGTH', sides)

    # 같은 플레이·운동의 여러 세트는 같은 공통 ID와 서로 다른 파일 ID를 가집니다.
    def test_shared_ids_join_sets_and_preserve_unity_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patient = root / 'v2.3.4(27)' / '0001' / '02'
            # Unity가 이미 기록한 파일을 합성하고 바이트가 바뀌지 않는지 확인합니다.
            sentinels = [patient / 'PlayData/2026-10-22.xlsx', patient / 'Management/Records/PLAY1.json', root / '_Registrations/0001.json']
            for path in sentinels:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'unity-owned-original')
            manager = DataManager('1', log_root=root, data_version='v2.3.4(27)', registration='02')
            summaries = []
            for set_number in (1, 2):
                info = manager.begin_session('knee_extension', 'MAIN', round_number=2, slot=3,
                    set_number=set_number, play_session_id='PLAY1', exercise_record_id='EX1',
                    started_at=datetime(2026, 10, 22, 12, tzinfo=KST))
                self.append(manager)
                summaries.append(manager.finish_session([]))
                # 일자·플레이·소스 폴더 및 문서의 파일명 구성을 검증합니다.
                csv_path = Path(info['frame_csv'])
                self.assertEqual(csv_path.parent, patient / 'Sensors/2026-10-22/PLAY1/Camera')
                self.assertTrue(csv_path.name.startswith(f'R02_S03_Set{set_number:02d}_Play_Camera_'))
                with csv_path.open(encoding='utf-8-sig', newline='') as file:
                    row = next(csv.DictReader(file))
                self.assertEqual(row['time'], '0.125')
                self.assertEqual(row['play_session_id'], 'PLAY1')
                self.assertEqual(row['exercise_record_id'], 'EX1')
                self.assertEqual(row['metric_unit'], 'body_length')
                self.assertNotIn('AccelX', row)
                self.assertNotIn('GyroX', row)
                with np.load(info['skeleton_npz'], allow_pickle=False) as archive:
                    self.assertEqual(str(archive['player_id']), '0001')
                    self.assertEqual(str(archive['registration']), '02')
                    self.assertEqual(str(archive['play_session_id']), 'PLAY1')
                # 운동명이 없는 새 경로도 명세를 통해 오버레이할 운동을 찾습니다.
                self.assertEqual(infer_exercise(csv_path.with_suffix('.mp4')), 'knee_extension')
                record = json.loads(Path(info['camera_record_json']).read_text(encoding='utf-8'))
                self.assertEqual(record['exercise_record_id'], 'EX1')
                # 절대 경로가 달라져도 등록 차수 폴더 기준 상대 경로로 원본을 찾을 수 있습니다.
                self.assertEqual(patient / record['frame_csv_relative'], csv_path)
            self.assertNotEqual(summaries[0]['sensor_file_id'], summaries[1]['sensor_file_id'])
            for path in sentinels:
                self.assertEqual(path.read_bytes(), b'unity-owned-original')

    # 같은 번호를 재등록하거나 버전을 바꿔도 과거 프로필이 유입되지 않아야 합니다.
    def test_registration_and_version_isolate_personal_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            first = DataManager('0001', log_root=temporary)
            first.save_custom_threshold('knee_extension', {'left': {'start_val': 1., 'target_val': .5}})
            self.assertIsNotNone(first.get_custom_threshold('knee_extension'))
            for parameters in ({'registration': 2}, {'data_version': 'v2.0.0(0)'}):
                other = DataManager('0001', log_root=temporary, **parameters)
                self.assertIsNone(other.get_custom_threshold('knee_extension'))

    # 파일 시작일과 요약 종료일이 달라도 공통 ID로 연결되어야 합니다.
    def test_midnight_keeps_sensor_date_and_uses_end_date_for_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            manager = DataManager('0001', log_root=temporary)
            info = manager.begin_session('knee_extension', 'CALIBRATION', play_session_id='NIGHT',
                started_at=datetime(2026, 10, 21, 23, 59, 59, tzinfo=KST))
            self.append(manager)
            summary = manager.finish_session([], end_elapsed_sec=2.)
            self.assertIn('2026-10-21', Path(info['frame_csv']).parts)
            self.assertEqual(Path(summary['gamedata_csv']).name, '2026-10-22.csv')
            self.assertEqual(summary['collection_phase'], 'Calibration')
            self.assertTrue(Path(info['frame_csv']).is_file())

    # 버전·등록·공통 ID에 경로 구분자가 들어오면 기록 전에 거부합니다.
    def test_invalid_context_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            for parameters in ({'data_version': '../v1'}, {'registration': 0}, {'registration': True}):
                with self.assertRaises(ValueError):
                    DataManager('0001', log_root=temporary, **parameters)
            manager = DataManager('0001', log_root=temporary)
            for parameters in ({'play_session_id': '../escape'}, {'exercise_record_id': 'CON'}, {'slot': -1}):
                with self.assertRaises(ValueError):
                    manager.begin_session('knee_extension', 'MAIN', **parameters)
