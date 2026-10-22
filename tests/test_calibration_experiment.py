# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `csv` 모듈을 불러옵니다.
import csv
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# `tests.evaluate_calibration_videos`에서 `CaptureEngine`, `analyze_frames` 기능을 가져옵니다.
from tests.evaluate_calibration_videos import CaptureEngine, analyze_frames


# `CalibrationExperimentTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class CalibrationExperimentTests(unittest.TestCase):
    # 회귀 테스트: `test_eof_is_not_success_even_when_both_counts_are_three` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_eof_is_not_success_even_when_both_counts_are_three(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 경로에 계산식 `Path(directory) / 'frames.csv'`의 결과를 저장합니다. 대상: `path`.
            path = Path(directory) / 'frames.csv'
            # `rows` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `rows`.
            rows = []
            # `enumerate((('COLLECTING', 2), ('COOLDOWN', 3)))`에서 항목을 차례로 꺼내 `(index, (step, reps))`에 넣고 아래 처리를 반복합니다.
            for index, (step, reps) in enumerate((('COLLECTING', 2), ('COOLDOWN', 3))):
                # `row` 값에 `dict(timestamp_ms=index * 100, is_occluded='False', calib_step=step)` 호출 결과를 저장합니다. 대상: `row`.
                row = dict(timestamp_ms=index * 100, is_occluded='False', calib_step=step)
                # `('left', 'right')`에서 항목을 차례로 꺼내 `side`에 넣고 아래 처리를 반복합니다.
                for side in ('left', 'right'):
                    # `row.update({f'{side}_rep_count': reps, f'{side}_value': 0.5, f'{side}_state': 'WAITING'})`을 호출하여 해당 단계의 작업을 수행합니다.
                    row.update({f'{side}_rep_count': reps, f'{side}_value': 0.5,
                                # 구성 항목: f'{side} 단계(`f'{side}_state'`)에 'WAITING'을 지정합니다.
                                f'{side}_state': 'WAITING'})
                # `rows.append(row)`을 호출하여 항목 하나를 끝에 추가합니다.
                rows.append(row)
            # `path.open('w', encoding='utf-8-sig', newline='')`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with path.open('w', encoding='utf-8-sig', newline='') as file:
                # CSV 또는 영상 출력 객체에 `csv.DictWriter(file, fieldnames=list(rows[0]))` 호출 결과를 저장합니다. 대상: `writer`.
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                # `writer.writeheader()`을 호출하여 CSV의 열 이름 행을 기록합니다.
                writer.writeheader()
                # `writer.writerows(rows)`을 호출하여 CSV에 여러 행을 기록합니다.
                writer.writerows(rows)
            # 함수 또는 처리 단계의 결과에 `analyze_frames(path)` 호출 결과를 저장합니다. 대상: `result`.
            result = analyze_frames(path)
            # 검증문 `self.assertFalse(result['finished'])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertFalse(result['finished'])
            # 검증문 `self.assertEqual(result['left_final_reps'], 3)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['left_final_reps'], 3)
            # 검증문 `self.assertEqual(result['final_step'], 'COOLDOWN')`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(result['final_step'], 'COOLDOWN')

    # 회귀 테스트: `test_raw_cache_preserves_missing_frames_and_joint_ids` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_raw_cache_preserves_missing_frames_and_joint_ids(self):
        # `capture` 값에 `CaptureEngine(None)` 호출 결과를 저장합니다. 대상: `capture`.
        capture = CaptureEngine(None)
        # `capture.raw` 값에 순서·구성을 지정한 `[None, [{'id': 9, 'x': 0.2, 'y': 0.7, 'score': 0.9}]]`을 저장합니다. 대상: `capture.raw`.
        capture.raw = [None, [{'id': 9, 'x': 0.2, 'y': 0.7, 'score': 0.9}]]
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as directory:
            # 경로에 계산식 `Path(directory) / 'raw.npz'`의 결과를 저장합니다. 대상: `path`.
            path = Path(directory) / 'raw.npz'
            # `capture.save(path, 30)`을 호출하여 해당 단계의 작업을 수행합니다.
            capture.save(path, 30)
            # `np.load(path)`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
            with np.load(path) as data:
                # 검증문 `self.assertEqual(data['keypoints'].shape, (2, 17, 3))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(data['keypoints'].shape, (2, 17, 3))
                # 검증문 `self.assertTrue(np.isnan(data['keypoints'][0, :, :2]).all())`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertTrue(np.isnan(data['keypoints'][0, :, :2]).all())
                # 검증문 `np.testing.assert_allclose(data['keypoints'][1, 9], [0.2, 0.7, 0.9])`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                np.testing.assert_allclose(data['keypoints'][1, 9], [0.2, 0.7, 0.9])
                # 검증문 `self.assertEqual(float(data['fps']), 30)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertEqual(float(data['fps']), 30)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
