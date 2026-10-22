# 한국어 코드 해설: 자동 회귀 테스트와 테스트 입력·기대 결과를 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 이 파일에서 사용할 `copy` 모듈을 불러옵니다.
import copy
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `tempfile` 모듈을 불러옵니다.
import tempfile
# 이 파일에서 사용할 `unittest` 모듈을 불러옵니다.
import unittest

# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np

# `core.data_manager`에서 `DataManager` 기능을 가져옵니다.
from core.data_manager import DataManager
# `core.motion_engine`에서 `MotionEngine` 기능을 가져옵니다.
from core.motion_engine import MotionEngine
# `utils.image_geometry`에서 `letterbox_preview`, `resize_within` 기능을 가져옵니다.
from utils.image_geometry import letterbox_preview, resize_within
# `utils.normalization`에서 `PoseNormalizer` 기능을 가져옵니다.
from utils.normalization import PoseNormalizer


# `pose` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def pose(width, height):
    # `(cx, cy)` 값에 순서·구성을 지정한 `(width / 2, height / 2)`을 저장합니다. 대상: `(cx, cy)`.
    cx, cy = width / 2, height / 2
    # `pixels` 값에 키 5, 6, 11, 12, 7, 9로 구성한 사전을 저장합니다. 대상: `pixels`.
    pixels = {5: (-40, -100), 6: (40, -100), 11: (-30, 0), 12: (30, 0),
              # 구성 항목: `7` 값(`7`)에 순서·구성을 지정한 `(-100, -50)`; `9` 값(`9`)에 순서·구성을 지정한 `(-20, -10)`을 지정합니다.
              7: (-100, -50), 9: (-20, -10)}
    # 현재 함수 처리를 끝내고 반복·조건식 `[dict(id=i, x=(cx + pixels.get(i, (0, 0))[0]) / width, y=(cy + pixels.get(i, (0, 0))[1]) / height, score=1.0, image_aspect_ratio=width / height) for i in range(17)]`으로 구성한 값을 호출한 쪽으로 반환합니다.
    return [dict(id=i, x=(cx + pixels.get(i, (0, 0))[0]) / width,
                 # 구성 항목: `y` 값(`y`)에 계산식 `(cy + pixels.get(i, (0, 0))[1]) / height`의 결과; 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`score`)에 1.0을 지정합니다.
                 y=(cy + pixels.get(i, (0, 0))[1]) / height, score=1.,
                 # 구성 항목: 입력 영상의 가로/세로 비율(`image_aspect_ratio`)에 계산식 `width / height`의 결과를 지정합니다.
                 image_aspect_ratio=width / height) for i in range(17)]


# `PortraitInputTests` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class PortraitInputTests(unittest.TestCase):
    # 회귀 테스트: `test_same_pixel_geometry_has_same_normalized_pose_and_angle_in_all_aspects` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_same_pixel_geometry_has_same_normalized_pose_and_angle_in_all_aspects(self):
        # 골반 중심·몸통 길이 기준 좌표 변환기에 `PoseNormalizer(invert_y=True)` 호출 결과를 저장합니다. 대상: `normalizer`.
        normalizer = PoseNormalizer(invert_y=True)
        # 생성 또는 검증할 분석 엔진에 `MotionEngine({'eval_config': {'metric_type': 'ANGLE', 'primary_kpts': {'left': [5, 7, 9]}}})` 호출 결과를 저장합니다. 대상: `engine`.
        engine = MotionEngine({'eval_config': {'metric_type': 'ANGLE',
                             # 구성 항목: `primary_kpts` 값(`'primary_kpts'`)에 키 'left'로 구성한 사전; 왼쪽(`'left'`)에 순서·구성을 지정한 `[5, 7, 9]`을 지정합니다.
                             'primary_kpts': {'left': [5, 7, 9]}}})
        # `reference` 값에 None을 저장합니다. 대상: `reference`.
        reference = None
        # `((1920, 1080), (1080, 1920), (640, 640))`에서 항목을 차례로 꺼내 `(width, height)`에 넣고 아래 처리를 반복합니다.
        for width, height in ((1920, 1080), (1080, 1920), (640, 640)):
            # 계산에 사용할 관절 또는 좌표 묶음에 `pose(width, height)` 호출 결과를 저장합니다. 대상: `points`.
            points = pose(width, height)
            # `original` 값에 `copy.deepcopy(points)` 호출 결과를 저장합니다. 대상: `original`.
            original = copy.deepcopy(points)
            # 신체 기준으로 정규화한 관절 목록에 `normalizer.normalize(points)` 호출 결과를 저장합니다. 대상: `normalized`.
            normalized = normalizer.normalize(points)
            # 검증문 `self.assertEqual(points, original)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
            self.assertEqual(points, original)
            # `matrix` 값에 `np.asarray([[p['x'], p['y']] for p in normalized])` 호출 결과를 저장합니다. 대상: `matrix`.
            matrix = np.asarray([[p['x'], p['y']] for p in normalized])
            # 측정값에 `engine._compute_side_value('left', {p['id']: p for p in normalized})` 호출 결과를 저장합니다. 대상: `value`.
            value = engine._compute_side_value('left', {p['id']: p for p in normalized})
            # 조건 `reference is None`을 만족하는 경우에만 아래 분기를 실행합니다.
            if reference is None:
                # `reference` 값에 순서·구성을 지정한 `(matrix, value)`을 저장합니다. 대상: `reference`.
                reference = matrix, value
            # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
            else:
                # 검증문 `np.testing.assert_allclose(matrix, reference[0], atol=0.0002)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                np.testing.assert_allclose(matrix, reference[0], atol=0.0002)
                # 검증문 `self.assertAlmostEqual(value, reference[1], delta=0.03)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                self.assertAlmostEqual(value, reference[1], delta=0.03)

    # 회귀 테스트: `test_portrait_preview_keeps_proportions_and_moves_skeleton_with_image` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_portrait_preview_keeps_proportions_and_moves_skeleton_with_image(self):
        # OpenCV BGR 영상 프레임에 `np.full((1920, 1080, 3), 255, np.uint8)` 호출 결과를 저장합니다. 대상: `frame`.
        frame = np.full((1920, 1080, 3), 255, np.uint8)
        # 검증문 `self.assertEqual(resize_within(frame, 400, 300).shape[:2], (300, 169))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(resize_within(frame, 400, 300).shape[:2], (300, 169))
        # 계산에 사용할 관절 또는 좌표 묶음에 순서·구성을 지정한 `[dict(id=5, x=0, y=0, score=1)]`을 저장합니다. 대상: `points`.
        points = [dict(id=5, x=0, y=0, score=1)]
        # `(canvas, remapped)` 값에 `letterbox_preview(frame, points)` 호출 결과를 저장합니다. 대상: `(canvas, remapped)`.
        canvas, remapped = letterbox_preview(frame, points)
        # 검증문 `self.assertEqual(canvas.shape, (480, 640, 3))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(canvas.shape, (480, 640, 3))
        # 검증문 `self.assertEqual(int(canvas[:, :185].max()), 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(int(canvas[:, :185].max()), 0)
        # 검증문 `self.assertEqual(int(canvas[:, 185:455].min()), 255)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(int(canvas[:, 185:455].min()), 255)
        # 검증문 `self.assertAlmostEqual(remapped[0]['x'] * 640, 185)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertAlmostEqual(remapped[0]['x'] * 640, 185)
        # 검증문 `self.assertEqual(points[0]['x'], 0)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
        self.assertEqual(points[0]['x'], 0)

    # 회귀 테스트: `test_npz_retains_aspect_ratio_needed_to_reconstruct_measurements` 시나리오의 조건을 구성하고 아래 단언문으로 기대 동작을 확인합니다.
    def test_npz_retains_aspect_ratio_needed_to_reconstruct_measurements(self):
        # `tempfile.TemporaryDirectory()`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
        with tempfile.TemporaryDirectory() as folder:
            # `manager` 값에 `DataManager('portrait', log_root=folder)` 호출 결과를 저장합니다. 대상: `manager`.
            manager = DataManager('portrait', log_root=folder)
            # `info` 값에 `manager.begin_session('shoulder_press', 'MAIN')` 호출 결과를 저장합니다. 대상: `info`.
            info = manager.begin_session('shoulder_press', 'MAIN')
            # 로그에 현재 세션의 CSV·NPZ·MP4 기록 객체(`manager.session_log`)을 저장합니다. 대상: `log`.
            log = manager.session_log
            # 좌우 가동 범위 시작값·목표값에 반복·조건식 `{side: {'start_val': 45, 'target_val': 160} for side in ('left', 'right')}`으로 구성한 값을 저장합니다. 대상: `thresholds`.
            thresholds = {side: {'start_val': 45, 'target_val': 160} for side in ('left', 'right')}
            # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
            try:
                # `log.append_frame(0, 0, pose(1080, 1920), {}, {}, 'ANGLE', thresholds)`을 호출하여 해당 단계의 작업을 수행합니다.
                log.append_frame(0, 0, pose(1080, 1920), {}, {}, 'ANGLE', thresholds)
                # `manager.finish_session([])`을 호출하여 해당 단계의 작업을 수행합니다.
                manager.finish_session([])
                # `np.load(info['skeleton_npz'])`의 컨텍스트 안에서 처리하여 파일·잠금·연결·테스트 패치의 진입과 정리가 짝을 이루게 합니다.
                with np.load(info['skeleton_npz']) as data:
                    # 검증문 `self.assertAlmostEqual(float(data['image_aspect_ratio'][0]), 1080 / 1920)`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertAlmostEqual(float(data['image_aspect_ratio'][0]), 1080 / 1920)
                    # 검증문 `self.assertEqual(data['keypoints'].shape, (1, 17, 3))`으로 실제 결과가 기대 조건과 일치하는지 확인합니다. 다르면 테스트가 실패합니다.
                    self.assertEqual(data['keypoints'].shape, (1, 17, 3))
            # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
            finally:
                # `manager.finish_session([])`을 호출하여 해당 단계의 작업을 수행합니다.
                manager.finish_session([])


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `unittest.main()`을 호출하여 해당 단계의 작업을 수행합니다.
    unittest.main()
