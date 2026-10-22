# 한국어 코드 해설: 골반 중심을 원점으로 옮기고 몸통 길이를 1로 맞춥니다. 영상 종횡비를 보정한 뒤 필요하면 y축 부호를 뒤집습니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# ==============================================================================
# [모듈 정보]
# 파일: utils/normalization.py
# 설명: 골반 중심을 원점으로 하고 몸통 길이를 1로 맞추는 등방성 정규화입니다.
# 기본 화면 좌표는 아래쪽이 +Y이며 invert_y=True이면 위쪽이 +Y가 됩니다.
# ==============================================================================

# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np


# `PoseNormalizer` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class PoseNormalizer:
    # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
    """
    골반 중심을 (0, 0)으로 설정하고 몸통(어깨-골반 거리) 길이를 1.0으로 스케일링하는
    등방성(Isotropic) 2D 포즈 정규화 클래스.
    화면 좌표계(위쪽 0, 아래쪽 +Y)를 그대로 유지하여 팔을 위로 올릴 시 Y값이 감소합니다.
    """

    # COCO 관절 인덱스 정의
    # 왼쪽 shoulder에 5을 저장합니다. 대상: `LEFT_SHOULDER`.
    LEFT_SHOULDER = 5
    # 오른쪽 shoulder에 6을 저장합니다. 대상: `RIGHT_SHOULDER`.
    RIGHT_SHOULDER = 6
    # 왼쪽 hip에 11을 저장합니다. 대상: `LEFT_HIP`.
    LEFT_HIP = 11
    # 오른쪽 hip에 12을 저장합니다. 대상: `RIGHT_HIP`.
    RIGHT_HIP = 12

    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, conf_threshold: float = 0.35, invert_y: bool = False):
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """
        Args:
            conf_threshold: 기준 관절(골반/어깨)의 최소 감지 신뢰도 임계값
            invert_y: False일 경우 화면 기본 좌표계 유지 (상향 이동 시 Y값 감소)
        """
        # 관절 검출 결과를 유효하게 사용할 최소 신뢰도에 관절 검출 결과를 유효하게 사용할 최소 신뢰도(`conf_threshold`)을 저장합니다. 대상: `self.conf_threshold`.
        self.conf_threshold = conf_threshold
        # 위쪽 이동을 양수로 만들기 위한 y축 부호 반전 여부에 위쪽 이동을 양수로 만들기 위한 y축 부호 반전 여부(`invert_y`)을 저장합니다. 대상: `self.invert_y`.
        self.invert_y = invert_y
        # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
        print(f"[DEBUG][NORM_INIT] PoseNormalizer configured. ConfThresh: {self.conf_threshold}, InvertY: {self.invert_y}")

    # 관절을 골반 중심 원점·몸통 길이 단위로 변환합니다. 유효한 기준 관절이 없으면 None을 반환합니다.
    def normalize(self, keypoints: list) -> list:
        # 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
        """
        17개 관절 딕셔너리 리스트를 받아 신체 중심 정규화 좌표계로 변환합니다.

        Args:
            keypoints: [{'id': int, 'x': float, 'y': float, 'score': float}, ...]
        Returns:
            list: 정규화된 관절 좌표 딕셔너리 리스트 (변환 불가 시 None 반환)
        """
        # 조건 `not keypoints or len(keypoints) < 17`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not keypoints or len(keypoints) < 17:
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print("[WARN][NORM] Input keypoints list is empty or invalid. Skipping normalization.")
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None

        # 고속 벡터 연산을 위해 NumPy 배열로 변환: Shape (17, 3) -> [x, y, score]
        # `kpt_matrix` 값에 `np.array([[kp['x'], kp['y'], kp.get('score', 0.0)] for kp in keypoints], dtype=np.float32)` 호출 결과를 저장합니다. 대상: `kpt_matrix`.
        kpt_matrix = np.array(
            # 앞줄에서 시작한 식·인수·자료 구조를 `[[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in keypoints],` 내용으로 이어서 구성합니다.
            [[kp["x"], kp["y"], kp.get("score", 0.0)] for kp in keypoints],
            # 구성 항목: `dtype` 값(`dtype`)에 `np.float32` 값(`np.float32`)을 지정합니다.
            dtype=np.float32
        # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
        )
        # 관절의 x·y 좌표 배열에 계산식 `kpt_matrix[:, :2]`의 결과를 저장합니다. 대상: `coords`.
        coords = kpt_matrix[:, :2]
        # 관절 신뢰도 배열 또는 좌우 게임 점수(사용 위치에 따름)에 계산식 `kpt_matrix[:, 2]`의 결과를 저장합니다. 대상: `scores`.
        scores = kpt_matrix[:, 2]
        # 가로·세로가 다른 영상에서 x/w와 y/h는 서로 다른 척도입니다.
        # 관절 각도나 거리를 계산하기 전에 두 축을 영상 높이 기준 단위로 맞춥니다.
        # 영상 비율 정보가 없는 기존 호출은 기본 비율 1로 기존 좌표를 유지합니다.
        # 입력 영상의 가로/세로 비율에 `float(keypoints[0].get('image_aspect_ratio', 1.0))` 호출 결과를 저장합니다. 대상: `aspect_ratio`.
        aspect_ratio = float(keypoints[0].get('image_aspect_ratio', 1.0))
        # 조건 `not np.isfinite(aspect_ratio) or aspect_ratio <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not np.isfinite(aspect_ratio) or aspect_ratio <= 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('image_aspect_ratio must be finite and positive')` 예외를 발생시킵니다.
            raise ValueError('image_aspect_ratio must be finite and positive')
        # x/w와 y/h의 축별 척도가 다르므로 x에 w/h를 곱해 두 축을 영상 높이 단위로 맞춥니다. 이 보정이 없으면 세로 영상의 각도·길이가 왜곡됩니다.
        coords[:, 0] *= aspect_ratio

        # 필수 기준 관절 유효 감지 여부 검증
        # 양쪽 골반이 신뢰도 기준을 만족하는지 여부에 계산식 `scores[self.LEFT_HIP] >= self.conf_threshold and scores[self.RIGHT_HIP] >= self.conf_threshold`의 결과를 저장합니다. 대상: `has_hips`.
        has_hips = (scores[self.LEFT_HIP] >= self.conf_threshold and scores[self.RIGHT_HIP] >= self.conf_threshold)
        # 양쪽 어깨가 신뢰도 기준을 만족하는지 여부에 계산식 `scores[self.LEFT_SHOULDER] >= self.conf_threshold and scores[self.RIGHT_SHOULDER] >= self.conf_threshold`의 결과를 저장합니다. 대상: `has_shoulders`.
        has_shoulders = (scores[self.LEFT_SHOULDER] >= self.conf_threshold and scores[self.RIGHT_SHOULDER] >= self.conf_threshold)

        # 조건 `not has_hips`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not has_hips:
            # 골반이 가려진 경우 정규화 기준 원점을 확정할 수 없으므로 무효 처리
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"[WARN][NORM] Pelvis keypoints occluded. Left Hip Score: {scores[self.LEFT_HIP]:.2f}, Right Hip Score: {scores[self.RIGHT_HIP]:.2f}")
            # 현재 함수 처리를 끝내고 None을 호출한 쪽으로 반환합니다.
            return None

        # 1. 골반 중심 원점(Origin) 계산
        # 좌우 골반 좌표의 평균을 신체 원점으로 삼아 화면 내 위치가 달라져도 같은 자세를 비교할 수 있게 합니다.
        hip_center = (coords[self.LEFT_HIP] + coords[self.RIGHT_HIP]) / 2.0

        # 2. 몸통 길이(Torso Length) 척도 계산
        # 조건 `has_shoulders`을 만족하는 경우에만 아래 분기를 실행합니다.
        if has_shoulders:
            # 좌우 어깨의 중간 좌표에 계산식 `(coords[self.LEFT_SHOULDER] + coords[self.RIGHT_SHOULDER]) / 2.0`의 결과를 저장합니다. 대상: `shoulder_center`.
            shoulder_center = (coords[self.LEFT_SHOULDER] + coords[self.RIGHT_SHOULDER]) / 2.0
            # 어깨 중심과 골반 중심 사이의 2D 거리를 신체 크기 척도로 사용합니다. 이후 이 길이를 1로 맞추므로 카메라 거리의 영향을 줄입니다.
            torso_length = float(np.linalg.norm(shoulder_center - hip_center))
        # 앞의 조건을 만족하지 않은 나머지 경우를 이 분기에서 처리합니다.
        else:
            # 어깨 결측 시 좌우 골반 너비를 기반으로 몸통 길이 대체 추정 (인체 비율 약 1.5배 보정)
            # 좌우 골반 사이의 거리에 `float(np.linalg.norm(coords[self.LEFT_HIP] - coords[self.RIGHT_HIP]))` 호출 결과를 저장합니다. 대상: `hip_width`.
            hip_width = float(np.linalg.norm(coords[self.LEFT_HIP] - coords[self.RIGHT_HIP]))
            # 어깨 중심과 골반 중심 사이의 몸통 길이에 계산식 `hip_width * 1.5`의 결과를 저장합니다. 대상: `torso_length`.
            torso_length = hip_width * 1.5
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print(f"[DEBUG][NORM] Shoulders missing. Estimated torso length from hip width: {torso_length:.4f}")

        # 0으로 나누기 방어
        # 조건 `torso_length < 0.0001`을 만족하는 경우에만 아래 분기를 실행합니다.
        if torso_length < 1e-4:
            # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
            print("[WARN][NORM] Measured scale factor is near zero. Falling back to 1.0.")
            # 어깨 중심과 골반 중심 사이의 몸통 길이에 1.0을 저장합니다. 대상: `torso_length`.
            torso_length = 1.0

        # 3. 원점 평행이동 및 등방성 스케일링 수행
        # 모든 관절에서 골반 중심을 빼고 같은 몸통 길이로 나눕니다. x·y에 같은 배율을 적용하므로 보정된 영상의 각도와 비율을 보존합니다.
        norm_coords = (coords - hip_center) / torso_length

        # 4. Y축 반전 비활성화 (화면 좌표계 유지)
        # 조건 `self.invert_y`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.invert_y:
            # invert_y=True일 때 y 부호를 뒤집습니다. 영상의 아래쪽 양수 좌표가 위쪽 양수 좌표로 바뀌며, 현재 세션 제어기는 이 옵션을 켭니다.
            norm_coords[:, 1] = -norm_coords[:, 1]

        # 5. 기존 파이프라인 호환 딕셔너리 구조체로 복원
        # `normalized_list` 값에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `normalized_list`.
        normalized_list = []
        # `range(17)`에서 항목을 차례로 꺼내 `i`에 넣고 아래 처리를 반복합니다.
        for i in range(17):
            # `normalized_list.append({'id': i, 'x': round(float(norm_coords[i, 0]), 4), 'y': round(float(norm_coords[i, 1]), 4), 'score': round(float(scores[i]), 3)})`을 호출하여 항목 하나를 끝에 추가합니다.
            normalized_list.append({
                # 구성 항목: `id` 값(`'id'`)에 `i` 값(`i`)을 지정합니다.
                "id": i,
                # 구성 항목: `x` 값(`'x'`)에 `round(float(norm_coords[i, 0]), 4)` 호출 결과를 지정합니다.
                "x": round(float(norm_coords[i, 0]), 4),
                # 구성 항목: `y` 값(`'y'`)에 `round(float(norm_coords[i, 1]), 4)` 호출 결과를 지정합니다.
                "y": round(float(norm_coords[i, 1]), 4),
                # 구성 항목: 판정 코드 또는 관절 검출 신뢰도(사용 위치에 따름)(`'score'`)에 `round(float(scores[i]), 3)` 호출 결과를 지정합니다.
                "score": round(float(scores[i]), 3)
            # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
            })

        # 현재 함수 처리를 끝내고 `normalized_list` 값(`normalized_list`)을 호출한 쪽으로 반환합니다.
        return normalized_list
