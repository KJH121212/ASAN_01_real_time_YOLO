# 한국어 코드 해설: 세로·가로 영상의 비율을 유지하면서 표시 크기를 계산하고 영상 메타데이터를 관절에 연결합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Aspect-preserving preview geometry; measurement coordinates remain untouched."""

# 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
import cv2
# 이 파일에서 사용할 `numpy`(`np`) 모듈을 불러옵니다.
import numpy as np


# `resize_within` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def resize_within(frame, width, height):
    # `(source_height, source_width)` 값에 계산식 `frame.shape[:2]`의 결과를 저장합니다. 대상: `(source_height, source_width)`.
    source_height, source_width = frame.shape[:2]
    # `scale` 값에 `min(width / source_width, height / source_height)` 호출 결과를 저장합니다. 대상: `scale`.
    scale = min(width / source_width, height / source_height)
    # 크기에 순서·구성을 지정한 `(max(1, round(source_width * scale)), max(1, round(source_height * scale)))`을 저장합니다. 대상: `size`.
    size = (max(1, round(source_width * scale)), max(1, round(source_height * scale)))
    # 현재 함수 처리를 끝내고 `cv2.resize(frame, size)` 호출 결과를 호출한 쪽으로 반환합니다.
    return cv2.resize(frame, size)


# `letterbox_preview` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def letterbox_preview(frame, keypoints, width=640, height=480):
    # `resized` 값에 `resize_within(frame, width, height)` 호출 결과를 저장합니다. 대상: `resized`.
    resized = resize_within(frame, width, height)
    # `(resized_height, resized_width)` 값에 계산식 `resized.shape[:2]`의 결과를 저장합니다. 대상: `(resized_height, resized_width)`.
    resized_height, resized_width = resized.shape[:2]
    # `(left, top)` 값에 순서·구성을 지정한 `((width - resized_width) // 2, (height - resized_height) // 2)`을 저장합니다. 대상: `(left, top)`.
    left, top = (width - resized_width) // 2, (height - resized_height) // 2
    # `canvas` 값에 `np.zeros((height, width, 3), dtype=frame.dtype)` 호출 결과를 저장합니다. 대상: `canvas`.
    canvas = np.zeros((height, width, 3), dtype=frame.dtype)
    # `canvas[top:top + resized_height, left:left + resized_width]` 값에 `resized` 값(`resized`)을 저장합니다. 대상: `canvas[top:top + resized_height, left:left + resized_width]`.
    canvas[top:top + resized_height, left:left + resized_width] = resized
    # 계산에 사용할 관절 또는 좌표 묶음에 반복·조건식 `[{**point, 'x': (point['x'] * resized_width + left) / width, 'y': (point['y'] * resized_height + top) / height} for point in keypoints]`으로 구성한 값을 저장합니다. 대상: `points`.
    points = [{**point, 'x': (point['x'] * resized_width + left) / width,
               # 구성 항목: `y` 값(`'y'`)에 계산식 `(point['y'] * resized_height + top) / height`의 결과를 지정합니다.
               'y': (point['y'] * resized_height + top) / height} for point in keypoints]
    # 현재 함수 처리를 끝내고 순서·구성을 지정한 `(canvas, points)`을 호출한 쪽으로 반환합니다.
    return canvas, points
