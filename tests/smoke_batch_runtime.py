# 한국어 코드 해설: `smoke_batch_runtime`에서 사용하는 실행·분석 보조 기능을 정의합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Manual smoke test with local YOLO/RTMPose weights and a real video."""

# 이 파일에서 사용할 `argparse` 모듈을 불러옵니다.
import argparse
# `pathlib`에서 `Path` 기능을 가져옵니다.
from pathlib import Path
# 이 파일에서 사용할 `sys` 모듈을 불러옵니다.
import sys
# 이 파일에서 사용할 `time` 모듈을 불러옵니다.
import time

# 최상위 경로에 계산식 `Path(__file__).resolve().parents[1]`의 결과를 저장합니다. 대상: `ROOT`.
ROOT = Path(__file__).resolve().parents[1]
# `sys.path.insert(0, str(ROOT))`을 호출하여 해당 단계의 작업을 수행합니다.
sys.path.insert(0, str(ROOT))


# 진입점 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def main():
    # `parser` 값에 `argparse.ArgumentParser(description=__doc__)` 호출 결과를 저장합니다. 대상: `parser`.
    parser = argparse.ArgumentParser(description=__doc__)
    # `parser.add_argument('--video', default=str(ROOT / 'data/test/biceps_curl/patient_1/calibration.mp4'))`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--video', default=str(ROOT / 'data/test/biceps_curl/patient_1/calibration.mp4'))
    # `parser.add_argument('--device', default=None)`을 호출하여 해당 단계의 작업을 수행합니다.
    parser.add_argument('--device', default=None)
    # `args` 값에 `parser.parse_args()` 호출 결과를 저장합니다. 대상: `args`.
    args = parser.parse_args()

    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print('Loading YOLO and RTMPose...', flush=True)
    # 이 파일에서 사용할 `cv2` 모듈을 불러옵니다.
    import cv2
    # `core.skeleton_engine`에서 `SkeletonEngine` 기능을 가져옵니다.
    from core.skeleton_engine import SkeletonEngine
    # 생성 또는 검증할 분석 엔진에 `SkeletonEngine(device=args.device)` 호출 결과를 저장합니다. 대상: `engine`.
    engine = SkeletonEngine(device=args.device)
    # OpenCV 영상 입력 장치에 `cv2.VideoCapture(args.video)` 호출 결과를 저장합니다. 대상: `cap`.
    cap = cv2.VideoCapture(args.video)
    # 영상 프레임 묶음에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `frames`.
    frames = []
    # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
    try:
        # `range(3)`에서 항목을 차례로 꺼내 `_`에 넣고 아래 처리를 반복합니다.
        for _ in range(3):
            # `(ok, frame)` 값에 `cap.read()` 호출 결과를 저장합니다. 대상: `(ok, frame)`.
            ok, frame = cap.read()
            # 조건 `not ok`을 만족하는 경우에만 아래 분기를 실행합니다.
            if not ok:
                # 이 조건에서는 처리를 계속할 수 없으므로 `RuntimeError('Smoke test needs a readable video with at least 3 frames')` 예외를 발생시킵니다.
                raise RuntimeError('Smoke test needs a readable video with at least 3 frames')
            # `frames.append(frame)`을 호출하여 항목 하나를 끝에 추가합니다.
            frames.append(frame)
    # 성공·실패와 관계없이 아래 자원 해제 및 임시 상태 정리를 실행합니다.
    finally:
        # `cap.release()`을 호출하여 영상 입력·출력 자원을 해제합니다.
        cap.release()

    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print('Testing a two-frame batch, a missing frame, and a one-frame tail...', flush=True)
    # 동작 시작 시각 또는 실행 여부에 고해상도 단조 시각(초)(`time.perf_counter()`)을 저장합니다. 대상: `started`.
    started = time.perf_counter()
    # `results` 값에 `engine.extract_keypoints_batch([None, frames[0], frames[1]])` 호출 결과를 저장합니다. 대상: `results`.
    results = engine.extract_keypoints_batch([None, frames[0], frames[1]])
    # `tail` 값에 `engine.extract_keypoints_batch([frames[2]])` 호출 결과를 저장합니다. 대상: `tail`.
    tail = engine.extract_keypoints_batch([frames[2]])
    # `len(results) == 3 and results[0] is None and (len(tail) == 1)`이 참인지 확인하고, 거짓이면 단언 오류로 기대 조건 위반을 알립니다.
    assert len(results) == 3 and results[0] is None and len(tail) == 1
    # `found` 값에 반복·조건식 `[pose for pose in results + tail if pose is not None]`으로 구성한 값을 저장합니다. 대상: `found`.
    found = [pose for pose in results + tail if pose is not None]
    # `found`이 참인지 확인하고, 거짓이면 단언 오류로 기대 조건 위반을 알립니다.
    assert found, 'No people detected in the first three frames; choose another video'
    # `all((len(pose) == 17 for pose in found))`이 참인지 확인하고, 거짓이면 단언 오류로 기대 조건 위반을 알립니다.
    assert all(len(pose) == 17 for pose in found)
    # 진행 상태나 진단 결과를 콘솔에 출력합니다. 출력값은 아래 인수와 문자열 형식을 따릅니다.
    print(f'BATCH_SMOKE_OK device={engine.device} poses={len(found)} '
          # 구성 항목: `flush` 값(`flush`)에 True을 지정합니다.
          f'seconds={time.perf_counter() - started:.3f}', flush=True)


# 조건 `__name__ == '__main__'`을 만족하는 경우에만 아래 분기를 실행합니다.
if __name__ == '__main__':
    # `main()`을 호출하여 해당 단계의 작업을 수행합니다.
    main()
