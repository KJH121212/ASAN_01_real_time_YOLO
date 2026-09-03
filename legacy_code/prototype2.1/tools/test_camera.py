import os
import sys
import cv2

# prototype2.1 경로 우선 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
proto_dir = os.path.dirname(current_dir)
if proto_dir not in sys.path:
    sys.path.insert(0, proto_dir)

from core.skeleton_engine import SkeletonEngine


def run_camera_test(camera_index: int = 0):
    print("[TestCamera] SkeletonEngine 로드 중...")
    engine = SkeletonEngine(yolo_interval=1)
    
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"[Error] 카메라({camera_index}번)를 열 수 없습니다.")
        return

    print("[TestCamera] 디버그 창 실행 (종료하려면 이미지 창 선택 후 'q' 누름)")
    
    # COCO Keypoint 연결 라인 (어깨, 팔, 골반 등)
    skeleton_links = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # 상체 및 팔
        (5, 11), (6, 12), (11, 12)                # 몸통
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[Error] 프레임을 읽어올 수 없습니다.")
            break

        keypoints = engine.extract_keypoints(frame)

        if keypoints:
            kpt_dict = {kp["id"]: (int(kp["x"]), int(kp["y"]), kp.get("score", 0.0)) for kp in keypoints}

            # 1. 관절 연결 선 그리기
            for p1_id, p2_id in skeleton_links:
                if p1_id in kpt_dict and p2_id in kpt_dict:
                    pt1, pt2 = kpt_dict[p1_id], kpt_dict[p2_id]
                    if pt1[2] >= 0.4 and pt2[2] >= 0.4:
                        cv2.line(frame, (pt1[0], pt1[1]), (pt2[0], pt2[1]), (255, 200, 0), 2)

            # 2. 관절 포인트 그리기
            for kp_id, (x, y, score) in kpt_dict.items():
                if score >= 0.4:
                    cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        cv2.imshow("Camera & Skeleton Debug (Press 'q' to Quit)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_test(camera_index=0)