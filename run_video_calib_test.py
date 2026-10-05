import os
import sys
from pathlib import Path
import cv2
import numpy as np

# 프로젝트 루트 경로 등록
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.skeleton_engine import SkeletonEngine
from core.session_controller import SessionController
from core.data_manager import DataManager, load_exercise_config
from core.motion_engine import MotionEngine
from utils.filters import RealtimeEMAFilter
from utils.overlay_renderer import OverlayRenderer


def draw_hud_and_bar(canvas: np.ndarray, val_l: float, val_r: float, fsm_l, fsm_r, mode: str, step: str, reps: int, target_reps: int):
    """
    영상 우측에 실시간 관절 Y값과 ROM 달성률 게이지 바를 시각화합니다.
    """
    h, w = canvas.shape[:2]

    # 상단 HUD 정보 박스
    cv2.rectangle(canvas, (15, 15), (340, 115), (20, 20, 20), -1)
    cv2.rectangle(canvas, (15, 15), (340, 115), (100, 100, 100), 1)

    title_str = f"MODE : {mode}"
    step_str = f"STEP : {step}" if mode == "CALIBRATION" else f"L_Q: {fsm_l.last_quality} | R_Q: {fsm_r.last_quality}"
    cv2.putText(canvas, title_str, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(canvas, f"REPS : {reps} / {target_reps}", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(canvas, step_str, (25, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # 우측 수직 프로그레스 바 설정
    bar_w = 28
    bar_top = int(h * 0.25)
    bar_bottom = int(h * 0.75)
    bar_h = bar_bottom - bar_top

    # (1) Left Bar
    l_bar_x = w - 90
    _render_single_bar(canvas, l_bar_x, bar_top, bar_w, bar_h, val_l, fsm_l, "L")

    # (2) Right Bar
    r_bar_x = w - 45
    _render_single_bar(canvas, r_bar_x, bar_top, bar_w, bar_h, val_r, fsm_r, "R")


def _render_single_bar(canvas: np.ndarray, x: int, y_top: int, w: int, h: int, val: float, fsm, label: str):
    y_bottom = y_top + h

    cv2.rectangle(canvas, (x, y_top), (x + w, y_bottom), (40, 40, 40), -1)
    cv2.rectangle(canvas, (x, y_top), (x + w, y_bottom), (150, 150, 150), 1)

    val_range = abs(fsm.target_val - fsm.start_val)
    if val is not None and val_range > 0:
        ratio = float(np.clip(abs(val - fsm.start_val) / val_range, 0.0, 1.0))
    else:
        ratio = 0.0

    fill_h = int(h * ratio)
    fill_y = y_bottom - fill_h

    # 상태별 색상 (PEAK: 녹색, PUSHING: 주황색, READY: 회색)
    if fsm.peak_reached:
        bar_color = (0, 255, 0)
    elif fsm.state == "PUSHING":
        bar_color = (0, 165, 255)
    else:
        bar_color = (180, 180, 180)

    if fill_h > 0:
        cv2.rectangle(canvas, (x, fill_y), (x + w, y_bottom), bar_color, -1)

    cv2.putText(canvas, label, (x + 7, y_top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    val_str = f"{val:.2f}" if val is not None else "N/A"
    cv2.putText(canvas, val_str, (x - 8, y_bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(canvas, f"{int(ratio * 100)}%", (x - 4, y_bottom + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


def run_stage_with_video(
    input_video_path: Path,
    output_video_path: Path,
    exercise_name: str,
    player_id: str,
    mode: str,
    target_reps: int,
    skeleton_engine: SkeletonEngine,
    custom_thresholds: dict = None
) -> dict:
    print("\n" + "=" * 70)
    print(f"[{mode} 세션 가동] 영상: {input_video_path.name} | 목표: {target_reps}회")
    print(f"  - 적용 임계값: {custom_thresholds if custom_thresholds else '기본 설정값 사용'}")
    print("=" * 70)

    data_manager = DataManager(player_id=player_id, patient_name="Tester")
    exercise_config = load_exercise_config(exercise_name)

    filter_engine = RealtimeEMAFilter(max_jump=0.35, alpha=0.85)
    renderer = OverlayRenderer(conf_threshold=0.35)

    motion_engine = MotionEngine(
        config=exercise_config,
        custom_thresholds=custom_thresholds,
        mode=mode,
        target_reps=target_reps
    )
    controller = SessionController(data_manager=data_manager, motion_engine=motion_engine)

    # 상체 전용 가림 조건 완화 패치 (상체 8개 관절만 신뢰도 검증)
    def upper_body_check(keypoints: list) -> bool:
        if not keypoints or len(keypoints) < 17:
            return True
        kpt_map = {kp["id"]: kp.get("score", 0.0) for kp in keypoints}
        return not all(kpt_map.get(i, 0.0) >= 0.35 for i in [5, 6, 7, 8, 9, 10, 11, 12])
    controller.check_occlusion = upper_body_check

    cap = cv2.VideoCapture(str(input_video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    frame_idx = 0
    is_finished = False
    rep_count = 0
    final_thresholds = None

    while cap.isOpened():
        ret, frame = cap.read()

        # 목표 횟수를 다 채우기 전에 영상이 끝나면 루프 재생
        if not ret or frame is None:
            if not is_finished:
                print(f"  -> [Loop] {input_video_path.name} 되감기 실행 (현재 달성: {rep_count}/{target_reps})")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            else:
                break

        if not ret or frame is None:
            break

        frame_idx += 1

        # 1. 포즈 추론 및 스무딩 필터
        raw_kpts = skeleton_engine.extract_keypoints(frame)
        filtered_kpts = filter_engine.update(raw_kpts)
        is_detected = filtered_kpts is not None and len(filtered_kpts) > 0

        # 2. 가림 판별 및 모드별 프레임 평가
        is_occluded = controller.check_occlusion(filtered_kpts) if is_detected else True

        step_name = ""
        if mode == "CALIBRATION":
            cal_res = controller.process_calibration_frame(filtered_kpts, is_occluded)
            step_name = cal_res["calib_step"]
            rep_count = cal_res["calib_rep_count"]
            if step_name == "FINISHED":
                is_finished = True
        else:  # TEST 모드
            l_data, r_data, finished_flag = controller.process_main_frame(filtered_kpts, is_occluded)
            rep_count = max(l_data["rep_count"], r_data["rep_count"])
            step_name = f"RUNNING ({rep_count}/{target_reps})"
            if finished_flag:
                is_finished = True

        # 3. Y값 추출
        val_l, val_r = None, None
        if is_detected:
            norm_kpts = controller.normalizer.normalize(filtered_kpts)
            if norm_kpts:
                kpt_map = {kp["id"]: kp for kp in norm_kpts}
                val_l = motion_engine._compute_side_value("left", kpt_map)
                val_r = motion_engine._compute_side_value("right", kpt_map)

        # 4. 스켈레톤 및 게이지 바 렌더링
        vis_frame = renderer.draw_skeleton(
            canvas=frame.copy(),
            keypoints=filtered_kpts if is_detected else [],
            is_occluded=is_occluded
        )
        draw_hud_and_bar(
            canvas=vis_frame,
            val_l=val_l,
            val_r=val_r,
            fsm_l=motion_engine.fsm_left,
            fsm_r=motion_engine.fsm_right,
            mode=mode,
            step=step_name,
            reps=rep_count,
            target_reps=target_reps
        )
        out.write(vis_frame)

        if frame_idx % 30 == 0:
            print(f"  -> [{mode}] {frame_idx:04d}F | Reps: {rep_count}/{target_reps} | Status: {step_name}")

        # 목표 완수 시 루프 종료
        if is_finished:
            print(f"\n[SUCCESS] {mode} 세션 목표 달성 완료! ({rep_count}/{target_reps}회)")
            if mode == "CALIBRATION":
                final_thresholds = controller.compute_and_save_thresholds(exercise_name)
            break

    cap.release()
    out.release()
    print(f"[저장 완료] -> {output_video_path.resolve()}\n")

    return final_thresholds


def main():
    player_id = "patient_1"
    exercise_name = "biceps_curl"
    data_dir = project_root / "data" / "test" / "patient_1"

    calib_video = data_dir / "calibration.mp4"
    # test.mp4가 있으면 test.mp4를 사용하고, 없으면 test_5times_1.mp4 사용
    test_video = data_dir / "test.mp4"
    if not test_video.exists():
        test_video = data_dir / "test_5times_1.mp4"

    calib_out = data_dir / "output_calib_visualized.mp4"
    test_out = data_dir / "output_test_visualized.mp4"

    # 무거운 AI 모델(YOLO/RTMPose)은 최초 1회만 초기화하여 공유
    print("[INIT] YOLO & RTMPose 관절 추론 엔진 로드 중...")
    skeleton_engine = SkeletonEngine(yolo_interval=3)

    # ---------------------------------------------------------
    # 1단계: CALIBRATION 실행 (3회 달성 후 맞춤 threshold 계산 및 저장)
    # ---------------------------------------------------------
    calib_thresholds = run_stage_with_video(
        input_video_path=calib_video,
        output_video_path=calib_out,
        exercise_name=exercise_name,
        player_id=player_id,
        mode="CALIBRATION",
        target_reps=3,
        skeleton_engine=skeleton_engine,
        custom_thresholds=None
    )

    print("=" * 70)
    print(">>> 1단계 캘리브레이션 산출 임계값 결과 <<<")
    print(calib_thresholds)
    print("=" * 70)

    # ---------------------------------------------------------
    # 2단계: TEST 실행 (1단계에서 산출된 calib_thresholds 즉시 주입, 5회 평가)
    # ---------------------------------------------------------
    run_stage_with_video(
        input_video_path=test_video,
        output_video_path=test_out,
        exercise_name=exercise_name,
        player_id=player_id,
        mode="TEST",
        target_reps=5,
        skeleton_engine=skeleton_engine,
        custom_thresholds=calib_thresholds
    )

    print("\n" + "=" * 70)
    print("[전체 파이프라인 완료]")
    print(f"1. 캘리브레이션 결과 영상: {calib_out}")
    print(f"2. 본 운동(TEST) 결과 영상: {test_out}")
    print(f"3. 환자 프로필 JSON 파일  : {project_root / 'data' / player_id / f'{player_id}.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()