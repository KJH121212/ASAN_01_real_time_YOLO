"""실험 전용 영상 출력·파일 분석. 이 파일을 삭제해도 수치 서버는 동작합니다."""

import asyncio
import base64
import json
import math
import time

import cv2
import numpy as np


from utils.pose_overlay import motion_progress, render_overlay

PREVIEW_SIZE = (640, 360)
PREVIEW_FPS = 10
PREVIEW_JPEG_QUALITY = 60


def encode_preview(frame, keypoints, pose, exercise_name, elapsed_sec):
    """Resize a separate preview before drawing; the captured frame stays full size."""
    height, width = frame.shape[:2]
    scale = min(PREVIEW_SIZE[0] / width, PREVIEW_SIZE[1] / height, 1.)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))),
                           interpolation=cv2.INTER_AREA)
    image = render_overlay(frame, keypoints, pose, exercise_name, elapsed_sec)
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_JPEG_QUALITY])
    if not ok:
        raise RuntimeError('미리보기 JPEG 인코딩에 실패했습니다.')
    return {
        'type': 'VIDEO_FRAME', 'encoding': 'jpeg',
        'image_base64': base64.b64encode(encoded).decode('ascii'),
        'width': image.shape[1], 'height': image.shape[0],
        'timestamp': elapsed_sec,
    }


async def send_preview(server, websocket, frame, keypoints, pose, elapsed_sec):
    """Send small previews at most 10 FPS, independently of recording resolution."""
    if server.output_mode != 'VISUAL':
        return
    now = time.perf_counter()
    if now - server._last_preview_at < 1 / PREVIEW_FPS:
        return
    server._last_preview_at = now
    exercise = server.controller.motion_engine.exercise_name
    packet = await asyncio.to_thread(encode_preview, frame, keypoints, pose, exercise, elapsed_sec)
    await websocket.send(json.dumps(packet))


async def _poll_video_stop(server, websocket):
    """비디오 분석 중 수신되는 중단 명령 확인"""
    if not hasattr(websocket, "recv"):
        return None
    try:
        packet = json.loads(await asyncio.wait_for(websocket.recv(), timeout=0.001))
    except asyncio.TimeoutError:
        return None

    if packet.get("type") == "CMD_STOP_SESSION":
        session_id = server.controller.data_manager.session_log.session_id
        if packet.get("session_id") in (None, session_id):
            return packet

    await websocket.send(json.dumps({
        "type": "ERROR",
        "command": packet.get("type"),
        "request_id": packet.get("request_id"),
        "code": "VIDEO_COMMAND_REJECTED",
        "message": "Only a stop command for the active session is supported during video analysis",
    }))
    return None


async def analyze_video(server, websocket, data: dict):
    video_path = data.get("video_path") or data.get("input_source")
    exercise_name = data.get("exercise_name", "biceps_curl")
    mode = data.get("mode", "CALIBRATION")
    batch_size = int(data.get("batch_size", 16))

    if batch_size <= 0:
        await websocket.send(json.dumps({
            "type": "ERROR",
            "code": "INVALID_BATCH_SIZE",
            "message": "batch_size must be greater than zero",
        }))
        return

    print(f"\n[MODE 3: VIDEO_BATCH] Processing '{video_path}' (Batch: {batch_size}, Mode: {mode})")
    await websocket.send(json.dumps({
        "type": "VIDEO_STATUS",
        "status": "PROCESSING",
        "message": "비디오를 고속 배치 분석 중입니다. 잠시만 기다려주세요...",
    }))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        await websocket.send(json.dumps({"type": "ERROR", "message": f"비디오 파일을 열 수 없습니다: {video_path}"}))
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_duration = 1.0 / fps

    try:
        data_manager = server._create_session(data, video_fps=fps)
    except Exception:
        cap.release()
        raise

    await websocket.send(json.dumps({"type": "SESSION_LOG_STARTED", **data_manager.session_log.info}))
    all_extracted_keypoints = []
    batch_frames = []
    decoded_frames = 0
    stop_request = None
    t_start = time.time()
    last_progress = time.perf_counter()

    from network.camera import downscale_frame_if_needed

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if batch_frames:
                    kpts_batch = await asyncio.to_thread(server.skeleton_engine.extract_keypoints_batch, batch_frames)
                    all_extracted_keypoints.extend(kpts_batch)
                break

            frame = downscale_frame_if_needed(frame, 1280, 720)
            data_manager.session_log.write_video_frame(frame, decoded_frames * frame_duration)
            decoded_frames += 1
            batch_frames.append(frame)

            if len(batch_frames) >= batch_size:
                kpts_batch = await asyncio.to_thread(server.skeleton_engine.extract_keypoints_batch, batch_frames)
                all_extracted_keypoints.extend(kpts_batch)
                batch_frames.clear()

                stop_request = await _poll_video_stop(server, websocket)
                if stop_request:
                    break

                if time.perf_counter() - last_progress >= 1.0:
                    await websocket.send(json.dumps({
                        "type": "VIDEO_PROGRESS",
                        "phase": "EXTRACTING",
                        "processed_frames": decoded_frames,
                        "total_frames": total_frames,
                    }))
                    last_progress = time.perf_counter()
    finally:
        cap.release()

    print(f"[MODE 3: VIDEO_BATCH] Extracted {len(all_extracted_keypoints)} frames in {time.time() - t_start:.2f}s")

    # Step 2: 누락 없는 관절 좌표 및 측정값 버퍼 적재 (count는 수행하지 않고 전체 프레임 적재)
    server.controller.timer_start = 0.0

    for frame_idx, raw_kpts in enumerate(all_extracted_keypoints):
        if stop_request:
            break

        virtual_time = frame_idx * frame_duration
        filtered_kpts = server.filter_engine.update(raw_kpts, timestamp=virtual_time)
        is_occluded = server.controller.check_occlusion(filtered_kpts) if filtered_kpts else True

        # 정규화 좌표 계산 및 프레임별 측정값(val) 도출 (카운트는 건너뜀)
        normalized = server.controller.measurement_points(filtered_kpts) if filtered_kpts and not is_occluded else None
        points = {p['id']: p for p in normalized} if normalized else {}
        val_left = server.controller.motion_engine._compute_side_value('left', points)
        val_right = server.controller.motion_engine._compute_side_value('right', points)

        left_data = {
            "val": val_left,
            "rep_count": 0,
            "total_score": 0.0,
            "score_delta": 0.0,
            "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "EXTRACTED",
            "is_count_updated": False,
        }
        right_data = {
            "val": val_right,
            "rep_count": 0,
            "total_score": 0.0,
            "score_delta": 0.0,
            "score": 0,
            "progress_ratio": 0.0,
            "state": "READY",
            "quality": "EXTRACTED",
            "is_count_updated": False,
        }
        server.controller.last_left_data = left_data
        server.controller.last_right_data = right_data

        server._append_to_buffers(
            left_data,
            right_data,
            raw_kpts,
            elapsed_sec=virtual_time,
            frame_idx=frame_idx,
            is_occluded=is_occluded,
            video_frame_idx=frame_idx if data_manager.session_log.record_video else -1,
        )

        stop_request = await _poll_video_stop(server, websocket)

        if frame_idx % 30 == 0 or frame_idx == len(all_extracted_keypoints) - 1:
            await websocket.send(json.dumps({
                'type': 'VIDEO_PROGRESS', 'phase': 'ANALYZING',
                'processed_frames': frame_idx + 1,
                'total_frames': len(all_extracted_keypoints),
            }))

    # Step 3: 파일 저장 및 완료 결과 일괄 전송 (카운트 없이 NPZ 저장 완결)
    if stop_request:
        server._record_client_result(stop_request)

    status = 'COMPLETED'
    if stop_request:
        status = 'COMPLETED' if stop_request.get('reason') == 'TARGET_REACHED' else 'STOPPED'
    summary_meta = server._save_session_files(status=status)

    payload = {
        "type": "SESSION_FINISHED" if stop_request else "VIDEO_ANALYSIS_COMPLETED",
        "session_id": data_manager.session_log.session_id,
        "request_id": stop_request.get("request_id") if stop_request else None,
        "reason": stop_request.get("reason", "USER_STOP") if stop_request else "END_OF_VIDEO",
        "mode": mode,
        "video_path": video_path,
        "total_frames_analyzed": len(all_extracted_keypoints),
        "source_duration_sec": decoded_frames * frame_duration,
        "summary": summary_meta,
    }
    if stop_request:
        payload["status"] = "COMPLETED" if stop_request.get("reason") == "TARGET_REACHED" else "STOPPED"

    await websocket.send(json.dumps(payload))
    npz_path = summary_meta.get('skeleton_npz') if summary_meta else 'NPZ'
    print(f"[MODE 3: VIDEO_BATCH] Completed successfully. Saved NPZ: {npz_path}")
    server.controller = None
