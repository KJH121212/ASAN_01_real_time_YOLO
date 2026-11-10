"""Keep camera preview independent of inference, calibration and session completion."""

import asyncio
import json
import time
import math

from websockets.exceptions import ConnectionClosed

from network.experimental_modes import encode_preview, PREVIEW_FPS
from network.session_commands import available_output_modes


async def watch_camera(server, websocket, packet):
    if 'VISUAL' not in available_output_modes():
        await websocket.send(json.dumps({'type': 'PREVIEW_STATUS', 'warning': '영상 미리보기가 비활성화되어 있습니다.'}))
        return
    camera_index = int(packet.get('camera_index', 0))
    max_width = int(packet.get('camera_max_width', 1280))
    max_height = int(packet.get('camera_max_height', 720))
    server._preview_watchers += 1
    try:
        if server.controller and server._camera_index not in (None, camera_index):
            await websocket.send(json.dumps({'type': 'PREVIEW_STATUS', 'warning': '측정 중인 카메라와 번호가 다릅니다.'}))
            return
        while True:
            started = time.perf_counter()
            camera_failed = True
            try:
                await asyncio.to_thread(server.start_camera, camera_index, max_width, max_height)
                ok, frame, captured_at, _ = server.camera.read_sample()
                if not ok:
                    raise RuntimeError('카메라에서 영상을 받지 못하고 있습니다. 연결을 확인해 주세요.')
                if time.perf_counter()-captured_at > 2.:
                    raise RuntimeError('카메라 영상 갱신이 중단되었습니다. 연결을 복구하고 있습니다.')
                camera_failed = False
                pose = dict(server._preview_pose)
                points = server._preview_keypoints
                warning, notice = '', ''
                hide_measurement = False
                model = server.model_loader.snapshot()
                if not model['ready']:
                    hide_measurement = True
                    notice = model.get('message', '모델을 준비하고 있습니다. 카메라 영상은 계속 표시됩니다.')
                    if model['model_state'] == 'FAILED':
                        warning = '모델 준비에 실패했습니다. 영상은 계속 표시됩니다. 서버 상태를 확인해 주세요.'
                elif server.controller is None:
                    hide_measurement = True
                    notice = '측정이 진행 중이 아닙니다. 현재 카메라 영상만 표시합니다.'
                elif server._measurement_ready_at and time.perf_counter() < server._measurement_ready_at:
                    hide_measurement = True
                    notice = '측정 준비 중입니다. 기본 자세를 취해 주세요.'
                elif time.perf_counter()-server._preview_updated_at > 1.:
                    hide_measurement = True
                    warning = '측정 결과를 기다리고 있습니다. 카메라를 향해 자세를 유지해 주세요.'
                    points = []
                elif pose.get('is_occluded') or missing_joints(server, points):
                    warning = '관절이 보이지 않습니다. 운동에 필요한 관절이 화면에 모두 보이도록 자세를 조정해 주세요.'
                if server.controller is None or not model['ready']:
                    points = []
                if hide_measurement:
                    points = []
                    pose = {**pose, 'is_occluded': False,
                            **{side: {**pose.get(side, {}), 'val': None, 'progress_ratio': None}
                               for side in ('left', 'right')}}
                controller = server.controller
                exercise = controller.motion_engine.exercise_name if controller else packet.get('exercise_name', '')
                elapsed = max(0., captured_at-getattr(server, '_session_started', captured_at))
                result = await asyncio.to_thread(encode_preview, frame, points, pose, exercise, elapsed)
                result.update(warning=warning, notice=notice)
                await websocket.send(json.dumps(result))
            except ConnectionClosed:
                raise
            except Exception as exc:
                await websocket.send(json.dumps({'type': 'PREVIEW_STATUS', 'warning': str(exc)}))
                # Reopen a failed camera on the next attempt, retaining the last client image.
                if camera_failed:
                    with server._camera_lock:
                        if server.camera:
                            server.camera.stop()
                            server.camera = None
                            server._camera_index = None
                await asyncio.sleep(.5)
            await asyncio.sleep(max(0., 1/PREVIEW_FPS-(time.perf_counter()-started)))
    finally:
        server._preview_watchers -= 1
        if server.controller is None:
            server.stop_camera()


def missing_joints(server, keypoints):
    if server.controller is None:
        return False
    engine = server.controller.motion_engine
    config = engine.config['eval_config']
    required = {5, 6, 11, 12} if engine.use_normalization else set()
    for field in ('target_kpt', 'reference_kpt', 'primary_kpts'):
        for ids in config.get(field, {}).values():
            required.update(ids if isinstance(ids, list) else [ids])
    points = {point['id']: point for point in keypoints or []}
    return any(j not in points or points[j].get('score', 0) < .35 or
               any(not math.isfinite(points[j].get(axis, math.nan)) or not 0 <= points[j][axis] <= 1
                   for axis in ('x', 'y')) for j in required)
