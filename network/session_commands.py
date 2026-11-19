"""GPU 모델이나 카메라를 켜지 않고도 검증·테스트할 수 있는 세션 명령 처리 모음."""

import math
from importlib.util import find_spec

# 운영 배포에서는 False로 바꾸거나 experimental_modes.py를 삭제합니다.
ENABLE_EXPERIMENTAL_MODES = True


def available_output_modes():
    """현재 서버에서 사용 가능한 출력 모드 목록을 반환한다."""
    if ENABLE_EXPERIMENTAL_MODES and find_spec('network.experimental_modes') is not None:
        return ('VISUAL', 'LIGHTWEIGHT', 'OFFLINE_VIDEO')
    return ('LIGHTWEIGHT',)


DEFAULT_PREPARATION_SECONDS = 5.0


def resolve_output_mode(packet):
    """클라이언트 요청 패킷에서 출력 모드(VISUAL / LIGHTWEIGHT / OFFLINE_VIDEO)를 결정한다."""
    output = str(packet.get('output_mode', 'LIGHTWEIGHT')).upper()
    if output not in {'VISUAL', 'LIGHTWEIGHT', 'OFFLINE_VIDEO'}:
        raise ValueError('output_mode must be VISUAL, LIGHTWEIGHT or OFFLINE_VIDEO')
    video = packet.get('video_path') or packet.get('input_source')
    if video:
        if output == 'VISUAL':
            raise ValueError('영상 파일은 OFFLINE_VIDEO 모드를 사용하세요.')
        output = 'OFFLINE_VIDEO'
    elif output == 'OFFLINE_VIDEO':
        raise ValueError('OFFLINE_VIDEO 모드에는 video_path가 필요합니다.')
    if output not in available_output_modes():
        raise ValueError('실험용 영상 모드가 꺼져 있습니다. LIGHTWEIGHT를 사용하세요.')
    return output


def get_target_score(packet):
    """목표 점수(target_score) 값을 꺼내 양의 유한수인지 검사한다."""
    value = packet.get('target_score', packet.get('target_reps', 10))
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError('target_score must be a finite positive number')
    return float(value)


def get_target_reps(packet):
    """목표 반복 횟수(target_reps)가 주어졌다면 양의 정수인지 검사한다."""
    value = packet.get('target_reps')
    if value is not None and (type(value) is not int or value <= 0):
        raise ValueError('target_reps must be a positive integer')
    return value


def get_preparation_seconds(packet):
    """측정 시작 전 준비 시간(초)을 꺼내 0 이상 유한수인지 검사한다."""
    value = packet.get('preparation_seconds', DEFAULT_PREPARATION_SECONDS)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('preparation_seconds must be a finite nonnegative number')
    return float(value)


COUNT_THRESHOLD_COMMANDS = {"CMD_GET_COUNT_THRESHOLD", "CMD_UPDATE_COUNT_THRESHOLD"}


def handle_count_threshold_command(controller, packet: dict) -> dict:
    """실행 중인 세션의 카운트 기준값(임계값)을 조회하거나 수정한다."""
    request_id = packet.get("request_id")

    def error(code, message):
        return {"type": "ERROR", "request_id": request_id,
                "command": packet.get("type"), "code": code, "message": message}

    command = packet.get("type")
    if command not in COUNT_THRESHOLD_COMMANDS:
        return error("UNKNOWN_COMMAND", "Unsupported count threshold command")
    if controller is None or controller.motion_engine is None:
        return error("NO_ACTIVE_SESSION", "Start a session before accessing count thresholds")
    engine = controller.motion_engine
    if packet.get("exercise_name") != engine.exercise_name:
        return error("EXERCISE_MISMATCH", "exercise_name must match the active session")

    changed_sides = []
    if command == "CMD_UPDATE_COUNT_THRESHOLD":
        if engine.mode != "MAIN":
            return error("INVALID_MODE", "Count threshold updates require MAIN mode")
        try:
            changed_sides = controller.update_count_thresholds(packet.get("thresholds"))
        except ValueError as exc:
            return error("INVALID_THRESHOLD", str(exc))

    return {
        "type": "COUNT_THRESHOLD_UPDATED" if command == "CMD_UPDATE_COUNT_THRESHOLD" else "COUNT_THRESHOLD_STATUS",
        "request_id": request_id,
        **engine.count_threshold_status(),
        "changed_sides": changed_sides,
        "states": {"left": engine.fsm_left.state, "right": engine.fsm_right.state},
        "scope": "SESSION",
    }
