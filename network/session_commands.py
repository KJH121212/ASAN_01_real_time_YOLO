"""Session commands that can be tested without loading GPU models or a camera."""

import math


# Set to 0 to disable preparation globally; each start request may override it.
DEFAULT_PREPARATION_SECONDS = 5.0


def get_target_score(packet):
    value = packet.get('target_score', packet.get('target_reps', 10))
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value <= 0):
        raise ValueError('target_score must be a finite positive number')
    return float(value)


def get_preparation_seconds(packet):
    value = packet.get('preparation_seconds', DEFAULT_PREPARATION_SECONDS)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0):
        raise ValueError('preparation_seconds must be a finite nonnegative number')
    return float(value)


COUNT_THRESHOLD_COMMANDS = {"CMD_GET_COUNT_THRESHOLD", "CMD_UPDATE_COUNT_THRESHOLD"}


def handle_count_threshold_command(controller, packet: dict) -> dict:
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
        if engine.mode not in {"MAIN", "TEST"}:
            return error("INVALID_MODE", "Count threshold updates require MAIN or TEST mode")
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
