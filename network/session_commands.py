# 한국어 코드 해설: 모델이나 카메라를 열지 않고 준비 시간·목표값·실시간 가동 범위 수정 명령을 검증합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Session commands that can be tested without loading GPU models or a camera."""

# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math


# 기본값을 0으로 바꾸면 준비 대기를 끕니다. 개별 시작 요청에서도 시간을 덮어쓸 수 있습니다.
# 기본 preparation 초에 5.0을 저장합니다. 대상: `DEFAULT_PREPARATION_SECONDS`.
DEFAULT_PREPARATION_SECONDS = 5.0


# 조회 목표 점수 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def get_target_score(packet):
    # 측정값에 `packet.get('target_score', packet.get('target_reps', 10))` 호출 결과를 저장합니다. 대상: `value`.
    value = packet.get('target_score', packet.get('target_reps', 10))
    # 조건 `isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value <= 0)`을 만족하는 경우에만 아래 분기를 실행합니다.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            # 앞줄에서 시작한 식·인수·자료 구조를 `or not math.isfinite(value) or value <= 0):` 내용으로 이어서 구성합니다.
            or not math.isfinite(value) or value <= 0):
        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('target_score must be a finite positive number')` 예외를 발생시킵니다.
        raise ValueError('target_score must be a finite positive number')
    # 현재 함수 처리를 끝내고 `float(value)` 호출 결과를 호출한 쪽으로 반환합니다.
    return float(value)


# 조회 preparation 초 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def get_preparation_seconds(packet):
    # 측정값에 `packet.get('preparation_seconds', DEFAULT_PREPARATION_SECONDS)` 호출 결과를 저장합니다. 대상: `value`.
    value = packet.get('preparation_seconds', DEFAULT_PREPARATION_SECONDS)
    # 조건 `isinstance(value, bool) or not isinstance(value, (int, float)) or (not math.isfinite(value)) or (value < 0)`을 만족하는 경우에만 아래 분기를 실행합니다.
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            # 앞줄에서 시작한 식·인수·자료 구조를 `or not math.isfinite(value) or value < 0):` 내용으로 이어서 구성합니다.
            or not math.isfinite(value) or value < 0):
        # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('preparation_seconds must be a finite nonnegative number')` 예외를 발생시킵니다.
        raise ValueError('preparation_seconds must be a finite nonnegative number')
    # 현재 함수 처리를 끝내고 `float(value)` 호출 결과를 호출한 쪽으로 반환합니다.
    return float(value)


# 횟수 가동 범위 기준 명령에 순서·구성을 지정한 `{'CMD_GET_COUNT_THRESHOLD', 'CMD_UPDATE_COUNT_THRESHOLD'}`을 저장합니다. 대상: `COUNT_THRESHOLD_COMMANDS`.
COUNT_THRESHOLD_COMMANDS = {"CMD_GET_COUNT_THRESHOLD", "CMD_UPDATE_COUNT_THRESHOLD"}


# 요청 처리 횟수 가동 범위 기준 명령 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
def handle_count_threshold_command(controller, packet: dict) -> dict:
    # 명령과 응답을 연결하는 요청 식별자에 `packet.get('request_id')` 호출 결과를 저장합니다. 대상: `request_id`.
    request_id = packet.get("request_id")

    # 오류 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def error(code, message):
        # 현재 함수 처리를 끝내고 키 'type', 'request_id', 'command', 'code', 'message'로 구성한 사전을 호출한 쪽으로 반환합니다.
        return {"type": "ERROR", "request_id": request_id,
                # 구성 항목: 명령(`'command'`)에 `packet.get('type')` 호출 결과; `code` 값(`'code'`)에 `code` 값(`code`); 메시지(`'message'`)에 메시지(`message`)을 지정합니다.
                "command": packet.get("type"), "code": code, "message": message}

    # 명령에 `packet.get('type')` 호출 결과를 저장합니다. 대상: `command`.
    command = packet.get("type")
    # 조건 `command not in COUNT_THRESHOLD_COMMANDS`을 만족하는 경우에만 아래 분기를 실행합니다.
    if command not in COUNT_THRESHOLD_COMMANDS:
        # 현재 함수 처리를 끝내고 `error('UNKNOWN_COMMAND', 'Unsupported count threshold command')` 호출 결과를 호출한 쪽으로 반환합니다.
        return error("UNKNOWN_COMMAND", "Unsupported count threshold command")
    # 조건 `controller is None or controller.motion_engine is None`을 만족하는 경우에만 아래 분기를 실행합니다.
    if controller is None or controller.motion_engine is None:
        # 현재 함수 처리를 끝내고 `error('NO_ACTIVE_SESSION', 'Start a session before accessing count thresholds')` 호출 결과를 호출한 쪽으로 반환합니다.
        return error("NO_ACTIVE_SESSION", "Start a session before accessing count thresholds")
    # 생성 또는 검증할 분석 엔진에 관절 측정값 계산 및 반복 판정 엔진(`controller.motion_engine`)을 저장합니다. 대상: `engine`.
    engine = controller.motion_engine
    # 조건 `packet.get('exercise_name') != engine.exercise_name`을 만족하는 경우에만 아래 분기를 실행합니다.
    if packet.get("exercise_name") != engine.exercise_name:
        # 현재 함수 처리를 끝내고 `error('EXERCISE_MISMATCH', 'exercise_name must match the active session')` 호출 결과를 호출한 쪽으로 반환합니다.
        return error("EXERCISE_MISMATCH", "exercise_name must match the active session")

    # 이번 명령으로 가동 범위가 바뀐 좌우 목록에 순서·구성을 지정한 `[]`을 저장합니다. 대상: `changed_sides`.
    changed_sides = []
    # 조건 `command == 'CMD_UPDATE_COUNT_THRESHOLD'`을 만족하는 경우에만 아래 분기를 실행합니다.
    if command == "CMD_UPDATE_COUNT_THRESHOLD":
        # 조건 `engine.mode not in {'MAIN', 'TEST'}`을 만족하는 경우에만 아래 분기를 실행합니다.
        if engine.mode not in {"MAIN", "TEST"}:
            # 현재 함수 처리를 끝내고 `error('INVALID_MODE', 'Count threshold updates require MAIN or TEST mode')` 호출 결과를 호출한 쪽으로 반환합니다.
            return error("INVALID_MODE", "Count threshold updates require MAIN or TEST mode")
        # 아래 작업을 시도하고 예외 또는 종료 시 뒤의 except·finally 정리 규칙을 적용합니다.
        try:
            # 이번 명령으로 가동 범위가 바뀐 좌우 목록에 `controller.update_count_thresholds(packet.get('thresholds'))` 호출 결과를 저장합니다. 대상: `changed_sides`.
            changed_sides = controller.update_count_thresholds(packet.get("thresholds"))
        # `ValueError` 오류를 받아 아래 복구·오류 안내 처리를 실행합니다.
        except ValueError as exc:
            # 현재 함수 처리를 끝내고 `error('INVALID_THRESHOLD', str(exc))` 호출 결과를 호출한 쪽으로 반환합니다.
            return error("INVALID_THRESHOLD", str(exc))

    # 현재 함수 처리를 끝내고 키 'type', 'request_id', 'changed_sides', 'states', 'scope'로 구성한 사전을 호출한 쪽으로 반환합니다.
    return {
        # 구성 항목: `type` 값(`'type'`)에 `command == 'CMD_UPDATE_COUNT_THRESHOLD'`이면 `'COUNT_THRESHOLD_UPDATED'`, 아니면 `'COUNT_THRESHOLD_STATUS'`을 지정합니다.
        "type": "COUNT_THRESHOLD_UPDATED" if command == "CMD_UPDATE_COUNT_THRESHOLD" else "COUNT_THRESHOLD_STATUS",
        # 구성 항목: 명령과 응답을 연결하는 요청 식별자(`'request_id'`)에 명령과 응답을 연결하는 요청 식별자(`request_id`)을 지정합니다.
        "request_id": request_id,
        # 앞줄에서 시작한 식·인수·자료 구조를 `**engine.count_threshold_status(),` 내용으로 이어서 구성합니다.
        **engine.count_threshold_status(),
        # 구성 항목: 이번 명령으로 가동 범위가 바뀐 좌우 목록(`'changed_sides'`)에 이번 명령으로 가동 범위가 바뀐 좌우 목록(`changed_sides`)을 지정합니다.
        "changed_sides": changed_sides,
        # 구성 항목: `states` 값(`'states'`)에 키 'left', 'right'로 구성한 사전; 왼쪽(`'left'`)에 처리 단계(`engine.fsm_left.state`); 오른쪽(`'right'`)에 처리 단계(`engine.fsm_right.state`)을 지정합니다.
        "states": {"left": engine.fsm_left.state, "right": engine.fsm_right.state},
        # 구성 항목: `scope` 값(`'scope'`)에 'SESSION'을 지정합니다.
        "scope": "SESSION",
    # 앞에서 여러 줄로 작성한 호출·자료 구조·표현식의 괄호를 닫아 구성을 마칩니다.
    }
