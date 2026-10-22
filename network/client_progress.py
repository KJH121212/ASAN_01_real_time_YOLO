# 한국어 코드 해설: 서버의 반복 이벤트를 중복 없이 게임 점수로 바꾸고 클라이언트가 종료를 요청할 시점을 판단합니다.
# 아래 설명은 실행식 앞에 붙인 주석입니다. 문자열·정규식·프로토콜 값과 계산 순서는 원본을 유지합니다.
# 다음 문자열은 모듈·클래스·함수의 설명 또는 문자열 리터럴입니다. 문자열 내용은 주석 추가로 변경하지 않습니다.
"""Client game rules, independent of the measurement server and UI reruns."""

# 이 파일에서 사용할 `math` 모듈을 불러옵니다.
import math


# `ClientProgress` 클래스: 관련 상태와 처리 함수를 하나의 객체에 묶습니다.
class ClientProgress:
    # 생성자: 전달받은 설정을 보관하고 이 객체가 사용할 상태·누적값·자원을 초기화합니다.
    def __init__(self, config):
        # 해당 기능을 사용할지 여부에 계산식 `config.get('mode') in {'MAIN', 'TEST'}`의 결과를 저장합니다. 대상: `self.enabled`.
        self.enabled = config.get('mode') in {'MAIN', 'TEST'}
        # 목표에 `float(config.get('target_score', config.get('target_reps', 40)))` 호출 결과를 저장합니다. 대상: `self.target`.
        self.target = float(config.get('target_score', config.get('target_reps', 40)))
        # 배치 영상 분석은 실제 재생 속도가 아니므로 목표를 초과하면 지연 없이 중지를 요청합니다.
        # 종료 요청 전에 기다리는 시간(초)에 `config.get('video_path')`이면 `0.0`, 아니면 `float(config.get('finish_delay_seconds', 1.5))`을 저장합니다. 대상: `self.delay`.
        self.delay = 0.0 if config.get('video_path') else float(config.get('finish_delay_seconds', 1.5))
        # 계산에 사용할 관절 또는 좌표 묶음에 `dict(config.get('quality_points', {'PERFECT': 1.0, 'GOOD': 0.5, 'BAD': 0.25}))` 호출 결과를 저장합니다. 대상: `self.points`.
        self.points = dict(config.get('quality_points', {'PERFECT': 1., 'GOOD': .5, 'BAD': .25}))
        # 조건 `not math.isfinite(self.target) or self.target <= 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(self.target) or self.target <= 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Client target must be finite and positive')` 예외를 발생시킵니다.
            raise ValueError('Client target must be finite and positive')
        # 조건 `not math.isfinite(self.delay) or self.delay < 0`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not math.isfinite(self.delay) or self.delay < 0:
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Client finish delay must be finite and nonnegative')` 예외를 발생시킵니다.
            raise ValueError('Client finish delay must be finite and nonnegative')
        # 조건 `set(self.points) != {'PERFECT', 'GOOD', 'BAD'} or any((isinstance(v, bool) or not isinstance(v, (int, float)) or (not math.isfinite(v)) or (v < 0) for v in self.points.v…`을 만족하는 경우에만 아래 분기를 실행합니다.
        if set(self.points) != {'PERFECT', 'GOOD', 'BAD'} or any(
                # 앞줄에서 시작한 식·인수·자료 구조를 `isinstance(v, bool) or not isinstance(v, (int, float))` 내용으로 이어서 구성합니다.
                isinstance(v, bool) or not isinstance(v, (int, float))
                # 앞줄에서 시작한 식·인수·자료 구조를 `or not math.isfinite(v) or v < 0 for v in self.points.values()):` 내용으로 이어서 구성합니다.
                or not math.isfinite(v) or v < 0 for v in self.points.values()):
            # 이 조건에서는 처리를 계속할 수 없으므로 `ValueError('Client quality points must be finite nonnegative numbers')` 예외를 발생시킵니다.
            raise ValueError('Client quality points must be finite nonnegative numbers')
        # 현재 측정 세션 식별자에 None을 저장합니다. 대상: `self.session_id`.
        self.session_id = None
        # 이미 점수에 반영한 반복 이벤트 식별자 집합에 `set()` 호출 결과를 저장합니다. 대상: `self.seen`.
        self.seen = set()
        # 관절 신뢰도 배열 또는 좌우 게임 점수(사용 위치에 따름)에 키 'left', 'right'로 구성한 사전을 저장합니다. 대상: `self.scores`.
        self.scores = {'left': 0., 'right': 0.}
        # 클라이언트가 반영한 좌우 반복 횟수에 키 'left', 'right'로 구성한 사전을 저장합니다. 대상: `self.repetitions`.
        self.repetitions = {'left': 0, 'right': 0}
        # 게임 목표를 초과한 최초 시각에 None을 저장합니다. 대상: `self.goal_at`.
        self.goal_at = None

    # bind 측정 세션 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def bind_session(self, session_id):
        # 조건 `session_id`을 만족하는 경우에만 아래 분기를 실행합니다.
        if session_id:
            # 현재 측정 세션 식별자에 현재 측정 세션 식별자(`session_id`)을 저장합니다. 대상: `self.session_id`.
            self.session_id = session_id

    # 세션·좌우·반복 번호·등급을 검증하고 중복 이벤트를 제외한 뒤 클라이언트 게임 점수를 반영합니다.
    def accept(self, packet, now):
        # 조건 `not self.enabled or packet.get('mode') == 'CALIBRATION'`을 만족하는 경우에만 아래 분기를 실행합니다.
        if not self.enabled or packet.get('mode') == 'CALIBRATION':
            # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
            return False
        # 측정 세션에 `packet.get('session_id')` 호출 결과를 저장합니다. 대상: `session`.
        session = packet.get('session_id')
        # `(side, rep_id, quality)` 값에 반복·조건식 `(packet.get(key) for key in ('side', 'rep_id', 'quality'))`으로 구성한 값을 저장합니다. 대상: `(side, rep_id, quality)`.
        side, rep_id, quality = (packet.get(key) for key in ('side', 'rep_id', 'quality'))
        # 조건 `not isinstance(session, str) or not session or side not in self.scores or (type(rep_id) is not int) or (rep_id <= 0) or (quality not in self.points)`을 만족하는 경우에만 아래 분기를 실행합니다.
        if (not isinstance(session, str) or not session or side not in self.scores
                # 앞줄에서 시작한 식·인수·자료 구조를 `or type(rep_id) is not int or rep_id <= 0 or quality not in self.points):` 내용으로 이어서 구성합니다.
                or type(rep_id) is not int or rep_id <= 0 or quality not in self.points):
            # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
            return False
        # 조건 `self.session_id is not None and session != self.session_id`을 만족하는 경우에만 아래 분기를 실행합니다.
        if self.session_id is not None and session != self.session_id:
            # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
            return False
        # `self.bind_session(session)`을 호출하여 해당 단계의 작업을 수행합니다.
        self.bind_session(session)
        # 세션 ID·좌우·반복 번호를 묶어 이벤트 식별자로 사용합니다. 좌우 반복 번호가 같아도 서로 다른 이벤트입니다.
        key = (session, side, rep_id)
        # 조건 `key in self.seen`을 만족하는 경우에만 아래 분기를 실행합니다.
        if key in self.seen:
            # 현재 함수 처리를 끝내고 False을 호출한 쪽으로 반환합니다.
            return False
        # `self.seen.add(key)`을 호출하여 집합에 항목을 등록합니다.
        self.seen.add(key)
        # `self.scores[side]` 값에 `self.points[quality]`을 더해 누적합니다. 대상: `self.scores[side]`.
        self.scores[side] += self.points[quality]
        # `self.repetitions[side]` 값에 `1`을 더해 누적합니다. 대상: `self.repetitions[side]`.
        self.repetitions[side] += 1
        # 목표를 초과해야 합니다. 목표 40점에서 정확히 40점에 도달한 것만으로 종료하지 않습니다.
        # 현재 게임 규칙은 목표 이상이 아니라 목표 초과입니다. 예를 들어 목표 40점에서 정확히 40점이면 계속하고 초과할 때 종료 대기를 시작합니다.
        if self.goal_at is None and self.scores[side] > self.target:
            # 게임 목표를 초과한 최초 시각에 이번 처리에 사용할 현재 시각(초)(`now`)을 저장합니다. 대상: `self.goal_at`.
            self.goal_at = now
        # 현재 함수 처리를 끝내고 True을 호출한 쪽으로 반환합니다.
        return True

    # 목표 초과가 발생한 뒤 설정된 종료 지연 시간이 지났는지 확인합니다.
    def finish_due(self, now):
        # 현재 함수 처리를 끝내고 계산식 `self.enabled and self.goal_at is not None and (now - self.goal_at >= self.delay)`의 결과를 호출한 쪽으로 반환합니다.
        return self.enabled and self.goal_at is not None and now - self.goal_at >= self.delay

    # `snapshot` 값 처리 함수입니다. 입력을 아래 순서로 처리하고 반환문 또는 객체 상태로 결과를 전달합니다.
    def snapshot(self):
        # 현재 함수 처리를 끝내고 키 'scores', 'repetitions', 'target_score', 'goal_reached'로 구성한 사전을 호출한 쪽으로 반환합니다.
        return {'scores': dict(self.scores), 'repetitions': dict(self.repetitions),
                # 구성 항목: 게임 목표 점수(`'target_score'`)에 목표(`self.target`); 목표 reached(`'goal_reached'`)에 계산식 `self.goal_at is not None`의 결과를 지정합니다.
                'target_score': self.target, 'goal_reached': self.goal_at is not None}
