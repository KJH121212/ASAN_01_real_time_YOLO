"""Client game rules, independent of the measurement server and UI reruns."""

import math


class ClientProgress:
    def __init__(self, config):
        self.enabled = config.get('mode') in {'MAIN', 'TEST'}
        self.target = float(config.get('target_score', config.get('target_reps', 40)))
        # Batch replay isn't wall-clock playback; request a stop as soon as the goal is exceeded.
        self.delay = 0.0 if config.get('video_path') else float(config.get('finish_delay_seconds', 1.5))
        self.points = dict(config.get('quality_points', {'PERFECT': 1., 'GOOD': .5, 'BAD': .25}))
        if not math.isfinite(self.target) or self.target <= 0:
            raise ValueError('Client target must be finite and positive')
        if not math.isfinite(self.delay) or self.delay < 0:
            raise ValueError('Client finish delay must be finite and nonnegative')
        if set(self.points) != {'PERFECT', 'GOOD', 'BAD'} or any(
                isinstance(v, bool) or not isinstance(v, (int, float))
                or not math.isfinite(v) or v < 0 for v in self.points.values()):
            raise ValueError('Client quality points must be finite nonnegative numbers')
        self.session_id = None
        self.seen = set()
        self.scores = {'left': 0., 'right': 0.}
        self.repetitions = {'left': 0, 'right': 0}
        self.goal_at = None

    def bind_session(self, session_id):
        if session_id:
            self.session_id = session_id

    def accept(self, packet, now):
        if not self.enabled or packet.get('mode') == 'CALIBRATION':
            return False
        session = packet.get('session_id')
        side, rep_id, quality = (packet.get(key) for key in ('side', 'rep_id', 'quality'))
        if (not isinstance(session, str) or not session or side not in self.scores
                or type(rep_id) is not int or rep_id <= 0 or quality not in self.points):
            return False
        if self.session_id is not None and session != self.session_id:
            return False
        self.bind_session(session)
        key = (session, side, rep_id)
        if key in self.seen:
            return False
        self.seen.add(key)
        self.scores[side] += self.points[quality]
        self.repetitions[side] += 1
        # Strictly exceed the goal: exactly 40 points does not end a 40-point game.
        if self.goal_at is None and self.scores[side] > self.target:
            self.goal_at = now
        return True

    def finish_due(self, now):
        return self.enabled and self.goal_at is not None and now - self.goal_at >= self.delay

    def snapshot(self):
        return {'scores': dict(self.scores), 'repetitions': dict(self.repetitions),
                'target_score': self.target, 'goal_reached': self.goal_at is not None}
