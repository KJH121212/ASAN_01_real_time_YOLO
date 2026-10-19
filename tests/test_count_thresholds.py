import copy
import json
import unittest

from core.motion_engine import MotionEngine
from core.session_controller import SessionController
from network.session_commands import handle_count_threshold_command
from network.threshold_test_client import make_command


def make_engine(direction="INCREASING", metric="RELATIVE_Y", target_reps=10):
    start, target = (0, 100) if direction == "INCREASING" else (180, 20)
    return MotionEngine({
        "exercise_name": "test_motion",
        "eval_config": {"metric_type": metric, "motion_direction": direction},
        "default_thresholds": {
            side: {"start_val": start, "target_val": target} for side in ("left", "right")
        },
    }, target_reps=target_reps)


def feed(fsm, values):
    return [fsm.update(value) for value in values]


class CountThresholdTests(unittest.TestCase):
    def test_reduced_rom_accepts_previously_insufficient_motion(self):
        engine = make_engine()
        fsm = engine.fsm_left
        feed(fsm, [0, 20, 30, 20, 0])
        self.assertEqual(fsm.rep_count, 0)  # 30 < 40% of the original 100.
        engine.update_count_thresholds({"left": {"target_val": 50}})
        feed(fsm, [0, 20, 30, 20])
        self.assertEqual(fsm.rep_count, 1)  # Now 60%: GOOD.
        self.assertEqual(fsm.last_quality, "GOOD")
        self.assertEqual(fsm.completed_reps_history[-1]['target_val'], 50)
        self.assertEqual(fsm.completed_reps_history[-1]['threshold_version'], 1)

    def test_edit_mid_motion_discards_old_peak_and_waits_for_return(self):
        engine = make_engine()
        fsm = engine.fsm_left
        feed(fsm, [0, 20, 70])
        self.assertEqual(fsm.state, "PUSHING")
        engine.update_count_thresholds({"left": {"target_val": 50}})
        results = feed(fsm, [60, 40, 30, 10])
        self.assertFalse(any(row['is_count_updated'] for row in results))
        self.assertEqual(fsm.state, "READY")
        feed(fsm, [20, 40, 30])
        self.assertEqual(fsm.rep_count, 1)

    def test_completed_reps_quality_history_and_other_side_are_preserved(self):
        engine = make_engine()
        feed(engine.fsm_left, [0, 20, 80, 60])
        feed(engine.fsm_right, [0, 20, 70])
        history = copy.deepcopy(engine.fsm_left.completed_reps_history)
        other_side = copy.deepcopy(engine.fsm_right.__dict__)
        changed = engine.update_count_thresholds({"left": {"target_val": 50}})
        self.assertEqual(changed, ["left"])
        self.assertEqual(engine.fsm_left.rep_count, 1)
        self.assertEqual(engine.fsm_left.last_quality, "PERFECT")
        self.assertEqual(engine.fsm_left.completed_reps_history, history)
        self.assertEqual(engine.fsm_right.__dict__, other_side)
        feed(engine.fsm_left, [0, 20, 40, 30])
        self.assertEqual(engine.fsm_left.rep_count, 2)

    def test_decreasing_direction_reduces_rom_by_raising_target(self):
        engine = make_engine("DECREASING", "ANGLE")
        feed(engine.fsm_left, [180, 150, 135, 150, 180])
        self.assertEqual(engine.fsm_left.rep_count, 0)
        engine.update_count_thresholds({"left": {"target_val": 100}})
        feed(engine.fsm_left, [180, 150, 135, 150])
        self.assertEqual(engine.fsm_left.rep_count, 1)
        self.assertEqual(engine.fsm_left.last_quality, "GOOD")

    def test_invalid_second_side_does_not_change_first_side(self):
        engine = make_engine()
        before = copy.deepcopy(engine.__dict__)
        with self.assertRaises(ValueError):
            engine.update_count_thresholds({"left": {"target_val": 50},
                                            "right": {"target_val": -1}})
        self.assertEqual(engine.get_count_thresholds(), {
            side: {"start_val": 0, "target_val": 100} for side in ("left", "right")})
        self.assertEqual(engine.threshold_version, 0)
        self.assertEqual(engine.fsm_left.__dict__, before['fsm_left'].__dict__)

    def test_invalid_values_shapes_and_direction_are_rejected(self):
        invalid = [None, [], {}, {'both': {'target_val': 50}}, {'left': {}},
                   {'left': {'typo': 50}}, {'left': None}]
        invalid += [{'left': {'target_val': v}} for v in
                    (0, -1, True, '50', None, float('nan'), float('inf'), 10 ** 400)]
        for patch in invalid:
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                make_engine().update_count_thresholds(patch)
        with self.assertRaises(ValueError):
            make_engine('DECREASING').update_count_thresholds({'right': {'target_val': 190}})
        with self.assertRaises(ValueError):
            make_engine(metric='ANGLE').update_count_thresholds({'left': {'target_val': 181}})
        with self.assertRaises(ValueError):
            make_engine(metric='DISTANCE').update_count_thresholds({'left': {'start_val': -1}})

    def test_signed_height_and_partial_updates_are_supported(self):
        engine = make_engine()
        engine.update_count_thresholds({'left': {'start_val': -1, 'target_val': 0}})
        self.assertEqual(engine.get_count_thresholds()['left'], {'start_val': -1, 'target_val': 0})
        engine.update_count_thresholds({'left': {'target_val': -0.5}})
        self.assertEqual(engine.fsm_left.start_val, -1)

    def test_identical_request_does_not_reset_current_motion(self):
        engine = make_engine()
        patch = {'left': {'target_val': 50}}
        engine.update_count_thresholds(patch)
        feed(engine.fsm_left, [0, 20, 40])
        before = copy.deepcopy(engine.fsm_left.__dict__)
        self.assertEqual(engine.update_count_thresholds(patch), [])
        self.assertEqual(engine.threshold_version, 1)
        self.assertEqual(engine.fsm_left.__dict__, before)

    def test_main_side_continues_after_legacy_target_and_update(self):
        engine = make_engine(target_reps=1)
        feed(engine.fsm_left, [0, 20, 80, 60])
        engine.update_count_thresholds({'left': {'target_val': 50}})
        self.assertEqual(engine.fsm_left.state, 'RETURNING')
        feed(engine.fsm_left, [0, 20, 40, 30])
        self.assertEqual(engine.fsm_left.rep_count, 2)

    def test_status_is_a_copy_and_both_sides_share_one_revision(self):
        engine = make_engine()
        engine.update_count_thresholds({side: {'target_val': 50} for side in ('left', 'right')})
        self.assertEqual(engine.threshold_version, 1)
        status = engine.count_threshold_status()
        status['thresholds']['left']['target_val'] = 999
        self.assertEqual(engine.fsm_left.target_val, 50)
        self.assertEqual(engine.fsm_left.threshold_version, engine.fsm_right.threshold_version)


class CountThresholdCommandTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.controller = SessionController(motion_engine=self.engine)

    def command(self, **overrides):
        packet = {'type': 'CMD_UPDATE_COUNT_THRESHOLD', 'exercise_name': 'test_motion',
                  'request_id': 'request-1', 'thresholds': {'left': {'target_val': 50}}}
        packet.update(overrides)
        # Exercise the same JSON conversion performed by Unity/the server.
        return handle_count_threshold_command(self.controller, json.loads(json.dumps(packet)))

    def test_update_acknowledges_request_and_refreshes_cached_state(self):
        feed(self.engine.fsm_left, [0, 20, 70])
        self.controller.last_left_data.update(val=70, state='PUSHING', is_count_updated=True)
        response = self.command()
        self.assertEqual(response['type'], 'COUNT_THRESHOLD_UPDATED')
        self.assertEqual(response['request_id'], 'request-1')
        self.assertEqual(response['threshold_version'], 1)
        self.assertEqual(response['thresholds']['left']['target_val'], 50)
        self.assertEqual(self.controller.last_left_data['state'], 'RETURNING')
        self.assertFalse(self.controller.last_left_data['is_count_updated'])
        self.assertEqual(self.controller.mode, 'MAIN')

    def test_query_does_not_modify_motion(self):
        feed(self.engine.fsm_left, [0, 20, 70])
        response = self.command(type='CMD_GET_COUNT_THRESHOLD')
        self.assertEqual(response['type'], 'COUNT_THRESHOLD_STATUS')
        self.assertEqual(self.engine.fsm_left.state, 'PUSHING')
        self.assertEqual(self.engine.threshold_version, 0)

    def test_errors_leave_session_usable(self):
        self.assertEqual(self.command(exercise_name='another_game')['code'], 'EXERCISE_MISMATCH')
        self.assertEqual(self.command(exercise_name=None)['code'], 'EXERCISE_MISMATCH')
        self.assertEqual(self.command(thresholds={})['code'], 'INVALID_THRESHOLD')
        self.assertEqual(self.command()['type'], 'COUNT_THRESHOLD_UPDATED')

    def test_no_session_and_calibration_are_rejected(self):
        response = handle_count_threshold_command(None, {'type': 'CMD_UPDATE_COUNT_THRESHOLD'})
        self.assertEqual(response['code'], 'NO_ACTIVE_SESSION')
        self.engine.mode = 'CALIBRATION'
        self.assertEqual(self.command()['code'], 'INVALID_MODE')
        self.assertEqual(self.command(type='CMD_GET_COUNT_THRESHOLD')['type'], 'COUNT_THRESHOLD_STATUS')

    def test_console_client_builds_unity_compatible_packets(self):
        response = handle_count_threshold_command(self.controller, make_command('both 0 50', 'test_motion'))
        self.assertEqual(response['changed_sides'], ['left', 'right'])
        self.assertEqual(self.engine.fsm_right.target_val, 50)
        response = handle_count_threshold_command(self.controller, make_command('show', 'test_motion'))
        self.assertEqual(response['type'], 'COUNT_THRESHOLD_STATUS')
        with self.assertRaises(ValueError):
            make_command('left bad 50', 'test_motion')


if __name__ == '__main__':
    unittest.main()
