"""Published-rule checks, separated from measured event/prediction accuracy."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .paper2022.core import Parameters, detect, prepare_ball
from .paper2022.evaluate import match_events
from .paper2022.events import classify
from .paper2022.run import read_status, viewer_labels, tactical
from .paper2022.setpieces import detect as setpieces, triggers
from .paper2022.sequences import native_sequences

EXACT = Parameters(smooth_window=1, smooth_order=0)
SIGNS = np.array([[0, 0], [1, -1], [-1, 1]])
PITCH = (105.0, 68.0)


class PossessionTests(unittest.TestCase):
    def core(self, ball, positions):
        ball = np.column_stack([ball, np.zeros(len(ball))]) if np.ndim(ball) == 1 else np.array(ball, float)
        players = np.broadcast_to(np.array(positions, float), (len(ball), len(positions), 2)).copy()
        return detect(players, np.arange(len(positions)) % 2, ball, np.ones(len(ball), np.int8), np.zeros(len(ball), bool), EXACT)

    def test_published_provider_parameters(self):
        a, bc = Parameters.profile('a'), Parameters.profile('bc')
        self.assertEqual((a.possession_radius, a.duel_radius, a.speed_change, a.displacement), (0.5, 1.0, 1.0, 0.05))
        self.assertEqual((bc.possession_radius, bc.duel_radius, bc.speed_change, bc.displacement), (1, 1, 5, 0.1))
        self.assertEqual((bc.smooth_window, bc.smooth_order, bc.turn_degrees), (7, 2, 10))

    def test_duel_overrides_single_player_proximity(self):
        result = self.core(np.zeros(10), [[0, 0], [0.8, 0]])
        np.testing.assert_array_equal(result['state'], 3)

    def test_straight_flyby_does_not_create_a_gain(self):
        result = self.core(np.arange(20) * 0.2, [[2, 0], [100, 0]])
        self.assertFalse(result['gains'])
        xy = np.tile([[2, 0], [100, 0]], (20, 1, 1))
        passing, _, _ = classify(result, xy, np.tile([0, 1], (20, 1)), np.ones(20),
                                  SIGNS, PITCH, np.zeros(20, bool), [], np.zeros((20, 2), bool))
        self.assertFalse(passing)

    def test_direction_change_confirms_gain_at_zone_entry_not_later_touch(self):
        ball = np.array([[i * 0.2, 0] if i <= 10 else [2, (i - 10) * 0.2] for i in range(21)])
        result = self.core(ball, [[2, 0], [100, 0]])
        gain = result['gains'][0]
        self.assertLess(gain['frame'], 10)
        self.assertTrue(gain['direction_change'])

    def test_speed_change_alone_can_confirm_a_gain(self):
        result = self.core(np.r_[np.arange(11) * 0.4, np.full(10, 4.0)], [[4, 0], [100, 0]])
        self.assertTrue(any(g['speed_change'] > 5 for g in result['gains']))

    def test_player_walking_away_from_static_ball_is_not_a_loss(self):
        ball = np.zeros((10, 2))
        players = np.zeros((10, 2, 2))
        players[:, 1, 0] = 10
        players[3:, 0, 0] = 5
        players[6:, 1, 0] = 0
        result = detect(players, np.array([0, 1]), ball, np.ones(10), np.zeros(10, bool), EXACT)
        self.assertFalse(result['losses'])

    def test_same_player_at_next_control_suppresses_loss(self):
        result = self.core(np.array([0, 0.5, 2, 2, 0.5, 0]), [[0, 0], [100, 0]])
        self.assertFalse(any(e['frame'] == 1 for e in result['losses']))

    def test_different_player_next_control_confirms_loss(self):
        result = self.core(np.array([0, 0.5, 2, 4, 8, 10, 10, 10]), [[0, 0], [10, 0]])
        loss = next(e for e in result['losses'] if e['slot'] == 0)
        self.assertEqual(loss['frame'], 1)
        self.assertEqual(loss['next_control'], 5)

    def test_false_flyby_does_not_terminate_previous_players_control(self):
        outward = np.r_[0.0, 0.2, np.arange(0.8, 8.1, 0.6)]
        returning = np.arange(outward[-1] - 0.6, -5.0, -0.6)
        result = self.core(np.r_[outward, returning], [[0, 0], [4, 0]])
        self.assertFalse(any(g['slot'] == 1 for g in result['gains']))
        self.assertFalse(any(loss['slot'] == 0 and loss['frame'] < len(outward) for loss in result['losses']))
        self.assertTrue(any(r['slot'] == 0 and not r['gain'] and r['verified'] for r in result['intervals']))

    def test_pass_to_stoppage_has_loss_without_another_receiver(self):
        ball = np.column_stack([[0, 0.5, 2, 4, 8, 10, 10, 10], np.zeros(8)])
        players = np.tile([[0, 0], [100, 0]], (8, 1, 1))
        dead = np.array([False] * 6 + [True] * 2)
        result = detect(players, np.array([0, 1]), ball, np.ones(8), dead, EXACT)
        loss = next(e for e in result['losses'] if e['slot'] == 0)
        self.assertEqual(loss['frame'], 1)
        self.assertIsNone(loss['next_control'])

    def test_same_player_restarting_after_dead_ball_cannot_cancel_previous_loss(self):
        ball = np.column_stack([[0, 0.5, 2, 4, 8, 10, 0, 0.5, 2], np.zeros(9)])
        players = np.tile([[0, 0], [100, 0]], (9, 1, 1))
        dead = np.array([False] * 4 + [True] * 2 + [False] * 3)
        result = detect(players, np.array([0, 1]), ball, np.ones(9), dead, EXACT)
        loss = next(e for e in result['losses'] if e['frame'] == 1)
        self.assertIsNone(loss['next_control'])

    def test_smoothing_does_not_fill_missing_ball_or_cross_halves(self):
        raw = np.column_stack([np.arange(20), np.zeros(20)]).astype(float)
        raw[9:11] = np.nan
        half = np.r_[np.ones(10), np.full(10, 2)]
        ball, before, after = prepare_ball(raw, half, Parameters())
        self.assertTrue(np.isnan(ball[9:11]).all())
        self.assertTrue(np.isnan(after[8]).all())
        self.assertTrue(np.isnan(before[11]).all())


class RestartTests(unittest.TestCase):
    def setup_scene(self, positions, ball_slot=0, n=5):
        xy = np.broadcast_to(np.array(positions, float), (n, 4, 2)).copy()
        teams = np.broadcast_to(np.array([0, 0, 1, 1]), (n, 4)).copy()
        half = np.ones(n, np.int8)
        dead = np.array([True, True] + [False] * (n - 2))
        pz = np.zeros((n, 4), bool)
        if ball_slot is not None:
            pz[2:, ball_slot] = True
        losses = [dict(frame=n - 1, slot=ball_slot, half=1)] if ball_slot is not None else []
        core = dict(pz=pz, distances=np.full((n, 4), 10.0), losses=losses, ball=np.tile([50.0, 34.0], (n, 1)))
        if ball_slot is not None:
            core['distances'][2:, ball_slot] = 0
        return xy, teams, half, dead, core

    def check_scene(self, positions, expected):
        xy, teams, half, dead, core = self.setup_scene(positions)
        rows = setpieces(xy, teams, half, SIGNS, PITCH, dead, core)
        self.assertEqual(rows[0]['kind'], expected)

    def test_kickoff_trigger_and_pattern(self):
        self.check_scene([[52, 34], [20, 20], [80, 30], [90, 40]], 'kickoff')

    def test_penalty_configuration(self):
        self.check_scene([[94, 34], [20, 20], [105, 34], [70, 60]], 'penalty')

    def test_corner_precedes_throw_in(self):
        self.check_scene([[104, 1], [20, 20], [105, 34], [70, 60]], 'corner')

    def test_throw_in(self):
        self.check_scene([[60, 0], [20, 20], [105, 34], [70, 60]], 'throw_in')

    def test_goal_kick(self):
        self.check_scene([[0, 34], [20, 20], [90, 34], [70, 60]], 'goal_kick')

    def test_failed_goal_kick_pattern_falls_back_to_free_kick(self):
        self.check_scene([[70, 34], [20, 20], [105, 34], [70, 60]], 'free_kick')

    def test_s1_missing_inbounder_uses_incomplete_corner(self):
        xy, teams, half, dead, core = self.setup_scene([[104, 1], [70, 20], [60, 30], [70, 60]], None)
        xy[1:, 0] = np.nan
        teams[1:, 0] = -1
        row = setpieces(xy, teams, half, SIGNS, PITCH, dead, core)[0]
        self.assertEqual(row['kind'], 'corner')
        self.assertEqual(row['reason'], 's1_incomplete_trigger_missing_executor')

    def test_repeated_kickoff_is_not_an_extra_goal(self):
        xy, teams, half, _, core = self.setup_scene([[52, 34], [20, 20], [80, 30], [90, 40]], n=10)
        dead = np.array([True, True, False, False, False, True, True, False, False, False])
        core['pz'][7:, 0] = True
        rows = setpieces(xy, teams, half, SIGNS, PITCH, dead, core)
        self.assertEqual(rows[0]['kind'], 'incorrect_kickoff')
        self.assertEqual(rows[1]['dead_event'], 'referee_interruption')

    def test_s1_complete_throw_in_precedes_incomplete_corner(self):
        xy, teams, half, dead, core = self.setup_scene([[104, 1], [60, 0], [80, 30], [70, 60]], None)
        xy[1:, 0] = [70, 20]
        row = setpieces(xy, teams, half, SIGNS, PITCH, dead, core)[0]
        self.assertEqual(row['kind'], 'throw_in')
        self.assertEqual(row['reason'], 's1_trigger_only_uncertain_pattern')


class GoalkeeperTests(unittest.TestCase):
    def scene(self, dead_type=None, release=None, shooter=(90, 34), dead_before=False):
        n = 100
        half = np.ones(n, np.int8)
        dead = np.zeros(n, bool)
        restarts = []
        if dead_type:
            a, b = (15, 24) if dead_before else (60, 70)
            dead[a:b] = True
            restarts = [dict(frame=b, dead_start=a, half=1, slot=1, team=1,
                             kind='goal_kick', dead_event=dead_type)]
        xy = np.tile(np.array([shooter, [103, 34], [100, 30], [70, 60]], float), (n, 1, 1))
        teams = np.tile([0, 1, 0, 1], (n, 1))
        gains = [dict(frame=1, slot=0, team=0, half=1, end=11, duel=False, kind='gain'),
                 dict(frame=25, slot=1, team=1, half=1, end=50, duel=False, kind='gain')]
        losses = [dict(frame=10, slot=0, team=0, half=1, kind='loss', duel=False)]
        if release is not None:
            losses.append(dict(frame=release, slot=1, team=1, half=1, kind='loss', duel=False))
        core = dict(gains=gains, losses=losses, ball=np.tile(shooter, (n, 1)).astype(float),
                    outgoing=np.tile([10.0, 0.0], (n, 1)))
        keeper = np.zeros((n, 4), bool)
        keeper[:, 1] = True
        return classify(core, xy, teams, half, SIGNS, PITCH, dead, restarts, keeper)

    def test_figure6b_goal_kick_rejects_keeper_gain(self):
        passing, receiving, rejected = self.scene('out_for_goal_kick')
        self.assertEqual(passing[0]['event'], 'shot_off_target')
        self.assertIn((25, 1), rejected)

    def test_figure6a_stoppage_before_keeper_is_not_a_save(self):
        passing, receiving, _ = self.scene('out_for_goal_kick', dead_before=True)
        self.assertEqual(passing[0]['event'], 'shot_off_target')
        self.assertFalse(next(e['event'] for e in receiving if e['slot'] == 1).startswith('save'))

    def test_figure6c_corner_is_save_deflect(self):
        _, receiving, _ = self.scene('out_for_corner')
        self.assertEqual(next(e['event'] for e in receiving if e['slot'] == 1), 'save_deflect')

    def test_figure6d_goal_is_unsuccessful_save(self):
        passing, receiving, _ = self.scene('goal')
        self.assertEqual(passing[0]['event'], 'shot_on_target')
        self.assertEqual(next(e['event'] for e in receiving if e['slot'] == 1), 'unsuccessful_save')

    def test_retention_and_deflection_one_second_rule(self):
        _, retained, _ = self.scene()
        _, deflected, _ = self.scene(release=30)
        self.assertEqual(next(e['event'] for e in retained if e['slot'] == 1), 'save_retain')
        self.assertEqual(next(e['event'] for e in deflected if e['slot'] == 1), 'save_deflect')

    def test_cross_to_keeper_is_claim_when_attacker_in_area(self):
        _, receiving, _ = self.scene(shooter=(90, 1))
        self.assertEqual(next(e['event'] for e in receiving if e['slot'] == 1), 'claim_retain')


class AdapterAndEvaluationTests(unittest.TestCase):
    def test_native_player_control_ends_at_release_and_retains_travel_as_no_possession(self):
        core = dict(gains=[dict(frame=1, slot=0, team=0), dict(frame=10, slot=1, team=1)],
                    duel=np.zeros(15, bool), pz=np.ones((15, 2), bool),
                    distances=np.ones((15, 2)), teams=np.tile([0, 1], (15, 1)))
        player, state, intervals, loose = native_sequences(core, np.ones(15), np.zeros(15, bool), [],
            [dict(frame=5, slot=0, event='pass')], set())
        np.testing.assert_array_equal(player[1:6], 0)
        np.testing.assert_array_equal(player[6:10], -1)
        np.testing.assert_array_equal(state[6:10], 1)
        self.assertEqual(intervals[0]['outcome'], 'pass')

    def test_tactical_default_does_not_load_mall_events_models_or_dead_mask(self):
        loaded = []
        n = 4
        z = dict(ball_xy_raw=np.zeros((n, 2)), player_xy_raw=np.zeros((n, 2, 2)),
                 half=np.ones(n), team_of_slot=np.array([0, 1]), pitch=PITCH, attack_sign=SIGNS)
        def load(layer):
            loaded.append(layer)
            return z
        empty = dict(gains=[], passing_events=[], restarts=[])
        core = dict(state=np.zeros(n), duel=np.zeros(n), player_owner=np.zeros(n), ball_control=np.zeros(n))
        with tempfile.TemporaryDirectory() as folder, \
             patch('pipeline.mall.common.CACHE', str(Path(folder) / 'mall')), \
             patch('pipeline.mall.common.load_layer', side_effect=load), \
             patch('pipeline.run.write_csv'), \
             patch('pipeline.paper2022.run.reproduce', return_value=(np.zeros(n), core, empty)):
            tactical(Path(folder) / 'output.csv')
        self.assertEqual(loaded, ['layer0'])

    def test_duel_does_not_immediately_override_previous_owner(self):
        core = dict(gains=[dict(frame=0, team=1, slot=1, end=5), dict(frame=10, team=0, slot=0, end=15)],
                    duel=np.array([False] * 10 + [True, True] + [False] * 8),
                    pz=np.ones((20, 2), bool), distances=np.ones((20, 2)), teams=np.tile([0, 1], (20, 1)))
        labels, _ = viewer_labels(core, np.ones(20), np.zeros(20, bool), [], set())
        np.testing.assert_array_equal(labels[:12], 1)
        np.testing.assert_array_equal(labels[12:], 0)

    def test_csv_adapter_does_not_backdate_a_future_gain_into_unknown_prefix(self):
        core = dict(gains=[dict(frame=10, team=1, slot=1, end=15)], duel=np.zeros(20, bool),
                    pz=np.ones((20, 2), bool), distances=np.tile([0.2, 4.0], (20, 1)),
                    teams=np.tile([0, 1], (20, 1)))
        labels, _ = viewer_labels(core, np.ones(20), np.zeros(20, bool), [], set())
        np.testing.assert_array_equal(labels[:10], 0)
        np.testing.assert_array_equal(labels[10:], 1)

    def test_status_file_cannot_supply_ownership_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'status.csv'
            path.write_text('frame,label\n0,H\n')
            with self.assertRaises(ValueError):
                read_status(path, 1)
            path.write_text('frame,in_play\n0,1\n1,0\n')
            np.testing.assert_array_equal(read_status(path, 2), [False, True])

    def test_event_matching_is_actor_aware_and_one_to_one(self):
        refs = [dict(frame=100, half=1, slot=1, event='pass')]
        preds = [dict(frame=100, half=1, slot=2, event='pass'),
                 dict(frame=103, half=1, slot=1, event='pass'), dict(frame=104, half=1, slot=1, event='pass')]
        result = match_events(preds, refs, 12)
        self.assertEqual(result['matched'], 1)
        self.assertEqual(result['median_absolute_error_frames'], 3)

    def test_event_matching_maximizes_count_before_minimizing_timing_error(self):
        refs = [dict(frame=f, half=1, slot=1, event='pass') for f in (0, 10)]
        preds = [dict(frame=f, half=1, slot=1, event='pass') for f in (9, 20)]
        self.assertEqual(match_events(preds, refs, 10)['matched'], 2)

    def test_paper_timing_tolerance_does_not_imply_frame_accuracy(self):
        refs = [dict(frame=100, half=1, slot=1, event='pass')]
        preds = [dict(frame=140, half=1, slot=1, event='pass')]
        self.assertEqual(match_events(preds, refs, 12)['matched'], 0)
        self.assertEqual(match_events(preds, refs, 250)['matched'], 1)


if __name__ == '__main__':
    unittest.main()
