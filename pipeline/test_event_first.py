"""Event-first ownership rules, independently of learned possession estimates."""
import unittest
from unittest.mock import patch

import numpy as np

from .event_first.events import detect, live_regions
from .event_first.run import build
from .event_first.sequence import reconstruct


def event(frame, teams, anchor=False, region=0):
    return dict(frame=frame, teams=teams, team=teams[0] if len(teams) == 1 else -1,
                anchor=anchor, region=region, kind='control' if anchor else 'contact')


class SequenceTests(unittest.TestCase):
    def decode(self, events, seed=1, n=100):
        return reconstruct(n, [(0, n)], events, {0: seed})

    def test_later_recovery_backdates_to_earlier_teammate_touch(self):
        labels, decisions = self.decode([event(10, [0]), event(30, [0], True)])
        np.testing.assert_array_equal(labels[:10], 1)
        np.testing.assert_array_equal(labels[10:], 0)
        self.assertEqual(decisions[-1]['confirmed_at'], 30)

    def test_opponent_rebound_does_not_erase_recovered_teams_earlier_contact(self):
        labels, _ = self.decode([event(10, [0]), event(14, [1]), event(30, [0], True)])
        np.testing.assert_array_equal(labels[10:], 0)

    def test_incidental_deflection_does_not_switch_without_control(self):
        labels, _ = self.decode([event(10, [0]), event(30, [1], True)])
        np.testing.assert_array_equal(labels, 1)

    def test_brief_deliberate_plays_can_switch_without_cooldown(self):
        labels, _ = self.decode([event(10, [0], True), event(12, [1], True)])
        np.testing.assert_array_equal(labels[10:12], 0)
        np.testing.assert_array_equal(labels[12:], 1)

    def test_ambiguous_contact_is_resolved_by_recovery(self):
        labels, _ = self.decode([event(10, [0, 1]), event(30, [0], True)])
        np.testing.assert_array_equal(labels[10:], 0)

    def test_control_settles_episode_and_stops_later_backdating_through_it(self):
        labels, _ = self.decode([event(10, [0]), event(15, [1], True), event(30, [0], True)])
        np.testing.assert_array_equal(labels[:30], 1)
        np.testing.assert_array_equal(labels[30:], 0)

    def test_confirmation_is_bounded_in_time(self):
        labels, _ = self.decode([event(10, [0]), event(200, [0], True)], n=250)
        np.testing.assert_array_equal(labels[:200], 1)

    def test_nearby_contact_cannot_backdate_through_established_control(self):
        control = dict(event(5, [1], True), end=20)
        labels, _ = self.decode([control, event(10, [0, 1]), event(30, [0], True)])
        np.testing.assert_array_equal(labels[:30], 1)
        np.testing.assert_array_equal(labels[30:], 0)

    def test_dead_ball_and_half_boundaries_break_chains(self):
        half = np.array([0, 1, 1, 1, 1, 1, 2, 2])
        dead = np.array([True, False, False, True, False, False, False, False])
        spans = live_regions(half, dead)
        self.assertEqual(spans, [(1, 3), (4, 6), (6, 8)])
        labels, _ = reconstruct(8, spans, [event(2, [0]), event(5, [0], True, region=1)],
                                {0: 1, 1: 1, 2: 1})
        np.testing.assert_array_equal(labels, [2, 1, 1, 2, 1, 0, 1, 1])

    def test_fallback_is_explicit_and_deterministic(self):
        labels, decisions = reconstruct(5, [(0, 5)], [])
        np.testing.assert_array_equal(labels, 0)
        self.assertIn('fallback', decisions[0]['reason'])

    def test_runtime_only_loads_tracking_features_and_dead_mask(self):
        loaded = []

        def load(layer, tag):
            loaded.append(layer)
            return {'half': np.ones(3)} if layer == 'layer0' else {'dead': np.zeros(3)}

        with patch('pipeline.mall.common.load_layer', side_effect=load), \
             patch('pipeline.event_first.run.predict', return_value=(np.zeros(3), {})):
            build()
        self.assertEqual(loaded, ['layer0', 'layer1', 'layer3'])


class DetectionTests(unittest.TestCase):
    def fixture(self, n=30):
        z = dict(half=np.ones(n, np.int8), ball_speed=np.full(n, 10.0),
                 ball_vel=np.tile([10.0, 0.0], (n, 1)))
        F = {key: np.full(n, value) for key, value in
             [('d_h', 0.8), ('d_a', 1.0), ('relv_h', 99.0), ('relv_a', 99.0),
              ('slot_h', 0), ('slot_a', 1)]}
        return z, F, np.zeros(n, bool)

    def test_short_control_is_an_anchor_but_not_a_single_frame_touch(self):
        z, F, dead = self.fixture()
        F['relv_h'][5:9] = 0
        F['relv_h'][15:18] = 0
        _, events = detect(z, F, dead)
        controls = [e for e in events if e['kind'] == 'control']
        self.assertEqual([(e['frame'], e['duration']) for e in controls], [(5, 4)])

    def test_control_cannot_join_different_players_of_the_same_team(self):
        z, F, dead = self.fixture()
        F['relv_h'][5:11] = 0
        F['slot_h'][8:] = 2
        _, events = detect(z, F, dead)
        self.assertFalse(any(e['kind'] == 'control' for e in events))

    def test_contested_redirect_preserves_both_teams_without_forcing_a_kick(self):
        z, F, dead = self.fixture()
        z['ball_speed'][15] = 2
        _, events = detect(z, F, dead)
        contact = next(e for e in events if e['motion_frame'] == 15)
        self.assertEqual(contact['teams'], [0, 1])
        self.assertEqual(contact['team'], -1)
        self.assertEqual(contact['frame'], 15)
        self.assertFalse(contact['anchor'])


if __name__ == '__main__':
    unittest.main()
