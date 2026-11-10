"""Behavioral checks for contact confirmation, boundaries and model isolation."""
import unittest
from unittest.mock import patch

import numpy as np

from .mall.interaction.features import contacts, design, regions
from .mall.interaction.model import predict, train
from .mall.layer2_touches.touches import RECEIVE, TOUCH, STOP
from .mall.layer4_possession_model import crossgame as X
from .mall.layer4_possession_model.model import PLAYER_BASE


def fixture():
    n = 80
    z = dict(half=np.ones(n, np.int8), ball_xy=np.zeros((n, 2)),
             ball_speed=np.zeros(n), ball_vel=np.zeros((n, 2)))
    keys = set(PLAYER_BASE) | set(X.BALL_NEUTRAL) | {'margin', 'ball_fwd_speed_home'}
    keys |= {f'{key}_{tag}' for key in X.BALL_PAIRS for tag in ('h', 'a')}
    F = {key: np.zeros(n) for key in keys}
    events = np.array([(1, RECEIVE, 1, 0.5), (10, TOUCH, 0, 0.8),
                       (14, TOUCH, 1, 1.0), (30, RECEIVE, 0, 0.5)],
                      dtype=[('frame', 'i4'), ('kind', 'i1'), ('team', 'i1'), ('dist', 'f4')])
    L2 = dict(events=events, control_team=np.full(n, -1), last_control_team=np.ones(n),
              frames_since_control=np.arange(n))
    return z, F, L2, dict(dead=np.zeros(n, bool))


class InteractionFeaturesTests(unittest.TestCase):
    def test_later_recovery_confirms_contact_and_survives_opponent_rebound(self):
        z, F, L2, L3 = fixture()
        matrix, names, events = design(z, F, L2, L3)
        self.assertEqual(matrix[12, names.index('turnover_chain_h')], 1)
        self.assertEqual(matrix[15, names.index('rebound_return_h')], 1)
        self.assertEqual(matrix[15, names.index('recovery_h')], 1)
        self.assertAlmostEqual(matrix[15, names.index('recovery_age')], 0.6)
        # This is evidence; feature construction never edits possession labels.
        self.assertNotIn('label', events[0])

    def test_chains_do_not_cross_dead_ball_or_half_boundary(self):
        for boundary in ('dead', 'half'):
            z, F, L2, L3 = fixture()
            if boundary == 'dead':
                L3['dead'][20:23] = True
            else:
                z['half'][20:] = 2
            matrix, names, _ = design(z, F, L2, L3)
            self.assertEqual(matrix[12, names.index('recovery_h')], 0)
            self.assertTrue(np.isnan(matrix[12, names.index('recovery_age')]))

    def test_suspected_stop_can_be_retained_without_becoming_confirmed_control(self):
        z, F, L2, L3 = fixture()
        with patch('pipeline.mall.interaction.features.kinematic_candidates',
                   return_value=(np.array([18]), np.array([STOP]), np.zeros(80))), \
             patch('pipeline.mall.interaction.features.attribute', return_value=[(0, 1, 1.2, 19)]):
            event = next(e for e in contacts(z, F, L2, L3['dead']) if e['frame'] == 19)
        self.assertFalse(event['anchor'])
        self.assertEqual(event['confidence'], 0.2)
        self.assertEqual(event['motion_frame'], 18)

    def test_home_away_swap_preserves_feature_meaning(self):
        z, F, L2, L3 = fixture()
        F['d_h'][:] = 1
        F['d_a'][:] = 2
        F['margin'][:] = 1
        matrix, names, _ = design(z, F, L2, L3)
        swapped = {}
        for key, values in F.items():
            other = key[:-2] + ('_a' if key.endswith('_h') else '_h') if key.endswith(('_h', '_a')) else key
            swapped[key] = -F[other] if key in X.FLIP else F[other].copy()
        L2['events']['team'] = 1 - L2['events']['team']
        L2['last_control_team'] = 1 - L2['last_control_team']
        second, second_names, _ = design(z, swapped, L2, L3)
        idx, sign = X.swap_index(names)
        self.assertEqual(names, second_names)
        np.testing.assert_allclose(matrix[:, idx] * sign, second, equal_nan=True)

    def test_missing_ball_events_are_neutral_features(self):
        z, F, L2, L3 = fixture()
        z['ball_xy'][20:25] = np.nan
        matrix, names, _ = design(z, F, L2, L3)
        self.assertGreater(matrix[20, names.index('lost_nearby')], 0)
        self.assertGreater(matrix[25, names.index('found_nearby')], 0)


class ModelIsolationTests(unittest.TestCase):
    def test_tactical_or_empty_training_is_rejected(self):
        for tags in [('tactical',), ()]:
            with self.assertRaises(ValueError):
                train(tags)

    def test_scoring_a_training_game_is_rejected(self):
        with self.assertRaises(ValueError):
            predict('bench1', {'train_games': ('bench1', 'bench2')})

    def test_hybrid_preserves_dead_mask_and_original_evidence(self):
        n = 20
        half = np.ones(n, np.int8)
        half[:2] = 0
        dead = np.zeros(n, bool)
        dead[7:10] = True
        events = np.empty(0, dtype=[('frame', 'i4'), ('kind', 'i1'), ('team', 'i1'),
                                    ('dist', 'f4'), ('duration', 'i4'), ('contest', 'f4')])
        L2 = dict(ball_state=np.zeros(n), events=events)
        frames = X.cp_candidates(dead, half)
        cp = np.tile([0.98, 0.01, 0.01], (len(frames), 1))
        legacy = dict(p_home_full=np.full(n, 0.7), p_home_players=np.full(n, 0.5),
                      cp_frames=frames, cp_p=cp.copy())
        trained = dict(train_games=('bench1', 'bench2'), names=['team_h', 'team_a'],
                       frame=(None, None), cp=(None, None), blend=0.5)
        with patch('pipeline.mall.interaction.model.inputs', return_value=(dict(half=half, ball_gap_len=np.zeros(n)), {}, L2, dict(dead=dead))), \
             patch('pipeline.mall.interaction.model.design', return_value=(np.zeros((n, 2)), trained['names'], [])), \
             patch('pipeline.mall.interaction.model.X.predict', return_value=np.full(n, 0.8)), \
             patch('pipeline.mall.interaction.model.X.predict_cp', return_value=cp.copy()):
            result, _, _ = predict('tactical', trained, legacy)
        np.testing.assert_array_equal(result['label'] == 2, dead | (half == 0))
        np.testing.assert_array_equal(legacy['cp_p'], cp)
        np.testing.assert_array_equal(legacy['p_home_full'], np.full(n, 0.7))


if __name__ == '__main__':
    unittest.main()
