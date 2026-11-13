"""Focused invariants for the optional segmentation path."""

import unittest
from unittest.mock import patch

import numpy as np

from .segmentation import change_candidates, revise_dead, stage_dead


class SegmentationTests(unittest.TestCase):
    def setup_inputs(self):
        n = 100
        z = dict(half=np.ones(n, np.int8), ball_xy=np.zeros((n, 2)))
        F = dict(ball_outside=np.zeros(n), d_h=np.full(n, 0.7), d_a=np.full(n, 0.8))
        event_dtype = [('kind', 'i1'), ('frame', 'i4'), ('duration', 'i4'),
                       ('contest', 'f4')]
        L2 = dict(events=np.empty(0, dtype=event_dtype), ball_state=np.zeros(n))
        L3 = dict(dead=np.ones(n, np.int8), stage_viterbi=np.ones(n, np.int8),
                  dead_llr=np.ones(n), hard_kickoff=np.zeros(n, np.int8))
        L4 = dict(p_home_players=np.full(n, 0.5), p_home_full=np.full(n, 0.5))
        return z, F, L2, L3, L4

    def test_reopens_live_path_removed_by_postprocessing(self):
        z, F, L2, L3, L4 = self.setup_inputs()
        L3['stage_viterbi'][20:50] = 0
        L3['dead_llr'][20:50] = -1.0
        dead, rows = revise_dead(z, F, L2, L3, L4)
        self.assertFalse(dead[20:50].any())
        self.assertTrue(dead[:20].all() and dead[50:].all())
        self.assertEqual(sum(r['reopened'] for r in rows), 1)

    def test_outside_and_kickoff_are_hard_constraints(self):
        z, F, L2, L3, L4 = self.setup_inputs()
        L3['stage_viterbi'][20:50] = 0
        L3['dead_llr'][20:50] = -3.0
        F['ball_outside'][30:35] = 1.0
        L3['hard_kickoff'][40:45] = 1
        dead, _ = revise_dead(z, F, L2, L3, L4)
        self.assertTrue(dead[30:35].all())
        self.assertTrue(dead[40:45].all())

    def test_contact_preserves_both_teams_without_forcing_change(self):
        z, F, L2, _, L4 = self.setup_inputs()
        event = dict(frame=25, motion_frame=24, team=0, kind=2, distance=0.7,
                     confidence=0.5, anchor=False, region=1)
        with patch('pipeline.wes.segmentation.contacts', return_value=[event]):
            rows = change_candidates(z, F, L2, np.zeros(100, bool), L4)
        self.assertEqual({r['source'] for r in rows}, {'contact', 'motion'})
        self.assertTrue(all(r['team_home'] > 0 and r['team_away'] > 0 for r in rows))
        self.assertTrue(all(r['bonus_home'] < 1 and r['bonus_away'] < 1 for r in rows))


if __name__ == '__main__':
    unittest.main()
