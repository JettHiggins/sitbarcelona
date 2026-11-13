"""Checks that diagnostic buckets retain official spell semantics."""

import unittest

import numpy as np

from .diagnose import missed_spells, spell_symptom
from ..vendor.challenge import possession_metrics as metrics


class DiagnoseTests(unittest.TestCase):
    def test_majority_symptoms(self):
        self.assertEqual(spell_symptom(np.array([2, 2, 0]), metrics.H)[0], 'D_majority')
        self.assertEqual(spell_symptom(np.array([1, 1, 0]), metrics.H)[0], 'wrong_team_majority')
        self.assertEqual(spell_symptom(np.array([0, 0, 2]), metrics.H)[0],
                         'right_team_majority_unmatched')

    def test_same_team_across_stoppage_is_two_missed_spells(self):
        gt = np.array([metrics.H] * 10 + [metrics.D] * 10 + [metrics.H] * 10)
        pred = np.array([metrics.H] * 30)
        result = missed_spells(gt, pred)
        self.assertEqual(result['gt_spells'], 2)
        self.assertEqual(result['predicted_spells'], 1)
        self.assertEqual(result['matched_at_0_5'], 0)
        self.assertEqual(result['symptoms'], {'right_team_majority_unmatched': 2})


if __name__ == '__main__':
    unittest.main()
