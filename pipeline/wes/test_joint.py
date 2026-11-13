"""Constrained joint-decoder behavior on small, interpretable sequences."""

import unittest

import numpy as np

from .joint import decode_window


class JointDecoderTests(unittest.TestCase):
    def test_reopens_dead_only_when_live_evidence_wins(self):
        base = np.array([0] * 10 + [2] * 10 + [0] * 10, np.int8)
        owner = np.full(30, 2.0)
        hard = np.zeros(30, bool)
        live, _ = decode_window(0, 30, base, np.full(30, -2.0), owner,
                                hard, {}, 'joint-d')
        self.assertTrue(np.all(live == 0))
        dead, _ = decode_window(0, 30, base, np.full(30, 2.0), owner,
                                hard, {}, 'joint-d')
        self.assertTrue(np.all(dead[10:20] == 2))

    def test_hard_dead_and_fixed_live_are_respected(self):
        base = np.array([0] * 10 + [2] * 10 + [1] * 10, np.int8)
        hard = np.zeros(30, bool)
        hard[13:17] = True
        labels, _ = decode_window(0, 30, base, np.full(30, -3.0),
                                  np.zeros(30), hard, {}, 'joint-d')
        self.assertTrue(np.all(labels[:10] == 0))
        self.assertTrue(np.all(labels[13:17] == 2))
        self.assertTrue(np.all(labels[20:] == 1))

    def test_team_only_can_recover_strong_opposing_evidence(self):
        base = np.zeros(50, np.int8)
        owner = np.full(50, 2.0)
        owner[15:35] = -4.0
        labels, _ = decode_window(0, 50, base, np.full(50, -2.0), owner,
                                  np.zeros(50, bool), {}, 'joint-team')
        self.assertEqual(labels[25], 1)
        self.assertEqual(labels[0], 0)
        self.assertEqual(labels[-1], 0)


if __name__ == '__main__':
    unittest.main()
