"""Prevent legacy benchmark results from silently changing production selection."""
import unittest

from ..mall import config as C
from .evaluate import baseline_settings, gate


class EvaluationProtocolTests(unittest.TestCase):
    def test_legacy_weights_restore_even_after_failure(self):
        keys = ('DEAD_W_LEARNED', 'DEAD_W_SPEED', 'SEQ_W_FULL', 'SEQ_W_CP')
        before = {key: getattr(C, key) for key in keys}
        with self.assertRaises(RuntimeError):
            with baseline_settings('legacy'):
                self.assertEqual(C.SEQ_W_FULL, 0.35)
                self.assertEqual(C.DEAD_W_LEARNED, 0)
                raise RuntimeError('interrupted evaluation')
        self.assertEqual(before, {key: getattr(C, key) for key in keys})

    def test_promotion_gate_includes_dfl_regressions(self):
        games = {tag: {'original': {'spellF1': 0.7}, 'wes': {'spellF1': 0.75}}
                 for tag in ('bench1', 'bench2', 'bench3')}
        games['dfl1'] = {'original': {'spellF1': 0.7}, 'wes': {'spellF1': 0.5}}
        result = gate({'games': games}, 'wes')
        self.assertFalse(result['score_gate_passed'])
        self.assertEqual(len(result['per_game_change']), 4)


if __name__ == '__main__':
    unittest.main()
