"""Checks for training weights and safe cached-prediction evaluation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from .mall.bench.ensemble import read_prediction
from .mall.layer4_possession_model import sequence as Q


class SpellWeightsTests(unittest.TestCase):
    def test_short_possessions_count_more_without_overweighting_one_frame(self):
        y = np.r_[np.zeros(250), np.ones(25), np.zeros(1), np.full(20, 2)]
        w = Q.spell_weights(y, np.ones(len(y)))
        np.testing.assert_allclose(w[:250], 1)
        np.testing.assert_allclose(w[250:275], np.sqrt(10))
        self.assertEqual(w[275], 4)
        self.assertTrue((w[276:] == 0).all())

    def test_halves_split_spells_and_team_swapping_preserves_weights(self):
        y = np.r_[np.zeros(500), np.ones(25)]
        half = np.r_[np.ones(250), np.full(275, 2)]
        w = Q.spell_weights(y, half)
        np.testing.assert_allclose(w[:500], 1)
        np.testing.assert_array_equal(w, Q.spell_weights(1 - y, half))
        self.assertTrue((Q.spell_weights(y, np.zeros(len(y))) == 0).all())

    def test_weighted_training_produces_finite_predictions(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        torch.set_num_threads(1)
        rng = np.random.default_rng(2)
        x = rng.normal(size=(100, Q.DIM)).astype(np.float32)
        y = np.tile(np.r_[np.zeros(15), np.ones(5)], 5).astype(np.int8)
        half = np.ones(100, np.int8)
        with patch.object(Q, "WINDOW", 20), patch.object(Q, "BATCH", 2), \
                patch.object(Q, "HIDDEN", 8), patch.object(Q, "LAYERS", 2), \
                patch.object(torch.cuda, "is_available", lambda: False):
            state = Q.train([(x, y, half)], steps=5, spell_balance=0.5)
            ph, pc = Q.predict(state, x, half)
        self.assertTrue(np.isfinite(ph).all())
        np.testing.assert_allclose(pc.sum(axis=1), 1, atol=1e-6)


class CachedPredictionsTests(unittest.TestCase):
    def test_rejects_misaligned_or_invalid_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.npz"
            np.savez(path, p_home=[0.5, 0.7], p_change=[[1, 0, 0], [0.5, 0.2, 0.3]])
            ph, pc = read_prediction(path, 2)
            self.assertEqual(len(ph), 2)
            with self.assertRaises(ValueError):
                read_prediction(path, 3)
            for bad in (np.full((2, 3), np.nan), np.ones((2, 3))):
                np.savez(path, p_home=ph, p_change=bad)
                with self.assertRaises(ValueError):
                    read_prediction(path, 2)


class TemporalModelTests(unittest.TestCase):
    def test_linear_dilation_matches_convolution_outputs_and_gradients(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from .mall.layer4_possession_model.temporal import build_net
        torch.set_num_threads(1)
        torch.manual_seed(3)
        net = build_net(3, hidden=4, blocks=3).double()
        for block in net.blocks:
            for conv in (block.conv1, block.conv2):
                reference = torch.nn.Conv1d(4, 4, 3, padding=conv.d,
                                            dilation=conv.d).double()
                with torch.no_grad():
                    reference.weight.copy_(conv.lin.weight.reshape(4, 3, 4).permute(0, 2, 1))
                    reference.bias.copy_(conv.lin.bias)
                for length in (1, 11):
                    x = torch.randn(2, length, 4, dtype=torch.float64, requires_grad=True)
                    actual = conv(x)
                    expected = reference(x.transpose(1, 2)).transpose(1, 2)
                    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                    da = torch.autograd.grad(actual.square().sum(), x)[0]
                    de = torch.autograd.grad(expected.square().sum(), x)[0]
                    torch.testing.assert_close(da, de, atol=1e-12, rtol=1e-12)

    def test_chunked_prediction_preserves_team_symmetry(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from .mall.layer4_possession_model.temporal import build_net
        torch.set_num_threads(1)
        x = np.random.default_rng(0).normal(size=(120, Q.DIM)).astype(np.float32)
        half = np.r_[np.zeros(10), np.ones(100), np.zeros(10)]
        with patch.object(Q, "build_net", lambda: build_net(Q.DIM, hidden=8, blocks=2)), \
                patch.object(Q, "PIECE", 30), patch.object(Q, "MARGIN", 25), \
                patch.object(torch.cuda, "is_available", lambda: False):
            state = Q.build_net().state_dict()
            ph, pc = Q.predict(state, x, half)
            sh, sc = Q.predict(state, Q.swap_teams(x), half)
        np.testing.assert_allclose(ph, 1 - sh, atol=1e-6)
        np.testing.assert_allclose(pc[:, 1], sc[:, 2], atol=1e-6)
        np.testing.assert_allclose(ph[half == 0], 0.5)
        np.testing.assert_allclose(pc.sum(axis=1), 1, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
