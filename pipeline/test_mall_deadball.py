"""Mall's learned dead-ball evidence and the extra bench games, on synthetic data only."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from .mall import config as C
from .mall.bench import degrade, idsse
from .mall.layer3_dead_ball import deadball, learned


def evidence_stub(gap_frames, n=2000):
    """A lost ball in `gap_frames` and no other rule cue."""
    parts = {"gap": np.zeros(n), "outside": np.full(n, -0.5)}      # mild evidence of play
    parts["gap"][gap_frames] = 3.0
    parts["kickoff_shape"] = np.zeros(n)
    return parts


class LearnedEvidenceTests(unittest.TestCase):
    def detect(self, p, gap=slice(300, 600), trace=False, **settings):
        n = 2000                                      # two halves; the scene is in the first
        if p is not None:
            p = np.r_[p, np.full(n - len(p), 0.02)]
        z = {"half": np.repeat([1, 2], n // 2).astype(np.int8),
             "ball_gap_len": np.zeros(n, np.int32)}
        F = {"ball_speed": np.full(n, 5.0), "ball_outside": np.zeros(n),
             "st_set_piece": np.zeros(n, np.int8)}
        parts = evidence_stub(gap, n)
        patches = [patch.object(deadball, "evidence",
                                lambda *a: (sum(parts.values()), dict(parts)))]
        patches += [patch.object(C, k, v) for k, v in settings.items()]
        for q in patches:
            q.start()
        try:
            return deadball.detect(z, F, {}, p, trace=trace)
        finally:
            for q in patches:
                q.stop()

    def test_lost_ball_alone_still_reads_as_a_stoppage_without_the_model(self):
        dead, _, parts = self.detect(None)
        self.assertTrue(dead[320:580].all())
        self.assertNotIn("learned", parts)

    def test_players_in_play_veto_a_lost_ball(self):
        p = np.full(1000, 0.02)                       # the players never look stopped
        dead, _, parts = self.detect(p, DEAD_W_LEARNED=1.0, DEAD_GAP_GATE=0.5)
        self.assertFalse(dead.any())
        self.assertLess(parts["gap"][400], 0.2)       # the gap cue has faded, not just lost

    def test_players_stopped_keep_the_lost_ball_cue_in_full(self):
        p = np.full(1000, 0.02)
        p[300:600] = 0.9
        dead, _, parts = self.detect(p, DEAD_W_LEARNED=1.0, DEAD_GAP_GATE=0.5)
        self.assertTrue(dead[320:580].all())
        self.assertEqual(parts["gap"][400], 3.0)
        self.assertFalse(dead[:250].any())

    def test_players_alone_can_mark_a_stoppage_with_the_ball_in_view(self):
        p = np.full(1000, 0.02)
        p[300:600] = 0.98
        dead, _, _ = self.detect(p, gap=slice(0, 0), DEAD_W_LEARNED=1.0, DEAD_GAP_GATE=0.5)
        self.assertTrue(dead[320:580].all())

    def test_learned_evidence_is_clipped(self):
        p = np.full(1000, 1e-9)
        _, _, parts = self.detect(p, DEAD_W_LEARNED=1.0, DEAD_LEARNED_CLIP=6.0)
        self.assertEqual(parts["learned"].min(), -6.0)

    def test_wes_trace_preserves_learned_predictions_and_evidence(self):
        p = np.full(1000, 0.02)
        p[300:600] = 0.98
        settings = dict(DEAD_W_LEARNED=1.0, DEAD_GAP_GATE=0.5)
        plain = self.detect(p, **settings)
        traced = self.detect(p, trace=True, **settings)
        for a, b in zip(plain[:2], traced[:2]):
            np.testing.assert_array_equal(a, b)
        for key in plain[2]:
            np.testing.assert_array_equal(plain[2][key], traced[2][key])
        self.assertIn('learned', traced[2])
        self.assertTrue(traced[3]['viterbi'][320:580].all())
        self.assertEqual(set(traced[3]), {'viterbi', 'snapped', 'rules', 'durations', 'hard_kickoff'})


class PlayerOnlyFeatureTests(unittest.TestCase):
    def test_stoppage_signals_ignore_the_ball_and_the_team_labels(self):
        rng = np.random.default_rng(0)
        n, k = 50, 22
        z = {"player_xy": rng.uniform(0, 60, (n, k, 2)), "player_vel": rng.normal(0, 2, (n, k, 2)),
             "player_team": np.tile(np.r_[np.zeros(11), np.ones(11)].astype(np.int8), (n, 1)),
             "pitch": np.array([105.0, 68.0]), "ball_xy": rng.uniform(0, 60, (n, 2))}
        F = {"mean_speed_all": rng.uniform(0, 3, n)}
        a = learned.stoppage_signals(z, F)
        z2 = dict(z, player_team=1 - z["player_team"], ball_xy=np.full((n, 2), np.nan))
        b = learned.stoppage_signals(z2, F)
        for key in a:
            np.testing.assert_allclose(a[key], b[key], err_msg=key)

    def test_a_player_behind_the_touchline_is_seen(self):
        n, k = 3, 4
        xy = np.full((n, k, 2), 30.0)
        xy[1, 0] = (40.0, -0.8)                       # a thrower, frame 1 only
        z = {"player_xy": xy, "player_vel": np.zeros((n, k, 2)),
             "player_team": np.zeros((n, k), np.int8), "pitch": np.array([105.0, 68.0])}
        S = learned.stoppage_signals(z, {"mean_speed_all": np.zeros(n)})
        np.testing.assert_array_equal(S["n_outside"], [0, 1, 0])
        self.assertLess(S["touch_min"][1], 0)


class BenchDegradationTests(unittest.TestCase):
    def setUp(self):
        self.n = 25 * 600
        self.dead = np.zeros(self.n, bool)
        for a in range(1000, self.n - 1500, 1500):
            self.dead[a:a + 500] = True               # a 20 s stoppage every minute
        self.period = np.ones(self.n, np.int8)
        self.ball = np.full((self.n, 2), 0.5)

    def test_simulated_losses_are_never_shorter_than_the_provider_bridges(self):
        out = degrade.ball_dropouts(self.ball, self.dead, self.period, seed=3)
        missing = np.isnan(out[:, 0])
        runs = degrade._runs(missing)
        self.assertTrue(runs)
        self.assertTrue(all(b - a >= 125 for a, b in runs))
        self.assertGreater(missing[self.dead].mean(), 0.4)       # mostly at stoppages
        self.assertLess(missing[~self.dead].mean(), 0.25)

    def test_dropouts_are_reproducible_and_leave_the_input_alone(self):
        a = degrade.ball_dropouts(self.ball, self.dead, self.period, seed=3)
        b = degrade.ball_dropouts(self.ball, self.dead, self.period, seed=3)
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        self.assertFalse(np.isnan(self.ball).any())

    def test_set_piece_windows_come_before_restarts_and_throw_ins_have_no_team(self):
        label = np.where(self.dead, 2, 0).astype(np.int8)
        ball = self.ball.copy()
        ball[:, 1] = 0.0                              # every restart on a touchline, not a corner
        st = degrade.set_piece_tags_from_key(label, ball, (105.0, 68.0), self.n, seed=1, p_tag=1.0)
        restarts = np.flatnonzero(self.dead[:-1] & ~self.dead[1:]) + 1
        later = restarts[1:]                          # the first restart is the kick-off
        self.assertTrue((st["st_set_piece"][later - 1] == 1).all())      # 1 = throw-in
        self.assertTrue((st["st_set_piece_team"][later - 1] == -1).all())


class ShortWindowFeatureTests(unittest.TestCase):
    def setUp(self):
        from .mall.layer2_touches.touches import KICK, RECEIVE
        rng = np.random.default_rng(1)
        n = 400
        self.F = {k: rng.uniform(0, 5, n) for k in
                  ["d_h", "d_a", "relv_h", "relv_a", "closing_h", "closing_a", "n2_h", "n2_a",
                   "n5_h", "n5_a", "dwell_h", "dwell_a", "nearest_fwd_speed_h",
                   "nearest_fwd_speed_a", "ball_ahead_h", "ball_ahead_a", "margin",
                   "ball_fwd_speed_home", "ball_known", "ball_speed", "ball_gap_len",
                   "ball_frames_since_seen", "ball_frames_until_seen", "ball_to_line",
                   "ball_outside"]}
        events = np.array([(100, RECEIVE, 0, 3, 0.5, 1, 1, np.nan, 40, 0.1),
                           (200, RECEIVE, 1, 7, 0.5, 1, 1, np.nan, 12, 0.6),
                           (150, KICK, 1, 7, 0.5, 1, 1, np.nan, 0, np.nan)],
                          dtype=[("frame", "i4"), ("kind", "i1"), ("team", "i1"), ("slot", "i1"),
                                 ("dist", "f4"), ("speed_before", "f4"), ("speed_after", "f4"),
                                 ("turn", "f4"), ("duration", "i4"), ("contest", "f4")])
        ctrl = np.full(n, -1, np.int8)
        ctrl[100:140], ctrl[200:212] = 0, 1
        self.L2 = {"events": events, "control_team": ctrl, "last_control_team": ctrl,
                   "frames_since_control": np.zeros(n, np.int32), "turn": rng.uniform(0, 90, n)}
        self.frames = np.arange(0, n, 3)

    def test_reception_starting_at_a_candidate_is_described(self):
        from .mall.layer4_possession_model import crossgame as X
        A, names = X.fine_features(self.F, self.L2, self.frames)
        self.assertEqual(A.shape, (len(self.frames), len(names)))
        row = list(self.frames).index(99)
        self.assertEqual(A[row, names.index("nextrec_dur_h")], 40)
        self.assertEqual(A[row, names.index("nextrec_in_h")], 1)
        self.assertEqual(A[row, names.index("nextrec_dur_a")], 0)       # away's is 4 s away
        self.assertFalse(np.isnan(A[:, names.index("dmin_fw0_12_h")]).any())

    def test_home_away_swap_is_defined_for_every_short_window_column(self):
        from .mall.layer4_possession_model import crossgame as X
        A, names = X.fine_features(self.F, self.L2, self.frames)
        idx, sgn = X.swap_index(names)
        self.assertEqual(sorted(idx), list(range(len(names))))          # a permutation
        np.testing.assert_array_equal(idx[idx], np.arange(len(names)))  # its own inverse
        flipped = [n for n, s_ in zip(names, sgn) if s_ < 0]
        self.assertTrue(flipped and all(n.startswith(("margin", "ball_fwd_speed_home"))
                                        for n in flipped))


def layer0_stub(n=600, seed=0):
    rng = np.random.default_rng(seed)
    k = 22
    xy = rng.uniform(5, 60, (n, k, 2)).astype(np.float32)
    ball = xy[:, 0] + rng.normal(0, 0.5, (n, 2)).astype(np.float32)   # home slot 0 has it
    ball[200:260] = np.nan
    since = np.zeros(n, np.int32)
    since[200:260] = np.arange(1, 61)
    return {"player_xy": xy, "player_vel": rng.normal(0, 2, (n, k, 2)).astype(np.float32),
            "player_team": np.tile(np.r_[np.zeros(11), np.ones(11)].astype(np.int8), (n, 1)),
            "ball_xy": ball, "ball_vel": rng.normal(0, 3, (n, 2)).astype(np.float32),
            "half": np.ones(n, np.int8), "pitch": np.array([105.0, 68.0]),
            "attack_sign": np.array([[0, 0], [1, -1], [-1, 1]], np.int8),
            "ball_frames_since_seen": since, "ball_frames_until_seen": since[::-1].copy()}


class TrajectoryFeatureTests(unittest.TestCase):
    def test_swapping_the_team_labels_swaps_the_two_blocks(self):
        from .mall.layer4_possession_model import sequence as Q
        z = layer0_stub()
        a = Q.features(z)
        z2 = dict(z, player_team=1 - z["player_team"], attack_sign=z["attack_sign"][:, ::-1])
        np.testing.assert_allclose(Q.features(z2), Q.swap_teams(a), atol=1e-6)
        self.assertEqual(a.shape, (600, Q.DIM))
        self.assertFalse(np.isnan(a).any())

    def test_nearest_player_is_first_and_a_lost_ball_blanks_the_ball_parts(self):
        from .mall.layer4_possession_model import sequence as Q
        a = Q.features(layer0_stub())
        self.assertLess(a[:200, 0].max(), 0.4)                 # home's nearest is within ~4 m
        self.assertTrue((a[:200, 0] <= a[:200, Q.PER_PLAYER]).all())    # sorted by distance
        self.assertTrue((a[200:260, :Q.K_NEAR * Q.PER_PLAYER] == 0).all())
        self.assertTrue((a[200:260, 2 * Q.BLOCK] == 0).all())  # ball_known

    def test_change_targets_mark_six_frames_either_side_and_skip_dead_ball(self):
        from .mall.layer4_possession_model import sequence as Q
        truth = np.r_[np.zeros(50), np.ones(50), np.full(20, 2), np.zeros(30)].astype(int)
        poss, ch = Q.targets(truth)
        np.testing.assert_array_equal(np.flatnonzero(ch == 2), np.arange(44, 57))   # to away at 50
        self.assertFalse((ch == 1).any())                      # away -> dead -> home is a restart
        self.assertTrue((ch[100:120] == -1).all() and (poss[100:120] == -1).all())
        self.assertTrue((poss[:50] == 1).all() and (poss[50:100] == 0).all())

    def test_training_is_repeatable_and_predictions_respect_team_order(self):
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed")
        from .mall.layer4_possession_model import sequence as Q
        z = layer0_stub(n=1400)
        X = Q.features(z)
        truth = np.zeros(1400, np.int64)
        with patch.object(Q, "BATCH", 4), patch.object(Q, "HIDDEN", 16),                 patch.object(torch.cuda, "is_available", lambda: False):
            a = Q.train([(X, truth, z["half"])], seed=3, steps=5)
            b = Q.train([(X, truth, z["half"])], seed=3, steps=5)
            for key in a:
                self.assertTrue(torch.equal(a[key], b[key]), key)
            p, c = Q.predict(a, X, z["half"])
            ps, cs = Q.predict(a, Q.swap_teams(X), z["half"])
        np.testing.assert_allclose(p, 1 - ps, atol=1e-5)       # swap the teams, swap the answer
        np.testing.assert_allclose(c[:, 1], cs[:, 2], atol=1e-5)
        np.testing.assert_allclose(c.sum(axis=1), 1.0, atol=1e-5)


POSITIONS = """<PutDataRequest><Positions>
<FrameSet GameSection="firstHalf" TeamId="H" PersonId="p1">
<Frame N="10000" X="0.0" Y="0.0"/>
<Frame N="10001" X="1.05" Y="0.0"/>
</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="A" PersonId="p2">
<Frame N="10000" X="-52.5" Y="-34.0"/>
<Frame N="10001" X="-52.5" Y="-34.0"/>
</FrameSet>
<FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="b">
<Frame N="10000" X="0.0" Y="0.0" Z="0.1" BallPossession="1" BallStatus="1"/>
<Frame N="10001" X="0.0" Y="34.0" Z="3.0" BallPossession="2" BallStatus="1"/>
</FrameSet>
<FrameSet GameSection="secondHalf" TeamId="BALL" PersonId="b">
<Frame N="100000" X="0.0" Y="0.0" Z="0.0" BallPossession="2" BallStatus="0"/>
</FrameSet>
<FrameSet GameSection="secondHalf" TeamId="referee" PersonId="r">
<Frame N="100000" X="5.0" Y="5.0"/>
</FrameSet>
</Positions></PutDataRequest>
"""
META = """<PutDataRequest><MatchInformation>
<General HomeTeamId="H" GuestTeamId="A" />
<Environment PitchX="105.00" PitchY="68.00" />
<Teams><Team TeamId="H" TeamName="Home FC" Role="home"><Players/></Team>
<Team TeamId="A" TeamName="Away FC" Role="guest"><Players/></Team></Teams>
</MatchInformation></PutDataRequest>
"""


class DflLoaderTests(unittest.TestCase):
    def test_key_comes_from_ball_status_and_possession(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, text in (("T_meta.xml", META), ("T_tracking.xml", POSITIONS)):
                with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
                    f.write(text)
            with patch.object(idsse, "SRC", folder):
                G = idsse.load("T")
        np.testing.assert_array_equal(G["label"], [0, 1, 2])       # home, away, dead
        np.testing.assert_array_equal(G["period"], [1, 1, 2])
        np.testing.assert_array_equal(G["team_of_slot"], [0, 1])   # the referee is not a player
        np.testing.assert_allclose(G["players"][1, 0], [0.51, 0.5])  # metres -> 0-1, centre origin
        np.testing.assert_allclose(G["players"][0, 1], [0.0, 0.0])
        self.assertTrue(np.isnan(G["players"][2]).all())
        np.testing.assert_allclose(G["ball"][1], [0.5, 1.0])
        self.assertEqual(G["names"], ("Home FC", "Away FC"))


if __name__ == "__main__":
    unittest.main()
