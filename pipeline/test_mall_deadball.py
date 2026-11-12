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
    def detect(self, p, gap=slice(300, 600), **settings):
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
            return deadball.detect(z, F, {}, p)
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
