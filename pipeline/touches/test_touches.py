import tempfile
import unittest
from pathlib import Path

import numpy as np

from .core import assign_teams, detect, features, signals
from .run import write_csv


class AlwaysTouch:
    def predict_proba(self, x):
        return np.tile([0.05, 0.95], (len(x), 1))


class TouchTests(unittest.TestCase):
    def test_immediate_opponent_switch_without_confirmation(self):
        half = np.ones(15, int)
        events = [dict(frame=3, team=0), dict(frame=4, team=1), dict(frame=5, team=0)]
        y = assign_teams(len(half), half, events)
        np.testing.assert_array_equal(y[:6], [-1, -1, -1, 0, 1, 0])
        self.assertTrue((y[6:] == 0).all())

    def test_half_reset_and_unknown_export(self):
        y = assign_teams(7, np.array([1, 1, 0, 2, 2, 2, 2]),
                         [dict(frame=1, team=1), dict(frame=5, team=0)])
        np.testing.assert_array_equal(y, [-1, 1, -1, -1, -1, 0, 0])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'out.csv'
            write_csv(y, path)
            self.assertEqual(path.read_text().splitlines()[1:4], ['0,X', '1,A', '2,X'])

    def test_gaps_and_jumps_cannot_create_touches(self):
        ball = np.column_stack([np.arange(100)*.1, np.zeros(100)])
        half = np.ones(100, int)
        ball[30] = np.nan
        ball[65] += 30
        valid = signals(ball, half)['valid']
        self.assertFalse(valid[18:43].any())
        self.assertFalse(valid[52:79].any())

    def test_static_ball_is_not_repeated_touch_even_with_positive_classifier(self):
        ball = np.zeros((80, 2))
        players = np.ones((80, 2, 2))*.2
        y, trace = detect(players, ball, np.ones(80, int), np.zeros((80, 2), int), AlwaysTouch())
        self.assertEqual(trace['touches'], [])
        self.assertTrue((y == -1).all())

    def test_contested_identical_actors_reject_instead_of_team_guess(self):
        t = np.arange(100)
        ball = np.column_stack([np.where(t < 50, t*.1, 5-(t-50)*.1), np.zeros(100)])
        players = np.broadcast_to(ball[:, None]+[0, .2], (100, 2, 2)).copy()
        _, trace = detect(players, ball, np.ones(100, int), np.tile([0, 1], (100, 1)), AlwaysTouch())
        self.assertEqual(trace['touches'], [])
        self.assertGreater(len(trace['rejected']), 0)

    def test_detected_touch_holds_team_across_later_ball_dropout(self):
        t = np.arange(100)
        ball = np.column_stack([np.where(t < 40, t*.1, 4-(t-40)*.1), np.zeros(100)])
        players = ball[:, None]+[0, .2]
        ball[70:] = np.nan
        y, trace = detect(players, ball, np.ones(100, int), np.ones((100, 1), int), AlwaysTouch())
        self.assertGreater(len(trace['touches']), 0)
        first = trace['touches'][0]['frame']
        self.assertTrue((y[:first] == -1).all())
        self.assertTrue((y[first:] == 1).all())

    def test_features_invariant_to_pitch_translation_and_reflection(self):
        t = np.arange(100)
        ball = np.column_stack([t*.1, np.sin(t*.1)])
        players = ball[:, None]+np.array([[[.5, .2], [-1, .1]]])
        half = np.ones(100, int)
        x = features(players, ball, [30, 60], [0, 1], signals(ball, half))
        reflected_ball, reflected_players = -ball+50, -players+50
        xx = features(reflected_players, reflected_ball, [30, 60], [0, 1], signals(reflected_ball, half))
        np.testing.assert_allclose(x, xx, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    unittest.main()
