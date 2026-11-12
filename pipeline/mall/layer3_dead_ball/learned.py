"""Learned dead-ball evidence from the players alone: P(ball not in play) per frame.

The rule cues in deadball.py read the ball: beyond a line, lost, at rest. That fails when the ball
is lost in open play or stays tracked while play is stopped. What the 22 players do - walk,
stand, gather in a box, one of them waiting at a line - marks a stoppage either way, and it
is the one signal that looks the same in every tracking feed. So this model sees no ball input
at all, and is trained on other games that have a frame-level answer key:

  three Metrica games   key rebuilt from the event log (bench/labels.py)
  seven DFL games       key from the league feed's own ball status (bench/idsse.py)

  bench game g      trained on the nine other games, never on g
  challenge match   trained on all ten; no label of the challenge match is involved

Why no ball input, measured rather than assumed: Metrica's ball is unknown on 92-99% of dead
frames and the DFL ball is frozen (speed exactly 0) on all of them, while the challenge's ball is
tracked and moving on most dead frames of its public window. A model given the ball scored
dead-ball F1 0.98 on Metrica and 0.44 on that window. Set-piece windows are left out for a
similar reason: the bench simulates them from the answer key more completely than the real
tagging covers the challenge match.

Features: layer 4's players-only context view (team shape, lines, pressing, movement, at the
frame, as 1.5 / 5 / 30 s means and 5 s before and after) plus stoppage signals - the share of
players standing, walking and running, players at or beyond a line, players in the penalty
areas - looking up to 20 s both ways.
"""

import os
import pickle

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from ..common import CACHE, TAG, cache_path, load_layer, train_tags
from ..layer4_possession_model import crossgame as X

MODELS = os.path.join(CACHE, "models")
WINDOWS = X.WINDOWS + ((-500, -200), (200, 500))
PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
              l2_regularization=1.0, random_state=0)
STRIDE = 10         # ~14k frames a game; with ten games and the swapped copy ~280k rows


def truth(tag):
    with open(os.path.join(CACHE, "bench", tag + "_gt.csv")) as f:
        f.readline()
        g = np.array([line.strip().split(",")[1] for line in f])
    return np.where(g == "H", 0, np.where(g == "A", 1, np.where(g == "D", 2, -1))).astype(np.int8)


def stoppage_signals(z, F):
    """name -> per-frame array; players only, symmetric in the two teams."""
    import warnings
    xy = z["player_xy"].astype(float)
    vel = z["player_vel"].astype(float)
    on = z["player_team"] >= 0
    L, W = (float(v) for v in z["pitch"])
    sp = np.where(on, np.hypot(vel[..., 0], vel[..., 1]), np.nan)
    x = np.where(on, xy[..., 0], np.nan)
    y = np.where(on, xy[..., 1], np.nan)
    n_on = np.maximum(on.sum(axis=1), 1)
    S = {}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        S["share_standing"] = np.nansum(sp < 0.5, axis=1) / n_on
        S["share_walking"] = np.nansum(sp < 2.0, axis=1) / n_on
        S["share_running"] = np.nansum(sp > 4.0, axis=1) / n_on
        S["speed_max"] = np.nanmax(sp, axis=1)
        S["speed_p25"] = np.nanpercentile(sp, 25, axis=1)
        S["speed_p75"] = np.nanpercentile(sp, 75, axis=1)
        to_touch = np.minimum(y, W - y)                      # < 0 beyond a touchline
        to_goal = np.minimum(x, L - x)
        S["touch_min"] = np.nanmin(to_touch, axis=1)         # a thrower stands on / behind it
        S["goal_min"] = np.nanmin(to_goal, axis=1)
        S["n_outside"] = np.nansum((to_touch < -0.3) | (to_goal < -0.3), axis=1).astype(float)
        slow_at_line = (to_touch < 1.0) & (sp < 1.5)
        S["n_slow_at_touch"] = np.nansum(slow_at_line, axis=1).astype(float)
        corner = np.hypot(to_touch, to_goal)
        S["corner_min"] = np.nanmin(corner, axis=1)
        in_box = (to_goal < 16.5) & (np.abs(y - W / 2) < 20.16)
        left = in_box & (x < L / 2)
        nl, nr = np.nansum(left, axis=1), np.nansum(in_box & ~left, axis=1)
        S["n_box_max"] = np.maximum(nl, nr).astype(float)
        S["n_box_min"] = np.minimum(nl, nr).astype(float)
        S["x_spread_all"] = np.nanstd(x, axis=1)
        S["y_spread_all"] = np.nanstd(y, axis=1)
    S["n_players"] = on.sum(axis=1).astype(float)
    S["mean_speed_all"] = F["mean_speed_all"].astype(float)
    return S


def matrix(tag):
    """(X, names) for one game: players only. Cached (derived data; gitignored)."""
    path = os.path.join(MODELS, "deadp_design_{}.npz".format(tag))
    newest = max(os.path.getmtime(cache_path(x, tag)) for x in ("layer0", "layer1"))
    if os.path.isfile(path) and os.path.getmtime(path) >= newest:    # not stale
        d = np.load(path)
        return d["X"], list(d["names"])
    z, F = (load_layer(x, tag) for x in ("layer0", "layer1"))
    S, base = X.sources(z, F, dyn=True)
    M, names = X.design(S, base, **X.CONTEXT)
    names = list(names)
    cols = []
    for k, x in stoppage_signals(z, F).items():
        cols.append(x)
        names.append("stop_" + k)
        for lo, hi in WINDOWS:
            cols.append(X.window_mean(x, lo, hi))
            names.append("stop_{}@w{}_{}".format(k, lo, hi).replace("-", "m"))
    M = np.column_stack([M, np.column_stack(cols).astype(np.float32)])
    os.makedirs(MODELS, exist_ok=True)
    np.savez(path, X=M, names=np.array(names))
    return M, names


def fit(tags):
    Xs, ys = [], []
    names = None
    for t in tags:
        M, names = matrix(t)
        y = truth(t)
        keep = np.flatnonzero(y >= 0)[::STRIDE]
        Xs.append(M[keep])
        ys.append((y[keep] == 2).astype(np.int8))
        del M
    idx, sgn = X.swap_index(names)
    n = sum(len(x) for x in Xs)
    A = np.empty((2 * n, len(names)), np.float32)     # every row, and its home/away swap
    a = 0
    for x in Xs:
        A[a:a + len(x)] = x
        A[n + a:n + a + len(x)] = x[:, idx] * sgn
        a += len(x)
    del Xs
    y = np.concatenate(ys)
    # no early stopping: it would hold out (and copy) a validation split of a matrix this wide
    clf = HistGradientBoostingClassifier(early_stopping=False,
                                         verbose=int(os.environ.get("MALL_VERBOSE", 0)), **PARAMS)
    clf.fit(A, np.r_[y, y])
    return clf, (idx, sgn)


def p_dead(tag=TAG):
    """P(dead) per frame of `tag` from a model that never saw it; None without a bench."""
    train = train_tags(tag)
    if not train:
        return None
    path = os.path.join(MODELS, "deadp_{}.pkl".format("+".join(train)))
    if os.path.isfile(path):
        with open(path, "rb") as f:
            clf, (idx, sgn) = pickle.load(f)
    else:
        clf, (idx, sgn) = fit(train)
        with open(path, "wb") as f:
            pickle.dump((clf, (idx, sgn)), f)
    M, _ = matrix(tag)
    out = np.empty(len(M), np.float32)
    for a in range(0, len(M), 50000):
        b = M[a:a + 50000]
        out[a:a + 50000] = 0.5 * (clf.predict_proba(b)[:, 1] + clf.predict_proba(b[:, idx] * sgn)[:, 1])
    return out
