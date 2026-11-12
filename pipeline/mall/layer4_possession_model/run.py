"""Layer 4 - self-trained possession model.

    python -m pipeline.mall.layer4_possession_model.run     # writes cache/layer4_tactical.npz

Input : cache/layer0..layer3
Output: p_home_players (players only - the evidence through ball gaps; an even blend of a
        moment model and a context model), p_home_full_self (players + ball): out-of-fold
        P(home has the ball) per graded frame from the self-trained model, label (the
        certain-possession frames it learned from: 1 home, 0 away, -1 none).
        p_home_full: P(home has the ball) from the cross-game model (crossgame.py), trained on
        every live frame of the bench games - never on this game.
        cp_frames, cp_p: candidate frames and P(no change / change to home / change to away)
        there, from the cross-game change-point model.
Needs the bench (python -m pipeline.mall.bench.build, then layers 1-3 on each bench game);
python -m pipeline.mall.bench.score does that in the right order. Trained models are cached in
cache/models/ (derived from other games; gitignored).
Gitignored derived data.
"""

import os
import time
import warnings

import numpy as np

from ..common import CACHE, ROOT, TAG, cache_path, load_layer, train_tags
from . import crossgame as X
from .model import PARAMS, design, labels, out_of_fold

MODELS = os.path.join(CACHE, "models")


def bench_tags():
    d = os.path.join(CACHE, "bench")
    return sorted(f[:-7] for f in os.listdir(d) if f.endswith("_gt.csv")) if os.path.isdir(d) else []


def truth(tag):
    with open(os.path.join(CACHE, "bench", tag + "_gt.csv")) as f:
        f.readline()
        g = np.array([line.rstrip("\n").split(",")[1] for line in f])
    return np.where(g == "H", 0, np.where(g == "A", 1, np.where(g == "D", 2, -1))).astype(np.int8)


def full_matrix(tag):
    """Cross-game feature matrix of one game, cached (derived data; gitignored)."""
    path = os.path.join(MODELS, "design_{}.npz".format(tag))
    if os.path.isfile(path):
        d = np.load(path)
        return d["X"], list(d["names"])
    z, F, L2 = (load_layer(x, tag) for x in ("layer0", "layer1", "layer2"))
    S, base = X.sources(z, F, dyn=False)
    M, names = X.full_design(S, base, F, L2)
    os.makedirs(MODELS, exist_ok=True)
    np.savez(path, X=M, names=np.array(names))
    return M, names


def cross_game(verbose=True):
    """p_home_full, cp_frames, cp_p for the current game from models trained on the others."""
    t0 = time.time()
    from .. import config as C
    train = train_tags(TAG, C.POSS_TRAIN_PREFIXES)
    if not train:
        raise IOError("no bench games: run python -m pipeline.mall.bench.build (see bench/README.md)")
    key = "+".join(train)
    M, names = full_matrix(TAG)
    ys = {t: truth(t) for t in train}
    stride = max(3, len(train))                 # more games, same number of rows
    step = max(1, len(train) // 3)              # ... and of change-point candidates

    def cp_data(t, mat):
        half = load_layer("layer0", t)["half"]
        # training games: candidates on truly live frames, so the cached model does not depend
        # on the dead-ball layer's output for those games
        dead = (ys[t] == 2) if t in ys else load_layer("layer3", t)["dead"]
        fr = X.cp_candidates(dead, half)
        Xc, cn = X.cp_design(mat, names, load_layer("layer2", t), fr)
        return fr, Xc, cn

    def fit_frame():
        Xs, yy = [], []
        for t in train:                         # one game's matrix in memory at a time
            mat = full_matrix(t)[0]
            y = np.where(ys[t] == 0, 1, np.where(ys[t] == 1, 0, -1))
            keep = np.flatnonzero(y >= 0)[::stride]
            Xs.append(mat[keep])
            yy.append(y[keep])
            del mat
        return X.fit(Xs, yy, names, stride=1)

    def fit_cp():
        Xs, yy, cn = [], [], None
        for t in train:
            fr_t, Xc_t, cn = cp_data(t, full_matrix(t)[0])
            Xs.append(Xc_t[::step])
            yy.append(X.cp_labels(ys[t], fr_t)[::step])
        return X.fit_cp(Xs, yy, cn)

    frame = X.cached(os.path.join(MODELS, "frame_{}.pkl".format(key)), fit_frame)
    p_full = X.predict(frame[0], frame[1], M).astype(np.float32)
    fr, Xc, cn = cp_data(TAG, M)
    cp = X.cached(os.path.join(MODELS, "cp_{}.pkl".format(key)), fit_cp)
    cp_p = X.predict_cp(cp[0], cp[1], Xc).astype(np.float32)
    if verbose:
        print("cross-game models trained on {}  ({:.0f}s)".format(key, time.time() - t0))
    return {"p_home_full": p_full, "cp_frames": fr.astype(np.int32), "cp_p": cp_p}

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build(verbose=True, with_passes=True, params=None):
    t0 = time.time()
    z = load_layer("layer0")
    F = load_layer("layer1")
    L2 = load_layer("layer2")
    L3 = load_layer("layer3")
    y, seg = labels(z, F, L2, L3, with_passes=with_passes)
    params = params or PARAMS
    out = {"label": y, "label_segment": seg}
    from .. import config as C
    if C.SEQ_W_PLAYERS:
        p = {}
        for view in ("players", "players_context", "full"):
            X, _ = design(F, view)
            p[view] = out_of_fold(X, y, z["half"], params)
        out["p_home_players"] = 0.5 * p["players"] + 0.5 * p["players_context"]
        out["p_home_full_self"] = p["full"]
    else:
        # The decoder gives the self-trained views no weight (SEQ_W_PLAYERS = 0, chosen on the
        # bench), so they are not trained: neutral evidence, identical decoded output.
        out["p_home_players"] = np.full(len(y), 0.5, np.float32)
        out["p_home_full_self"] = np.full(len(y), 0.5, np.float32)
    out.update(cross_game(verbose))
    if verbose:
        print("labels: {} frames ({:.0%} home)  ({:.0f}s)".format(
            int((y >= 0).sum()), float(np.mean(y[y >= 0] == 1)), time.time() - t0))
    return out


def main():
    out = build()
    path = cache_path("layer4")
    np.savez_compressed(path, **out)
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
