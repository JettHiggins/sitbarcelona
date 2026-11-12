"""Build the bench: three open Metrica games, degraded to look like the challenge's tracking,
with frame-level answer keys from their event data.

    python -m pipeline.mall.bench.build                 # all three games
    python -m pipeline.mall.bench.build --games 1 3

Writes, per game g (all gitignored derived data):
  cache/layer0_bench{g}.npz                layer 0 output, as for the challenge match
  cache/bench/bench{g}_smart_tagging.npz   simulated set-piece windows (degrade.py)
  cache/bench/bench{g}_gt.csv              answer key, frame,label (H/A/D/X)
  cache/bench/bench{g}_baseline.csv        the official baseline (nearest player, +/- 2 s vote)
                                           on the degraded tracking

Then run layers 1-5 on it with POSSESSION_GAME=bench{g} (bench/score.py does all of it).
"""

import argparse
import os
import time

import numpy as np

from ..common import CACHE
from ..layer0_clean.run import clean
from . import degrade, idsse, labels, metrica

BENCH = os.path.join(CACHE, "bench")


def write_csv(path, letters):
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.write("".join("{},{}\n".format(i, c) for i, c in enumerate(letters)))


def baseline(players_norm, team_of_slot, ball_norm, pitch, window=50):
    """The starter's baseline_smooth50 on arrays: team of the nearest slot, D without a ball,
    majority vote over +/- 2 s."""
    s = np.array(pitch)
    P = players_norm * s
    b = ball_norm * s
    d = np.hypot(*(P - b[:, None, :]).transpose(2, 0, 1))
    d = np.where(np.isnan(d), np.inf, d)
    k = np.argmin(d, axis=1)
    ok = ~np.isnan(b[:, 0]) & np.isfinite(d.min(axis=1))
    codes = np.where(ok, team_of_slot[k], -1)
    hc = np.r_[0, np.cumsum(codes == 0)]
    ac = np.r_[0, np.cumsum(codes == 1)]
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - window), np.minimum(len(codes), i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    return np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))


def build_game(g, verbose=True):
    t0 = time.time()
    G = metrica.load(g)
    lab, dead, graded = labels.answer_key(G)
    n = len(lab)
    Q, tos = degrade.players(G["players"], G["team_of_slot"], seed=1000 + g, pitch=G["pitch"])
    st = degrade.set_piece_tags(G["events"], n, 1000 + g, labels.restart)
    half = np.where(graded, G["period"], 0).astype(np.int8)
    raw = {"players": Q, "ball": G["ball"], "present": np.ones(n, bool)}
    z = clean(raw, tos, G["names"], {}, half, G["pitch"], "bench{}".format(g), verbose=verbose)
    os.makedirs(BENCH, exist_ok=True)
    tag = "bench{}".format(g)
    np.savez_compressed(os.path.join(CACHE, "layer0_{}.npz".format(tag)), **z)
    np.savez_compressed(os.path.join(BENCH, tag + "_smart_tagging.npz"), **st)
    letters = labels.to_letters(lab)
    write_csv(os.path.join(BENCH, tag + "_gt.csv"), letters)
    base = baseline(Q, tos, G["ball"], G["pitch"])
    base = np.where(graded, base, "X")
    write_csv(os.path.join(BENCH, tag + "_baseline.csv"), base)
    if verbose:
        print("{}: {} frames, {} slots after degradation, {:.0f}s".format(tag, n, Q.shape[1],
                                                                     time.time() - t0))
    return tag


def build_dfl(i, verbose=True):
    """One open DFL match (bench/idsse.py): frame-exact key from the feed's ball status and
    possession, players degraded as for Metrica, ball hidden as the challenge's tracker does."""
    t0 = time.time()
    tag = "dfl{}".format(i)
    G = idsse.load(idsse.MATCHES[i - 1])
    n = len(G["label"])
    graded = G["period"] > 0
    lab = np.where(graded, G["label"], -1).astype(np.int8)
    seed = 2000 + i
    Q, tos = degrade.players(G["players"], G["team_of_slot"], seed=seed, pitch=G["pitch"])
    ball = degrade.ball_dropouts(G["ball"], lab == 2, G["period"], seed, G["ball_z"])
    st = degrade.set_piece_tags_from_key(lab, G["ball"], G["pitch"], n, seed)
    half = np.where(graded, G["period"], 0).astype(np.int8)
    raw = {"players": Q, "ball": ball, "present": np.ones(n, bool)}
    z = clean(raw, tos, G["names"], {}, half, G["pitch"], tag, verbose=verbose)
    os.makedirs(BENCH, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, "layer0_{}.npz".format(tag)), **z)
    np.savez_compressed(os.path.join(BENCH, tag + "_smart_tagging.npz"), **st)
    write_csv(os.path.join(BENCH, tag + "_gt.csv"), labels.to_letters(lab))
    base = np.where(graded, baseline(Q, tos, ball, G["pitch"]), "X")
    write_csv(os.path.join(BENCH, tag + "_baseline.csv"), base)
    if verbose:
        print("{}: {} frames, {} slots after degradation, {:.0f}s".format(tag, n, Q.shape[1],
                                                                     time.time() - t0))
    return tag


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--games", type=int, nargs="*", default=[1, 2, 3], help="Metrica games")
    p.add_argument("--dfl", type=int, nargs="*", default=[], help="DFL matches 1-7")
    a = p.parse_args()
    for g in a.games:
        build_game(g)
    for i in a.dfl:
        build_dfl(i)


if __name__ == "__main__":
    main()
