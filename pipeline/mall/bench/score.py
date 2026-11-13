"""Score the Mall layers on the bench games (leave-one-game-out) and, as a read-only sanity
check, on the public five-minute window of the challenge match.

    python -m pipeline.mall.bench.score                    # rerun layers 3-5 everywhere, then score
    python -m pipeline.mall.bench.score --from 5           # only redecode
    python -m pipeline.mall.bench.score --from 0 --note "what changed"

The bench is the selection set: changes are kept when they help the three bench games, whose
answer keys cover full matches. The public window (17 possessions) is printed beside it but is
never used to choose a setting - see RULES.md and EVALUATION.md.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

from ...vendor.challenge import possession_metrics as pm
from ..common import CACHE, REPORTS, ROOT

LAYERS = ["layer1_features", "layer2_touches", "layer3_dead_ball", "layer4_possession_model",
          "layer5_sequence_decoder"]
def all_games():
    d = os.path.join(CACHE, "bench")
    tags = sorted(f[:-7] for f in os.listdir(d) if f.endswith("_gt.csv")) if os.path.isdir(d) else []
    return tags + ["tactical"]


GAMES = all_games()
PUBLIC_GT = os.environ.get("MTAH_PUBLIC_GT") or os.path.join(
    os.path.dirname(ROOT), "more-than-a-hack-2027", "evaluation", "gt",
    "BAR-ATH_possession_gt_public.csv")


def run_layers(tag, first):
    env = dict(os.environ, POSSESSION_GAME=tag)
    if tag == "tactical" and first <= 1:
        design = os.path.join(CACHE, "models", "design_tactical.npz")
        if os.path.isfile(design):
            os.remove(design)
    for i, layer in enumerate(LAYERS, start=1):
        if i < first:
            continue
        subprocess.run([sys.executable, "-m", "pipeline.mall.{}.run".format(layer)], cwd=ROOT,
                       env=env, check=True, stdout=subprocess.DEVNULL)


def truth_codes(tag):
    path = PUBLIC_GT if tag == "tactical" else os.path.join(CACHE, "bench", tag + "_gt.csv")
    return pm.label_codes(pm.read_label_csv(path))


def predicted_codes(tag, layer="layer5", key="label"):
    lab = np.load(os.path.join(CACHE, "{}_{}.npz".format(layer, tag)))[key].astype(int)
    half = np.load(os.path.join(CACHE, "layer0_{}.npz".format(tag)))["half"]
    return np.where(half > 0, lab, pm.X)


def summary(gt, pred):
    r = pm.evaluate(gt, pred)
    return {"spell": r["spell_f1_mean"], "macro": r["macro_f1"], "acc": r["accuracy"],
            "dead_f1": r["per_class"]["D"]["f1"], "team": r["team_acc_both_alive"],
            "n_pred": r["n_pred_spells"], "n_gt": r["n_gt_spells"]}


def score_all(games=GAMES):
    return {tag: summary(truth_codes(tag), predicted_codes(tag)) for tag in games}


def show(rows):
    keys = ("spell", "macro", "acc", "dead_f1", "team")
    print("{:9s} {:>7s} {:>7s} {:>7s} {:>7s} {:>7s}  spells".format(
        "game", "spellF1", "macroF1", "acc", "deadF1", "team"))
    line = "{:9s} {:7.4f} {:7.4f} {:7.4f} {:7.4f} {:7.4f}"
    for tag, r in rows.items():
        print(line.format(tag, *(r[k] for k in keys)) + "  {}/{}".format(r["n_pred"], r["n_gt"]))
    for name, pre in (("metrica", "bench"), ("dfl", "dfl")):
        grp = [r for t, r in rows.items() if t.startswith(pre)]
        if grp:
            print(line.format(name + " avg", *(float(np.mean([r[k] for r in grp])) for k in keys)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="first", type=int, default=3,
                   help="first layer to rerun (1-5); 6 = score existing caches only")
    p.add_argument("--games", nargs="+", default=GAMES)
    p.add_argument("--note", default="")
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args()
    t0 = time.time()
    if a.first <= 5:
        # games are independent (the model cache is keyed by train set); a few at a time
        pending, running, failed = list(a.games), [], False
        while pending or running:
            while pending and len(running) < a.jobs:
                tag = pending.pop(0)
                running.append(subprocess.Popen(
                    [sys.executable, "-c",
                     "from pipeline.mall.bench.score import run_layers; run_layers({!r}, {})".format(
                         tag, a.first)], cwd=ROOT))
            failed |= bool(running.pop(0).wait())
        if failed:
            raise SystemExit("a layer failed")
    rows = score_all(a.games)
    show(rows)
    os.makedirs(os.path.join(REPORTS, "bench"), exist_ok=True)
    with open(os.path.join(REPORTS, "bench", "history.jsonl"), "a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "note": a.note,
                            "rows": rows}) + "\n")
    print("({:.0f}s)".format(time.time() - t0))


if __name__ == "__main__":
    main()
