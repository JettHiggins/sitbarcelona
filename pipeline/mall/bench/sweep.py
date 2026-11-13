"""Evaluate postprocessing settings on benchmark caches, without overwriting them.

The JSON settings file is a list of config overrides. Public labels cannot be
used in this selection tool. Reports retain every setting and per-game score.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import config as C
from ..common import CACHE, load_layer
from ..layer3_dead_ball.deadball import detect
from ..layer5_sequence_decoder.decoder import decode
from .ensemble import read_prediction
from .score import summary, truth_codes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--settings", type=Path, required=True)
    p.add_argument("--games", nargs="+", default=["bench1", "bench2", "bench3", "dfl1", "dfl4"])
    p.add_argument("--prediction", help="optional cached full-frame trajectory prefix")
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    if "tactical" in a.games:
        p.error("selection must use benchmark games")
    settings = json.loads(a.settings.read_text(encoding="utf-8-sig"))
    keys = set().union(*(s.keys() for s in settings))
    if any(not k.startswith(("SEQ_", "DEAD_")) or not hasattr(C, k) for k in keys):
        p.error("unknown postprocessing setting")
    baseline = {k: getattr(C, k) for k in keys}
    rows = []
    for tag in a.games:
        z, f, l2, l3, l4 = (dict(load_layer(layer, tag)) for layer in
                            ("layer0", "layer1", "layer2", "layer3", "layer4"))
        gt = truth_codes(tag)
        if a.prediction:
            ph, pc = read_prediction(Path(CACHE) / f"{a.prediction}_{tag}.npz", len(z["half"]))
            # Keep dense change evidence available if the dead-ball mask changes.
            fr = np.arange(0, len(ph), 3, dtype=np.int32)
            l4.update(p_home_full=ph, cp_frames=fr, cp_p=pc[fr])
        for settings_id, overrides in enumerate(settings):
            for k, v in dict(baseline, **overrides).items():
                setattr(C, k, v)
            dead = l3
            if any(k.startswith("DEAD_") for k in overrides):
                dead = dict(l3, dead=detect(z, f, l2, l3.get("cue_p_learned"))[0])
            labels = decode(z, l2, dead, l4)
            scores = summary(gt, np.where(z["half"] > 0, labels, 3))
            rows.append(dict(game=tag, settings_id=settings_id, **scores))
            print(tag, settings_id, json.dumps(overrides), f"spell={scores['spell']:.4f}", flush=True)
        # Save completed games so a long sweep is still inspectable if interrupted.
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(dict(settings=settings, rows=rows), indent=2), encoding="utf-8")
    means = [dict(settings_id=i, settings=s,
                  spell=float(np.mean([r["spell"] for r in rows if r["settings_id"] == i])),
                  macro=float(np.mean([r["macro"] for r in rows if r["settings_id"] == i])))
             for i, s in enumerate(settings)]
    for row in sorted(means, key=lambda r: -r["spell"]):
        print(json.dumps(row), flush=True)
    a.report.write_text(json.dumps(dict(settings=settings, rows=rows, means=means), indent=2),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
