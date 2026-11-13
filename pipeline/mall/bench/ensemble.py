"""Compare cached trajectory predictions without replacing any pipeline output.

Example: python -m pipeline.mall.bench.ensemble --members seqB seqS1 seqS2
All members must have been trained without either evaluation game. Their archives
contain p_home (one value per frame) and p_change (three probabilities per frame).
The public challenge window is deliberately excluded from this selection command.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..common import CACHE, load_layer
from ..layer5_sequence_decoder.decoder import decode
from .score import summary, truth_codes


def read_prediction(path, n):
    with np.load(path, allow_pickle=False) as data:
        ph = data["p_home"].astype(np.float64)
        pc = data["p_change"].astype(np.float64)
    if ph.shape != (n,) or pc.shape != (n, 3):
        raise ValueError(f"{path}: prediction/frame alignment mismatch")
    if not (np.isfinite(ph).all() and np.isfinite(pc).all()):
        raise ValueError(f"{path}: non-finite probabilities")
    if np.any((ph < 0) | (ph > 1)) or np.any((pc < 0) | (pc > 1)):
        raise ValueError(f"{path}: probabilities outside [0, 1]")
    if not np.allclose(pc.sum(axis=1), 1, atol=1e-5):
        raise ValueError(f"{path}: change probabilities do not sum to one")
    return ph, pc


def compare(games, members):
    rows, inputs = {}, {}
    for tag in games:
        if tag == "tactical":
            raise ValueError("Use held-out benchmark games for ensemble selection")
        z, l2, l3, l4 = (dict(load_layer(layer, tag))
                         for layer in ("layer0", "layer2", "layer3", "layer4"))
        n = len(z["half"])
        gt = truth_codes(tag)
        predictions = {}
        for name in members:
            path = Path(CACHE) / f"{name}_{tag}.npz"
            predictions[name] = read_prediction(path, n)
            inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        predictions["ensemble"] = tuple(np.mean(a, axis=0) for a in
                                         zip(*predictions.values()))
        rows[tag] = {}
        for name in ["current", *predictions]:
            evidence = dict(l4)
            if name != "current":
                ph, pc = predictions[name]
                evidence.update(p_home_full=ph, cp_p=pc[l4["cp_frames"]])
            labels = decode(z, l2, l3, evidence)
            pred = np.where(z["half"] > 0, labels, 3)
            rows[tag][name] = summary(gt, pred)
            print(f"{tag:9s} {name:12s} spellF1={rows[tag][name]['spell']:.4f} "
                  f"macroF1={rows[tag][name]['macro']:.4f} "
                  f"spells={rows[tag][name]['n_pred']}/{rows[tag][name]['n_gt']}",
                  flush=True)
    means = {name: {metric: float(np.mean([r[name][metric] for r in rows.values()]))
                    for metric in ("spell", "macro", "acc")}
             for name in next(iter(rows.values()))}
    return {"members": members, "games": games, "inputs_sha256": inputs,
            "note": "Current uses leave-one-game-out models; scratch members may use a "
                    "smaller training set. Compare ensemble against its own members too.",
            "rows": rows, "means": means}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--members", nargs="+", required=True)
    parser.add_argument("--games", nargs="+", default=["bench1", "dfl1"])
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.members)) != len(args.members) or any(
            name in {"ensemble", "current"} or Path(name).name != name for name in args.members):
        parser.error("member names must be unique cache prefixes, excluding current/ensemble")
    result = compare(args.games, args.members)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print("MEANS", json.dumps(result["means"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
