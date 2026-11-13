"""Reproducible trajectory experiments, excluding all test games from training.

Writes separate caches and a manifest; leaves production models and CSVs alone.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from ..common import CACHE, bench_tags, load_layer
from ..layer4_possession_model import sequence as Q
from ..layer4_possession_model.run import sequence_features, truth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--test", nargs="+", default=["bench1", "dfl1"])
    parser.add_argument("--steps", type=int, default=Q.STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spell-balance", type=float, default=0.0)
    parser.add_argument("--network", choices=["gru", "temporal"], default="gru")
    args = parser.parse_args()
    if Path(args.name).name != args.name or not args.name or args.steps < 2:
        parser.error("use a simple name and at least two steps")
    tags = bench_tags()
    if not args.test or not set(args.test).issubset(tags):
        parser.error("test games must be existing benchmark games")
    train = [tag for tag in tags if tag not in args.test]
    if not train:
        parser.error("no training games remain")
    import torch
    torch.set_num_threads(2)
    if args.network == "temporal":
        from ..layer4_possession_model import temporal
        Q.build_net = lambda: temporal.build_net(Q.DIM, Q.HIDDEN)
    started = time.time()
    path = Path(CACHE) / "models" / (args.name + ".pt")
    manifest_path = path.with_suffix(".json")
    if path.exists() or manifest_path.exists():
        parser.error(f"experiment {args.name} already exists; use a new name")
    manifest = dict(vars(args), train=train, status="training", started=started,
                    torch=torch.__version__, device="cuda" if torch.cuda.is_available() else "cpu",
                    source_sha256=hashlib.sha256(Path(Q.__file__).read_bytes()).hexdigest(),
                    architecture={k: getattr(Q, k) for k in
                                  ("DIM", "HIDDEN", "LAYERS", "WINDOW", "BATCH", "LR")})
    if args.network == "temporal":
        manifest["temporal_source_sha256"] = hashlib.sha256(Path(temporal.__file__).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest), flush=True)
    games = [(sequence_features(t), truth(t), load_layer("layer0", t)["half"]) for t in train]
    state = Q.train(games, seed=args.seed, steps=args.steps, verbose=True,
                    spell_balance=args.spell_balance)
    torch.save(state, str(path) + ".tmp")
    os.replace(str(path) + ".tmp", path)
    del games
    for tag in args.test:
        ph, pc = Q.predict(state, sequence_features(tag), load_layer("layer0", tag)["half"])
        dest = Path(CACHE) / f"{args.name}_{tag}.npz"
        with open(str(dest) + ".tmp", "wb") as f:
            np.savez(f, p_home=ph, p_change=pc)
        os.replace(str(dest) + ".tmp", dest)
        print(f"wrote {dest}", flush=True)
    manifest.update(status="complete", seconds=time.time() - started,
                    model_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"complete in {manifest['seconds']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
