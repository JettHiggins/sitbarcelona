"""Layer 3 - dead-ball detection.

    python -m pipeline.mall.layer3_dead_ball.run        # writes cache/layer3_tactical.npz

Input : cache/layer0, layer1, layer2
Output: dead (per frame, 1 = ball not in play), dead_llr (the evidence), one array per cue.
Gitignored derived data.
"""

import os
import time
import warnings

import numpy as np

from ..common import ROOT, cache_path, load_layer
from .deadball import detect

warnings.filterwarnings("ignore", category=RuntimeWarning)


def build(verbose=True):
    t0 = time.time()
    z = load_layer("layer0")
    F = load_layer("layer1")
    L2 = load_layer("layer2")
    dead, llr, parts, stages = detect(z, F, L2, trace=True)
    g = z["half"] > 0
    if verbose:
        print("dead on {:.1%} of graded frames  ({:.0f}s)".format(dead[g].mean(), time.time() - t0))
    return dead, llr, parts, stages


def main():
    dead, llr, parts, stages = build()
    out = cache_path("layer3")
    np.savez_compressed(out, dead=dead, dead_llr=llr.astype(np.float32),
                        hard_kickoff=stages["hard_kickoff"].astype(np.int8),
                        **{"cue_" + k: v.astype(np.float32) for k, v in parts.items()},
                        **{"stage_" + k: v.astype(np.int8) for k, v in stages.items()
                           if k != "hard_kickoff"})
    print("wrote", os.path.relpath(out, ROOT))


if __name__ == "__main__":
    main()
