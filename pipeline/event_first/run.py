"""Run the independent event-first ownership algorithm on prepared tracking."""
import hashlib
import json
from pathlib import Path

import numpy as np

from .events import detect
from .sequence import reconstruct


def predict(z, F, dead):
    spans, events = detect(z, F, dead)
    seeds = {}
    for region, (start, end) in enumerate(spans):
        # Prefer the tagged restarting team. Otherwise use an observed near-ball
        # team at the start; never borrow a model's ownership estimate.
        for frame in range(start, min(end, start + 25)):
            tagged = int(F['st_set_piece_team'][frame])
            if F['st_set_piece'][frame] > 0 and tagged in (0, 1):
                seeds[region] = tagged
                break
        if region not in seeds:
            for frame in range(start, min(end, start + 25)):
                distances = np.array([F['d_h'][frame], F['d_a'][frame]], float)
                if np.isfinite(distances).any() and np.nanmin(distances) <= 2.5:
                    seeds[region] = int(np.nanargmin(distances))
                    break
    labels, decisions = reconstruct(len(z['half']), spans, events, seeds)
    return labels, dict(spans=spans, events=events, decisions=decisions)


def build(tag='tactical', diagnostics=None):
    from ..mall.common import load_layer
    # Explicit contract: layers 4/5/6 and their probabilities are never loaded.
    z, F, L3 = [load_layer(layer, tag) for layer in ('layer0', 'layer1', 'layer3')]
    labels, trace = predict(z, F, L3['dead'])
    if diagnostics:
        files = [Path(__file__).parent / name for name in ('events.py', 'sequence.py', 'run.py')]
        fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in files)).hexdigest()
        trace.update(schema=1, game=tag, fps=25, source_fingerprint=fingerprint,
                     ownership='independent event state machine; no possession probabilities',
                     dead_ball='shared Mall layer 3; held fixed for ownership comparison')
        path = Path(diagnostics)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(trace, indent=2, allow_nan=False) + '\n')
    return labels, z['half'], trace
