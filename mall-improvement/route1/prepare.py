"""Build layers 1-3 and the model inputs for an outside game, as delivered ('clean') or degraded
to look like the challenge match's tracking ('matched', see degrade.py).

    python route1/prepare.py matched J03WOH J03WOY ...
"""
import os, sys, json, time, warnings
os.environ.setdefault('OMP_NUM_THREADS', '1'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
warnings.filterwarnings('ignore')
from pipeline.mall.layer0_clean.ball import clean_ball
from pipeline.mall.layer1_features.features import build_features
from pipeline.mall.layer2_touches.touches import detect as touches
from pipeline.mall.layer3_dead_ball.deadball import detect as deadball
from pipeline.mall.layer4_possession_model import crossgame as X
import degrade as G

SRC = os.path.join(ROOT, 'experiments', 'idsse')
DATA = os.path.join(HERE, 'data')
IDS = ['J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ', 'J03WR9', 'J03WMX', 'J03WN1']
PARAMS = dict(G.DEFAULT, sigma=0.25, tail=1.2, smooth=1.5, hide_min=60)


def build(mid, sensor, seed_base=31000):
    out = os.path.join(DATA, sensor, mid)
    if os.path.isfile(os.path.join(out, 'done.json')):
        return
    t0 = time.time()
    z = dict(np.load(os.path.join(SRC, mid, 'layer0.npz')))
    F0 = np.load(os.path.join(SRC, mid, 'layer1.npz'))
    st = {k: F0[k] for k in ('st_possessor', 'st_set_piece', 'st_set_piece_team')}
    labels = np.load(os.path.join(SRC, mid, 'labels.npy'))
    audit = {'sensor': sensor}
    if sensor == 'matched':
        rng = np.random.default_rng(seed_base + IDS.index(mid))
        raw = G.ball(z['ball_xy_raw'].astype(float), labels, z['half'], z['pitch'], rng, PARAMS)
        clean = clean_ball(raw, z['half'])
        z['ball_xy_raw'] = raw.astype('float32')
        for k, v in clean.items():
            if 'ball_' + k in z:
                z['ball_' + k] = v.astype('float32') if v.dtype.kind == 'f' else v
        z = G.players({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in z.items()}, rng, PARAMS)
        audit.update(params=PARAMS, ball_seen=float(np.isfinite(raw[z['half'] > 0, 0]).mean()))
    F, _ = build_features(z, st)
    ev, states = touches(z, {k: v.astype(float) if v.dtype.kind == 'f' else v for k, v in F.items()})
    L2 = {'events': ev, **states}
    dead, llr, parts = deadball(z, F, L2)
    L3 = {'dead': dead, 'llr': llr}
    os.makedirs(out, exist_ok=True)
    for name, data in (('layer0', z), ('layer1', F), ('layer2', L2), ('layer3', L3)):
        np.savez_compressed(os.path.join(out, name + '.npz'), **data)
    S, base = X.sources(z, F, dyn=False)
    M, names = X.full_design(S, base, F, L2)
    np.save(os.path.join(out, 'design.npy'), M.astype('float32'))
    json.dump(list(names), open(os.path.join(out, 'names.json'), 'w'))
    audit['seconds'] = round(time.time() - t0)
    json.dump(audit, open(os.path.join(out, 'done.json'), 'w'))
    print('BUILT', sensor, mid, audit.get('ball_seen'), audit['seconds'], 's', flush=True)


if __name__ == '__main__':
    for mid in sys.argv[2:]:
        build(mid, sys.argv[1])
