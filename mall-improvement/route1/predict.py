"""Route 1 inference on the challenge match. Reads tracking caches only; never a reference file.

    python route1/predict.py            # writes possession-viewer/data/mall-matched.csv

Models (route1/run/*.pkl) were trained on four public DFL games whose tracking was degraded to the
statistics in tracking_diagnostics_report.txt. Setting: route1/run/frozen.json, chosen on a fifth
degraded DFL game and tested on two more.
"""
import os, sys, json, argparse, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import numpy as np
from pathlib import Path
import experiment as E
from decoder_experiment import load
from pipeline.mall.common import CACHE
from pipeline.mall.layer4_possession_model import crossgame as X
from pipeline.run import write_csv


def predict_game(parts, M, names, setting, tag='matched'):
    z, F, L2, L3 = parts
    train_names = json.load(open(os.path.join(E.RUN, 'feature_names.json')))
    assert list(names) == train_names, 'feature columns differ from the ones the models were trained on'
    M = np.asarray(M)
    p = {'frame_' + tag: X.predict(*E.pk(os.path.join(E.RUN, 'frame_{}.pkl'.format(tag))), M).astype('float32')}
    fr = X.cp_candidates(np.zeros(len(z['half'])), z['half'])
    cx, _ = X.cp_design(M, list(names), L2, fr)
    p['cp_frames'] = fr
    p['cp_' + tag] = X.predict_cp(*E.pk(os.path.join(E.RUN, 'cp_{}.pkl'.format(tag))), cx).astype('float32')
    p['dead_' + tag] = E.pk(os.path.join(E.RUN, 'dead_{}.pkl'.format(tag))).predict_proba(E.dead_design(z, F, L2)[0])[:, 1].astype('float32')
    return E.infer(parts, p, setting)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path); a = ap.parse_args()
    setting = json.load(open(os.path.join(E.RUN, 'frozen.json')))['setting']
    z, F, L2, L3, _ = load('tactical')
    d = np.load(Path(CACHE) / 'models' / 'design_tactical.npz')
    pred = predict_game((z, F, L2, L3), d['X'], list(d['names']), setting)
    out = a.output or Path(ROOT).parent / 'possession-viewer' / 'data' / 'mall-matched.csv'
    write_csv(pred, z['half'], out)
    g = z['half'] > 0
    runs = np.flatnonzero(np.diff(pred[g]) != 0)
    seg = np.split(pred[g], runs + 1)
    print('EXPORTED', out, hashlib.sha256(Path(out).read_bytes()).hexdigest())
    print('shares on graded frames  H {:.3f}  A {:.3f}  D {:.3f} | possessions {} | stoppages {}'.format(
        *(np.mean(pred[g] == k) for k in (0, 1, 2)), sum(1 for s in seg if s[0] < 2), sum(1 for s in seg if s[0] == 2)))


if __name__ == '__main__':
    main()
