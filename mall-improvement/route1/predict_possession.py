"""Route 1 possession models with the existing (referee-supervised) stoppage detector.

    python route1/predict_possession.py     # writes possession-viewer/data/mall-matched-possession.csv

Stoppages: exactly as predict_referee_supervised.py. Who has the ball: route 1's frame and
change-point models (trained on degraded DFL games). Reads tracking caches only.
"""
import os, sys, json, argparse, hashlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
import numpy as np
from pathlib import Path
import experiment as E
from referee_supervised import R, design
from decoder_experiment import load
from pipeline.mall.common import CACHE
from pipeline.mall.layer4_possession_model import crossgame as X
import idsse_experiment as I
from pipeline.run import write_csv


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--output', type=Path); a = ap.parse_args()
    selected = json.loads((R / 'frozen.json').read_text())
    z, F, L2, L3, _ = load('tactical')
    mat, _, parts = design('tactical')
    dead = I.model(R / 'final_firsthalves.pkl').predict_proba(mat)[:, 1].astype('float32')
    d = np.load(Path(CACHE) / 'models' / 'design_tactical.npz'); M = d['X']; names = list(d['names'])
    assert names == json.load(open(os.path.join(E.RUN, 'feature_names.json'))), 'feature columns differ'
    frame = X.predict(*E.pk(os.path.join(E.RUN, 'frame_matched.pkl')), M).astype('float32')
    fr = X.cp_candidates(np.zeros(len(dead)), z['half'])
    cx, _ = X.cp_design(M, names, L2, fr)
    cp = X.predict_cp(*E.pk(os.path.join(E.RUN, 'cp_matched.pkl')), cx).astype('float32')
    p = dict(old=frame, new=frame, cp_frames=fr, cp_p=cp, dead=dead)
    pred = I.infer('tactical', selected['setting'], parts, p)
    out = a.output or Path(ROOT).parent / 'possession-viewer' / 'data' / 'mall-matched-possession.csv'
    write_csv(pred, z['half'], out)
    g = z['half'] > 0
    seg = np.split(pred[g], np.flatnonzero(np.diff(pred[g]) != 0) + 1)
    print('EXPORTED', out, hashlib.sha256(Path(out).read_bytes()).hexdigest())
    print('shares on graded frames  H {:.3f}  A {:.3f}  D {:.3f} | possessions {} | stoppages {}'.format(
        *(np.mean(pred[g] == k) for k in (0, 1, 2)), sum(1 for s in seg if s[0] < 2), sum(1 for s in seg if s[0] == 2)))


if __name__ == '__main__':
    main()
