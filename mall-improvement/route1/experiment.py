"""Route 1: train the possession, change-point and dead-ball models on outside games degraded to
look like the challenge match's tracking, and score every variant on held-out degraded games.

    python route1/experiment.py train      # models on J03WOH, J03WOY, J03WPY, J03WQQ (matched)
    python route1/experiment.py screen     # choose the blend on J03WR9 (matched)
    python route1/experiment.py test       # frozen choice on J03WMX, J03WN1 (matched and clean)

No file from the challenge match is read.
"""
import os, sys, json, pickle, time, warnings
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, HERE]
warnings.filterwarnings('ignore')
from sklearn.ensemble import HistGradientBoostingClassifier
from pipeline.mall.layer4_possession_model import crossgame as X
from pipeline.mall.layer3_dead_ball import deadball as B
from pipeline.mall.layer5_sequence_decoder.decoder import logit
from decoder_experiment import predict, quick
from deadball_experiment import design as dead_design, PARAMS as DEAD_PARAMS

DATA = os.path.join(HERE, 'data'); RUN = os.path.join(HERE, 'run'); os.makedirs(RUN, exist_ok=True)
SRC = os.path.join(ROOT, 'experiments', 'idsse')
TRAIN = ['J03WOH', 'J03WOY', 'J03WPY', 'J03WQQ']; VALID = ['J03WR9']; TEST = ['J03WMX', 'J03WN1']
BENCH_FRAME = os.path.join(ROOT, 'pipeline', 'cache', 'mall', 'models', 'frame_bench1+bench2+bench3.pkl')
BENCH_CP = os.path.join(ROOT, 'pipeline', 'cache', 'mall', 'models', 'cp_bench1+bench2+bench3.pkl')
IDSSE_FRAME = os.path.join(SRC, 'experiment', 'frame.pkl'); IDSSE_DEAD = os.path.join(SRC, 'experiment', 'dead.pkl')


def pk(path, obj=None):
    if obj is None:
        with open(path, 'rb') as f:
            return pickle.load(f)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load(sensor, mid):
    d = os.path.join(DATA, sensor, mid)
    parts = [dict(np.load(os.path.join(d, k + '.npz'))) for k in ('layer0', 'layer1', 'layer2', 'layer3')]
    return parts, np.load(os.path.join(d, 'design.npy'), mmap_mode='r'), json.load(open(os.path.join(d, 'names.json')))


def labels(mid):
    return np.load(os.path.join(SRC, mid, 'labels.npy'))


def train(sensors=('matched',), tag='matched'):
    fp, cp, dp = (os.path.join(RUN, '{}_{}.pkl'.format(k, tag)) for k in ('frame', 'cp', 'dead'))
    if not os.path.isfile(fp):
        t0 = time.time(); mats, ys = [], []
        for s in sensors:
            for mid in TRAIN:
                _, M, names = load(s, mid); y = labels(mid)
                y = np.where(y == 0, 1, np.where(y == 1, 0, -1)); ok = y >= 0
                st = 3 * len(sensors)
                mats.append(np.asarray(M[ok][::st])); ys.append(y[ok][::st])
        pk(fp, X.fit(mats, ys, names, stride=1)); print('FRAME', tag, round(time.time() - t0), 's', flush=True); del mats
    if not os.path.isfile(cp):
        t0 = time.time(); mats, ys = [], []
        for s in sensors:
            for mid in TRAIN:
                (z, F, L2, L3), M, names = load(s, mid); y = labels(mid)
                truth = np.where(y < 0, 2, y)
                fr = X.cp_candidates((truth == 2).astype(np.int8), z['half'])
                fr = fr[::len(sensors)]
                cx, cn = X.cp_design(M, names, L2, fr)
                mats.append(cx); ys.append(X.cp_labels(truth, fr))
        pk(cp, X.fit_cp(mats, ys, cn)); print('CP', tag, round(time.time() - t0), 's', flush=True); del mats
    if not os.path.isfile(dp):
        t0 = time.time(); mats, ys = [], []
        for s in sensors:
            for mid in TRAIN:
                (z, F, L2, L3), _, _ = load(s, mid); y = labels(mid); ok = y >= 0
                a = dead_design(z, F, L2)[0]
                mats.append(a[ok][::5 * len(sensors)]); ys.append((y[ok] == 2)[::5 * len(sensors)])
        clf = HistGradientBoostingClassifier(**DEAD_PARAMS); clf.fit(np.vstack(mats), np.concatenate(ys))
        pk(dp, clf); print('DEAD', tag, round(time.time() - t0), 's', flush=True)


def probs(sensor, mid, tags=('matched',)):
    path = os.path.join(RUN, 'prob_{}_{}.npz'.format(sensor, mid))
    have = dict(np.load(path)) if os.path.isfile(path) else {}
    (z, F, L2, L3), M, names = load(sensor, mid)
    need = {'frame_bench': BENCH_FRAME, 'frame_idsse': IDSSE_FRAME}
    need.update({'frame_' + t: os.path.join(RUN, 'frame_{}.pkl'.format(t)) for t in tags})
    changed = False
    Mfull = None
    for k, p in need.items():
        if k not in have:
            Mfull = np.asarray(M) if Mfull is None else Mfull
            have[k] = X.predict(*pk(p), Mfull).astype('float32'); changed = True
    fr = X.cp_candidates(np.zeros(len(z['half'])), z['half'])
    cpm = {'cp_bench': BENCH_CP}; cpm.update({'cp_' + t: os.path.join(RUN, 'cp_{}.pkl'.format(t)) for t in tags})
    cx = None
    for k, p in cpm.items():
        if k not in have:
            if cx is None:
                cx, _ = X.cp_design(M, names, L2, fr)
            have[k] = X.predict_cp(*pk(p), cx).astype('float32'); changed = True
    dm = {'dead_idsse': IDSSE_DEAD}; dm.update({'dead_' + t: os.path.join(RUN, 'dead_{}.pkl'.format(t)) for t in tags})
    dx = None
    for k, p in dm.items():
        if k not in have:
            dx = dead_design(z, F, L2)[0] if dx is None else dx
            have[k] = pk(p).predict_proba(dx)[:, 1].astype('float32'); changed = True
    have['cp_frames'] = fr
    if changed:
        np.savez_compressed(path, **have)
    return (z, F, L2, L3), have


def infer(parts, p, s):
    z, F, L2, L3 = parts
    l4 = dict(p_home_full=p['frame_' + s.get('frame', 'bench')], p_home_players=np.full(len(z['half']), .5),
              cp_frames=p['cp_frames'], cp_p=p['cp_' + s.get('cp', 'bench')])
    b = s.get('dead_blend', 0)
    if b:
        llr, comp = B.evidence(z, F, L2)
        llr = (1 - b) * llr + b * s.get('dead_weight', .25) * logit(p['dead_' + s['dead']])
        L3 = {'dead': B.decode_evidence(z, F, llr, comp['kickoff_shape'])}
    return predict((z, F, L2, L3, l4), s.get('decoder', {}))


def score(pred, gt):
    r = quick(pred, gt)
    d = gt == 2; g = gt >= 0
    r['dead_recall'] = float(np.mean(pred[d] == 2)); r['dead_precision'] = float(np.mean(gt[g & (pred == 2)] == 2))
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}


def settings(tags=('matched',)):
    S = {'current': {}, 'idsse_clean_models': {'frame': 'idsse', 'dead': 'idsse', 'dead_blend': .5, 'dead_weight': .5}}
    for t in tags:
        for cp in ('bench', t):
            S['{}_frame_cp-{}'.format(t, cp)] = {'frame': t, 'cp': cp}
            for b in (.5, 1.0):
                for w in (.1, .25, .5):
                    S['{}_frame_cp-{}_dead{}_w{}'.format(t, cp, b, w)] = {'frame': t, 'cp': cp, 'dead': t, 'dead_blend': b, 'dead_weight': w}
    return S


def run(mids, sensors, S, out, tags=('matched',)):
    rows = {}
    for sensor in sensors:
        for mid in mids:
            parts, p = probs(sensor, mid, tags); gt = labels(mid)
            for name, s in S.items():
                r = score(infer(parts, p, s), gt); rows.setdefault(name, {})[sensor + ':' + mid] = r
                print(sensor, mid, name, r, flush=True)
            json.dump(rows, open(os.path.join(RUN, out), 'w'), indent=1)
    return rows


if __name__ == '__main__':
    stage = sys.argv[1]
    if stage == 'train':
        train()
    elif stage == 'screen':
        rows = run(VALID, ('matched',), settings(), 'validation.json')
        base = rows['current']
        ok = [k for k, v in rows.items() if all(v[g]['macro'] >= base[g]['macro'] - .003 for g in v)]
        best = max(ok, key=lambda k: np.mean([r['spell'] for r in rows[k].values()]))
        json.dump({'name': best, 'setting': settings()[best]}, open(os.path.join(RUN, 'frozen.json'), 'w'), indent=1)
        print('FROZEN', best, flush=True)
    elif stage == 'test':
        fz = json.load(open(os.path.join(RUN, 'frozen.json')))
        S = {'current': {}, 'idsse_clean_models': settings()['idsse_clean_models'], 'selected': fz['setting']}
        run(TEST, ('matched', 'clean'), S, 'test.json')
