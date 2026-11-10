"""Supervised contact-chain evidence. Training is restricted to other Metrica games."""
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from ..common import CACHE, load_layer
from .. import config as C
from ..layer4_possession_model import crossgame as X
from ..layer4_possession_model.run import truth, full_matrix
from ..layer5_sequence_decoder.decoder import decode, logit
from .features import candidates, design

TRAIN_GAMES = ('bench1', 'bench2', 'bench3')
PARAMS = dict(max_iter=150, lr=0.05, leaves=15, min_leaf=100)
BLEND = 0.5  # Fixed equal log-odds blend; not selected on public labels.


def inputs(tag):
    return [load_layer(layer, tag) for layer in ('layer0', 'layer1', 'layer2', 'layer3')]


def fingerprint(tags):
    digest = hashlib.sha256()
    files = list(Path(__file__).parent.parent.rglob('*.py'))
    for tag in tags:
        files.extend(Path(CACHE) / f'{layer}_{tag}.npz' for layer in ('layer0', 'layer1', 'layer2', 'layer3'))
        files.append(Path(CACHE) / 'bench' / f'{tag}_gt.csv')
    for path in sorted(files):
        digest.update(str(path.name).encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    digest.update(json.dumps(PARAMS, sort_keys=True).encode())
    return digest.hexdigest()


def train(tags):
    tags = tuple(sorted(tags))
    if not tags or not set(tags).issubset(TRAIN_GAMES):
        raise ValueError('Interaction training requires other Metrica benchmark games only')
    key = fingerprint(tags)
    path = Path(CACHE) / 'interaction' / f'model-{key}.pkl'
    if path.exists():
        with path.open('rb') as handle:
            return pickle.load(handle)
    matrices, labels, cp_matrices, cp_targets = [], [], [], []
    names = None
    for tag in tags:
        z, F, L2, L3 = inputs(tag)
        matrix, columns, events = design(z, F, L2, L3)
        if names is not None and names != columns:
            raise ValueError('Interaction feature schemas differ between games')
        names = columns
        y = truth(tag)
        matrices.append(matrix)
        labels.append(np.where(y == 0, 1, np.where(y == 1, 0, -1)))
        frames = candidates(events, L3['dead'], z['half'])
        cp_matrices.append(matrix[frames])
        cp_targets.append(X.cp_labels(y, frames))
        print(f'Interaction training: {tag}, {len(events)} contacts, {len(frames)} candidates', flush=True)
    frame = X.fit(matrices, labels, names, params=PARAMS)
    cp = X.fit_cp(cp_matrices, cp_targets, names, params=PARAMS)
    result = dict(frame=frame, cp=cp, names=names, fingerprint=key, train_games=tags,
                  params=PARAMS, blend=BLEND)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as handle:
        pickle.dump(result, handle)
    return result


def legacy_evidence(tag, train_tags):
    """Frozen Mall evidence, with proper game exclusion for held-out comparisons."""
    if tag == 'tactical':
        return dict(load_layer('layer4', tag))
    # Independent models for each fold; never reuse the all-three-games tactical model.
    mats = {t: full_matrix(t) for t in (*train_tags, tag)}
    names = mats[tag][1]
    key = '+'.join(train_tags)
    ys = {t: truth(t) for t in train_tags}
    frame = X.cached(str(Path(CACHE) / 'models' / f'frame_{key}.pkl'), lambda: X.fit(
        [mats[t][0] for t in train_tags],
        [np.where(ys[t] == 0, 1, np.where(ys[t] == 1, 0, -1)) for t in train_tags], names))

    def cp_data(t):
        z, _, L2, L3 = inputs(t)
        fr = X.cp_candidates(L3['dead'], z['half'])
        matrix, columns = X.cp_design(mats[t][0], names, L2, fr)
        return fr, matrix, columns

    cp_train = {t: cp_data(t) for t in train_tags}
    fr, matrix, columns = cp_data(tag)
    cp = X.cached(str(Path(CACHE) / 'models' / f'cp_{key}.pkl'), lambda: X.fit_cp(
        [cp_train[t][1] for t in train_tags],
        [X.cp_labels(ys[t], cp_train[t][0]) for t in train_tags], columns))
    return dict(p_home_full=X.predict(*frame, mats[tag][0]),
                p_home_players=np.full(len(mats[tag][0]), 0.5),
                cp_frames=fr, cp_p=X.predict_cp(*cp, matrix))


def predict(tag, trained, legacy=None):
    if tag in trained['train_games']:
        raise ValueError('Cannot evaluate interaction model on a training game')
    z, F, L2, L3 = inputs(tag)
    matrix, names, events = design(z, F, L2, L3)
    if names != trained['names']:
        raise ValueError('Interaction feature schema changed; retrain the model')
    legacy = legacy if legacy is not None else legacy_evidence(tag, trained['train_games'])
    p = X.predict(*trained['frame'], matrix)
    # Pool log odds rather than treating the two correlated models as independent evidence.
    blend = trained['blend']
    combined_logit = (1 - blend) * logit(legacy['p_home_full']) + blend * logit(p)
    combined = 1 / (1 + np.exp(-combined_logit))
    frames = candidates(events, L3['dead'], z['half'])
    cp = X.predict_cp(*trained['cp'], matrix[frames])
    # Include old CP evidence only where it exists; new contact candidates get new evidence.
    positions = np.searchsorted(frames, legacy['cp_frames'])
    if len(positions):
        if np.any(positions >= len(frames)) or np.any(frames[positions] != legacy['cp_frames']):
            raise ValueError('Interaction candidates must contain every legacy change point')
        pooled = np.sqrt(np.clip(cp[positions], 1e-6, 1) * np.clip(legacy['cp_p'], 1e-6, 1))
        cp[positions] = pooled / pooled.sum(axis=1, keepdims=True)
    evidence = dict(legacy, p_home_full=combined, cp_frames=frames, cp_p=cp)
    labels = decode(z, L2, L3, evidence)
    return dict(label=labels, p_home_interaction=p, p_home_combined=combined,
                cp_frames=frames, cp_p=cp), events, legacy


def save_diagnostics(path, tag, trained, result, events, legacy):
    """Compact contact/recovery explanation table alongside the viewer CSV."""
    z, F, L2, L3 = inputs(tag)
    matrix, names, _ = design(z, F, L2, L3)
    from ..layer2_touches.touches import KIND_NAMES
    from ..layer5_sequence_decoder.decoder import decode as original_decode
    original = original_decode(z, L2, L3, legacy)
    letters = ['H', 'A', 'D']
    contacts_out = []
    for event in events:
        frame = event['frame']
        row = dict(event, kind=KIND_NAMES[event['kind']], team=letters[event['team']],
                   p_home_original=float(legacy['p_home_full'][frame]),
                   p_home_interaction=float(result['p_home_interaction'][frame]),
                   p_home_combined=float(result['p_home_combined'][frame]),
                   original_label=letters[original[frame]], label=letters[result['label'][frame]])
        for key in ('recovery_h', 'recovery_a', 'recovery_age', 'confirmed_contact_h',
                    'confirmed_contact_a', 'rebound_return_h', 'rebound_return_a'):
            value = float(matrix[frame, names.index(key)])
            row[key] = value if np.isfinite(value) else None
        contacts_out.append(row)
    changes = np.flatnonzero(result['label'][1:] != result['label'][:-1]) + 1
    data = dict(schema=1, game=tag, fps=C.FPS, train_games=list(trained['train_games']),
                fingerprint=trained['fingerprint'], params=trained['params'], blend=trained['blend'],
                dead_ball='unchanged Mall layer 3', contacts=contacts_out,
                transitions=[dict(frame=int(f), label=letters[result['label'][f]]) for f in changes],
                changed_frames=int(((original != result['label']) & (z['half'] > 0)).sum()))
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
