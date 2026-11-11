"""Weak touch supervision from Metrica event endpoints; never challenge labels."""
import csv
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .core import Parameters, features, signals
from ..mall.bench import metrica


def load_game(number, source):
    metrica.SRC = str(source)
    game = metrica.load(number)
    folder = source / 'data' / f'Sample_Game_{number}'
    stem = f'Sample_Game_{number}_'
    if number < 3:
        ids = []
        for team in ('Home', 'Away'):
            with (folder / f'{stem}RawTrackingData_{team}_Team.csv').open() as f:
                next(f); next(f)
                ids.extend(v for v in next(f).strip().split(',') if v.startswith('Player'))
        with (folder / f'{stem}RawEventsData.csv').open() as f:
            raw = list(csv.DictReader(f))
            actors, receivers = [r['From'] for r in raw], [r['To'] for r in raw]
    else:
        root = ET.parse(folder / f'{stem}metadata.xml').getroot()
        ids = [c.get('playerId') for c in root.iter('PlayerChannel') if c.get('channelId') == 'x']
        raw = json.loads((folder / f'{stem}events.json').read_text())['data']
        actors = [(e.get('from') or {}).get('id') for e in raw]
        receivers = [(e.get('to') or {}).get('id') for e in raw]
    index = {name: i for i, name in enumerate(ids)}
    refs = set()
    for event, actor, receiver in zip(game['events'], actors, receivers):
        if event['type'] in ('PASS', 'SHOT', 'RECOVERY', 'SET PIECE') and actor in index:
            refs.add((event['start'], event['period'], index[actor]))
        if event['type'] == 'PASS' and receiver in index:
            refs.add((event['end'], event['period'], index[receiver]))
    reference = [dict(frame=f, half=h, slot=s, event='touch') for f, h, s in sorted(refs)]
    players, ball = game['players']*game['pitch'], game['ball']*game['pitch']
    teams = np.broadcast_to(game['team_of_slot'], players.shape[:2]).copy()
    teams[~np.isfinite(players).all(axis=2)] = -1
    return players, ball, game['period'], teams, reference


def training_rows(players, ball, half, teams, reference, p=Parameters()):
    """Use local annotated-event windows, not all unannotated frames as negatives.

    +/-2 frames around an annotated actor contact are positive. Samples 8/12
    frames away are timing negatives; other nearby actors at a labeled contact
    are attribution negatives. Endpoints are weak labels, not physical truth.
    """
    sig = signals(ball, half, p)
    near_reference = set()
    for e in reference:
        near_reference.update((e['frame']+d, e['slot']) for d in range(-3, 4))
    rows = {}
    for e in reference:
        s = e['slot']
        for offset in (-12, -8, -2, -1, 0, 1, 2, 8, 12):
            f = e['frame']+offset
            if not (0 <= f < len(ball) and sig['valid'][f] and half[f] == e['half']):
                continue
            distance = np.linalg.norm(players[f]-ball[f], axis=1)
            if distance[s] <= p.radius:
                if abs(offset) <= 2:
                    rows[f, s] = 1
                    for other in np.flatnonzero((distance <= p.radius) & (teams[f] >= 0)):
                        if other != s and (f, int(other)) not in near_reference:
                            rows.setdefault((f, int(other)), 0)
                elif (f, s) not in near_reference:
                    rows.setdefault((f, s), 0)
    keys = sorted(rows)
    ff, ss = np.asarray(keys).T
    x = features(players, ball, ff, ss, sig, p)
    y = np.array([rows[k] for k in keys])
    return x, y


def fit(datasets):
    x = np.concatenate([d[0] for d in datasets])
    y = np.concatenate([d[1] for d in datasets])
    # Balanced local training distribution; output is a score, not a calibrated
    # probability of contact during arbitrary match frames.
    weights = np.where(y == 1, len(y)/(2*max(1, y.sum())), len(y)/(2*max(1, (y == 0).sum())))
    model = HistGradientBoostingClassifier(max_iter=120, max_leaf_nodes=15,
              min_samples_leaf=35, l2_regularization=1.0, random_state=42,
              early_stopping=False)
    model.fit(x, y, sample_weight=weights)
    return model
