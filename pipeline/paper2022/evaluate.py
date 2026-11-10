"""Event-level timing/actor evaluation on original Metrica sample tracking."""
import argparse
import csv
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import linear_sum_assignment

from .core import Parameters
from .run import reproduce


def match_events(predicted, reference, tolerance):
    """Maximum-cardinality, then minimum-time-error, same-actor one-to-one matches."""
    groups = {(e['half'], e['slot']) for e in reference + predicted if e['slot'] is not None}
    pairs = []
    for key in sorted(groups):
        gt = [i for i, e in enumerate(reference) if (e['half'], e['slot']) == key]
        pr = [i for i, e in enumerate(predicted) if (e['half'], e['slot']) == key]
        if not gt or not pr:
            continue
        dt = np.abs(np.subtract.outer([reference[i]['frame'] for i in gt], [predicted[i]['frame'] for i in pr]))
        unmatched = (max(len(gt), len(pr)) + 1) * (tolerance + 1)
        cost = np.full((len(gt), len(pr) + len(gt)), float(unmatched))
        cost[:, :len(pr)] = np.where(dt <= tolerance, dt, unmatched * (len(gt) + 1))
        rows, cols = linear_sum_assignment(cost)
        pairs.extend((gt[r], pr[c]) for r, c in zip(rows, cols) if c < len(pr) and dt[r, c] <= tolerance)
    tp = len(pairs)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(reference) if reference else 0.0
    errors = [predicted[p]['frame'] - reference[g]['frame'] for g, p in pairs]
    confusion = {}
    for g, p in pairs:
        key = reference[g]['event'] + ' -> ' + predicted[p]['event']
        confusion[key] = confusion.get(key, 0) + 1
    per_class = {}
    for name in sorted({e['event'] for e in reference + predicted}):
        correct = sum(reference[g]['event'] == predicted[p]['event'] == name for g, p in pairs)
        ng = sum(e['event'] == name for e in reference)
        npred = sum(e['event'] == name for e in predicted)
        per_class[name] = dict(reference=ng, predicted=npred, correct=correct,
                               precision=correct / npred if npred else 0.0,
                               recall=correct / ng if ng else 0.0,
                               f1=2 * correct / (ng + npred) if ng + npred else 0.0)
    return dict(predicted=len(predicted), reference=len(reference), matched=tp,
                precision=precision, recall=recall,
                f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
                median_signed_error_frames=float(np.median(errors)) if errors else None,
                median_absolute_error_frames=float(np.median(np.abs(errors))) if errors else None,
                matched_type_confusion=confusion,
                per_class=per_class,
                exact_type_matches=sum(reference[g]['event'] == predicted[p]['event'] for g, p in pairs))


def original_game(number, source):
    from ..mall.bench import metrica
    metrica.SRC = str(source)
    game = metrica.load(number)
    folder = source / 'data' / f'Sample_Game_{number}'
    stem = f'Sample_Game_{number}_'
    if number in (1, 2):
        slot_ids = []
        for team in ('Home', 'Away'):
            with (folder / (stem + f'RawTrackingData_{team}_Team.csv')).open() as handle:
                next(handle); next(handle)
                slot_ids.extend(name for name in next(handle).strip().split(',') if name.startswith('Player'))
        with (folder / (stem + 'RawEventsData.csv')).open() as handle:
            actors = [r['From'] for r in csv.DictReader(handle)]
        role_slots = None
    else:
        root = ET.parse(folder / (stem + 'metadata.xml')).getroot()
        slot_ids = [c.get('playerId') for c in root.iter('PlayerChannel') if c.get('channelId') == 'x']
        events = json.loads((folder / (stem + 'events.json')).read_text())['data']
        actors = [(e.get('from') or {}).get('id') for e in events]
        role_slots = []
        for player in root.iter('Player'):
            params = {p.findtext('Name'): p.findtext('Value') for p in player.iter('ProviderParameter')}
            if 'goalkeeper' in (params.get('position_type') or '').lower() and player.get('id') in slot_ids:
                role_slots.append(slot_ids.index(player.get('id')))
    index = {name: i for i, name in enumerate(slot_ids)}
    for event, actor in zip(game['events'], actors):
        event['slot'] = index.get(actor)
    return game, role_slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bench-src', type=Path, required=True)
    parser.add_argument('--games', type=int, choices=[1, 2, 3], nargs='+', default=[1, 2, 3])
    parser.add_argument('--output', type=Path, default=Path('pipeline/cache/paper2022/benchmark.json'))
    args = parser.parse_args()
    from ..mall.bench.labels import answer_key
    from ..mall.layer0_clean.direction import attack_direction
    from ..vendor.challenge import possession_metrics as metrics
    report = dict(profile='bc', input='original undegraded Metrica tracking; seven-frame SG ball',
                  protocol='fixed published parameters; no fitting; actor-aware event matching', games={})
    for number in args.games:
        G, role_slots = original_game(number, args.bench_src)
        xy, ball = G['players'] * G['pitch'], G['ball'] * G['pitch']
        half = G['period']
        teams = np.broadcast_to(G['team_of_slot'], xy.shape[:2]).copy()
        teams[~np.isfinite(xy).all(axis=2)] = -1
        directions = attack_direction(xy, np.isfinite(xy).all(axis=2), G['team_of_slot'], half, G['pitch'][0])
        signs = np.zeros((3, 2), np.int8)
        for h, row in directions.items():
            signs[h] = row['home'], row['away']
        y, derived_dead, graded = answer_key(G)
        keepers = None
        if role_slots:
            keepers = np.zeros(teams.shape, bool)
            keepers[:, role_slots] = np.isfinite(xy[:, role_slots]).all(axis=2)
        reference = [dict(frame=e['start'], half=e['period'], slot=e['slot'],
                          event='shot' if e['type'] == 'SHOT' else ('cross' if 'CROSS' in e['subtype'].upper() else 'pass'))
                     for e in G['events'] if e['type'] in ('PASS', 'SHOT') and e['slot'] is not None]
        restart_reference = []
        for e in G['events']:
            subtype = e['subtype'].upper()
            if e['slot'] is None or not (e['type'] == 'SET PIECE' or (e['type'] == 'RECOVERY' and 'GOAL KICK' in subtype)):
                continue
            name = next((kind for token, kind in [('KICK OFF', 'kickoff'), ('PENALTY', 'penalty'),
                         ('CORNER', 'corner'), ('THROW', 'throw_in'), ('GOAL KICK', 'goal_kick'),
                         ('FREE KICK', 'free_kick')] if token in subtype), None)
            if name:
                restart_reference.append(dict(frame=e['start'], half=e['period'], slot=e['slot'], event=name))
        game_report = {}
        for source, dead in [('missing', ~np.isfinite(ball).all(axis=1)), ('event_derived_status', derived_dead)]:
            labels, core, trace = reproduce(xy, teams, ball, half, G['pitch'], signs, dead, Parameters(), keepers)
            pred = [dict(frame=e['frame'], half=e['half'], slot=e['slot'],
                         event='shot' if e['event'].startswith('shot') or e['event'] == 'own_goal' else e['event'])
                    for e in trace['passing_events'] if graded[e['frame']]]
            score = metrics.evaluate(np.where(graded, y, metrics.X), np.where(graded, labels, metrics.X), bootstrap_n=0)
            row = dict(gains=len(trace['gains']), passing_events=len(pred), restarts=len(trace['restarts']),
                       spell_f1_mean=score['spell_f1_mean'], macro_f1=score['macro_f1'],
                       event_matching={str(t): match_events(pred, reference, t) for t in (12, 25, 250)},
                       restart_matching={str(t): match_events(
                           [dict(frame=e['frame'], half=e['half'], slot=e['slot'], event=e['kind'])
                            for e in trace['restarts'] if graded[e['frame']]], restart_reference, t) for t in (12, 25, 250)},
                       keeper_source=trace['keeper_source'])
            game_report[source] = row
            print(f"Game {number} {source}: spellF1={row['spell_f1_mean']:.4f}; "
                  f"actor event F1 @12/25/250f=" + '/'.join(f"{row['event_matching'][str(t)]['f1']:.3f}" for t in (12, 25, 250)), flush=True)
        report['games'][str(number)] = game_report
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
