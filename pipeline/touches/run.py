"""Train/evaluate leave-one-game-out, then export literal last-touch predictions."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

from .core import Parameters, detect, signals
from .train import fit, load_game, training_rows
from ..paper2022.evaluate import match_events


def write_csv(labels, output):
    if not np.isin(labels, [-1, 0, 1]).all():
        raise ValueError('Touch-only labels must be unknown/home/away')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w') as f:
        f.write('frame,label\n')
        f.writelines(f'{i},{"XHA"[int(y)+1]}\n' for i, y in enumerate(labels))


def build(source, output, evaluate=True):
    cache = Path('pipeline/cache/touches')
    cache.mkdir(parents=True, exist_ok=True)
    datasets, samples = {}, {}
    for number in (1, 2, 3):
        datasets[number] = load_game(number, source)
        samples[number] = training_rows(*datasets[number])
        x, y = samples[number]
        print(f'Game {number}: {len(y)} training windows, {int(y.sum())} positive endpoint proxies', flush=True)
    report = dict(protocol='leave-one-game-out; weak event endpoint labels; fixed parameters; no challenge truth', games={})
    if evaluate:
        for number, data in datasets.items():
            model = fit([samples[k] for k in samples if k != number])
            players, ball, half, teams, refs = data
            _, trace = detect(players, ball, half, teams, model)
            row = dict(touches=len(trace['touches']), ambiguous=len(trace['rejected']),
                       endpoint_proxy_matching={str(t): match_events(trace['touches'], refs, t) for t in (3, 12, 25)})
            # Proposal coverage is distinct from classification/actor errors.
            sig = signals(ball, half)
            covered = 0
            for e in refs:
                f, s = e['frame'], e['slot']
                idx = np.arange(max(0, f-3), min(len(ball), f+4))
                covered += bool(np.any(sig['valid'][idx] & (sig['impulse'][idx] >= Parameters().minimum_impulse)
                               & (np.linalg.norm(players[idx, s]-ball[idx], axis=1) <= Parameters().radius)))
            row['proposal_coverage_at_3_frames'] = covered/len(refs) if refs else 0
            report['games'][str(number)] = row
            (cache / f'game{number}.touches.json').write_text(json.dumps(trace, indent=2, allow_nan=False)+'\n')
            print(f"Held-out {number}: {row['touches']} touches; actor endpoint proxy F1 @3/12/25f=" +
                  '/'.join(f"{row['endpoint_proxy_matching'][str(t)]['f1']:.3f}" for t in (3, 12, 25)), flush=True)
    model = fit(list(samples.values()))
    (cache / 'model.pkl').write_bytes(pickle.dumps(model))
    z = np.load('pipeline/cache/mall/layer0_tactical.npz')
    players = z['player_xy_raw'].copy()
    players[~z['player_valid'].astype(bool)] = np.nan
    if 'player_duplicate' in z:
        players[z['player_duplicate'].astype(bool)] = np.nan
    labels, trace = detect(players, z['ball_xy_raw'], z['half'], z['player_team'], model,
                           evidence_path=cache / 'tactical.evidence.npz')
    trace['training'] = dict(games=[1, 2, 3], source=str(source.resolve()),
                             features='25-frame distance polynomial and local 2D ball kinematics',
                             supervision='weak Metrica event endpoints, not all physical touches',
                             challenge_truth_used=False, calibrated_probability=False)
    trace['input_sha256'] = hashlib.sha256(Path('pipeline/cache/mall/layer0_tactical.npz').read_bytes()).hexdigest()
    trace['training']['source_files_sha256'] = {
        str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((source / 'data').glob('Sample_Game_*/*')) if path.is_file()}
    trace['training']['model_sha256'] = hashlib.sha256((cache / 'model.pkl').read_bytes()).hexdigest()
    (cache / 'tactical.touches.json').write_text(json.dumps(trace, indent=2, allow_nan=False)+'\n')
    (cache / 'benchmark.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    write_csv(labels, output)
    print(f"{output}: {len(labels)} frames; {len(trace['touches'])} touches; {len(trace['rejected'])} ambiguous; no dead-ball detector", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bench-src', required=True, type=Path)
    p.add_argument('--output', type=Path, default=Path('possession-viewer/data/touches.csv'))
    p.add_argument('--skip-evaluation', action='store_true')
    args = p.parse_args()
    build(args.bench_src, args.output, not args.skip_evaluation)


if __name__ == '__main__':
    main()
