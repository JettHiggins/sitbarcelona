"""Leave-one-game-out comparison. Does not read BAR–ATH public truth."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--games', type=int, choices=[1, 2, 3], nargs='+', default=[1, 2, 3])
    parser.add_argument('--output', type=Path, default=Path('pipeline/reports/mall/interaction-evaluation.json'))
    args = parser.parse_args()
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    import numpy as np
    from ...vendor.challenge import possession_metrics as metrics
    from ..common import load_layer
    from ..layer4_possession_model.run import truth
    from ..layer5_sequence_decoder.decoder import decode
    from .model import TRAIN_GAMES, predict, train

    report = dict(protocol='leave-one-game-out; fixed settings; no public labels', games={})
    for game in args.games:
        tag = f'bench{game}'
        training = tuple(t for t in TRAIN_GAMES if t != tag)
        trained = train(training)
        result, contacts, legacy = predict(tag, trained)
        z, L2, L3 = [load_layer(layer, tag) for layer in ('layer0', 'layer2', 'layer3')]
        original = decode(z, L2, L3, legacy)
        y = truth(tag)
        y = np.where(y < 0, metrics.X, y)
        row = dict(train_games=list(training), fingerprint=trained['fingerprint'], contacts=len(contacts))
        for name, labels in [('original', original), ('interaction', result['label'])]:
            scored = metrics.evaluate(y, np.where(z['half'] == 0, metrics.X, labels), bootstrap_n=0)
            row[name] = {key: float(scored[key]) for key in ('spell_f1_mean', 'macro_f1', 'accuracy')}
            boundaries = np.r_[True, labels[1:] != labels[:-1]]
            row[name]['possessions'] = int((boundaries & (labels < 2) & (z['half'] > 0)).sum())
        report['games'][tag] = row
        print(json.dumps({tag: row}), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    for name in ('original', 'interaction'):
        report[name + '_mean'] = {key: float(np.mean([r[name][key] for r in report['games'].values()]))
                                 for key in ('spell_f1_mean', 'macro_f1', 'accuracy')}
    args.output.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
