"""Score the independent event-first pipeline on the three prepared benchmark games."""
import json
import os
from pathlib import Path


def main():
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
    import numpy as np
    from ..mall.common import CACHE, load_layer
    from ..mall.interaction.model import legacy_evidence, TRAIN_GAMES
    from ..mall.layer4_possession_model.run import truth
    from ..mall.layer5_sequence_decoder.decoder import decode
    from ..vendor.challenge import possession_metrics as metrics
    from .run import build

    report = dict(protocol='fixed event rules; no fitting; original Mall leave-one-game-out', games={})
    for tag in TRAIN_GAMES:
        print(f'Scoring {tag}: event-first then excluded-game Mall reference', flush=True)
        labels, half, trace = build(tag)
        legacy = legacy_evidence(tag, tuple(t for t in TRAIN_GAMES if t != tag))
        z, L2, L3 = [load_layer(layer, tag) for layer in ('layer0', 'layer2', 'layer3')]
        original = decode(z, L2, L3, legacy)
        y = truth(tag)
        y = np.where(y < 0, metrics.X, y)
        row = dict(events=len(trace['events']))
        for name, predictions in [('original', original), ('event_first', labels)]:
            result = metrics.evaluate(y, np.where(half > 0, predictions, metrics.X), bootstrap_n=0)
            row[name] = {key: float(result[key]) for key in ('spell_f1_mean', 'macro_f1', 'accuracy')}
            row[name]['possessions'] = int(result['n_pred_spells'])
        report['games'][tag] = row
        print(json.dumps({tag: row}), flush=True)
    for name in ('original', 'event_first'):
        report[name + '_mean'] = {key: float(np.mean([r[name][key] for r in report['games'].values()]))
                                 for key in ('spell_f1_mean', 'macro_f1', 'accuracy')}
    path = Path(CACHE).parent / 'event_first/benchmark.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
