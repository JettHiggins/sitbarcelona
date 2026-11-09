"""Validate viewer CSVs and compare them with the unchanged official scorer."""
import argparse
import csv
from pathlib import Path

import numpy as np

from .mall.games import GAME, half_video_blocks, source_frames
from .vendor.challenge import possession_metrics as metrics


def read(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ['frame', 'label']:
            raise ValueError(f'{path}: expected frame,label header')
        rows = list(reader)
    n = source_frames()
    if [int(r['frame']) for r in rows] != list(range(n)):
        raise ValueError(f'{path}: expected {n} ordered, unique frames')
    labels = np.array([r['label'] for r in rows])
    if not np.isin(labels, ['H', 'A', 'D', 'X']).all():
        raise ValueError(f'{path}: invalid labels')
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('csv', nargs='+', type=Path)
    parser.add_argument('--truth', type=Path, help='Optional official public ground-truth CSV; evaluation only')
    args = parser.parse_args()
    predictions = [(p, read(p)) for p in args.csv]
    truth = metrics.label_codes(metrics.read_label_csv(str(args.truth))) if args.truth else None
    graded = np.zeros(source_frames(), bool)
    for _, first, last in half_video_blocks(GAME):
        graded[first:last + 1] = True
    for path, labels in predictions:
        counts = ', '.join(f'{k}={int((labels == k).sum())}' for k in 'HADX')
        print(f'{path.name}: VALID ({len(labels)} frames), {counts}')
        print(f"  mask: {int((labels[~graded] != 'X').sum())} non-X outside graded halves; {int((labels[graded] == 'X').sum())} abstentions inside")
        if truth is not None:
            result = metrics.evaluate(truth, metrics.label_codes(labels), bootstrap_n=0)
            print(f"  public spellF1={result['spell_f1_mean']:.4f}, macroF1={result['macro_f1']:.4f}")
    for i, (a, x) in enumerate(predictions):
        for b, y in predictions[i + 1:]:
            print(f'{a.name} vs {b.name}: {np.mean(x[graded] != y[graded]):.2%} graded-frame disagreement')


if __name__ == '__main__':
    main()
