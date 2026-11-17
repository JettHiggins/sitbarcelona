"""Held-out evaluation of the selected Wes decoder against Mall.

Run after `python -m pipeline.run prepare-mall --bench-src PATH`:
    python -m pipeline.wes.evaluate
"""

import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from ..vendor.challenge import possession_metrics as metrics
from ..mall.common import CACHE, ROOT, load_layer
from ..mall import config as C
from ..mall.interaction.model import legacy_evidence
from ..mall.layer4_possession_model.run import truth
from ..mall.layer5_sequence_decoder.decoder import decode
from .segmentation import predict

GAMES = ('bench1', 'bench2', 'bench3')
VARIANTS = ('wes',)


@contextmanager
def baseline_settings(baseline):
    """Keep the historical three-game comparison separate from production."""
    settings = dict(DEAD_W_LEARNED=0.0, DEAD_W_SPEED=1.5,
                    SEQ_W_FULL=0.35, SEQ_W_CP=1.0) if baseline == 'legacy' else {}
    previous = {key: getattr(C, key) for key in settings}
    try:
        for key, value in settings.items():
            setattr(C, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(C, key, value)


def evaluation_inputs(tag, baseline):
    if baseline == 'production':
        from ..run import run_module
        # Rebuild rather than mistake the old three-game caches for current evidence.
        run_module('pipeline.mall.layer3_dead_ball.run', tag)
        run_module('pipeline.mall.layer4_possession_model.run', tag)
    z, F, L2, L3 = (dict(load_layer(layer, tag)) for layer in
                    ('layer0', 'layer1', 'layer2', 'layer3'))
    if baseline == 'legacy':
        from ..mall.layer3_dead_ball.deadball import detect
        dead, llr, _, stages = detect(z, F, L2, trace=True)
        if not np.array_equal(dead, L3['dead']):
            raise ValueError(f'{tag}: Layer 3 cache is not the historical rules baseline; '
                             'use a separate checkout/cache to reproduce legacy results')
        L3.update(dead_llr=llr, hard_kickoff=stages['hard_kickoff'],
                  **{'stage_' + key: value for key, value in stages.items()
                     if key != 'hard_kickoff'})
        evidence = legacy_evidence(tag, tuple(t for t in GAMES if t != tag))
    else:
        evidence = dict(load_layer('layer4', tag))
    return z, F, L2, L3, evidence


def source_revision():
    files = [Path(ROOT) / 'pipeline' / 'run.py']
    files += sorted((Path(ROOT) / 'pipeline' / 'mall').rglob('*.py'))
    files += sorted((Path(ROOT) / 'pipeline' / 'wes').rglob('*.py'))
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    return dict(commit=commit, source_sha256=digest.hexdigest())


def short_recall(y, labels, max_seconds=5):
    runs = metrics.valid_runs(y)
    gt = metrics.possession_spells(y, runs)
    pred = metrics.possession_spells(labels, runs)
    flags = metrics.matched_gt_flags(metrics.spell_overlaps(gt, pred), len(gt), 0.5)
    short = np.array([b - a <= max_seconds * metrics.FPS for a, b, _ in gt], bool)
    return float(flags[short].mean()) if short.any() else None


def score(y, labels):
    result = metrics.evaluate(y, labels, bootstrap_n=0)
    cm = metrics.scored_confusion(metrics.confusion(y, labels))
    alive_tp = float(cm[:2, :2].sum())
    alive_fp = float(cm[2, :2].sum())
    alive_fn = float(cm[:2, 2:].sum())
    t12 = result['transitions'][12]
    t25 = result['transitions'][25]
    return dict(spellF1=result['spell_f1_mean'], macroF1=result['macro_f1'],
                alive_precision=alive_tp / (alive_tp + alive_fp) if alive_tp + alive_fp else 0.0,
                alive_recall=alive_tp / (alive_tp + alive_fn) if alive_tp + alive_fn else 0.0,
                D_precision=result['per_class']['D']['precision'],
                D_recall=result['per_class']['D']['recall'],
                predicted_possessions=result['n_pred_spells'],
                true_possessions=result['n_gt_spells'],
                short_spell_recall_5s=short_recall(y, labels),
                boundary_12f=dict(precision=t12['precision'], recall=t12['recall'],
                                  mean_abs_latency=t12['mean_abs_latency']),
                boundary_25f=dict(precision=t25['precision'], recall=t25['recall'],
                                  mean_abs_latency=t25['mean_abs_latency']))


def gate(report, name):
    changes = [row[name]['spellF1'] - row['original']['spellF1']
               for row in report['games'].values()]
    score_gate = bool(np.mean(changes) >= 0.02 and sum(v > 0 for v in changes) >= 2
                      and min(changes) >= -0.02)
    return dict(mean_change=float(np.mean(changes)), per_game_change=changes,
                gains=sum(v > 0 for v in changes), score_gate_passed=score_gate,
                note='Full-match 199,319-row validation and reproducibility remain separate gates.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', choices=('production', 'legacy'), default='production',
                        help='Production rebuilds ten-game evidence; legacy reproduces the older three-game experiment')
    parser.add_argument('--output', type=Path,
                        default=Path(ROOT) / 'pipeline/reports/wes/segmentation-evaluation.json')
    args = parser.parse_args()
    if args.baseline == 'production':
        from ..run import BENCH_TAGS
        from ..mall.common import bench_tags
        missing = sorted(set(BENCH_TAGS) - set(bench_tags()))
        if missing:
            parser.error('Production comparison needs all ten prepared games; missing ' +
                         ', '.join(missing) + '. Legacy results cannot select the production model.')
    with baseline_settings(args.baseline):
        evaluate(args)


def evaluate(args):
    from ..mall.common import train_tags
    games = GAMES if args.baseline == 'legacy' else (*GAMES, 'dfl1', 'dfl4')
    report = dict(protocol='leave-one-game-out on other-game keys; no challenge labels',
                  baseline=args.baseline,
                  revision=source_revision(), inputs={}, games={}, gates={})
    provenance = Path(CACHE) / 'bench' / 'provenance.json'
    report['bench_source'] = json.loads(provenance.read_text()) if provenance.exists() else None
    for tag in games:
        key = Path(CACHE) / 'bench' / f'{tag}_gt.csv'
        if not key.exists():
            raise ValueError(f'{key} missing; run prepare-mall with Metrica sample-data first')
        report['inputs'][tag] = dict(key=str(key), key_sha256=hashlib.sha256(key.read_bytes()).hexdigest(),
                                     training_games=([t for t in GAMES if t != tag]
                                                     if args.baseline == 'legacy' else train_tags(tag)))
        z, F, L2, L3, evidence = evaluation_inputs(tag, args.baseline)
        y = truth(tag)
        y = np.where(y < 0, metrics.X, y)
        row = {}
        original = decode(z, L2, L3, evidence)
        row['original'] = score(y, np.where(z['half'] == 0, metrics.X, original))
        for name in VARIANTS:
            labels, diagnostics = predict(z, F, L2, L3, evidence)
            row[name] = score(y, np.where(z['half'] == 0, metrics.X, labels))
            row[name]['changed_dead_frames'] = diagnostics['changed_dead_frames']
            row[name]['reopened_spans'] = sum(d['reopened'] for d in diagnostics['dead_revisions'])
            row[name]['new_candidate_rows'] = len(diagnostics['change_candidates'])
            row[name]['joint_windows'] = len(diagnostics.get('windows', []))
            row[name]['joint_changed_frames'] = diagnostics.get('changed_frames', 0)
        report['games'][tag] = row
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
        print(f'{tag}: ' + ' '.join(f'{name}={row[name]["spellF1"]:.4f}' for name in
                                   ('original', *VARIANTS)), flush=True)
    for name in VARIANTS:
        report['gates'][name] = gate(report, name)
    report['mean'] = {name: {metric: float(np.mean([report['games'][tag][name][metric]
                                                for tag in games]))
                             for metric in ('spellF1', 'macroF1')}
                      for name in ('original', *VARIANTS)}
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'wrote {args.output}', flush=True)


if __name__ == '__main__':
    main()
