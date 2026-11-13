"""Attribute missed benchmark possessions to observable failure symptoms.

This is a diagnostic on the three OTHER games, not a source of match labels or
a tuning objective. Categories describe the prediction inside an unmatched
ground-truth spell; they do not establish the causal pipeline component.

Run after prepare-mall:
    python -m pipeline.wes.diagnose
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from ..mall.common import CACHE, ROOT, load_layer
from ..mall.interaction.model import legacy_evidence
from ..mall.layer4_possession_model.run import truth
from ..mall.layer5_sequence_decoder.decoder import candidate_points, decode, frame_evidence
from ..vendor.challenge import possession_metrics as metrics
from .evaluate import GAMES, score, source_revision
from .segmentation import predict

VARIANTS = ('original', 'revisable', 'candidates')
HEADLINE_IOU = 0.5
BOUNDARY_TOLERANCE = 12


def spell_symptom(window, correct_team):
    """A mutually exclusive majority description, with mixed as the remainder."""
    n = len(window)
    same = int(np.sum(window == correct_team))
    dead = int(np.sum(window == metrics.D))
    other = int(np.sum(window == 1 - correct_team))
    if 2 * dead >= n:
        category = 'D_majority'
    elif 2 * other >= n:
        category = 'wrong_team_majority'
    elif 2 * same >= n:
        category = 'right_team_majority_unmatched'
    else:
        category = 'mixed'
    return category, dict(right_team=same / n, wrong_team=other / n, D=dead / n)


def missed_spells(gt, pred):
    runs = metrics.valid_runs(gt)
    filled = metrics.fill_abstentions(pred, runs)
    gsp = metrics.possession_spells(gt, runs)
    psp = metrics.possession_spells(filled, runs)
    overlaps = metrics.spell_overlaps(gsp, psp)
    matched = metrics.matched_gt_flags(overlaps, len(gsp), HEADLINE_IOU)
    best = {}
    for iou, gi, _ in overlaps:
        best[gi] = max(best.get(gi, 0.0), iou)
    rows = []
    for i, (start, end, team) in enumerate(gsp):
        if matched[i]:
            continue
        category, fractions = spell_symptom(filled[start:end], team)
        rows.append(dict(start=int(start), end=int(end), team='H' if team == 0 else 'A',
                         duration_s=(end - start) / metrics.FPS, symptom=category,
                         fractions=fractions, best_same_team_iou=best.get(i, 0.0)))
    categories = Counter(r['symptom'] for r in rows)
    return dict(gt_spells=len(gsp), predicted_spells=len(psp),
                matched_at_0_5=int(matched.sum()), missed_at_0_5=len(rows),
                missed_short_5s=sum(r['duration_s'] <= 5 for r in rows),
                symptoms=dict(sorted(categories.items())), spells=rows)


def false_D_provenance(gt, pred, L3):
    false_dead = (gt < metrics.D) & (pred == metrics.D)
    stages = ('viterbi', 'snapped', 'rules', 'durations')
    remaining = false_dead.copy()
    provenance = {}
    for stage in stages:
        key = 'stage_' + stage
        if key not in L3:
            raise ValueError(f'{key} missing; rebuild other-game Layer 3 caches')
        selected = remaining & L3[key].astype(bool)
        provenance[stage] = int(selected.sum())
        remaining &= ~selected
    provenance['final_or_wes_constraint'] = int(remaining.sum())
    return dict(false_D_frames=int(false_dead.sum()),
                first_stage_with_D=provenance,
                Mall_D_survived=int(np.sum(false_dead & L3['dead'].astype(bool))),
                Wes_added_D=int(np.sum(false_dead & ~L3['dead'].astype(bool))))


def missed_team_switches(gt, pred, z, L2, L4, extra_candidates):
    runs = metrics.valid_runs(gt)
    gt_trans = [t for t in metrics.transitions(gt, runs)
                if t[1] in (metrics.H, metrics.A) and t[2] in (metrics.H, metrics.A)]
    pred_trans = [t for t in metrics.transitions(pred, runs)
                  if t[1] in (metrics.H, metrics.A) and t[2] in (metrics.H, metrics.A)]
    matched, missed, spurious = metrics.match_transitions(
        gt_trans, pred_trans, BOUNDARY_TOLERANCE, typed='pair')
    pts = candidate_points(L2, z, L4, extra_candidates)
    frames = np.array(sorted(pts), int)
    owner_llr = frame_evidence(L2, L4)
    rows = []
    for frame, old, new in missed:
        near = frames[(frames >= frame - BOUNDARY_TOLERANCE) &
                      (frames <= frame + BOUNDARY_TOLERANCE)]
        accessible = [int(f) for f in near if pred[f] in (metrics.H, metrics.A)]
        bonus = max((pts[f][new] for f in accessible), default=None)
        # A 1-second local ownership margin is a diagnostic, not a trained label.
        hi = min(frame + metrics.FPS, len(owner_llr))
        owner_margin = float(np.nanmean(owner_llr[frame:hi])) * (1 if new == metrics.H else -1)
        rows.append(dict(frame=int(frame), from_team='H' if old == 0 else 'A',
                         to_team='H' if new == 0 else 'A',
                         accessible_candidate=bool(accessible),
                         positive_target_bonus=bonus is not None and bonus > 0,
                         max_target_bonus=float(bonus) if bonus is not None else None,
                         owner_margin_next_1s=owner_margin,
                         owner_evidence_favors_target=owner_margin > 0))
    return dict(gt_direct_switches=len(gt_trans), matched_12f=len(matched),
                missed_12f=len(missed), spurious_12f=len(spurious),
                missed_with_accessible_candidate=sum(r['accessible_candidate'] for r in rows),
                missed_with_positive_target_bonus=sum(r['positive_target_bonus'] for r in rows),
                missed_with_target_favored_by_owner_evidence=sum(
                    r['owner_evidence_favors_target'] for r in rows),
                switches=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path(ROOT) / 'pipeline/reports/wes/missed-spell-diagnosis.json')
    args = parser.parse_args()
    provenance = Path(CACHE) / 'bench/provenance.json'
    report = dict(protocol='other-game event-derived keys; no BAR-ATH labels',
                  revision=source_revision(),
                  bench_source=json.loads(provenance.read_text()) if provenance.exists() else None,
                  definitions=dict(headline_iou=HEADLINE_IOU,
                                   direct_switch_tolerance_frames=BOUNDARY_TOLERANCE,
                                   symptom='majority predicted label inside unmatched GT spell; not a causal diagnosis'),
                  games={})
    for tag in GAMES:
        key = Path(CACHE) / 'bench' / f'{tag}_gt.csv'
        if not key.exists():
            parser.error(f'{key} missing; run prepare-mall first')
        z, F, L2, L3 = (load_layer(layer, tag) for layer in
                        ('layer0', 'layer1', 'layer2', 'layer3'))
        L4 = legacy_evidence(tag, tuple(t for t in GAMES if t != tag))
        raw_truth = truth(tag)
        gt = np.where(raw_truth < 0, metrics.X, raw_truth)
        baseline = decode(z, L2, L3, L4)
        outputs = {'original': (baseline, [])}
        for variant in ('revisable', 'candidates'):
            labels, diagnostics = predict(z, F, L2, L3, L4, variant=variant)
            outputs[variant] = (labels, diagnostics['change_candidates'])
        row = dict(input_key_sha256=hashlib.sha256(key.read_bytes()).hexdigest(),
                   training_games=[t for t in GAMES if t != tag], variants={})
        for variant, (labels, extra) in outputs.items():
            labels = np.where(z['half'] == 0, metrics.X, labels)
            row['variants'][variant] = dict(
                metrics=score(gt, labels),
                missed_spells=missed_spells(gt, labels),
                false_D=false_D_provenance(gt, labels, L3),
                direct_switches=missed_team_switches(gt, labels, z, L2, L4, extra))
        report['games'][tag] = row
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, default=numpy_scalar) + '\n')
        for variant, data in row['variants'].items():
            spells = data['missed_spells']
            print(f'{tag} {variant}: {spells["missed_at_0_5"]}/{spells["gt_spells"]} missed; '
                  f'{spells["symptoms"]}', flush=True)
    print(f'wrote {args.output}', flush=True)


def numpy_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'{type(value).__name__} is not JSON serializable')


if __name__ == '__main__':
    main()
