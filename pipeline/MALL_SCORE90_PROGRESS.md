# Work toward 90% possession spell-F1

2026-11-12, continuing on `score-90` from commit `36a9c5f`.
The target is the official scorer's `spell_f1_mean`, not frame accuracy or
macro-F1. The target has **not** been reached. Model selection uses other
matches; the challenge's five-minute public window is not a tuning set.

## Confirmed starting point

Rescored existing layer-5 caches with the unchanged official scorer:

| Game | spell-F1 | macro-F1 |
|---|---:|---:|
| Metrica 1 | 0.7961 | 0.9172 |
| Metrica 2 | 0.8161 | 0.9457 |
| Metrica 3 | 0.8291 | 0.9470 |
| DFL 1 | 0.7058 | 0.9185 |
| DFL 4 | 0.6924 | 0.9140 |
| Mean | 0.7679 | 0.9285 |

The production trajectory models exclude the game being evaluated. These five
games have already informed model and decoder selection; they are development
benchmarks, not a previously untouched final test set.

## Three-seed ensemble: no useful improvement

Finished the existing `seqS1` and `seqS2` jobs and compared their probability
average with `seqB`. All three scratch models trained on the same eight games,
excluding both Metrica 1 and DFL 1. Decode settings were fixed at the existing
defaults. Production leave-one-game-out models train on nine games, so the
fairest ensemble comparison is with `seqB` itself.

| Prediction | Metrica 1 spell-F1 | DFL 1 spell-F1 | Mean |
|---|---:|---:|---:|
| Production | 0.7961 | 0.7058 | 0.7510 |
| Original scratch model, seqB | 0.8050 | 0.7004 | 0.7527 |
| Extra seed, seqS1 | 0.7972 | 0.6703 | 0.7338 |
| Extra seed, seqS2 | 0.7933 | 0.7102 | 0.7518 |
| Three-seed probability average | 0.8006 | 0.7049 | 0.7527 |

The ensemble gains only 0.000055 mean spell-F1 over seqB. It is not adopted:
tripling training and inference has no demonstrated spell-F1 benefit here.
Full results and prediction hashes are retained locally in
`pipeline/reports/mall/ensemble/three-members.json`.

Reproduce the comparison without replacing model caches or CSVs:

```powershell
$env:PYTHONPATH="$PWD\.deps_torch;$PWD\.deps;$PWD"
python -m pipeline.mall.bench.ensemble --members seqB seqS1 seqS2 --report pipeline/reports/mall/ensemble/three-members.json
```

## Diagnosis and experiments

At temporal IoU 0.5, the existing pipeline matches only 15-33% of sub-second
possessions across these five games. Recall for possessions over ten seconds is
90-96%. This is a diagnosis on benchmark labels, not an inference-time rule.

Replacing predicted dead-ball labels with the true dead-ball mask while retaining
the model's team evidence gives spell-F1 0.875, 0.863, 0.898, 0.756, and 0.754.
Replacing the team decisions with true teams while retaining predicted stoppages
gives 0.910, 0.944, 0.929, 0.920, and 0.896. These deliberately label-assisted
diagnostics are not submission results or guaranteed upper bounds: the former
retains the old change-point grid, and the latter fills false-live dead frames
with the last known true team.

The next experiment weights the training loss by the square root of inverse
possession duration (relative to ten seconds), clipped to [0.25, 4]. This reduces
long spells' dominance without allowing one-frame labels to dominate training.
Only training-game labels determine these weights. No truth is needed at inference.
The option defaults to zero, preserving ordinary training unless requested.

```powershell
python -m pipeline.mall.bench.train_sequence --name codex_balanced05 --spell-balance 0.5 --steps 6000
```

Each experiment records its training/test split, seed, settings, source hash,
PyTorch version, device, and checkpoint hash in a separate model manifest.
Training prints elapsed time and estimated remaining time every 250 steps.
The default model and both existing exported prediction files remain unchanged.

## Paused at the user's request

The completed short-possession-weighted run took 350 seconds. With the same
eight-game training split and fixed decoder, mean spell-F1 on Metrica 1 / DFL 1
was 0.7534 (0.8011 / 0.7057), versus seqB's 0.7527. Averaging the two gave
0.7541. These small differences do not justify replacing the production model.

A ten-setting stoppage sweep across the five development games topped out at
0.7705 mean spell-F1, versus 0.7679 for the current settings. A separate decoder
ablation on the two test games topped out at 0.7567 versus 0.7527. Neither sweep
has established a meaningful generalization gain, and neither changed defaults.
Reports are in `pipeline/reports/mall/ensemble/` as `balanced05-scores.json`,
`dead-sweep.json`, and `decoder-sweep.json`.

A new experimental temporal-convolution architecture was implemented and its
prediction shape, team symmetry, and ungraded-frame behavior were tested. Its
first training job (`codex_temporal`) was stopped at the user's request before
completion. It produced no scored predictions or completed checkpoint; training
would need to restart. No claim of improvement is made for it.

All 95 pipeline tests passed before pausing. The work remains uncommitted on
`score-90`; no push was made. The existing production model and CSVs remain the
usable result. No new experiment has reached 90% spell-F1.
