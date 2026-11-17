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

## 2026-11-17: temporal convolutions, completed

The first temporal-convolution run was slow because `torch.backends.cudnn.deterministic`
selects a deterministic cuDNN kernel for dilated `Conv1d` that took 2.37 s per training
step (0.11 s without determinism). `temporal.py` now writes each kernel-3 dilated
convolution as one linear map of `[x(t-d), x(t), x(t+d)]` with zero padding. Output
matches `nn.Conv1d` to 1.2e-6, and deterministic training takes 0.13 s per step. The
6,000-step run (`codex_temporal3`, same eight-game split and seed 0) took 608 s. The
stopped `codex_temporal` and `codex_temporal2` manifests have no checkpoints.

| Prediction | Metrica 1 spell-F1 | DFL 1 spell-F1 | Mean |
|---|---:|---:|---:|
| seqB (GRU) | 0.8050 | 0.7004 | 0.7527 |
| codex_temporal3 | 0.7844 | 0.7112 | 0.7478 |
| Average of the two | 0.8112 | 0.7022 | 0.7567 |

Alone, the temporal network is 0.005 worse than the GRU. The two-model average gains
0.004, the same size as the earlier ensemble, loss-weighting and sweep gains. Not adopted.
Report: `pipeline/reports/mall/ensemble/temporal3-scores.json`.

### Where the remaining spell-F1 is

Sub-second true possessions are 12-14% of all possessions in these games, and the
pipeline matches only 8-16 of the 46-57 per game at IoU 0.5. Of these, 23-36 per game are
one touch between two possessions of the other team (an interception, deflection or
clearance). Another 13-21 per game are a touch just before the ball goes dead. Many last
under 0.2 s. The predictions contain fewer than half as many sub-second possessions as the
truth (17-30 per game). Missed ones are either absent from the prediction or are absorbed
into a long same-team predicted possession (overlap under 0.1).

For about half of the missed ones, layer 2 detects no event by the correct team within
0.5 s. The rest have such an event that the decoder does not turn into a possession.
Changing the network architecture moves spell-F1 by about ±0.005, and the decoder sweep
found no gain, so 0.90 needs a better touch signal for these moments (detection and
team attribution of brief contacts), not another possession network.

Touch-bonus sweep on the five development games (production evidence; report
`pipeline/reports/mall/ensemble/touch-sweep.json`). Raising `SEQ_W_TOUCH` from 1 to 3, 6 or 10
nats gave 0.7685, 0.7702 and 0.7692 mean spell-F1, against 0.7679. Weakening the duration
prior (`SEQ_W_DUR` 0.1) lowered every combination (0.7639-0.7668). The +0.002 best is not
adopted. Decoder settings cannot recover the short possessions the evidence does not support.

Status: 0.90 not reached. Best development mean remains 0.7679 (production).

### Network evidence at true possession changes (seqB, Metrica 1 / DFL 1)

The decoder already reads the network's change probabilities on a 3-frame grid: decoding
seqB with change points every 3 frames gives the same 0.8050 / 0.7004.

At the true start of each possession that follows another possession, the largest
probability of a change to the correct team within ±3 frames is:

| True possession length | Metrica 1: median, share > 0.5 | DFL 1: median, share > 0.5 |
|---|---:|---:|
| under 1 s | 0.93, 56% | 0.14, 36% |
| 1-3 s | 1.00, 84% | 0.90, 60% |
| 3 s or more | 1.00, 83% | 0.96, 66% |

Inside sub-second possessions, the network's probability for the true team averages 0.55
(Metrica 1) and 0.50 (DFL 1). For possessions of 3 s or more it is 0.95 and 0.90.

Making short possessions cheaper for the decoder lowers spell-F1 (report
`switch-seqB.json`, against 0.7527): switch cost 2 gives 0.7472, switch cost 1 gives 0.7429,
switch cost 0 with no duration prior gives 0.6435, and change-point offset +1 / +2 gives
0.7510 / 0.7505. The change signal fires at too many non-changes, and the team signal for
sub-second possessions is near chance. Both need new input evidence for brief contacts.
Decoding cannot supply it.

### Feasibility of a tracking-based contact detector

Before building one, I checked who is nearest the ball within ±3 frames of each true
possession start (nearest-player distance per team from the sequence features; ball unknown
excluded). A player of the new team is the nearer one in only 47-63% of sub-second
possessions (Metrica 1: 61%, Metrica 2: 47%, Metrica 3: 63%, DFL 1: 55%, DFL 4: 61%). For
possessions of 3 s or more it is 81-93%. These short possessions are contested contacts with
both teams equally close. The 25 Hz positions do not show who touched the ball, which matches
the network's near-chance team probability (0.50-0.55) inside them. A detector trained on the
same tracking geometry would face the same ambiguity, so it was not built.

Conclusion so far: with this tracking data, the sub-second possessions that separate 0.77
from 0.90 do not carry a reliable team signal. Reaching 0.90 would need information the
current inputs do not contain (e.g. event data at inference time, or higher-quality ball and
body tracking).

### Restart-team rule (tested, rejected)

When the ball leaves the pitch, the restart goes to the team that did not touch it last. On
predicted out-of-play stoppages (ball within 1 m of a line or beyond, within 0.5 s of the stop)
with the same team before and after, two post-processing rules were scored on the five
development games (production predictions, mean spell-F1 0.7679):

| Rule | Mean spell-F1 |
|---|---:|
| Give the last 0.5 / 1 / 2 s before the stop to the other team, from its closest approach to the ball | 0.7621 / 0.7637 / 0.7658 |
| Same, fixed 0.5 / 1 s | 0.7620 / 0.7624 |
| Flip the team of the possession after the restart | 0.7376 |

Every variant is worse on every game. In the ground truth, the same team before and after
an out-of-play stoppage happens 1-9 times per Metrica game but 11-23 times per DFL game (that
labelling credits such stoppages differently). The model already uses restarts well, so this
signal adds nothing.
