# November 17 integration and submission decision

Integrated sources:

- `main` at `05fbb1d`: portable setup, ten-game Mall model, shared prediction
  snapshots and current playback implementation.
- Lenny `4c7381a`: deterministic temporal-convolution acceleration and completed
  experiments. The experimental temporal model is not the production GRU.
- Wes `c41da73`: constrained dead-ball revision, diagnostics and viewer comparisons.
- Earlier Lenny, Mall and November 11 work was already integrated on `main`.

## Decision

Keep the supplied `possession-viewer/data/mall-model.csv` trajectory-model
snapshot as the recommended submission candidate. Its SHA-256 is
`ce99b78b250e73eaa2f0f6a724d743ef54a47b61da84e6f277ff25a7222d26f9`.
It is unchanged from Lenny's validated export. No prediction blending, manual
frame edits, public-window selection or algorithm retuning was performed.

Lenny's reported five-game development mean is 0.7679 spell-F1. His temporal
network, ensembles and decoder sweeps did not establish a meaningful gain.
See [the recorded experiments](MALL_SCORE90_PROGRESS.md). Those training runs
were not repeated on this machine.

The merged Wes implementation was run against the locally available historical
three-game evidence, with the historical rules and decoder weights:

| Game | Legacy Mall spell-F1 | Wes spell-F1 |
|---|---:|---:|
| Metrica 1 | 0.6902 | 0.6888 |
| Metrica 2 | 0.6936 | 0.6961 |
| Metrica 3 | 0.6616 | 0.6625 |
| Mean | 0.6818 | 0.6825 |

Gain: 0.000666, below the +0.02 promotion gate. The absolute Game 2 scores differ
from Wes's original report; these local older caches and environment do not
establish bit-exact reproduction of his run. The machine-readable local report
is `pipeline/reports/wes/nov17-legacy-integration.json` (Git-ignored).

This is not a test against the current production model. The seven DFL training
games and ten-game trained checkpoints are absent locally. The production
comparison refuses to use the old caches as substitutes. Consequently no current
Mall-plus-Wes superiority is established, and the production CSV is preserved.

## Integration details

The dead-ball detector retains learned probabilities and gap gating while
optionally returning Wes's stage traces. The runner retains ten-game training,
boosted and interaction modes, and adds the Wes command. The viewer retains
native playback, scrubbing and worker decoding with the new comparison controls.

The evaluator now defaults to rebuilding production evidence on the five
development games, with each game excluded from training. Use
`python -m pipeline.wes.evaluate --baseline legacy` only to reproduce the older
experiment. Promotion averages and regression checks include every evaluated
game, including DFL. Missing stage traces produce an explicit cache-rebuild error.

## Checks

- 104 pipeline tests pass, including learned-evidence trace equivalence,
  temporal convolution output/input-gradient equivalence, and evaluation guards.
- Node comparison utility and JavaScript syntax checks pass.
- Browser checks cover full CSV comparison/navigation/filtering, scrubbing,
  repeated native pause/resume, and reverse worker decoding.
- The exact selected CSV is copied to `submission/sitbarcelona.csv` and validated
  with the official format validator and the repository's mask checks.

## Reproduction limitation

The source includes training and inference commands, but the author's production
checkpoints and complete originating environment are not supplied. Training on
another device can change the final CSV. The shipped snapshot is byte-identical
and auditable; bit-exact regeneration from raw match data has not been verified.
This was the state before the checkpoint update below.

## Production checkpoint update

Lenny's `7357580` now supplies the production GRU and learned dead-ball models,
SHA-256 hashes for both checkpoints and all three raw input files, pinned
production requirements and inference-only installation commands. Both bundled
checkpoint hashes and all three local raw-input hashes match the manifest.
The author reports byte-identical reproduction in a clean worktree with an
empty cache on his recorded Windows/CUDA environment. The update merges cleanly
with the integrated code, and all 104 pipeline tests still pass.

The author-supplied checkpoints enable challenge inference without downloading
or training on the benchmark data. They do not supply the nine-game-exclusion
models needed to complete the current production Mall/Wes development comparison.
The selected submission CSV remains unchanged.

An independent clean replay of the integrated code at `ac6c5cf` completed on
Linux ARM64 CPU with Python 3.10.22 and PyTorch 2.6.0+cpu. It started with no
tracking caches or benchmark data, installed the bundled checkpoints, and
rebuilt all six layers from the raw challenge tracking and Smart Tagging files.
The generated CSV is byte-identical to the selected submission snapshot:
`ce99b78b250e73eaa2f0f6a724d743ef54a47b61da84e6f277ff25a7222d26f9`.
The raw-data reproduction gap identified earlier is therefore closed for this
verified environment. See `submission/reproduction-verification.json` and the
resolved CPU environment file for the recorded evidence.
