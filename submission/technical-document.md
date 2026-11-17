# Possession estimation from tracking

## Approach and inputs

The selected algorithm is the repository's Mall trajectory pipeline. It uses
the provided tactical player/ball tracking and Smart Tagging. Per-half offsets
align tracking with the tactical video's 25 Hz frame index. Tracking cleanup
removes implausible jumps, repairs some player/team identity errors, and fills
only short ball gaps. Shape, motion and proximity features provide event cues
for contacts, receptions, kicks and restarts.

A players-only gradient-boosted model supplies dead/live evidence, combined
with rule cues such as pitch boundaries, ball movement, gaps and kickoff
formations. Excluding ball features from the learned stoppage model reduces
dependence on the training feeds' different ball-missingness conventions.

A bidirectional recurrent network estimates ownership and possession-change
probabilities from the nearest three players of each team, ball-relative
geometry, motion and missingness. Team swapping and pitch mirroring augment
training; prediction averages the two team orientations. A semi-Markov decoder
combines learned evidence, event-anchored transitions, switching costs and a
soft duration prior within each predicted live interval.

No video appearance model, skeleton detector or manually corrected match
labels contribute to the selected output. The viewer supports inspection only.

## Training and evaluation data

Training uses ten other matches: three open Metrica Sports sample games and
seven DFL Bundesliga games from the Bassek et al. 2025 dataset. Metrica labels
are derived algorithmically from the published event annotations; DFL labels
come from the provider's ball-status and possession fields. This project did
not create new human labels. The exact number of supervised frames is not
recorded in the supplied production export; the preparation code constructs
the frame-level keys from the source feeds.

Sources and parsing are documented in `pipeline/mall/bench/README.md`,
`metrica.py`, `labels.py` and `idsse.py`. The Metrica source revision is
`e706dd506b360d69d9d123d5b8026e7294b13996`; the seven DFL file mappings are
fixed in `idsse.py`. The challenge match's labels are excluded from training
and parameter selection.

Five other matches informed model and decoder development, each predicted
with models trained on the other games. These are reused development games,
not an untouched test set. The supplied report records mean spell-F1 0.7679
for the production trajectory model. Full-match challenge accuracy is unknown.

## Possession and stoppage rules

Home/away ownership is decoded within live intervals using learned team
evidence and contact/control cues. Loose balls can remain with the current
team until contrary evidence supports a change. Receptions and kicks supply
candidate changes, but proximity alone does not force every switch. The
duration prior is soft; the dead/live stage separately merges very short
stoppages and live islands. Kickoff, outside-pitch and half boundaries constrain
the timeline. Frames outside the published halves are exported as X.

## Alternatives and limitations

The repository also includes boosted ownership, referee-supervised,
interaction, event-first, published-method and touch-only experiments. They
are not blended into this output. The experimental temporal network did not
outperform the production GRU. Wes's uncertain-dead-ball decoder was integrated,
but its small historical three-game gain failed the promotion gate; comparison
against the current ten-game model could not be completed with the local assets.

Sub-second possessions and contested touches remain difficult. Missing or
smoothed ball trajectories, anonymous or switched player identities, and weak
event-derived training labels limit attribution and boundary timing. More
reliable contact evidence and independently labeled other-game tests would be
the next priorities.

## Reproduction and submitted artifact

Install with `python dev.py setup`, prepare the ten training games using
`python dev.py prepare`, then run
`python dev.py model --data-dir PATH_TO_CHALLENGE_BUNDLE`.
The output is `possession-viewer/data/mall-model.csv`.

The submission candidate is an unchanged copy of the author's exported
production predictions, SHA-256
`ce99b78b250e73eaa2f0f6a724d743ef54a47b61da84e6f277ff25a7222d26f9`.
It contains 199,319 ordered predictions. It is not a newly trained hybrid.

The author's recorded environment includes Python 3.10.11, NumPy 1.26.4,
SciPy 1.14.1, scikit-learn 1.5.2 and PyTorch 2.6.0+cu124 on an RTX 4060 Laptop
GPU. The production GRU and players-only dead-ball checkpoints are bundled in
`pipeline/mall/production/`, with their hashes, raw-input hashes and output hash
in `MANIFEST.json`. Direct dependency versions are pinned in
`pipeline/requirements-production.txt`; the PyTorch version and installation
command are specified separately in the production README.

For inference without benchmark downloads or training, follow
`pipeline/mall/production/README.md`: create the production environment, run
`python -m pipeline.mall.production` to verify and install the checkpoints, then
run `python -m pipeline.run mall --mode model --data-dir PATH --output OUTPUT.csv`
from a clean checkout without prepared benchmark caches. Existing prepared
benchmark tags take precedence over the installed checkpoint manifest.

The author reports byte-for-byte reproduction from raw inputs in a fresh
worktree with an empty cache, using the recorded Windows/CUDA environment.
Numerical behavior on other hardware can differ; the expected output hash
above is the verification target.

An independent check of the integrated code at `ac6c5cf` also regenerated the
CSV byte for byte on Linux ARM64 CPU, using Python 3.10.22, PyTorch 2.6.0+cpu
and the other pinned direct dependencies. The checkout began with an empty
cache and no prepared benchmark games; all preprocessing was rerun from the
three raw inputs and the bundled models. The generated output passed format
and half-mask validation and matched the expected SHA-256 exactly. The resolved
CPU environment and verification record are included under `submission/`.
