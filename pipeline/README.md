# Possession pipelines

Possession implementations feed the existing possession viewer. Run commands from the
repository root. Generated CSVs and caches stay local; viewer code is unchanged.

## Install and run

```bash
python3 -m venv pipeline/.venv
source pipeline/.venv/bin/activate
pip install -r pipeline/requirements.txt
export MTAH_DATA_DIR=/path/to/challenge-data
python -m pipeline.run lenny --mode model
python -m pipeline.run mall --mode events
```

The data bundle must contain `tactical/BAR-ATH_tactical_atd.txt`,
`BAR-ATH_tactical_atd.xml`, and `BAR-ATH_tactical_smart_tagging.xml`.
Alternatively pass `--data-dir /path/to/data` (or its tactical folder) to either run.
Video is not used by the pipelines.

Outputs default to `possession-viewer/data/lenny-model.csv` and
`possession-viewer/data/mall-events.csv`. Use `--output PATH` to select another destination.
Open the tactical video and one CSV using the viewer's existing file pickers. Switch the CSV
picker to compare predictions at the same video frame. Serve with
`python3 possession-viewer/server.py` and open <http://127.0.0.1:8081/>.

## Mall's complete model

For the separate **touch-only experiment**, run
`python -m pipeline.run touches --bench-src /path/to/metrica-sample-data --reuse-cache`.
It learns player contact scores from the three sample games and immediately switches
to each detected toucher's team, with no possession decoder or dead-ball labels.
See [research, training, diagnostics and measured limitations](touches/README.md).
Its viewer output is `touches.csv`; Mall remains the stronger possession model.

The default `mall --mode model` uses supervised models trained on three other Metrica games.
It needs the benchmark preparation step; it never silently substitutes event-only predictions.

```bash
git clone --depth 1 https://github.com/metrica-sports/sample-data.git /tmp/metrica-sample-data
python -m pipeline.run prepare-mall --bench-src /tmp/metrica-sample-data
python -m pipeline.run mall --mode model
```

Preparation builds benchmark labels and layers 0–3 for all three games. The model run then
trains the frame and change-point models and exports `possession-viewer/data/mall-model.csv`.
`mall --mode events` is an additional ablation: Mall's current decoder with learned frame and
change-point evidence disabled. It is neither the full model nor Mall's historical v0 model.

`mall --mode interaction` adds an experimental learned contact/recovery layer to the full
model and exports `mall-interaction.csv`, leaving `mall-model.csv` available for comparison.
See [the interaction model](mall/interaction/README.md) for its training protocol, diagnostic
output, and the explicit `--reuse-cache` option. The original `model` mode remains the default.

For an independent event-first ownership system, use `python -m pipeline.run event-first`.
It exports `event-first.csv` without Mall's ownership probabilities or sequence decoder;
tracking cleanup and dead-ball detection are shared. See [event-first rules and commands](event_first/README.md)
and [measured results](EVENT_FIRST_RESULTS.md). Both variants remain experimental.

`python -m pipeline.run paper --reuse-cache` runs the separate reconstruction of Vidal-Codina
et al. (2022), using the published gain/loss rules, parameter profiles and event trees. It
exports `paper.csv`. Read [the fidelity map and input assumptions](paper2022/README.md): the
paper expects a game-status signal and outputs richer player states than the challenge's H/A/D.

Lenny also supports `--mode nearest`, `dead`, and `decoder` (without its learned model).
Every mode has a separate default output filename.

## Compare and validate

```bash
python -m pipeline.compare possession-viewer/data/lenny-model.csv possession-viewer/data/mall-model.csv
# Optional: official public-sample scores; labels are used only by this comparison command.
python -m pipeline.compare possession-viewer/data/lenny-model.csv possession-viewer/data/mall-model.csv \
  --truth /path/to/challenge/evaluation/gt/BAR-ATH_possession_gt_public.csv
python -m unittest pipeline.test_adapters pipeline.lenny.test_invariants
```

The comparison checks all 199,319 ordered frame IDs, valid labels, then reports the ungraded mask,
label counts and pairwise disagreement within the published graded halves. Official baselines
may label ungraded frames D instead of X; those frames are excluded from disagreement. The unchanged official metrics implementation
is included for optional scoring. Counts verify alignment, not prediction accuracy. The public
sample has only 17 possessions; full-match agreement with Smart Tagging is a noisy proxy.

## What differs

| Area | Lenny | Mall |
|---|---|---|
| Tracking ingestion | Official kloppy loader | Direct vectorized ATD parser |
| Cleaning | Ball jump rejection, short linear gap filling, smoothing | Ball tracklets, Hermite gap filling, player smoothing, duplicates and team-label repair |
| Touch evidence | Sustained control/co-motion and abrupt kicks | Speed minima, heading changes, receptions, releases, kicks and gap events |
| Dead ball | Weighted cues, minimum durations and restart inference | Multiple geometry/movement/tag cues, temporal decoding and boundary snapping |
| Learned possession | Self-training on certain controls/passes in this match; purged time blocks | Cross-game supervised frame and change-point models; also computes self-trained reference models |
| Final sequence | Event-constrained dynamic programming with short-spell penalties | Event and learned change points with log-normal duration prior |
| External training games | None | Three Metrica sample games for full mode |
| CSV export | Adapted original writer | Added numeric-label adapter, with ungraded frames mapped to X |

Both pipelines are offline: smoothing, future events and contextual windows use later frames.
They are not causal live possession detectors.

For the current Mall implementation, measured public errors, evaluation limitations, and
recommended additions, see [the pipeline audit](MALL_PIPELINE_AUDIT.md).

## Isolation and source provenance

- `lenny/`: extracted from `origin/lenny-pipeline`, commit
  `757f519e6d8cc32bffad770ec604d3020eb56833`.
- `mall/`: runtime layers and benchmark preparation from `origin/mall-pipeline`, commit
  `735502706dffc0a93f199659edcfbf78e6f2f591`.
- `vendor/challenge/`: unchanged official tracking/tagging loaders, game metadata, metrics and
  MIT license, as included in Mall's source branch. `mall/games.py` is the same metadata with
  its default data path adapted; its license is preserved beside it.
- Relocation changes package imports and paths, not algorithm constants. The event-only Mall
  mode changes model weights explicitly inside its process. The common runner adds orchestration
  and separate viewer filenames. Source layer READMEs and Lenny's RESULTS describe the original
  branches; their evaluation/sweep commands are not included in this runtime extraction.
- No copied challenge repo, notebooks, benchmark CSVs, match assets, or source-branch prediction
  CSV is imported. Benchmark-building code is retained because Mall's full model requires it.

Caches live under `pipeline/cache/lenny/` and `pipeline/cache/mall/`; they cannot overwrite each
other. Match stages are rebuilt on each common-runner invocation. Mall rebuilds its tactical feature matrix on full runs and invalidates trained models after
benchmark preparation. When changing cleaning, features or model parameters, prepare again
so the benchmark caches and trained models stay consistent. Lenny fingerprints raw tracking inputs for its loader
cache; clear its cache when changing loader versions or pitch/frame metadata. Each output path
is overwritten when that same mode is rerun.

`requirements-tested.txt` records the exact environment used for the integration check (Python
3.14 on Linux ARM64). It differs from Lenny's original Python 3.10 environment; identical
full-match model output across those environments is not promised. Public-sample reproduction
results are recorded in `VERIFICATION.md`.

Both source branches should be retired only after this extraction has been committed and merged
into main. Their commit IDs above retain a record of the original work.
