# Wes possession experiments

Wes is the experimental pipeline on `wes_deadball+video`. Mall's `mall --mode model`
CSV is the fixed comparison. Wes reuses Mall Layers 0–4 (clean tracking, physical
features, touch events, dead-ball cue scores and trained ownership/change-point
probabilities), then tests a different segmentation decision before using the
existing sequence objective. This avoids training and maintaining a second copy of
the same baseline model. The small Mall hooks record Layer 3's intermediate states
and let Layer 5 accept an optional dead mask and extra candidates; its default
Mall path is unchanged.

## Run from the repository root

The local setup has Python 3.12 in `pipeline/.venv`, the gated tactical assets in
`data/tactical/`, and three other Metrica games in `data/sample-data/`. All generated
data below is Git-ignored.

```bash
git branch --show-current                              # wes_deadball+video
pipeline/.venv/bin/python -m pipeline.run mall --mode model
pipeline/.venv/bin/python -m pipeline.run wes --mode revisable --reuse-cache
pipeline/.venv/bin/python -m pipeline.run wes --mode candidates --reuse-cache
pipeline/.venv/bin/python -m pipeline.run wes --mode joint-d --reuse-cache
pipeline/.venv/bin/python -m pipeline.run wes --mode joint-team --reuse-cache
pipeline/.venv/bin/python -m pipeline.run wes --mode joint --reuse-cache
pipeline/.venv/bin/python -m pipeline.compare \
  possession-viewer/data/mall-model.csv possession-viewer/data/wes-candidates.csv
```

Mall writes `possession-viewer/data/mall-model.csv`. Wes writes separate
`wes-revisable.csv` and `wes-candidates.csv` files, each with an adjacent
`.segmentation.json` trace. The compare command checks row count, labels and
disagreement; it is not a ground-truth score. `wes --mode candidates` is an
experiment that **failed** the current held-out promotion gate, so do not treat its
CSV as the chosen submission.

The joint experiments write separate `wes-joint-d.csv`, `wes-joint-team.csv`,
and `wes-joint.csv` outputs. Their local semi-Markov decoder scores D/H/A
segments together only in windows selected from tracking/model disagreement.
`joint-d` permits D revision while keeping existing live team labels fixed;
`joint-team` permits H/A revision while keeping D fixed; `joint` permits both.
Outside selected windows, Mall's labels are unchanged. Hard kickoff, ungraded,
and clearly outside-ball frames remain protected. The existing ownership score
chooses H versus A but cannot independently turn D into live. The first joint
trial uses existing score scales rather than fitting new calibration parameters.
None of these three modes passes the current held-out promotion gate.

Omit `--reuse-cache` on a Wes run to rebuild the shared Mall match stages and
refresh the baseline CSV first. Use `--reuse-cache` after changing only files in
`pipeline/wes/`: it reuses the already-built Mall evidence and finishes much faster.
If you edit Mall cleaning, cues, features or trained model code, rebuild the three
other-game cache and then the match baseline:

```bash
pipeline/.venv/bin/python -m pipeline.run prepare-mall --bench-src data/sample-data
pipeline/.venv/bin/python -m pipeline.run mall --mode model
```

## Evaluate a change

Change **one Wes setting at a time** in `pipeline/wes/config.py`. The dead/live
settings govern minimum reopened span length, transition penalty, outside-ball
protection, observed-control support and bounded team evidence. The candidate
settings govern ambiguous-team radius, later-recovery lookahead/blend, and the
soft contact bonus. Each candidate still enters Mall's sequence objective as a
hypothesis, never an automatic turnover.

```bash
pipeline/.venv/bin/python -m unittest pipeline.wes.test_segmentation
pipeline/.venv/bin/python -m unittest pipeline.wes.test_joint
pipeline/.venv/bin/python -m pipeline.wes.evaluate
pipeline/.venv/bin/python -m pipeline.wes.diagnose
```

The evaluation is leave-one-game-out: for each of `bench1`, `bench2`, `bench3`,
the ownership/change-point models train on the other two. It reads **only** the
other-game event-derived answer keys. Review each game and the gate in
`pipeline/reports/wes/segmentation-evaluation.json`, then record the hypothesis,
source revision, scores, counts, precision/recall, boundary timing, regressions
and explanation in `pipeline/WES_EXPERIMENT_LOG.md`. These games were used in
earlier development and are not an untouched test. Do not tune on the public
BAR–ATH five-minute labels or hand-correct the competition match.

`diagnose` writes the ignored
`pipeline/reports/wes/missed-spell-diagnosis.json`. It compares unmatched
possessions at IoU 0.5, records their majority predicted label, traces false-D
frames through Layer 3, and checks whether missed direct H/A switches had an
accessible candidate, positive target bonus, or favorable ownership evidence.
These are overlapping error mechanisms summarized into descriptive buckets,
not causal proof or a score improvement by themselves.

The project promotion gate requires mean held-out `spellF1` at least +0.02,
improvement in at least two of three games, no per-game loss worse than −0.02,
and a reproducible valid 199,319-row output. This is a project rule, not a
competition threshold. The revisable and candidate modes scored slightly below
Mall on all three games. `joint-d` had a negligible mean gain and the other
joint modes regressed, so Mall remains the selected version.

## Inspect in the viewer

```bash
pipeline/.venv/bin/python possession-viewer/server.py
```

Open `http://127.0.0.1:8081/`. Use the file pickers for
`data/tactical/BAR-ATH_tactical.mp4` and first `mall-model.csv`, then switch to a
`wes-*.csv` to compare the same tactical frames. The adjacent JSON trace explains
which dead spans Wes reopened and which candidate points it considered. Visual
inspection is diagnostic; do not make competition-match labels by hand.

## Later video stage

The tactical BAR–ATH video is available, but the public three-game sample-data
checkout has no synchronized other-game videos. Run
`python -m pipeline.wes.video.audit --video-root PATH` when separately permitted
other-game footage is available. A visual evidence layer and a TCN remain separate
experiments; neither has a measured score or a submission-ready output yet.
