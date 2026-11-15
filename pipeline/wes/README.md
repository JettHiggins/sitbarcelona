# Wes possession pipeline

`wes` has one supported output: the former `joint-d` decoder, now called
`wes-model.csv`. It shares Mall's cleaned tracking, event cues and learned
ownership evidence (Layers 0–4). It first obtains Mall's Layer 5 prediction,
then revisits uncertain dead-ball windows with a local semi-Markov decoder.
Already-live H/A labels remain fixed. Clear outside-ball, kickoff and halftime
constraints remain protected. The CSV and adjacent `.segmentation.json` trace
are generated locally and ignored by Git.

The historical revisable, candidate, joint-team and unrestricted joint
experiments are documented in [the experiment log](../WES_EXPERIMENT_LOG.md).
Their score regressions are why they are no longer selectable. The previous
`joint-d` algorithm had mean held-out `spellF1` 0.6833 versus Mall's 0.6826
over three reused other-game keys. This tiny numerical gain does **not** pass
the project's +0.02 promotion gate and is not evidence of private-match
superiority. The user selected it as the single Wes collaboration path; Mall
remains available as a separate comparison baseline.

## Run from the repository root

The local setup uses Python 3.12 in `pipeline/.venv`, competition assets in
`data/tactical/`, and permitted other-game sample data in `data/sample-data/`.
On another machine, prepare the three training games first:

```bash
pipeline/.venv/bin/python -m pipeline.run prepare-mall --bench-src data/sample-data
pipeline/.venv/bin/python -m pipeline.run wes
```

The Wes run rebuilds shared Mall match evidence and writes both
`possession-viewer/data/mall-model.csv` and
`possession-viewer/data/wes-model.csv`. After changing **only** Wes code, reuse
the existing evidence:

```bash
pipeline/.venv/bin/python -m pipeline.run wes --reuse-cache
```

To choose an explicit submission path, add `--output /path/to/submission.csv`.
The output has `frame,label` columns for frames 0–199318. The diagnostic JSON
uses the same basename. Do not commit licensed data or generated outputs.

## Validate and compare

```bash
pipeline/.venv/bin/python -m unittest pipeline.wes.test_joint pipeline.wes.test_diagnose
pipeline/.venv/bin/python -m pipeline.wes.evaluate
pipeline/.venv/bin/python -m pipeline.wes.diagnose
pipeline/.venv/bin/python -m pipeline.compare possession-viewer/data/mall-model.csv possession-viewer/data/wes-model.csv
```

Evaluation is leave-one-game-out: the learned models for each of `bench1`–
`bench3` train on the other two. The event-derived answer keys are approximate
and have already informed development; they are not an untouched test. The
evaluation and diagnosis reports under `pipeline/reports/wes/` are ignored by
Git. The public five-minute BAR–ATH labels are for alignment checks only, not
training or choosing settings. No competition-match frames may be hand-labeled
or corrected for the pipeline.

## Inspect in the viewer

```bash
pipeline/.venv/bin/python possession-viewer/server.py
```

Open `http://127.0.0.1:8081/`, select the tactical MP4 locally, then choose
Wes under **Compare with Mall**. The model outputs and local diagnostic JSON
are served from `possession-viewer/data/`; the video stays in your browser.
The viewer highlights differences, not private-match correctness.

The later video evidence layer remains an experiment. The current sample-data
checkout has no synchronized other-game video, so it has no measured score or
submission-ready output.
