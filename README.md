The central repository for our project for MoreThanAHack2026

To use the possession viewer
```
cd possession-viewer
python3 -m http.server 8080
```

## MTAH 2027 possession pipeline

Use Python 3.10 (verified on Windows). Clone the official challenge alongside this
repository; the pipeline imports its MIT-licensed starter loaders and uses its
unchanged validator and scorer:

```powershell
git clone https://github.com/metrica-sports/more-than-a-hack-2027.git ../more-than-a-hack-2027
python -m venv .venv
.venv/Scripts/Activate.ps1
python -m pip install -r requirements.txt
```

Enroll and accept the data terms via the [official challenge](https://github.com/metrica-sports/more-than-a-hack-2027#getting-the-data).
Download the licensed bundle from the participant link and unpack its `tactical/`
folder into `data/`. Required files are `BAR-ATH_tactical_atd.txt`,
`BAR-ATH_tactical_atd.xml`, and `BAR-ATH_tactical_smart_tagging.xml`. Video and
homography are not used. Data, video, caches and local dependencies are ignored.

```powershell
# Optional: data bundle root, or the tactical folder itself
$env:MTAH_DATA_DIR = "$PWD/data"
# Optional: another official challenge checkout
$env:MTAH_CHALLENGE_DIR = "$PWD/../more-than-a-hack-2027"
python pipeline/run_all.py
python ../more-than-a-hack-2027/evaluation/validate_submission.py possession-viewer/data/predictions.csv
Copy-Item possession-viewer/data/predictions.csv ../more-than-a-hack-2027/evaluation/results/lenny_pipeline.csv
python ../more-than-a-hack-2027/evaluation/run_evaluation.py --bootstrap 1000
python -m unittest pipeline.test_invariants
```

The generated file is `possession-viewer/data/predictions.csv`: exactly 199,319
ordered frame rows, plus `frame,label`. Open it through the viewer's **Predictions
CSV** picker alongside your tactical video. The viewer code is unchanged.
For staged reproduction use `--stage nearest`, `--stage dead`, `--stage decoder`,
then the default `--stage model`. Each command overwrites the output CSV.
All settings and the fixed seed are in `pipeline/config.py`. Delete `cache/` to
force the starter loader to reload the raw tracking; the cache is keyed by the
SHA-256 of both tracking inputs and holds only normalized source arrays.
The verified official challenge revision is `f8f2629ff1e0985f81d7e59d8413678514ee1530`;
check out that revision in the sibling repository to reproduce the loader and scorer.

### Method and limits

Layer 0 aligns `frame_id - 1`, scales to 105.3 by 68 metres, rejects impossible
ball jumps, interpolates gaps shorter than one second, and smooths separate
available ball stretches. Layer 1 measures proximity and collective team geometry.
Layer 2 confirms receptions after 0.3 seconds of uncontested co-motion, detects
kicks, and excludes fast reversals and unresolved duels. Layer 3 combines line,
speed, restart geometry, kickoff formations and automatic set-piece evidence,
with minimum dead/play durations and dead time before each detected kickoff.
Restart ownership comes from players at the relevant spot.

Layer 4 trains gradient boosting on algorithmically certain control and completed
same-team passes. It uses only player centroids, spreads, speeds and pressing
distances averaged over 3/10/30 seconds. Each time block is predicted by a model
trained outside that block, with 30 seconds purged at both edges, team-swapped
training examples and symmetric prediction. No human labels, public ground truth,
or external match annotations enter the pipeline. Layer 5 uses dynamic programming
over reception/kick/restart/gap boundaries with switching and duration costs.
Loose and contested balls retain possession until admissible evidence resolves them.
Layer 6 masks only the published ungraded frame ranges with `X`.

The public five-minute ground truth is used only by the official scorer for sanity
checks. Stage scores are observations, never a parameter-selection loop. Automatic
tags can be late or wrong, ball gaps remain ambiguous, duplicate tracks affect
team geometry, and pseudo-label errors can propagate. The public sample contains
only 17 possessions; its score does not establish full-match performance.

See [stage results, bootstrap scoreboard and reproducibility hash](pipeline/RESULTS.md).
