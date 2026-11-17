# Production Mall checkpoints

These are the two learned models behind `possession-viewer/data/mall-model.csv`. With them installed,
you can reproduce that CSV from the challenge tracking data alone. You do not need the ten
benchmark games (about 2.6 GB) and no training runs.

| File | Model | Code |
|---|---|---|
| `seq_bench1+…+dfl7.pt` | Trajectory model: a bidirectional GRU over raw tracking, as a torch `state_dict` with seed 0 | `layer4_possession_model/sequence.py` |
| `deadp_bench1+…+dfl7.pkl` | Players-only learned dead-ball model: a pickled `HistGradientBoostingClassifier` plus its home/away swap index | `layer3_dead_ball/learned.py` |

Both were trained on all ten benchmark games: Metrica bench1–3 and DFL dfl1–7. `MANIFEST.json`
records their SHA-256 hashes, the hashes of the input files and of the CSV, and the environment
they were made in.

## Environment

| | Version used |
|---|---|
| OS | Windows 11 x64 (10.0.26300) |
| Python | 3.10.11, 64-bit |
| PyTorch | 2.6.0+cu124 |
| CUDA / cuDNN | 12.4 / 9.1.0 |
| GPU | NVIDIA RTX 4060 Laptop, driver 621.14 |
| Python packages | [`pipeline/requirements-production.txt`](../../requirements-production.txt), with every version pinned exactly |

The dead-ball model is a pickle, so it needs **scikit-learn 1.5.2**. A newer
`pipeline/requirements-tested.txt` environment may fail to load it. A GPU is not required:
inference falls back to the CPU, which is slower. The CPU output has not been checked against
the CSV by the original author. The integrated repository was independently
verified on Linux ARM64 CPU below.

```bash
python -m venv .venv-prod && . .venv-prod/bin/activate     # Windows: .venv-prod\Scripts\activate
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r pipeline/requirements-production.txt
```

## Inputs (not in the repo)

The challenge data is licensed, so it is not committed. Put these three files in `data/tactical/`,
or pass `--data-dir` (or set `MTAH_DATA_DIR`) to point at the folder that holds `tactical/`:

- `BAR-ATH_tactical_atd.txt`: raw tracking
- `BAR-ATH_tactical_atd.xml`: tracking metadata
- `BAR-ATH_tactical_smart_tagging.xml`: Smart Tagging phases

Their SHA-256 hashes are in `MANIFEST.json`.

## Inference

Run these from the repository root:

```bash
python -m pipeline.mall.production                 # check hashes, copy into pipeline/cache/mall/models
python -m pipeline.run mall --mode model           # writes possession-viewer/data/mall-model.csv
# optional: --data-dir /path/to/data  --output some/other.csv
```

The run rebuilds every preprocessing layer from the raw tracking:

- layer 0 cleans the tracking
- layer 1 builds the features
- layer 2 detects touches
- layer 3 detects dead balls

It then loads the two checkpoints (layers 3 and 4) and decodes the result (layer 5). No other
assets are needed. Layer 4 prints "trajectory model trained on …" even though it only loaded the
checkpoint.

**Verified 2026-11-17:** I made a fresh `git worktree` with an empty cache and no benchmark games,
then ran the two commands above with the environment listed here. The result matched the
committed `mall-model.csv` byte for byte (SHA-256 `ce99b78b…`). The run took 2.9 minutes on the
RTX 4060. All derived data goes to `pipeline/cache/mall/`, which is gitignored and safe
to delete. The run needs about 8 GB of free RAM.

**Independent integrated-code verification, 2026-11-17:** a fresh worktree at
`ac6c5cf`, with an initially empty cache and no prepared benchmark games, also
reproduced the exact same CSV on Linux ARM64 using Python 3.10.22 and
PyTorch 2.6.0+cpu. All three raw-input hashes and both checkpoint hashes matched
this manifest. The run used `OMP_NUM_THREADS=2` and `OPENBLAS_NUM_THREADS=2`,
with the other direct requirements pinned as above. Its resolved environment
is recorded in `submission/cpu-verification-environment.txt`, and the evidence
is in `submission/reproduction-verification.json`. No training was performed.

## Cache notes

- The checkpoints are found by filename: `{seq,deadp}_<training games joined by +>.{pt,pkl}` in
  `pipeline/cache/mall/models/`. If a file with that name is already there, it is used, and no
  training runs. If you run `python -m pipeline.mall.production` and an older cached file has
  different contents, that file is kept as `*.replaced`.
- If you have no benchmark games prepared, `pipeline/cache/mall/models/production.json` (written
  by the installer) tells the pipeline which games the checkpoints were trained on. When you do
  prepare the benchmark (`python -m pipeline.run prepare-mall`), those games take precedence.
- To retrain from scratch instead, delete the two installed files and follow the
  `prepare-mall` steps in `pipeline/README.md`.
- The `boosted` and `interaction` modes use other models and still need the prepared benchmark.
