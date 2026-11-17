# Possession pipelines and viewer

Team repository for MoreThanAHack2026. Clone, install, and test without any private
match data or another developer's local caches.

## Quick start

Install **Git** and **64-bit Python 3.10** (the tested baseline), available on your
PATH. These commands work in Windows PowerShell, macOS, and Linux. On systems
where Python is named `python3`, use that instead of `python` throughout.
Linux distributions may require their `python3-venv` package first.

```sh
git clone https://github.com/JettHiggins/sitbarcelona.git
cd sitbarcelona
python dev.py setup
```

Setup creates `pipeline/.venv`, installs the pipeline dependencies and CPU PyTorch,
checks dependency compatibility, and runs the complete pipeline unit test suite.
No environment activation, CUDA, match downloads, or copied `.deps` folders are
required. Internet access is required for installation. Re-running setup reuses
the environment; a failing install or test exits with an error.

The tests use synthetic inputs, including a small training/prediction test, team
symmetry checks, and CSV export checks. They verify functionality, not real-match
accuracy. Run them again after edits:

```sh
python dev.py test
```

GitHub Actions runs this same setup on fresh Windows and Linux machines with
Python 3.10. Dependencies are resolved from `pipeline/requirements.txt`; exact
versions may vary across platforms and installation dates. The older
`requirements-tested.txt` is a historical environment snapshot, not the setup lockfile.

## Open the viewer

```sh
python dev.py viewer
```

Open **http://127.0.0.1:8081/**. Stop the server with Ctrl+C. The viewer itself
needs only Python's standard library and can launch before setup. A fresh clone
includes three prediction snapshots in `possession-viewer/data/`, available in the
viewer's prediction selector:

- `mall-model.csv`: the current trajectory-model predictions.
- `mall-boosted.csv`: the gradient-boosted model predictions for comparison.
- `mall-referee-supervised.csv`: Mall's November 11 referee-motion candidate.

Lenny's original pipeline, Mall's pipeline, and both teammates' November updates
are integrated on `main`. The referee candidate's CSV is usable directly, but its
inference scripts still need missing upstream dependencies; see the
[integration notes](mall-improvement/INTEGRATION.md). The trajectory and boosted
snapshots above are supplied by Lenny's November 12 branch.

Each file contains 199,319 ordered frame predictions with H/A/D labels and X for
ungraded frames. These are the existing model outputs, not results from the
unfinished temporal-model experiment. No training or match tracking data is
needed to load them. Watching the actual match requires the separately supplied
video. A small synthetic CSV is also available at
`possession-viewer/tests/fixtures/comparison-fixture.csv`.

## Generate predictions for the real match

The challenge data bundle is distributed separately and is not in Git. Obtain it
from your team's existing challenge-data access. Keep these files together:

```text
challenge-data/
  tactical/
    BAR-ATH_tactical_atd.txt
    BAR-ATH_tactical_atd.xml
    BAR-ATH_tactical_smart_tagging.xml
```

After setup, prepare the public training games, then run Mall:

```sh
python dev.py prepare
python dev.py model --data-dir "C:/path/to/challenge-data"
```

Replace the example path with your bundle's location; macOS/Linux paths work too.
Alternatively set `MTAH_DATA_DIR`. Preparation clones Metrica sample data at
revision `e706dd506b360d69d9d123d5b8026e7294b13996`, downloads the seven DFL games
(about 2.6 GB in addition to Metrica), and builds the training features. An existing
clean Metrica checkout can be supplied with `python dev.py prepare --bench-src PATH`.
Preparation never resets an existing checkout.

The first model run trains the models; allow substantial time on CPU and about
8 GB of free memory. Later runs reuse trained models. The output is
`possession-viewer/data/mall-model.csv`; start/reload the viewer to select it.
Running the model replaces this tracked snapshot locally; review the resulting
CSV change before committing an updated prediction snapshot.
Match data, downloaded benchmarks, trained models, and generated reports remain
local and are recreated through these commands.

See [pipeline commands and alternatives](pipeline/README.md) and the
[optimization experiment results](pipeline/MALL_SCORE90_PROGRESS.md).

The November 17 integration also includes Wes's experimental dead-ball decoder
and viewer comparisons. `python -m pipeline.run wes` uses the current Mall
evidence; it is not promoted over Mall without a production-baseline comparison.
See [Wes evaluation and reproduction](pipeline/wes/README.md).
The [November 17 integration decision](pipeline/NOV17_INTEGRATION.md) records
the checks and remaining limitations; [submission artifacts](submission/README.md)
include the recommended candidate's manifest and a draft technical document.
