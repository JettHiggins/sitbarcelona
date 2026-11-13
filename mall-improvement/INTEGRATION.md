# November 11 integration

`main` already contains Lenny (`757f519`) and Mall (`7355027`) through the
pipeline integration commit `5c72081`. Their original branches are retained as
`archive/lenny-pipeline` and `archive/mall-pipeline` tags.

This merge imports all 21 additions from `origin/nov11push` (`2b12f19`):
experiments, diagnostics, trained Route 1 models, reports, and the supplied
`possession-viewer/data/mall-referee-supervised.csv`.
The branch starts from an unrelated challenge-repository history. Its initial
challenge scaffold is intentionally excluded; the current viewer, isolated
pipelines, requirements, and documentation remain the project foundation.
Both histories are recorded as merge parents.

## Use now

Run `python3 possession-viewer/server.py`, open http://127.0.0.1:8081/,
and select `mall-referee-supervised.csv` in the prediction selector. The existing
server discovers this file automatically. Existing Mall and Lenny output options
remain separate; no model blending or frame-specific correction was applied.

Validated with `python -m pipeline.compare` and the official public truth:

| Prediction | Public spell F1 | Public macro F1 |
|---|---:|---:|
| Existing local mall-model.csv | 0.5375 | 0.8503 |
| Imported mall-referee-supervised.csv | 0.5867 | 0.8826 |

Both CSVs have 199,319 ordered frames, valid labels, 40,462 ungraded X frames,
and no abstentions inside graded halves. They disagree on 2.32% of graded frames.
These scores concern the small public sample, not full-match accuracy. The
existing local Mall output is generated data, not a new committed artifact.
The 73 existing pipeline unit tests pass after integration.

## Reproduction gaps in the source branch

The imported experiment scripts are preserved as research source. They are not
yet runnable end to end from this checkout. The branch did not supply:

- `decoder_experiment.py`, `deadball_experiment.py`, `idsse_experiment.py`, or
  `predict_robust.py`, which its scripts import or invoke.
- `experiments/referee_supervised` frozen settings, final trained referee model,
  and deployment manifest; `referee-model/referee_features.npz`.
- IDSSE source/prepared data and supporting experiment caches.

The PowerShell runner also hardcodes its author's Windows Python path. Some
Route 1 cache paths assume a different directory layout. The upstream README's
`pipeline.run mall --mode improved` command is not implemented in either this
checkout or the supplied branch. Continue using the documented existing modes
in `pipeline/README.md`; view the imported CSV directly for the new candidate.

Recover the missing modules, model assets, environment versions, and provenance
from the author before adding these experiments to the common runner. In
particular, Route 1's supplied pickle models alone do not recreate the missing
referee detector or inference helpers. No successful model replay is claimed.
