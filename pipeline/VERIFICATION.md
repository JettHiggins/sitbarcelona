# Integration verification — 2026-11-08

Both source implementations were selectively extracted; no algorithm constants were tuned.
Commands ran on the local BAR-ATH tactical tracking and Smart Tagging files, using Python 3.14
on Linux ARM64 and the versions in `requirements-tested.txt`.

## Measured comparison

The unchanged official scorer was run against the same public five-minute ground truth.
These are measured integration results, not copied from the branches' reports.

| Output | Public spellF1 | Public macroF1 | Dead frames in graded halves |
|---|---:|---:|---:|
| Official baseline smooth50 | 0.3256 | 0.6207 | 31,058 |
| Lenny decoder, without model | 0.4571 | 0.6795 | 17,621 |
| Lenny full model | 0.4571 | 0.6795 | 17,621 |
| Mall events-only ablation | 0.3238 | 0.7753 | 83,249 |
| Mall full model | 0.5375 | 0.8503 | 83,249 |

Lenny and Mall's full models disagree on 46.94% of the 158,857 graded frames. Their dead-ball
coverage differs substantially: 11.09% versus 52.41%. This is a reason to inspect the viewer,
not evidence that either is correct across the full match. Public truth contains only 17
possessions. No public labels were used to fit models or select parameters during integration.

Lenny's model changes 2.45% of graded predictions relative to its decoder-only mode, while
leaving the public scores unchanged. Mall's model changes 7.52% of graded predictions relative
to the additional events-only ablation. That ablation is not Mall's historical v0 pipeline.

Mall's source report records 0.5625 spellF1 / 0.8482 macroF1. This integration does not exactly
reproduce that result. Its source also records a subsequent revision to benchmark answer keys;
the integration builds those keys from the current source. Dependency versions differ too.
The difference has not been isolated to one cause; parameters were left unchanged.
Lenny's public scores reproduce its source report, but its full-match model CSV hash differs
from the original Python 3.10 environment.

## Checks completed

- Nine unit tests pass: the original four Lenny invariants, viewer export/masking, independent
  cache paths, baseline/abstention acceptance, and duplicate-frame rejection.
- Both full model CSVs pass the unchanged official `validate_submission.py`.
- All four generated variants have exactly 199,319 ordered frames, valid labels, and exactly
  40,462 ungraded frames marked X. None abstains inside the graded halves.
- Benchmark preparation successfully built all three Metrica games and layers 0–3 for each.
  Sample-data checkout: `e706dd506b360d69d9d123d5b8026e7294b13996`.
- Mall trained frame and change-point models on all three benchmark games and completed decoding.
- Vendored official loaders, game metadata, and metrics match their source bytes.
- All Python modules compile. Git whitespace checks pass.
- The possession viewer's HTML, JavaScript, CSS, and frame verifier are unchanged.

Generated CSVs, tracking caches, benchmark labels, models, and the virtual environment are
ignored by Git. The local CSVs are ready to load through the possession viewer's file picker.
Video playback was not retested because the viewer application was not modified.

## Output SHA-256

| CSV | SHA-256 |
|---|---|
| lenny-decoder.csv | `e613c6df4f93f58c850f3707e5f1819d22c8ae1b34321ec4bccc3348f13dc6ca` |
| lenny-model.csv | `4abe66c5a9b76e31c18ff0138fa916982a2e5c2f3af4a4c99d67779bd52fa932` |
| mall-events.csv | `aab2277a8389d8e823d4d2f3486692465a22997649356360979acedd2c9bab0d` |
| mall-model.csv | `fd07fd4e2e57686668a001fa42cb516ebe930f50f74a8afb5c809aa3ed49a898` |

To rerun validation and scores, use `python -m pipeline.compare` as documented in README.md.
