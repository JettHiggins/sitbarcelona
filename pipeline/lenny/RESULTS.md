# Verification results

Official challenge revision: `f8f2629ff1e0985f81d7e59d8413678514ee1530`.
Python 3.10.11, Windows, pinned `requirements.txt`, seed 2027.
All four stages passed the unchanged official validator (199,319 ordered rows).
No scorer output was used to select settings or train the model.

## Requested stage checks

| Stage | macroF1 | spellF1 | Public possessions | Mean duration (s) |
|---|---:|---:|---:|---:|
| baseline_smooth50 | 0.6207 | 0.3256 | 26 | 10.8 |
| Layer 0 + nearest team | 0.5976 | 0.0376 | 132 | 2.0 |
| + Layer 3 dead ball | 0.6291 | 0.0424 | 134 | 1.9 |
| + Layer 5 decoder | 0.6795 | 0.4571 | 18 | 15.4 |
| + Layer 4 model (final) | 0.6795 | 0.4571 | 18 | 15.4 |

The player model changes full-match predictions, but leaves the public window
unchanged. The final output remains the model stage, independent of public ranking.

## Official scoreboard, 1,000 bootstrap draws

Command: `python ../more-than-a-hack-2027/evaluation/run_evaluation.py --bootstrap 1000 --no-detail`.
The generated CSV was copied into that checkout's `evaluation/results/lenny_pipeline.csv`.

| Rank | Submission | macroF1 | spellF1 | spellF1 bootstrap interval | Possessions | Mean duration (s) |
|---|---|---:|---:|---|---:|---:|
| 1 | lenny_pipeline | 0.6795 | 0.4571 | 0.2971–0.6057 | 18 | 15.4 |
| 2 | baseline_smooth125 | 0.5812 | 0.4242 | 0.2667–0.5939 | 16 | 18.5 |
| 3 | baseline_smooth50 | 0.6207 | 0.3256 | 0.2140–0.4465 | 26 | 10.8 |
| 4 | baseline_smooth25 | 0.6200 | 0.2868 | 0.1962–0.3849 | 36 | 7.6 |
| 5 | baseline_raw | 0.5972 | 0.0378 | 0.0189–0.0595 | 131 | 2.0 |

This is only the public 7,500-frame sample: 17 true possessions, mean 14.2 s.
Intervals overlap; the numerical improvement over smooth50 is not established as
statistically significant, and no private/full-match score is available.

## Output and reproducibility

Two consecutive complete `python pipeline/run_all.py` runs produced SHA-256:

```text
657d2950979b9617146a9a462af6cc8497008e2c3191de46bef11dec836dfa3d
```

The source tracking/XML concatenation has SHA-256
`6985dbedc9b1d47d306ca73523eec17d049eda5502a6802f38fd9e5e3c5ec10f`.
Local `cache/alignment.json` records that fingerprint and the inferred attacking
directions: home attacks +x in half 1, -x in half 2; away is opposite.
Detected kickoff frames are 11187 and 113741, inferred from centre geometry in
the automatic tagging intervals. Before those estimates, each graded half is dead.

Final labels: H 79,236; A 62,000; D 17,621; X 40,462. All X frames are outside
the published graded halves. Four synthetic tests check event-only switches,
duel retention, dead states and deterministic serialization. Runtime assertions
also check every decoded live switch against the event mask and every model
training block against the purged held-out interval.

The only file added under the viewer is the generated predictions CSV. Licensed
match files, video, cache, model arrays and local dependencies are not committed.
