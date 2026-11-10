# Mall interaction variant — first measured results

2026-11-10. Implemented in `mall/interaction/`; run with `mall --mode interaction`.
This is an experimental addition to Mall, not a replacement or a confirmed improvement on
BAR–ATH. All model settings were fixed before looking at the variant's public score.

## What was built

- Contact chains linking the preceding play, recent contacts, possible opponent rebounds and
  subsequent recovery/play anchors within the same live region and half.
- Low-confidence preservation of nearby kinematic contacts that Mall otherwise suppresses or
  classifies as STOP/ARTIFACT. They do not automatically become possession changes.
- New supervised frame-ownership and change-point classifiers trained on the other Metrica games.
- Equal pooling with Mall's original learned evidence, followed by the unchanged sequence decoder.
- A separate CSV, probability cache, and contact/recovery diagnostics JSON. No additional UI
  controls or dependencies. Existing `mall-model.csv` and the default model mode are preserved.

The new layer expresses later-recovery evidence explicitly, but does not assume that all
deflections switch ownership or unconditionally backdate possession. Dead-ball inference,
decoder penalties and the original 1 m touch-bonus gate were deliberately held fixed.

## Fresh leave-one-game-out comparison

For each row, both original Mall and the new variant were trained on the other two games.
The tactical model trained on all three was never used to score a benchmark game. Results
below were freshly computed, rather than copied from Mall's historical report.

| Held-out game | Original spellF1 | Interaction spellF1 | Original macroF1 | Interaction macroF1 |
|---|---:|---:|---:|---:|
| Metrica 1 | 0.6902 | 0.7007 | 0.8910 | 0.8956 |
| Metrica 2 | 0.6936 | 0.7088 | 0.9053 | 0.9097 |
| Metrica 3 | 0.6616 | 0.6974 | 0.9065 | 0.9135 |
| **Mean** | **0.6818** | **0.7023** | **0.9010** | **0.9062** |

Mean frame accuracy is 0.9029 → 0.9077. SpellF1 improves on every game in this development
benchmark, by 0.0205 on average. This supports further testing of the representation; three
games and event-derived labels are insufficient to establish general superiority.

Predicted possession counts are 285 → 292, 277 → 274, and 323 → 318. Better interval matching
does not necessarily mean more possessions. The underlying benchmark has 389, 366, and 411
true possessions: the undercount remains substantial.

Full-precision results and each fold's training fingerprint are in the generated local report
`pipeline/reports/mall/interaction-evaluation.json`. Training sets are bench2+3, bench1+3 and
bench1+2 respectively. No public BAR–ATH ground truth is read by the training or benchmark
evaluation modules.

## BAR–ATH public window

The match model trains on all three Metrica games. The unchanged official scorer evaluates
the same 7,500 public frames afterward:

| | Original Mall | Interaction |
|---|---:|---:|
| spellF1 | **0.5375** | 0.5212 |
| macroF1 | **0.8503** | 0.8478 |
| Frame accuracy | **0.8649** | 0.8621 |
| Predicted public possessions | 15 | 16 |

The variant changes 1,266 graded match frames (0.80%). The dead-ball mask and ungraded mask
are identical. It **does not fix** the opening late BAR switch, the following short ATH spell,
or the brief BAR spells at 15346–15364 and 15766–15798. It also cannot recover the later BAR
spell masked D by layer 3.

The diagnostics show why sequence reasoning alone remains limited:

- Near the opening expected BAR start, a retained artifact candidate at 11351 is attributed
  to ATH, and the next inferred control is ATH at 11418. The new model therefore sees an ATH
  chain, despite the expected BAR label. At 11466 it supports the correctly detected BAR
  reception more strongly, but that is still too late.
- At 15353/15358 it retains BAR touch hypotheses, but the next inferred anchor is ATH. This
  supplies no confirmed BAR recovery under the new features. There is no guaranteed switch
  merely because a touch hypothesis exists.
- Around 15766, its contact and reception hypotheses remain ATH. Preserving more candidates
  cannot by itself identify a BAR contact when the nearest-player attribution is wrong.

These are model-input observations, not proof that the expected ground truth is wrong.
They also do not establish that video inference is necessary: better tracking-based contact
attribution and boundary evidence remain possible next steps.

## Interpretation and next work

Keep the current original Mall model as the default. The new representation has a positive
development-benchmark result but a small public regression, so it is useful for comparison,
not ready to be promoted.

The immediate remaining issue is identifying and timing brief contacts correctly, rather than
adding more lookahead or weakening all switching penalties. The contact diagnostics now make
those errors reviewable. Next experiments should isolate attribution/uncertainty and dead-ball
postprocessing separately, using game-level comparisons. They should not repeatedly select
settings to recover these particular public spells.

## Validation and provenance

- 17 unit tests pass, including eight interaction checks: recovery/rebound evidence, no links
  across stoppages or halves, preservation of uncertain contacts, home/away symmetry, neutral
  tracking-gap features, training/evaluation isolation, and preservation of the dead mask and
  original evidence.
- Official submission validator passes: 199,319 ordered frames, valid H/A/D/X labels, no
  abstentions in graded halves.
- The viewer successfully lists and loads `mall-interaction.csv` through its existing dropdown.
- A repeat run to a newly created output directory produces identical CSV bytes.
- Python compilation and Git whitespace checks pass.

SHA-256:

- Original CSV: `fd07fd4e2e57686668a001fa42cb516ebe930f50f74a8afb5c809aa3ed49a898` (unchanged).
- Interaction CSV: `6103684b45c0b6dbd007dec80f5708f242f6b9ecc2a9df5dde92e11fbe120aa2`.
- Tactical training fingerprint:
  `1e61ac91bb9704289f32de8d0a2f9d24bf80ee07c6c3902eca0e458b55ac2612`.

The full match run explicitly reused the existing Mall caches with `--reuse-cache`. Original
preprocessing was not changed. New classifiers were trained locally; the benchmark also trained
the excluded-game original classifiers for each fold. Generated CSVs, diagnostics, weights,
caches and full-precision reports are ignored by Git; source, tests and this measured summary
are committed. See `mall/interaction/README.md` for reproduction commands and cache limitations.
