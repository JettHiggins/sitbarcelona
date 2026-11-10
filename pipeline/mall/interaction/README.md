# Mall interaction model (experimental)

An optional event-centered hybrid. The original Mall model remains the default.
This variant adds supervised contact-chain ownership and change-point classifiers; it does
not require video inference or public BAR–ATH labels.

The [first measured comparison](../../MALL_INTERACTION_RESULTS.md) improves all three held-out
Metrica games but slightly regresses on the BAR–ATH public window. It is experimental.

## Representation

1. Retain Mall receptions, attributed touches and kicks. Receptions/kicks are control/play
   anchors, not independently verified physical contacts.
2. Revisit kinematic candidates without Mall's suppression near existing control intervals.
   Retain candidates within the existing 2.5 m attribution radius, including low-confidence
   STOP/ARTIFACT candidates. These remain hypotheses, not automatic possession changes.
   Confidence values are heuristic indicators, not calibrated contact probabilities.
3. Preserve both the motion-change frame and closest-approach frame.
4. Link the last two contacts, next contact, preceding anchor and next anchor, within eight
   seconds and the same predicted-live region and half. Represent possible confirmed contacts,
   opponent rebounds followed by recovery, and changes from the previous playing team.
5. Combine these links with current ball/control and team-shape features. Tracking loss and
   reappearance have neutral features, since they are not inherently home/away events.

An example candidate chain is `ATH play → BAR touch → ATH rebound → BAR recovery`. Features
describe that chain; the model learns how it maps to ownership from other games. Nothing
unconditionally moves a possession boundary back to the first contact.

## Learning and decoding

Two HistGradientBoosting classifiers learn frame ownership and change-point probabilities.
The tactical model trains on bench1–3, with home/away augmentation and symmetric prediction.
Training samples every third labeled live frame for the ownership model. Change points use
the existing three-frame live grid plus new contact and closest-approach candidates, with
the existing ±6-frame change-point targets.

Ownership log odds are pooled equally with frozen Mall evidence. Change-point probabilities
are geometrically pooled where both models have a candidate; additional contacts use the new
model alone. Pooling avoids simply doubling correlated evidence. Mall's final decoder,
switching costs, touch bonuses, duration preference, and dead-ball mask are unchanged.

Settings were fixed before inspecting this variant's public score. They are implementation
defaults, not proven optimal values. The current event-derived training keys remain a
limitation for deflection semantics. This is not a reproduction of either research paper.

## Run

After preparing the three benchmark games as described in `pipeline/README.md`:

```bash
pipeline/.venv/bin/python -m pipeline.run mall --mode interaction --data-dir /path/to/data
```

This rebuilds Mall's match layers and runs the new layer. To explicitly use already-prepared
Mall tracking and full-model caches instead:

```bash
pipeline/.venv/bin/python -m pipeline.run mall --mode interaction --reuse-cache
```

Only use reuse mode when the inputs and original Mall code match the existing caches. Missing
caches fail rather than falling back to events-only mode. The new model's cache is keyed by
Mall Python source and the benchmark layer/label contents; changing those invalidates it.

Outputs:

- `possession-viewer/data/mall-interaction.csv`: select alongside `mall-model.csv` in the viewer.
- `possession-viewer/data/mall-interaction.interactions.json`: contact hypotheses, anchor
  status/confidence, later-recovery features, original/new probabilities and labels, final
  transitions, training games, model fingerprint and changed-frame count.
- `pipeline/cache/mall/layer6_interaction_tactical.npz`: frame probabilities, change-point
  probabilities and final numeric labels, separate from Mall's layer-5 cache.

`--output PATH` changes the CSV destination and places the diagnostics beside it. Generated
match output, model weights and benchmark inputs remain local/ignored in the default paths.

## Verify

```bash
pipeline/.venv/bin/python -m unittest pipeline.test_mall_interaction pipeline.test_adapters pipeline.lenny.test_invariants
pipeline/.venv/bin/python -m pipeline.mall.interaction.evaluate
pipeline/.venv/bin/python -m pipeline.compare possession-viewer/data/mall-model.csv \
  possession-viewer/data/mall-interaction.csv --truth /path/to/BAR-ATH_possession_gt_public.csv
```

The benchmark evaluator trains **both** original Mall and interaction models on the other two
games for each held-out game. It never uses the all-three-game tactical model on a benchmark
training game. Results are written incrementally to
`pipeline/reports/mall/interaction-evaluation.json`. Its three-game comparison is a development
benchmark, not an untouched external test.

Tests check rebound/recovery links, half/stoppage isolation, uncertain candidate preservation,
home/away symmetry, neutral gap features and rejection of training-game evaluation. Actual
match runs and held-out scoring are necessary in addition to these synthetic invariants.
