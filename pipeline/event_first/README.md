# Independent event-first possession

This is a deterministic ownership algorithm, not a blend with Mall's estimated possession.
It reads cleaned tracking/features and the existing dead-ball mask. It never loads Mall's
layer-2 ownership events, layer-4 probabilities or change points, layer-5 predictions, or the
interaction model. It requires no model training or public ground truth at inference.

Tracking cleanup/features and **Mall's layer-3 dead-ball mask are shared** to isolate the team
ownership comparison. Rebuilding that mask uses Mall's original layer 2 upstream; it is not
input to the new ownership state machine. False dead balls remain an explicit limitation.

## Event stream

`events.py` detects its own primitives from tracking:

- **Control:** the same nearest tracked player remains within 1.5 m and below 3 m/s relative
  ball speed for at least four frames (0.16 s). Player changes split intervals.
- **Contact hypothesis:** a speed dip or direction change, with a player within 2.5 m over a
  ±4-frame attribution neighborhood. Suspected fast reversals are not automatically discarded.
- **Ambiguous contact:** both teams remain candidates when their closest distances differ by
  at most 0.6 m. This does not itself award possession.
- **Kick/play anchor:** an unambiguous nearby contact, within 1.5 m, followed by at least a
  5 m/s speed rise over ten frames. This is a tracking proxy for playing the ball, not proof
  of intent. Headers and deflections can remain contact hypotheses.

The motion-change timestamp and nearest-approach timestamp are both retained. There is no
suppression of contacts merely because an existing control interval is nearby.
These are contact/control primitives, not a complete pass/shot/duel event classifier or a
full reproduction of the papers' event taxonomy.

## Sequence rules

`sequence.py` operates on those events:

1. Establish an owner at the restart/first observation, falling back to the first anchor.
   With no evidence at all, first attributed contact or home is an explicitly logged fallback.
2. Keep the current team during a loose ball. Preserve tentative contacts as a pending episode.
3. A control or kick anchor confirms a play. If it belongs to the opponent, switch to that team.
4. If a pending contact belongs to the newly confirmed team, backdate to its earliest eligible
   contact, allowing intervening opponent rebounds. Confirmation is limited to six seconds.
5. A confirmed play settles that episode. Do not backdate through established control or a
   subsequent confirmed play. Stoppages and half boundaries break chains.

There are no learned frame probabilities, switch costs, possession-duration priors, minimum
H/A spell lengths, or cooldowns. Two clear plays can create two rapid ownership switches.
The rules are hypotheses to test against the labels; they are not the ground-truth algorithm.
In particular, the current rules require a control/play anchor to confirm an opponent's touch.

## Run and inspect

```bash
pipeline/.venv/bin/python -m pipeline.run event-first --data-dir /path/to/data
# Or explicitly reuse the existing tracking/features/dead-ball caches:
pipeline/.venv/bin/python -m pipeline.run event-first --reuse-cache
```

Default output: `possession-viewer/data/event-first.csv`. Select it in the viewer alongside
`mall-model.csv` and `baseline_smooth50.csv`. The older `mall-interaction.csv` hybrid from the
first experiment was moved locally to `pipeline/cache/event_first/previous-hybrid/` to keep
the data picker at three CSVs. Its source mode remains available.

Event and decision trace: `pipeline/cache/event_first/tactical.events.json`. Each ownership
change records its boundary, confirming event, reason, and intervening contacts. The trace also
contains all contact/control events, live spans, source fingerprint and shared-mask disclosure.
It stays outside the viewer data folder.

`--output PATH` changes the CSV destination. Reuse mode assumes the original tracking and
feature caches match the inputs and preprocessing; missing caches fail explicitly. Default
mode rebuilds layers 0–3 and does not train or run Mall's possession models.

## Evaluation

```bash
pipeline/.venv/bin/python -m unittest pipeline.test_event_first pipeline.test_mall_interaction pipeline.test_adapters pipeline.lenny.test_invariants
pipeline/.venv/bin/python -m pipeline.event_first.evaluate
pipeline/.venv/bin/python -m pipeline.compare possession-viewer/data/mall-model.csv \
  possession-viewer/data/event-first.csv --truth /path/to/BAR-ATH_possession_gt_public.csv
```

The benchmark command scores the fixed rules on each Metrica game and compares with original
Mall trained on the other two games. The original comparison models are reused if cached, or
trained if absent. They are comparison references only, never inputs to event-first ownership.
Results go to `pipeline/cache/event_first/benchmark.json`.

See [measured results](../EVENT_FIRST_RESULTS.md). This version catches some Mall errors but
is less accurate overall on both the benchmark and public window; it remains experimental.
