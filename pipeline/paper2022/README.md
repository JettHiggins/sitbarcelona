# Published-method reconstruction: Vidal-Codina et al. (2022)

Independent implementation from the final paper and supplementary methods:

- [Automatic event detection in football using tracking data](https://doi.org/10.1007/s12283-022-00381-6).
  F. Vidal-Codina, N. Evans, B. El Fakir and J. Billingham, Sports Engineering 25, 18 (2022).
- [Supplementary Information, S1–S6](https://static-content.springer.com/esm/art%3A10.1007%2Fs12283-022-00381-6/MediaObjects/12283_2022_381_MOESM1_ESM.pdf).

The article is CC BY 4.0. This is newly written code reconstructing the described method,
not the authors' source or a claim of bit-for-bit reproduction. No DataBallPy code is copied
or required. See [measured results and limitations](../PAPER2022_RESULTS.md).

## Fidelity map

| Published component | Implementation | Qualification |
|---|---|---|
| §2.3.1 possession/duel zones, duel priority | `core.py` | Opponent memberships retained; no arbitrary duel winner |
| Noisy-ball smoothing | Seven-frame, order-2 SG on observed runs | No interpolation; already-filtered feeds may need filtering disabled via Parameters |
| §2.3.3 gain | Incoming/outgoing direction over PZ interval OR sufficient adjacent speed change | Gain at PZ entry; no minimum control duration |
| Reject geometric fly-bys | Gain validation before accepted-control loss linking | Same-player return remains continuation if no verified opponent took over |
| §2.3.2 loss | Ball leaves zone, displacement exceeds epsilon_s, next control excludes player or live interval ends | No loss for walking away from a static ball or regaining before the stoppage |
| S4 parameters | Published A and B/C profiles | BC is a reference preset, not an identified/optimized provider match |
| §2.3.4/Fig. 5 | Kickoff, penalty, corner, throw-in, goal-kick triggers and inbound patterns | Published positional tolerances |
| S1 error paths | Stationary executor/release recheck, missing inbounder, trigger-only uncertainty | Follow Fig. S1's extension of free-kick fallback; unknown patterns remain marked |
| Kickoff exceptions | Repeated kickoff correction and period-start handling | No repetition inference if intervening ball trajectory is entirely unobserved |
| Fig. 6a–e | Shot/save/claim/loose-ball reception, retain/deflect, unsuccessful save, rejected keeper gain | Requires known or inferred goalkeeper identity |
| Fig. 6f/§2.4.3 | Cross/shot zones and receiver/context checks | Numerical zone dimensions not supplied; explicit interpretation below |
| S5 | Passing/receiving events, control/no-possession sequences, basic outcomes | Advanced analytics attributes, e.g. opponents overtaken/assists, omitted |
| S2 | Actor-aware one-to-one timing matching; type confusion and class P/R/F1 | Private STS/FIFA data unavailable; Metrica used instead |

Gains, loss candidates, accepted events, restart reasons and control validations are exported.
No Mall ownership probabilities, touch stream, decoder, duration prior, change-point model,
or six-second contact backdating is used. No classifier is trained.

## Published parameters

| | Provider A | Provider B/C (default) |
|---|---:|---:|
| Possession radius | 0.5 m | 1.0 m |
| Duel radius | 1.0 m | 1.0 m |
| Direction change | 10° | 10° |
| Speed change | 1 m/s | 5 m/s |
| Per-frame displacement | 0.05 m | 0.10 m |

S4.2 restart tolerances: kickoff own-half 3 m (published optimal range 3–5), center 2 m;
penalty 2/2/4 m; corner 3 m; throw-in/goal-kick 0.5 m; stationary executor 1 m.
Keeper retention/deflection uses one second; goal-ray testing allows 0.25 m beyond posts.
No threshold is selected using public BAR–ATH labels.

## Input limitations and explicit adaptations

**Game status is an input in the paper.** It uses a provider boolean or a feed whose ball is
absent when play is dead. The paper does not supply a general replacement for that signal:

- `--status missing` (default): missing raw ball is D, following that feed convention.
  BAR–ATH violates it: the ball disappears in live play and remains visible in some stoppages.
- `--status provided --status-file PATH`: full ordered `frame,in_play` CSV, binary 0/1.
  Ownership-label CSVs are rejected.
- `--status mall`: explicitly reuse Mall's D for a controlled sensitivity comparison.
  This preserves its dead-ball errors and is not independent status detection.

**Raw coordinates.** The tactical adapter reads layer 0 only for `ball_xy_raw`,
`player_xy_raw`, original slot teams, half/pitch metadata and attack directions. Mall's
interpolated ball and repaired teams are not used. Non-reuse mode parses layer 0 first,
but its other cleaned outputs are ignored.

**Identity.** The paper assumes identifiable players/goalkeepers. BAR–ATH has anonymous
slots with unknown roles; identity can change within a slot. Goalkeepers are inferred from
median defensive position among currently observed slots. That is an input adaptation, not
a published keeper-identification method. `reproduce(..., keepers=...)` accepts supplied
roles. Metrica game 3 uses actual role metadata; games 1–2 use the disclosed inference.

**Geometry/occupants.** Fig. 6f has no numerical cross/shot-zone dimensions. We interpret it
as the last 40% of pitch length, with a central penalty-area-width shot band. Native
simultaneous same-team control gains use nearest-player tie breaking; opponents form a duel.

**Viewer possession is a separate adapter.** The native method includes no possession and
duels; the challenge demands H/A/D. The adapter holds the last unambiguous verified gain
through travel/duels and accepts confirmed restart patterns, not an uncertain trigger alone.
An unresolved restart carries the last team; an uninitialized half uses the current nearest
observation or a logged home fallback. It never fills an early interval with a future
owner's identity. These choices are explicit in `viewer_decisions`, not attributed to the paper.

## Run

```bash
pipeline/.venv/bin/python -m pipeline.run paper --reuse-cache
# Fresh parsing when raw tracking is not cached:
pipeline/.venv/bin/python -m pipeline.run paper --data-dir /path/to/challenge-data
```

Select `paper.csv` alongside `mall-model.csv` and `baseline_smooth50.csv`. The previous
event-first prototype is archived locally outside the data dropdown.

Optional published profile or status input:

```bash
pipeline/.venv/bin/python -m pipeline.run paper --reuse-cache --profile a --output /tmp/paper-a.csv
pipeline/.venv/bin/python -m pipeline.run paper --reuse-cache --status provided --status-file /path/to/status.csv
pipeline/.venv/bin/python -m pipeline.run paper --reuse-cache --status mall \
  --output pipeline/cache/paper2022/paper-mall-status.csv
```

Trace: `pipeline/cache/paper2022/tactical-bc-missing.events.json`. Native arrays/CSV labels:
`tactical-bc-missing.npz` (zone_state, ball_control, player_owner, duel, label). Different
profiles/status inputs have separate cache names. Generated assets remain ignored by Git.

## Verify and reproduce measurements

```bash
pipeline/.venv/bin/python -m unittest pipeline.test_paper2022 pipeline.test_event_first pipeline.test_mall_interaction pipeline.test_adapters pipeline.lenny.test_invariants
git clone --depth 1 https://github.com/metrica-sports/sample-data.git /tmp/paper-metrica-sample
pipeline/.venv/bin/python -m pipeline.paper2022.evaluate --bench-src /tmp/paper-metrica-sample
pipeline/.venv/bin/python -m pipeline.compare possession-viewer/data/mall-model.csv \
  possession-viewer/data/paper.csv --truth /path/to/BAR-ATH_possession_gt_public.csv
```

The benchmark uses **original undegraded Metrica tracking and stable actor IDs**, not Mall's
corrupted training benchmark. Missing-ball status and an event-derived status proxy are tested
separately. The latter supplies benchmark binary status, not ownership labels, to inference.
These are neither the private FIFA data nor a fair direct comparison with degraded Mall inputs.

Matching uses 12-, 25-, and 250-frame windows, same actor, one-to-one assignment, type confusion
and class scores. The 250-frame window follows the paper's ±10-second timing tolerance; it
does not establish frame-level boundary accuracy. Metrica PASS/SHOT and restart annotations
are mapped explicitly. Keeper event accuracy is not claimed because equivalent independent
keeper ground truth is absent.
