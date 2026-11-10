# Mall pipeline audit — 2026-11-10

This is an analysis of the integrated implementation and its existing outputs. No model,
threshold, training label or prediction was changed. The public ground truth is the expected
output for the public window; disagreement is an error, including brief ownership flips.
Whether a proposed rule explains those labels is a separate question from whether that rule
will generalize to other games.

## Findings

Mall is a useful foundation: learned cross-game possession evidence materially improves the
events-only version. Its weaknesses are distributed across touch attribution, learned evidence,
dead-ball inference and final decoding. It is already temporal and offline, but does not
explicitly model a chain of touch → deflection → loose ball → recovery.

The public errors do not support changing one global possession radius or simply reducing
switch resistance. Several brief possessions lack supporting evidence upstream. Incorrect
dead-ball decisions also override all later possession evidence.

## What actually runs

The runtime uses tracking and Smart Tagging, not video pixels. Frames are zero-based at 25 fps.
The tactical parser subtracts one from the source ATD frame IDs. Video inspection helps us
diagnose mistakes, but the pipeline cannot see a foot, a header or intent directly.

| Stage | Actual behavior | Main limitation |
|---|---|---|
| 0: cleanup | Rejects implausible ball motion; fills consistent gaps up to 25 frames with Hermite interpolation; smooths ball over 5 frames and players over 9; repairs some duplicate/team tracks | Position and velocity around a real contact can be uncertain; interpolation is not recovered observation |
| 1: features | Distances, relative velocities, nearby-player counts, team movement/shape, ball gaps/geometry, and tagging cues | Nearest player and co-motion are proxies for playing the ball, particularly weak for aerial balls |
| 2: events | Sustained controls, receptions/releases, speed dips, direction changes, kicks, tracking loss/reappearance | Hard classifications and nearest-player attribution can discard or assign the wrong contact |
| 3: dead ball | Weighted cues → two-state temporal decoder → boundary snapping → minimum durations | Missing tracking can become a stoppage; later stages cannot overturn D |
| 4: learned evidence | Cross-game boosted-tree frame classifier and change-point classifier | Learns from only three games, with event-derived ownership labels |
| 5: final sequence | Global optimization of team evidence, event/change-point support, switching cost and possession duration | Can merge short spells; cannot repair unsupported attribution reliably or recover possession inside D |

### Touch, control and radius are different things

There is no single possession radius:

- **Control:** nearest player within 1.5 m, relative speed below 3 m/s, retained for at least
  8 frames (0.32 s). Same-player control segments can bridge interruptions up to 4 frames.
- **Free-touch attribution:** chooses the closest player over ±5 frames, accepting attribution
  out to 2.5 m. It stores the kinematic event frame, which can differ from that closest approach.
- **Touch switching support:** the final decoder gives the touch its explicit bonus only if
  the attributed distance is at most 1 m. A 1.36 m touch is detected but gets zero such bonus.
- **Kick:** a sufficient ball-speed rise within 12 frames, attributed within 1.5 m.

Direction is compared six frames before and after the candidate. Speed dips and turns are
merged locally. Candidates within six frames of a control interval are suppressed as free
touches. Certain sharp, fast reversals become ARTIFACT, and sufficiently deep speed minima
can become STOP. Neither classification directly supplies a touch-based switch candidate.
The learned change-point grid can still allow a switch there.

These rules help reject tracking noise, but rapid contacts by opposing players may be merged,
suppressed or attributed to the closest player at a different point in the neighborhood.
The runtime uses planar x/y tracking; it has no direct ball-height or body-contact evidence.

### Existing temporal context

The active full frame classifier uses present features plus ball/control averages over windows
extending eight seconds backward and forward. It explicitly receives the previous control team,
next control team and time until next control. Team-movement features also use centered means.
Thus “add future context” alone would duplicate an existing capability.

The change-point classifier scores a grid every three live frames (0.12 s), augmented by local
event context. Training marks candidates within ±6 frames of a labeled ownership change as
positive. It learns a neighborhood of a boundary, rather than an exact-contact frame target.
Direct touch/reception/kick candidates can fall between grid points.

The final decoder combines:

- per-frame control evidence and 0.35 × log-odds from the full cross-game model;
- a base switch cost of 4, modified by reception, touch, kick and learned change-point scores;
- a soft duration preference centered on 10 seconds.

There is **no hard minimum H/A possession duration**. The duration penalty is approximately
1.00 for a 0.76 s spell and 0.62 for a 1.32 s spell. However, a brief interruption requires
switching into and out of a team; unsupported switches each incur the base cost and often
additional negative learned change-point scores. This can make two switches unattractive.
It is a global sequence decision, not a fixed cooldown after the previous switch.

Self-trained players-only and full-reference models are also computed, but the players-only
decoder weight is zero and the full self-trained reference is not used in final frame evidence.
The active learned frame evidence comes from the supervised cross-game full model.

## Current public performance

Rechecked the existing CSV with the integrated official scorer:

| Output | spellF1 | macroF1 |
|---|---:|---:|
| Mall events-only, measured during integration | 0.3238 | 0.7753 |
| Mall full model, rechecked for this audit | 0.5375 | 0.8503 |

The public window has 7,500 frames and 17 team possessions. Frame accuracy is 86.49%.
At spell IoU 0.5, 9 of 17 true possessions match 15 predicted possessions. Eight unmatched
possessions do not mean eight wholly absent possessions; see PUBLIC_SAMPLE_ANALYSIS.md.

Confusion matrix: rows are ground truth, columns are predictions.

| | Predicted BAR | Predicted ATH | Predicted D | Total |
|---|---:|---:|---:|---:|
| True BAR | 3,081 | 180 | 193 | 3,454 |
| True ATH | 148 | 2,138 | 291 | 2,577 |
| True D | 83 | 118 | 1,268 | 1,469 |

Of **1,013 erroneous frames**:

- 484 live frames are incorrectly D;
- 201 D frames are incorrectly live;
- 328 live frames have the wrong team while predicted live.

Thus alive/dead disagreement accounts for 685 errors (67.6% of frame errors). This is an
accounting breakdown, not a causal estimate or a prediction of spellF1 gains after a fix.
One short team error may damage more spell matches than a longer boundary error.

Class F1 is BAR 0.9107, ATH 0.8530, D 0.7873. The learned frame classifier's simple 0.5 team
threshold is 94.11% accurate on true-live frames. That number excludes dead-ball classification
and sequence matching, so it must not be compared directly with final three-state accuracy.

Prediction SHA-256:
`fd07fd4e2e57686668a001fa42cb516ebe930f50f74a8afb5c809aa3ed49a898`.

## The opening 30 seconds

Ground truth begins at frame 11185. These are cached pipeline observations, not claims that
the detected events are the real physical contacts:

| Expected interval | Mall evidence/output | Diagnosis |
|---|---|---|
| BAR starts 11351 | ARTIFACT at 11356, unattributed event at 11363, STOP at 11394; ATH reception at 11418 lasting 48 frames; BAR reception and output switch at 11466 | 115-frame / 4.60 s late switch, with missing/misattributed evidence before decoding |
| ATH 11553–11623 inclusive | Mean learned P(BAR) is 0.702; accumulated team evidence favors BAR; final output stays BAR | The entire 2.84 s ATH spell is missing; reducing duration preference alone does not fix the opposing upstream evidence |
| BAR starts 11624, keeper collection sequence | BAR touch at 11601; BAR reception at 11621 | Reception detection is close to this boundary, but the output was already BAR, hiding the turnover |

For the first BAR interval, mean learned P(BAR) is only 0.477 and accumulated evidence over
the true interval favors ATH. This directly limits an explanation based only on a decoder
being reluctant to switch back quickly.

## The two brief BAR flips

Intervals below are inclusive. Visible-contact frames are the user's observations before the
viewer synchronization fix; those contact timings need rechecking in the corrected viewer.
The GT boundaries and cache measurements do not depend on those video observations.

| | Around video 10:13 | Around video 10:30 |
|---|---|---|
| Reported BAR contact / ATH rebound | 15331 / 15335 | 15748 / 15751 |
| Expected BAR possession | 15346–15364, 0.76 s | 15766–15798, 1.32 s |
| Mall output during that interval | ATH throughout | ATH throughout |
| Detected relevant events | BAR TOUCH 15353 at 1.36 m; no touch switching bonus | ATH TOUCH 15760; ATH reception 15777 lasting 24 frames |
| Mean learned P(BAR) over the true interval | 0.396 | 0.198 |
| Interpretation | Some BAR touch evidence exists, but both learned evidence and switch support are weak | Attribution/control evidence itself favors ATH |

Neither brief interval has missing raw ball coordinates. At the start of each, the cleaned
tracking puts ATH closer to the ball than BAR. Nearest-player assignment alone therefore
cannot reproduce the expected labels. Conversely, the labels' delay after the reported contacts
does not establish a fixed delay rule: corrected tracking ownership and raw tracking may differ.

## Why dead ball can occur while someone sprints

The movement cue uses a centered mean of **mean speed across the observed players**, not the
fastest player. One sprinter can coexist with a low average. Even a strong live movement cue
can be outweighed by missing-ball, boundary, stationary-ball or tagging cues.

The temporal processing matters too. The audited failure at frames 17047–17379 unfolds as:

1. The initial two-state decoder produces D at 17114–17278 and 17305–17392.
2. Boundary snapping moves the first start back to tracking loss at 17047, adding 67 frames.
3. The live gap 17279–17304 is only 26 frames (1.04 s). The minimum-live rule requires two
   seconds, so it fills that gap with D, joining the stoppages into 17047–17392.
4. Ground truth is live until 17379. The result is 333 false-D frames, including all of BAR
   17293–17379. Tracking has already returned for the final 101 of those erroneous frames.

At 17047 the movement cue favors live by 3.12, but a long missing-ball gap near a line adds
2.5 toward D. Its instantaneous total still favors live; snapping nevertheless assigns D there.
This illustrates why an isolated frame's player speed cannot explain the final decision.
Layer 5 treats D as fixed, and layer-4 change-point candidates also exclude predicted-dead frames.

## How strong is the evaluation?

The three Metrica games are used in two ways:

- Their event logs are converted into team/dead labels. These are useful derived answer keys,
  not the challenge's corrected tracking ownership.
- Player tracks receive seeded dropouts, fragmentation, team mix-ups and jitter. Simulated
  set-piece tags are generated from events. Ball tracking is left as delivered.

Cross-game training excludes the game being scored when the target is a benchmark game. The
BAR–ATH model trains on all three benchmarks. Home/away-swapped training and averaged predictions
are sensible protections against team-side bias. No public BAR–ATH labels are used by that
training path.

The archived source reports mean leave-one-game-out spellF1 0.681 / macroF1 0.901 with its
revised event-derived keys. It predicts 287/389, 276/366 and 323/411 benchmark possessions.
This undercount is consistent with merged/missed spells beyond our public examples. Those are
**historical source results**, not a newly reproduced benchmark run in this audit. Integration
reproduced the current tactical score, not all historical benchmark scores.

Limits on interpreting those numbers:

1. The decoder settings were selected using the three-game benchmark. Leave-one-game-out
   model training is better than training on the test game, but the same games are not an
   untouched final test of all tuning decisions.
2. Event-derived “last team that played the ball” labels do not guarantee the same deflection
   and loose-ball ownership semantics as corrected tracking GT. Event-log consistency checks
   verify construction, not independent frame-accurate possession correctness.
3. Simulated player corruption and set-piece tags approximate one domain shift. They do not
   independently validate long live-ball gaps or noisy rebound trajectories in BAR–ATH.
4. Seventeen public possessions are too few for confident generalization. Treat their labels
   as expected outputs and diagnostic examples, not an optimization set for constants.
5. The local tactical result differs from the source's reported 0.5625 / 0.8482. Label-key
   revisions and environment differences exist; the cause has not been isolated.

## Concrete implementation issues worth isolating

- Change-point event features build home/away counts for LOST and FOUND, but layer 2 assigns
  those events team -1. Both team-specific feature groups are therefore always zero: this
  match has 151 LOST and 150 FOUND events, none team-attributed. Gap features still reach the
  model elsewhere. Neutral loss/reappearance event features would require retraining.
- `frames_since_control` measures age since the **start** of the most recent control segment,
  including while control is ongoing. That is different from time since control ended or the
  last controlled frame. It is consistent across training/inference, but its name can invite
  incorrect interpretation when adding temporal features.
- Some documentation is stale: config's blanket “none tuned against ground truth” statement
  conflicts with its explicit benchmark-selected decoder settings. The crossgame module's
  players-only description does not describe the active full model. Evaluation claims should
  follow the actual run path rather than those summaries.

## Recommended additions, after agreeing on the hypothesis

1. **Export decision evidence.** A diagnostic sidecar should show observed/filled/missing ball,
   attributed events, control teams, next recovery, frame probabilities, change-point support,
   dead-ball cues and postprocessing. This makes each failure attributable before tuning.
2. **Represent loose-ball episodes explicitly.** Preserve the last confirmed owner, candidate
   contact team(s), intervening contacts, tracking quality, and eventual control or deliberate
   play. Use later recovery to support an earlier candidate boundary. Do not overwrite the
   existing output with unconditional backdating or automatic switches on every touch.
3. **Improve contact attribution.** Preserve uncertainty around close successive contacts and
   reconsider suppressing a free touch solely because a control interval is nearby. Separate
   contact confidence from possession confidence; investigate raw versus cleaned evidence.
4. **Audit dead-ball snapping and short-live merging independently.** A possession recovery
   layer cannot fix intervals hard-masked D. Test live gaps with moving players and ball return
   as well as real stoppages, so improved recall does not erase genuine dead balls.
5. **Then revisit decoding.** Add sequence/recovery evidence to candidates and switch support
   before globally weakening penalties. Measure short-spell recall, false flips, and onset/end
   error in addition to official spellF1 and macroF1.

Change one component at a time, rebuild dependent benchmark features/models when needed, and
evaluate with game-level holdouts. Keep the public window fixed for diagnosis. A new untouched
game with independently compatible ownership labels would strengthen the final comparison.
The existing full model remains the baseline; the proposed recovery layer should be an
optional variant until its benefits and regressions are measured.

## Evidence locations and reproduction

- `mall/layer0_clean/{parse,ball,players,teams}.py`
- `mall/layer1_features/features.py`
- `mall/layer2_touches/touches.py`
- `mall/layer3_dead_ball/deadball.py`
- `mall/layer4_possession_model/{crossgame,run,model}.py`
- `mall/layer5_sequence_decoder/decoder.py`
- `mall/bench/{labels,degrade,build}.py`
- `VERIFICATION.md` and `PUBLIC_SAMPLE_ANALYSIS.md`
- Challenge `CHALLENGE.md`, Possession rules: expected public labels are derived mechanically
  from corrected tracking; loose/contested phases and deflections are explicitly discussed.
- Archived Mall source commit `735502706dffc0a93f199659edcfbf78e6f2f591`,
  `reports/bench/report.md` and `reports/bench/history.md`.

```bash
pipeline/.venv/bin/python -m pipeline.compare possession-viewer/data/mall-model.csv \
  --truth ../more-than-a-hack-2027/evaluation/gt/BAR-ATH_possession_gt_public.csv
git show 735502706dffc0a93f199659edcfbf78e6f2f591:reports/bench/report.md
```

Stage-level measurements were read from the current ignored `pipeline/cache/mall/layer0` through
`layer5_tactical.npz` caches. The dead-ball intermediate trace reran the existing evidence,
Viterbi, snapping and minimum-duration functions on those cached inputs without changing weights.
