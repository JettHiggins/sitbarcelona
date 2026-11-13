# Wes segmentation activity log

Branch: `wes_deadball+video`. The unchanged `mall --mode model` is the comparison
baseline. This log records hypotheses, exact comparisons and setbacks; it does not
promote a variant on the strength of an estimated score. Detailed traces, generated
CSVs, model files and restricted data stay under Git-ignored paths.

## Protocol and decision gate

- Run `python -m pipeline.wes.evaluate` after `prepare-mall` has built
  `bench1`, `bench2`, and `bench3` from the
  [Metrica sample-data repository](https://github.com/metrica-sports/sample-data).
  Each game is scored by models trained on the other two. The answer keys are derived
  from event logs and may disagree with the competition's corrected-tracking key.
  These games have already informed development, so they are **not** an untouched test.
- The evaluation uses the vendored official possession metric, with five spell IoU
  thresholds. It reports `spellF1`, `macroF1`, dead/live precision and recall,
  predicted possession counts, recall of spells up to five seconds, and boundary
  precision/recall/latency at 12 and 25 frames. It includes the source SHA-256,
  benchmark key SHA-256, code commit, and training-game list in the JSON report.
- Promote a variant only if mean held-out `spellF1` rises by at least **0.02**,
  at least two games improve, no game falls by more than **0.02**, the full-match
  output validates as 199,319 ordered rows, and the run is reproducible from
  authorized inputs. These are project gates, not contest thresholds. The public
  five-minute labels are for alignment/file checks only and are not used to
  choose any variant or parameter.

## Initial implementation, 2026-11-12

Reference before this change: branch commit `80f24a9`. Current experimental code:
`pipeline/wes/segmentation.py`, `evaluate.py`, the optional Mall Layer 5
arguments, and Mall Layer 3 stage traces. The exact source SHA-256 and revision are written
by the evaluation command. The evaluated Python source fingerprint was
`95233e81064c27871c1c41030af7e7cb4d143ea28e1428887cc9073583388842`.
Inputs were Metrica sample-data commit `e706dd506b360d69d9d123d5b8026e7294b13996`,
degraded with fixed seeds 1001–1003 and event-derived answer keys. Each game used
models trained only on the other two. The machine-readable report and input-key
hashes from the original run are in ignored
`pipeline/reports/mall/segmentation-evaluation.json`. A fresh run after the Wes
split writes `pipeline/reports/wes/segmentation-evaluation.json`.

| Variant | Hypothesis and changed component | `spellF1` / `macroF1` by game | Mean change | Dead/live, counts, short spells, timing | Outcome |
|---|---|---|---|---|---|
| Original Mall | Fixed Layer 3 D mask and existing Layer 5 decoder; comparison only | 1: .6902/.8910; 2: .6961/.9056; 3: .6616/.9065 | .6826/.9011 mean | Predicted possessions 285/277/323; live recall .8722/.8924/.9127 | Baseline retained |
| `viterbi` | Remove all Layer 3 postprocessing except hard kickoff/half constraints to isolate Viterbi output | .6531/.8790; .6613/.8957; .6073/.8890 | −.0421 spell | Live recall falls in all games; 297/283/355 possessions | Rejected: raw path loses too many live frames and boundaries |
| `snapped` | Isolate effect of boundary snapping | .6709/.8879; .6876/.9057; .6315/.8996 | −.0193 spell | Recovers most of raw-path loss, but still below final in every game | Diagnostic: snapping helps overall |
| `rules` | Add restart/enter-pitch rules, before short-live merging | .6703/.8877; .6840/.9034; .6310/.8996 | −.0209 spell | Similar to snapped, slightly worse on this bench | Diagnostic: no separate gain here |
| `durations` | Add minimum-duration merging, before final hard constraints | .6902/.8910; .6961/.9056; .6616/.9065 | 0 spell | Exact match to final baseline on all three games | Diagnostic: duration merging restores the remaining loss |
| `revisable` | Compare uncertain D subspans against a live path using the original dead/live score, observed control/ownership support, and transition cost; protect outside/kickoff | .6864/.8922; .6947/.9070; .6569/.9087 | −.0033 spell | Reopened 7/12/17 spans (333/577/748 frames); live recall +.0030/+.0042/+.0041; possessions 287/280/338 | Rejected: macroF1 rises, but extra spells and small live-precision losses coincide with lower spellF1 in every game |
| `candidates` | Above, plus soft kinematic contact and later-recovery hypotheses with separate contact, team, and change confidence; Layer 5 objective/costs unchanged | .6892/.8928; .6930/.9072; .6574/.9088 | −.0027 spell | 5516/5841/7178 candidate diagnostic rows, including existing CPs; possessions 290/281/338 | Rejected: tiny recovery against `revisable`, no gain against original in any game |

This is a measured **stalemate** for the two new stages. The scoring gate fails for
every variant: no mean +.02 gain, no two improving games. Stage ablations indicate
that the existing boundary snapping and duration merge are beneficial on these
keys, even though an individual false-D case remains possible. The revisable path
raises live recall and per-frame `macroF1`, but `spellF1` penalizes its extra
possession boundaries. The candidate stage slightly improves the revised path on
Games 1 and 3 while worsening Game 2; all three still trail original Mall.
These are associations in the diagnostics, not a claim that every reopened span
is wrong. The key is approximate, and there is no private full-match ground truth.
The licensed BAR–ATH tactical bundle is now present. The unchanged Mall model
generated `possession-viewer/data/mall-model.csv`; the official format validator
accepted all 199,319 ordered rows, with `X` on all 40,462 ungraded frames. This
validates output format, not private score. The public five-minute score from older
Mall runs is not a substitute for this held-out test.

## Wes split and reproduction, 2026-11-12

The experimental decoder, evaluator, tests, and video audit now live under
`pipeline/wes/`, with a separate `wes` command and `wes-<mode>.csv` outputs.
Mall Layers 0–4 provide shared tracking and learned evidence; Mall Layer 3 only
records intermediate dead-ball decisions, and Mall Layer 5 accepts optional
arguments that its default path does not pass. Wes-specific settings live in
`pipeline/wes/config.py`. The source fingerprint for the rerun is
`951c6a6e26bb93e2112852ddef33132a5101806a47db686fdc0bec9d2132d6ad`
from pre-commit worktree revision `80f24a9`; the input and game-held-out protocol
are as above. The report is
`pipeline/reports/wes/segmentation-evaluation.json`.

The three-game scores reproduced the original table to four decimal places:
Mall mean `spellF1` .6826 / `macroF1` .9011; Wes `revisable` .6793 / .9026;
Wes `candidates` .6799 / .9029. `revisable` lost .0033 mean spellF1 and
`candidates` lost .0027. Both lost spellF1 on all three games, so neither
passes the project gate. The full BAR–ATH `wes-revisable.csv` and
`wes-candidates.csv` each passed the 199,319-row validator. On graded frames,
Wes candidates differs from Mall on .73% of frames; this is disagreement,
not an accuracy score. Mall remains the selected variant. Video evidence and
the optional TCN are still unimplemented; no video score is implied.

## Missed-spell diagnosis, 2026-11-13

`python -m pipeline.wes.diagnose` reproduced the original/revisable/candidates
`spellF1` and `macroF1` exactly from the earlier three-game report. The new
ignored report, `pipeline/reports/wes/missed-spell-diagnosis.json`, contains
per-spell and per-transition traces for **only** the three other-game keys.
The committed source records the deterministic category definitions. At the
official headline IoU 0.5, a missed spell is assigned its majority predicted
label *inside the true interval*; this is a symptom, not a causal label.

| Variant | Missed / true spells | Mostly D | Mostly wrong team | Mostly right team but unmatched | Mixed | Missed spells ≤5 s |
|---|---:|---:|---:|---:|---:|---:|
| Original Mall | 453 / 1166 | 153 | 176 | 120 | 4 | 371 |
| Wes revisable | 451 / 1166 | 145 | 181 | 122 | 3 | 368 |
| Wes candidates | 450 / 1166 | 145 | 180 | 122 | 3 | 367 |

For original Mall, 28,414 true-live benchmark frames were predicted D.
Of these, 26,922 were **already D in Layer 3's raw Viterbi path**;
1,179 first became D at snapping, 186 at the duration step, and 127 at
the final step. This provenance is not causal attribution, and event-derived
dead/live keys may differ from the competition key. It does show that a
snapping-only fix cannot cover most false-D frames on this benchmark.
Wes revisable lowered false-D frames to 27,403, but its extra live islands
still lowered mean `spellF1`.

There were 852 true direct H/A changes in the approximate benchmark keys.
Mall missed 428 within the official 12-frame tolerance. Of those missed
changes, 374 already had a decoder candidate accessible within 12 frames,
but only 158 had positive support for the destination team; 221 had
frame ownership evidence favoring that team over the next second. Wes
candidates raised positive target support to 309 of its 425 missed changes,
yet barely altered the number missed. This is consistent with a combined
ownership/sequence-score problem, not just missing candidate timestamps.
The diagnostic uses the key for analysis on other games only; it does not
modify model predictions or use BAR–ATH public labels.

**Planned next experiment at this stage:** compare a local three-state `D/H/A` decoder
with Mall and both Wes modes. In uncertain windows, jointly score dead/live
and team evidence, event-anchored transitions, and state durations, so a
reopened D span pays for the entire resulting possession pattern. Keep clear
half/kickoff constraints and the original output as an ablation. Calibrate
the evidence scales on training games only, avoid counting correlated
control/change-point evidence twice, and preserve the current project
promotion gate. Report changes in all three missed-spell symptom groups,
short-spell recall, spurious transitions, boundary timing, `spellF1` and
`macroF1`. Wrong-team-majority spells may require better upstream ownership
evidence even if decoding improves.

## Constrained joint decoder, 2026-11-13

Hypothesis: scoring `D/H/A` segments together in uncertain windows could
reopen genuine play without creating the extra unsupported possession islands
seen in `revisable`. `pipeline/wes/joint.py` adds a sparse-boundary local
semi-Markov decoder. Windows come from a disagreement between Mall D and its
raw Viterbi/control evidence, or a sustained disagreement between the decoded
team and smoothed owner evidence. No benchmark or BAR–ATH labels select windows.
Outside them, Mall is frozen. Dead/live emissions use existing Layer 3 log odds;
conditional H/A emissions use the existing Layer 5 owner evidence, normalized
so owner confidence alone does not make a dead frame live. Transition costs,
event bonuses and the soft live-duration prior come from Mall. This first trial
did **not** fit new evidence scales. The three ablations isolate D, H/A, or both.
No video or TCN is involved. Input provenance and held-out training are the
same as the initial experiment. Python source SHA-256:
`2f24d62ec1f51d5b4fb9f31521973c9bceaf86e270628ae4d568f50c0daea116`.

| Variant | Game 1 `spellF1` | Game 2 | Game 3 | Mean `spellF1` | Mean change vs Mall | Mean `macroF1` | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|
| Mall baseline | .6902 | .6961 | .6616 | .6826 | — | .9011 | Selected |
| `joint-d` | .6888 | .6986 | .6625 | .6833 | +.0007 | .9029 | Failed +.02 gate; diagnostic only |
| `joint-team` | .6888 | .6967 | .6503 | .6786 | −.0040 | .9006 | Rejected: Game 3 and short-spell regressions |
| `joint` | .6858 | .6952 | .6513 | .6774 | −.0052 | .9022 | Rejected: combined changes compound boundary errors |

`joint-d` reduced benchmark false-D frames from 28,414 to 27,249 and
D-majority missed spells from 153 to 146. It still increased mostly-right-but-
unmatched spells from 120 to 124 and mostly-wrong-team spells from 176 to 178;
total missed spells fell only 453 → 451 at IoU .5. Its small mean gain is far
below the project gate and below the official full-match score noise band.
`joint-team` left the D mask fixed and increased missed spells to 458;
`joint` increased them to 460. The local objective can alter H/A boundaries,
but the upstream owner evidence is wrong or weak on many short spells, and
window seams plus the weak duration prior can create poor interval matches.
These are measured associations, not proof of one exact cause for each miss.

All three full BAR–ATH CSVs passed the 199,319-row format check. Their graded-
frame disagreements with Mall are .58%, .11%, and .76%, respectively; private
full-match accuracy is unknown. Keep Mall selected. Avoid tuning this decoder
against BAR–ATH's public five minutes or repeatedly searching constants on the
same three development games. The next meaningful improvement needs stronger
ownership/control evidence or an additional compatible other-game benchmark.

## Next entries

The JSON report carries the complete dead/live precision and recall, predicted
counts, short-spell recall, and 12/25-frame boundary precision/recall/latency for
each variant and game. Keep `mall-model.csv` as the selected output under the
current gate. If later experiments pass all gates, add another dated entry with
the new source fingerprint, per-game scores and explanation before freezing it.

## Separate later experiments

- **TCN:** an optional compact temporal dead/live model on other-game tracking,
  with explicit missingness and order-independent player aggregation. Train and
  evaluate game-held-out, then feed its probabilities to the revisable decoder only
  if it adds consistent value. It is not needed to remove the hard D bottleneck.
- **Video:** first run `python -m pipeline.wes.video.audit --video-root PATH` on
  permitted other-game footage. A synchronized tracking/video mapping and usable
  other-game labels are unconfirmed. No visual model or score is claimed. If the
  audit cannot confirm these, begin only with locally run pretrained detection and
  inspectable appearance evidence in short uncertain windows; keep separate
  tracking-only and video-assisted outputs. Never hand-label/correct the competition
  match or use its public five-minute labels to select settings.
