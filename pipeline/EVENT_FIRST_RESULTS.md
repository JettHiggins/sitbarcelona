# Event-first ownership: first measured comparison

2026-11-10. This replaces the experimental hybrid as the "new" CSV in the local viewer picker.
Original Mall remains the default pipeline. No existing model settings or CSVs were overwritten.

## Independence

This version detects its own contact/control events from tracking and reconstructs ownership
with an explicit state machine. It does not blend Mall's possession scores or use its decoder,
change-point model, switching cost, duration prior, or original touch-event ownership stream.
There is no new classifier to train. The event rules and parameters were chosen before the
comparison, not fitted to the public ground truth. During implementation, two correctness
fixes prevented backdating through established control and avoided an early timestamp bias on
flat distance plateaus; these have synthetic regression checks.

Shared inputs are Mall's cleaned tracking, geometric features and dead-ball mask. Holding D
fixed makes this an independent **ownership** experiment, not an independent dead-ball system.

## Public BAR–ATH window

| Metric | Mall original | Event-first |
|---|---:|---:|
| Official mean spellF1 | **0.5375** | 0.5297 |
| macroF1 | **0.8503** | 0.8153 |
| Frame accuracy | **0.8649** | 0.8243 |
| Predicted possessions (17 expected) | 15 | 20 |
| Matched possessions at IoU 0.5 | 9 | **12** |

The increased matching count at a single threshold does not establish a better overall model:
there are more false/poorly bounded spells, and the official mean over IoU thresholds is lower.

Full-match disagreement with Mall is **4.67% of graded frames**, versus 0.80% for the earlier
hybrid. The new CSV contains 46,408 H, 29,200 A, 83,249 D and 40,462 X frames. D/X masks are
identical to Mall's. The event stream contains 2,195 events and 319 recorded ownership decisions
(including initial ownership at each live-region start).

Within the public window, it corrects 49 frames Mall got wrong, regresses 354 previously correct
frames, and changes 136 frames where both predictions remain wrong. This is an error breakdown,
not an estimate of causal improvements from any particular rule.

### A concrete recovered Mall miss

Expected ATH frames **13704–13784** (video **9:08.16–9:11.40**, end exclusive):

- Mall correctly labels 33 of the 81 frames.
- Event-first correctly labels all 81.
- However, the event-first ATH spell is **13692–13837**, starting 12 frames early and ending
  53 frames late. It recovers the short spell while delaying the subsequent BAR recovery.

This is useful for visual comparison, but is not a perfectly timed fix.

### The previously discussed deflections

- First BAR start at 11351: still switches at 11466, 4.60 seconds late.
- ATH 11553–11623: still missing.
- BAR 15346–15364: still missing.
- BAR 15766–15798: only one frame correct; this is not a recovered possession.
- BAR 17293–17379: still masked D by the shared dead-ball stage.

These failures show that changing the sequence rules does not automatically recover the right
touch attribution or the ground truth's boundary timing. The new trace allows those two issues
to be inspected separately from sequence smoothing. In particular, a brief contact followed
by ATH recovery will not become a BAR spell unless the detector also establishes BAR control
or a BAR play. That assumption remains testable against the expected labels.

## Three Metrica games

The event-first rules are fixed and untrained. The Mall reference for each row is trained on
the other two games, using the existing excluded-game model caches. These are fresh evaluations,
not scores copied from historical reports.

| Game | Mall spellF1 | Event-first spellF1 | Mall macroF1 | Event-first macroF1 |
|---|---:|---:|---:|---:|
| 1 | 0.6902 | 0.5951 | 0.8910 | 0.8775 |
| 2 | 0.6936 | 0.6105 | 0.9053 | 0.8926 |
| 3 | 0.6616 | 0.6142 | 0.9065 | 0.8993 |
| **Mean** | **0.6818** | **0.6066** | **0.9010** | **0.8898** |

Unlike the previous hybrid, this independent approach regresses on the benchmark. It is a
working event-first baseline for testing hypotheses, not a validated replacement for Mall.

## Verification and reproduction

- 31 tests pass: 14 event-first checks plus the existing 17 checks. These cover backdating,
  rebounds, incidental deflections, rapid confirmed switches, ambiguous attribution, bounded
  lookahead, control protection, half/stoppage isolation, explicit fallback, detector behavior
  and a runtime input contract excluding learned ownership layers.
- Official CSV validator passes all 199,319 ordered rows and valid labels. There are no
  abstentions inside graded halves.
- A separate browser session checks that the existing CSV dropdown lists and loads the new file.
- The full match run uses explicit `--reuse-cache`; original preprocessing has not changed.
- Generated CSV/trace/benchmark reports stay local and ignored by Git.

Event-first CSV SHA-256:
`7ab9632db905093f264d05ba7a43997e0f2b9ff2954cfd45f4782df45534980c`.

Original Mall CSV SHA-256 remains:
`fd07fd4e2e57686668a001fa42cb516ebe930f50f74a8afb5c809aa3ed49a898`.

See [commands and rule definitions](event_first/README.md). Full event/decision traces and
benchmark values are in `pipeline/cache/event_first/`. The new CSV is `event-first.csv`;
`mall-interaction.csv` is the older hybrid and is archived outside the local viewer data folder.
