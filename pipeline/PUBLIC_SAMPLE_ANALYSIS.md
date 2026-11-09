# Mall: public-sample diagnosis

Computed from the current `mall-model.csv`, the official public labels, and Mall's cached
`layer0_tactical.npz` (`ball_xy_raw` and `ball_xy`). This is a diagnostic worked example,
not a tuning run or a full-match evaluation. No pipeline parameters or predictions were changed.
Frame ranges below are inclusive, at 25 fps.

Prediction SHA-256: `fd07fd4e2e57686668a001fa42cb516ebe930f50f74a8afb5c809aa3ed49a898`.

## Missing tracking and prediction errors

The public window is frames 11185–18684: 7,500 frames. Raw ball coordinates are missing in
885 frames (11.8%); after cleaning, 880 remain missing. Of the raw missing frames, 391 are
live play and 494 are dead ball.

| Raw ball tracking | Public frames | Wrong predictions | Correct predictions |
|---|---:|---:|---:|
| Observed | 6,615 | 759 | 88.53% |
| Missing | 885 | 254 | 71.30% |
| All | 7,500 | 1,013 | 86.49% |

These are conditional frame accuracies, not macroF1, spellF1, or estimates of causation.
The missing group contains many correctly classified dead-ball frames: all 494 missing-ball
D frames are correct. All 254 errors in that group occur during ATH possession. Five missing
BAR frames were filled by cleaning and classified correctly. During live play alone, frame
accuracy is 90.11% with observed raw ball positions (5,640 frames) versus 35.04% with missing
positions (391 frames). Most of that missing-live error comes from one 9.28 s gap, so it should
not be generalized into a match-wide causal estimate.

| Raw gap | Length | Ground truth during gap | Wrong frames |
|---|---:|---|---:|
| 11185–11186 | 0.08 s | ATH | 2 |
| 11508–11512 | 0.20 s | BAR | 0 |
| 12831–12978 | 5.92 s | ATH: 20 frames; D: 128 frames | 20 |
| 16646–16777 | 5.28 s | ATH | 0 |
| 17047–17278 | 9.28 s | ATH | 232 |
| 17460–17825 | 14.64 s | D | 0 |

The most informative case is 17047–17278. Mall's layer-3 dead-ball flag switches on at 17047
and stays on through 17392. Ground truth remains live until 17380, so frames 17047–17379 are
333 false-dead predictions. This includes all 87 frames of the BAR possession at 17293–17379.
There are 101 erroneous observed-ball frames after tracking returns at 17279.

The preceding 5.28 s ATH gap was handled correctly. Thus neither "all gaps cause failure" nor
"ball observed during a missed possession rules out gap effects" is supported. Missing tracking
is associated with some errors, and a temporal dead-ball decision can persist beyond a gap.
A controlled comparison with genuinely recovered ball data would be needed to establish its
causal contribution. Simply removing those frames or inventing interpolated trajectories would
change the evaluation problem and would not be such a comparison.

## What "eight missed possessions" means

At the scorer's IoU threshold 0.5, 9 of 17 ground-truth possessions match Mall's 15 predicted
possessions. Eight ground-truth spells are unmatched; six predicted spells are unmatched.
These are interval-matching counts, not counts of possessions containing zero correct frames.

| True spell | Frames | Duration | Best same-team IoU | Visible issue |
|---|---|---:|---:|---|
| BAR | 11351–11552 | 8.08 s | 0.082 | Later BAR prediction merges across intervening ATH possession |
| ATH | 11553–11623 | 2.84 s | 0 | No overlapping ATH prediction |
| ATH | 13704–13784 | 3.24 s | 0.371 | ATH prediction ends early |
| BAR | 15346–15364 | 0.76 s | 0 | No overlapping BAR prediction |
| ATH | 15365–15765 | 16.04 s | 0.317 | Shares an overlong prediction with the preceding ATH spell |
| BAR | 15766–15798 | 1.32 s | 0 | No overlapping BAR prediction |
| ATH | 15799–15888 | 3.60 s | 0.363 | Predicted spell extends too far around the true interval |
| BAR | 17293–17379 | 3.48 s | 0 | Layer 3 still declares dead ball after the preceding gap |

Best IoU here is calculated before enforcing one-to-one matching. A prediction cannot earn
credit for multiple true possessions. Four unmatched spells have no same-team overlap; four
have some overlap but do not qualify. The four with no overlap total only 8.4 seconds
(2.8% of the five-minute window), despite representing four separate possession errors. All eight have cleaned ball positions available throughout;
only the first has any raw missing positions (five frames). Earlier gaps can still affect them.

## Comparison with the cited paper

[Automatic event detection in football using tracking data (Vidal-Codina et al., 2022)](https://doi.org/10.1007/s12283-022-00381-6)
reports event-detection results, not this challenge's possession spellF1. Section 3.1 says player
possession was not benchmarked. Section 2.1 describes feeds with in-play/dead information, supplied
explicitly or via feed-specific missing-ball conventions. That is an important input difference
when our ball can disappear during live play. Mall also adds learned models and a sequence decoder,
so its implementation is not a direct reproduction of the paper's deterministic decision tree.
