# Why corrected frames can lower possession spellF1

Date: 2026-11-13. Branch: `wes_deadball+video`. The comparisons use only
`bench1`–`bench3`, with each game's learned models trained on the other two.
Their event-derived keys are approximate and have already guided development;
they are not an untouched test or the private BAR–ATH answer key. The public
five-minute BAR–ATH labels were not used for these comparisons.

The [official evaluation](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md)
scores each predicted H/A possession as an interval. At each temporal IoU
threshold (0.3, 0.4, 0.5, 0.6, 0.7), predicted and true intervals of the same
team match **one-to-one**. With `M` matches, `P` predicted possessions, and `G`
true possessions, the F1 at that threshold is `2M / (P + G)`. `spellF1` is
the mean of those five F1 values. Its unit is the possession, not the frame.
A 2-second missed possession and a 20-second missed possession each contribute
one false negative; duration matters through overlap/IoU, not as a separate
per-frame penalty. `macroF1` instead summarizes frame-level H/A/D accuracy.

For a concrete example, suppose the true possession is ten seconds of H. If a
model predicts H for seconds 0–4, D for 4–6, and H for 6–10, it gets 80% of
those frames right, but produces **two** H possessions. Each overlaps only 40%
of the true interval. At IoU .3 and .4, one of the two predictions can match;
the other is a false positive. At .5, .6 and .7 neither matches. The five
spell-F1 values are 2/3, 2/3, 0, 0, 0, averaging about .267 for this simple
one-possession example. This is why a short false D gap can be expensive.

`X` has another important asymmetry. The official scorer counts an X on a
graded frame as a frame-level miss, but forward-fills it for temporal scoring
within each graded run, with back-fill at the run head. An `H X X H` sequence
can therefore remain one H possession for `spellF1` while accumulating
`macroF1` errors. Conversely, carrying H through an actual handover via X can
miss a possession. The viewer's full-match Mall/Wes differences use their raw
H/A/D labels and exclude X; they are a navigation aid, not a replay of the
private spell scorer, whose full-match graded runs are unavailable.

## Aggregate counts across three development games

These counts are summed to explain the score. The reported mean `spellF1` is
the mean of each game's official score, not a new F1 computed from this table.

| Variant | Predicted possessions | Matched at IoU .3 / .4 / .5 / .6 / .7 | Frames corrected vs Mall | Frames broken vs Mall | Mean `spellF1` | Mean `macroF1` |
|---|---:|---|---:|---:|---:|---:|
| Mall | 885 | 772 / 748 / 713 / 663 / 600 | — | — | .6826 | .9011 |
| Wes revisable | 905 | 779 / 754 / 715 / 663 / 601 | 1,200 | 550 | .6793 | .9026 |
| Wes candidates | 909 | 781 / 757 / 716 / 665 / 603 | 1,340 | 577 | .6799 | .9029 |
| Wes joint D only | 896 | 778 / 754 / 715 / 665 / 606 | 1,130 | 375 | .6833 | .9029 |
| Wes joint team only | 893 | 777 / 746 / 708 / 660 / 596 | 287 | 491 | .6786 | .9006 |
| Wes joint D/H/A | 900 | 779 / 749 / 706 / 659 / 600 | 1,433 | 949 | .6774 | .9022 |

"Corrected" means Mall disagreed with the other-game key on a frame and the
Wes variant agreed; "broken" means Mall agreed and Wes disagreed. These counts
are diagnostic, not independent observations or estimates for BAR–ATH.

At IoU .5, the other-game keys contain 1,166 true possessions. Mall predicts
885 and matches 713. Revisable predicts 905 (+20) and matches 715 (+2);
candidates predicts 909 (+24) and matches 716 (+3). The extra matches do not
compensate for the extra predicted spells in `2M/(P+G)`. Joint D predicts 896
(+11) and matches 715 (+2), while its strict .7 matches improve 600 → 606;
that combination explains its very small mean advantage. This arithmetic
explains the score direction without claiming every new spell is wrong.

`joint-d` is numerically 0.0007 above Mall, but fails the project promotion
gate (+0.02 mean gain). The difference is too small to establish superiority
on three reused approximate keys. Mall remains the selected output.

## Which missed possessions matter here

At IoU .5, Mall has 453 unmatched true possessions across the three games.
The diagnostic groups them by the **majority predicted label inside the true
interval**, a symptom rather than a proven cause:

| Majority symptom | Mall misses | Revisable misses | Joint D misses |
|---|---:|---:|---:|
| D covers most of the interval | 153 | 145 | 146 |
| Wrong team covers most | 176 | 181 | 178 |
| Right team covers most, but interval still fails matching | 120 | 122 | 124 |
| Mixed, no majority | 4 | 3 | 3 |

Reopening D removes some D-majority misses but also creates extra predicted
spells and shifts some misses into ownership or boundary/fragmentation cases.
Wrong-team majority and right-team-but-unmatched together account for 296 of
Mall's 453 misses, so the hard D decision is only part of the bottleneck.
The latter category is particularly important: mostly right-team frames
still fail when a spell is too short, too long, split, badly timed, or loses
one-to-one matching to another predicted interval. This is why adding correct
frames alone does not guarantee a possession match.

## Two concrete examples

- **Game 1, revisable:** 275 frames corrected and 101 broken, a net gain of
  174 correct frames. Predicted possessions rose 285 → 287, while matched
  possessions fell 236 → 235 at IoU .5 and 225 → 223 at IoU .6.
  `spellF1` fell .6902 → .6864. The added live frames did not form enough
  accurately timed, same-team possessions to offset the extra predictions.
- **Game 3, revisable:** 528 frames corrected and 223 broken, a net gain of
  305. At IoU .5, matches rose 249 → 253, but predicted possessions rose
  323 → 338. At IoU .7, matches stayed 203. The new intervals helped some
  loose matches but did not yield enough precise matches; `spellF1` fell
  .6616 → .6569.

The missed-spell diagnosis found 371 of Mall's 453 unmatched true
possessions at IoU .5 lasted at most five seconds. A short H/A error can
erase a whole possession match while affecting few frames. Conversely,
correcting many D frames inside a long interval may improve `macroF1` but
leave its spell unmatched, or create a new fragment counted as a false
positive. Boundary movement changes interval IoU at several thresholds.
Thus the ranking can change even when frame accuracy moves in the opposite
direction.

A long incorrect stretch matters when it covers several true possessions or
pushes an interval below multiple IoU thresholds. There is **no extra
duration-weighted spell penalty** simply for being wrong longer. A single
long wrong-team possession can produce a false positive and a false negative;
several short turnovers hidden inside a long prediction can produce several
false negatives. Mall's smoothing and duration constraints appear to avoid
some false boundaries, preserving enough interval matches to outweigh the
extra frame misses. That is an inference from the matched counts and frame
changes, not proof that each individual Mall boundary is correct.

## What the viewer can and cannot show

The local viewer compares full-match Mall and Wes CSVs and highlights exactly
where their H/A/D labels differ. Its experiment trace explains which Wes
window, dead-ball proposal or change hypothesis was active. This identifies
**where an experiment changed the answer**, not which answer is correct.
Only five minutes of public BAR–ATH reference are available. Competition
rules prohibit using those labels to choose a variant or tune settings, and
there is no full-match ground truth in this repository. Choose variants on
permitted other-game evidence and the established promotion gate.
