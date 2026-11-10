# Published-method reconstruction: measured results

2026-11-10. Source: Vidal-Codina et al., *Automatic event detection in football using tracking
data*, [final paper](https://doi.org/10.1007/s12283-022-00381-6) and
[supplement](https://static-content.springer.com/esm/art%3A10.1007%2Fs12283-022-00381-6/MediaObjects/12283_2022_381_MOESM1_ESM.pdf).

This is an independent reconstruction, not the authors' code. The [fidelity map](paper2022/README.md)
distinguishes reproduced rules from input adapters and unspecified geometry. No settings were
selected by optimizing the public window. Original Mall remains the default pipeline.

## Implemented and tested

Published possession/duel zones, seven-frame SG smoothing, interval motion-based gains,
validated next-control losses, rejection of fly-bys, and same-player reacquisition without
spurious loss. Published restart tolerances/hierarchy and S1 error paths; repeated-kickoff
handling; Fig. 6 shot/keeper branches with retain/deflect distinction; native player control,
no-possession sequences and event outcomes. No learned ownership model or Mall decoder.

The earlier prototype's four-frame control requirement, peak-based kick heuristic and
six-second backdating have been removed from this implementation. Native gains occur at
the entry of a motion-validated possession-zone interval, as in the paper.

Limits: numerical cross/shot-zone boundaries are interpreted from Fig. 6f; tactical keeper
roles are inferred because metadata is anonymous; and converting native duels/no-possession
to the challenge's H/A/D requires a separately documented policy. It does not backdate a
future gain into an unknown initial interval. The paper's private FIFA/STS data are not used.

## Event-level validation: original Metrica tracking

Original undegraded games with stable actor IDs, fixed B/C reference parameters. Passing-event
matching groups Metrica PASS/SHOT (including cross subtypes), requires the correct player and
half, and is one-to-one. The table is **event detection**, not guaranteed correct event type;
type confusion and per-class precision/recall/F1 are included in the generated benchmark JSON.

| Game, missing-ball status | Precision at ±10 s | Recall at ±10 s | Detection F1 at ±10 s | F1 at ±12 frames (~0.5 s) |
|---|---:|---:|---:|---:|
| 1 | 0.6644 | 0.8445 | 0.7437 | 0.7298 |
| 2 | 0.7200 | 0.8613 | 0.7843 | 0.7705 |
| 3 | 0.6949 | 0.8405 | 0.7608 | 0.7219 |
| **Mean F1** | | | **0.7630** | **0.7407** |

Median absolute timing error among matched events is 2, 2 and 3 frames. This excludes missed
and incorrectly attributed events and must not be interpreted as accuracy on all touches.
The paper's S2 benchmark uses ±10 seconds; matching at that tolerance alone does not establish
frame-accurate possession boundaries. These results do not reproduce the paper's >90% claims.

Actor-matched restart detection F1 at ±10 s is approximately 0.476, 0.435 and 0.546. Many
tracking gaps are not genuine restarts, and some restart patterns are not observed. With
event-derived binary status instead, restart detection is approximately 0.552/0.533/0.680.
That status proxy is reconstructed from annotations, not a provider's authoritative boolean.
There is no independently comparable keeper-event ground truth; save/claim accuracy is not claimed.

For completeness, the viewer adapter's mean possession spellF1/macroF1 on these original
games is 0.7035/0.9315 with missing-ball status and 0.7045/0.9458 with event-derived status.
These are **not comparable directly** with Mall's degraded benchmark results. Original inputs,
stable identities and the status source differ. Per-game values remain in the local report.

## BAR–ATH: data assumptions matter

| Output / status input | Public spellF1 | macroF1 | Frame accuracy |
|---|---:|---:|---:|
| Original Mall | **0.5375** | **0.8503** | **0.8649** |
| `paper.csv`: missing-ball feed convention | 0.3810 | 0.6596 | 0.7283 |
| Paper rules with Mall D, diagnostic only | 0.5158 | 0.8176 | 0.8265 |

Default `paper.csv` is the disclosed missing-ball convention, not a claim that we possess an
independent accurate dead-ball detector. It has 391 false-dead live frames and 975 dead frames
predicted live in the public window. The paper assumes a game-status signal; our ball disappears
in live play and remains visible during stoppages, so its feed convention does not hold here.

The same-D diagnostic isolates more of the ownership change, but remains worse than original
Mall. Status mismatch is therefore not the only issue. It is stored outside the viewer data
directory at `pipeline/cache/paper2022/paper-mall-status.csv`.

Default output: 199,319 frames, H=58,396, A=35,818, D=64,643, X=40,462. There are 25 predicted
public possessions against 17 expected, and 20.93% full-match graded-frame disagreement with Mall.

## The reviewed plays

- First BAR possession: first verified gain is still 11466 versus expected 11351, 4.60 s late.
- Short ATH possession: gain at **11575**, 22 frames/0.88 s after expected 11553. It now
  correctly labels 49 of 71 frames, versus Mall's zero, but has not solved the late boundary.
- Keeper collection: BAR gain at **11625**, one frame after expected 11624. The receiving
  label is an interception in the trace; do not infer that keeper identity/classification is validated.
- BAR 15346–15364: still no verified BAR gain. An ATH gain at 15332 reflects the detector's
  attribution, not the expected labels.
- BAR 15766–15798: 17 correct frames, but no BAR gain in that interval. ATH gain at 15783 is
  16 frames/0.64 s earlier than expected 15799; partial frame agreement is not a recovered event.
- BAR 17293–17379: default status allows 71 correct frames; native BAR gain is 17309, 16
  frames late. Mall-status mode still masks the entire possession D.

The remaining misses are consistent with player-ball distances outside the published PZ,
anonymous/misassigned raw tracks, motion ambiguity, and differences between native player
events and corrected-tracking ownership. Ground truth remains the expected output.

## Verification and provenance

66 tests pass (35 paper-specific plus 31 existing), including all Fig. 6 paths, gain/loss
conditions, fly-by rejection, control continuity, restart branches, duel handling, input-layer
isolation, non-backdated CSV initialization, terminal passes into stoppages, actor-aware one-to-one
evaluation and timing tolerances.
The official CSV validator passes; the existing viewer dropdown loads the new file. A repeat
run is checked for identical CSV bytes. No model weights or original Mall output are replaced.

- Metrica sample-data checkout: `e706dd506b360d69d9d123d5b8026e7294b13996`.
- Final paper PDF SHA-256: `984d0ef8ed95e1ae12fb6a81af773c5882cb7e3ff61c9ef80b60f807fbe23e92`.
- Supplement PDF SHA-256: `a6e27544ba2d4b06a01b33f349d47926ad601b7c112d2c2686b27be25612d3cf`.
- Default CSV SHA-256: `72f90f9c36f746923395fa7600be0aea3540c99e319d388c3a3c8455b481d07d`.
- Mall-status diagnostic SHA-256: `b2ff11afc5cf1bbf64e1045289ab9bb25692132ca4e3f99f5b73cdd4e67c8823`.
- Original Mall SHA-256 remains `fd07fd4e2e57686668a001fa42cb516ebe930f50f74a8afb5c809aa3ed49a898`.

Native JSON/NPZ and benchmark reports live under `pipeline/cache/paper2022/`; downloaded
PDFs under `pipeline/cache/paper-reproduction/`. All are ignored by Git. Source, tests,
fidelity documentation and this summary are committed. The viewer data folder keeps only
`paper.csv`, `mall-model.csv` and `baseline_smooth50.csv`; the previous prototype is archived
under `pipeline/cache/paper2022/previous-prototype/`.
