# Mall: players-only dead-ball evidence, ten training games and a trajectory model

2026-11-12, branch `score-90`. Everything here is produced by `python -m pipeline.run mall`
after `prepare-mall`; no label of the challenge match enters training, model
selection or any constant.

## Result

Official scorer (`pipeline/vendor/challenge/possession_metrics.py`). Two outputs are kept:
`mall-model.csv` (the default, with the trajectory model) and `mall-boosted.csv` (the same
pipeline with the gradient-boosted possession models, `mall --mode boosted`).

| Public five-minute window (17 possessions) | Mall before | Boosted | Trajectory (default) |
|---|---:|---:|---:|
| spellF1 (ranking metric) | 0.5375 | 0.6710 | 0.6182 |
| spellF1, bootstrap 5-95% | | 0.48 - 0.84 | 0.42 - 0.79 |
| macroF1 | 0.8503 | 0.9149 | 0.9151 |
| Frame accuracy | 0.8649 | 0.9152 | 0.9153 |
| Dead-ball F1 (class D) | 0.7873 | 0.9227 | 0.9227 |
| Predicted possessions | 15 | 14 | 16 |

Frame accuracy and macroF1 are above 0.90 with either output. spellF1, the metric the board is
ranked on, is not, and will not be with this approach (see "What still limits spellF1").

**The two outputs disagree about which is better, and the default follows the larger sample.**
On five full held-out matches the trajectory model is ahead on every one, by 0.09 spellF1 on
average. On the public window it is 0.05 behind. That window holds 17 possessions; the scorer's
own bootstrap puts each output's spellF1 anywhere in a range 0.36 wide, and the two ranges
overlap almost entirely, so the window cannot rank them. The frame-level scores on it are equal
(macroF1 0.9151 against 0.9149). RULES.md says the window is for checking, not choosing, and
choosing by it here would mean preferring 17 possessions to about 2,000. Both files are produced
by the repository, so the team can submit either.

## Held-out games

Five other full matches, each scored by models that never saw it (three Metrica sample games,
two of the seven DFL games). All settings were chosen on these, not on the public window.

| Mean of five held-out games | spellF1 | macroF1 | Dead-ball F1 |
|---|---:|---:|---:|
| Before: rule cues only, possession models trained on Metrica | 0.6225 | 0.8870 | 0.9237 |
| + players-only dead-ball evidence | 0.6469 | 0.9053 | 0.9468 |
| + possession models trained on ten games | 0.6674 | 0.9141 | 0.9468 |
| + short-window change-point features, decoder re-swept (= boosted output) | 0.6794 | 0.9155 | 0.9468 |
| + trajectory model in place of the boosted models (= default output) | 0.7679 | 0.9285 | 0.9468 |

| Per game, spellF1 / macroF1 | Metrica 1 | Metrica 2 | Metrica 3 | DFL 1 | DFL 4 |
|---|---:|---:|---:|---:|---:|
| Boosted | 0.704 / 0.900 | 0.715 / 0.932 | 0.688 / 0.928 | 0.635 / 0.907 | 0.656 / 0.912 |
| Trajectory | 0.796 / 0.917 | 0.816 / 0.946 | 0.829 / 0.947 | 0.706 / 0.919 | 0.692 / 0.914 |

The scorer's noise band is about 0.04 spellF1 and 0.01 macroF1 per match. The dead-ball step is
a clear macroF1 gain; the trajectory model is a clear spellF1 gain (larger than the band on four
of five games, and in the same direction on the fifth); the steps between are small.

The ten-game training is not a uniform win for the boosted models: it lifts the two DFL games
(mean spellF1 0.556 to 0.636 at that step) and costs the three Metrica games a little (0.707 to
0.689, inside the noise band). It is kept because the challenge match is neither source.

## What changed

1. **Seven more training games** (`mall/bench/idsse.py`). The open DFL / IDSSE Bundesliga
   dataset (Bassek et al., Scientific Data 2025, CC-BY 4.0) carries ball status and team in
   possession on every frame of its position feed: a frame-exact H / A / D key on seven full
   matches. Players are degraded as for the Metrica games; the ball is hidden the way the
   challenge's tracker loses it (`degrade.ball_dropouts`).
2. **A dead-ball model that sees only the players** (`mall/layer3_dead_ball/learned.py`). Team
   shape and movement plus stoppage signals - the share of players standing, walking and
   running, players at or beyond a line, players in the penalty areas - looking up to 20 s
   both ways. Its log-odds are added to the existing rule cues, the rules' own player-speed cue
   is dropped (the model replaces it), and the lost-ball cue fades where the players look in
   play (`DEAD_GAP_GATE`).
3. **Possession and change-point models trained on all ten games** instead of three.
4. **A sharper change-point model** (`crossgame.fine_features`): the quarter- and half-second
   before and after each candidate frame (ball-to-player distances and their minima, relative
   speed, control, ball acceleration, events, the reception that starts there), every candidate
   of every training game, 600 boosting rounds. Trained on eight games and tested on two it
   never saw, average precision for "possession changes here" went 0.52 to 0.65 (Metrica 1)
   and 0.48 to 0.54 (DFL 1). (Boosted output only.)
5. **A trajectory model** (`mall/layer4_possession_model/sequence.py`, default). A three-layer
   bidirectional GRU over every frame of a half. Its input is the raw 25 Hz geometry: for each
   team the three players nearest the ball - distance, offset, velocity relative to the ball -
   plus counts within 2 m and 5 m, the ball's position and motion, and coarse team shape for
   when the ball is lost. It outputs P(home in possession) and P(possession changes within 6
   frames) per frame, which replace the boosted models' evidence in the same decoder. Trained
   on the other games' live frames with team-swap and pitch-mirror augmentation; about 6 minutes
   on a laptop GPU. On the two test games: share of live frames with the right team 0.941 to
   0.971 and 0.908 to 0.930; average precision for a change 0.65 to 0.76 and 0.54 to 0.62.

## Why the dead-ball model gets no ball input

This was measured, twice, before it was decided.

| Dead frames on which the ball is tracked | |
|---|---:|
| Metrica sample games | 0.2 - 8% |
| DFL games as delivered | 100%, frozen in place (speed exactly 0) |
| Challenge match, public window | 66%, moving (median 2.7 m/s) |

In both training sources the ball gives the answer away, in opposite ways, and neither way
holds on the challenge match. A first model that was given the ball scored dead-ball F1 0.98 on
held-out Metrica games and 0.44 on the public window (the rules alone: 0.79). Hiding the DFL
ball at stoppages does not help either, because a visible DFL ball is always frozen. What the
22 players do at a stoppage looks the same in every feed, so that is all the model reads: alone
it scores 0.93 on held-out games and 0.91 on the public window; blended with the rules, 0.947
and 0.923.

The largest single error in the earlier analysis (`PUBLIC_SAMPLE_ANALYSIS.md`), 333 frames
called dead through a 9 s loss of the ball in open play, is gone: the players were visibly in
play, which now outweighs the lost ball.

## How the settings were chosen

Dead-ball F1 averaged over all ten bench games, each by a model that never saw it:

| Setting | Bench F1 |
|---|---:|
| Rules only | 0.9239 |
| + learned evidence, weight 0.5 / 1 / 2 | 0.9465 / 0.9463 / 0.9428 |
| weight 1, rules' speed cue dropped | 0.9485 |
| weight 1, speed cue dropped, gap gate 0.5 (kept) | 0.9465 |

Everything with the learned evidence sits within 0.006: the bench cannot rank those settings.
The kept one is at the top of that group and is the one with a reason behind each part. Larger
learned weights score higher on the public window (dead-ball F1 0.94 at weight 2); they were
not chosen, because that would be tuning on the window.

The sequence decoder was swept on the five held-out games (switch cost, duration prior,
change-point weight and offset, frame-evidence weight, event bonuses). Before the short-window
change-point features nothing beat the existing constants by more than 0.004 spellF1. With them,
trusting the learned evidence more helps a little: `SEQ_W_CP` 1.0 to 2.0 and `SEQ_W_FULL` 0.35
to 0.5 take the mean from 0.6709 to 0.6794. Everything else stayed best where it was.

With the trajectory model the same sweep gives `SEQ_W_FULL` 1.0 and `SEQ_W_CP` 1.0 (mean
0.768); every setting tried around that pair scores between 0.760 and 0.768. Three training
lengths were compared on the two test games (1,500 / 3,000 / 6,000 steps) and the longest was
best on both; a wider network was not finished and is untested.

## What still limits spellF1

Two measurements on held-out games (and the same picture on the public window) say where the
rest of the metric is.

**It is the team decision, not the stoppages.** Measured with the boosted models: keeping our
dead-ball mask and putting the true team on every live frame gives spellF1 0.92 (0.87 on the
public window). Keeping our team decisions and using the true dead-ball mask gives 0.73. That
is why the trajectory model, which only changes the team decision, moved the metric when
nothing done to the decoder did.

**And within that, it is the short possessions.** Take the answer key itself and erase only the
possessions shorter than a threshold that sit between two spells of the other team:

| Key with short possessions erased | < 1 s | < 2 s | < 3 s | < 5 s |
|---|---:|---:|---:|---:|
| spellF1, Metrica 1 | 0.919 | 0.835 | 0.761 | 0.648 |
| spellF1, DFL 1 | 0.911 | 0.807 | 0.747 | 0.633 |
| spellF1, public window | 0.912 | 0.827 | 0.743 | 0.615 |

A submission that is frame-perfect except for possessions under one second already loses 0.09;
0.67-0.70 is what "right about everything longer than about four seconds" scores, and the
trajectory model's 0.77 on held-out games is roughly the three-second line. So 0.90
on the ranking metric means finding essentially every possession longer than a second and
timing it to a few frames.

The evidence for those is thin. On a held-out Metrica game the improved boosted change-point
model is
confident (p > 0.5) at 44% of the turnovers that start a possession under one second and at
about two thirds of the rest; a control segment by the new team - the decoder's strongest cue -
exists for 14% of the sub-second ones and 75% of those over five seconds. The decoder passes
most of what the model is sure of (it misses 12-32% of confident turnovers) and lowering its
switch cost loses more to false possessions than it gains. Better constants will not close
this; a touch detector that works from the trajectories themselves might.

## Limits

- The DFL ball degradation is a simulation with stated ranges, not a fit to the challenge match.
- The dead-ball model's players-only premise was checked on two leagues' feeds; a third feed
  could still differ in how players are tracked at stoppages.
- Training now needs the 2.6 GB DFL download, about 8 GB of free memory and PyTorch. The default
  run trains the dead-ball model and the trajectory model (about 10 minutes with a GPU; far
  longer on CPU).
- The trajectory model's weights depend on the hardware: a different GPU, or the CPU, will give
  slightly different weights and so a slightly different CSV. `mall --mode boosted` has no such
  dependence.
- The default output scores lower than the boosted one on the public window. The reasons for
  keeping it as the default are above; if the full match behaves like the window rather than
  like the five held-out matches, the boosted file is the better submission.
- Full-match quality on the challenge match is unknown; only five minutes of it are labelled.

## Reproduce

```bash
python -m pipeline.run prepare-mall --bench-src /path/to/metrica-sample-data
python -m pipeline.run mall --mode model      # trajectory model -> mall-model.csv
python -m pipeline.run mall --mode boosted    # boosted models    -> mall-boosted.csv
python -m pipeline.mall.bench.score          # held-out games and the public window
python -m unittest pipeline.test_mall_deadball
```

Output SHA-256 (Python 3.10.11 on Windows, numpy 1.26.4, scipy 1.14.1, scikit-learn 1.5.2,
torch 2.6.0+cu124 on an RTX 4060 Laptop GPU):

- `mall-model.csv` `ce99b78b250e73eaa2f0f6a724d743ef54a47b61da84e6f277ff25a7222d26f9`
- `mall-boosted.csv` `f26c3ce33a3cbfc0485ed6deacf922e1b0254bc872f1d970aa0d2e895e0c74bc`
