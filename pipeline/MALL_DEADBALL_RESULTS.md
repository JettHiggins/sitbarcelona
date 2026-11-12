# Mall: players-only dead-ball evidence and ten training games

2026-11-12, branch `score-90`. Everything here is produced by `python -m pipeline.run mall
--mode model` after `prepare-mall`; no label of the challenge match enters training, model
selection or any constant.

## Result

Official scorer (`pipeline/vendor/challenge/possession_metrics.py`).

| Public five-minute window (17 possessions) | Mall before | Mall now |
|---|---:|---:|
| spellF1 (ranking metric) | 0.5375 | 0.6710 |
| macroF1 | 0.8503 | 0.9149 |
| Frame accuracy | 0.8649 | 0.9152 |
| Dead-ball F1 (class D) | 0.7873 | 0.9227 |
| Predicted possessions | 15 | 14 |

Frame accuracy and macroF1 are above 0.90. spellF1, the metric the board is ranked on, is not:
it rose by about 0.13 and is still far from 0.90. The window is five minutes and has been looked
at many times by this team; it is a read-out, not validation. The numbers that carry weight are
the held-out games below.

## Held-out games

Five other full matches, each scored by models that never saw it (three Metrica sample games,
two of the seven DFL games). All settings were chosen on these, not on the public window.

| Mean of five held-out games | spellF1 | macroF1 | Dead-ball F1 |
|---|---:|---:|---:|
| Before: rule cues only, possession models trained on Metrica | 0.6225 | 0.8870 | 0.9237 |
| + players-only dead-ball evidence | 0.6469 | 0.9053 | 0.9468 |
| + possession models trained on ten games | 0.6674 | 0.9141 | 0.9468 |
| + short-window change-point features, decoder re-swept (final) | 0.6794 | 0.9155 | 0.9468 |

Per game, final: Metrica 1-3 spellF1 0.704 / 0.715 / 0.688, macroF1 0.900 / 0.932 / 0.928;
DFL 1 and 4 spellF1 0.635 / 0.656, macroF1 0.907 / 0.912. The scorer's own noise band is about
0.04 spellF1 and 0.01 macroF1 per match, so the dead-ball step is a clear macroF1 gain and a
modest spellF1 gain, and the last step is small: every one of the five games moves the same way,
by 0.002 to 0.026 spellF1, and the public window does not move at all.

The ten-game training is not a uniform win: it lifts the two DFL games a lot (mean spellF1
0.556 to 0.636 at that step) and costs the three Metrica games a little (0.707 to 0.689, inside the noise
band). It is kept because the challenge match is neither source, and a model that has seen two
leagues' labelling is the safer choice for a third. On the public window the two variants are
indistinguishable (spellF1 0.667 / macroF1 0.921 with Metrica-only possession models).

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
   and 0.48 to 0.54 (DFL 1).

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

## What still limits spellF1

Two measurements on held-out games (and the same picture on the public window) say where the
rest of the metric is.

**It is the team decision, not the stoppages.** Keeping our dead-ball mask and putting the true
team on every live frame gives spellF1 0.92 (0.87 on the public window). Keeping our team
decisions and using the true dead-ball mask gives 0.73.

**And within that, it is the short possessions.** Take the answer key itself and erase only the
possessions shorter than a threshold that sit between two spells of the other team:

| Key with short possessions erased | < 1 s | < 2 s | < 3 s | < 5 s |
|---|---:|---:|---:|---:|
| spellF1, Metrica 1 | 0.919 | 0.835 | 0.761 | 0.648 |
| spellF1, DFL 1 | 0.911 | 0.807 | 0.747 | 0.633 |
| spellF1, public window | 0.912 | 0.827 | 0.743 | 0.615 |

A submission that is frame-perfect except for possessions under one second already loses 0.09;
our 0.67-0.70 is what "right about everything longer than about four seconds" scores. So 0.90
on the ranking metric means finding essentially every possession longer than a second and
timing it to a few frames.

The evidence for those is thin. On a held-out Metrica game the improved change-point model is
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
- Training now needs the 2.6 GB DFL download, about 8 GB of free memory, and roughly a quarter
  of an hour for the three models once the ten games are prepared.
- Full-match quality on the challenge match is unknown; only five minutes of it are labelled.

## Reproduce

```bash
python -m pipeline.run prepare-mall --bench-src /path/to/metrica-sample-data
python -m pipeline.run mall --mode model
python -m pipeline.mall.bench.score          # held-out games and the public window
python -m unittest pipeline.test_mall_deadball
```

Output SHA-256 (`possession-viewer/data/mall-model.csv`; Python 3.10.11 on Windows, numpy 1.26.4,
scipy 1.14.1, scikit-learn 1.5.2): `f26c3ce33a3cbfc0485ed6deacf922e1b0254bc872f1d970aa0d2e895e0c74bc`.
