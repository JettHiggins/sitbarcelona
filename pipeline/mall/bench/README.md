# Bench: scoring the pipeline on other games

The challenge match has only 5 public minutes of ground truth (17 possessions). We can't tune on
them, and they're too few to tell two versions apart. The bench gives us a full-match score on
three other games: Metrica Sports' open sample data (RULES.md allows other games).

    git clone https://github.com/metrica-sports/sample-data   # set BENCH_SRC to its path
    python -m pipeline.mall.bench.build      # degrade the tracking, write answer keys, ~10 s per game
    python -m pipeline.mall.bench.score      # rerun layers 3-5 on each game, official scorer

Seven more games come from the open DFL / IDSSE Bundesliga dataset (Bassek et al., Scientific
Data 2025, CC-BY 4.0), whose position feed carries the ball status (in play / dead) and the team
in possession on every frame - a frame-exact answer key with no event-log reconstruction:

    python -m pipeline.mall.bench.idsse --download      # ~2.6 GB into data/idsse (gitignored)
    python -m pipeline.mall.bench.build --games --dfl 1 2 3 4 5 6 7

| file | what it does |
|---|---|
| idsse.py | downloads and reads the seven DFL matches: players, ball, ball height, H/A/D key |
| degrade.py `ball_dropouts` | the DFL ball is tracked through every stoppage; this hides it the way the challenge's tracker loses it (long losses at stoppages, occasional 5-20 s losses in open play, nothing shorter than 5 s) |
| degrade.py `set_piece_tags_from_key` | Smart-Tagging-like set-piece windows for a game with no event log |
| score.py | reruns layers 3-5 on every bench game (each scored by models that never saw it) and on the challenge match, and prints the official metrics. The bench is the selection set; the public window is a read-out |

Two properties of these feeds decide what may be learned from them (measured, see
`layer3_dead_ball/learned.py`): Metrica's ball is unknown on 92-99% of dead frames, and the DFL
ball is frozen in place on all of them. The challenge's ball is neither. So a dead-ball model must
not see the ball of a bench game; the players transfer, the ball does not.

| file | what it does |
|---|---|
| metrica.py | reads the three games: CSV for games 1-2, EPTS + JSON for game 3 (handles substitution blocks) |
| labels.py | builds H/A/D answer keys from the event log, following Metrica's own event definitions (`sample-data/documentation/events-definitions.pdf`). `verify()` checks every key against its event log, event by event |
| degrade.py | makes the tracking look like ours: player dropouts, fresh track slots on re-acquisition, cross-team mix-ups, jitter, and simulated Smart Tagging set-piece windows. The ball is left as delivered because it already matches ours |
| build.py | writes `cache/layer0_bench{g}.npz` plus the answer key and the baseline, all gitignored |

How the degraded bench compares with the challenge match:

| | challenge | bench |
|---|---|---|
| ball seen while alive / dead | 97.8% / 24% | 96-98% / 1-15% |
| players seen per team | 10.2-10.3 | 10.3 |
| player tracks per minute | 11.5 | 12.8 |

The answer keys come from human-tagged events, not from corrected tracking, so they only
approximate the private ground truth. The bench is for comparing versions of the pipeline, not
for predicting the leaderboard number.

## Result (v0, the pipeline as of layer 5)

| mean of 3 games | baseline | pipeline | our team + true dead | true team + our dead |
|---|---|---|---|---|
| spellF1 | 0.355 | 0.508 | 0.532 | 0.872 |
| macroF1 | 0.795 | 0.870 | 0.919 | 0.929 |

The two swapped columns are diagnostics:

- **Our team + true dead** takes our team calls and uses the true stoppages. It only gains 0.024.
- **True team + our dead** takes our stoppages and uses the true team calls. It gains 0.36.

So most of the remaining loss comes from who has the ball (layers 2, 4 and 5), not from when the
ball is dead (layer 3). We also find too few possessions: about 330 against a truth of 370-410.

## History

| version | bench spellF1 | macroF1 | team acc. (live) | public 5 min spellF1 / macroF1 |
|---|---|---|---|---|
| v0: self-trained layer 4, hand-set decoder | 0.508 | 0.870 | 0.906 | 0.545 / 0.830 |
| v1: cross-game layer 4 (frame model + change points), decoder chosen on the bench | **0.672** | **0.899** | **0.952** | **0.562 / 0.848** |

The v1 models are scored leave-one-game-out. Nothing was chosen on the public window.

## Answer keys v2: Metrica's own definitions

`labels.py` now follows `events-definitions.pdf` rule by rule:

- **Possession** starts at a player's touch and passes only to another player taking over.
- **Out of play** means the ball went out of bounds *or* the referee stopped play (fouls without
  advantage, offsides, forced stops).
- **A RECOVERY can be a restart.** It "brings the ball back into play via a dead-ball situation
  that is not a SET PIECE" (goal kicks, drop balls).

Checked against the event logs:

| | game 1 | game 2 | game 3 |
|---|---|---|---|
| possession changes on a ball-playing event of the new team | 283 / 283 | 264 / 264 | 305 / 305 |
| ball-playing events inside a dead spell | 0 | 0 | 0 |
| ball-playing events whose frame has their own team | 1434 / 1435 | 1545 / 1549 | 3196 / 3208 |
| stoppages followed by dead time | 151 / 153 | 140 / 144 | 133 / 135 |

The remaining mismatches are all in the source data:

- **Team mismatches** are hand-overs where both teams play the ball on the same frame.
- **Stoppages without dead time** are end-of-half stops, or restarts tagged on the very next frame.

The v1 keys are kept as `cache/bench/bench{g}_gt_v1.csv`; 1.5-5% of frames changed. With the v2
keys and models retrained on them, the pipeline scores spellF1 **0.681**, macroF1 **0.901**
(baseline 0.361 / 0.816).
