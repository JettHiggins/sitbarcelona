"""Make a bench game look like the challenge's tracking, so a score on it means something.

Measured on the challenge match (cache/layer0_tactical.npz, graded frames) vs the bench:

                                    challenge          bench (as delivered)
  ball seen while alive / dead      97.8% / 24%        96-98% / 1-15%     -> left as it is
  ball speed p50 / p90 / p99        5 / 14 / 25 m/s    5-7 / 15 / 21-24   -> left as it is
  players seen per team             10.2-10.3 of 11    11.0               -> dropouts
  player tracks (slot runs) / min   11.5               ~0 (one per player) -> fragmentation
  cross-team slot mix-ups           66 repaired        none               -> mix-ups

So the ball is not touched, and the players get:
  dropouts       each player is lost for a while every ~100 s (log-normal, median 4 s), which
                 leaves ~6% of player-frames missing
  fragmentation  every time a player is re-acquired he comes back in a new slot (the lowest free
                 slot of his team), as an optical tracker does
  mix-ups        6% of re-acquisitions start in a slot of the *other* team, for a log-normal
                 ~3 s, then hand back to a slot of his own team - the 12-vs-10 episodes layer 0's
                 team repair exists for
  jitter         0.1 m white noise on every position

and Smart-Tagging-like set-piece windows are simulated from the answer key's restarts: 95% of
restarts get one, starting a log-normal ~7 s (2-25 s) before the restart, 35 s long (5 s for a
kick-off), tagged with the restarting team except for throw-ins (the real tagging says N/A).
Measured on the challenge match: 56/59 set pieces tagged, the restart a median 7 s into a window
of median 35 s.

Everything is seeded per game, so the bench is reproducible.
"""

import numpy as np

SET_PIECE_TYPES = ["Throw-in", "Corner Kick", "Goal Kick", "Free Kick", "Kick Off (After Goal)",
                   "Kick Off (Start)"]
FPS = 25


def _runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def players(P, team_of_slot, seed, on_mean_s=100.0, off_median_s=4.0, p_mix=0.06,
            mix_median_s=3.0, jitter_m=0.1, pitch=(105.0, 68.0)):
    """(n, k, 2) one column per real player -> (n, slots, 2) tracker-like slots, team_of_slot."""
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    pieces = []                                     # (start, end, true column, slot team)
    for k in range(P.shape[1]):
        t = int(team_of_slot[k])
        for a, b in _runs(~np.isnan(P[:, k, 0])):
            f = a
            first = True
            while f < b:
                on = max(25, int(rng.exponential(on_mean_s) * FPS))
                e = min(b, f + on)
                if not first and rng.random() < p_mix:   # re-acquired under the wrong team
                    m = min(e, f + max(5, int(rng.lognormal(np.log(mix_median_s), 0.7) * FPS)))
                    pieces.append((f, m, k, 1 - t))
                    if m < e:
                        pieces.append((m, e, k, t))
                else:
                    pieces.append((f, e, k, t))
                first = False
                f = e + int(rng.lognormal(np.log(off_median_s), 1.0) * FPS)
    pieces.sort()
    out_cols, out_team = [], []
    free_at = {0: [], 1: []}                        # per team: [(frame slot becomes free, slot)]
    for a, b, k, t in pieces:
        slots = free_at[t]
        free = [i for i, (ff, s) in enumerate(slots) if ff <= a]
        if free:
            i = min(free, key=lambda i: slots[i][1])
            s = slots[i][1]
            slots[i] = (b, s)
        else:
            s = len(out_cols)
            out_cols.append(np.full((n, 2), np.nan))
            out_team.append(t)
            slots.append((b, s))
        out_cols[s][a:b] = P[a:b, k]
    Q = np.stack(out_cols, axis=1)
    Q = Q + rng.normal(0, jitter_m, Q.shape) / np.array(pitch)
    return Q, np.array(out_team, np.int8)


def set_piece_tags(events, n, seed, restart_fn, p_tag=0.95):
    rng = np.random.default_rng(seed + 1)
    sp = np.zeros(n, np.int8)
    sp_team = np.full(n, -1, np.int8)
    started = set()
    for e in sorted(events, key=lambda e: e["start"]):
        if not restart_fn(e) or rng.random() > p_tag:
            continue
        st = e["subtype"].upper()
        if e["type"] != "SET PIECE":
            kind = "Goal Kick"
        elif "THROW" in st:
            kind = "Throw-in"
        elif "CORNER" in st:
            kind = "Corner Kick"
        elif "KICK OFF" in st:
            kind = "Kick Off (After Goal)" if e["period"] in started else "Kick Off (Start)"
            started.add(e["period"])
        else:
            kind = "Free Kick"
        lead = np.clip(rng.lognormal(np.log(7.0), 0.6), 2, 25)
        length = 5.0 if "Kick Off" in kind else 35.0
        a = max(0, int(e["start"] - lead * FPS))
        b = min(n, int(a + length * FPS))
        sp[a:b] = 1 + SET_PIECE_TYPES.index(kind)
        if kind != "Throw-in" and e["team"] is not None:
            sp_team[a:b] = e["team"]
    return {"st_possessor": np.zeros(n, np.int8), "st_set_piece": sp, "st_set_piece_team": sp_team}


def ball_dropouts(ball, dead, period, seed, ball_z=None, min_gap_s=5.0, p_long=0.85, p_short=0.3):
    """Hide the ball the way the challenge's optical tracker does. Returns a copy of `ball`.

    Measured on the challenge match without any labels (cleaned layer 0, graded frames): the
    ball is unknown on 41% of frames, in ~150 gaps that are almost never shorter than 5 s (the
    provider bridges shorter losses itself) - median 11 s, 90th percentile 33 s - and it is slow
    (median 1.4 m/s) when it is lost. So most of the loss is the ball being picked up or leaving
    the picture at a stoppage, plus occasional long losses in open play.

    A feed that tracks the ball through every stoppage (the DFL games) is degraded with:
      stoppages  a stoppage of >= 8 s loses the ball with p = 0.85, a shorter one with p = 0.3,
                 from shortly around the stop until either some seconds before the restart (the
                 ball is put down and seen again) or just after it (it re-appears already moving)
      open play  0.1-0.6 losses a minute (drawn per game) of 5-20 s, half of them starting while
                 the ball is in the air if heights are known
    A simulated loss shorter than `min_gap_s` is dropped, as the provider would have bridged it.
    These ranges are an augmentation, not a fit: no label of the challenge match is involved.

    Caveat for anything trained on the result: the DFL feed freezes the ball where play stopped
    (its speed is exactly 0 on every dead frame), which a real tracker never shows. So the ball
    of a dead frame - hidden or not - must not be a model input (layer3_dead_ball/learned.py).
    """
    rng = np.random.default_rng(seed + 7)
    out = ball.copy()
    n = len(ball)
    hide = np.zeros(n, bool)
    for a, b in _runs(dead & (period > 0)):
        length = (b - a) / FPS
        if rng.random() > (p_long if length >= 8 else p_short):
            continue
        s = a + int(rng.uniform(-1.0, 3.0) * FPS)
        e = b - int(rng.uniform(0.0, 4.0) * FPS) if rng.random() < 0.6 else             b + int(rng.uniform(0.0, 2.0) * FPS)
        s, e = max(0, s), min(n, e)
        if e - s >= min_gap_s * FPS:
            hide[s:e] = True
    alive = ~dead & (period > 0)
    rate = rng.uniform(0.1, 0.6)                              # losses per minute of open play
    k = rng.poisson(rate * alive.sum() / FPS / 60.0)
    idx = np.flatnonzero(alive)
    high = np.flatnonzero(alive & (np.nan_to_num(ball_z) > 3.0)) if ball_z is not None else []
    for _ in range(k):
        pool = high if len(high) and rng.random() < 0.5 else idx
        s = int(pool[rng.integers(len(pool))])
        e = min(n, s + int(np.clip(rng.lognormal(np.log(8.0), 0.5), min_gap_s, 20.0) * FPS))
        hide[s:e] = True
    out[hide] = np.nan
    return out


def set_piece_tags_from_key(label, ball, pitch, n, seed, p_tag=0.95):
    """Smart-Tagging-like set-piece windows for a game with a frame-level key but no event log:
    one per restart after a stoppage of >= 4 s, typed by where the ball is put back into play."""
    rng = np.random.default_rng(seed + 1)
    sp = np.zeros(n, np.int8)
    sp_team = np.full(n, -1, np.int8)
    L, W = pitch
    dead = label == 2
    first = True
    for a, b in _runs(dead):
        if b >= n or (b - a) < 4 * FPS and not first:
            continue
        if rng.random() > p_tag:
            first = False
            continue
        xy = ball[b:b + 13] * np.array(pitch)
        xy = xy[~np.isnan(xy[:, 0])]
        x, y = (xy[0] if len(xy) else (L / 2, W / 2))
        if first or (abs(x - L / 2) < 2 and abs(y - W / 2) < 2):
            kind = "Kick Off (Start)" if first else "Kick Off (After Goal)"
        elif min(x, L - x) < 2 and min(y, W - y) < 2:
            kind = "Corner Kick"
        elif min(y, W - y) < 1.5:
            kind = "Throw-in"
        elif min(x, L - x) < 6.5 and abs(y - W / 2) < 10:
            kind = "Goal Kick"
        else:
            kind = "Free Kick"
        first = False
        lead = np.clip(rng.lognormal(np.log(7.0), 0.6), 2, 25)
        length = 5.0 if "Kick Off" in kind else 35.0
        s = max(0, int(b - lead * FPS))
        e = min(n, int(s + length * FPS))
        sp[s:e] = 1 + SET_PIECE_TYPES.index(kind)
        team = label[min(n - 1, b)]
        if kind != "Throw-in" and team in (0, 1):
            sp_team[s:e] = team
    return {"st_possessor": np.zeros(n, np.int8), "st_set_piece": sp, "st_set_piece_team": sp_team}
