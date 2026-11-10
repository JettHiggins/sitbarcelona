"""Paper §§2.3.1–2.3.3 and supplementary §S4: zones, player gains and losses."""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.signal import savgol_filter


@dataclass(frozen=True)
class Parameters:
    fps: int = 25
    possession_radius: float = 1.0
    duel_radius: float = 1.0
    turn_degrees: float = 10.0
    speed_change: float = 5.0
    displacement: float = 0.10
    smooth_window: int = 7
    smooth_order: int = 2

    def __post_init__(self):
        if self.possession_radius <= 0 or self.duel_radius < self.possession_radius:
            raise ValueError('Require 0 < possession radius <= duel radius')

    @classmethod
    def profile(cls, name):
        if name == 'a':
            return cls(possession_radius=0.5, speed_change=1.0, displacement=0.05)
        if name == 'bc':
            return cls()
        raise ValueError('Unknown paper provider profile')


def runs(mask):
    bounds = np.flatnonzero(np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8)))
    return list(zip(bounds[::2], bounds[1::2]))


def prepare_ball(raw, half, params):
    """Seven-frame SG on uninterrupted observed runs; no interpolation or Mall cleaning."""
    ball = np.asarray(raw, float).copy()
    known = np.isfinite(ball).all(axis=1)
    for h in (1, 2):
        for a, b in runs(known & (half == h)):
            if b - a >= params.smooth_window:
                ball[a:b] = savgol_filter(ball[a:b], params.smooth_window, params.smooth_order,
                                          axis=0, mode='interp')
    incoming = np.full_like(ball, np.nan)
    outgoing = np.full_like(ball, np.nan)
    delta = ball[1:] - ball[:-1]
    valid = known[1:] & known[:-1] & (half[1:] == half[:-1]) & (half[1:] > 0)
    delta[~valid] = np.nan
    incoming[1:] = delta * params.fps
    outgoing[:-1] = delta * params.fps
    return ball, incoming, outgoing


def detect(players, teams, ball_raw, half, dead, params=Parameters()):
    """Native states: 0 dead, 1 no possession, 2 possession-zone, 3 duel.

    Returned gains/losses are player events, not inferred team possession. No
    co-motion duration, learned probabilities, arbitrary touch peaks or decoder.
    """
    half = np.asarray(half)
    dead = np.asarray(dead, bool) | (half == 0)
    players = np.asarray(players)
    teams = np.asarray(teams)
    if teams.ndim == 1:
        teams = np.broadcast_to(teams, players.shape[:2])
    ball, vin, vout = prepare_ball(ball_raw, half, params)
    distances = np.linalg.norm(players - ball[:, None, :], axis=2)
    distances[~np.isin(teams, [0, 1])] = np.nan
    seen = np.isfinite(players).all(axis=2) & np.isin(teams, [0, 1]) & ~dead[:, None]
    pz = seen & (distances <= params.possession_radius)
    dz = seen & (distances <= params.duel_radius)
    duel = (dz & (teams == 0)).any(axis=1) & (dz & (teams == 1)).any(axis=1)
    state = np.where(dead, 0, np.where(duel, 3, np.where(pz.any(axis=1), 2, 1))).astype(np.int8)
    indices = np.arange(len(ball))
    speed_in = np.linalg.norm(vin, axis=1)
    speed_out = np.linalg.norm(vout, axis=1)
    gains, losses, intervals = [], [], []
    for slot in range(players.shape[1]):
        for h in (1, 2):
            # A provider slot changing teams is a separate interval.
            for team in (0, 1):
                for a, b in runs(pz[:, slot] & (half == h) & (teams[:, slot] == team)):
                    a, b = int(a), int(b)
                    norm = np.linalg.norm(vin[a]) * np.linalg.norm(vout[b - 1])
                    cosine = float(np.dot(vin[a], vout[b - 1]) / norm) if np.isfinite(norm) and norm > 0 else None
                    speed_diffs = np.abs(speed_in[a:b] - speed_out[a:b])
                    change = float(np.nanmax(speed_diffs)) if np.isfinite(speed_diffs).any() else 0.0
                    angle_gain = cosine is not None and cosine < np.cos(np.deg2rad(params.turn_degrees))
                    gain = angle_gain or change > params.speed_change
                    interval = dict(start=a, end=b, slot=slot, team=team, half=h,
                                    gain=bool(gain), cosine=cosine, speed_change=change,
                                    distance=float(distances[a:b, slot].min()))
                    intervals.append(interval)
                    if gain:
                        gains.append(dict(frame=a, slot=slot, team=team, half=h, end=b,
                                          kind='gain', duel=bool(duel[a]),
                                          speed_change=change, direction_change=bool(angle_gain)))
    # §2.3.3 excludes fly-bys from the possession step. They must not terminate
    # another player's possession simply by being the next geometrical PZ hit.
    marks = np.full(len(ball), -1, np.int32)
    for gain in gains:
        marks[gain['frame']] = gain['frame']
    last_any = np.maximum.accumulate(marks)
    alive = ~dead & (half > 0)
    region = np.cumsum(alive & np.r_[True, ~alive[:-1] | (half[1:] != half[:-1])])
    verified = np.zeros_like(pz)
    for slot in range(players.shape[1]):
        own_marks = np.full(len(ball), -1, np.int32)
        for gain in gains:
            if gain['slot'] == slot:
                own_marks[gain['frame']] = gain['frame']
        last_own = np.maximum.accumulate(own_marks)
        for interval in (r for r in intervals if r['slot'] == slot):
            a, b = interval['start'], interval['end']
            earlier = int(last_own[a])
            continuation = earlier >= 0 and earlier == last_any[a] and region[earlier] == region[a]
            interval['verified'] = bool(interval['gain'] or continuation)
            if interval['verified']:
                verified[a:b, slot] = True
    control = verified | (duel[:, None] & dz)
    occupied = control.any(axis=1)
    nxt = np.minimum.accumulate(np.where(occupied, indices, len(ball))[::-1])[::-1]
    for slot in range(players.shape[1]):
        # Loss requires actual ball displacement AND absence of A at next control.
        for a, b in runs(control[:, slot]):
            f = int(b - 1)
            if f + 1 >= len(ball) or half[f + 1] != half[f]:
                continue
            if np.isfinite(distances[f + 1, slot]) and distances[f + 1, slot] <= params.possession_radius:
                continue
            displacement = np.linalg.norm(ball[f + 1] - ball[f])
            following = int(nxt[f + 1])
            if not np.isfinite(displacement) or displacement <= params.displacement:
                continue
            # A pass/shot may end at a stoppage or half without a next receiver.
            # Only a same-player return WITHIN this live interval suppresses loss.
            same_live = following < len(ball) and half[following] == half[f] and not dead[f + 1:following + 1].any()
            if same_live and control[following, slot]:
                continue
            losses.append(dict(frame=f, slot=slot, team=int(teams[f, slot]), half=int(half[f]),
                               kind='loss', next_control=following if same_live else None, duel=bool(duel[f]),
                               displacement=float(displacement)))
    gains.sort(key=lambda e: (e['frame'], e['slot']))
    losses.sort(key=lambda e: (e['frame'], e['slot']))
    return dict(ball=ball, incoming=vin, outgoing=vout, distances=distances, pz=pz, verified_pz=verified,
                duel=duel, state=state, gains=gains, losses=losses, intervals=intervals,
                parameters=asdict(params))
