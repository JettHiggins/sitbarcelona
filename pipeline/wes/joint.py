"""Constrained local decoder for uncertain Mall dead-ball windows.

The original Mall prediction is frozen outside selected windows. Existing
live H/A labels remain fixed. No answer key or competition-match label is read.
"""

import numpy as np

from ..mall import config as C
from ..mall.layer2_touches.touches import RECEIVE
from ..mall.layer5_sequence_decoder.decoder import (
    candidate_points, decode, duration_prior, frame_evidence,
)
from . import config as W

D, H, A = 2, 0, 1
NEG = -1e30


def _runs(mask):
    edge = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edge == 1), np.flatnonzero(edge == -1)))


def _control_mask(events, n):
    result = np.zeros(n, bool)
    for event in events:
        if (int(event['kind']) == RECEIVE and
                float(np.nan_to_num(event['contest'])) <= W.CONTROL_MAX_CONTEST):
            a = max(0, int(event['frame']))
            b = min(n, int(event['frame'] + event['duration']))
            result[a:b] = True
    return result


def windows(z, L2, L3, base):
    """Select short uncertainty neighborhoods from the model's own disagreement."""
    n = len(base)
    half = z['half']
    raw_live = ~L3['stage_viterbi'].astype(bool)
    d_trigger = (base == D) & (raw_live | _control_mask(L2['events'], n))
    trigger = d_trigger
    proposed = []
    for h in (1, 2):
        indices = np.flatnonzero(half == h)
        if not len(indices):
            continue
        lo, hi = int(indices[0]), int(indices[-1] + 1)
        for start, stop in _runs(trigger[lo:hi]):
            if stop - start < W.JOINT_MIN_TRIGGER_FRAMES:
                continue
            proposed.append((max(lo, lo + int(start) - W.JOINT_CONTEXT_FRAMES),
                             min(hi, lo + int(stop) + W.JOINT_CONTEXT_FRAMES)))
    merged = []
    for start, stop in proposed:
        if merged and start <= merged[-1][1] and half[start] == half[merged[-1][0]]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    return merged, dict(D_trigger_frames=int(d_trigger.sum()),
                        team_trigger_frames=0)


def _transition_cost(old, new, frame, points):
    if old is None or new is None or old == new:
        return 0.0
    if old == D or new == D:
        return float(C.DEAD_SWITCH_COST)
    return float(C.SEQ_SWITCH_COST - points.get(frame, (0.0, 0.0))[new])


def _allowed(base, hard):
    allowed = np.zeros((3, len(base)), bool)
    allowed[D] = base == D
    allowed[H] = ((base == D) & ~hard) | (base == H)
    allowed[A] = ((base == D) & ~hard) | (base == A)
    return allowed


def decode_window(start, stop, base, dead_llr, owner_llr, hard, points,
                  left=None, right=None):
    """Exact segment DP on a sparse boundary grid inside one uncertain window."""
    local = base[start:stop]
    allowed = _allowed(local, hard[start:stop])
    dead = np.nan_to_num(dead_llr[start:stop].astype(float))
    owner = np.nan_to_num(owner_llr[start:stop].astype(float))
    # Ownership chooses H versus A but cannot, by itself, turn a stoppage live.
    emissions = np.stack((
        -dead / 2 + owner / 2 - np.abs(owner) / 2,
        -dead / 2 - owner / 2 - np.abs(owner) / 2,
        dead / 2,
    ))
    cum = np.pad(np.cumsum(emissions, axis=1), ((0, 0), (1, 0)))
    forbidden = np.pad(np.cumsum(~allowed, axis=1), ((0, 0), (1, 0)))
    grid = range(((start + W.JOINT_BOUNDARY_GRID - 1) // W.JOINT_BOUNDARY_GRID) *
                 W.JOINT_BOUNDARY_GRID, stop, W.JOINT_BOUNDARY_GRID)
    edges = np.flatnonzero(local[1:] != local[:-1]) + start + 1
    event_points = (f for f, bonus in points.items()
                    if start < f < stop and max(bonus) > 0)
    bounds = np.array(sorted({start, stop, *grid, *edges, *event_points}), int)
    K = len(bounds) - 1
    rel = bounds - start
    dp = np.full((K + 1, 3), NEG)
    back_i = np.full((K + 1, 3), -1, int)
    back_t = np.full((K + 1, 3), -1, int)
    for j in range(1, K + 1):
        for state in (H, A, D):
            best = NEG
            for i in range(j):
                if forbidden[state, rel[j]] != forbidden[state, rel[i]]:
                    continue
                seg = cum[state, rel[j]] - cum[state, rel[i]]
                if state != D and not ((i == 0 and left == state) or
                                       (j == K and right == state)):
                    seg += float(duration_prior(bounds[j] - bounds[i]))
                if i == 0:
                    value = seg - _transition_cost(left, state, start, points)
                    prev_state = -1
                else:
                    values = [(dp[i, old] - _transition_cost(old, state, bounds[i], points), old)
                              for old in (H, A, D) if old != state]
                    value, prev_state = max(values)
                    value += seg
                if value > best:
                    best, back_i[j, state], back_t[j, state] = value, i, prev_state
            dp[j, state] = best
    final = [dp[K, state] - _transition_cost(state, right, stop, points)
             for state in (H, A, D)]
    state = int(np.argmax(final))
    if final[state] <= NEG / 2:
        return local.copy(), len(bounds)
    result = local.copy()
    j = K
    while j > 0:
        i = back_i[j, state]
        result[rel[i]:rel[j]] = state
        j, state = i, back_t[j, state]
    return result, len(bounds)


def predict_joint(z, F, L2, L3, L4):
    if 'stage_viterbi' not in L3:
        raise ValueError('Wes needs traced Layer 3 evidence. Rebuild the Mall layers with '
                         'the integrated code before using --reuse-cache.')
    base = decode(z, L2, L3, L4)
    owner = frame_evidence(L2, L4)
    selected, triggers = windows(z, L2, L3, base)
    outside = np.nan_to_num(F['ball_outside'], nan=0.0) > W.MAX_BALL_OUTSIDE_M
    hard = (z['half'] == 0) | outside
    if 'hard_kickoff' in L3:
        hard |= L3['hard_kickoff'].astype(bool)
    points = candidate_points(L2, z, L4)
    result = base.copy()
    trace = []
    for start, stop in selected:
        h = z['half'][start]
        left = int(base[start - 1]) if start and z['half'][start - 1] == h else None
        right = int(base[stop]) if stop < len(base) and z['half'][stop] == h else None
        revised, n_bounds = decode_window(start, stop, base, L3['dead_llr'], owner,
                                          hard, points, left, right)
        result[start:stop] = revised
        trace.append(dict(start=start, end=stop, boundaries=n_bounds,
                          changed_frames=int(np.sum(revised != base[start:stop])),
                          changed_D_frames=int(np.sum((revised == D) !=
                                                      (base[start:stop] == D)))))
    return result, dict(variant='wes', algorithm='joint-d', windows=trace, triggers=triggers,
                        changed_frames=int(np.sum(result != base)),
                        changed_dead_frames=int(np.sum((result == D) != (base == D))),
                        dead_revisions=[], change_candidates=[])
