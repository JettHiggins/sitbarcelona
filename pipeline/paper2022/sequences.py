"""Native control/no-possession sequences from verified player gains and losses (S5)."""
import numpy as np

from .core import runs


def native_sequences(core, half, dead, restarts, passing, rejected):
    n = len(half)
    player = np.full(n, -1, np.int16)
    gaining, losing = {}, {}
    for event in core['gains']:
        if (event['frame'], event['slot']) not in rejected:
            gaining.setdefault(event['frame'], []).append(event)
    for restart in restarts:
        if restart['slot'] is not None and restart['team'] in (0, 1):
            gaining.setdefault(restart['frame'], []).append(restart)
    for event in passing:
        losing.setdefault(event['frame'], set()).add(event['slot'])
    current, previous_half = -1, None
    pending_duel = set()
    for f in range(n):
        if dead[f] or half[f] == 0 or half[f] != previous_half:
            current = -1
            pending_duel.clear()
        previous_half = half[f]
        if dead[f] or half[f] == 0:
            continue
        gains = gaining.get(f, [])
        if core['duel'][f]:
            pending_duel.update(e['slot'] for e in gains)
        else:
            candidates = {e['slot'] for e in gains}
            candidates |= {slot for slot in pending_duel if core['pz'][f, slot]}
            if candidates:
                # Same-team multiple PZ occupants: nearest player is the tie break.
                # Opponent ambiguity has already been marked as a duel geometrically.
                current = min(candidates, key=lambda slot: (core['distances'][f, slot], slot))
            pending_duel.clear()
        player[f] = current
        if current in losing.get(f, set()):
            current = -1
    control = np.where(dead | (half == 0), 0,
                       np.where(core['duel'], 3, np.where(player >= 0, 2, 1))).astype(np.int8)
    sequences = []
    boundaries = np.r_[0, np.flatnonzero((player[1:] != player[:-1]) | (half[1:] != half[:-1]) |
                                        (np.asarray(dead)[1:] != np.asarray(dead)[:-1])) + 1, n]
    pass_keys = {(e['frame'], e['slot']): e for e in passing}
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        a, b, slot = int(a), int(b), int(player[a])
        if slot < 0 or dead[a] or half[a] == 0:
            continue
        team = int(core['teams'][a, slot])
        ending_pass = pass_keys.get((b - 1, slot))
        if ending_pass:
            outcome = ending_pass['event']
        elif b == n or half[b] != half[a]:
            outcome = 'end_of_half'
        elif dead[b]:
            outcome = 'dead_ball'
        elif player[b] >= 0:
            outcome = 'maintained' if core['teams'][b, player[b]] == team else 'lost'
        else:
            outcome = 'no_possession'
        sequences.append(dict(start=a, end=b, slot=slot, team=team, half=int(half[a]), outcome=outcome))
    loose = [dict(start=int(a), end=int(b)) for a, b in runs(control == 1)]
    return player, control, sequences, loose
