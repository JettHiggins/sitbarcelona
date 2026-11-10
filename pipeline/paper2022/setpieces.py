"""Paper §2.3.4, Fig. 5 and supplement S1: positional triggers and restart patterns."""
import numpy as np

from .core import runs

# Supplement S4.2. Kickoff epsilon_k1 is in the published optimal range 3–5 m.
TOLERANCES = dict(k1=3.0, k2=2.0, p1=2.0, p2=2.0, p3=4.0, c=3.0, t=0.5, g=0.5, f=1.0)
ORDER = ('kickoff', 'penalty', 'corner', 'throw_in', 'goal_kick')
DEAD_EVENT = dict(kickoff='goal', penalty='penalty_awarded', corner='out_for_corner',
                  throw_in='out_for_throw_in', goal_kick='out_for_goal_kick',
                  free_kick='foul', free_kick_uncertain='foul_uncertain')


def triggers(xy, teams, half, signs, pitch):
    """Boolean (frame,slot) candidate executors for each restart configuration."""
    n, k = xy.shape[:2]
    L, W = map(float, pitch)
    valid = np.isfinite(xy).all(axis=2) & np.isin(teams, [0, 1])
    result = {name: np.zeros((n, k), bool) for name in ORDER}
    center = np.array([L / 2, W / 2])
    for team in (0, 1):
        mine = valid & (teams == team)
        opponents = valid & (teams == 1 - team)
        sign = signs[half.astype(int), team]
        x = np.where(sign[:, None] > 0, xy[:, :, 0], L - xy[:, :, 0])
        y = xy[:, :, 1]
        # Both teams must be in their own halves, with the published tolerance.
        opp_x = L - x
        own_ok = (~mine | (x <= L / 2 + TOLERANCES['k1'])).all(axis=1)
        opp_ok = (~opponents | (opp_x <= L / 2 + TOLERANCES['k1'])).all(axis=1)
        formation = own_ok & opp_ok & mine.any(axis=1) & opponents.any(axis=1)
        result['kickoff'] |= mine & formation[:, None] & (np.linalg.norm(xy - center, axis=2) <= TOLERANCES['k2'])
        result['corner'] |= mine & (x >= L - TOLERANCES['c']) & ((y <= TOLERANCES['c']) | (y >= W - TOLERANCES['c']))
        result['throw_in'] |= mine & ((y <= TOLERANCES['t']) | (y >= W - TOLERANCES['t']))
        result['goal_kick'] |= mine & (x <= 5.5 + TOLERANCES['g']) & (np.abs(y - W / 2) <= 9.16 + TOLERANCES['g'])
        # Penalty executor lies in the asymmetric 2 m square around the penalty mark.
        mark = L - 11.0
        taker = mine & (x >= mark - 3 * TOLERANCES['p2'] / 4) & (x <= mark + TOLERANCES['p2'] / 4) & (np.abs(y - W / 2) <= TOLERANCES['p2'] / 2)
        keeper = opponents & (np.abs(x - L) <= TOLERANCES['p1']) & (np.abs(y - W / 2) <= 3.66 + TOLERANCES['p1'])
        penalty_inside = (x >= L - 16.5 + TOLERANCES['p3']) & (np.abs(y - W / 2) <= 20.16 - TOLERANCES['p3'])
        near_mark = np.hypot(x - mark, y - W / 2) < 9.15 - TOLERANCES['p3']
        forbidden = valid & ~taker & ~keeper & (penalty_inside | near_mark)
        configuration = (taker.sum(axis=1) == 1) & (keeper.sum(axis=1) == 1) & ~forbidden.any(axis=1)
        result['penalty'] |= taker & configuration[:, None]
    return result


def detect(xy, teams, half, signs, pitch, dead, core):
    trig = triggers(xy, teams, half, signs, pitch)
    restarts = []
    for h in (1, 2):
        for a, b in runs(np.asarray(dead, bool) & (half == h)):
            a, b = int(a), int(b)
            if b >= len(half) or half[b] != h or dead[b]:
                continue
            complete = {kind: np.flatnonzero(mask[b - 1]) for kind, mask in trig.items()}
            incomplete = {}
            for kind in ('corner', 'throw_in'):
                candidates = []
                for slot in np.flatnonzero(trig[kind][a:b].any(axis=0)):
                    last = a + int(np.flatnonzero(trig[kind][a:b, slot])[-1])
                    missing_after = last + 1 < b and not np.isfinite(xy[last + 1:b, slot]).all(axis=1).any()
                    candidates.append((int(slot), bool(missing_after)))
                incomplete[kind] = candidates
            pz = core['pz'][b]

            def choose(slots, require_pattern=True):
                slots = np.array(slots, int)
                if require_pattern:
                    slots = slots[pz[slots]]
                return int(slots[np.nanargmin(core['distances'][b, slots])]) if len(slots) and np.isfinite(core['distances'][b, slots]).any() else (int(slots[0]) if len(slots) and not require_pattern else None)

            kind, slot, reason = None, None, None
            # Fig. 5: a failed kickoff/penalty pattern goes directly to free kick;
            # corner/throw-in failures may continue down the hierarchy.
            for name in ORDER:
                if not len(complete[name]):
                    continue
                candidate = choose(complete[name])
                if candidate is not None:
                    kind, slot, reason = name, candidate, 'complete_trigger_and_pattern'
                    break
                if name in ('kickoff', 'penalty'):
                    kind, slot, reason = 'free_kick', choose(np.flatnonzero(pz)), 'complete_trigger_failed_pattern'
                    break
            if kind == 'free_kick':
                kind = None
            if kind is None:
                # Fig. S1 extends the free-kick fallback after the main hierarchy
                # fails to confirm a restart, including failed trigger patterns.
                free_slot = choose(np.flatnonzero(pz))
                later_loss = next((e for e in core['losses'] if free_slot == e['slot'] and e['half'] == h and e['frame'] >= b), None)
                stationary = False
                if free_slot is not None and later_loss is not None:
                    end = later_loss['frame'] + 1
                    if not np.asarray(dead)[b:end].any():
                        displacement = np.linalg.norm(xy[b:end, free_slot] - xy[b, free_slot], axis=1)
                        stationary = np.isfinite(displacement).all() and displacement.max() <= TOLERANCES['f']
                if stationary:
                    # S1 issue 1: recheck other triggers/patterns at the first release.
                    f = later_loss['frame']
                    selected = next((name for name in ORDER if trig[name][f, free_slot] and core['pz'][f, free_slot]), None)
                    kind, slot, reason = selected or 'free_kick', free_slot, 's1_stationary_executor_rechecked_at_release'
                if kind is None and free_slot is None:
                    for name in ('corner', 'throw_in'):
                        missing = [s for s, absent in incomplete[name] if absent]
                        if missing:
                            kind, slot, reason = name, missing[0], 's1_incomplete_trigger_missing_executor'
                            break
                if kind is None:
                    # S1 issue 3: no usable inbound pattern; inspect trigger evidence.
                    options = [(name, np.flatnonzero(trig[name][a:b].any(axis=0))) for name in ('kickoff', 'penalty')]
                    options += [(name, complete[name]) for name in ('corner', 'throw_in')]
                    options += [(name, np.array([s for s, _ in incomplete[name]], int)) for name in ('corner', 'throw_in')]
                    options += [('goal_kick', np.flatnonzero(trig['goal_kick'][a:b].any(axis=0)))]
                    for name, candidates in options:
                        if len(candidates):
                            kind, slot, reason = name, choose(candidates, False), 's1_trigger_only_uncertain_pattern'
                            break
                if kind is None:
                    kind, slot, reason = 'free_kick_uncertain', free_slot, 's1_unresolved'
            team = int(teams[b, slot]) if slot is not None and teams[b, slot] in (0, 1) else None
            if team is None and slot is not None:
                seen = teams[a:b, slot]
                seen = seen[np.isin(seen, [0, 1])]
                team = int(seen[-1]) if len(seen) else None
            first_in_half = not any(e['half'] == h and e['kind'] == 'kickoff' for e in restarts)
            dead_event = 'period_start' if kind == 'kickoff' and first_in_half else DEAD_EVENT[kind]
            restarts.append(dict(frame=b, dead_start=a, dead_end=b, half=h, kind=kind,
                                 slot=slot, team=team, dead_event=dead_event, reason=reason,
                                 complete_triggers={k: list(map(int, v)) for k, v in complete.items()}))
    # Repeated kickoff: no penalty-area visit between kickoffs means earlier was incorrect.
    for h in (1, 2):
        kickoffs = [e for e in restarts if e['half'] == h and e['kind'] == 'kickoff']
        for before, after in zip(kickoffs[:-1], kickoffs[1:]):
            ball = core['ball'][before['frame']:after['frame']]
            L, W = pitch
            in_area = ((ball[:, 0] <= 16.5) | (ball[:, 0] >= L - 16.5)) & (np.abs(ball[:, 1] - W / 2) <= 20.16)
            if np.isfinite(ball).all(axis=1).any() and not in_area.any():
                before['kind'] = 'incorrect_kickoff'
                after['dead_event'] = 'referee_interruption'
    return restarts
