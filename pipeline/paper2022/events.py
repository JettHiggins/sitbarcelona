"""Paper Fig. 6 and §2.4.3: linked passing, receiving and goalkeeper events."""
import numpy as np


def attack_xy(point, team, half, signs, pitch):
    L, W = pitch
    return (float(point[0]) if signs[half, team] > 0 else L - float(point[0]), float(point[1]))


def in_penalty(point, team, half, signs, pitch):
    x, y = attack_xy(point, team, half, signs, pitch)
    return x >= pitch[0] - 16.5 and abs(y - pitch[1] / 2) <= 20.16


def zone(point, team, half, signs, pitch):
    """Fig. 6f digitization: metric zone boundary not specified in the text.

    Last 40% of pitch, central penalty-area-width band = shot zone; outer
    bands = cross zone. This geometric interpretation is explicitly an approximation.
    """
    x, y = attack_xy(point, team, half, signs, pitch)
    if x < 0.6 * pitch[0]:
        return 'other'
    return 'shot' if abs(y - pitch[1] / 2) <= 20.16 else 'cross'


def trajectory(core, frame, team, half, signs, pitch):
    point = core['ball'][frame]
    velocity = core['outgoing'][frame]
    x, y = attack_xy(point, team, half, signs, pitch)
    vx = velocity[0] * signs[half, team]
    forward = bool(np.isfinite(vx) and vx > 0)
    goal_y = y + velocity[1] * (pitch[0] - x) / vx if forward else float('nan')
    on_target = forward and np.isfinite(goal_y) and abs(goal_y - pitch[1] / 2) <= 3.66 + 0.25
    return forward, bool(on_target)


def infer_goalkeepers(xy, teams, half, signs, pitch):
    """Input adaptation: anonymous ATD slots lack explicit goalkeeper identities."""
    keeper = np.zeros(teams.shape, bool)
    assignments = []
    for h in (1, 2):
        for team in (0, 1):
            mask = (half[:, None] == h) & (teams == team) & np.isfinite(xy[:, :, 0])
            score = []
            for slot in range(xy.shape[1]):
                seen = mask[:, slot]
                if not seen.any():
                    continue
                x = xy[seen, slot, 0]
                depth = x if signs[h, team] > 0 else pitch[0] - x
                score.append((float(np.median(depth)), slot))
            # At each frame choose the most defensively stationed available slot.
            assigned = np.zeros(len(half), bool)
            for median, slot in sorted(score):
                use = mask[:, slot] & ~assigned
                keeper[use, slot] = True
                assigned |= use
                if use.any():
                    assignments.append(dict(half=h, team=team, slot=slot, frames=int(use.sum()), median_depth=median))
    return keeper, assignments


def classify(core, xy, teams, half, signs, pitch, dead, restarts, keepers):
    # Confirmed gains and restart executors, not every geometric fly-by, authorize a loss.
    timeline = [(e['frame'], 1, 'gain', e) for e in core['gains']]
    timeline += [(e['frame'], 2, 'loss', e) for e in core['losses']]
    timeline += [(e['frame'], 0, 'restart', e) for e in restarts if e['slot'] is not None]
    active, active_half, accepted_losses = set(), None, []
    last_time = None
    for f, _, kind, event in sorted(timeline, key=lambda t: (t[0], t[1])):
        if active_half != event['half'] or (last_time is not None and np.asarray(dead)[last_time:f + 1].any()):
            active = set()
        active_half, last_time = event['half'], f
        slot = event['slot']
        if kind in ('gain', 'restart'):
            if kind == 'restart' or not event.get('duel', False):
                active = {slot}
            else:
                active.add(slot)
        elif slot in active:
            accepted_losses.append(event)
            active.discard(slot)

    passing, receiving, rejected = [], {}, set()
    for g in core['gains']:
        receiving[(g['frame'], g['slot'])] = dict(g, event='reception', origin='unresolved')

    for loss in accepted_losses:
        f, team, h, actor = loss['frame'], loss['team'], loss['half'], loss['slot']
        later_dead = np.flatnonzero(np.asarray(dead)[f + 1:] | (half[f + 1:] != h))
        stop = f + 1 + int(later_dead[0]) if len(later_dead) else len(half)
        gain = next((g for g in core['gains'] if f <= g['frame'] < stop and g['half'] == h
                     and not (g['frame'] == f and g['slot'] == actor)), None)
        restart = next((r for r in restarts if r['half'] == h and r['dead_start'] == stop), None)
        origin_zone = zone(core['ball'][f], team, h, signs, pitch)
        forward, target = trajectory(core, f, team, h, signs, pitch)
        shot_name = 'shot_on_target' if target else 'shot_off_target'
        name, reason = 'pass', 'ordinary_loss'
        outcome = 'end_of_half' if stop >= len(half) or half[stop] != h else 'dead_ball'
        saving = None
        next_gain = None
        goalkeeper_gain = (gain is not None and gain['team'] != team and keepers[gain['frame'], gain['slot']]
                           and in_penalty(xy[gain['frame'], gain['slot']], team, h, signs, pitch))
        if gain is not None:
            outcome = 'duel' if gain['duel'] else ('completed' if gain['team'] == team else 'intercepted')
            next_gain = gain['frame']
        if goalkeeper_gain:
            gf, gs = gain['frame'], gain['slot']
            key = (gf, gs)
            after = next((e for e in accepted_losses if e['slot'] == gs and e['half'] == h and e['frame'] >= gf), None)
            takeover = next((g for g in core['gains'] if g['half'] == h and g['frame'] > gf and g['slot'] != gs), None)
            end_control = min(after['frame'] if after else len(half), takeover['frame'] if takeover else len(half))
            retention = 'deflect' if end_control - gf <= 25 else 'retain'
            dead_type = restart['dead_event'] if restart else None
            if dead_type == 'out_for_goal_kick':
                name = 'shot_off_target' if origin_zone == 'shot' else ('cross' if origin_zone == 'cross' else 'pass')
                reason = 'figure6b_goalkeeper_gain_false_positive'
                rejected.add(key)
                receiving[key]['event'] = 'rejected_goalkeeper_gain'
            elif dead_type == 'out_for_corner':
                name = 'cross' if origin_zone == 'cross' else shot_name
                reason, saving = 'figure6c', 'save_deflect'
            elif dead_type == 'goal':
                name, reason, saving = 'shot_on_target', 'figure6d', 'unsuccessful_save'
            elif origin_zone == 'shot' and forward:
                name, reason, saving = shot_name, 'figure6e_shot', 'save_' + retention
            elif origin_zone == 'cross':
                attackers = sum(teams[gf, s] == team and in_penalty(xy[gf, s], team, h, signs, pitch)
                                for s in range(xy.shape[1]) if np.isfinite(xy[gf, s]).all())
                if attackers:
                    name, reason, saving = 'cross', 'figure6e_cross', 'claim_' + retention
                else:
                    name, reason, saving = 'pass', 'figure6e_no_attacker', 'reception_from_loose_ball'
            else:
                reason, saving = 'figure6e_other', 'reception_from_loose_ball'
            if saving:
                receiving[key]['event'] = saving
        elif gain is None and restart is not None:
            # Fig. 6a: a stoppage intervenes before the goalkeeper controls it.
            if restart['dead_event'] == 'goal':
                name = 'own_goal' if restart['team'] == team else 'shot_on_target'
                outcome = 'goal'
            elif restart['dead_event'] in ('out_for_corner', 'out_for_goal_kick'):
                name = 'shot_off_target' if origin_zone == 'shot' else ('cross' if origin_zone == 'cross' else 'pass')
            reason = 'figure6a_dead_before_reception'
        elif gain is not None:
            if origin_zone == 'cross' and in_penalty(core['ball'][gain['frame']], team, h, signs, pitch):
                attackers = any(teams[gain['frame'], s] == team and in_penalty(xy[gain['frame'], s], team, h, signs, pitch)
                                for s in range(xy.shape[1]) if np.isfinite(xy[gain['frame'], s]).all())
                if attackers:
                    name = 'cross'
            key = (gain['frame'], gain['slot'])
            receiving[key]['event'] = 'reception' if gain['team'] == team else 'interception'
            receiving[key]['origin'] = outcome
        elif origin_zone == 'shot' and forward:
            # Last-minute goal uncertainty: require an actual 2D goal-line crossing.
            segment = core['ball'][f:stop]
            x = segment[:, 0] if signs[h, team] > 0 else pitch[0] - segment[:, 0]
            crossed = (x >= pitch[0]) & (np.abs(segment[:, 1] - pitch[1] / 2) <= 3.66)
            if crossed.any():
                name, outcome, reason = shot_name, 'goal_uncertain', 'period_end_2d_crossing'
        source_restart = next((r for r in reversed(restarts) if r['half'] == h and r['frame'] <= f
                               and not np.asarray(dead)[r['frame']:f + 1].any()), None)
        from_set_piece = None
        if source_restart is not None:
            earlier = any(e['half'] == h and source_restart['frame'] <= e['frame'] < f for e in passing)
            if not earlier:
                from_set_piece = source_restart['kind']
        passing.append(dict(loss, event=name, outcome=outcome, next_gain=next_gain,
                            receiver_slot=gain['slot'] if gain else None, from_set_piece=from_set_piece,
                            zone=origin_zone, towards_goal=forward, reason=reason))
    # Fig. 6b rejects the supposed keeper possession, including a release that
    # depended only on that rejected gain. Keep rejected reception in the trace.
    filtered = []
    for event in passing:
        previous = [g for g in core['gains'] if g['slot'] == event['slot'] and g['half'] == event['half'] and g['frame'] <= event['frame']]
        last = previous[-1] if previous else None
        valid_restart = any(r['half'] == event['half'] and r['slot'] == event['slot'] and r['frame'] <= event['frame']
                            and (last is None or r['frame'] >= last['frame']) for r in restarts)
        if last is None or (last['frame'], last['slot']) not in rejected or valid_restart:
            filtered.append(event)
    return filtered, list(receiving.values()), rejected
