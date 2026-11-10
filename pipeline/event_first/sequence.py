"""Explicit touch → loose ball → recovery ownership state machine.

No learned frame/transition probabilities, duration priors or switching costs.
"""
import numpy as np

MAX_LOOSE_FRAMES = 6 * 25


def reconstruct(n, spans, events, initial_teams=None):
    labels = np.full(n, 2, np.int8)
    decisions = []
    initial_teams = initial_teams or {}
    for region, (start, end) in enumerate(spans):
        local = [dict(e) for e in events if e['region'] == region]
        anchors = [e for e in local if e['anchor'] and e['team'] in (0, 1)]
        seed = initial_teams.get(region)
        if seed is None:
            seed = anchors[0]['team'] if anchors else None
        if seed is None:
            possible = next((e['team'] for e in local if e['team'] in (0, 1)), 0)
            seed = possible
            seed_reason = 'no control/play anchor; first attributed contact or home fallback'
        else:
            seed_reason = 'restart/initial observation' if region in initial_teams else 'first control/play anchor'
        owner = int(seed)
        labels[start:end] = owner
        decisions.append(dict(region=region, frame=start, team=owner, reason=seed_reason))
        pending = []
        settled_until = start
        # Tentative opponent contacts are not promoted until a control or play confirms them.
        for event in local:
            frame = event['frame']
            pending = [p for p in pending if frame - p['frame'] <= MAX_LOOSE_FRAMES
                       and p['frame'] >= settled_until]
            if not event['anchor']:
                if frame >= settled_until:
                    pending.append(event)
                continue
            team = event['team']
            if team not in (0, 1):
                continue
            if team != owner:
                matching = [p for p in pending if team in p['teams']]
                boundary = matching[0]['frame'] if matching else frame
                labels[boundary:end] = team
                decisions.append(dict(region=region, frame=int(boundary), team=int(team),
                                      confirmed_at=int(frame),
                                      reason='earlier contact confirmed by recovery/play' if matching else 'control/play turnover',
                                      intervening_contacts=[dict(frame=p['frame'], teams=p['teams']) for p in pending]))
                owner = int(team)
            # A confirmed play settles the preceding episode. Later contacts may start a new one.
            pending.clear()
            settled_until = max(settled_until, event.get('end', frame + 1))
    return labels, decisions
