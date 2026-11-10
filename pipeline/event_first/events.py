"""Detect contact hypotheses and control/play anchors directly from tracking features."""
import numpy as np
from scipy.signal import find_peaks

FPS = 25
CONTACT_RADIUS = 2.5
CONTROL_RADIUS = 1.5
CONTROL_RELV = 3.0
CONTROL_FRAMES = 4
ATTRIBUTION_WINDOW = 4
AMBIGUOUS_MARGIN = 0.6
KICK_RISE = 5.0


def live_regions(half, dead):
    live = (np.asarray(half) > 0) & ~np.asarray(dead, bool)
    boundaries = np.r_[0, np.flatnonzero((live[1:] != live[:-1]) | (half[1:] != half[:-1])) + 1, len(live)]
    return [(int(a), int(b)) for a, b in zip(boundaries[:-1], boundaries[1:]) if live[a]]


def detect(z, F, dead):
    """No Mall event labels, possession probabilities, change points or GT are inputs.

    A contact is not automatically possession. Brief controls and plausible kicks
    are the anchors used to interpret preceding loose-ball contacts.
    """
    speed = np.asarray(z['ball_speed'], float)
    vel = np.asarray(z['ball_vel'], float)
    rows = []
    spans = live_regions(z['half'], dead)
    for region, (start, end) in enumerate(spans):
        d = np.column_stack([F['d_h'][start:end], F['d_a'][start:end]])
        rv = np.column_stack([F['relv_h'][start:end], F['relv_a'][start:end]])
        nearest = np.argmin(np.nan_to_num(d, nan=np.inf), axis=1)
        idx = np.arange(end - start)
        good = (d[idx, nearest] < CONTROL_RADIUS) & (rv[idx, nearest] < CONTROL_RELV)
        # A control must remain with the same tracked player, not just the same team.
        slots = np.where(nearest == 0, F['slot_h'][start:end], F['slot_a'][start:end])
        changes = (good[1:] != good[:-1]) | (nearest[1:] != nearest[:-1]) | (slots[1:] != slots[:-1])
        bounds = np.r_[0, np.flatnonzero(changes) + 1, end - start]
        for a, b in zip(bounds[:-1], bounds[1:]):
            if good[a] and b - a >= CONTROL_FRAMES:
                team = int(nearest[a])
                rows.append(dict(frame=int(start + a), end=int(start + b), region=region,
                                 kind='control', team=team, teams=[team], anchor=True,
                                 distance=float(d[a, team]), duration=int(b - a)))

        # Motion candidates are detected separately from controls, so adjacent
        # opponent contacts are not suppressed by a control segment.
        s = speed[start:end]
        dips, _ = find_peaks(-np.nan_to_num(s, nan=99), prominence=2.0, distance=3)
        turns = np.zeros(len(s))
        if len(s) > 6:
            before, after = vel[start:end - 6], vel[start + 6:end]
            norm = np.linalg.norm(before, axis=1) * np.linalg.norm(after, axis=1)
            cosine = np.divide(np.sum(before * after, axis=1), norm,
                               out=np.ones(len(norm)), where=norm > 1)
            turns[3:-3] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
        redirects, _ = find_peaks(np.where(s > 1, turns, 0), height=25, distance=3)
        candidates = sorted(set(map(int, dips)) | set(map(int, redirects)))
        for local in candidates:
            frame = start + local
            lo, hi = max(start, frame - ATTRIBUTION_WINDOW), min(end, frame + ATTRIBUTION_WINDOW + 1)
            distances, approaches = [], []
            for tag in ('h', 'a'):
                values = np.asarray(F['d_' + tag][lo:hi], float)
                # A flat distance plateau should not systematically pick the
                # earliest frame of the attribution window.
                rank = np.nan_to_num(values, nan=np.inf) + 1e-9 * np.abs(np.arange(lo, hi) - frame)
                k = int(np.argmin(rank))
                distances.append(float(values[k]) if np.isfinite(values[k]) else float('inf'))
                approaches.append(lo + k)
            closest = min(distances)
            teams = [team for team in (0, 1) if distances[team] <= CONTACT_RADIUS
                     and distances[team] <= closest + AMBIGUOUS_MARGIN]
            if not teams:
                continue
            team = teams[0] if len(teams) == 1 else -1
            # Keep motion timing distinct from nearest approach for diagnosis.
            contact_frame = approaches[int(np.argmin(distances))]
            future = speed[frame:min(end, frame + 11)]
            rise = float(np.nanmax(future) - speed[frame]) if np.isfinite(speed[frame]) else 0.0
            kick = team >= 0 and distances[team] <= CONTROL_RADIUS and rise >= KICK_RISE
            rows.append(dict(frame=int(contact_frame), motion_frame=int(frame), region=region,
                             kind='kick' if kick else 'contact', team=team, teams=teams,
                             anchor=bool(kick), distance=float(closest), duration=0,
                             turn_degrees=float(turns[local]), speed_rise=rise))

    # Deduplicate motion detectors, while preserving separate control/contact evidence.
    unique = {}
    for event in rows:
        key = (event['region'], event['frame'], event['kind'], tuple(event['teams']))
        previous = unique.get(key)
        if previous is None or event['distance'] < previous['distance']:
            unique[key] = event
    events = sorted(unique.values(), key=lambda e: (e['region'], e['frame'], not e['anchor'], e['kind']))
    return spans, events
