"""Contact chains and later recovery evidence. No labels or video are read here."""
import numpy as np

from .. import config as C
from ..layer2_touches.touches import (
    RECEIVE, TOUCH, KICK, ARTIFACT, STOP, UNATTRIBUTED,
    attribute, kinematic_candidates,
)
from ..layer4_possession_model import crossgame as X
from ..layer4_possession_model.model import PLAYER_BASE

LOOKAROUND = 8 * C.FPS


def regions(half, dead):
    """Unique live-region IDs; contacts must never link across a stoppage or half."""
    live = (half > 0) & ~np.asarray(dead, bool)
    starts = live & np.r_[True, (~live[:-1]) | (half[1:] != half[:-1])]
    return np.where(live, np.cumsum(starts), -1)


def contacts(z, F, L2, dead):
    """Preserve confirmed controls/kicks and soft kinematic contacts suppressed by Mall.

    Suspected artifacts/stops remain low-confidence hypotheses, not confirmed touches.
    Both the motion-change frame and the closest-approach frame remain available.
    """
    region = regions(z['half'], dead)
    rows = {}

    def add(frame, team, kind, distance, confidence, anchor=False, motion_frame=None):
        frame = int(frame)
        if not 0 <= frame < len(region) or region[frame] < 0 or team not in (0, 1):
            return
        motion_frame = frame if motion_frame is None else int(motion_frame)
        key = (frame, int(team))
        row = dict(frame=frame, team=int(team), kind=int(kind), distance=float(distance),
                   confidence=float(confidence), anchor=bool(anchor), region=int(region[frame]),
                   motion_frame=motion_frame)
        previous = rows.get(key)
        if previous is None or (anchor, confidence) > (previous['anchor'], previous['confidence']):
            rows[key] = row

    for event in L2['events']:
        kind = int(event['kind'])
        if kind not in (RECEIVE, TOUCH, KICK):
            continue
        add(event['frame'], int(event['team']), kind, event['dist'],
            1.0 if kind in (RECEIVE, KICK) else 0.6,
            anchor=kind in (RECEIVE, KICK))

    frames, kinds, _ = kinematic_candidates(z['ball_speed'], z['ball_vel'], z['half'])
    for frame, kind, (team, slot, distance, at) in zip(frames, kinds, attribute(frames, F)):
        if team < 0 or not np.isfinite(distance) or distance > C.TOUCH_ATTRIB_RADIUS:
            continue
        # Reject cross-stoppage attribution, but retain uncertainty about event semantics.
        if region[frame] < 0 or region[at] != region[frame]:
            continue
        confidence = 0.5 if kind == TOUCH else 0.2
        add(at, team, kind, distance, confidence, motion_frame=frame)
    return sorted(rows.values(), key=lambda r: (r['frame'], r['team']))


def _neighbor(rows, region, n, future=False, offset=0):
    """Nearest contact/anchor within the same live region and bounded time window."""
    result = np.full(n, -1, np.int32)
    if not rows:
        return result
    frames = np.array([r['frame'] for r in rows])
    ids = np.searchsorted(frames, np.arange(n), side='left' if future else 'right')
    ids += offset if future else -1 - offset
    valid = (ids >= 0) & (ids < len(rows))
    clipped = np.clip(ids, 0, len(rows) - 1)
    reg = np.array([r['region'] for r in rows])
    valid &= (reg[clipped] == region) & (region >= 0)
    valid &= np.abs(frames[clipped] - np.arange(n)) <= LOOKAROUND
    result[valid] = ids[valid]
    return result


def design(z, F, L2, L3):
    n = len(z['half'])
    region = regions(z['half'], L3['dead'])
    events = contacts(z, F, L2, L3['dead'])
    anchors = [r for r in events if r['anchor']]
    past = _neighbor(events, region, n)
    prior = _neighbor(events, region, n, offset=1)
    future = _neighbor(events, region, n, future=True)
    recovery = _neighbor(anchors, region, n, future=True)
    previous_control = _neighbor(anchors, region, n)
    cols = {k: np.asarray(F[k], float) for k in PLAYER_BASE}
    # Use explicit observations, not baseline model probabilities or target labels.
    cols.update(X.ball_signals(F, L2))
    # Signed fields have a defined home/away swap in crossgame.swap_index.
    for prefix, rows, index in [('last_contact', events, past), ('prior_contact', events, prior),
                                ('next_contact', events, future), ('recovery', anchors, recovery),
                                ('previous_play', anchors, previous_control)]:
        valid = index >= 0
        teams = np.array([r['team'] for r in rows] + [-1])
        times = np.array([r['frame'] for r in rows] + [0])
        conf = np.array([r['confidence'] for r in rows] + [0.0])
        dist = np.array([r['distance'] for r in rows] + [np.nan])
        idx = np.where(valid, index, len(rows))
        for team, tag in [(0, 'h'), (1, 'a')]:
            cols[f'{prefix}_{tag}'] = (valid & (teams[idx] == team)).astype(float)
        cols[prefix + '_age'] = np.where(valid, np.abs(np.arange(n) - times[idx]) / C.FPS, np.nan)
        cols[prefix + '_confidence'] = conf[idx]
        cols[prefix + '_distance'] = dist[idx]

    for tag in ('h', 'a'):
        other = 'a' if tag == 'h' else 'h'
        # Later recovery can support an earlier candidate contact without forcing backdating.
        cols['confirmed_contact_' + tag] = cols['last_contact_' + tag] * cols['recovery_' + tag]
        cols['rebound_return_' + tag] = (cols['prior_contact_' + tag] * cols['last_contact_' + other]
                                       * cols['recovery_' + tag])
        cols['turnover_chain_' + tag] = (cols['previous_play_' + other] * cols['last_contact_' + tag]
                                       * cols['recovery_' + tag])
    # Loss/reappearance are neutral events; Mall's team-specific versions are always zero.
    known = np.isfinite(z['ball_xy'][:, 0])
    for name, mask in [('lost', np.r_[False, known[:-1] & ~known[1:]]),
                       ('found', np.r_[False, ~known[:-1] & known[1:]])]:
        cols[name + '_nearby'] = X.window_mean(mask.astype(float), -12, 13)
    names = list(cols)
    matrix = np.column_stack([cols[k] for k in names]).astype(np.float32)
    return matrix, names, events


def candidates(events, dead, half):
    """Existing three-frame fallback grid plus contact/closest-approach candidates."""
    frames = set(map(int, X.cp_candidates(dead, half)))
    for event in events:
        frames.add(event['frame'])
        frames.add(event['motion_frame'])
    return np.array(sorted(f for f in frames if half[f] > 0 and not dead[f]), np.int32)
