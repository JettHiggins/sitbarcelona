"""Wes dead/live revision and change-candidate experiments.

No competition labels are read here. The original Layer 5 decoder remains the default.
The proposal scores below are fixed, deliberately conservative starting hypotheses;
only held-out other-game evaluation can justify using them for submission.
"""

import numpy as np

from ..mall.interaction.features import contacts
from ..mall.layer2_touches.touches import RECEIVE
from ..mall.layer5_sequence_decoder.decoder import decode, frame_evidence
from . import config as W


def _runs(mask):
    edges = np.diff(np.r_[False, np.asarray(mask, bool), False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def stage_dead(z, L3, stage):
    """Layer 3 state before/after individual postprocessing steps; final is the baseline."""
    if stage == 'final':
        return L3['dead'].astype(bool)
    key = 'stage_' + stage
    if key not in L3:
        raise ValueError(f'{key} missing: rebuild Layer 3 with the current code')
    out = L3[key].astype(bool).copy()
    out[z['half'] == 0] = True
    if 'hard_kickoff' in L3:
        out[L3['hard_kickoff'].astype(bool)] = True
    return out


def revise_dead(z, F, L2, L3, L4):
    """Reopen a baseline D subspan only when a competing live path scores higher.

    Candidate intervals come from the Layer 3 Viterbi path or sustained controls.
    The score compares the original dead/live log odds with two state changes, and
    gives bounded credit for coherent ownership while the ball is actually observed.
    A missing ball and a confident team model alone cannot reopen a stoppage.
    """
    n = len(z['half'])
    original = L3['dead'].astype(bool)
    raw_live = ~stage_dead(z, L3, 'viterbi')
    hard = z['half'] == 0
    if 'hard_kickoff' in L3:
        hard |= L3['hard_kickoff'].astype(bool)
    outside = np.nan_to_num(F['ball_outside'], nan=0.0) > W.MAX_BALL_OUTSIDE_M
    hard |= outside
    control = np.zeros(n, bool)
    for event in L2['events']:
        if int(event['kind']) == RECEIVE and float(np.nan_to_num(event['contest'])) <= W.CONTROL_MAX_CONTEST:
            a, b = int(event['frame']), int(event['frame'] + event['duration'])
            control[max(a, 0):min(b, n)] = True
    proposed = original & ~hard & (raw_live | control)
    ownership = frame_evidence(L2, L4)
    observed = np.isfinite(z['ball_xy'][:, 0]) & ~outside
    llr = L3['dead_llr'].astype(float)
    result = original.copy()
    decisions = []
    for a, b in _runs(proposed):
        a, b = int(a), int(b)
        # A live island in D pays two boundary costs. If it touches an existing
        # live spell, it pays only one. Segments at a half edge remain protected.
        if b - a < W.MIN_REOPEN_FRAMES:
            continue
        switches = int(a > 0 and original[a - 1]) + int(b < n and original[b])
        if switches == 0:
            switches = 1
        # Team evidence is usable only with a visible ball and a real control.
        # Cap it so an overconfident owner model cannot overwhelm dead-ball cues.
        evidence = ownership[a:b][observed[a:b] & control[a:b]]
        team_credit = min(W.TEAM_CREDIT_CAP,
                          W.TEAM_CREDIT_WEIGHT * abs(float(np.nansum(evidence))))
        dead_advantage = float(np.nansum(llr[a:b]))
        margin = -dead_advantage + team_credit - switches * W.DEAD_SWITCH_COST
        reopen = margin > 0
        if reopen:
            result[a:b] = False
        decisions.append(dict(start=a, end=b, source='raw_viterbi+control' if
                              np.any(raw_live[a:b] & control[a:b]) else
                              ('raw_viterbi' if np.any(raw_live[a:b]) else 'control'),
                              dead_log_odds=dead_advantage, team_credit=team_credit,
                              transition_cost=switches * W.DEAD_SWITCH_COST,
                              live_margin=margin, reopened=bool(reopen)))
    result[hard] = True
    return result, decisions


def change_candidates(z, F, L2, dead, L4):
    """Soft candidate locations with separate contact, team, and change evidence.

    Contacts are hypotheses. Both teams survive near a contested ball; a later
    recovery raises attribution confidence but never forces a turnover.
    """
    events = contacts(z, F, L2, dead)
    recovery = [(int(r['frame']), int(r['team'])) for r in events if r['anchor']]
    cp_frames = np.asarray(L4['cp_frames'], int) if 'cp_frames' in L4 else np.array([], int)
    cp_p = np.asarray(L4['cp_p'], float) if 'cp_p' in L4 else np.empty((0, 3))
    rows = []
    for event in events:
        primary = 'recovery_anchor' if event['anchor'] and event['kind'] == RECEIVE else 'contact'
        for frame, source in ((event['frame'], primary), (event['motion_frame'], 'motion')):
            if not 0 <= frame < len(dead) or dead[frame]:
                continue
            dh, da = float(F['d_h'][frame]), float(F['d_a'][frame])
            near = np.array([dh, da], float)
            near = np.where(np.isfinite(near) & (near <= W.TOUCH_ATTRIB_RADIUS),
                            np.exp(-np.maximum(near, 0)), 0.0)
            if near.sum() > 0:
                team = near / near.sum()
            else:
                team = np.eye(2)[event['team']]
            # Confirm an attribution only from a later control by the same team.
            later = [t for f, t in recovery
                     if frame < f <= frame + W.RECOVERY_LOOKAHEAD_FRAMES]
            if later:
                team = (1 - W.RECOVERY_BLEND) * team + W.RECOVERY_BLEND * np.eye(2)[later[0]]
            change = W.CONTACT_CHANGE_PRIOR  # uncertain contact alone is weak turnover evidence
            if len(cp_frames):
                k = int(np.argmin(np.abs(cp_frames - frame)))
                if abs(int(cp_frames[k]) - frame) <= 3:
                    change = float(np.clip(1.0 - cp_p[k, 0], 0, 1))
            contact = float(np.clip(event['confidence'], 0, 1))
            bonus = W.CONTACT_BONUS_WEIGHT * contact * change * team
            rows.append(dict(frame=int(frame), source=source,
                             contact_observed=contact, team_home=float(team[0]),
                             team_away=float(team[1]), possession_changed=change,
                             bonus_home=float(bonus[0]), bonus_away=float(bonus[1])))
    # The original decoder already uses every learned CP frame. Retain strong
    # suppressed-touch CPs in the source/confidence trace without double counting.
    for frame, p in zip(cp_frames, cp_p):
        if (not 0 <= frame < len(dead) or dead[frame] or
                1.0 - p[0] < W.STRONG_CP_PROBABILITY):
            continue
        team = p[1:] / max(float(p[1:].sum()), 1e-9)
        rows.append(dict(frame=int(frame), source='learned_cp_existing',
                         contact_observed=0.0, team_home=float(team[0]),
                         team_away=float(team[1]), possession_changed=float(1.0 - p[0]),
                         bonus_home=0.0, bonus_away=0.0))
    return rows


def predict(z, F, L2, L3, L4, variant='revisable'):
    """Run an isolated ablation; return labels and local diagnostics."""
    if variant in ('joint-d', 'joint-team', 'joint'):
        from .joint import predict_joint
        return predict_joint(z, F, L2, L3, L4, variant)
    if variant in ('viterbi', 'snapped', 'rules', 'durations', 'final'):
        dead = stage_dead(z, L3, variant)
        decisions = []
    elif variant in ('revisable', 'candidates'):
        dead, decisions = revise_dead(z, F, L2, L3, L4)
    else:
        raise ValueError(f'unknown Mall segmentation variant: {variant}')
    candidates = change_candidates(z, F, L2, dead, L4) if variant == 'candidates' else []
    labels = decode(z, L2, L3, L4, dead_override=dead, extra_candidates=candidates)
    return labels, dict(variant=variant, dead_revisions=decisions,
                        change_candidates=candidates,
                        changed_dead_frames=int(np.sum(dead != L3['dead'])))
