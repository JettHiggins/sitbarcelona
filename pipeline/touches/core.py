"""Paper-inspired window classifier and literal last-touch team assignment.

This is an offline 2D adaptation, not a reproduction of SwiftEvent or SALT.
All constants are fixed independently of the challenge possession ground truth.
"""
from dataclasses import asdict, dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d
from scipy.signal import find_peaks


@dataclass(frozen=True)
class Parameters:
    fps: int = 25
    radius: float = 3.0
    window: int = 12  # 25 samples; 0.48 seconds future context
    maximum_speed: float = 60.0
    minimum_impulse: float = 0.05  # stationary numerical/noise floor, not a kick threshold
    threshold: float = 0.65
    actor_margin: float = 0.10
    same_actor_separation: int = 3


def signals(ball, half, p=Parameters()):
    """No interpolation: any gap, half boundary or jump invalidates its window."""
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1) & (half > 0)
    step = np.linalg.norm(np.diff(ball, axis=0), axis=1) * p.fps
    bad = np.r_[False, (step > p.maximum_speed) | (half[1:] != half[:-1])]
    good = observed & ~bad & ~np.r_[bad[1:], False]
    valid = minimum_filter1d(good.astype(int), size=2*p.window+1, mode='constant', cval=0).astype(bool)
    before = np.full((n, 2), np.nan)
    after = before.copy()
    before[3:] = (ball[3:] - ball[:-3]) * p.fps/3
    after[:-3] = (ball[3:] - ball[:-3]) * p.fps/3
    impulse = np.linalg.norm(after-before, axis=1)
    return dict(valid=valid, before=before, after=after, impulse=impulse,
                observed=observed, jump=bad)


def features(players, ball, frames, slots, sig, p=Parameters()):
    """Distance-window polynomial + 2D before/after kinematics; no team inputs."""
    frames, slots = np.asarray(frames), np.asarray(slots)
    offsets = np.arange(-p.window, p.window+1)
    idx = frames[:, None] + offsets
    d = np.linalg.norm(players[idx, slots[:, None]] - ball[idx], axis=2)
    # Missing player samples remain NaN. No track extrapolation or zero filling.
    poly = np.linalg.pinv(np.vander(np.linspace(-1, 1, len(offsets)), 6))
    coeff = d @ poly.T
    pre, post = sig['before'][frames], sig['after'][frames]
    speed0, speed1 = np.linalg.norm(pre, axis=1), np.linalg.norm(post, axis=1)
    cosine = np.sum(pre*post, axis=1) / np.maximum(speed0*speed1, 0.01)
    center = d[:, p.window]
    with np.errstate(invalid='ignore'):
        # Completeness flags distinguish absent player history from proximity.
        minimum = np.min(np.where(np.isfinite(d), d, np.inf), axis=1)
    # Round sub-micrometre noise before argmin so constant-distance windows
    # have a stable minimum position under a pitch translation/reflection.
    minimum_position = np.argmin(np.where(np.isfinite(d), np.round(d, 6), np.inf), axis=1)/p.fps
    return np.column_stack([coeff, center, minimum, minimum_position,
                            d[:, 0], d[:, -1], d[:, p.window-3], d[:, p.window+3],
                            (d[:, p.window+1]-d[:, p.window-1])*p.fps/2,
                            speed0, speed1, sig['impulse'][frames], cosine,
                            (speed1-speed0)/np.maximum(speed0+speed1, .1),
                            np.isfinite(d).mean(axis=1)]).astype(np.float32)


def assign_teams(n, half, touches):
    """Switch exactly at accepted touch frames; no D, smoothing or backdating."""
    labels = np.full(n, -1, np.int8)
    by_frame = {e['frame']: e for e in touches}
    last, previous_half = -1, 0
    for f in range(n):
        if half[f] == 0 or half[f] != previous_half:
            last = -1
        if half[f] > 0 and f in by_frame:
            last = by_frame[f]['team']
        labels[f] = last if half[f] > 0 else -1
        previous_half = half[f]
    return labels


def detect(players, ball, half, teams, model, p=Parameters(), evidence_path=None):
    sig = signals(ball, half, p)
    n, count = players.shape[:2]
    scores = np.zeros((n, count), np.float32)
    candidates = sig['valid'] & (sig['impulse'] >= p.minimum_impulse)
    # Bounded batches keep full-match feature memory small.
    for start in range(0, n, 4000):
        stop = min(n, start+4000)
        distance = np.linalg.norm(players[start:stop]-ball[start:stop, None], axis=2)
        ff, ss = np.where((distance <= p.radius) & candidates[start:stop, None] & (teams[start:stop] >= 0))
        if len(ff):
            ff += start
            scores[ff, ss] = model.predict_proba(features(players, ball, ff, ss, sig, p))[:, 1]
    proposals = {}
    for slot in range(count):
        peaks, _ = find_peaks(scores[:, slot], height=p.threshold, distance=p.same_actor_separation)
        for f in peaks:
            proposals.setdefault(int(f), []).append(slot)
    accepted, rejected = [], []
    for f, slots in sorted(proposals.items()):
        slot = max(slots, key=lambda s: scores[f, s])
        others = scores[f].copy()
        others[slot] = 0
        runner = float(others.max())
        score = float(scores[f, slot])
        distance = float(np.linalg.norm(players[f, slot]-ball[f]))
        event = dict(frame=f, half=int(half[f]), slot=int(slot), team=int(teams[f, slot]),
                     event='touch', score=score, runner_up_score=runner, margin=score-runner,
                     distance_m=distance, impulse_m_s=float(sig['impulse'][f]))
        if score-runner < p.actor_margin:
            event['reason'] = 'ambiguous_actor'
            rejected.append(event)
        else:
            accepted.append(event)
    if evidence_path is not None:
        np.savez_compressed(evidence_path, actor_scores=scores, valid_window=sig['valid'],
                            impulse_m_s=sig['impulse'], observed_ball=sig['observed'],
                            jump=sig['jump'], candidate=candidates)
    return assign_teams(n, half, accepted), dict(parameters=asdict(p), touches=accepted,
             rejected=rejected, observed_ball_frames=int(sig['observed'].sum()),
             valid_window_frames=int(sig['valid'].sum()), candidate_frames=int(candidates.sum()))
