"""Starter loading, video alignment, metric geometry and ball quality control."""
import hashlib
import json
import numpy as np
from scipy.signal import savgol_filter
from . import config as c

def build():
    from starter.load_atd import load, check_alignment
    paths = [c.DATA / 'tactical' / ('BAR-ATH_tactical_atd.' + ext) for ext in ('xml', 'txt')]
    fingerprint = hashlib.sha256()
    for path in paths:
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                fingerprint.update(chunk)
    key = fingerprint.hexdigest()
    cached = c.CACHE / 'tracking.npz'
    if cached.exists():
        with np.load(cached) as z:
            if str(z['fingerprint']) == key:
                raw = {name: z[name] for name in ('players', 'teams', 'ball_raw')}
            else:
                raw = None
    else:
        raw = None
    if raw is None:
        dataset = load(data_dir=str(c.DATA))
        assert check_alignment(dataset)['agrees'], 'ATD alignment mismatch'
        players = [p for team in dataset.metadata.teams for p in team.players]
        # Team metadata order is H/A, as documented by the starter.
        teams = np.array([dataset.metadata.teams.index(p.team) for p in players])
        xy = np.full((c.N, len(players), 2), np.nan, dtype=np.float32)
        ball = np.full((c.N, 2), np.nan, dtype=np.float32)
        for record in dataset.records:
            i = int(record.frame_id) - 1
            if record.ball_coordinates is not None:
                ball[i] = record.ball_coordinates.x, record.ball_coordinates.y
            for j, player in enumerate(players):
                pdata = record.players_data.get(player)
                if pdata is not None and pdata.coordinates is not None:
                    xy[i, j] = pdata.coordinates.x, pdata.coordinates.y
        raw = dict(players=xy * c.PITCH, teams=teams, ball_raw=ball * c.PITCH)
        np.savez_compressed(cached, **raw, fingerprint=key)
        del dataset
    ball = raw['ball_raw'].copy()
    visible = np.isfinite(ball).all(axis=1)
    # Remove isolated spikes on both sides; remove discontinuous observations
    # relative to the last accepted sample, using elapsed time across gaps.
    ids = np.flatnonzero(visible)
    for k in range(1, len(ids)-1):
        a, i, b = ids[k-1:k+2]
        v1 = np.linalg.norm(ball[i]-ball[a]) * c.FPS / (i-a)
        v2 = np.linalg.norm(ball[b]-ball[i]) * c.FPS / (b-i)
        bridge = np.linalg.norm(ball[b]-ball[a]) * c.FPS / (b-a)
        if min(v1, v2) > c.MAX_BALL_SPEED and bridge <= c.MAX_BALL_SPEED:
            ball[i] = np.nan
    last = None
    for i in np.flatnonzero(np.isfinite(ball).all(axis=1)):
        if last is not None and np.linalg.norm(ball[i]-ball[last])*c.FPS/(i-last) > c.MAX_BALL_SPEED:
            ball[i] = np.nan
        else:
            last = i
    accepted = np.isfinite(ball).all(axis=1)
    for a, b in c.runs(accepted):
        if not accepted[a] and 0 < a and b < c.N and b-a < c.MAX_GAP:
            ball[a:b] = np.linspace(ball[a-1], ball[b], b-a+2)[1:-1]
    available = np.isfinite(ball).all(axis=1)
    velocity = np.full_like(ball, np.nan)
    for a, b in c.runs(available):
        if available[a]:
            if b-a >= c.SMOOTH_WINDOW:
                ball[a:b] = savgol_filter(ball[a:b], c.SMOOTH_WINDOW, c.SMOOTH_ORDER, axis=0)
            if b-a >= 2:
                velocity[a:b] = np.gradient(ball[a:b], axis=0)*c.FPS
    p = raw['players']
    pv = np.full_like(p, np.nan)
    pv[1:] = np.diff(p, axis=0)*c.FPS
    speed = np.linalg.norm(pv, axis=2)
    direction = np.zeros((c.N, 2))
    attacks = {}
    for half, (a,b) in enumerate(c.HALVES, 1):
        # Goalkeepers are the extreme persistent team positions; centroids
        # provide a robust team-level fallback with fragmented track slots.
        h = np.nanmedian(p[a:b, raw['teams']==0, 0])
        av = np.nanmedian(p[a:b, raw['teams']==1, 0])
        attacks[str(half)] = [1 if h < av else -1, -1 if h < av else 1]
        direction[a:b] = attacks[str(half)]
    print('attacking directions:', attacks, 'ball available:', int(available.sum()), flush=True)
    (c.CACHE/'alignment.json').write_text(json.dumps({'frame_alignment':'frame_id - 1', 'attacks':attacks, 'source_sha256':key}, indent=2))
    return dict(**raw, ball=ball, visible=visible, available=available, velocity=velocity,
                ball_speed=np.linalg.norm(velocity,axis=1), ball_direction=np.arctan2(velocity[:,1],velocity[:,0]),
                player_velocity=pv, player_speed=speed, attack=direction)
