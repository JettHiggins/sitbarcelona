"""Fuse noisy restart tags with geometry; enforce stoppage/play durations."""
import numpy as np
import pandas as pd
from . import config as c

def restart_team(track, frame, kind, spot):
    a,b=max(0,frame-c.FPS),min(c.N,frame+c.FPS)
    xy=np.nanmedian(track['players'][a:b],axis=0)
    dist=np.linalg.norm(xy-spot,axis=1)
    if 'throw' in kind:
        eligible=(xy[:,1]<0)|(xy[:,1]>c.PITCH[1])
    elif 'corner' in kind:
        eligible=(np.minimum(xy[:,0],c.PITCH[0]-xy[:,0])<c.RESTART_RADIUS)&(np.minimum(xy[:,1],c.PITCH[1]-xy[:,1])<c.RESTART_RADIUS)
    elif 'goal kick' in kind:
        attack=track['attack'][frame,track['teams']]
        ownx=np.where(attack>0,xy[:,0],c.PITCH[0]-xy[:,0])
        eligible=(ownx<c.RESTART_RADIUS)&(np.abs(xy[:,1]-c.PITCH[1]/2)<c.RESTART_RADIUS*2)
    else:
        eligible=dist<c.RESTART_RADIUS
    restricted=np.where(eligible,dist,np.inf)
    if np.isfinite(restricted).any():
        return int(track['teams'][restricted.argmin()])
    if np.isfinite(dist).any():
        return int(track['teams'][np.nanargmin(dist)])
    return -1

def build(track, features):
    from starter.load_smart_tagging import load
    events=load(data_dir=str(c.DATA))
    ball=track['ball']; x,y=ball.T
    last=pd.DataFrame(ball).ffill().to_numpy()
    beyond=(last[:,0]<0)|(last[:,0]>c.PITCH[0])|(last[:,1]<0)|(last[:,1]>c.PITCH[1])
    slow=features.median_player_speed.to_numpy()<c.SLOW_PLAYERS
    rest=track['ball_speed']<c.REST_BALL
    centre=np.linalg.norm(ball-np.array(c.PITCH)/2,axis=1)<c.CENTRE_RADIUS
    line=(np.minimum(np.abs(x),np.abs(c.PITCH[0]-x))<c.RESTART_RADIUS)|(np.minimum(np.abs(y),np.abs(c.PITCH[1]-y))<c.NEAR)
    # Kickoff formation: teams separated at centre with ball stationary.
    separated=np.abs(features.centroid_x_H-features.centroid_x_A).to_numpy()>c.RESTART_RADIUS
    score=c.OUT_WEIGHT*beyond+c.SLOW_WEIGHT*slow+c.SPOT_WEIGHT*(rest&(centre|line))+c.FORMATION_WEIGHT*(centre&rest&separated)
    tag=np.zeros(c.N)
    restart=np.full(c.N,-1,dtype=int)
    kickoff_candidates=[]
    for event in events:
        if (event['code'] or '').lower()!='set pieces':
            continue
        a=max(0,event['start_frame']); b=min(c.N,event['end_frame'])
        if b<=a: continue
        kind=event['tags'].get('Type','').lower()
        tag[a:b]=c.TAG_WEIGHT
        # A tag surrounds the restart. Prefer a sustained movement after rest;
        # its interval must never be copied wholesale as dead ball.
        moving=track['available'][a:b]&(track['ball_speed'][a:b]>c.REST_BALL*2)
        candidate=None
        for u,v in c.runs(moving):
            if moving[u] and v-u>=c.RECEPTION_FRAMES:
                candidate=a+u; break
        if candidate is None:
            candidate=(a+b)//2
        if 'kick off' in kind:
            # Pre-kickoff tracking can follow a different moving object.
            # A kickoff is anchored at centre, not at the first moving sample
            # in its approximate tag interval.
            centre_distance=np.linalg.norm(ball[a:b]-np.array(c.PITCH)/2,axis=1)
            if np.isfinite(centre_distance).any():
                candidate=a+int(np.nanargmin(centre_distance))
        spot=ball[candidate]
        if 'kick off' in kind:
            spot=np.array(c.PITCH)/2
        if not np.isfinite(spot).all():
            spot=last[candidate]
        restart[candidate]=restart_team(track,candidate,kind,spot)
        if 'kick off (start)' in kind:
            kickoff_candidates.append(candidate)
    score+=tag
    # Hysteresis is based on sustained evidence, not ball visibility alone.
    raw=score>=c.DEAD_THRESHOLD
    dead=np.zeros(c.N,dtype=bool)
    for a,b in c.runs(raw):
        if raw[a] and b-a>=c.MIN_DEAD:
            dead[a:b]=True
    # Close very short play holes between stoppages, then recheck lengths.
    for a,b in list(c.runs(dead)):
        if not dead[a] and a>0 and b<c.N and b-a<c.MIN_PLAY:
            dead[a:b]=True
    for ha,hb in c.HALVES:
        kicks=[k for k in kickoff_candidates if ha<=k<hb]
        if kicks:
            dead[ha:min(kicks)]=True
    # Every dead->live boundary is a restart, even if no tag exists.
    for i in np.flatnonzero(dead[:-1]&~dead[1:])+1:
        if restart[i]<0:
            spot=last[i]
            restart[i]=restart_team(track,i,'restart',spot)
    print('dead-ball frames:',int(dead.sum()), 'tagged kickoff estimates:',kickoff_candidates,flush=True)
    np.savez_compressed(c.CACHE/'dead_ball.npz',dead=dead,restart=restart,score=score)
    return dead,restart,score
