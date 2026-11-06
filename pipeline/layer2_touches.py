"""Controlled receptions, kicks, delayed duels and pseudo-label evidence."""
import numpy as np
from . import config as c

def build(track,f,distances,nearest):
    n=c.N
    owner=np.full(n,-1,dtype=int)
    player=np.full(n,-1,dtype=int)
    reception=np.full(n,-1,dtype=int)
    kick=np.full(n,-1,dtype=int)
    certain=np.full(n,-1,dtype=int)
    duel=(f.dH.to_numpy()<c.REACH)&(f.dA.to_numpy()<c.REACH)
    direction=track['ball_direction']
    turn=np.abs(np.angle(np.exp(1j*np.diff(direction,prepend=direction[0]))))
    accel=np.abs(np.diff(track['ball_speed'],prepend=np.nan))
    fast=track['ball_speed']>c.FAST_BALL
    glitch=fast&np.roll(fast,1)&(turn>c.GLITCH_TURN)
    for team,name in enumerate(('H','A')):
        idx=nearest[:,team]
        relative=np.linalg.norm(track['player_velocity'][np.arange(n),idx]-track['velocity'],axis=1)
        control=(f['d'+name].to_numpy()<c.REACH)&(relative<c.COMOTION)&~duel&~glitch
        # Require continuity in both team and track slot; slots are not identities.
        key=np.where(control,idx,-1)
        for a,b in c.runs(key):
            if key[a]<0: continue
            if b-a>=c.RECEPTION_FRAMES:
                owner[a:b]=team; player[a:b]=key[a]; reception[a]=team
            if b-a>=c.CERTAIN_FRAMES:
                certain[a:b]=team
        nearby=(f['d'+name].to_numpy()<c.REACH)&~duel&~glitch
        abrupt=(accel>c.KICK_ACCEL)|((turn>c.KICK_TURN)&fast)
        kick[nearby&abrupt]=team
    # Completed pass: two stable, distinct receivers from the same team,
    # separated by a short flight, with no opposing control in between.
    recs=np.flatnonzero(reception>=0)
    for a,b in zip(recs[:-1],recs[1:]):
        if reception[a]==reception[b] and player[a]!=player[b] and b-a<=c.MODEL_WINDOWS[0] and not glitch[a:b+1].any():
            certain[a:b+1]=reception[a]
    gaps=np.r_[False,track['available'][1:]!=track['available'][:-1]]
    print('receptions:',len(recs),'certain frames:',int((certain>=0).sum()),flush=True)
    return dict(owner=owner,player=player,reception=reception,kick=kick,duel=duel,glitch=glitch,gap_edge=gaps,certain=certain)
