"""Semi-Markov dynamic programming on admissible event boundaries."""
import numpy as np
from . import config as c

def build(track,f,events,dead,restart,probability=None):
    labels=np.full(c.N,'D')
    owner=events['owner']; rec=events['reception']; kick=events['kick']
    if probability is None:
        probability=np.full(c.N,.5)
    p=np.clip(probability,.05,.95)
    # Player model supplies weak context. Control supplies direct evidence.
    evidence=np.column_stack((np.log(p),np.log1p(-p)))*c.MODEL_WEIGHT/c.FPS
    for team in (0,1):
        evidence[:,team]+=c.CONTROL_WEIGHT*(owner==team)/c.FPS
        evidence[:,team]-=c.CONTROL_WEIGHT*(owner==1-team)/c.FPS
    allowed=(rec>=0)|(kick>=0)|(restart>=0)|events['gap_edge']
    # Duel outcomes must resolve to a reception before switching.
    allowed&=~events['duel']&~events['glitch']
    for ha,hb in c.HALVES:
        alive=~dead[ha:hb]
        for aa,bb in c.runs(alive):
            if not alive[aa]: continue
            a,b=ha+aa,ha+bb
            bounds=np.r_[a,np.flatnonzero(allowed[a+1:b])+a+1,b]
            k=len(bounds)-1
            cumulative=np.vstack((np.zeros(2),np.cumsum(evidence[a:b],axis=0)))
            # State includes last switch boundary, allowing an explicit duration
            # prior when closing each possession. Quadratic in event count.
            best=np.full((k+1,2),-np.inf)
            previous=np.full((k+1,2),-1,dtype=int)
            for j in range(1,k+1):
                end=bounds[j]
                for team in (0,1):
                    starts=np.arange(j)
                    frames=bounds[starts]
                    emission=cumulative[end-a,team]-cumulative[frames-a,team]
                    length=end-frames
                    prior=-c.SHORT_PENALTY*np.maximum(0,c.MIN_POSSESSION-length)/c.MIN_POSSESSION
                    scores=best[starts,1-team]-c.SWITCH_COST+emission+prior
                    bonus=c.RECEPTION_REWARD*(rec[frames]==team)+c.KICK_REWARD*(kick[frames]==team)
                    scores+=bonus
                    # A kick by the other team or an uninformative gap edge
                    # cannot by itself overturn retained possession.
                    valid=(rec[frames]==team)|(kick[frames]==team)|(restart[frames]==team)|events['gap_edge'][frames]
                    scores[~valid]=-np.inf
                    initial=emission[0]+prior[0]
                    if restart[a]>=0:
                        initial+=c.RECEPTION_REWARD*(restart[a]==team)
                    else:
                        known=owner[a:b]; known=known[known>=0]
                        if len(known): initial+=c.KICK_REWARD*(known[0]==team)
                    scores[0]=initial
                    ii=int(np.argmax(scores)); best[j,team]=scores[ii];previous[j,team]=ii
            team=int(np.argmax(best[k])); j=k
            while j>0:
                i=previous[j,team]
                labels[bounds[i]:bounds[j]]='H' if team==0 else 'A'
                j=i;team=1-team
        switches=np.flatnonzero((labels[ha+1:hb]!=labels[ha:hb-1])&~dead[ha+1:hb]&~dead[ha:hb-1])+ha+1
        assert allowed[switches].all(), 'Possession switched outside an admissible event'
    return labels
