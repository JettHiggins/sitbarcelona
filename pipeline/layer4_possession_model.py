"""Out-of-block self-training using only collective player geometry."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from . import config as c

def build(track,f,events,dead):
    # Distances to the ball, margin and ball-neighbour counts are deliberately
    # excluded: the model is useful precisely when the ball disappears.
    stems=('centroid_x','centroid_y','spread','speed','press')
    columns=[stem+'_'+team for stem in stems for team in ('H','A')]
    base=f[columns]
    x=np.full((c.N,len(columns)*len(c.MODEL_WINDOWS)),np.nan,dtype=np.float32)
    for a,b in c.HALVES:
        x[a:b]=np.hstack([base.iloc[a:b].rolling(w,center=True,min_periods=1).mean().to_numpy() for w in c.MODEL_WINDOWS])
    # Swapping teams includes all matching feature pairs, with complementary labels.
    permutation=np.arange(x.shape[1]).reshape(-1,2)[:,::-1].ravel()
    target=events['certain'].copy()
    target[dead]=-1
    valid=np.zeros(c.N,dtype=bool)
    for a,b in c.HALVES: valid[a:b]=True
    training=np.flatnonzero((target>=0)&valid)[::c.TRAIN_STRIDE]
    p=np.full(c.N,.5)
    audits=[]
    with threadpool_limits(limits=1):
        for ha,hb in c.HALVES:
            for a in range(ha,hb,c.MODEL_BLOCK):
                b=min(a+c.MODEL_BLOCK,hb)
                # Purge both sides so centred 30 s feature windows and pseudo
                # pass labels cannot share observations across the held-out edge.
                train=training[(training<a-c.MODEL_PURGE)|(training>=b+c.MODEL_PURGE)]
                assert not ((train>=a-c.MODEL_PURGE)&(train<b+c.MODEL_PURGE)).any()
                if len(np.unique(target[train]))<2:
                    audits.append(dict(start=a,end=b,training=len(train),fallback=True));continue
                xx=np.concatenate((x[train],x[train][:,permutation]))
                yy=np.r_[target[train]==0,target[train]!=0].astype(int)
                model=HistGradientBoostingClassifier(max_iter=c.MODEL_ITER,max_leaf_nodes=c.MODEL_LEAVES,learning_rate=c.MODEL_RATE,early_stopping=False,random_state=c.SEED)
                model.fit(xx,yy)
                # Symmetric test-time prediction keeps mirror consistency exact.
                p[a:b]=(model.predict_proba(x[a:b])[:,1]+1-model.predict_proba(x[a:b][:,permutation])[:,1])/2
                audits.append(dict(start=a,end=b,training=len(train),fallback=False))
                print('held-out block',a,b,'training',len(train),flush=True)
    (c.CACHE/'model_blocks.json').write_text(json.dumps(audits,indent=2))
    np.save(c.CACHE/'probability_home.npy',p)
    return p
