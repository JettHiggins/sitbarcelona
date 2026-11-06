"""Ball proximity and collective, identity-independent team geometry."""
import numpy as np
import pandas as pd
from . import config as c

def build(track):
    p, ball, teams = track['players'], track['ball'], track['teams']
    distances = np.linalg.norm(p-ball[:,None,:], axis=2)
    features = {}
    nearest = np.full((c.N,2), -1, dtype=int)
    for team, name in enumerate(('H','A')):
        mask = teams==team
        d = np.where(np.isfinite(distances[:,mask]),distances[:,mask],np.inf)
        features['d'+name] = d.min(axis=1)
        nearest[:,team] = np.flatnonzero(mask)[d.argmin(axis=1)]
        features['within2_'+name] = (d<c.NEAR).sum(axis=1)
        xy = p[:,mask]
        centre = np.nanmean(xy,axis=1)
        features['centroid_x_'+name],features['centroid_y_'+name] = centre.T
        features['spread_'+name] = np.sqrt(np.nanmean(np.sum((xy-centre[:,None])**2,axis=2),axis=1))
        features['speed_'+name] = np.nanmean(track['player_speed'][:,mask],axis=1)
        press = np.full(c.N,np.nan)
        # Bound memory: pairwise distances across teams in chunks.
        for a in range(0,c.N,1000):
            b=min(a+1000,c.N)
            pair=np.linalg.norm(p[a:b,mask,None,:]-p[a:b,None,~mask,:],axis=3)
            press[a:b]=np.nanmean(np.nanmin(pair,axis=2),axis=1)
        features['press_'+name]=press
    features['margin']=features['dA']-features['dH']
    features['mean_player_speed']=np.nanmean(track['player_speed'],axis=1)
    features['median_player_speed']=np.nanmedian(track['player_speed'],axis=1)
    return pd.DataFrame(features).replace([np.inf,-np.inf],np.nan), distances, nearest

def nearest_labels(features):
    known=features[['dH','dA']].notna().any(axis=1)
    labels=np.full(c.N,'D')
    labels[known]=np.where(features.loc[known,'dH'].fillna(np.inf)<=features.loc[known,'dA'].fillna(np.inf),'H','A')
    return labels
